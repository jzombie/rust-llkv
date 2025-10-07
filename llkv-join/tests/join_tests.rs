//! Integration tests for table join operations.

use arrow::array::{Int32Array, Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use llkv_column_map::store::Projection;
use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::{CompareOp, Expr, ScalarExpr};
use llkv_join::{JoinKey, JoinOptions, TableJoinExt};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::table::{ScanProjection, ScanStreamOptions};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Helper to create a test table with row-id, user_id, and name columns.
fn create_test_table(
    table_id: u16,
    pager: &Arc<MemPager>,
    data: Vec<(u64, i32, &str)>,
) -> Table<MemPager> {
    let table = Table::new(table_id, Arc::clone(pager)).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("user_id", DataType::Int32, false).with_metadata(HashMap::from([(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        Field::new("name", DataType::Utf8, false).with_metadata(HashMap::from([(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]));

    let row_ids: Vec<u64> = data.iter().map(|(rid, _, _)| *rid).collect();
    let ids: Vec<i32> = data.iter().map(|(_, id, _)| *id).collect();
    let names: Vec<&str> = data.iter().map(|(_, _, name)| *name).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(row_ids)),
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
        ],
    )
    .unwrap();

    table.append(&batch).unwrap();
    table
}

#[test]
fn test_inner_join_simple() {
    let pager = Arc::new(MemPager::default());

    // Left: (1, "Alice"), (2, "Bob"), (3, "Charlie")
    let left = create_test_table(
        1,
        &pager,
        vec![(0, 1, "Alice"), (1, 2, "Bob"), (2, 3, "Charlie")],
    );

    // Right: (2, "Beta"), (3, "Gamma"), (4, "Delta")
    let right = create_test_table(
        2,
        &pager,
        vec![(0, 2, "Beta"), (1, 3, "Gamma"), (2, 4, "Delta")],
    );

    let keys = vec![JoinKey::new(1, 1)]; // Join on id column (field_id=1)
    let options = JoinOptions::inner();

    let mut result_batches = Vec::new();
    left.join_stream(&right, &keys, &options, |batch| {
        result_batches.push(batch);
    })
    .unwrap();

    // Should match: (2, "Bob") with (2, "Beta") and (3, "Charlie") with (3, "Gamma")
    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2, "Inner join should produce 2 rows");

    // Verify schema: id, name (left), id, name (right) - row_id excluded
    let schema = &result_batches[0].schema();
    assert_eq!(
        schema.fields().len(),
        4,
        "Output should have 4 columns (2 left + 2 right)"
    );
}

#[test]
fn test_left_join() {
    let pager = Arc::new(MemPager::default());

    let left = create_test_table(1, &pager, vec![(0, 1, "Alice"), (1, 2, "Bob")]);
    let right = create_test_table(2, &pager, vec![(0, 2, "Beta")]);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::left();

    let mut result_batches = Vec::new();
    left.join_stream(&right, &keys, &options, |batch| {
        result_batches.push(batch);
    })
    .unwrap();

    // Should have 2 rows: (1, "Alice", NULL) and (2, "Bob", 2, "Beta")
    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 2,
        "Left join should produce 2 rows (all left rows)"
    );
}

#[test]
fn test_semi_join() {
    let pager = Arc::new(MemPager::default());

    let left = create_test_table(
        1,
        &pager,
        vec![(0, 1, "Alice"), (1, 2, "Bob"), (2, 3, "Charlie")],
    );
    let right = create_test_table(2, &pager, vec![(0, 2, "Beta")]);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::semi();

    let mut result_batches = Vec::new();
    left.join_stream(&right, &keys, &options, |batch| {
        result_batches.push(batch);
    })
    .unwrap();

    // Should have 1 row: (2, "Bob") — only left rows with matches, no right columns
    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 1, "Semi join should produce 1 row");

    // Verify schema: only left columns (id, name) - row_id excluded
    let schema = &result_batches[0].schema();
    assert_eq!(
        schema.fields().len(),
        2,
        "Semi join should only have left columns"
    );
}

#[test]
fn test_anti_join() {
    let pager = Arc::new(MemPager::default());

    let left = create_test_table(
        1,
        &pager,
        vec![(0, 1, "Alice"), (1, 2, "Bob"), (2, 3, "Charlie")],
    );
    let right = create_test_table(2, &pager, vec![(0, 2, "Beta")]);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::anti();

    let mut result_batches = Vec::new();
    left.join_stream(&right, &keys, &options, |batch| {
        result_batches.push(batch);
    })
    .unwrap();

    // Should have 2 rows: (1, "Alice") and (3, "Charlie") — left rows with NO match
    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2, "Anti join should produce 2 rows");

    // Verify schema: only left columns (id, name) - row_id excluded
    let schema = &result_batches[0].schema();
    assert_eq!(
        schema.fields().len(),
        2,
        "Anti join should only have left columns"
    );
}

#[test]
fn test_many_to_many_join() {
    let pager = Arc::new(MemPager::default());

    // Left has 2 rows with id=1
    let left = create_test_table(1, &pager, vec![(0, 1, "Alice"), (1, 1, "Alice2")]);

    // Right has 3 rows with id=1
    let right = create_test_table(
        2,
        &pager,
        vec![(0, 1, "Alpha"), (1, 1, "Alpha2"), (2, 1, "Alpha3")],
    );

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::inner();

    let mut result_batches = Vec::new();
    left.join_stream(&right, &keys, &options, |batch| {
        result_batches.push(batch);
    })
    .unwrap();

    // Should have 2 * 3 = 6 rows (Cartesian product of matching keys)
    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 6,
        "Many-to-many join should produce 6 rows (2 * 3)"
    );
}

#[test]
fn test_empty_left_table() {
    let pager = Arc::new(MemPager::default());

    let left = create_test_table(1, &pager, vec![]);
    let right = create_test_table(2, &pager, vec![(0, 1, "Alpha")]);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::inner();

    let mut result_batches = Vec::new();
    left.join_stream(&right, &keys, &options, |batch| {
        result_batches.push(batch);
    })
    .unwrap();

    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0, "Join with empty left should produce 0 rows");
}

#[test]
fn test_empty_right_table() {
    let pager = Arc::new(MemPager::default());

    let left = create_test_table(1, &pager, vec![(0, 1, "Alice")]);
    let right = create_test_table(2, &pager, vec![]);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::inner();

    let mut result_batches = Vec::new();
    left.join_stream(&right, &keys, &options, |batch| {
        result_batches.push(batch);
    })
    .unwrap();

    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0, "Join with empty right should produce 0 rows");
}

#[test]
fn test_no_matching_rows() {
    let pager = Arc::new(MemPager::default());

    let left = create_test_table(1, &pager, vec![(0, 1, "Alice"), (1, 2, "Bob")]);
    let right = create_test_table(2, &pager, vec![(0, 3, "Gamma"), (1, 4, "Delta")]);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::inner();

    let mut result_batches = Vec::new();
    left.join_stream(&right, &keys, &options, |batch| {
        result_batches.push(batch);
    })
    .unwrap();

    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0, "Join with no matches should produce 0 rows");
}

#[test]
fn test_join_validation_errors() {
    let pager = Arc::new(MemPager::default());
    let left = create_test_table(1, &pager, vec![(0, 1, "Alice")]);
    let right = create_test_table(2, &pager, vec![(0, 1, "Alpha")]);

    // Empty keys should error
    let result = left.join_stream(&right, &[], &JoinOptions::default(), |_| {});
    assert!(result.is_err());

    // Bad batch size should error
    let bad_opts = JoinOptions {
        batch_size: 0,
        ..Default::default()
    };
    let result = left.join_stream(&right, &[JoinKey::new(1, 1)], &bad_opts, |_| {});
    assert!(result.is_err());
}

fn proj(table: &Table<MemPager>, field_id: llkv_table::types::FieldId) -> Projection {
    Projection::from(LogicalFieldId::for_user(table.table_id(), field_id))
}

#[test]
fn test_join_with_expression_filters() {
    const LEFT_TABLE_ID: u16 = 41;
    const RIGHT_TABLE_ID: u16 = 84;

    const LEFT_CUSTOMER_ID: llkv_table::types::FieldId = 11;
    const LEFT_SEGMENT: llkv_table::types::FieldId = 12;
    const LEFT_ANNUAL_REVENUE: llkv_table::types::FieldId = 13;
    const LEFT_LOYALTY: llkv_table::types::FieldId = 14;

    const RIGHT_ORDER_ID: llkv_table::types::FieldId = 21;
    const RIGHT_CUSTOMER_ID: llkv_table::types::FieldId = 22;
    const RIGHT_AVG_ORDER_VALUE: llkv_table::types::FieldId = 23;
    const RIGHT_TRAILING_SPEND: llkv_table::types::FieldId = 24;

    let pager = Arc::new(MemPager::default());

    let customer_table = Table::new(LEFT_TABLE_ID, Arc::clone(&pager)).unwrap();
    let customer_schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("customer_id", DataType::Int32, false).with_metadata(HashMap::from([(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            LEFT_CUSTOMER_ID.to_string(),
        )])),
        Field::new("segment", DataType::Utf8, false).with_metadata(HashMap::from([(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            LEFT_SEGMENT.to_string(),
        )])),
        Field::new("annual_revenue", DataType::Int64, false).with_metadata(HashMap::from([(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            LEFT_ANNUAL_REVENUE.to_string(),
        )])),
        Field::new("loyalty_score", DataType::Int32, false).with_metadata(HashMap::from([(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            LEFT_LOYALTY.to_string(),
        )])),
    ]));

    let customer_batch = RecordBatch::try_new(
        Arc::clone(&customer_schema),
        vec![
            Arc::new(UInt64Array::from((0u64..8).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(vec![
                1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008,
            ])),
            Arc::new(StringArray::from(vec![
                "emerging",
                "growth",
                "enterprise",
                "starter",
                "enterprise",
                "growth",
                "strategic",
                "growth",
            ])),
            Arc::new(Int64Array::from(vec![
                610, 820, 940, 560, 1_200, 710, 1_320, 680,
            ])),
            Arc::new(Int32Array::from(vec![45, 62, 88, 51, 91, 57, 97, 63])),
        ],
    )
    .unwrap();
    customer_table.append(&customer_batch).unwrap();

    let orders_table = Table::new(RIGHT_TABLE_ID, Arc::clone(&pager)).unwrap();
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("order_id", DataType::Int32, false).with_metadata(HashMap::from([(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            RIGHT_ORDER_ID.to_string(),
        )])),
        Field::new("customer_id", DataType::Int32, false).with_metadata(HashMap::from([(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            RIGHT_CUSTOMER_ID.to_string(),
        )])),
        Field::new("avg_order_value", DataType::Int64, false).with_metadata(HashMap::from([(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            RIGHT_AVG_ORDER_VALUE.to_string(),
        )])),
        Field::new("trailing_spend", DataType::Int64, false).with_metadata(HashMap::from([(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            RIGHT_TRAILING_SPEND.to_string(),
        )])),
    ]));

    let orders_batch = RecordBatch::try_new(
        Arc::clone(&orders_schema),
        vec![
            Arc::new(UInt64Array::from((0u64..10).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(vec![
                5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010,
            ])),
            Arc::new(Int32Array::from(vec![
                1002, 1003, 1003, 1004, 1005, 1007, 1008, 1010, 1006, 1011,
            ])),
            Arc::new(Int64Array::from(vec![
                125, 140, 155, 70, 205, 95, 110, 180, 118, 145,
            ])),
            Arc::new(Int64Array::from(vec![
                360, 420, 455, 165, 520, 210, 205, 600, 295, 330,
            ])),
        ],
    )
    .unwrap();
    orders_table.append(&orders_batch).unwrap();

    let high_value_filter = Expr::Compare {
        left: ScalarExpr::column(LEFT_ANNUAL_REVENUE),
        op: CompareOp::GtEq,
        right: ScalarExpr::literal(900_i64),
    };

    let mut high_value_customers: HashSet<i32> = HashSet::new();
    customer_table
        .scan_stream(
            &[ScanProjection::column(proj(
                &customer_table,
                LEFT_CUSTOMER_ID,
            ))],
            &high_value_filter,
            ScanStreamOptions::default(),
            |batch| {
                let ids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                for idx in 0..ids.len() {
                    high_value_customers.insert(ids.value(idx));
                }
            },
        )
        .unwrap();

    let premium_order_filter = Expr::Compare {
        left: ScalarExpr::column(RIGHT_AVG_ORDER_VALUE),
        op: CompareOp::GtEq,
        right: ScalarExpr::literal(120_i64),
    };

    let mut high_avg_order_customers: HashSet<i32> = HashSet::new();
    orders_table
        .scan_stream(
            &[ScanProjection::column(proj(
                &orders_table,
                RIGHT_CUSTOMER_ID,
            ))],
            &premium_order_filter,
            ScanStreamOptions::default(),
            |batch| {
                let ids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                for idx in 0..ids.len() {
                    high_avg_order_customers.insert(ids.value(idx));
                }
            },
        )
        .unwrap();

    assert_eq!(high_value_customers.len(), 3);
    assert!(high_value_customers.contains(&1003));
    assert!(high_value_customers.contains(&1005));
    assert!(high_value_customers.contains(&1007));

    assert_eq!(high_avg_order_customers.len(), 5);
    assert!(high_avg_order_customers.contains(&1002));
    assert!(high_avg_order_customers.contains(&1003));
    assert!(high_avg_order_customers.contains(&1005));
    assert!(high_avg_order_customers.contains(&1010));

    let join_keys = vec![JoinKey::new(LEFT_CUSTOMER_ID, RIGHT_CUSTOMER_ID)];
    let options = JoinOptions::inner();

    let mut joined_batches: Vec<RecordBatch> = Vec::new();
    customer_table
        .join_stream(&orders_table, &join_keys, &options, |batch| {
            joined_batches.push(batch);
        })
        .unwrap();

    assert!(!joined_batches.is_empty());
    let total_join_rows: usize = joined_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_join_rows, 8,
        "expected eight joined rows across batches"
    );

    let mut both_filters = 0usize;
    let mut left_only = 0usize;
    let mut right_only = 0usize;
    let mut neither = 0usize;
    let mut combined_scores: Vec<(i32, i64, i64)> = Vec::new();

    for batch in joined_batches {
        let left_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let left_revenue = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let right_ids = batch
            .column(5)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let right_avg = batch
            .column(6)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        for idx in 0..batch.num_rows() {
            let cid_left = left_ids.value(idx);
            let cid_right = right_ids.value(idx);
            assert_eq!(cid_left, cid_right);

            let left_pass = high_value_customers.contains(&cid_left);
            let right_pass = high_avg_order_customers.contains(&cid_right);

            match (left_pass, right_pass) {
                (true, true) => {
                    both_filters += 1;
                    combined_scores.push((cid_left, left_revenue.value(idx), right_avg.value(idx)));
                }
                (true, false) => left_only += 1,
                (false, true) => right_only += 1,
                (false, false) => neither += 1,
            }
        }
    }

    assert_eq!(
        both_filters, 3,
        "expected three joined rows passing both filters"
    );
    assert_eq!(
        left_only, 1,
        "expected one row passing only the customer filter"
    );
    assert_eq!(
        right_only, 1,
        "expected one row passing only the order filter"
    );
    assert_eq!(neither, 3, "expected three rows failing both filters");

    let mut seen_customers: HashSet<i32> = HashSet::new();
    for (cid, revenue, avg_order) in combined_scores {
        seen_customers.insert(cid);
        assert!(revenue >= 900);
        assert!(avg_order >= 120);
    }

    assert_eq!(seen_customers, HashSet::from([1003, 1005]));
}
