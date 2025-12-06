//! Integration tests for table join operations.

use arrow::array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use llkv_column_map::store::Projection;
use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_expr::{CompareOp, Expr, ScalarExpr};
use llkv_join::{
    JoinIndexBatch, JoinKey, JoinOptions, JoinSide, JoinType, TableJoinRowIdExt,
    project_join_columns,
};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::table::{ScanProjection, ScanStreamOptions};
use llkv_table::types::TableId;
use llkv_types::LogicalFieldId;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Materialize a `JoinIndexBatch` into a `RecordBatch` for assertions using the
/// projection helpers to keep contiguous slices zero-copy.
fn materialize_join_index_batch(
    batch: &JoinIndexBatch<'_>,
    output_schema: &Arc<Schema>,
    join_type: JoinType,
) -> RecordBatch {
    let include_right = !matches!(join_type, JoinType::Semi | JoinType::Anti);

    let left_col_count = batch.left_batch.num_columns();
    let left_projection: Vec<usize> = (0..left_col_count).collect();
    let mut arrays: Vec<ArrayRef> =
        project_join_columns(batch, JoinSide::Left, &left_projection).expect("left projection");

    if include_right {
        let right_col_count = batch
            .right_batches
            .first()
            .map(|b| b.num_columns())
            .unwrap_or(0);
        let right_projection: Vec<usize> = (0..right_col_count).collect();
        let right_arrays = project_join_columns(batch, JoinSide::Right, &right_projection)
            .expect("right projection");
        arrays.extend(right_arrays);
    }

    RecordBatch::try_new(output_schema.clone(), arrays).unwrap()
}

fn build_output_schema(
    left_schema: &Schema,
    right_schema: &Schema,
    join_type: JoinType,
) -> Arc<Schema> {
    let mut fields = Vec::new();
    let mut field_names: HashSet<String> = HashSet::new();

    if matches!(join_type, JoinType::Semi | JoinType::Anti) {
        for field in left_schema.fields() {
            if field
                .metadata()
                .get(llkv_column_map::store::FIELD_ID_META_KEY)
                .is_some()
            {
                fields.push(field.clone());
                field_names.insert(field.name().clone());
            }
        }
        return Arc::new(Schema::new(fields));
    }

    for field in left_schema.fields() {
        if field
            .metadata()
            .get(llkv_column_map::store::FIELD_ID_META_KEY)
            .is_some()
        {
            fields.push(field.clone());
            field_names.insert(field.name().clone());
        }
    }

    for field in right_schema.fields() {
        if field
            .metadata()
            .get(llkv_column_map::store::FIELD_ID_META_KEY)
            .is_some()
        {
            let field_name = field.name();
            let new_name = if field_names.contains(field_name) {
                format!("{}_1", field_name)
            } else {
                field_name.clone()
            };

            let new_field = Arc::new(
                arrow::datatypes::Field::new(
                    new_name.clone(),
                    field.data_type().clone(),
                    field.is_nullable(),
                )
                .with_metadata(field.metadata().clone()),
            );

            fields.push(new_field);
            field_names.insert(new_name);
        }
    }

    Arc::new(Schema::new(fields))
}

fn run_join(
    left: &Table<MemPager>,
    right: &Table<MemPager>,
    keys: &[JoinKey],
    options: &JoinOptions,
) -> Vec<RecordBatch> {
    let output_schema = build_output_schema(
        &left.schema().unwrap(),
        &right.schema().unwrap(),
        options.join_type,
    );
    let mut out = Vec::new();
    left.join_rowid_stream(right, keys, options, |index_batch| {
        let batch = materialize_join_index_batch(&index_batch, &output_schema, options.join_type);
        out.push(batch);
    })
    .unwrap();
    out
}

/// Helper to create a test table with row-id, user_id, and name columns.
fn create_test_table(
    table_id: TableId,
    pager: &Arc<MemPager>,
    data: Vec<(u64, i32, &str)>,
) -> Table<MemPager> {
    let table = Table::from_id(table_id, Arc::clone(pager)).unwrap();

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

    let result_batches = run_join(&left, &right, &keys, &options);

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

    let result_batches = run_join(&left, &right, &keys, &options);

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

    let result_batches = run_join(&left, &right, &keys, &options);

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

    let result_batches = run_join(&left, &right, &keys, &options);

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

    let result_batches = run_join(&left, &right, &keys, &options);

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

    let result_batches = run_join(&left, &right, &keys, &options);

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

    let result_batches = run_join(&left, &right, &keys, &options);

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

    let result_batches = run_join(&left, &right, &keys, &options);

    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0, "Join with no matches should produce 0 rows");
}

#[test]
fn test_join_validation_errors() {
    let pager = Arc::new(MemPager::default());
    let left = create_test_table(1, &pager, vec![(0, 1, "Alice")]);
    let right = create_test_table(2, &pager, vec![(0, 1, "Alpha")]);

    // Empty keys is valid (cross product) - should succeed
    let result = left.join_rowid_stream(&right, &[], &JoinOptions::default(), |_| {});
    assert!(result.is_ok());

    // Bad batch size should error
    let bad_opts = JoinOptions {
        batch_size: 0,
        ..Default::default()
    };
    let result = left.join_rowid_stream(&right, &[JoinKey::new(1, 1)], &bad_opts, |_| {});
    assert!(result.is_err());
}

fn proj(table: &Table<MemPager>, field_id: llkv_table::types::FieldId) -> Projection {
    Projection::from(LogicalFieldId::for_user(table.table_id(), field_id))
}

#[test]
fn test_join_with_expression_filters() {
    const LEFT_TABLE_ID: TableId = 41;
    const RIGHT_TABLE_ID: TableId = 84;

    const LEFT_CUSTOMER_ID: llkv_table::types::FieldId = 11;
    const LEFT_SEGMENT: llkv_table::types::FieldId = 12;
    const LEFT_ANNUAL_REVENUE: llkv_table::types::FieldId = 13;
    const LEFT_LOYALTY: llkv_table::types::FieldId = 14;

    const RIGHT_ORDER_ID: llkv_table::types::FieldId = 21;
    const RIGHT_CUSTOMER_ID: llkv_table::types::FieldId = 22;
    const RIGHT_AVG_ORDER_VALUE: llkv_table::types::FieldId = 23;
    const RIGHT_TRAILING_SPEND: llkv_table::types::FieldId = 24;

    let pager = Arc::new(MemPager::default());

    let customer_table = Table::from_id(LEFT_TABLE_ID, Arc::clone(&pager)).unwrap();
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

    let orders_table = Table::from_id(RIGHT_TABLE_ID, Arc::clone(&pager)).unwrap();
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

    let joined_batches = run_join(&customer_table, &orders_table, &join_keys, &options);

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

// ============================================================================
// Cartesian Product / Cross Join Tests
// ============================================================================

#[test]
fn test_cartesian_product_basic() {
    let pager = Arc::new(MemPager::default());
    let left = create_test_table(1, &pager, vec![(0, 1, "A"), (1, 2, "B")]);
    let right = create_test_table(2, &pager, vec![(0, 10, "X"), (1, 20, "Y")]);

    let result_batches = run_join(&left, &right, &[], &JoinOptions::default());

    // Should produce 2 × 2 = 4 rows
    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 4,
        "Cartesian product should produce 2 × 2 = 4 rows"
    );

    // Verify we have columns from both tables
    let schema = result_batches[0].schema();
    assert!(schema.column_with_name("user_id").is_some());
    assert!(schema.column_with_name("name").is_some());
    assert!(schema.column_with_name("user_id_1").is_some());
    assert!(schema.column_with_name("name_1").is_some());
}

#[test]
fn test_cartesian_product_asymmetric() {
    let pager = Arc::new(MemPager::default());
    // Left: 3 rows
    let left = create_test_table(1, &pager, vec![(0, 1, "A"), (1, 2, "B"), (2, 3, "C")]);
    // Right: 2 rows
    let right = create_test_table(2, &pager, vec![(0, 10, "X"), (1, 20, "Y")]);

    let result_batches = run_join(&left, &right, &[], &JoinOptions::default());

    // Should produce 3 × 2 = 6 rows
    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 6,
        "Cartesian product should produce 3 × 2 = 6 rows"
    );
}

#[test]
fn test_cartesian_product_with_filters() {
    let pager = Arc::new(MemPager::default());
    let left = create_test_table(1, &pager, vec![(0, 1, "A"), (1, 2, "B"), (2, 3, "C")]);
    let right = create_test_table(2, &pager, vec![(0, 10, "X"), (1, 20, "Y"), (2, 30, "Z")]);

    let mut all_rows = Vec::new();

    for batch in run_join(&left, &right, &[], &JoinOptions::default()) {
        let left_ids = batch
            .column_by_name("user_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let right_ids = batch
            .column_by_name("user_id_1")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        for i in 0..batch.num_rows() {
            all_rows.push((left_ids.value(i), right_ids.value(i)));
        }
    }

    // Should produce 3 × 3 = 9 rows
    assert_eq!(all_rows.len(), 9);

    // Verify all combinations exist
    let expected_combinations: HashSet<(i32, i32)> = [
        (1, 10),
        (1, 20),
        (1, 30),
        (2, 10),
        (2, 20),
        (2, 30),
        (3, 10),
        (3, 20),
        (3, 30),
    ]
    .iter()
    .cloned()
    .collect();

    let actual_combinations: HashSet<(i32, i32)> = all_rows.into_iter().collect();
    assert_eq!(actual_combinations, expected_combinations);
}

#[test]
fn test_cartesian_product_with_empty_left() {
    let pager = Arc::new(MemPager::default());
    let left = create_test_table(1, &pager, vec![]); // Empty
    let right = create_test_table(2, &pager, vec![(0, 10, "X"), (1, 20, "Y")]);

    let result_batches = run_join(&left, &right, &[], &JoinOptions::default());

    // Empty left table means no results
    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 0,
        "Cartesian product with empty left should produce 0 rows"
    );
}

#[test]
fn test_cartesian_product_with_empty_right() {
    let pager = Arc::new(MemPager::default());
    let left = create_test_table(1, &pager, vec![(0, 1, "A"), (1, 2, "B")]);
    let right = create_test_table(2, &pager, vec![]); // Empty

    let result_batches = run_join(&left, &right, &[], &JoinOptions::default());

    // Empty right table means no results
    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 0,
        "Cartesian product with empty right should produce 0 rows"
    );
}

#[test]
fn test_cartesian_product_larger_dataset() {
    let pager = Arc::new(MemPager::default());

    // Create larger tables
    let left_data: Vec<_> = (0..10).map(|i| (i, i as i32, "L")).collect();
    let right_data: Vec<_> = (0..8).map(|i| (i, (i * 10) as i32, "R")).collect();

    let left = create_test_table(1, &pager, left_data);
    let right = create_test_table(2, &pager, right_data);

    let result_batches = run_join(&left, &right, &[], &JoinOptions::default());

    // Should produce 10 × 8 = 80 rows
    let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 80,
        "Cartesian product should produce 10 × 8 = 80 rows"
    );
}

#[test]
fn test_cartesian_product_data_integrity() {
    let pager = Arc::new(MemPager::default());
    let left = create_test_table(1, &pager, vec![(0, 1, "Alice"), (1, 2, "Bob")]);
    let right = create_test_table(2, &pager, vec![(0, 100, "X"), (1, 200, "Y")]);

    let mut all_rows = Vec::new();

    for batch in run_join(&left, &right, &[], &JoinOptions::default()) {
        let left_ids = batch
            .column_by_name("user_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let left_names = batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let right_ids = batch
            .column_by_name("user_id_1")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let right_names = batch
            .column_by_name("name_1")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        for i in 0..batch.num_rows() {
            all_rows.push((
                left_ids.value(i),
                left_names.value(i).to_string(),
                right_ids.value(i),
                right_names.value(i).to_string(),
            ));
        }
    }

    assert_eq!(all_rows.len(), 4);

    // Verify data integrity - all combinations should exist with correct data
    let expected = [
        (1, "Alice", 100, "X"),
        (1, "Alice", 200, "Y"),
        (2, "Bob", 100, "X"),
        (2, "Bob", 200, "Y"),
    ];

    for (exp_lid, exp_lname, exp_rid, exp_rname) in &expected {
        assert!(
            all_rows.iter().any(|(lid, lname, rid, rname)| {
                lid == exp_lid && lname == exp_lname && rid == exp_rid && rname == exp_rname
            }),
            "Missing expected combination: {:?}",
            (exp_lid, exp_lname, exp_rid, exp_rname)
        );
    }
}
