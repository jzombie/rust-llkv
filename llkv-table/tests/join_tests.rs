//! Integration tests for table join operations.

use arrow::array::{Int32Array, RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::join::{JoinKey, JoinOptions};
use std::collections::HashMap;
use std::sync::Arc;

/// Helper to create a test table with row_id, id, and name columns.
fn create_test_table(
    table_id: u16,
    pager: &Arc<MemPager>,
    data: Vec<(u64, i32, &str)>,
) -> Table<MemPager> {
    let table = Table::new(table_id, Arc::clone(pager)).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("id", DataType::Int32, false)
            .with_metadata(HashMap::from([("field_id".to_string(), "1".to_string())])),
        Field::new("name", DataType::Utf8, false)
            .with_metadata(HashMap::from([("field_id".to_string(), "2".to_string())])),
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
