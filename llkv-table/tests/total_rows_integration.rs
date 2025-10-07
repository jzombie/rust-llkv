use std::sync::Arc;

use arrow::array::{StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_column_map::types::LogicalFieldId;
use llkv_storage::pager::MemPager;
use llkv_table::Table;

#[test]
fn test_total_rows_before_after_append_and_delete() {
    // Create table with a single user column and append rows with explicit row_ids
    let pager = Arc::new(MemPager::new());
    let table = Table::new(1000, Arc::clone(&pager)).expect("create table");

    // Build schema: row_id + name (field_id 1) + age (field_id 2)
    let mut md1 = std::collections::HashMap::new();
    md1.insert(
        llkv_table::constants::FIELD_ID_META_KEY.to_string(),
        "1".to_string(),
    );
    let name_field = Field::new("name", DataType::Utf8, false).with_metadata(md1);

    let mut md2 = std::collections::HashMap::new();
    md2.insert(
        llkv_table::constants::FIELD_ID_META_KEY.to_string(),
        "2".to_string(),
    );
    let age_field = Field::new("age", DataType::UInt64, false).with_metadata(md2);

    let schema = {
        let rid = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
        Arc::new(Schema::new(vec![rid, name_field, age_field]))
    };

    // Append 5 rows with row_ids 0..4 and both columns populated
    let row_ids = Arc::new(UInt64Array::from(vec![0u64, 1, 2, 3, 4]));
    let names = Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"]));
    let ages = Arc::new(UInt64Array::from(vec![10u64, 11, 12, 13, 14]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![row_ids.clone(), names.clone(), ages.clone()],
    )
    .expect("batch");
    table.append(&batch).expect("append");

    // Both columns should report 5 rows
    let col1_total = table.total_rows_for_col(1).expect("col1 total");
    let col2_total = table.total_rows_for_col(2).expect("col2 total");
    let table_total = table.total_rows().expect("table total");
    assert_eq!(col1_total, 5);
    assert_eq!(col2_total, 5);
    assert_eq!(table_total, 5);

    // Delete row with global position 2 (third row) from column 1 only
    let store = table.store();
    let lfid1 = LogicalFieldId::for_user(table.table_id(), 1);
    store.delete_rows(&[lfid1], &[2u64]).expect("delete col1");

    // col1 should be 4, col2 still 5
    let col1_after = table.total_rows_for_col(1).expect("col1 after");
    let col2_after = table.total_rows_for_col(2).expect("col2 after");
    let table_total = table.total_rows().expect("table total");
    assert_eq!(col1_after, 4);
    assert_eq!(col2_after, 5);
    assert_eq!(table_total, 5);

    // Delete global position 3 from column 2 only
    let lfid2 = LogicalFieldId::for_user(table.table_id(), 2);
    store.delete_rows(&[lfid2], &[3u64]).expect("delete col2");

    // col1 still 4, col2 should now be 4
    let col1_after2 = table.total_rows_for_col(1).expect("col1 after2");
    let col2_after2 = table.total_rows_for_col(2).expect("col2 after2");
    let table_total = table.total_rows().expect("table total");
    assert_eq!(col1_after2, 4);
    assert_eq!(col2_after2, 4);
    assert_eq!(table_total, 4);

    // Append one more row (row_id 5) with both columns
    let row_ids2 = Arc::new(UInt64Array::from(vec![5u64]));
    let names2 = Arc::new(StringArray::from(vec!["f"]));
    let ages2 = Arc::new(UInt64Array::from(vec![15u64]));
    let batch2 =
        RecordBatch::try_new(schema.clone(), vec![row_ids2, names2, ages2]).expect("batch2");
    table.append(&batch2).expect("append2");

    // Both columns should now be 5 again
    let col1_final = table.total_rows_for_col(1).expect("col1 final");
    let col2_final = table.total_rows_for_col(2).expect("col2 final");
    let table_total = table.total_rows().expect("table total");
    assert_eq!(col1_final, 5);
    assert_eq!(col2_final, 5);
    assert_eq!(table_total, 5);
}
