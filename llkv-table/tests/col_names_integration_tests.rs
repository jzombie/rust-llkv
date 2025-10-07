use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};

use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::types::{FieldId, TableId};

#[test]
fn col_names_persist_across_reopen() {
    // Arrange: prepare pager and create a table, write a small batch with
    // human-friendly column names stored in field metadata.
    let pager = Arc::new(MemPager::default());
    const TID: TableId = 9000;
    const COL_NAME_FID: FieldId = 1;
    const COL_AGE_FID: FieldId = 2;

    // Create a table instance, append data, then drop that instance so the
    // reopened `Table::new` simulates a fresh process reading the same pager.
    {
        let table = Table::new(TID, Arc::clone(&pager)).expect("create table");

        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("name", DataType::Utf8, true).with_metadata(HashMap::from([(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                COL_NAME_FID.to_string(),
            )])),
            Field::new("age", DataType::UInt64, true).with_metadata(HashMap::from([(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                COL_AGE_FID.to_string(),
            )])),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["alice", "bob", "carol"])),
                Arc::new(UInt64Array::from(vec![30u64, 40u64, 50u64])),
            ],
        )
        .expect("build batch");

        table.append(&batch).expect("append");

        // Ensure `table` is dropped before reopen to simulate independent instances.
    }

    {
        // Act: reopen the table using the same pager to simulate a new process.
        let reopened = Table::new(TID, Arc::clone(&pager)).expect("reopen table");

        let schema_after = reopened.schema().expect("schema after reopen");

        // We don't have the original `table` instance here (it was dropped). The
        // important assertion is that the reopened schema contains the persisted
        // human-friendly column names.

        // Assert: both schemas include the human-friendly names for user columns.
        // Field 0 is row_id; user fields follow.
        assert_eq!(schema_after.fields().len(), 3);
        assert_eq!(schema_after.fields()[1].name(), "name");
        assert_eq!(schema_after.fields()[2].name(), "age");
    }
}
