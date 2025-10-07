use arrow::array::{Array, FixedSizeListArray, Float32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::LogicalFieldId;
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn ingest_and_scan_multiple_columns() {
    // 1. Set up the store with an in-memory pager.
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    // 2. Define the first column (u64 scalars) and its metadata.
    // The "field_id" metadata is critical for the store to identify
    // the column. We now construct a full LogicalFieldId for it.
    let field_id_1 = LogicalFieldId::for_user_table_0(101);
    let mut metadata1 = HashMap::new();
    metadata1.insert(
        llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
        u64::from(field_id_1).to_string(),
    );
    let field1 = Field::new("user_id", DataType::UInt64, false).with_metadata(metadata1);
    let array1 = Arc::new(UInt64Array::from(vec![1001, 1002, 1003]));

    // 3. Define the second column (2D f32 vectors) and its metadata.
    let field_id_2 = LogicalFieldId::for_user_table_0(102);
    let mut metadata2 = HashMap::new();
    metadata2.insert(
        llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
        u64::from(field_id_2).to_string(),
    );
    let list_value_field = Arc::new(Field::new("item", DataType::Float32, true));
    let field2_type = DataType::FixedSizeList(list_value_field.clone(), 2);
    let field2 = Field::new("embedding", field2_type, true).with_metadata(metadata2);
    let values = Arc::new(Float32Array::from(vec![1.1, 1.2, 2.1, 2.2, 3.1, 3.2]));

    let array2 = Arc::new(FixedSizeListArray::new(
        list_value_field, // Field describing the list items
        2,                // The size of each list
        values,           // The flattened values array
        None,             // No nulls in this example
    ));

    // 3b. Add required row_id column (UInt64, non-null, same length).
    let row_id_field = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    let row_ids = Arc::new(UInt64Array::from(vec![0u64, 1, 2]));

    // 4. Create the RecordBatch from the schema and arrays.
    // The schema must include the row_id column.
    let schema = Arc::new(Schema::new(vec![row_id_field, field1, field2]));
    let batch = RecordBatch::try_new(schema, vec![row_ids, array1, array2.clone()]).unwrap();

    // 5. Ingest the entire batch.
    store.append(&batch).unwrap();

    // 6. Verify the first column (field_id: 101).
    let mut results1_iter = store.scan(field_id_1).unwrap();
    let scanned_array1 = results1_iter
        .next()
        .expect("iterator should have one batch")
        .unwrap();
    // Cast the generic ArrayRef to the concrete type and check contents.
    let scanned_u64_array = scanned_array1
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("array should be a UInt64Array");
    assert_eq!(scanned_u64_array.values(), &[1001, 1002, 1003]);
    assert!(results1_iter.next().is_none(), "iterator should be empty");

    // 7. Verify the second column (field_id: 102).
    let mut results2_iter = store.scan(field_id_2).unwrap();
    let scanned_array2 = results2_iter
        .next()
        .expect("iterator should have one batch")
        .unwrap();
    // Cast to the concrete vector type and check its contents.
    let scanned_fsl_array = scanned_array2
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("array should be a FixedSizeListArray");
    // Sanity: fixed-size=2, row count=3
    assert_eq!(scanned_fsl_array.value_length(), 2);
    assert_eq!(scanned_fsl_array.len(), 3);
    // Compare flattened child values (ignores field metadata differences)
    let scanned_child = scanned_fsl_array
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("child must be Float32");
    let expected_child = array2
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("child must be Float32");
    assert_eq!(scanned_child.len(), expected_child.len());
    for i in 0..expected_child.len() {
        // Float compare; these are constructed from the same literals so exact is fine.
        // Use approx if you prefer:
        // assert!((scanned_child.value(i) - expected_child.value(i)).abs() < 1e-6);
        assert_eq!(scanned_child.value(i), expected_child.value(i));
    }

    assert!(results2_iter.next().is_none(), "iterator should be empty");
}
