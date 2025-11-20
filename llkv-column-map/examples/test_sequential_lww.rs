//! Test sequential LWW updates with NULL values to reproduce SQL UPDATE bug

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::{
    ColumnStore, FIELD_ID_META_KEY, GatherNullPolicy, ROW_ID_COLUMN_NAME,
};
use llkv_column_map::types::LogicalFieldId;
use llkv_storage::pager::MemPager;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager))?;

    // Use field_id 4294967297 (table_id=1, field_id=1) like SQL tables do
    let field_id = LogicalFieldId::for_user(1, 1);
    println!("Using field_id: {}", u64::from(field_id));

    let mut md = HashMap::new();
    md.insert(
        FIELD_ID_META_KEY.to_string(),
        u64::from(field_id).to_string(),
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("x", DataType::Int64, true).with_metadata(md),
    ]));

    // Step 1: Insert [1, 0, NULL] for row_ids [1, 2, 3]
    println!("\n=== STEP 1: Initial INSERT ===");
    {
        let rowid_array: ArrayRef = Arc::new(UInt64Array::from(vec![1, 2, 3]));
        let value_array: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(0), None]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid_array, value_array])?;
        println!("Inserting batch:");
        println!("  rowids: {:?}", batch.column(0));
        println!("  values: {:?}", batch.column(1));

        store.append(&batch)?;
    }

    // Verify initial state
    println!("\nAfter INSERT, gathering all 3 rows:");
    let batch = store.gather_rows(&[field_id], &[1, 2, 3], GatherNullPolicy::IncludeNulls)?;
    println!("  Result: {:?}", batch.column(0));

    // Step 2: Update row 1 to value 2 (simulating WHERE clause update)
    println!("\n=== STEP 2: UPDATE row 1 SET x=2 (WHERE clause) ===");
    {
        let rowid_array: ArrayRef = Arc::new(UInt64Array::from(vec![1]));
        let value_array: ArrayRef = Arc::new(Int64Array::from(vec![2]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid_array, value_array])?;
        println!("Updating batch:");
        println!("  rowids: {:?}", batch.column(0));
        println!("  values: {:?}", batch.column(1));

        store.append(&batch)?;
    }

    // Verify after first update
    println!("\nAfter UPDATE row 1, gathering all 3 rows:");
    let batch = store.gather_rows(&[field_id], &[1, 2, 3], GatherNullPolicy::IncludeNulls)?;
    println!("  Result: {:?}", batch.column(0));

    // Step 3: Update all 3 rows to value 99 (simulating no-WHERE update)
    println!("\n=== STEP 3: UPDATE all rows SET x=99 (no WHERE) ===");
    {
        let rowid_array: ArrayRef = Arc::new(UInt64Array::from(vec![1, 2, 3]));
        let value_array: ArrayRef = Arc::new(Int64Array::from(vec![99, 99, 99]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid_array, value_array])?;
        println!("Updating batch:");
        println!("  rowids: {:?}", batch.column(0));
        println!("  values: {:?}", batch.column(1));

        store.append(&batch)?;
    }

    // Verify final state
    println!("\n=== FINAL STATE ===");
    println!("After UPDATE all rows, gathering all 3 rows:");
    let batch = store.gather_rows(&[field_id], &[1, 2, 3], GatherNullPolicy::IncludeNulls)?;
    println!("  Result: {:?}", batch.column(0));

    // Gather one at a time to see if there's a difference
    println!("\nGathering individually:");
    for rid in [1, 2, 3] {
        let batch = store.gather_rows(&[field_id], &[rid], GatherNullPolicy::IncludeNulls)?;
        println!("  Row {}: {:?}", rid, batch.column(0));
    }

    // Check if row 3 has value 99
    let final_batch = store.gather_rows(&[field_id], &[1, 2, 3], GatherNullPolicy::IncludeNulls)?;
    let values = final_batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    println!("\nFinal verification:");
    println!("  Row 1 value: {:?}", values.value(0));
    println!("  Row 2 value: {:?}", values.value(1));
    println!(
        "  Row 3 value: {:?} (is_null: {})",
        if values.is_null(2) {
            "NULL"
        } else {
            &values.value(2).to_string()
        },
        values.is_null(2)
    );

    if values.is_null(2) {
        println!("\n❌ BUG REPRODUCED: Row 3 is NULL after UPDATE!");
        std::process::exit(1);
    } else if values.value(2) == 99 {
        println!("\n✅ SUCCESS: Row 3 has value 99!");
    } else {
        println!(
            "\n❌ UNEXPECTED: Row 3 has value {} (expected 99)",
            values.value(2)
        );
        std::process::exit(1);
    }

    Ok(())
}
