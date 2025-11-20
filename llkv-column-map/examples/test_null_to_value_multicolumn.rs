//! Test updating NULL to value in a multi-column scenario

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, StringArray, UInt64Array};
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

    let field_id_x = LogicalFieldId::for_user(1, 1);
    let field_id_y = LogicalFieldId::for_user(1, 2);

    let mut md_x = HashMap::new();
    md_x.insert(
        FIELD_ID_META_KEY.to_string(),
        u64::from(field_id_x).to_string(),
    );

    let mut md_y = HashMap::new();
    md_y.insert(
        FIELD_ID_META_KEY.to_string(),
        u64::from(field_id_y).to_string(),
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("x", DataType::Int64, true).with_metadata(md_x),
        Field::new("y", DataType::Utf8, true).with_metadata(md_y),
    ]));

    // Step 1: INSERT 3 rows like SQL does
    println!("=== INSERT 3 rows ===");
    {
        let rowid = Arc::new(UInt64Array::from(vec![1, 2, 3]));
        let x = Arc::new(Int64Array::from(vec![Some(1), Some(0), None]));
        let y = Arc::new(StringArray::from(vec![
            Some("true"),
            Some("false"),
            Some("NULL"),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid, x, y])?;
        store.append(&batch)?;
        println!("Inserted: x={:?}, y={:?}", batch.column(1), batch.column(2));
    }

    // Step 2: UPDATE row 1 several times (like WHERE clause updates)
    println!("\n=== UPDATE row 1 (WHERE x>0) ===");
    {
        let rowid = Arc::new(UInt64Array::from(vec![1]));
        let x = Arc::new(Int64Array::from(vec![2]));
        let y = Arc::new(StringArray::from(vec![Some("true")]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid, x, y])?;
        store.append(&batch)?;
    }

    {
        let rowid = Arc::new(UInt64Array::from(vec![1]));
        let x = Arc::new(Int64Array::from(vec![2]));
        let y = Arc::new(StringArray::from(vec![Some("unknown")]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid, x, y])?;
        store.append(&batch)?;
    }

    println!("After WHERE updates:");
    let batch = store.gather_rows(
        &[field_id_x, field_id_y],
        &[1, 2, 3],
        GatherNullPolicy::IncludeNulls,
    )?;
    println!("  x={:?}", batch.column(0));
    println!("  y={:?}", batch.column(1));

    // Step 3: UPDATE all 3 rows x=99 (keeping current y values)
    println!("\n=== UPDATE all rows SET x=99 ===");
    {
        let rowid = Arc::new(UInt64Array::from(vec![1, 2, 3]));
        let x = Arc::new(Int64Array::from(vec![99, 99, 99]));
        let y = Arc::new(StringArray::from(vec![
            Some("unknown"),
            Some("false"),
            Some("NULL"),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid, x, y])?;
        println!(
            "Appending: rowid={:?}, x={:?}, y={:?}",
            batch.column(0),
            batch.column(1),
            batch.column(2)
        );
        store.append(&batch)?;
    }

    // Step 4: Gather and verify
    println!("\n=== VERIFICATION ===");
    let batch = store.gather_rows(
        &[field_id_x, field_id_y],
        &[1, 2, 3],
        GatherNullPolicy::IncludeNulls,
    )?;
    let x_vals = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let y_vals = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    println!("Final state:");
    for i in 0..3 {
        println!(
            "  Row {}: x={:?}, y={:?}",
            i + 1,
            if x_vals.is_null(i) {
                "NULL".to_string()
            } else {
                x_vals.value(i).to_string()
            },
            if y_vals.is_null(i) {
                "NULL".to_string()
            } else {
                y_vals.value(i).to_string()
            }
        );
    }

    // Check row 3
    if x_vals.is_null(2) {
        println!("\n❌ BUG REPRODUCED: Row 3 x is NULL!");
        println!("Expected: x=99, y='NULL'");
        println!(
            "Got: x=NULL, y={:?}",
            if y_vals.is_null(2) {
                "NULL"
            } else {
                y_vals.value(2)
            }
        );
        std::process::exit(1);
    }

    if x_vals.value(2) != 99 {
        println!(
            "\n❌ Row 3 x has wrong value: {} (expected 99)",
            x_vals.value(2)
        );
        std::process::exit(1);
    }

    println!("\n✅ SUCCESS: All rows updated correctly!");
    Ok(())
}
