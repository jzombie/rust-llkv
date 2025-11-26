use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::{Projection, ROW_ID_COLUMN_NAME};
use llkv_expr::{Expr, Filter, Operator};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::table::ScanStreamOptions;
use llkv_table::types::{FieldId, RowId, TableId};
use llkv_types::LogicalFieldId;
use std::ops::Bound;

/// Attach required `field_id` metadata to a data column field.
fn field_with_fid(name: &str, dt: DataType, fid: FieldId, nullable: bool) -> Field {
    Field::new(name, dt, nullable).with_metadata(HashMap::from([(
        llkv_table::constants::FIELD_ID_META_KEY.to_string(),
        fid.to_string(),
    )]))
}

#[allow(clippy::print_stdout)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing streaming optimization with real data...");

    // Set up test data
    let pager = Arc::new(MemPager::default());
    const TABLE_ID: TableId = 1;
    const FIELD_ID: FieldId = 10;

    let table = Table::from_id(TABLE_ID, pager)?;

    // Create test data - 10,000 rows for performance testing
    let row_count = 10_000;
    let rows: Vec<RowId> = (1..=row_count).collect();
    let values: Vec<i64> = (0..row_count).map(|i| (i * 2) as i64).collect(); // Some pattern

    println!("Creating {} rows of test data...", row_count);

    // Create schema with proper field metadata
    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        field_with_fid("test_col", DataType::Int64, FIELD_ID, false),
    ]));

    // Create RecordBatch
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(rows)),
            Arc::new(Int64Array::from(values)),
        ],
    )?;

    // Insert data
    table.append(&batch)?;
    println!("Data inserted successfully!");

    // Now test the streaming scan
    let lfid = LogicalFieldId::for_user(TABLE_ID, FIELD_ID);
    let projection = Projection::with_alias(lfid, "test_col");

    // Create unbounded filter (should trigger streaming optimization)
    let filter_expr = Expr::Pred(Filter {
        field_id: FIELD_ID,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    });

    let options = ScanStreamOptions {
        include_nulls: false,
        order: None,
        row_id_filter: None,
    };

    println!("Starting streaming scan...");

    let mut batch_count = 0;
    let mut total_rows = 0;
    let mut sum: i64 = 0;

    let start = Instant::now();

    table.scan_stream(&[projection], &filter_expr, options.clone(), |batch| {
        batch_count += 1;
        total_rows += batch.num_rows();

        // Sum up values to verify correctness
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..arr.len() {
            sum += arr.value(i);
        }

        if batch_count <= 3 {
            println!("Batch {}: {} rows", batch_count, batch.num_rows());
        }
    })?;

    let elapsed = start.elapsed();

    println!("Streaming scan completed:");
    println!("  Time: {:?}", elapsed);
    println!("  Batches: {}", batch_count);
    println!("  Total rows: {}", total_rows);
    println!(
        "  Sum: {} (expected: {})",
        sum,
        (row_count * (row_count - 1))
    ); // sum of 0,2,4,...,2*(n-1) = n*(n-1)
    println!(
        "  Throughput: {:.0} rows/sec",
        total_rows as f64 / elapsed.as_secs_f64()
    );

    if total_rows == row_count as usize {
        println!("✅ All rows scanned successfully!");
        if elapsed.as_millis() < 10 {
            println!("🚀 FAST! Streaming optimization likely activated");
        } else {
            println!("⚠️  Slower than expected - might be using materialization path");
        }
    } else {
        println!("❌ Row count mismatch!");
    }

    // Test 2: Use multiple projections to force the slow path
    println!("\n--- Testing slow path with multiple projections ---");

    let lfid2 = LogicalFieldId::for_user(TABLE_ID, FIELD_ID);
    let projections = vec![
        Projection::with_alias(lfid, "col1"),
        Projection::with_alias(lfid2, "col2"), // Same field but different alias to trigger slow path
    ];

    let mut batch_count = 0;
    let mut total_rows = 0;

    let start = Instant::now();

    match table.scan_stream(&projections, &filter_expr, options, |batch| {
        batch_count += 1;
        total_rows += batch.num_rows();
    }) {
        Ok(()) => {
            let elapsed = start.elapsed();
            println!("Multi-projection scan completed:");
            println!("  Time: {:?}", elapsed);
            println!("  Batches: {}", batch_count);
            println!("  Total rows: {}", total_rows);
            println!(
                "  Throughput: {:.0} rows/sec",
                total_rows as f64 / elapsed.as_secs_f64()
            );
        }
        Err(e) => {
            println!("Multi-projection scan failed: {}", e);
        }
    }

    Ok(())
}
