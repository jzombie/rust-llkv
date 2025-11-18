use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::{Projection, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::{Expr, Filter, Operator};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::table::ScanStreamOptions;
use llkv_table::types::{FieldId, RowId, TableId};
use std::ops::Bound;

/// Attach required `field_id` metadata to a data column field.
fn field_with_fid(name: &str, dt: DataType, fid: FieldId, nullable: bool) -> Field {
    Field::new(name, dt, nullable).with_metadata(HashMap::from([(
        llkv_table::constants::FIELD_ID_META_KEY.to_string(),
        fid.to_string(),
    )]))
}

#[allow(clippy::print_stdout)]
fn benchmark_scenario(
    name: &str,
    table: &Table<MemPager>,
    projections: &[Projection],
    filter_expr: &Expr<FieldId>,
    options: ScanStreamOptions,
) {
    println!("\n=== {} ===", name);

    let mut batch_count = 0;
    let mut total_rows = 0;
    let mut total_values: i64 = 0;

    let start = Instant::now();

    match table.scan_stream(projections, filter_expr, options, |batch| {
        batch_count += 1;
        total_rows += batch.num_rows();

        // Sum first column to verify correctness
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..arr.len() {
            total_values += arr.value(i);
        }
    }) {
        Ok(()) => {
            let elapsed = start.elapsed();
            println!("  Time: {:?}", elapsed);
            println!("  Batches: {}", batch_count);
            println!("  Total rows: {}", total_rows);
            println!("  Sum: {}", total_values);
            println!(
                "  Throughput: {:.0} rows/sec",
                total_rows as f64 / elapsed.as_secs_f64()
            );

            if elapsed.as_millis() < 5 {
                println!("  🚀 VERY FAST - likely using optimized path");
            } else if elapsed.as_millis() < 20 {
                println!("  ✅ Fast - good performance");
            } else {
                println!("  ⚠️  Slower - might be using materialization");
            }
        }
        Err(e) => {
            println!("  ❌ Failed: {}", e);
        }
    }
}

#[allow(clippy::print_stdout)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Table Layer Performance Optimization Benchmark");
    println!("============================================");

    // Set up test data
    let pager = Arc::new(MemPager::default());
    const TABLE_ID: TableId = 1;
    const FIELD_A: FieldId = 10;
    const FIELD_B: FieldId = 11;

    let table = Table::from_id(TABLE_ID, pager)?;

    // Create test data - 1,000,000 rows to match column-map benchmarks
    let row_count = 1_000_000;
    let rows: Vec<RowId> = (1..=row_count).collect();
    let values_a: Vec<i64> = (0..row_count).map(|i| (i * 2) as i64).collect();
    let values_b: Vec<i64> = (0..row_count).map(|i| (i * 3) as i64).collect();

    println!("Creating {} rows of test data...", row_count);

    // Create schema with two columns
    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        field_with_fid("col_a", DataType::Int64, FIELD_A, false),
        field_with_fid("col_b", DataType::Int64, FIELD_B, false),
    ]));

    // Create RecordBatch
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(rows)),
            Arc::new(Int64Array::from(values_a)),
            Arc::new(Int64Array::from(values_b)),
        ],
    )?;

    // Insert data
    table.append(&batch)?;
    println!("Data inserted successfully!");

    // Benchmark different scenarios
    let lfid_a = LogicalFieldId::for_user(TABLE_ID, FIELD_A);
    let lfid_b = LogicalFieldId::for_user(TABLE_ID, FIELD_B);

    // 1. Single column, unbounded scan (should use fastest path)
    let single_proj = vec![Projection::with_alias(lfid_a, "col_a")];
    let unbounded_filter = Expr::Pred(Filter {
        field_id: FIELD_A,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    });
    let default_options = ScanStreamOptions {
        include_nulls: false,
        order: None,
        row_id_filter: None,
    };

    benchmark_scenario(
        "Single Column, Unbounded Scan",
        &table,
        &single_proj,
        &unbounded_filter,
        default_options.clone(),
    );

    // 2. Multiple columns (should use materialization path)
    let multi_proj = vec![
        Projection::with_alias(lfid_a, "col_a"),
        Projection::with_alias(lfid_b, "col_b"),
    ];

    benchmark_scenario(
        "Multi-Column Scan",
        &table,
        &multi_proj,
        &unbounded_filter,
        default_options.clone(),
    );

    // 3. Single column with nulls included (should bypass streaming optimization)
    let with_nulls_options = ScanStreamOptions {
        include_nulls: true,
        order: None,
        row_id_filter: None,
    };

    benchmark_scenario(
        "Single Column with Nulls",
        &table,
        &single_proj,
        &unbounded_filter,
        with_nulls_options.clone(),
    );

    // 4. Single column with bounded filter (should bypass simple optimizations)
    let bounded_filter = Expr::Pred(Filter {
        field_id: FIELD_A,
        op: Operator::Range {
            lower: Bound::Included(1000.into()),
            upper: Bound::Excluded(2000.into()),
        },
    });

    #[allow(clippy::print_stdout)]
    benchmark_scenario(
        "Single Column, Bounded Filter",
        &table,
        &single_proj,
        &bounded_filter,
        default_options,
    );

    println!("\n=== Summary ===");
    println!("This benchmark shows different optimization paths in the Table layer:");
    println!("• Single column unbounded scans should be fastest (direct streaming)");
    println!("• Multi-column scans use materialization but are still fast");
    println!("• Complex filters and null handling add overhead");
    println!("• All scenarios should be significantly faster than 10x overhead we had before");

    Ok(())
}
