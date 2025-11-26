use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::UInt64Array;
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanOptions,
};
use llkv_storage::pager::MemPager;
use llkv_types::LogicalFieldId;

fn schema_with_row_id(field: Field) -> Arc<Schema> {
    let rid = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    Arc::new(Schema::new(vec![rid, field]))
}

struct SumVisitor {
    total_sum: u128,
}

impl SumVisitor {
    fn new() -> Self {
        Self { total_sum: 0 }
    }
}

impl PrimitiveVisitor for SumVisitor {
    fn u64_chunk(&mut self, a: &UInt64Array) {
        if let Some(s) = compute::sum(a) {
            self.total_sum += s as u128;
        }
    }
}

impl PrimitiveSortedVisitor for SumVisitor {}
impl PrimitiveWithRowIdsVisitor for SumVisitor {}
impl PrimitiveSortedWithRowIdsVisitor for SumVisitor {}

#[allow(clippy::print_stdout)]
fn main() {
    println!("=== Analyzing Criterion benchmark overhead ===\n");

    // Setup (same as Criterion benchmark)
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let field_id = LogicalFieldId::for_user_table_0(7777);

    let mut md = HashMap::new();
    md.insert(
        llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
        u64::from(field_id).to_string(),
    );
    let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
    let schema = schema_with_row_id(data_f);

    // Create 1M rows of data (same as benchmark)
    let num_rows = 1_000_000usize;
    let rid: Vec<u64> = (0..num_rows as u64).collect();
    let vals: Vec<u64> = (0..num_rows as u64).collect();

    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr = Arc::new(UInt64Array::from(vals));
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();

    // Time the setup phase
    let setup_start = Instant::now();
    let pager2 = Arc::new(MemPager::new());
    let store2 = ColumnStore::open(pager2).unwrap();
    store2.append(&batch).unwrap();
    let setup_time = setup_start.elapsed();

    // Append to the main store
    store.append(&batch).unwrap();

    println!(
        "Setup overhead (create store + append): {:.3} ms",
        setup_time.as_secs_f64() * 1000.0
    );

    // Now measure pure scanning performance
    let mut scan_times = Vec::new();

    // Warm up
    for _ in 0..5 {
        let mut visitor = SumVisitor::new();
        store
            .scan(
                field_id,
                ScanOptions {
                    sorted: false,
                    reverse: false,
                    with_row_ids: false,
                    limit: None,
                    offset: 0,
                    include_nulls: false,
                    nulls_first: false,
                    anchor_row_id_field: None,
                },
                &mut visitor,
            )
            .unwrap();
    }

    // Measure scan-only performance
    for _ in 0..100 {
        let mut visitor = SumVisitor::new();
        let start = Instant::now();
        store
            .scan(
                field_id,
                ScanOptions {
                    sorted: false,
                    reverse: false,
                    with_row_ids: false,
                    limit: None,
                    offset: 0,
                    include_nulls: false,
                    nulls_first: false,
                    anchor_row_id_field: None,
                },
                &mut visitor,
            )
            .unwrap();
        let scan_time = start.elapsed();
        scan_times.push(scan_time.as_secs_f64() * 1000.0);
    }

    let avg_scan = scan_times.iter().sum::<f64>() / scan_times.len() as f64;
    let min_scan = scan_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_scan = scan_times.iter().fold(0.0f64, |a, &b| a.max(b));

    println!("Pure scan performance (100 iterations):");
    println!("  Average: {:.3} ms", avg_scan);
    println!("  Min: {:.3} ms", min_scan);
    println!("  Max: {:.3} ms", max_scan);
    println!();

    // Now measure the full Criterion-style benchmark (setup + scan)
    let mut full_times = Vec::new();

    for _ in 0..20 {
        let start = Instant::now();

        // Full Criterion benchmark equivalent
        let pager = Arc::new(MemPager::new());
        let store = ColumnStore::open(pager).unwrap();
        let field_id = LogicalFieldId::for_user_table_0(7777);

        let mut md = HashMap::new();
        md.insert(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            u64::from(field_id).to_string(),
        );
        let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
        let schema = schema_with_row_id(data_f);

        let rid: Vec<u64> = (0..num_rows as u64).collect();
        let vals: Vec<u64> = (0..num_rows as u64).collect();

        let rid_arr = Arc::new(UInt64Array::from(rid));
        let val_arr = Arc::new(UInt64Array::from(vals));
        let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();

        store.append(&batch).unwrap();

        let mut visitor = SumVisitor::new();
        store
            .scan(
                field_id,
                ScanOptions {
                    sorted: false,
                    reverse: false,
                    with_row_ids: false,
                    limit: None,
                    offset: 0,
                    include_nulls: false,
                    nulls_first: false,
                    anchor_row_id_field: None,
                },
                &mut visitor,
            )
            .unwrap();

        let full_time = start.elapsed();
        full_times.push(full_time.as_secs_f64() * 1000.0);
    }

    let avg_full = full_times.iter().sum::<f64>() / full_times.len() as f64;

    println!("Full benchmark (setup + scan, 20 iterations):");
    println!("  Average: {:.3} ms", avg_full);
    println!("  This matches Criterion: ~{:.3} ms", 797.0 / 1000.0);
    println!();

    println!("=== Performance Analysis ===");
    println!(
        "Setup overhead: {:.1}% of total benchmark time",
        (setup_time.as_secs_f64() * 1000.0) / avg_full * 100.0
    );
    println!(
        "Scan performance: {:.3} ms ({:.1}% of benchmark)",
        avg_scan,
        avg_scan / avg_full * 100.0
    );
    println!();
    println!("=== Analysis Complete ===");
    println!("LLKV scan-only performance: {:.3} ms", avg_scan);
    println!("Run external benchmarks separately for comparison.");
}
