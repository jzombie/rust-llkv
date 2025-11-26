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
use llkv_types::LogicalFieldId;
use llkv_storage::pager::MemPager;

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

fn benchmark_rust_llkv() -> (u128, f64) {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let field_id = LogicalFieldId::for_user_table_0(9999);

    let mut md = HashMap::new();
    md.insert(
        llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
        u64::from(field_id).to_string(),
    );
    let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
    let schema = schema_with_row_id(data_f);

    // Create the same data as Python: 1M integers from 0-99
    let total_rows = 1_000_000u64;
    let mut data = Vec::with_capacity(total_rows as usize);
    for i in 0..total_rows {
        data.push(i % 100); // Same pattern as Python: 0-99 repeating
    }

    let rid: Vec<u64> = (0..total_rows).collect();
    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr = Arc::new(UInt64Array::from(data));
    let batch = RecordBatch::try_new(schema.clone(), vec![rid_arr, val_arr]).unwrap();

    // Insert the data
    store.append(&batch).unwrap();

    // Warm up with a few runs
    for _ in 0..3 {
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

    // Now measure 100 iterations like Python
    let mut times = Vec::new();
    let mut final_sum = 0u128;

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
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64() * 1000.0); // Convert to ms
        final_sum = visitor.total_sum;
    }

    let avg_time = times.iter().sum::<f64>() / times.len() as f64;
    (final_sum, avg_time)
}

fn benchmark_arrow_direct() -> (u128, f64) {
    // Direct Arrow benchmark - no storage layer
    let total_rows = 1_000_000u64;
    let mut data = Vec::with_capacity(total_rows as usize);
    for i in 0..total_rows {
        data.push(i % 100); // Same pattern as Python
    }

    let arr = UInt64Array::from(data);

    // Warm up
    for _ in 0..3 {
        let _ = compute::sum(&arr);
    }

    // Measure 100 iterations
    let mut times = Vec::new();
    let mut final_sum = 0u128;

    for _ in 0..100 {
        let start = Instant::now();
        if let Some(s) = compute::sum(&arr) {
            final_sum = s as u128;
        }
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64() * 1000.0); // Convert to ms
    }

    let avg_time = times.iter().sum::<f64>() / times.len() as f64;
    (final_sum, avg_time)
}

#[allow(clippy::print_stdout)]
fn main() {
    println!("=== Direct comparison with Python benchmarks ===\n");

    println!("Testing direct Arrow compute::sum (no storage overhead)...");
    let (arrow_sum, arrow_time) = benchmark_arrow_direct();
    println!(
        "Arrow direct sum: {} (avg of 100 runs: {:.3} ms)\n",
        arrow_sum, arrow_time
    );

    println!("Testing LLKV ColumnStore scan...");
    let (llkv_sum, llkv_time) = benchmark_rust_llkv();
    println!(
        "LLKV scan sum: {} (avg of 100 runs: {:.3} ms)\n",
        llkv_sum, llkv_time
    );

    println!(
        "Storage overhead: {:.1}x (LLKV vs Arrow direct)",
        llkv_time / arrow_time
    );
    println!();
    println!("Run the Python benchmark separately for external comparison.");
}
