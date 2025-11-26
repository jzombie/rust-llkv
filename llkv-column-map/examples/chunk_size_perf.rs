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

struct PerfTester {
    total_sum: u128,
    chunk_count: usize,
}

impl PerfTester {
    fn new() -> Self {
        Self {
            total_sum: 0,
            chunk_count: 0,
        }
    }
}

impl PrimitiveVisitor for PerfTester {
    fn u64_chunk(&mut self, a: &UInt64Array) {
        self.chunk_count += 1;
        if let Some(s) = compute::sum(a) {
            self.total_sum += s as u128;
        }
    }
}

impl PrimitiveSortedVisitor for PerfTester {}
impl PrimitiveWithRowIdsVisitor for PerfTester {}
impl PrimitiveSortedWithRowIdsVisitor for PerfTester {}

#[allow(clippy::print_stdout)]
fn test_chunk_size(target_mb: usize, name: &str) {
    println!("=== Testing {} ===", name);

    // Temporarily modify the TARGET_CHUNK_BYTES by creating stores with different batch sizes
    // Since we can't easily change the constant, we'll simulate different effective chunk sizes
    // by varying how we send the data

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

    let total_rows = 1_000_000u64;
    let target_bytes = target_mb * 1024 * 1024;
    let rows_per_batch = (target_bytes / 8).min(total_rows as usize); // 8 bytes per u64
    let num_batches = (total_rows as usize).div_ceil(rows_per_batch);

    println!("  Rows per batch: {}", rows_per_batch);
    println!("  Number of batches: {}", num_batches);

    // Insert data
    let start_insert = Instant::now();
    let mut row_offset = 0u64;

    for _ in 0..num_batches {
        let batch_size = (total_rows - row_offset).min(rows_per_batch as u64) as usize;
        if batch_size == 0 {
            break;
        }

        let rid: Vec<u64> = (row_offset..row_offset + batch_size as u64).collect();
        let vals: Vec<u64> = (row_offset..row_offset + batch_size as u64).collect();

        let rid_arr = Arc::new(UInt64Array::from(rid));
        let val_arr = Arc::new(UInt64Array::from(vals));
        let batch = RecordBatch::try_new(schema.clone(), vec![rid_arr, val_arr]).unwrap();

        store.append(&batch).unwrap();
        row_offset += batch_size as u64;
    }
    let insert_time = start_insert.elapsed();

    // Scan and measure performance
    let start_scan = Instant::now();
    let mut tester = PerfTester::new();
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
            &mut tester,
        )
        .unwrap();
    let scan_time = start_scan.elapsed();

    println!("  Insert time: {:?}", insert_time);
    println!("  Scan time: {:?}", scan_time);
    println!("  Chunks created: {}", tester.chunk_count);
    println!(
        "  Avg rows per chunk: {}",
        total_rows / tester.chunk_count as u64
    );
    println!("  Sum result: {}", tester.total_sum);
    println!(
        "  Scan throughput: {:.1} M rows/sec",
        total_rows as f64 / scan_time.as_secs_f64() / 1_000_000.0
    );
    println!();
}

#[allow(clippy::print_stdout)]
fn main() {
    println!("Testing different effective chunk sizes...\n");

    // Test different effective chunk sizes by varying batch sizes
    test_chunk_size(1, "1MB batches (current default)");
    test_chunk_size(4, "4MB batches");
    test_chunk_size(16, "16MB batches");
    test_chunk_size(64, "64MB batches");
    test_chunk_size(1000, "Single 1GB batch (all data at once)");

    println!("Note: Run corresponding Python benchmarks separately for comparison.");
}
