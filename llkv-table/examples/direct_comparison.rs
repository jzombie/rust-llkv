use std::collections::HashMap;
use std::env;
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::ScanBuilder;
use llkv_column_map::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanOptions,
};
use llkv_column_map::store::{Projection, ROW_ID_COLUMN_NAME};
use llkv_types::LogicalFieldId;
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

/// Simple visitor that just counts rows (like column-map benchmark)
struct CountingVisitor {
    total_rows: usize,
}

impl CountingVisitor {
    fn new() -> Self {
        Self { total_rows: 0 }
    }
}

impl PrimitiveVisitor for CountingVisitor {
    fn u64_chunk(&mut self, values: &UInt64Array) {
        self.total_rows += values.len();
    }

    fn u32_chunk(&mut self, values: &arrow::array::UInt32Array) {
        self.total_rows += values.len();
    }

    fn u16_chunk(&mut self, values: &arrow::array::UInt16Array) {
        self.total_rows += values.len();
    }

    fn u8_chunk(&mut self, values: &arrow::array::UInt8Array) {
        self.total_rows += values.len();
    }

    fn i64_chunk(&mut self, values: &Int64Array) {
        self.total_rows += values.len();
    }

    fn i32_chunk(&mut self, values: &arrow::array::Int32Array) {
        self.total_rows += values.len();
    }

    fn i16_chunk(&mut self, values: &arrow::array::Int16Array) {
        self.total_rows += values.len();
    }

    fn i8_chunk(&mut self, values: &arrow::array::Int8Array) {
        self.total_rows += values.len();
    }

    fn f64_chunk(&mut self, values: &arrow::array::Float64Array) {
        self.total_rows += values.len();
    }

    fn f32_chunk(&mut self, values: &arrow::array::Float32Array) {
        self.total_rows += values.len();
    }
}

impl PrimitiveWithRowIdsVisitor for CountingVisitor {
    fn u64_chunk_with_rids(&mut self, _values: &UInt64Array, row_ids: &UInt64Array) {
        self.total_rows += row_ids.len();
    }

    fn u32_chunk_with_rids(&mut self, _values: &arrow::array::UInt32Array, row_ids: &UInt64Array) {
        self.total_rows += row_ids.len();
    }

    fn u16_chunk_with_rids(&mut self, _values: &arrow::array::UInt16Array, row_ids: &UInt64Array) {
        self.total_rows += row_ids.len();
    }

    fn u8_chunk_with_rids(&mut self, _values: &arrow::array::UInt8Array, row_ids: &UInt64Array) {
        self.total_rows += row_ids.len();
    }

    fn i64_chunk_with_rids(&mut self, _values: &Int64Array, row_ids: &UInt64Array) {
        self.total_rows += row_ids.len();
    }

    fn i32_chunk_with_rids(&mut self, _values: &arrow::array::Int32Array, row_ids: &UInt64Array) {
        self.total_rows += row_ids.len();
    }

    fn i16_chunk_with_rids(&mut self, _values: &arrow::array::Int16Array, row_ids: &UInt64Array) {
        self.total_rows += row_ids.len();
    }

    fn i8_chunk_with_rids(&mut self, _values: &arrow::array::Int8Array, row_ids: &UInt64Array) {
        self.total_rows += row_ids.len();
    }

    fn f64_chunk_with_rids(&mut self, _values: &arrow::array::Float64Array, row_ids: &UInt64Array) {
        self.total_rows += row_ids.len();
    }

    fn f32_chunk_with_rids(&mut self, _values: &arrow::array::Float32Array, row_ids: &UInt64Array) {
        self.total_rows += row_ids.len();
    }
}

impl PrimitiveSortedVisitor for CountingVisitor {
    fn u64_run(&mut self, _values: &UInt64Array, _start: usize, len: usize) {
        self.total_rows += len;
    }

    fn u32_run(&mut self, _values: &arrow::array::UInt32Array, _start: usize, len: usize) {
        self.total_rows += len;
    }

    fn u16_run(&mut self, _values: &arrow::array::UInt16Array, _start: usize, len: usize) {
        self.total_rows += len;
    }

    fn u8_run(&mut self, _values: &arrow::array::UInt8Array, _start: usize, len: usize) {
        self.total_rows += len;
    }

    fn i64_run(&mut self, _values: &Int64Array, _start: usize, len: usize) {
        self.total_rows += len;
    }

    fn i32_run(&mut self, _values: &arrow::array::Int32Array, _start: usize, len: usize) {
        self.total_rows += len;
    }

    fn i16_run(&mut self, _values: &arrow::array::Int16Array, _start: usize, len: usize) {
        self.total_rows += len;
    }

    fn i8_run(&mut self, _values: &arrow::array::Int8Array, _start: usize, len: usize) {
        self.total_rows += len;
    }

    fn f64_run(&mut self, _values: &arrow::array::Float64Array, _start: usize, len: usize) {
        self.total_rows += len;
    }

    fn f32_run(&mut self, _values: &arrow::array::Float32Array, _start: usize, len: usize) {
        self.total_rows += len;
    }
}

impl PrimitiveSortedWithRowIdsVisitor for CountingVisitor {
    fn u64_run_with_rids(
        &mut self,
        _values: &UInt64Array,
        _row_ids: &UInt64Array,
        _start: usize,
        len: usize,
    ) {
        self.total_rows += len;
    }

    fn u32_run_with_rids(
        &mut self,
        _values: &arrow::array::UInt32Array,
        _row_ids: &UInt64Array,
        _start: usize,
        len: usize,
    ) {
        self.total_rows += len;
    }

    fn u16_run_with_rids(
        &mut self,
        _values: &arrow::array::UInt16Array,
        _row_ids: &UInt64Array,
        _start: usize,
        len: usize,
    ) {
        self.total_rows += len;
    }

    fn u8_run_with_rids(
        &mut self,
        _values: &arrow::array::UInt8Array,
        _row_ids: &UInt64Array,
        _start: usize,
        len: usize,
    ) {
        self.total_rows += len;
    }

    fn i64_run_with_rids(
        &mut self,
        _values: &Int64Array,
        _row_ids: &UInt64Array,
        _start: usize,
        len: usize,
    ) {
        self.total_rows += len;
    }

    fn i32_run_with_rids(
        &mut self,
        _values: &arrow::array::Int32Array,
        _row_ids: &UInt64Array,
        _start: usize,
        len: usize,
    ) {
        self.total_rows += len;
    }

    fn i16_run_with_rids(
        &mut self,
        _values: &arrow::array::Int16Array,
        _row_ids: &UInt64Array,
        _start: usize,
        len: usize,
    ) {
        self.total_rows += len;
    }

    fn i8_run_with_rids(
        &mut self,
        _values: &arrow::array::Int8Array,
        _row_ids: &UInt64Array,
        _start: usize,
        len: usize,
    ) {
        self.total_rows += len;
    }

    fn f64_run_with_rids(
        &mut self,
        _values: &arrow::array::Float64Array,
        _row_ids: &UInt64Array,
        _start: usize,
        len: usize,
    ) {
        self.total_rows += len;
    }

    fn f32_run_with_rids(
        &mut self,
        _values: &arrow::array::Float32Array,
        _row_ids: &UInt64Array,
        _start: usize,
        len: usize,
    ) {
        self.total_rows += len;
    }

    fn null_run(&mut self, _row_ids: &UInt64Array, _start: usize, len: usize) {
        self.total_rows += len;
    }
}

#[allow(clippy::print_stdout)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pause_before_scan = env::args().skip(1).any(|arg| arg == "--pause-before-scan");

    println!("Direct Performance Comparison: ColumnStore vs Table Layer");
    println!("========================================================");

    let pager = Arc::new(MemPager::default());
    const TABLE_ID: TableId = 1;
    const FIELD_ID: FieldId = 10;

    let row_count = 1_000_000_u64;
    println!("Testing with {} rows", row_count);

    // Create test data
    let rows: Vec<RowId> = (1..=row_count).collect();
    let values: Vec<i64> = (0..row_count).map(|i| (i * 2) as i64).collect();
    let lfid = LogicalFieldId::for_user(TABLE_ID, FIELD_ID);

    // Create table and append data once (ensures catalog metadata matches Table expectations)
    let table = Table::from_id(TABLE_ID, Arc::clone(&pager))?;
    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        field_with_fid("test_col", DataType::Int64, FIELD_ID, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(rows.clone())),
            Arc::new(Int64Array::from(values.clone())),
        ],
    )?;
    table.append(&batch)?;

    if pause_before_scan {
        println!(
            "\nPausing before scans for external profiling. PID: {}",
            std::process::id()
        );
        println!("Attach your profiler now, then press Enter to continue.");
        print!("Waiting... ");
        io::stdout().flush().ok();
        let mut buf = String::new();
        let _ = io::stdin().read_line(&mut buf);
        println!("Resuming scans.\n");
    }

    // Test 1: Direct ColumnStore access via ScanBuilder
    println!("\n=== Direct ColumnStore Test ===");
    let direct_elapsed = {
        let store = table.store();

        let mut visitor = CountingVisitor::new();
        let scan_opts = ScanOptions::default();

        let start = Instant::now();
        ScanBuilder::new(store, lfid)
            .options(scan_opts)
            .run(&mut visitor)?;
        let elapsed = start.elapsed();

        println!("  Time: {:?}", elapsed);
        println!("  Rows: {}", visitor.total_rows);
        println!(
            "  Throughput: {:.0} rows/sec",
            visitor.total_rows as f64 / elapsed.as_secs_f64()
        );

        elapsed
    };

    // Test 2: Table Layer access using the same data
    println!("\n=== Table Layer Test ===");
    let table_elapsed = {
        let projection = Projection::with_alias(lfid, "test_col");

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

        let mut total_rows = 0;

        let start = Instant::now();
        table.scan_stream(&[projection], &filter_expr, options, |batch| {
            total_rows += batch.num_rows();
            // Just count rows, no processing like column-map benchmark
        })?;
        let elapsed = start.elapsed();

        println!("  Time: {:?}", elapsed);
        println!("  Rows: {}", total_rows);
        println!(
            "  Throughput: {:.0} rows/sec",
            total_rows as f64 / elapsed.as_secs_f64()
        );
        elapsed
    };

    let overhead_ratio = table_elapsed.as_secs_f64() / direct_elapsed.as_secs_f64();
    println!("  Overhead vs ColumnStore: {:.1}x", overhead_ratio);

    println!("\n=== Analysis ===");
    println!("Measured ColumnStore time: {:?}", direct_elapsed);
    println!("Measured Table time: {:?}", table_elapsed);
    println!(
        "Planner adds {:.1}x overhead for this single-column scan.",
        overhead_ratio
    );

    Ok(())
}
