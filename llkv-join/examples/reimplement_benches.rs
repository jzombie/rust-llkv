//! Example that reimplements the `join_bench` Criterion benches using std timing.
//! Run with: cargo run --example reimplement_benches -p llkv-join --release

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Int32Array, UInt64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_join::{JoinKey, JoinOptions, TableJoinExt};
use llkv_storage::pager::MemPager;
use llkv_table::Table;

fn create_table_with_rows(
    table_id: u16,
    pager: &Arc<MemPager>,
    num_rows: usize,
    id_offset: i32,
) -> Table<MemPager> {
    let table = Table::new(table_id, Arc::clone(pager)).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("id", DataType::Int32, false)
            .with_metadata(HashMap::from([("field_id".to_string(), "1".to_string())])),
        Field::new("value", DataType::Utf8, false)
            .with_metadata(HashMap::from([("field_id".to_string(), "2".to_string())])),
    ]));

    let batch_size = 10_000;
    for batch_start in (0..num_rows).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_rows);

        let row_ids: Vec<u64> = (batch_start..batch_end).map(|i| i as u64).collect();
        let ids: Vec<i32> = (batch_start..batch_end)
            .map(|i| (i as i32) + id_offset)
            .collect();
        let values: Vec<String> = (batch_start..batch_end)
            .map(|i| format!("value_{}", i))
            .collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(row_ids)),
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(values)),
            ],
        )
        .unwrap();

        table.append(&batch).unwrap();
    }

    table
}

fn run_inner_join(size: usize) -> Duration {
    let start = Instant::now();
    let pager = Arc::new(MemPager::default());

    let left = create_table_with_rows(1, &pager, size, 0);
    let right = create_table_with_rows(2, &pager, size, (size / 2) as i32);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::inner();

    let mut total_rows = 0usize;
    left.join_stream(&right, &keys, &options, |batch| {
        total_rows += batch.num_rows();
    })
    .unwrap();
    black_box(total_rows);
    start.elapsed()
}

fn run_left_join(size: usize) -> Duration {
    let start = Instant::now();
    let pager = Arc::new(MemPager::default());

    let left = create_table_with_rows(1, &pager, size, 0);
    let right = create_table_with_rows(2, &pager, size / 2, (size / 4) as i32);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::left();

    let mut total_rows = 0usize;
    left.join_stream(&right, &keys, &options, |batch| {
        total_rows += batch.num_rows();
    })
    .unwrap();
    black_box(total_rows);
    start.elapsed()
}

fn run_semi_join(size: usize) -> Duration {
    let start = Instant::now();
    let pager = Arc::new(MemPager::default());

    let left = create_table_with_rows(1, &pager, size, 0);
    let right = create_table_with_rows(2, &pager, size / 10, 0);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::semi();

    let mut total_rows = 0usize;
    left.join_stream(&right, &keys, &options, |batch| {
        total_rows += batch.num_rows();
    })
    .unwrap();
    black_box(total_rows);
    start.elapsed()
}

fn run_anti_join(size: usize) -> Duration {
    let start = Instant::now();
    let pager = Arc::new(MemPager::default());

    let left = create_table_with_rows(1, &pager, size, 0);
    let right = create_table_with_rows(2, &pager, size / 10, size as i32);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::anti();

    let mut total_rows = 0usize;
    left.join_stream(&right, &keys, &options, |batch| {
        total_rows += batch.num_rows();
    })
    .unwrap();
    black_box(total_rows);
    start.elapsed()
}

fn run_many_to_many(size: usize) -> Duration {
    let start = Instant::now();
    let pager = Arc::new(MemPager::default());

    let left = create_table_with_rows(1, &pager, size, 0);
    let right = create_table_with_rows(2, &pager, size, 0);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::inner();

    let mut total_rows = 0usize;
    left.join_stream(&right, &keys, &options, |batch| {
        total_rows += batch.num_rows();
    })
    .unwrap();
    black_box(total_rows);
    start.elapsed()
}

fn run_no_matches(size: usize) -> Duration {
    let start = Instant::now();
    let pager = Arc::new(MemPager::default());

    let left = create_table_with_rows(1, &pager, size, 0);
    let right = create_table_with_rows(2, &pager, size, (size * 2) as i32);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::inner();

    let mut total_rows = 0usize;
    left.join_stream(&right, &keys, &options, |batch| {
        total_rows += batch.num_rows();
    })
    .unwrap();
    black_box(total_rows);
    start.elapsed()
}

fn print_result(name: &str, dur: Duration) {
    println!("{:<30} {:>10.3} s", name, dur.as_secs_f64());
}

fn main() {
    println!("reimplement_benches: manual join benches (std timing)");

    for &size in &[1_000usize, 10_000usize, 100_000usize] {
        print_result(&format!("inner size={}", size), run_inner_join(size));
        print_result(&format!("left size={}", size), run_left_join(size));
        print_result(&format!("semi size={}", size), run_semi_join(size));
        print_result(&format!("anti size={}", size), run_anti_join(size));
    }

    for &size in &[100usize, 1_000usize, 10_000usize] {
        print_result(&format!("many_to_many size={}", size), run_many_to_many(size));
    }

    for &size in &[1_000usize, 10_000usize, 100_000usize] {
        print_result(&format!("no_match size={}", size), run_no_matches(size));
    }

    println!("done");
}
