// Small example whose purpose is to exercise public APIs so that dependencies are compiled.
// Run with: cargo build --example compile_deps -p llkv-join

use arrow::array::{Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_join::{JoinKey, JoinOptions, TableJoinExt};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use std::collections::HashMap;
use std::sync::Arc;

fn create_table_with_rows(
    table_id: u16,
    pager: &Arc<MemPager>,
    num_rows: usize,
) -> Table<MemPager> {
    let table = Table::new(table_id, Arc::clone(pager)).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("id", DataType::Int32, false)
            .with_metadata(HashMap::from([("field_id".to_string(), "1".to_string())])),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(
                (0..num_rows).map(|i| i as u64).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                (0..num_rows).map(|i| i as i32).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();

    table.append(&batch).unwrap();
    table
}

fn main() {
    let pager = Arc::new(MemPager::default());
    let left = create_table_with_rows(1, &pager, 16);
    let right = create_table_with_rows(2, &pager, 8);

    let keys = vec![JoinKey::new(1, 1)];
    let options = JoinOptions::inner();

    // use a mutable named closure so it compiles in both generic and trait-object variants
    let mut total = 0usize;
    left.join_stream(&right, &keys, &options, |batch| {
        total += batch.num_rows();
    })
    .unwrap();

    println!("joined rows: {}", total);
}
