#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{criterion_group, criterion_main, Criterion};
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_storage::pager::MemPager;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::hint::black_box;

const ROW_COUNT: usize = 250_000;
const FIELD_COUNT: usize = 4;
const MULTI_FIELD_TAKE: usize = 3;
const SAMPLE_ROW_IDS: usize = 1024;
const SEED: u64 = 0x9E37_79B9_F701_3CAB;

fn logical_fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

fn schema_with_row_id(field_id: LogicalFieldId, name: &str) -> Arc<Schema> {
    let rid = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data = Field::new(name, DataType::UInt64, false).with_metadata(md);
    Arc::new(Schema::new(vec![rid, data]))
}

fn build_fixture() -> (ColumnStore<MemPager>, Vec<LogicalFieldId>, Vec<u64>) {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager)).expect("open column store");

    let base_row_ids: Vec<u64> = (0..ROW_COUNT as u64).collect();
    let rid_array = Arc::new(UInt64Array::from(base_row_ids));

    let mut fields = Vec::with_capacity(FIELD_COUNT);
    for idx in 0..FIELD_COUNT {
        let fid = logical_fid(idx as u32);
        fields.push(fid);
        let schema = schema_with_row_id(fid, &format!("col_{idx}"));
        let values: Vec<u64> = (0..ROW_COUNT as u64)
            .map(|v| v.wrapping_mul(31).wrapping_add(idx as u64))
            .collect();
        let val_array = Arc::new(UInt64Array::from(values));

        let batch = RecordBatch::try_new(
            schema,
            vec![rid_array.clone() as ArrayRef, val_array.clone() as ArrayRef],
        )
        .expect("record batch");
        store.append(&batch).expect("append");
    }

    let mut sample_rows: Vec<u64> = (0..ROW_COUNT as u64).collect();
    let mut rng = StdRng::seed_from_u64(SEED);
    sample_rows.shuffle(&mut rng);
    sample_rows.truncate(SAMPLE_ROW_IDS);

    (store, fields, sample_rows)
}

fn bench_gather_rows(c: &mut Criterion) {
    let (store, field_ids, sample_rows) = build_fixture();
    let mut group = c.benchmark_group("gather_rows");

    let single_field = field_ids[0];
    group.bench_function("single_column", |b| {
        b.iter(|| {
            let result = store.gather_rows(single_field, &sample_rows).expect("gather");
            black_box(result);
        });
    });

    group.bench_function("multi_column_sequential", |b| {
        b.iter(|| {
            for &fid in field_ids.iter().take(MULTI_FIELD_TAKE) {
                let result = store.gather_rows(fid, &sample_rows).expect("gather");
                black_box(result);
            }
        });
    });

    group.finish();
}

criterion_group!(gather_rows_benches, bench_gather_rows);
criterion_main!(gather_rows_benches);
