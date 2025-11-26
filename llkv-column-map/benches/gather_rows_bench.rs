#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{Criterion, criterion_group, criterion_main};
use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_column_map::store::{ColumnStore, GatherNullPolicy};
use llkv_storage::pager::MemPager;
use llkv_types::LogicalFieldId;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::hint::black_box;

const ROW_COUNT: usize = 250_000;
const FIELD_COUNT: usize = 4;
const MULTI_FIELD_TAKE: usize = 3;
const SPARSE_FIELD_COUNT: usize = 3;
const SPARSE_FIELD_TAKE: usize = 3;
const SAMPLE_ROW_IDS: usize = 1024;
const SEED: u64 = 0x9E37_79B9_F701_3CAB;

fn schema_with_row_id(field_id: LogicalFieldId, name: &str) -> Arc<Schema> {
    let rid = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    let mut md = HashMap::new();
    md.insert(
        llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
        u64::from(field_id).to_string(),
    );
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
        let fid = LogicalFieldId::for_user_table_0(idx as u32);
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

fn build_sparse_fixture() -> (ColumnStore<MemPager>, Vec<LogicalFieldId>, Vec<u64>) {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager)).expect("open column store");

    let base_row_ids: Vec<u64> = (0..ROW_COUNT as u64).collect();
    let mut fields = Vec::with_capacity(SPARSE_FIELD_COUNT);
    for idx in 0..SPARSE_FIELD_COUNT {
        let fid = LogicalFieldId::for_user_table_0((100 + idx) as u32);
        fields.push(fid);
        let schema = schema_with_row_id(fid, &format!("sparse_col_{idx}"));
        let (rid_slice, values): (Vec<u64>, Vec<u64>) = if idx == SPARSE_FIELD_COUNT - 1 {
            let present: Vec<u64> = base_row_ids
                .iter()
                .copied()
                .filter(|v| v % 3 != 1)
                .collect();
            let vals: Vec<u64> = present
                .iter()
                .map(|v| v.wrapping_mul(43).wrapping_add(idx as u64))
                .collect();
            (present, vals)
        } else {
            let vals: Vec<u64> = base_row_ids
                .iter()
                .map(|v| v.wrapping_mul(29).wrapping_add(idx as u64))
                .collect();
            (base_row_ids.clone(), vals)
        };

        let rid_slice_arr: ArrayRef = Arc::new(UInt64Array::from(rid_slice.clone()));
        let value_arr: ArrayRef = Arc::new(UInt64Array::from(values));

        let batch =
            RecordBatch::try_new(schema, vec![rid_slice_arr, value_arr]).expect("record batch");
        store.append(&batch).expect("append");
    }

    let mut sample_rows = (0..SAMPLE_ROW_IDS as u64).collect::<Vec<u64>>();
    let mut rng = StdRng::seed_from_u64(SEED);
    sample_rows.shuffle(&mut rng);
    sample_rows.truncate(SAMPLE_ROW_IDS);
    // Ensure at least one missing row id for sparse column
    if !sample_rows.iter().any(|v| v % 3 == 1) {
        sample_rows[0] = 1;
    }

    (store, fields, sample_rows)
}

fn gather_single(
    store: &ColumnStore<MemPager>,
    field_id: LogicalFieldId,
    row_ids: &[u64],
    include_nulls: bool,
) -> ArrayRef {
    let policy = if include_nulls {
        GatherNullPolicy::IncludeNulls
    } else {
        GatherNullPolicy::ErrorOnMissing
    };
    store
        .gather_rows(&[field_id], row_ids, policy)
        .expect("gather single")
        .column(0)
        .clone()
}

fn bench_gather_rows(c: &mut Criterion) {
    let (store, field_ids, sample_rows) = build_fixture();
    let (sparse_store, sparse_fields, sparse_rows) = build_sparse_fixture();
    let mut group = c.benchmark_group("gather_rows");

    let single_field = field_ids[0];
    group.bench_function("single_column", |b| {
        b.iter(|| {
            let result = gather_single(&store, single_field, &sample_rows, false);
            black_box(result);
        });
    });

    group.bench_function("multi_column_sequential", |b| {
        b.iter(|| {
            for &fid in field_ids.iter().take(MULTI_FIELD_TAKE) {
                let result = gather_single(&store, fid, &sample_rows, false);
                black_box(result);
            }
        });
    });

    group.bench_function("multi_column_batched", |b| {
        let fids: Vec<LogicalFieldId> = field_ids.iter().take(MULTI_FIELD_TAKE).copied().collect();
        b.iter(|| {
            let result = store
                .gather_rows(&fids, &sample_rows, GatherNullPolicy::ErrorOnMissing)
                .expect("gather multi");
            black_box(result);
        });
    });

    group.bench_function("multi_column_batched_with_nulls", |b| {
        let fids: Vec<LogicalFieldId> = sparse_fields
            .iter()
            .take(SPARSE_FIELD_TAKE)
            .copied()
            .collect();
        b.iter(|| {
            let result = sparse_store
                .gather_rows(&fids, &sparse_rows, GatherNullPolicy::IncludeNulls)
                .expect("gather multi nulls");
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(gather_rows_benches, bench_gather_rows);
criterion_main!(gather_rows_benches);
