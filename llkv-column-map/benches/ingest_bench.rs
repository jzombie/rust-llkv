//! Ingest benchmark: append multiple 1M-row columns across mixed types.
//!
//! What it measures
//! - Builds an in-memory ColumnStore (MemPager) and ingests N rows for a
//!   multi-column schema (several integer types + one binary column).
//! - Varies the number of append batches (1, 4, 16, 64) to expose batching
//!   overhead vs. throughput.
//! - Uses the current `append(&RecordBatch)` API; every batch contains
//!   `rowid` plus all data columns.
//!
//! Run:
//!   cargo bench --bench ingest_bench

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{BinaryBuilder, Int8Array, Int32Array, UInt16Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_column_map::debug::ColumnStoreDebug;
use llkv_column_map::store::ColumnStore;
use llkv_storage::pager::MemPager;
use llkv_types::LogicalFieldId;

const N_ROWS: usize = 1_000_000;

#[derive(Clone, Copy, Debug)]
enum ColKind {
    U64(u32),
    I32(u32),
    U16(u32),
    I8(u32),
    BinShort(u32), // 5..=25 bytes
}

fn column_kinds() -> Vec<ColKind> {
    vec![
        ColKind::U64(1001),
        ColKind::U64(1002),
        ColKind::I32(2001),
        ColKind::I32(2002),
        ColKind::U16(3001),
        ColKind::U16(3002),
        ColKind::I8(4001),
        ColKind::I8(4002),
        ColKind::BinShort(5001),
    ]
}

fn schema_for(cols: &[ColKind]) -> Arc<Schema> {
    let mut fields = Vec::with_capacity(cols.len() + 1);
    fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));
    for c in cols {
        let (fid_raw, dt) = match *c {
            ColKind::U64(fid) => (fid, DataType::UInt64),
            ColKind::I32(fid) => (fid, DataType::Int32),
            ColKind::U16(fid) => (fid, DataType::UInt16),
            ColKind::I8(fid) => (fid, DataType::Int8),
            ColKind::BinShort(fid) => (fid, DataType::Binary),
        };
        let mut md = HashMap::new();
        md.insert(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            u64::from(LogicalFieldId::for_user_table_0(fid_raw)).to_string(),
        );
        fields.push(Field::new("data", dt, false).with_metadata(md));
    }
    Arc::new(Schema::new(fields))
}

#[inline]
fn var_len_for(row: u64, field: u64, min: usize, max: usize) -> usize {
    let span = (max - min + 1) as u64;
    let mix = row
        .wrapping_mul(1103515245)
        .wrapping_add(field)
        .rotate_left(13);
    (min as u64 + (mix % span)) as usize
}

fn build_batch_for_range(
    schema: &Arc<Schema>,
    cols: &[ColKind],
    start: usize,
    end: usize,
) -> RecordBatch {
    // let n = end - start;
    // row_id 0..N_ROWS-1
    let rid: Vec<u64> = (start as u64..end as u64).collect();
    let rid_arr = Arc::new(UInt64Array::from(rid));

    let mut arrays: Vec<arrow::array::ArrayRef> = Vec::with_capacity(cols.len() + 1);
    arrays.push(rid_arr);

    for c in cols {
        match *c {
            ColKind::U64(fid) => {
                // Deterministic content: (row_index ^ fid)
                let vals: Vec<u64> = (start..end).map(|i| (i as u64) ^ (fid as u64)).collect();
                arrays.push(Arc::new(UInt64Array::from(vals)));
            }
            ColKind::I32(fid) => {
                let vals: Vec<i32> = (start..end)
                    .map(|i| ((i as u64) ^ (fid as u64)) as i32)
                    .collect();
                arrays.push(Arc::new(Int32Array::from(vals)));
            }
            ColKind::U16(fid) => {
                let vals: Vec<u16> = (start..end)
                    .map(|i| (((i as u64) ^ (fid as u64)) & 0xFFFF) as u16)
                    .collect();
                arrays.push(Arc::new(UInt16Array::from(vals)));
            }
            ColKind::I8(fid) => {
                let vals: Vec<i8> = (start..end)
                    .map(|i| (((i as u64) ^ (fid as u64)) & 0x7F) as i8)
                    .collect();
                arrays.push(Arc::new(Int8Array::from(vals)));
            }
            ColKind::BinShort(fid) => {
                let mut b = BinaryBuilder::new();
                let fid_u64 = u64::from(LogicalFieldId::for_user_table_0(fid));
                for r in start as u64..end as u64 {
                    let len = var_len_for(r, fid_u64, 5, 25);
                    let byte = (((r).wrapping_add(fid_u64)) & 0xFF) as u8;
                    b.append_value(vec![byte; len]);
                }
                arrays.push(Arc::new(b.finish()));
            }
        }
    }

    RecordBatch::try_new(Arc::clone(schema), arrays).unwrap()
}

fn bench_ingest_mixed_columns(c: &mut Criterion) {
    let cols = column_kinds();
    let schema = schema_for(&cols);
    let num_cols = cols.len();
    let total_cells = (N_ROWS as u64) * (num_cols as u64);

    let mut group = c.benchmark_group("ingest_mixed_1M");
    group.sample_size(10);
    group.throughput(Throughput::Elements(total_cells));

    for &batches in &[1usize, 4, 16, 64] {
        group.bench_function(
            BenchmarkId::from_parameter(format!("batches={}", batches)),
            |b| {
                b.iter(|| {
                    let pager = Arc::new(MemPager::new());
                    let store = ColumnStore::open(pager).unwrap();

                    let rows_per = N_ROWS.div_ceil(batches);
                    let mut off = 0usize;
                    while off < N_ROWS {
                        let end = (off + rows_per).min(N_ROWS);
                        let batch = build_batch_for_range(&schema, &cols, off, end);
                        store.append(&batch).unwrap();
                        off = end;
                    }

                    // Touch storage so optimizer can’t elide work. Returns ASCII table.
                    let layout = store.render_storage_as_formatted_string();
                    black_box(layout.len());
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_ingest_mixed_columns);
criterion_main!(benches);
