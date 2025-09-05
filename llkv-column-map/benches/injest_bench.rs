use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

use bitcode::{Decode, Encode};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use llkv_column_map::{
    AppendOptions, ColumnStore, Put, ValueMode, pager::Pager, types::PhysicalKey,
};

// ----------------- minimal in-memory Pager for the bench -----------------
#[derive(Default)]
struct MemPager {
    map: HashMap<PhysicalKey, Vec<u8>>,
    next: PhysicalKey,
}
impl Pager for MemPager {
    fn alloc_many(&mut self, n: usize) -> Vec<PhysicalKey> {
        if self.next == 0 {
            self.next = 1;
        } // reserve 0 for bootstrap
        let start = self.next;
        self.next += n as u64;
        (0..n).map(|i| start + i as u64).collect()
    }
    fn batch_put_raw(&mut self, items: &[(PhysicalKey, Vec<u8>)]) {
        for (k, v) in items {
            self.map.insert(*k, v.clone());
        }
    }
    fn batch_get_raw<'a>(&'a self, keys: &[PhysicalKey]) -> Vec<&'a [u8]> {
        keys.iter()
            .map(|k| self.map.get(k).expect("missing key").as_slice())
            .collect()
    }
    fn batch_put_typed<T: Encode>(&mut self, items: &[(PhysicalKey, T)]) {
        let enc: Vec<(PhysicalKey, Vec<u8>)> = items
            .iter()
            .map(|(k, v)| (*k, bitcode::encode(v)))
            .collect();
        self.batch_put_raw(&enc);
    }
    fn batch_get_typed<T>(&self, keys: &[PhysicalKey]) -> Vec<T>
    where
        for<'a> T: Decode<'a>,
    {
        self.batch_get_raw(keys)
            .into_iter()
            .map(|b| bitcode::decode(b).expect("bitcode decode failed"))
            .collect()
    }
}

// ----------------- dataset shape: 21 columns x 150_000 rows ---------------
#[derive(Clone, Copy)]
enum ColKind {
    Fixed(u32),
    Var { min: usize, max: usize },
}

fn col_spec_21() -> Vec<(u32, ColKind)> {
    let mut spec = Vec::new();
    for i in 0..8 {
        spec.push((100 + i, ColKind::Fixed(8)));
    } // 8 x fixed(8)
    for i in 0..4 {
        spec.push((200 + i, ColKind::Fixed(4)));
    } // 4 x fixed(4)
    for i in 0..5 {
        spec.push((300 + i, ColKind::Var { min: 5, max: 25 }));
    } // 5 x var short
    for i in 0..4 {
        spec.push((400 + i, ColKind::Var { min: 50, max: 200 }));
    } // 4 x var med
    spec
}

#[inline]
fn row_key(row: u64) -> Vec<u8> {
    row.to_be_bytes().to_vec()
}

#[inline]
fn var_len_for(row: u64, field: u32, min: usize, max: usize) -> usize {
    let span = (max - min + 1) as u64;
    let mix = row
        .wrapping_mul(1103515245)
        .wrapping_add(field as u64)
        .rotate_left(13);
    (min as u64 + (mix % span)) as usize
}

fn build_puts_for_range(start: u64, end: u64, spec: &[(u32, ColKind)]) -> Vec<Put> {
    let mut puts = Vec::with_capacity(spec.len());
    for (field_id, kind) in spec.iter().copied() {
        let mut items = Vec::with_capacity((end - start) as usize);
        for r in start..end {
            let k = row_key(r);
            let v = match kind {
                // Handle ANY fixed width (4, 8, or whatever you put in ColKind::Fixed)
                ColKind::Fixed(w) => {
                    let w = w as usize;
                    // derive a stable 8-byte seed from row & field, then repeat/truncate to w bytes
                    let seed = (r ^ field_id as u64).to_le_bytes();
                    if w <= 8 {
                        seed[..w].to_vec()
                    } else {
                        let mut buf = Vec::with_capacity(w);
                        while buf.len() < w {
                            let take = std::cmp::min(8, w - buf.len());
                            buf.extend_from_slice(&seed[..take]);
                        }
                        buf
                    }
                }
                ColKind::Var { min, max } => {
                    let len = var_len_for(r, field_id, min, max);
                    let byte = (((r as u32).wrapping_add(field_id)) & 0xFF) as u8;
                    vec![byte; len]
                }
            };
            items.push((k, v));
        }
        puts.push(Put { field_id, items });
    }
    puts
}

fn bench_ingest_by_batches(c: &mut Criterion) {
    let total_rows: u64 = 150_000;
    let spec = col_spec_21();

    let opts = AppendOptions {
        mode: ValueMode::Auto,
        segment_max_entries: 16_384,
        segment_max_bytes: 2 * 1024 * 1024,
        last_write_wins_in_batch: true,
    };

    // num_batches: 1 means "ALL rows in ONE append_many call"
    let num_batches_options = [1usize, 2, 5, 10, 20, 50, 100];

    let mut group = c.benchmark_group("ingest_21x150k_by_num_batches");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(total_rows * 21));

    for &batches in &num_batches_options {
        let rows_per_batch = (total_rows as usize + batches - 1) / batches;
        let label = format!(
            "num_batches={} (rows_per_batch≈{})",
            batches, rows_per_batch
        );
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                let mut pager = MemPager::default();
                let mut store = ColumnStore::init_empty(&mut pager);

                let mut start = 0u64;
                for _ in 0..batches {
                    let end = std::cmp::min(total_rows, start + rows_per_batch as u64);
                    if start >= end {
                        break;
                    }
                    let puts = build_puts_for_range(start, end, &spec);
                    store.append_many(puts, opts.clone());
                    start = end;
                }

                black_box(pager.map.len());
            });
        });
    }
    group.finish();
}

fn bench_ingest_by_rows_per_batch(c: &mut Criterion) {
    let total_rows: u64 = 150_000;
    let spec = col_spec_21();

    let opts = AppendOptions {
        mode: ValueMode::Auto,
        segment_max_entries: 16_384,
        segment_max_bytes: 2 * 1024 * 1024,
        last_write_wins_in_batch: true,
    };

    // Explicit rows-per-batch. 150_000 means ALL AT ONCE.
    let rows_per_batch_options = [150_000usize, 75_000, 50_000, 10_000, 2_000, 1_000];

    let mut group = c.benchmark_group("ingest_21x150k_by_rows_per_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(total_rows * 21));

    for &rows_per_batch in &rows_per_batch_options {
        let label = format!("rows_per_batch={}", rows_per_batch);
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                let mut pager = MemPager::default();
                let mut store = ColumnStore::init_empty(&mut pager);

                let mut start = 0u64;
                while start < total_rows {
                    let end = std::cmp::min(total_rows, start + rows_per_batch as u64);
                    let puts = build_puts_for_range(start, end, &spec);
                    store.append_many(puts, opts.clone());
                    start = end;
                }

                black_box(pager.map.len());
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ingest_by_batches,
    bench_ingest_by_rows_per_batch
);
criterion_main!(benches);
