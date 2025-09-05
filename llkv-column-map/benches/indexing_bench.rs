use bitcode::{Decode, Encode};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use llkv_column_map::index::{
    Bootstrap, ColumnEntry, ColumnIndex, IndexSegment, IndexSegmentRef, Manifest,
};
use llkv_column_map::pager::Pager;
use llkv_column_map::types::{LogicalFieldId, LogicalKeyBytes, PhysicalKey};
use std::collections::HashMap;
use std::hint::black_box;

/// Physical key reserved for the tiny bootstrap record.
const BOOTSTRAP_PKEY: PhysicalKey = 0;

/// A tiny in-memory pager used for the benchmark (batch-only API).
struct MemPager {
    map: HashMap<PhysicalKey, Vec<u8>>,
    next: PhysicalKey,
}

impl Default for MemPager {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
            next: 1,
        } // reserve 0 for Bootstrap
    }
}

impl Pager for MemPager {
    fn alloc_many(&mut self, n: usize) -> Vec<PhysicalKey> {
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
        let mut enc: Vec<(PhysicalKey, Vec<u8>)> = Vec::with_capacity(items.len());
        for (k, v) in items {
            enc.push((*k, bitcode::encode(v)));
        }
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

/// Make monotonically increasing numeric logical keys (already sorted).
#[inline]
fn make_numeric_keys(n: usize) -> Vec<LogicalKeyBytes> {
    (0..n).map(|i| (i as u64).to_be_bytes().to_vec()).collect()
}

/// Build *and persist* many columns in batches:
/// - For each column: one sealed IndexSegment (fixed-width values),
///   one ColumnIndex pointing to that segment, and one Manifest entry.
/// - Finally write Manifest and Bootstrap (key 0).
fn build_many_columns_fixed_width(
    columns: usize,
    entries_per_col: usize,
    chunk_cols: usize,
    value_width: u32,
) {
    let mut pager = MemPager::default();

    // Manifest gets its own physical key up front.
    let manifest_pkey = pager.alloc_many(1)[0];

    // We collect the manifest entries as we go; writing it once at the end.
    let mut manifest_entries: Vec<ColumnEntry> = Vec::with_capacity(columns);

    // Per-column work is chunked (to cap peak memory).
    let mut remaining = columns;
    let mut next_field_id: LogicalFieldId = 0;

    while remaining > 0 {
        let batch = remaining.min(chunk_cols);

        // For each column in this batch we need:
        //   data_pkey, index_segment_pkey, column_index_pkey  → 3 keys/column
        let ids = pager.alloc_many(batch * 3);

        // Stage typed writes in two homogenous batches:
        //   1) IndexSegment
        //   2) ColumnIndex
        let mut seg_puts: Vec<(PhysicalKey, IndexSegment)> = Vec::with_capacity(batch);
        let mut colidx_puts: Vec<(PhysicalKey, ColumnIndex)> = Vec::with_capacity(batch);

        // Pre-build logical keys once per batch (same key set for all columns).
        let logical_keys = make_numeric_keys(entries_per_col);

        for i in 0..batch {
            let data_pkey = ids[i * 3 + 0];
            let seg_pkey = ids[i * 3 + 1];
            let colidx_pkey = ids[i * 3 + 2];

            // Values are fixed-width dummy payloads: the data blob is opaque in this model.
            let seg = IndexSegment::build_fixed(data_pkey, logical_keys.clone(), value_width);

            // Reference to that segment (newest-first list with 1 element).
            let segref = IndexSegmentRef {
                index_physical_key: seg_pkey,
                logical_key_min: seg.logical_key_min.clone(),
                logical_key_max: seg.logical_key_max.clone(),
                n_entries: seg.n_entries,
            };

            let col_index = ColumnIndex {
                field_id: next_field_id,
                segments: vec![segref],
            };

            seg_puts.push((seg_pkey, seg));
            colidx_puts.push((colidx_pkey, col_index));

            manifest_entries.push(ColumnEntry {
                field_id: next_field_id,
                column_index_physical_key: colidx_pkey,
            });

            next_field_id = next_field_id.wrapping_add(1);
        }

        // Persist the two homogenous batches.
        pager.batch_put_typed::<IndexSegment>(&seg_puts);
        pager.batch_put_typed::<ColumnIndex>(&colidx_puts);

        remaining -= batch;
    }

    // Write Manifest and Bootstrap (key 0).
    pager.batch_put_typed::<Manifest>(&[(
        manifest_pkey,
        Manifest {
            columns: manifest_entries,
        },
    )]);
    pager.batch_put_typed::<Bootstrap>(&[(
        BOOTSTRAP_PKEY,
        Bootstrap {
            manifest_physical_key: manifest_pkey,
        },
    )]);

    // Keep side-effects alive for the optimizer.
    black_box(pager);
}

fn criterion_build_columns(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_many_columns_fixed");

    // Tune these safely; 50k x 128 is sizable in RAM (keys+offsets).
    let column_counts = [5_000usize, 50_000usize];
    let entries_per_column = [8usize, 64usize, 128usize];

    // Chunking keeps peak allocations in check; adjust as needed.
    let chunk_cols = 2_048usize;
    let value_width = 8u32; // pretend 8-byte fixed-width values

    for &cols in &column_counts {
        for &n in &entries_per_column {
            let id = BenchmarkId::new("cols_entries", format!("cols={}_entries={}", cols, n));
            group.throughput(Throughput::Elements(cols as u64));
            group.bench_with_input(id, &(cols, n), |b, &(cols, n)| {
                b.iter(|| build_many_columns_fixed_width(cols, n, chunk_cols, value_width));
            });
        }
    }

    group.finish();
}

criterion_group!(benches, criterion_build_columns);
criterion_main!(benches);
