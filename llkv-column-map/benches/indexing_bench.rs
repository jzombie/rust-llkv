//! # Benchmark: Build Many Columns (fixed-width, one segment per column)
//!
//! **Purpose**
//! Stress the *metadata construction path* by creating lots of columns quickly.
//! Each column receives exactly one sealed `IndexSegment` (fixed-width values) and a
//! `ColumnIndex` pointing to it, then we write a single `Manifest` and `Bootstrap`.
//! This isolates the cost of **allocating keys**, **encoding/decoding typed blobs**,
//! and **batched puts** without mixing in high-volume value writes.
//!
//! **What it does**
//! Parameterized by:
//! - `columns` (e.g., 5k, 50k), i.e., how many columns to create.
//! - `entries_per_col` (e.g., 8, 64, 128), i.e., logical rows per column.
//! - `value_width` (e.g., 8), i.e., fixed width for every value in that column.
//! - `chunk_cols` (e.g., 2,048): columns processed per loop to cap peak allocations.
//!
//! For each chunk of columns:
//! 1. Allocate **3 keys per column**: data blob key, index segment key, column index key.
//! 2. Build an `IndexSegment::build_fixed` for that column (one per column).
//! 3. Build a `ColumnIndex` that contains a single `IndexSegmentRef` (newest-first list).
//! 4. Persist all segments and column indexes in **one unified batch** (mixed typed puts).
//! - After all chunks: write **one `Manifest`** containing all column entries and **`Bootstrap`**
//!   (physical key 0) in a **single batch**.
//!
//! **What it measures**
//! - Overhead of creating large numbers of small typed objects (`IndexSegment`, `ColumnIndex`,
//!   `Manifest`, `Bootstrap`) and shuttling them through the **batched puts** path.
//! - Effectiveness of batching: we minimize round-trips by staging puts into a single
//!   batch call per chunk (and once more at the end for `Manifest` + `Bootstrap`).
//!
//! **Why fixed-width + one segment per column?**
//! - Keeps the test simple and deterministic (no variable-width offset building).
//! - Focuses on index+manifest object counts and encoding costs rather than data slicing.
//!
//! **Knobs to tweak**
//! - `columns`: increase to stress the manifest and the volume of typed puts.
//! - `entries_per_col`: increases `logical_key_bytes` and offsets size per segment.
//! - `chunk_cols`: trade off peak memory vs. number of batched calls.
//! - `value_width`: impacts how large each data blob *would* be (opaque here).
//!
//! **Interpreting results**
//! - Larger `chunk_cols` ⇒ fewer batch calls, larger per-batch payloads.
//! - More `columns` or more `entries_per_col` ⇒ more/larger typed blobs;
//!   timings should scale roughly with total bytes encoded + map insertions.
//!
//! **Caveats**
//! - Uses an in-memory pager; it exercises encode/decode and batching but not I/O latency.
//! - Data blobs are treated as opaque; this benchmark targets **metadata path**, not reads.
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use llkv_column_map::codecs::big_endian::u64_be_array;
use llkv_column_map::column_index::{
    Bootstrap, ColumnEntry, ColumnIndex, IndexSegment, IndexSegmentRef, Manifest,
};
use llkv_column_map::constants::BOOTSTRAP_PKEY;
use llkv_column_map::storage::pager::{BatchPut, MemPager, Pager};
use llkv_column_map::types::{LogicalFieldId, PhysicalKey, TypedValue};
use std::hint::black_box;
/// Make monotonically increasing numeric logical keys (already sorted).
#[inline]
fn make_numeric_keys(n: usize) -> Vec<[u8; 8]> {
    (0..n).map(|i| u64_be_array(i as u64)).collect()
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
    let pager = MemPager::default();
    // Manifest gets its own physical key up front.
    let manifest_pkey = pager.alloc_many(1).unwrap()[0];
    // We collect the manifest entries as we go; writing it once at the end.
    let mut manifest_entries: Vec<ColumnEntry> = Vec::with_capacity(columns);

    // Per-column work is chunked (to cap peak memory).
    let mut remaining = columns;
    let mut next_field_id: LogicalFieldId = 0;

    while remaining > 0 {
        let batch = remaining.min(chunk_cols);
        // For each column in this batch we need:
        //   data_pkey, index_segment_pkey, column_index_pkey  → 3 keys/column
        let ids = pager.alloc_many(batch * 3).unwrap();
        // Stage typed writes in two homogenous batches:
        //   1) IndexSegment
        //   2) ColumnIndex
        let mut seg_puts: Vec<(PhysicalKey, IndexSegment)> = Vec::with_capacity(batch);
        let mut colidx_puts: Vec<(PhysicalKey, ColumnIndex)> = Vec::with_capacity(batch);

        // Pre-build logical keys once per batch (same key set for all columns).
        let logical_keys = make_numeric_keys(entries_per_col);

        // Pre-compute key bounds (they live on IndexSegmentRef in the new schema).
        // Note: .to_vec() is needed here because the Ref needs an owned Vec, but this
        // is only two allocations per chunk, not per-key.
        let key_min = logical_keys.first().unwrap().to_vec();
        let key_max = logical_keys.last().unwrap().to_vec();

        for i in 0..batch {
            let data_pkey = ids[i * 3];
            let seg_pkey = ids[i * 3 + 1];
            let colidx_pkey = ids[i * 3 + 2];
            // Values are fixed-width dummy payloads: the data blob is opaque in this model.
            let seg = IndexSegment::build_fixed(data_pkey, &logical_keys, value_width);
            // Reference to that segment (newest-first list with 1 element).
            let segref = IndexSegmentRef {
                index_physical_key: seg_pkey,
                data_physical_key: seg.data_physical_key,
                logical_key_min: key_min.clone(),
                logical_key_max: key_max.clone(),
                value_min: None, // metadata bench: no value bounds
                value_max: None, // metadata bench: no value bounds
                n_entries: seg.n_entries,
            };
            let col_index = ColumnIndex {
                field_id: next_field_id,
                value_order: llkv_column_map::types::ValueOrderPolicy::Raw,
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

        // Persist the two homogenous groups in ONE batched put call.
        let mut puts: Vec<BatchPut> = Vec::with_capacity(seg_puts.len() + colidx_puts.len());
        for (k, seg) in seg_puts {
            puts.push(BatchPut::Typed {
                key: k,
                value: TypedValue::IndexSegment(seg),
            });
        }
        for (k, colidx) in colidx_puts {
            puts.push(BatchPut::Typed {
                key: k,
                value: TypedValue::ColumnIndex(colidx),
            });
        }
        pager.batch_put(&puts).unwrap();

        remaining -= batch;
    }

    // Write Manifest and Bootstrap (key 0) in a single batched put.
    let final_puts = vec![
        BatchPut::Typed {
            key: manifest_pkey,
            value: TypedValue::Manifest(Manifest {
                columns: manifest_entries,
            }),
        },
        BatchPut::Typed {
            key: BOOTSTRAP_PKEY,
            value: TypedValue::Bootstrap(Bootstrap {
                manifest_physical_key: manifest_pkey,
            }),
        },
    ];
    pager.batch_put(&final_puts).unwrap();

    // Keep side-effects alive for the optimizer.
    black_box(manifest_pkey);
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
