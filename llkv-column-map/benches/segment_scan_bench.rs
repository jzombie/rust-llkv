//! Bench: read-only (scan) performance vs segment size & duplication.
//!
//! Measures ONLY scan time. Seeding happens once per config *outside* timers.
//!
//! Datasets per segment size:
//!   • CLEAN:        1 generation (no cross-segment dup winners)
//!   • FRAGMENTED:   G generations rewriting all keys (cross-segment dups),
//!                   plus optional *in-batch* duplicates (more fragmentation).
//!
//! We time full-range LWW scans (forward + reverse). Iterator creation is
//! included (it’s part of the read path). Seeding is *not* timed.
//!
//! Env knobs (examples):
//!   ROWS=1000000                 # total logical rows (default: 1_000_000)
//!   SEG_SIZES=200000,50000,20000 # segment_max_entries list
//!   GENS=4                       # generations for FRAGMENTED (default: 4)
//!   KEEP_DUP_IN_BATCH=0          # 1 => keep in-batch dup keys (default: 0)
//!
//! Run:
//!   cargo bench --bench segment_scan_read_bench
//!
//! Correctness:
//!   We assert once per dataset (outside timers) that LWW returns exactly
//!   ROWS winners. After you verify locally, you can comment out those asserts.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use llkv_column_map::ColumnStore;
use llkv_column_map::codecs::big_endian::u64_be_array;
use llkv_column_map::column_store::read_scan::{Direction, OrderBy, ValueScanOpts};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};

use std::borrow::Cow;
use std::env;
use std::hint::black_box;
use std::ops::Bound;

// ------------------------- config helpers -------------------------

fn parse_env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|s| s.replace('_', "").parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_env_bool01(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .and_then(|s| match s.as_str() {
            "1" | "true" | "TRUE" | "yes" | "YES" => Some(true),
            "0" | "false" | "FALSE" | "no" | "NO" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

fn parse_env_list_usize(key: &str, default: &[usize]) -> Vec<usize> {
    match env::var(key) {
        Ok(s) => s
            .split(',')
            .filter_map(|x| x.trim().replace('_', "").parse::<usize>().ok())
            .collect(),
        Err(_) => default.to_vec(),
    }
}

// ------------------------- data helpers -------------------------

#[inline]
fn row_key(i: u64) -> Vec<u8> {
    // "r00000000" style, fixed width
    format!("r{:08}", i).into_bytes()
}

#[inline]
fn be_u64_vec(x: u64) -> Vec<u8> {
    u64_be_array(x).to_vec()
}

// Distinct value band per generation so value-ordered scans keep ranges separated.
#[inline]
fn value_for_generation(generation_idx: usize, i: u64) -> u64 {
    (generation_idx as u64) * 10_000_000_000 + i
}

// Seed a single column into an existing store (no borrowed returns).
fn seed_column_into(
    store: &ColumnStore<'_, MemPager>,
    field_id: LogicalFieldId,
    rows: u64,
    segment_max_entries: usize,
    generations: usize,
    keep_dup_in_batch: bool,
) {
    assert!(generations >= 1);

    let opts = AppendOptions {
        mode: ValueMode::ForceFixed(8),
        segment_max_entries,
        segment_max_bytes: 64 << 20,
        last_write_wins_in_batch: !keep_dup_in_batch,
    };

    // Use a moderate chunk; sealing is controlled by segment_max_entries.
    let chunk: u64 = 100_000;

    for gidx in 0..generations {
        let mut start = 0u64;
        while start < rows {
            let end = std::cmp::min(start + chunk, rows);
            let mut items = Vec::with_capacity((end - start) as usize);

            for i in start..end {
                items.push((
                    Cow::<[u8]>::Owned(row_key(i)),
                    Cow::<[u8]>::Owned(be_u64_vec(value_for_generation(gidx, i))),
                ));
            }

            if keep_dup_in_batch && (end - start) >= 64 {
                // sprinkle some in-batch dups on the first ~1/64th of the chunk
                let dup_end = start + ((end - start) / 64);
                for i in start..dup_end {
                    items.push((
                        Cow::<[u8]>::Owned(row_key(i)),
                        Cow::<[u8]>::Owned(be_u64_vec(value_for_generation(gidx, i) + 7)),
                    ));
                }
            }

            store.append_many(vec![Put { field_id, items }], opts.clone());
            start = end;
        }
    }
}

fn scan_full_count_forward(store: &ColumnStore<'_, MemPager>, fid: LogicalFieldId) -> usize {
    let it = store
        .scan_values_lww(
            fid,
            ValueScanOpts {
                order_by: OrderBy::Value,
                dir: Direction::Forward,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        )
        .expect("forward iterator");
    let mut c = 0usize;
    for v in it {
        black_box(v.value.start()); // touch to avoid DCE
        c += 1;
    }
    c
}

fn scan_full_count_reverse(store: &ColumnStore<'_, MemPager>, fid: LogicalFieldId) -> usize {
    let it = store
        .scan_values_lww(
            fid,
            ValueScanOpts {
                order_by: OrderBy::Value,
                dir: Direction::Reverse,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        )
        .expect("reverse iterator");
    let mut c = 0usize;
    for v in it {
        black_box(v.value.end()); // touch to avoid DCE
        c += 1;
    }
    c
}

// ----------------------------- the bench ------------------------------

pub fn segment_scan_read_bench(c: &mut Criterion) {
    let rows = parse_env_usize("ROWS", 1_000_000) as u64; // default: 1M
    let seg_sizes = parse_env_list_usize("SEG_SIZES", &[200_000, 50_000, 20_000]);
    let gens_fragmented = {
        let v = parse_env_list_usize("GENS", &[4]); // default: 4 gens
        *v.first().unwrap_or(&4)
    };
    let keep_dup_in_batch = parse_env_bool01("KEEP_DUP_IN_BATCH", false);

    let mut group = c.benchmark_group("read_only_scan_vs_segment_size");
    group.throughput(Throughput::Elements(rows as u64));
    group.sample_size(10);

    for seg in seg_sizes {
        // ---------- CLEAN dataset (1 generation, no in-batch dups) ----------
        let pager_clean = MemPager::default();
        let store_clean = ColumnStore::init_empty(&pager_clean);
        let fid_clean: LogicalFieldId = 777;
        seed_column_into(&store_clean, fid_clean, rows, seg, 1, false);

        // One-time correctness (not timed)
        {
            let fwd = scan_full_count_forward(&store_clean, fid_clean);
            let rev = scan_full_count_reverse(&store_clean, fid_clean);
            assert_eq!(
                fwd as u64, rows,
                "clean forward winners mismatch (seg={seg})"
            );
            assert_eq!(
                rev as u64, rows,
                "clean reverse winners mismatch (seg={seg})"
            );
        }

        group.bench_function(BenchmarkId::new("CLEAN_forward_full_scan", seg), |b| {
            b.iter(|| black_box(scan_full_count_forward(&store_clean, fid_clean)))
        });
        group.bench_function(BenchmarkId::new("CLEAN_reverse_full_scan", seg), |b| {
            b.iter(|| black_box(scan_full_count_reverse(&store_clean, fid_clean)))
        });

        // ------ FRAGMENTED dataset (G gens + optional in-batch dups) -------
        let pager_frag = MemPager::default();
        let store_frag = ColumnStore::init_empty(&pager_frag);
        let fid_frag: LogicalFieldId = 778;
        seed_column_into(
            &store_frag,
            fid_frag,
            rows,
            seg,
            gens_fragmented,
            keep_dup_in_batch,
        );

        // One-time correctness (not timed)
        {
            let fwd = scan_full_count_forward(&store_frag, fid_frag);
            let rev = scan_full_count_reverse(&store_frag, fid_frag);
            assert_eq!(
                fwd as u64, rows,
                "frag forward winners mismatch (seg={seg})"
            );
            assert_eq!(
                rev as u64, rows,
                "frag reverse winners mismatch (seg={seg})"
            );
        }

        let label_fwd = format!(
            "FRAG_forward_full_scan(seg={},gens={},dup_in_batch={})",
            seg, gens_fragmented, keep_dup_in_batch as u8
        );
        let label_rev = format!(
            "FRAG_reverse_full_scan(seg={},gens={},dup_in_batch={})",
            seg, gens_fragmented, keep_dup_in_batch as u8
        );

        group.bench_function(BenchmarkId::from_parameter(label_fwd), |b| {
            b.iter(|| black_box(scan_full_count_forward(&store_frag, fid_frag)))
        });
        group.bench_function(BenchmarkId::from_parameter(label_rev), |b| {
            b.iter(|| black_box(scan_full_count_reverse(&store_frag, fid_frag)))
        });
    }

    group.finish();
}

criterion_group!(benches, segment_scan_read_bench);
criterion_main!(benches);
