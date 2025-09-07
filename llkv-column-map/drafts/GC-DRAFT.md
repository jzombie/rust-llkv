# ColumnStore GC (Reclaim) Notes

This summarizes a pragmatic, low-risk approach to reclaiming space in the append-only `ColumnStore`. It assumes the current design:

* Writes create **segments** (index + data blob).
* Deletes are **tombstones** (variable layout entries with zero-length values).
* Reads are **correct** without GC: newest-first shadowing and `[min,max]` pruning; zero-len values = “missing”.

GC here is purely an optimization for disk usage and read I/O, not correctness.

---

## Goals

* **Reclaim space** by removing segments that are completely shadowed by newer data/tombstones.
* **Reduce read cost** when many overlapping segments exist.
* **Never load “everything”**: operate with strict per-run budgets; fetch only a few typed segments at a time.
* **Crash-safe & simple**: prefer safe leaks over risk of breaking reads.

---

## When is GC needed?

* You can **run for a long time with no GC**. Reads prune by `[min,max]` and stop after the newest hit (value or tombstone).
* Start GC when you observe:

  * Growing **segments per column** (esp. overlapping ranges).
  * Rising `io_stats.get_typed_ops` for the same queries (keys match multiple segments before a hit).
  * Storage pressure.

---

## Strategy: Index-Only, Budgeted “Fully Shadowed” Reclaim

**Key idea:** Decide reclaimability using **typed index data only** (logical keys + layouts). No data blobs required.

A segment **S\_old** is *fully shadowed* if, for **every key** in it, there exists a **newer** segment containing that key **or** a tombstone for it. If so, removing **S\_old** is safe.

### Why index-only?

* To check presence of a key you only need the newer segments’ **logical keys**.
* To detect tombstones you only need the newer segment’s **value layout offsets** (zero-length window). No need to touch data blobs.

---

## Per-Run Budget & Scheduling

Run GC **opportunistically** with a **small budget**:

* Trigger after `append_many` **only if** a column’s segment count exceeds a threshold, or run on a timer/maintenance thread.
* Each run touches at most:

  * **N columns**, starting with those having the most segments.
  * **M candidate segments per column** (oldest first).
  * **K newer segments per candidate** (newest first) during checks.
  * **T keys per candidate** (optional sampling/early-exit).

This keeps memory and I/O bounded and prevents “load everything”.

---

## Safe Update Ordering

When reclaiming a **fully shadowed** segment:

1. **Update the ColumnIndex** (remove the segment ref) and **persist** it.
2. **Free** the segment’s **index** and **data** physical keys.

If a crash happens after (1) but before (2): safe leak.
If you reversed the order you could break reads—so don’t.

---

## Pseudocode (sketch)

```rust
struct ReclaimOpts {
    max_columns: usize,              // how many columns to consider this run
    max_candidates_per_col: usize,   // oldest segments to test per column
    max_newer_to_check: usize,       // how many newer segments to consult (newest-first)
    max_keys_to_check: Option<usize>,// Some(N)=>sample/early-exit; None=>check all keys
    min_shadowed_fraction: f32,      // e.g. 1.0 for “fully”; or <1.0 if you later add partial rewrite
}

fn reclaim_budgeted<P: Pager>(store: &ColumnStore<P>, opts: ReclaimOpts) {
    // 1) Pick columns with largest segment counts.
    let columns = pick_columns_by_segment_count(store, opts.max_columns);

    for fid in columns {
        // 2) Oldest-first candidates.
        let segs = store.colindex(fid).segments.clone();
        for cand in segs.iter().rev().take(opts.max_candidates_per_col) {
            // 3) Collect up to K newer segments that overlap [min,max] by key-range.
            let newer = newest_overlapping_segments(&segs, cand, opts.max_newer_to_check);

            // 4) Load typed index data for {cand} + {newer}. No data blobs.
            let typed = batch_load_index_segments(newer.iter().chain([cand]));

            // 5) For each key in cand (or a sample):
            //    - binary_search across newer (newest-first)
            //    - count as shadowed if found OR if found with zero-length (tombstone).
            let shadowed_ratio = check_shadowed_fraction(&typed, cand, opts.max_keys_to_check);

            if shadowed_ratio >= opts.min_shadowed_fraction {
                // 6) Persist ColumnIndex without cand, then free cand.index_pk & cand.data_pk.
                persist_colindex_without(store, fid, cand.index_physical_key);
                store.do_frees(&[cand.index_physical_key, cand.data_physical_key]);
                // budgeted: optionally stop early if we hit a per-run free limit
            }
        }
    }
}
```

**Notes**

* `newest_overlapping_segments` uses `[min,max]` to prune aggressively.
* `check_shadowed_fraction` uses binary search with the keys from `cand` against each newer segment’s key layout; tombstone detection = `a == b` in `value_offsets`.
* Start with `min_shadowed_fraction = 1.0` (only full reclaims). Partial compaction is optional future work.

---

## Performance Characteristics

* **Memory:** bounded by the number of typed segments you load concurrently (1 candidate + K newer). No data blobs are fetched.
* **I/O:** one batched typed get for that small set, plus a single typed put (ColumnIndex) and 2 frees when reclaiming.
* **Time:** roughly `O(|keys_in_candidate| * log K)` per candidate (binary search over ≤K newer segments).

---

## Practical Defaults (good starting points)

* `max_columns`: 4–16
* `max_candidates_per_col`: 2–8 (oldest first)
* `max_newer_to_check`: 4–16 (newest first)
* `max_keys_to_check`: `None` (exact) or `Some(4_096)` with early-exit once “not fully shadowed” is proven
* `min_shadowed_fraction`: `1.0` (full reclaim)

These are **reclaim-only knobs** (unrelated to ingest or read paths).

---

## Metrics to Watch

Use `IoStats`:

* **`get_typed_ops`** during reads: stays ≈1/key when overlap is small; climbs as overlap grows.
* **`free_ops`** during GC runs: confirms reclaimed keys.
* Consider tracking “segments per column” and “bytes freed” (estimated from `describe_storage()`).

---

## What This GC Does Not Do (yet)

* **Partial compaction** (rewriting a candidate with only its live subset). That’s a follow-up once full-shadow reclaim is stable.
* **Global dedupe** across many segments at once.
* **Background threads**: you can still run it periodically or after `append_many` when segment count crosses a threshold.

---

## TL;DR

* You don’t **need** GC for correctness; reads are already correct and efficient when overlap is small.
* Add a **budgeted, index-only** reclaim pass that removes **fully shadowed** old segments.
* Run it **occasionally** (threshold-triggered or scheduled) with small budgets so it never tries to load “everything.”
* Update ColumnIndex **first**, then free pkeys—safe and simple.
