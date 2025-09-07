A per-segment “value order” (a permutation array that sorts entries by **value bytes**) is a practical way to unlock efficient value-range scans, top-K, and percentile queries without reordering your data blob. It plays nicely with your append-only, segment-sealed design and your existing `ValueLayout` (fixed/variable). Below is what it buys you, what it costs, and how to wire it in.

# What it is

For each sealed `IndexSegment`, store a permutation `value_order` so that:

* `value_order[r] = i` maps a **rank by value** (0..n-1) to the **row index** `i` within the segment.
* Comparisons use **raw value bytes** (lexicographic) via existing `ValueLayout` slicing. Works for any logical type since you already compare bytes.

You don’t reorder the data blob—this is just an index.

# Why it helps

* **Value range scans**: with segment-level `value_min/value_max` you already prune segments; inside a kept segment you can binary-search the **rank space** to find `[low, high]` in `O(log n)` compares (each compare reads one value slice).
* **Top-K / bottom-K**: walk `value_order` from one end; early-exit without touching all entries.
* **Percentiles/quantiles**: pick rank ≈ `p * n`.

# Space & build cost (realistic)

Let `n = seg.n_entries`.

* **Space (uncompressed)**:

  * If `n ≤ 65,536` (your default `segment_max_entries`): store `u16` → **2·n bytes**. With 65,536 entries: **128 KiB** per segment.
  * If larger: `u32` → **4·n bytes** (262 KiB at 65,536; 4 MB at 1 M).
* **Build**:

  * Create indices `0..n-1` and `sort_by` a comparator that slices values via `ValueLayout`.

    * Fixed 4/8 B: very fast (CPU can inline memcmp).
    * Variable length: still fine; compares exit at first differing byte.
  * Time ≈ `O(n log n)` comparisons. For fixed-width numeric you can radix-sort in `O(n)` if you want later.

Because segments are sealed once, this cost is a one-time build at ingest.

# Where to store it

Keep it **optional** so you don’t pay for columns that don’t need value scans.

Minimal schema tweak:

```rust
#[derive(Debug, Clone, Encode, Decode)]
pub struct IndexSegment {
    pub data_physical_key: PhysicalKey,
    pub n_entries: IndexEntryCount,
    pub logical_key_bytes: Vec<u8>,
    pub key_layout: KeyLayout,
    pub value_layout: ValueLayout,

    // New (optional) auxiliary index:
    pub value_order_physical_key: Option<PhysicalKey>, // raw blob of u16/u32
    pub value_order_width: Option<u8>,                 // 2 or 4 (bytes per entry)
}
```

Why on `IndexSegment` (not `IndexSegmentRef`)? The order array can be 100+ KiB; it’s natural to co-locate it with other segment-local metadata, and you can fetch it along with the segment in one batched read.

# How to build (ingest path)

Add a knob to `AppendOptions`:

```rust
#[derive(Clone, Debug)]
pub struct AppendOptions {
    // ...
    pub build_value_order: bool, // default false
}
```

When sealing a segment in `append_many`:

```rust
if opts.build_value_order {
    let order_width = if seg.n_entries <= 65_536 { 2 } else { 4 };
    let aux_key = self.pager.alloc_many(1)?.[0];

    // Build permutation without copying values
    let mut idx: Vec<u32> = (0..seg.n_entries as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        let (aa, bb) = (a as usize, b as usize);
        let a_slice = value_slice(&seg.value_layout, aa); // [a..b) into data blob
        let b_slice = value_slice(&seg.value_layout, bb);
        a_slice.cmp(b_slice)
    });

    // Downcast to u16 if possible, then write as RAW blob
    let bytes: Vec<u8> = if order_width == 2 {
        let mut out = Vec::with_capacity(idx.len()*2);
        for v in idx { out.extend_from_slice(&(v as u16).to_le_bytes()); }
        out
    } else {
        let mut out = Vec::with_capacity(idx.len()*4);
        for v in idx { out.extend_from_slice(&v.to_le_bytes()); }
        out
    };

    puts_batch.push(BatchPut::Raw { key: aux_key, bytes });
    seg.value_order_physical_key = Some(aux_key);
    seg.value_order_width = Some(order_width);
}
```

`value_slice(...)` is just your existing offset math from `ValueLayout` (you already do it in `get_many`).

# How to query a value range

Within a segment:

1. **Load** the segment (typed), its **data blob** (raw), and, if present, its **value\_order** (raw). You already batch these.
2. **Binary search rank space** `[0..n)` using a comparator that reads the value at `value_order[mid]`:

```rust
fn rank_lower_bound(seg: &IndexSegment, order: &[u8], low: &[u8]) -> usize { /* O(log n) */ }
fn rank_upper_bound(seg: &IndexSegment, order: &[u8], high: &[u8]) -> usize { /* O(log n) */ }
```

3. The rank interval `[lo_rank .. hi_rank)` gives **positions** in value order. Map ranks to row indices, then to slices via `ValueLayout`, and stream results (respecting `limit`).

This keeps per-segment work \~`O(log n + hits)` and no full scans.

# When it’s worth it (rules of thumb)

Build `value_order` if **any** is true:

* You expect **value-range predicates** (e.g., “value between A and B”) on the column.
* You need **top-K/percentiles** over the column without full scans.
* Values are **fixed-width ≤ 16 B** or short strings; build time and comparators are cheap.

Probably **skip** it if:

* Workload is **point-lookups by key** only (your current benchmark).
* Column has **very large values** (comparisons are expensive) and range scans are rare.
* Column is **very low cardinality** → use **dictionary + postings** instead (far smaller).

# Size math examples (default 65,536-entry segments)

* `u16` order array per segment: **128 KiB**.
* 16 segments per million rows per column → **\~2 MiB per column** for value order.
* 10 such columns → **\~20 MiB** of auxiliary order data. Often acceptable; make it opt-in.

(If you raise `segment_max_entries`, recompute accordingly; `u32` doubles cost.)

# Alternatives / complements

* **Histograms / KLL / t-digest** per segment for fast **approximate** pruning or percentile estimates with tiny footprints.
* **Dictionary encoding** (value → list of row indices) for low-cardinality text/enums.
* **Block-wise order** (e.g., order within 4 K blocks) to cut footprint by \~×B but keep good selectivity.

# TL;DR recommendation

* Add **optional** `value_order` to `IndexSegment` as above.
* Build it only when `AppendOptions.build_value_order = true` (or via heuristic).
* Use it to implement an efficient `get_by_value_range(field_id, low, high, limit)` that:

  * prunes segments via `IndexSegmentRef.value_min/value_max`,
  * binary-searches **rank space** inside each kept segment.

You’ll get strong value-range performance with modest, predictable per-segment overhead and no changes to your data layout or write path semantics.
