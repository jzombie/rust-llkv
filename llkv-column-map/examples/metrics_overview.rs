//! Example: observing pager-hit metrics collected by ColumnStore (IoStats).
//!
//! What this does:
//! 1) Initializes an empty store (Bootstrap->Manifest).
//! 2) Appends fixed-width columns in one batch.
//! 3) Appends variable-width columns (chunked) in one batch.
//! 4) Adds new columns later (forces a Manifest update).
//! 5) Performs a mixed append across existing + new columns at once.
//! 6) Runs a multi-column read with `get_many`.
//! 7) Calls `render_storage_ascii` to show current on-disk layout
//!    (note: this also issues pager gets, and you’ll see the extra metrics).
//!
//! After each phase we print the IoStats counters and reset them so you can
//! see the cost of just that phase.

use llkv_column_map::{AppendOptions, ColumnStore, Put, ValueMode, pager::MemPager};

type Field = u32;

// ---------- tiny helpers to build synthetic data ----------

#[inline]
fn be_key(row: u64) -> Vec<u8> {
    row.to_be_bytes().to_vec()
}

/// Fixed-width value of exactly `w` bytes, deterministically derived from (row, field).
fn fixed_value(row: u64, field: Field, w: usize) -> Vec<u8> {
    let seed = (row ^ field as u64).to_le_bytes();
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

/// Variable-length value in [min, max], filled with a repeating byte from (row, field).
fn var_value(row: u64, field: Field, min: usize, max: usize) -> Vec<u8> {
    let span = (max - min + 1) as u64;
    let mix = row
        .wrapping_mul(1103515245)
        .wrapping_add(field as u64)
        .rotate_left(13);
    let len = (min as u64 + (mix % span)) as usize;
    let b = (((row as u32).wrapping_add(field)) & 0xFF) as u8;
    vec![b; len]
}

/// Build a `Put { field_id, items }` for rows [start, end) with fixed width `w`.
fn build_put_fixed(field: Field, start: u64, end: u64, w: usize) -> Put {
    let mut items = Vec::with_capacity((end - start) as usize);
    for r in start..end {
        items.push((be_key(r), fixed_value(r, field, w)));
    }
    Put {
        field_id: field,
        items,
    }
}

/// Build a `Put` for rows [start, end) with variable length values in [min, max].
fn build_put_var(field: Field, start: u64, end: u64, min: usize, max: usize) -> Put {
    let mut items = Vec::with_capacity((end - start) as usize);
    for r in start..end {
        items.push((be_key(r), var_value(r, field, min, max)));
    }
    Put {
        field_id: field,
        items,
    }
}

/// Pretty-print the IoStats and then (optionally) reset them.
fn show_stats(label: &str, store: &mut ColumnStore<'_, MemPager>, reset_after: bool) {
    let s = store.io_stats().clone();
    println!(
        "\n== {} ==\n  batches: {}\n  puts:   raw={} typed={}\n  gets:   raw={} typed={}",
        label, s.batches, s.put_raw_ops, s.put_typed_ops, s.get_raw_ops, s.get_typed_ops
    );
    if reset_after {
        store.reset_io_stats();
    }
}

fn main() {
    // Use the in-memory pager from the crate.
    let mut pager = MemPager::default();
    let mut store = ColumnStore::init_empty(&mut pager);

    // A common append configuration; we’ll tweak the mode per phase.
    let mut opts = AppendOptions {
        mode: ValueMode::Auto,
        segment_max_entries: 16_384,
        segment_max_bytes: 2 * 1024 * 1024,
        last_write_wins_in_batch: true,
    };

    // ---------------- Phase 1: fixed-width append ----------------
    // Two columns, each 10_000 rows of width-8 payloads, in one append_many.
    {
        let puts = vec![
            build_put_fixed(100, 0, 10_000, 8),
            build_put_fixed(101, 0, 10_000, 8),
        ];
        opts.mode = ValueMode::ForceFixed(8);
        store.append_many(puts, opts.clone());
        show_stats(
            "Phase 1: append fixed-width cols 100 & 101",
            &mut store,
            true,
        );
    }

    // ---------------- Phase 2: variable-width append (chunked) ----------------
    // Two columns with variable-length values in [5, 25], still in one append_many.
    // Chunking will likely split into multiple segments depending on thresholds.
    {
        let puts = vec![
            build_put_var(200, 0, 12_345, 5, 25),
            build_put_var(201, 0, 12_345, 5, 25),
        ];
        opts.mode = ValueMode::Auto; // let it detect variable width
        opts.segment_max_entries = 4_000; // encourage multiple segments
        opts.segment_max_bytes = 128 * 1024;
        store.append_many(puts, opts.clone());
        show_stats(
            "Phase 2: append variable-width cols 200 & 201 (chunked)",
            &mut store,
            true,
        );
    }

    // ---------------- Phase 3: add new columns later ----------------
    // This will cause a Manifest update and new ColumnIndex blobs to be created.
    {
        let puts = vec![
            build_put_fixed(300, 0, 2_000, 4),
            build_put_var(301, 0, 2_000, 50, 200),
        ];
        opts.mode = ValueMode::Auto;
        store.append_many(puts, opts.clone());
        show_stats(
            "Phase 3: append new cols 300 (fixed) & 301 (var) — forces Manifest update",
            &mut store,
            true,
        );
    }

    // ---------------- Phase 4: mixed append into existing + new ----------------
    {
        // We intentionally write ONLY three keys for column 100 (5, 7, 9).
        // Later, if the probe asks for any other key in col 100, it will (correctly) be MISSING.
        //
        // Also demonstrate "last-write-wins" inside a single Put: we first stage a different
        // value for key=7, then overwrite it before the append. Only the LAST value for the
        // same logical key is kept within the batch (dedup happens before layout is chosen).
        let mut items_100 = vec![
            (be_key(5), fixed_value(5, 100, 8)),
            (be_key(7), b"OVERWRITE7!".to_vec()), // temp; will be replaced to keep width=8
            (be_key(9), fixed_value(9, 100, 8)),
        ];
        // Keep column 100 fixed-width (8 bytes). If we left "OVERWRITE7!" (11 bytes),
        // AUTO layout would flip this chunk to variable width or (with ForceFixed) panic.
        // By replacing it with an 8-byte value, all three entries are uniform → fixed(8).
        items_100[1].1 = fixed_value(7, 100, 8);
        let put_100 = Put {
            field_id: 100,
            items: items_100,
        };

        // New variable-width column 999:
        // We insert keys in the inclusive range [0, 1233] (i.e., 1_234 keys total).
        // Any probe against col 999 that asks for a key OUTSIDE that range will be MISSING.
        //
        // NOTE on "min/max vs existence":
        // Segment min/max bounds are a coarse prune. A key can fall within a segment's
        // logical_key_min..=logical_key_max and STILL be MISSING if that exact logical key
        // isn't present in the segment (binary search returns None). That's expected.
        let put_999 = build_put_var(999, 0, 1_234, 10, 30);

        // IMPORTANT: use Auto so each put chooses its own layout independently:
        // - col 100 chunk stays fixed(8)
        // - col 999 chunk becomes variable (10..30 bytes per value)
        opts.mode = ValueMode::Auto;
        store.append_many(vec![put_100, put_999], opts.clone());

        // After this append:
        // - IoStats "puts typed" includes:
        //     * 1 IndexSegment for col 100
        //     * 1 IndexSegment for col 999
        //     * ColumnIndex rewrites for any columns in cache that changed
        //     * (and if col 999 was new, a Manifest rewrite)
        // - Reads that show `MISSING` are usually:
        //     * a key never written in col 100 (we only wrote 5,7,9 here), or
        //     * a key outside [0,1233] for col 999, or
        //     * a key within a segment’s min/max that simply doesn’t exist there
        //       (min/max is prune-only; exact membership is decided by binary search).
        show_stats(
            "Phase 4: mixed append (existing col 100 + new col 999)",
            &mut store,
            true,
        );
    }

    // ---------------- Phase 5: multi-column read ----------------
    // Query across multiple fields; internally this batches segment + data loads.
    {
        let query_items = vec![
            (
                100u32,
                vec![be_key(0), be_key(7), be_key(9), be_key(123456)],
            ),
            (200u32, vec![be_key(123), be_key(9999), be_key(12_344)]),
            (301u32, vec![be_key(100), be_key(1_999), be_key(2_000)]), // last is missing
            (999u32, vec![be_key(0), be_key(777), be_key(1_233)]),
        ];

        let results = store.get_many(query_items);
        // Quick sanity print (don’t spam; just show presence/lengths).
        println!("\nSome read results (len or MISSING):");
        for (rowset_i, rowset) in results.iter().enumerate() {
            print!("  field[{}]:", rowset_i);
            for v in rowset {
                match v {
                    Some(bytes) => print!(" {}", bytes.len()),
                    None => print!(" MISSING"),
                }
            }
            println!();
        }

        show_stats("Phase 5: get_many over multiple fields", &mut store, true);
    }

    // ---------------- Phase 6: introspection (storage ASCII) ----------------
    // The `describe_storage`/`render_storage_ascii` are read-only but do issue
    // batched raw/typed gets internally. We isolate their cost with fresh stats.
    {
        let ascii = store.render_storage_ascii();
        println!("\n--- Storage Layout (ASCII) ---\n{}\n", ascii);
        show_stats("Phase 6: render_storage_ascii()", &mut store, true);
    }

    println!("\nDone.");
}
