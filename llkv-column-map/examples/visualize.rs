//! This example ingests data in 4 batches using the crate's high-level API
//! (ColumnStore), then writes ONE final `.dot` file where nodes are color-
//! coded by the batch in which they were created.
//!
//! --- How to View the Output ---
//!
//! Online: https://dreampuf.github.io/GraphvizOnline/?engine=dot
//!
//! or:
//!   1) Install Graphviz
//!      - macOS (Homebrew):      `brew install graphviz`
//!      - Ubuntu/Debian:         `sudo apt-get update && sudo apt-get install graphviz`
//!      - Windows (Chocolatey):  `choco install graphviz`
//!        or (Scoop):            `scoop install graphviz`
//!
//!   2) Run this example
//!      `cargo run --example visualize`
//!
//!      This creates one `storage_layout.dot` in the project root and prints
//!      an ASCII summary to stdout. Nodes are colored by batch of creation:
//!      - batch 0 (bootstrap/manifest): white
//!      - batch 1: lightskyblue
//!      - batch 2: palegreen
//!      - batch 3: khaki
//!      - batch 4: lightpink
//!
//!   3) Generate an image
//!      - PNG:  `dot -Tpng storage_layout.dot -o storage_layout.png`
//!      - SVG:  `dot -Tsvg storage_layout.dot -o storage_layout.svg`

use std::collections::HashMap;
use std::time::Instant;

use llkv_column_map::storage::{StorageKind, pager::MemPager};
use llkv_column_map::types::PhysicalKey;
use llkv_column_map::{
    ColumnStore,
    column_store::write::{AppendOptions, Put, ValueMode},
};

// ---------------- Workload config (small, but shows batching clearly) --------

const BATCHES: usize = 4;

// Column 1: 500 rows, fixed width (4 bytes, auto-detected)
// Column 2: 500 rows, variable length (1..=21 bytes, auto-detected)
// Column 3: 5000 rows, fixed width (8 bytes, auto-detected)
const C1_ROWS: usize = 500;
const C2_ROWS: usize = 500;
const C3_ROWS: usize = 5_000;

fn build_put_for_col1(start: usize, end: usize) -> Option<Put<'static>> {
    let s = start.min(C1_ROWS);
    let e = end.min(C1_ROWS);
    if s >= e {
        return None;
    }
    let mut items = Vec::with_capacity(e - s);
    for i in s..e {
        let key = format!("id:{:06}", i).into_bytes();
        let val = (i as u32).to_le_bytes().to_vec(); // width=4
        items.push((key.into(), val.into()));
    }
    Some(Put { field_id: 1, items })
}

fn build_put_for_col2(start: usize, end: usize) -> Option<Put<'static>> {
    let s = start.min(C2_ROWS);
    let e = end.min(C2_ROWS);
    if s >= e {
        return None;
    }
    let mut items = Vec::with_capacity(e - s);
    for i in s..e {
        let key = format!("id:{:06}", i).into_bytes();
        let len = i % 21 + 1; // 1..21
        items.push((key.into(), vec![b'A' + (i % 26) as u8; len].into()));
    }
    Some(Put { field_id: 2, items })
}

fn build_put_for_col3(start: usize, end: usize) -> Option<Put<'static>> {
    let s = start.min(C3_ROWS);
    let e = end.min(C3_ROWS);
    if s >= e {
        return None;
    }
    let mut items = Vec::with_capacity(e - s);
    for i in s..e {
        let key = format!("k{:06}", i).into_bytes();
        let val = vec![0x55; 8]; // width=8
        items.push((key.into(), val.into()));
    }
    Some(Put { field_id: 3, items })
}

// ---------------- DOT rendering with batch coloring -------------------------

// TODO: Migrate to lib?
fn color_for_batch(b: usize) -> &'static str {
    match b {
        0 => "white", // bootstrap/manifest
        1 => "lightskyblue",
        2 => "palegreen",
        3 => "khaki",
        4 => "lightpink",
        _ => "lightgray",
    }
}

// TODO: Migrate to lib?
fn render_one_colored_dot(
    store: &ColumnStore<'_, MemPager>,
    created_in_batch: &HashMap<PhysicalKey, usize>,
) -> String {
    use std::fmt::Write;

    let nodes = store.describe_storage();

    // find bootstrap and manifest
    let mut bootstrap_pk: Option<PhysicalKey> = None;
    let mut manifest_pk: Option<PhysicalKey> = None;
    for n in &nodes {
        match n.kind {
            StorageKind::Bootstrap => bootstrap_pk = Some(n.pk),
            StorageKind::Manifest { .. } => manifest_pk = Some(n.pk),
            _ => {}
        }
    }

    let mut s = String::new();
    writeln!(&mut s, "digraph storage {{").unwrap();
    writeln!(&mut s, "  rankdir=LR;").unwrap();
    writeln!(&mut s, "  node [shape=box, fontname=\"monospace\"];").unwrap();

    // nodes
    for n in &nodes {
        let b = created_in_batch.get(&n.pk).copied().unwrap_or(0);
        let fill = color_for_batch(b);
        match &n.kind {
            StorageKind::Bootstrap => {
                writeln!(
                    &mut s,
                    "  n{} [label=\"Bootstrap pk={} bytes={}\" style=filled fillcolor={}];",
                    n.pk, n.pk, n.stored_len, fill
                )
                .unwrap();
            }
            StorageKind::Manifest { column_count } => {
                writeln!(
                    &mut s,
                    "  n{} [label=\"Manifest pk={} columns={} bytes={}\" style=filled fillcolor={}];",
                    n.pk, n.pk, column_count, n.stored_len, fill
                ).unwrap();
            }
            StorageKind::ColumnIndex {
                field_id,
                n_segments,
            } => {
                writeln!(
                    &mut s,
                    "  n{} [label=\"ColumnIndex pk={} field={} segs={} bytes={}\" style=filled fillcolor={}];",
                    n.pk, n.pk, field_id, n_segments, n.stored_len, fill
                ).unwrap();
            }
            StorageKind::IndexSegment {
                field_id,
                n_entries,
                layout,
                data_pkey,
                owner_colindex_pk,
            } => {
                let lay = match layout.fixed_width {
                    Some(w) => format!("fixed({})", w),
                    None => "variable".to_string(),
                };
                writeln!(
                    &mut s,
                    "  n{} [label=\"IndexSegment pk={} field={} entries={} layout={} idx_bytes={} (key_bytes={}, key_offs={}, val_meta={})\" style=filled fillcolor={}];",
                    n.pk,
                    n.pk,
                    field_id,
                    n_entries,
                    lay,
                    n.stored_len,
                    layout.key_bytes,
                    layout.key_offs_bytes,
                    layout.value_meta_bytes,
                    fill
                ).unwrap();
                // edges for this segment
                writeln!(&mut s, "  n{} -> n{};", owner_colindex_pk, n.pk).unwrap();
                writeln!(&mut s, "  n{} -> n{};", n.pk, data_pkey).unwrap();
            }
            StorageKind::DataBlob { owner_index_pk } => {
                writeln!(
                    &mut s,
                    "  n{} [label=\"DataBlob pk={} bytes={}\" style=filled fillcolor={}];",
                    n.pk, n.pk, n.stored_len, fill
                )
                .unwrap();
                writeln!(&mut s, "  n{} -> n{};", owner_index_pk, n.pk).unwrap();
            }
        }
    }

    // edges: bootstrap -> manifest, manifest -> all column indexes
    if let (Some(bpk), Some(mpk)) = (bootstrap_pk, manifest_pk) {
        writeln!(&mut s, "  n{} -> n{};", bpk, mpk).unwrap();
        for n in &nodes {
            if let StorageKind::ColumnIndex { .. } = n.kind {
                writeln!(&mut s, "  n{} -> n{};", mpk, n.pk).unwrap();
            }
        }
    }

    // legend
    writeln!(&mut s, "  subgraph cluster_legend {{").unwrap();
    writeln!(&mut s, "    label=\"Batch legend\";").unwrap();
    for b in 0..=4 {
        writeln!(
            &mut s,
            "    l{} [label=\"batch {}\" shape=box style=filled fillcolor={}];",
            b,
            b,
            color_for_batch(b)
        )
        .unwrap();
    }
    writeln!(&mut s, "    l0 -> l1 -> l2 -> l3 -> l4 [style=invis];").unwrap();
    writeln!(&mut s, "  }}").unwrap();

    writeln!(&mut s, "}}").unwrap();
    s
}

// ---------------- Main: multi-batch ingest then ONE colored DOT -------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pager = MemPager::default();
    let store = ColumnStore::init_empty(&pager);

    // record creation batch for each physical key we ever see
    // batch 0 = pre-existing (bootstrap, manifest)
    let mut created_in_batch: HashMap<PhysicalKey, usize> = HashMap::new();
    for n in store.describe_storage() {
        created_in_batch.insert(n.pk, 0);
    }

    let rows_per_batch = C3_ROWS.div_ceil(BATCHES);

    println!(
        "Ingesting col1={} rows, col2={} rows, col3={} rows in {} batches ({} rows/batch on col3)...",
        C1_ROWS, C2_ROWS, C3_ROWS, BATCHES, rows_per_batch
    );

    let t_total = Instant::now();
    for b in 0..BATCHES {
        let start = b * rows_per_batch;
        let end = (start + rows_per_batch).min(C3_ROWS);

        let mut puts: Vec<Put> = Vec::new();
        if let Some(p) = build_put_for_col1(start, end) {
            puts.push(p);
        }
        if let Some(p) = build_put_for_col2(start, end) {
            puts.push(p);
        }
        if let Some(p) = build_put_for_col3(start, end) {
            puts.push(p);
        }
        if puts.is_empty() {
            continue;
        }

        let t_batch = Instant::now();
        store.append_many(
            puts,
            AppendOptions {
                mode: ValueMode::Auto,
                segment_max_entries: 512,
                segment_max_bytes: 64 * 1024,
                last_write_wins_in_batch: true,
            },
        );
        let dt = t_batch.elapsed();
        println!("  batch {}: rows [{}..{}) in {:?}", b + 1, start, end, dt);

        // mark new physical keys as created in this batch (b+1)
        for n in store.describe_storage() {
            created_in_batch.entry(n.pk).or_insert(b + 1);
        }
    }
    println!("Total ingest time: {:?}", t_total.elapsed());

    // simple probes (get_many returns Arc<[u8]> values)
    let got1 = store.get_many(vec![(
        1,
        vec![b"id:000010".to_vec(), b"id:009999".to_vec()],
    )]);
    println!("col=1 id:000010 -> {:?}", got1[0][0]);
    println!("col=1 id:009999 -> {:?}", got1[0][1]);

    let got2 = store.get_many(vec![(
        2,
        vec![b"id:000000".to_vec(), b"id:000100".to_vec()],
    )]);
    println!(
        "col=2 id:000000 len={:?}",
        got2[0][0].as_deref().map(|v| v.len())
    );
    println!(
        "col=2 id:000100 len={:?}",
        got2[0][1].as_deref().map(|v| v.len())
    );

    let got3 = store.get_many(vec![(3, vec![b"k000100".to_vec(), b"k004999".to_vec()])]);
    println!(
        "col=3 k000100 -> len={:?}",
        got3[0][0].as_deref().map(|v| v.len())
    );
    println!(
        "col=3 k004999 -> len={:?}",
        got3[0][1].as_deref().map(|v| v.len())
    );

    // ASCII summary of final layout
    let ascii = store.render_storage_ascii();
    println!("\n==== STORAGE ASCII ====\n{}", ascii);

    // ONE final DOT with batch-colored nodes
    let dot = render_one_colored_dot(&store, &created_in_batch);
    std::fs::write("storage_layout.dot", dot)?;
    println!("Wrote storage_layout.dot (single graph, nodes colored by batch)");

    Ok(())
}
