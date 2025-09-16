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
use std::sync::Arc;
use std::time::Instant;

use llkv_column_map::{
    ColumnStore,
    storage::pager::{MemPager, Pager},
    store::catalog::ColumnCatalog,
    store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorPageHeader},
    types::{LogicalFieldId, Namespace, PhysicalKey},
};

// ---------------- Workload config (small, but shows batching clearly) --------

const BATCHES: usize = 4;

// Column 1: 500 rows, fixed width (4 bytes, auto-detected)
// Column 2: 500 rows, variable length (1..=21 bytes, auto-detected)
// Column 3: 5000 rows, fixed width (8 bytes, auto-detected)
const C1_ROWS: usize = 500;
const C2_ROWS: usize = 500;
const C3_ROWS: usize = 5_000;

use arrow::array::{Array, ArrayRef, BinaryBuilder, UInt32Array, UInt64Array};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;

/// Helper to create a standard user-data LogicalFieldId.
fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

fn build_put_for_col1(start: usize, end: usize) -> Option<(LogicalFieldId, ArrayRef)> {
    let s = start.min(C1_ROWS);
    let e = end.min(C1_ROWS);
    if s >= e {
        return None;
    }
    let vals: Vec<u32> = (s..e).map(|i| i as u32).collect();
    Some((fid(1), Arc::new(UInt32Array::from(vals)) as ArrayRef))
}

fn build_put_for_col2(start: usize, end: usize) -> Option<(LogicalFieldId, ArrayRef)> {
    let s = start.min(C2_ROWS);
    let e = end.min(C2_ROWS);
    if s >= e {
        return None;
    }
    // NOTE: arrow 56 BinaryBuilder::new() takes no capacity arg.
    let mut b = BinaryBuilder::new();
    for i in s..e {
        let len = i % 21 + 1; // 1..21
        b.append_value(&vec![b'A' + (i % 26) as u8; len]);
    }
    Some((fid(2), Arc::new(b.finish()) as ArrayRef))
}

fn build_put_for_col3(start: usize, end: usize) -> Option<(LogicalFieldId, ArrayRef)> {
    let s = start.min(C3_ROWS);
    let e = end.min(C3_ROWS);
    if s >= e {
        return None;
    }
    let vals: Vec<u64> = (s..e).map(|_| 0x55u64).collect(); // width=8
    Some((fid(3), Arc::new(UInt64Array::from(vals)) as ArrayRef))
}

/// Build a RecordBatch from per-column arrays (same row count).
fn batch_from_pairs(pairs: &[(LogicalFieldId, ArrayRef)]) -> RecordBatch {
    let fields: Vec<Field> = pairs
        .iter()
        .enumerate()
        .map(|(i, (fid, arr))| {
            let mut md = std::collections::HashMap::new();
            md.insert("field_id".to_string(), u64::from(*fid).to_string());
            Field::new(&format!("c{}", i), arr.data_type().clone(), false).with_metadata(md)
        })
        .collect();

    // The store's append logic requires a `row_id` column.
    let num_rows = if pairs.is_empty() {
        0
    } else {
        pairs[0].1.len()
    };
    let row_id_field = Field::new("row_id", arrow::datatypes::DataType::UInt64, false);
    let row_id_array = Arc::new(UInt64Array::from_iter_values(0..num_rows as u64)) as ArrayRef;
    let mut final_fields = vec![row_id_field];
    final_fields.extend(fields);
    let mut final_arrays = vec![row_id_array];
    final_arrays.extend(pairs.iter().map(|(_, a)| Arc::clone(a)));

    let schema = Arc::new(Schema::new(final_fields));
    RecordBatch::try_new(schema, final_arrays).unwrap()
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

/// Discover all physical keys reachable from the catalog.
///
/// NOTE: CATALOG_ROOT_PKEY is pub(crate) and equals 0 in this crate. We
/// cannot import it here, so we inline the same value locally. If that
/// ever changes in the library, update ROOT_PK here accordingly.
const ROOT_PK: PhysicalKey = 0;

fn discover_all_pks<P: Pager>(pager: &P) -> Vec<PhysicalKey> {
    let mut out = Vec::new();
    out.push(ROOT_PK);

    if let Some(cat_blob) = pager.get_raw(ROOT_PK).unwrap() {
        let cat = ColumnCatalog::from_bytes(cat_blob.as_ref()).unwrap();
        for (_fid, desc_pk) in cat.map.iter() {
            out.push(*desc_pk);

            // Walk descriptor pages
            let desc_blob = pager.get_raw(*desc_pk).unwrap().unwrap();
            let desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
            let mut page_pk = desc.head_page_pk;
            while page_pk != 0 {
                out.push(page_pk);
                let page_blob = pager.get_raw(page_pk).unwrap().unwrap();
                let bytes = page_blob.as_ref();
                let hdr_sz = DescriptorPageHeader::DISK_SIZE;
                let hd = DescriptorPageHeader::from_le_bytes(&bytes[..hdr_sz]);

                // Collect chunk-related pkeys
                for i in 0..(hd.entry_count as usize) {
                    let off = hdr_sz + i * ChunkMetadata::DISK_SIZE;
                    let end = off + ChunkMetadata::DISK_SIZE;
                    let meta = ChunkMetadata::from_le_bytes(&bytes[off..end]);
                    out.push(meta.chunk_pk);
                    if meta.value_order_perm_pk != 0 {
                        out.push(meta.value_order_perm_pk);
                    }
                }

                page_pk = hd.next_page_pk;
            }
        }
    }

    out.sort_unstable();
    out.dedup();
    out
}

// TODO: Migrate to lib?
fn render_one_colored_dot<P: Pager>(
    pager: &P,
    created_in_batch: &HashMap<PhysicalKey, usize>,
) -> String {
    use std::fmt::Write;

    let mut s = String::new();

    // Nodes
    writeln!(&mut s, "digraph storage {{").unwrap();
    writeln!(&mut s, "  rankdir=LR;").unwrap();
    writeln!(&mut s, "  node [shape=box, fontname=\"monospace\"];").unwrap();

    // Catalog
    let cat_pk = ROOT_PK;
    let cat_color = color_for_batch(*created_in_batch.get(&cat_pk).unwrap_or(&0));
    if let Some(cat_blob) = pager.get_raw(cat_pk).unwrap() {
        let cat = ColumnCatalog::from_bytes(cat_blob.as_ref()).unwrap();
        writeln!(
            &mut s,
            "  n{} [label=\"Catalog pk={} entries={}\" style=filled \
             fillcolor={}];",
            cat_pk,
            cat_pk,
            cat.map.len(),
            cat_color
        )
        .unwrap();

        // Columns and descriptor chains
        for (fid, desc_pk) in cat.map.iter() {
            let desc_blob = pager.get_raw(*desc_pk).unwrap().unwrap();
            let desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
            let dcol = color_for_batch(*created_in_batch.get(desc_pk).unwrap_or(&0));
            writeln!(
                &mut s,
                "  n{} [label=\"ColumnDescriptor pk={} field={:?} rows={} \
                 chunks={}\" style=filled fillcolor={}];",
                desc_pk, desc_pk, fid, desc.total_row_count, desc.total_chunk_count, dcol
            )
            .unwrap();
            writeln!(&mut s, "  n{} -> n{};", cat_pk, desc_pk).unwrap();

            // Pages in chain
            let mut page_pk = desc.head_page_pk;
            let mut prev_page: Option<PhysicalKey> = None;
            while page_pk != 0 {
                let pcol = color_for_batch(*created_in_batch.get(&page_pk).unwrap_or(&0));
                let page_blob = pager.get_raw(page_pk).unwrap().unwrap();
                let bytes = page_blob.as_ref();
                let hdr_sz = DescriptorPageHeader::DISK_SIZE;
                let hd = DescriptorPageHeader::from_le_bytes(&bytes[..hdr_sz]);

                writeln!(
                    &mut s,
                    "  n{} [label=\"DescPage pk={} entries={}\" style=filled \
                     fillcolor={}];",
                    page_pk, page_pk, hd.entry_count, pcol
                )
                .unwrap();

                if let Some(ppk) = prev_page {
                    writeln!(&mut s, "  n{} -> n{};", ppk, page_pk).unwrap();
                } else {
                    writeln!(&mut s, "  n{} -> n{};", desc_pk, page_pk).unwrap();
                }

                // Entries on page
                for i in 0..(hd.entry_count as usize) {
                    let off = hdr_sz + i * ChunkMetadata::DISK_SIZE;
                    let end = off + ChunkMetadata::DISK_SIZE;
                    let meta = ChunkMetadata::from_le_bytes(&bytes[off..end]);

                    // Data blob
                    if let Some(b) = pager.get_raw(meta.chunk_pk).unwrap() {
                        let len = b.as_ref().len();
                        let col =
                            color_for_batch(*created_in_batch.get(&meta.chunk_pk).unwrap_or(&0));
                        writeln!(
                            &mut s,
                            "  n{} [label=\"Data pk={} bytes={}\" style=filled \
                             fillcolor={}];",
                            meta.chunk_pk, meta.chunk_pk, len, col
                        )
                        .unwrap();
                        writeln!(&mut s, "  n{} -> n{};", page_pk, meta.chunk_pk).unwrap();
                    }

                    // Value-order permutation
                    if meta.value_order_perm_pk != 0 {
                        if let Some(b) = pager.get_raw(meta.value_order_perm_pk).unwrap() {
                            let len = b.as_ref().len();
                            let col = color_for_batch(
                                *created_in_batch
                                    .get(&meta.value_order_perm_pk)
                                    .unwrap_or(&0),
                            );
                            writeln!(
                                &mut s,
                                "  n{} [label=\"Perm pk={} bytes={}\" style=filled \
                                 fillcolor={}];",
                                meta.value_order_perm_pk, meta.value_order_perm_pk, len, col
                            )
                            .unwrap();
                            writeln!(&mut s, "  n{} -> n{};", page_pk, meta.value_order_perm_pk)
                                .unwrap();
                        }
                    }
                }

                prev_page = Some(page_pk);
                page_pk = hd.next_page_pk;
            }
        }
    } else {
        writeln!(
            &mut s,
            "  n{} [label=\"Catalog pk={} (empty)\" style=filled fillcolor={}];",
            cat_pk, cat_pk, cat_color
        )
        .unwrap();
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
    let pager = Arc::new(MemPager::default());
    // Keep a handle to the pager so we can walk storage for visualization.
    let store = ColumnStore::open(Arc::clone(&pager)).unwrap();

    // record creation batch for each physical key we ever see
    // batch 0 = pre-existing (bootstrap, manifest)
    let mut created_in_batch: HashMap<PhysicalKey, usize> = HashMap::new();
    for pk in discover_all_pks(&*pager) {
        created_in_batch.insert(pk, 0);
    }

    let rows_per_batch = C3_ROWS.div_ceil(BATCHES);

    println!(
        "Ingesting col1={} rows, col2={} rows, col3={} rows in {} batches \
         ({} rows/batch on col3)...",
        C1_ROWS, C2_ROWS, C3_ROWS, BATCHES, rows_per_batch
    );

    let t_total = Instant::now();
    for b in 0..BATCHES {
        let start = b * rows_per_batch;
        let end = (start + rows_per_batch).min(C3_ROWS);

        // Build per-column puts for this window; some will be empty.
        let mut pairs: Vec<(LogicalFieldId, ArrayRef)> = Vec::new();
        if let Some(p) = build_put_for_col1(start, end) {
            pairs.push(p);
        }
        if let Some(p) = build_put_for_col2(start, end) {
            pairs.push(p);
        }
        if let Some(p) = build_put_for_col3(start, end) {
            pairs.push(p);
        }
        if pairs.is_empty() {
            continue;
        }

        // Group columns by equal row count; make one RecordBatch per group.
        let mut by_len: HashMap<usize, Vec<(LogicalFieldId, ArrayRef)>> = HashMap::new();
        for (fid, arr) in pairs {
            by_len.entry(arr.len()).or_default().push((fid, arr));
        }

        let t_batch = Instant::now();
        for (_len, group) in by_len {
            let batch = batch_from_pairs(&group);
            store.append(&batch).unwrap();
        }
        let dt = t_batch.elapsed();
        println!("  batch {}: rows [{}..{}) in {:?}", b + 1, start, end, dt);

        // mark new physical keys as created in this batch (b+1)
        for pk in discover_all_pks(&*pager) {
            created_in_batch.entry(pk).or_insert(b + 1);
        }
    }
    println!("Total ingest time: {:?}", t_total.elapsed());

    // Simple probes by counting rows per column via scan()
    for id in [1u32, 2, 3] {
        let field_id = fid(id);
        let mut rows = 0usize;
        if let Ok(it) = store.scan(field_id) {
            for arr in it.flatten() {
                rows += arr.len();
            }
        }
        println!("col={:?} -> total rows scanned: {}", field_id, rows);
    }

    // ASCII summary of final layout (rough byte sizes per pk)
    let ascii = {
        use llkv_column_map::storage::pager::BatchGet;
        use std::fmt::Write;
        let mut s = String::new();
        let all_pks = discover_all_pks(&*pager);
        let gets: Vec<BatchGet> = all_pks.iter().map(|&k| BatchGet::Raw { key: k }).collect();
        let res = pager.batch_get(&gets).unwrap();
        writeln!(&mut s, "==== STORAGE ASCII ====").unwrap();
        for r in res {
            match r {
                llkv_column_map::storage::pager::GetResult::Raw { key, bytes, .. } => {
                    writeln!(&mut s, "pk={} bytes={}", key, bytes.as_ref().len()).unwrap();
                }
                llkv_column_map::storage::pager::GetResult::Missing { key } => {
                    writeln!(&mut s, "pk={} <missing>", key).unwrap();
                }
            }
        }
        s
    };
    println!("{}", ascii);

    // ONE final DOT with batch-colored nodes
    let dot = render_one_colored_dot(&*pager, &created_in_batch);
    std::fs::write("storage_layout.dot", dot)?;
    println!("Wrote storage_layout.dot (single graph, nodes colored by batch)");

    Ok(())
}
