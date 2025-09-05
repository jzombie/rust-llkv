//! This example demonstrates how to generate a moderately large columnar layout
//! using the crate's high-level API (ColumnStore), and then write its structure
//! to a `.dot` file for visualization.
//!
//! --- How to View the Output ---
//!
//! Online: https://dreampuf.github.io/GraphvizOnline/?engine=dot
//!
//! or:
//!
//! 1. Install Graphviz
//!    - macOS (Homebrew):      `brew install graphviz`
//!    - Ubuntu/Debian:         `sudo apt-get update && sudo apt-get install graphviz`
//!    - Windows (Chocolatey):  `choco install graphviz`
//!      or (Scoop):            `scoop install graphviz`
//!
//! 2. Run this example
//!    `cargo run --example visualize`
//!
//!    This will create `storage_layout.dot` in the project root and print an
//!    ASCII summary to stdout.
//!
//! 3. Generate an image
//!    - PNG:  `dot -Tpng storage_layout.dot -o storage_layout.png`
//!    - SVG:  `dot -Tsvg storage_layout.dot -o storage_layout.svg`
//!
//! 4. Open the PNG/SVG (or paste the DOT text into the online viewer).

use bitcode::{Decode, Encode};
use std::collections::HashMap;

use llkv_column_map::pager::Pager;
use llkv_column_map::types::PhysicalKey;
use llkv_column_map::{AppendOptions, ColumnStore, Put, ValueMode};

// ---------------- Tiny in-memory pager for the example ----------------

#[derive(Default)]
struct MemPager {
    map: HashMap<PhysicalKey, Vec<u8>>,
    next: PhysicalKey,
}

impl Pager for MemPager {
    fn alloc_many(&mut self, n: usize) -> Vec<PhysicalKey> {
        let start = if self.next == 0 { 1 } else { self.next }; // keep 0 for bootstrap
        self.next = start + n as u64;
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

// ---------------- Main: use the high-level API end-to-end ---------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pager = MemPager::default();
    let mut store = ColumnStore::init_empty(&mut pager);

    // Column 1: fixed-width values (auto-detected)
    let mut items_c1 = Vec::new();
    for i in 0..500u32 {
        let key = format!("id:{:06}", i).into_bytes();
        let val = (i.to_le_bytes()).to_vec(); // width=4
        items_c1.push((key, val));
    }

    // Column 2: variable-length values (auto-detected)
    let mut items_c2 = Vec::new();
    for i in 0..500u32 {
        let key = format!("id:{:06}", i).into_bytes();
        let len = (i % 21 + 1) as usize; // 1..21 bytes
        items_c2.push((key, vec![b'A' + (i % 26) as u8; len]));
    }

    // Column 3: lots of entries to force multiple segments by thresholds
    let mut items_c3 = Vec::new();
    for i in 0..5_000u32 {
        let key = format!("k{:06}", i).into_bytes();
        let val = vec![0x55; 8]; // fixed 8 bytes
        items_c3.push((key, val));
    }

    // One call writes all three columns. Options keep segments small for clarity.
    store.append_many(
        vec![
            Put {
                field_id: 1,
                items: items_c1,
            },
            Put {
                field_id: 2,
                items: items_c2,
            },
            Put {
                field_id: 3,
                items: items_c3,
            },
        ],
        AppendOptions {
            mode: ValueMode::Auto,
            segment_max_entries: 512, // keep segments modest
            segment_max_bytes: 64 * 1024,
            last_write_wins_in_batch: true,
        },
    );

    // Query a few keys back (point lookups)
    let got1 = store.get_in_column(1, vec![b"id:000010".to_vec(), b"id:009999".to_vec()]);
    println!("col=1 id:000010 -> {:?}", got1[0].as_deref());
    println!("col=1 id:009999 -> {:?}", got1[1].as_deref()); // likely None

    let got2 = store.get_in_column(2, vec![b"id:000000".to_vec(), b"id:000100".to_vec()]);
    println!(
        "col=2 id:000000 len={:?}",
        got2[0].as_ref().map(|v| v.len())
    );
    println!(
        "col=2 id:000100 len={:?}",
        got2[1].as_ref().map(|v| v.len())
    );

    let got3 = store.get_in_column(3, vec![b"k000100".to_vec(), b"k004999".to_vec()]);
    println!(
        "col=3 k000100 -> len={:?}",
        got3[0].as_ref().map(|v| v.len())
    );
    println!(
        "col=3 k004999 -> len={:?}",
        got3[1].as_ref().map(|v| v.len())
    );

    // Print the ASCII storage summary
    let ascii = store.render_storage_ascii();
    println!("\n==== STORAGE ASCII ====\n{}", ascii);

    // Write a DOT file
    let dot = store.render_storage_dot();
    std::fs::write("storage_layout.dot", dot)?;
    println!("Wrote storage_layout.dot");

    Ok(())
}
