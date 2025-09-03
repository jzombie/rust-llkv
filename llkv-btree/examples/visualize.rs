//! This example demonstrates how to generate a moderately large B+Tree,
//! and then write its structure to a `.dot` file for visualization.
//!
//! --- How to View the Output ---
//!
//! Online: https://dreampuf.github.io/GraphvizOnline/?engine=dot
//!
//! or:
//!
//! 1.  **Install Graphviz:**
//!     Graphviz is a graph visualization software. You'll need to install it
//!     on your system.
//!
//!     -   **macOS (using Homebrew):**
//!         `brew install graphviz`
//!
//!     -   **Ubuntu/Debian:**
//!         `sudo apt-get update && sudo apt-get install graphviz`
//!
//!     -   **Windows (using Chocolatey or Scoop):**
//!         `choco install graphviz` or `scoop install graphviz`
//!
//! 2.  **Run this Example:**
//!     Execute this example from your terminal using cargo:
//!     `cargo run --example visualize`
//!
//!     This will create a file named `large_tree.dot` in the root of your project.
//!
//! 3.  **Generate the Image:**
//!     Open your terminal in the project root and run the `dot` command
//!     to convert the `.dot` file into a viewable image format like PNG or SVG.
//!
//!     -   **To create a PNG image:**
//!         `dot -Tpng large_tree.dot -o large_tree.png`
//!
//!     -   **To create an SVG image (better for zooming):**
//!         `dot -Tsvg large_tree.dot -o large_tree.svg`
//!
//! 4.  **View the File:**
//!     Open `large_tree.png` or `large_tree.svg` in any image viewer to see
//!     the visual structure of your B+Tree.

#[cfg(feature = "debug")]
use llkv_btree::{
    BPlusTree,
    codecs::{BigEndianIdCodec, BigEndianKeyCodec},
    pager::MemPager64,
};

#[cfg(feature = "debug")]
use llkv_btree::traits::GraphvizExt;

#[cfg(feature = "debug")]
use rand::{
    rngs::StdRng,
    {SeedableRng, seq::SliceRandom},
};

#[cfg(feature = "debug")]
use std::{fs::File, io::Write};

#[cfg(feature = "debug")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Set up the pager and create an empty tree.
    // We now use the standard `MemPager64` from the library.
    // A smaller page size will cause more splits and create a deeper tree.
    let pager = MemPager64::default();

    // The BPlusTree is now instantiated with the standard BigEndian codecs.
    let tree =
        BPlusTree::<_, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>>::create_empty(pager, None)?;

    // 2. Insert a large number of random keys to build the tree.
    let mut rng = StdRng::seed_from_u64(1337);
    let mut keys: Vec<u64> = (0..150).collect();
    keys.shuffle(&mut rng);

    println!("Inserting 150 random keys into the B+Tree...");
    let items_to_insert: Vec<_> = keys
        .iter()
        .map(|k| (*k, format!("value_{}", k).into_bytes()))
        .collect();
    let owned_items: Vec<_> = items_to_insert
        .iter()
        .map(|(k, v)| (*k, v.as_slice()))
        .collect();
    tree.insert_many(&owned_items)?;

    println!("Insertion complete.");

    // 3. Generate the DOT string representation of the tree.
    println!("Generating DOT file for visualization...");
    let dot_string = tree.to_dot()?;

    // 4. Write the DOT string to a file.
    let mut file = File::create("large_tree.dot")?;
    file.write_all(dot_string.as_bytes())?;

    println!("\nSuccess! ✨");
    println!("A file named 'large_tree.dot' has been created in your project directory.");
    println!("Follow the instructions at the top of 'examples/visualize.rs' to view it.");

    Ok(())
}

#[cfg(not(feature = "debug"))]
fn main() {
    use std::io::Write as _;

    let msg = "\n\
============================================================\n\
  This example requires the `debug` feature.\n\
------------------------------------------------------------\n\
  Try one of the following:\n\
    cargo run --example visualize --features debug\n\
    cargo build --examples --features debug\n\
============================================================\n\n";

    let _ = std::io::stderr().write_all(msg.as_bytes());
    std::process::exit(1);
}
