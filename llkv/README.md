# LLKV: Arrow-Native SQL over Key-Value Storage

[![made-with-rust][rust-logo]][rust-src-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

`llkv` is the primary entrypoint crate for the LLKV database toolkit. It provides both a command-line interface and a library that re-exports high-level APIs from the underlying workspace crates.

Arrow column chunks are stored under pager-managed physical keys, so pagers that already offer zero-copy reads—like the SIMD-backed [`simd-r-drive`](https://crates.io/crates/simd-r-drive)—can return contiguous buffers that stay SIMD-friendly.

## Design Notes

- Execution stays synchronous by default. Query workloads rely on hot-path parallelism from [Rayon](https://crates.io/crates/rayon) and [Crossbeam](https://crates.io/crates/crossbeam) instead of a global async runtime so scheduler costs stay predictable, while still embedding cleanly inside Tokio—the SLT harness uses Tokio to stage concurrent connection simulations.
- Storage is built atop the [simd-r-drive](https://crates.io/crates/simd-r-drive) project rather than Parquet, prioritizing fast point updates over integration with existing data lake tooling.
- The crate shares Apache DataFusion's SQL parser and Arrow memory model but intentionally avoids Tokio. The goal is to explore how far a DataFusion-like stack can go without a task scheduler while still shipping MVCC transactions and [sqllogictest](https://sqlite.org/sqllogictest/doc/trunk/about.wiki) coverage.
- LLKV remains alpha software. DataFusion offers a broader production footprint today, while this project trails new ideas for specialized workloads.

## Command-Line Interface

The `llkv` binary supports three modes:

### Interactive REPL

Start an interactive session with an in-memory database:

```bash
cargo run -p llkv
```

Available commands:
- `.help` — Show usage information
- `.open FILE` — Open a persistent database file (planned)
- `.exit` or `.quit` — Exit the REPL
- SQL statements are executed directly

### Stream Processing

Pipe SQL scripts to stdin for batch execution:

```bash
echo "SELECT 42 AS answer" | cargo run -p llkv
```

### SQL Logic Test Runner

Execute SLT test files or directories:

```bash
# Run a single test
cargo run -p llkv -- --slt tests/slt/example.slt

# Run a test directory
cargo run -p llkv -- --slt tests/slt/

# Use multi-threaded runtime
cargo run -p llkv -- --slt tests/slt/ --slt-runtime multi
```

## Library Usage

The `llkv` crate re-exports the core SQL engine and storage abstractions for programmatic use.

### Quick Start

```rust
use std::sync::Arc;
use llkv::{SqlEngine, storage::MemPager};

// Create an in-memory SQL engine
let engine = SqlEngine::new(Arc::new(MemPager::default()));

// Execute SQL statements
let results = engine.execute("CREATE TABLE users (id INT, name TEXT)").unwrap();
let results = engine.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')").unwrap();

// Query data
let batches = engine.sql("SELECT * FROM users WHERE id > 0").unwrap();
```

### Re-exported APIs

The following is a non-exhaustive list of some of the APIs this crate re-exports. Enable the `simd-r-drive-support` feature flag when you need durable storage via the SIMD R-Drive pager.

- **`SqlEngine`** — Main SQL execution engine from [`llkv-sql`](../llkv-sql/)
- **`storage::MemPager`** — In-memory pager for transient databases
- **`storage::Pager`** — Trait for custom storage backends
- **`storage::SimdRDrivePager`** — Durable pager backed by [`simd-r-drive`](https://crates.io/crates/simd-r-drive) (requires the `simd-r-drive-support` feature)
- **`Error` / `Result`** — Error handling types

### Using Persistent Storage

For persistent databases, enable the `simd-r-drive-support` feature and use a file-backed pager:

```toml
[dependencies]
llkv = { version = "0.8.0-alpha", features = ["simd-r-drive-support"] }
```

```rust
use std::sync::Arc;
use llkv::{SqlEngine, storage::SimdRDrivePager};

let pager = SimdRDrivePager::open("database.llkv").unwrap();
let engine = SqlEngine::new(Arc::new(pager));
```

## Architecture

LLKV is organized as a layered workspace with each crate focused on a specific responsibility:

- **SQL Interface** ([`llkv-sql`](../llkv-sql/)) — Parses SQL and manages the `SqlEngine` API
- **Planning** ([`llkv-plan`](../llkv-plan/), [`llkv-expr`](../llkv-expr/)) — Logical plans and expression ASTs
- **Runtime** ([`llkv-runtime`](../llkv-runtime/), [`llkv-transaction`](../llkv-transaction/)) — Transaction orchestration and MVCC
- **Execution** ([`llkv-executor`](../llkv-executor/), [`llkv-aggregate`](../llkv-aggregate/), [`llkv-join`](../llkv-join/)) — Query evaluation and streaming results
- **Storage** ([`llkv-table`](../llkv-table/), [`llkv-column-map`](../llkv-column-map/), [`llkv-storage`](../llkv-storage/)) — Columnar storage and pager abstractions

See the [workspace root README](../) and [DeepWiki documentation][deepwiki-page] for detailed architecture information.

## License

Licensed under the [Apache-2.0 License](../LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
