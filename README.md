# LLKV: Arrow-Native SQL over Key-Value Storage

[![made-with-rust][rust-logo]][rust-src-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

LLKV is an experimental SQL database built as a Rust workspace. It layers Apache Arrow columnar storage, a streaming execution engine, and MVCC transaction management on top of generic key-value pagers.

Arrow arrays are persisted as column chunks addressed by pager-managed physical keys, so backends that already expose zero-copy reads—such as [`simd-r-drive`](https://crates.io/crates/simd-r-drive)—can hand out contiguous buffers for SIMD-friendly scans. Development focuses on correctness, layered modularity, and end-to-end Arrow interoperability.

## Goals and Status

- Provide a modular SQL stack with Arrow `RecordBatch` as the universal interchange format.
- Support transactional semantics via multi-version concurrency control (MVCC) with snapshot isolation.
- Keep each layer focused on a single responsibility so crates can evolve independently.
- Maintain a portable test harness with SQL Logic Tests and multi-OS CI coverage.
- Status: active WIP; core data layout, planner, runtime, and SLT harness are under construction.

## Using the Toolkit

- Navigate to [llkv/](./llkv/) for the workspace entrypoint. That crate houses the CLI binary and the high-level library surface.
- From the workspace root run `cargo run -p llkv` for the REPL, or `cargo run -p llkv -- --help` to see additional modes.
- See [llkv/README.md](./llkv/README.md) for installation flags, persistent pager setup, and API examples.

There's also a [`demos/`](./demos/) directory. Those projects are closer to publishable showcases than quick-start snippets, so they live alongside (and in addition to) crate-specific `examples/` trees.

## Design Tradeoffs

- Synchronous execution is the default. Hot paths lean on Rayon work-stealing and Crossbeam coordination instead of a pervasive async runtime so individual queries can keep scheduler overhead low, yet the engine still embeds cleanly inside Tokio—our SQL Logic Test runner spins up a Tokio runtime to simulate concurrent connections.
- Persistent storage backs onto the [SIMD R Drive](https://crates.io/crates/simd-r-drive) project rather than Parquet files. That keeps point updates fast without background compaction, but it does trade off the broader ecosystem tooling that Parquet enjoys.
- The project reuses the same SQL parser and Arrow memory model as Apache DataFusion while deliberately skipping Tokio. It grew out of an experiment to see how a DataFusion-style stack behaves without a task scheduler in the middle.
- Full SQL Logic Test coverage and MVCC transactions are core requirements, yet the crate remains alpha-quality. DataFusion is still the safer pick for production deployments with mature connectors and ecosystem support.

## Layered Architecture

The workspace is organized into six layers; higher layers depend on the ones below and communicate through Arrow data structures.

- **SQL Interface** (`llkv-sql`): Parses SQL, normalizes dialect quirks, batches `INSERT` workloads, and exposes `SqlEngine` as the main entry point.
- **Query Planning** (`llkv-plan`, `llkv-expr`): Translates parsed SQL into typed plans, models subqueries and scalar expressions, and keeps correlation plumbing shared.
- **Runtime and Orchestration** (`llkv-runtime`, `llkv-transaction`): Manages sessions, namespaces, MVCC snapshots, and coordinates plan execution across storage and execution layers.
- **Query Execution** (`llkv-executor`, `llkv-aggregate`, `llkv-join`): Streams Arrow batches through projection, filtering, aggregation, and join pipelines without buffering whole result sets.
- **Table and Metadata** (`llkv-table`, `llkv-column-map`): Adds schema-aware table APIs, system catalog management, and logical field tracking atop the column store.
- **Storage and I/O** (`llkv-storage`, `simd-r-drive`): Provides the `Pager` trait and concrete backends for zero-copy reads with SIMD-friendly alignment.

See [dev-docs/high-level-crate-linkage.md](./dev-docs/high-level-crate-linkage.md) and the [DeepWiki Architecture][deepwiki-page] page for dependency details.

## End-to-End Query Flow

1. `SqlEngine::execute` preprocesses SQL for SQLite and DuckDB dialect quirks, parses with `sqlparser`, and batches compatible `INSERT` statements.
2. Plans are built by `llkv-plan`, which annotates correlated subqueries, scalar programs, and DML metadata.
3. `llkv-runtime` acquires a transaction snapshot, injects MVCC metadata, and dispatches plans to the appropriate subsystem.
4. The executor layer materializes streaming Arrow `RecordBatch`es, invoking aggregation and join helpers as needed.
5. Results are returned straight to callers as Arrow batches; CTAS and INSERT workloads append Arrow data back through the table layer.

## Storage and MVCC Model

- Every table stores user columns alongside hidden `row_id`, `created_by`, and `deleted_by` metadata maintained by the runtime.
- Logical field IDs namespace user data, MVCC metadata, and row-id shadows so catalog lookups remain stable.
- `llkv-column-map` persists column chunks as Arrow-serialized blobs keyed by pager-managed physical IDs.
- The `ColumnStore::append` path sorts incoming `RecordBatch`es by `row_id`, rewrites conflicting rows with last-writer-wins semantics, and commits updates atomically via the pager.
- `llkv-transaction` allocates monotonic transaction IDs, tracks commits, and enforces snapshot visibility during scans and DML replay.
- Sessions use dual contexts: a persistent pager for existing tables and an in-memory pager for objects created within the active transaction that are replayed on commit.

## Testing, CI, and Benchmarks

- SQL Logic Tests: `llkv-slt-tester` wraps sqllogictest suites with `LlkvSltRunner`, pointer (`.slturl`) support, and optional query statistics (`LLKV_SLT_STATS=1`).
- CI: GitHub Actions workflows cover linting (`cargo fmt`, `clippy`, `cargo doc`, `cargo deny`, `cargo audit`) and run the full test matrix on Linux, macOS, and Windows.
- Benchmarks: Criterion benchmarks run on a self-hosted macOS ARM64 runner, with CodSpeed ingesting results for trend tracking.

## Developing Locally

- Build and test the workspace:
	- `cargo test --workspace --all-features --lib --bins --tests --examples -- --include-ignored`
	- Enable SLT stats with `LLKV_SLT_STATS=1` when running integration suites.
- Lint and docs:
	- `cargo fmt --all -- --check`
	- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
	- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items`
- Benchmark locally:
	- `cargo bench --workspace` (benchmarks are currently run via Criterion).
- Refer to [dev-docs](./dev-docs/) for more information.

## License

Licensed under the [Apache-2.0 License](./LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
