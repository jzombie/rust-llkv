# LLKV SQL

[![made-with-rust][rust-logo]][rust-src-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

`llkv-sql` exposes `SqlEngine`, the primary entry point for executing SQL against the [LLKV](../) stack. It converts SQL text into typed plans, handles dialect quirks, batches insert workloads, and delegates execution to the runtime layer.

## Responsibilities

- Parse SQL statements, translating source text to `sqlparser` ASTs and then to [`llkv-plan`](../llkv-plan/) structures.
- Provide user-facing APIs for ad-hoc queries (`sql`) and multi-statement execution (`execute`).
- Manage SQL preprocessing so SQLite/DuckDB conveniences map cleanly onto `sqlparser` and the planner.
- Coordinate with [`llkv-runtime`](../llkv-runtime/) for transaction control, MVCC visibility, and plan execution.

## SqlEngine Highlights

- `SqlEngine::new` wires a pager-backed runtime; `SqlEngine::with_context` reuses an existing runtime session when embedding the engine.
- `execute(&self, sql: &str)` accepts batches of statements, runs the preprocessing pipeline, parses each statement, manages the `InsertBuffer`, and returns `Vec<SqlStatementResult>`.
- `sql(&self, sql: &str)` enforces single `SELECT` semantics and yields `Vec<RecordBatch>` produced by the executor.
- `session(&self)` surfaces the underlying [`llkv-runtime::RuntimeSession`] for advanced transaction control.

## SQL Preprocessing Pipeline

The engine normalizes dialect-specific syntax before parsing:

- Converts `CREATE TYPE` to `CREATE DOMAIN`, rewrites SQLite trigger shorthands, and patches `REINDEX` statements.
- Expands SQLite `expr IN tablename` shorthands into `SELECT` subqueries so the planner sees a consistent AST.
- Strips SQLite `INDEXED BY`/`NOT INDEXED` hints and removes trailing commas in `VALUES` clauses for DuckDB compatibility.
- Handles `EXCLUDE` qualified names and other minor dialect mismatches to keep the rest of the stack parser-agnostic.

## INSERT Buffering

- Disabled by default; enable with `set_insert_buffering(true)` when bulk-loading data.
- Buffers consecutive `INSERT ... VALUES` statements targeting the same table, columns, and conflict action to reduce planning churn.
- Flush triggers include buffer size limits (8,192 rows), cross-table inserts, non-`INSERT` statements, explicit transaction boundaries, expectation hints, or manual `flush_pending_inserts()` calls.
- The buffered rows coalesce into one `InsertPlan`, improving batch ingestion throughput without bypassing MVCC.

## Transaction Support

- Standard `BEGIN`, `COMMIT`, and `ROLLBACK` statements wire through the runtime to allocate or finalize MVCC snapshots.
- Auto-commit mode executes each statement in its own transaction when no explicit transaction is active.
- Result variants map to [`llkv-runtime`](../llkv-runtime/) statement outcomes, including `Select`, `Insert`, `Update`, `Delete`, and catalog DDL responses.

## Testing Hooks

- The SLT harness ([`llkv-slt-tester`](../llkv-slt-tester/)) uses `SqlEngine` via an `EngineHarness` that enables insert buffering and translates results to sqllogictest expectations.
- Query-duration metrics (`LLKV_SLT_STATS=1`) surface per-statement timing when running sqllogictest workloads.

## License

Licensed under the [Apache-2.0 License](../LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
