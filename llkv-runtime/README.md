# LLKV Runtime

[![made-with-rust][rust-logo]][rust-src-page]
[![rust-docs][rust-docs-badge]][rust-docs-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

`llkv-runtime` coordinates the end-to-end execution of SQL statements for the [LLKV](../) stack. It bridges plans, storage, and MVCC so the higher-level SQL engine can remain stateless.

## Responsibilities

- Execute all statement types (DDL, DML, transaction control, and queries) using the lower-layer crates.
- Inject and maintain MVCC columns (`row_id`, `created_by`, `deleted_by`) across every mutation.
- Manage session lifecycle, namespaces, and transaction snapshots for both auto-commit and explicit transactions.
- Orchestrate plan evaluation by invoking [`llkv-executor`](../llkv-executor/) for streaming `SELECT` workloads while handling side effects elsewhere.

## Transaction Model

- Built on [`llkv-transaction`](../llkv-transaction/) to allocate monotonic transaction IDs and track commit watermarks.
- Sessions capture a `TransactionSnapshot` at `BEGIN`; visibility rules follow MVCC snapshot isolation.
- Auto-commit statements run with `TXN_ID_AUTO_COMMIT = 1`, observing the latest committed snapshot without staging.
- Explicit transactions maintain a durable catalog snapshot and replay staging operations on commit after conflict checks.

## Dual-Context Execution

- Each active transaction holds two runtime contexts:
	- **Persistent context (`RuntimeContext<BoxedPager>`)**: Operates on existing tables directly, tagging all writes with MVCC metadata.
	- **Staging context (`RuntimeContext<MemPager>`)**: Buffers tables created inside the transaction so they remain isolated until commit.
- On commit the runtime replays staged operations into the persistent context after [`TxnIdManager`] confirms the commit and advances the global watermark.
- Rollback drops the staging context and clears tracked operations, so uncommitted objects never leak.

## Statement Dispatch

- The runtime routes statements produced by [`llkv-plan`](../llkv-plan/) into subsystem-specific executors:
	- `SELECT` plans stream through [`llkv-executor`](../llkv-executor/), returning Arrow `RecordBatch`es.
	- Mutations append Arrow batches via [`llkv-table`](../llkv-table/) which delegates to [`llkv-column-map`](../llkv-column-map/).
	- Catalog updates leverage the system catalog (`SysCatalog`, table 0) to persist schema metadata alongside user data.
- Result reporting uses enums in [`llkv-result`](../llkv-result/) so callers receive structured responses (`Select`, `Insert`, `CreateTable`, etc.).

## Integration Points

- Consumed by [`llkv-sql`](../llkv-sql/) to power `SqlEngine` APIs.
- Supplies the session handle returned by `SqlEngine::session()` for advanced transaction control.
- Provides helper hooks used by the SLT harness to expose detailed query statistics when `LLKV_SLT_STATS=1` is set.

## License

Licensed under the [Apache-2.0 License](../LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[rust-docs-page]: https://docs.rs/llkv-runtime
[rust-docs-badge]: https://img.shields.io/docsrs/llkv-runtime

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
