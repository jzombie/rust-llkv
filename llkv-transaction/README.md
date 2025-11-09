# LLKV Transaction

[![made-with-rust][rust-logo]][rust-src-page]
[![rust-docs][rust-docs-badge]][rust-docs-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

`llkv-transaction` implements the MVCC substrate used across the [LLKV](../) stack. It allocates transaction IDs, enforces snapshot isolation, and persists commit metadata so row visibility stays deterministic.

## Responsibilities

- Allocate monotonic `TxnId`s and expose the current `last_committed` watermark.
- Capture `TransactionSnapshot`s for sessions so every statement runs against a consistent view.
- Track commit, abort, and rollback status for active transactions.
- Surface MVCC visibility rules to higher layers through helpers that evaluate `created_by` and `deleted_by` metadata.

## MVCC Metadata Columns

- Each table automatically manages three hidden columns stored by [`llkv-table`](../llkv-table/) and [`llkv-column-map`](../llkv-column-map/):
	- `row_id`: monotonically increasing identifier assigned during inserts.
	- `created_by`: `TxnId` that wrote the row.
	- `deleted_by`: `TxnId` that removed the row, or `TXN_ID_NONE` if the row is visible.
- Visibility is determined by comparing these columns against the transaction snapshot (`snapshot_id`) and the current transaction identifier.

## Snapshot Isolation Rules

A row is visible to transaction `T` with snapshot `S` when:

- `created_by <= S.last_committed`
- `deleted_by == TXN_ID_NONE` or `deleted_by > S.last_committed`
- `deleted_by != T.txn_id`

These rules implement snapshot isolation with optimistic conflict detection at commit.

## Transaction Lifecycle

- `begin_transaction` captures the current watermark, allocates a new `TxnId`, and freezes a catalog snapshot for schema stability.
- `commit_transaction` performs conflict detection (write/write conflicts, dropped tables, DDL races), replays staged operations, and persists the updated watermark.
- `rollback_transaction` discards staged operations and leaves persistent data untouched so uncommitted changes never leak.
- Auto-commit mode uses `TXN_ID_AUTO_COMMIT` to run single statements without replay.

## Integration

- Embedded by [`llkv-runtime`](../llkv-runtime/) for session control and MVCC metadata injection.
- Works with [`llkv-table`](../llkv-table/) scans to filter rows according to snapshot visibility.
- Supports the SQLogic Test harness and benchmarking flows by exposing stats hooks consumed through runtime APIs.

## License

Licensed under the [Apache-2.0 License](../LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[rust-docs-page]: https://docs.rs/llkv-transaction
[rust-docs-badge]: https://img.shields.io/docsrs/llkv-transaction

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
