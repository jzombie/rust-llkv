# LLKV Table

[![made-with-rust][rust-logo]][rust-src-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

`llkv-table` provides the schema-aware table abstraction that sits between SQL plans and the column store. It accepts Arrow `RecordBatch`es, coordinates catalog metadata, and exposes streaming scan APIs used by the runtime and executor layers.

## Responsibilities

- Persist table definitions and column metadata through the system catalog (table `0`).
- Validate Arrow schemas during `CREATE TABLE` and append operations.
- Inject and maintain MVCC columns (`row_id`, `created_by`, `deleted_by`) alongside user data.
- Translate logical field requests into `ColumnStore` lookups and stream Arrow batches back to callers.

## ColumnStore Integration

- Tables wrap an `Arc<ColumnStore>` from [`llkv-column-map`](../llkv-column-map/); `ColumnStore::append` handles sorting by `row_id`, last-writer-wins rewrites, and pager commits.
- Logical fields are namespaced to segregate user data, MVCC bookkeeping, and row-id shadows.
- `Table::append` performs schema validation, prepares MVCC metadata, and forwards the batch to the column store for persistence.
- `Table::scan_stream` projects requested columns, applies predicate pushdown via `ColumnStream`, and yields fixed-size batches to avoid buffering entire results.

## Catalog and Metadata

- `SysCatalog` (table `0`) stores `TableMeta` and `ColMeta` entries that describe user tables.
- Metadata changes (create, alter, drop) flow through the runtime and land in the catalog via the same Arrow append path, preserving crash consistency.

## MVCC Visibility

- Scan operations cooperate with [`llkv-transaction`](../llkv-transaction/) to filter rows based on transaction snapshots.
- Hidden MVCC columns remain present in storage; higher layers decide whether to expose them to callers.

## Usage in the Stack

- [`llkv-runtime`](../llkv-runtime/) uses tables for all DML and DDL operations.
- [`llkv-executor`](../llkv-executor/) relies on `Table::scan_stream` during `SELECT` evaluation, including joins and aggregations.
- Bulk ingestion paths (e.g., INSERT buffering) ultimately append Arrow batches through this crate, ensuring durability and MVCC tagging remain centralized.

## License

Licensed under the [Apache-2.0 License](../LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
