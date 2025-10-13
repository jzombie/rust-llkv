# LLKV Table

**Work in Progress**

Columnar table using the [LLKV](https://github.com/jzombie/rust-llkv) toolkit.

This crate is designed to work directly with Arrow `RecordBatch` and does not provide any additional abstraction over the batch data model beyond how batches are queried and streamed. Data is fed into tables and retrieved from tables as batches of `RecordBatch`.

## Purpose

- Provide schema-aware table abstraction built on [`llkv-column-map`](../llkv-column-map/) columnar storage.
- Manage table metadata via the system catalog (table 0).
- Handle MVCC columns (`created_by`, `deleted_by`) for transaction support.
- Support table scans with projection, filtering, and ordering.

## Design Notes

- Tables use [`llkv-column-map`](../llkv-column-map/) for physical storage but add schema validation and field ID tracking.
- Integrates with [`llkv-transaction`](../llkv-transaction/) for row-level visibility filtering.
- Used by [`llkv-runtime`](../llkv-runtime/) for executing SQL operations and by [`llkv-executor`](../llkv-executor/) for SELECT queries.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
