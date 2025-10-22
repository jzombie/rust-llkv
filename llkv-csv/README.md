# LLKV CSV

**Work in Progress**

`llkv-csv` is a small frontend and ingestion helper for the [LLKV](../) toolkit.

## Purpose

- Provides CSV materialization and ingest paths that translate CSV files into Arrow `RecordBatch` (with an injected `row_id` column) and append them into [`llkv-table`](../llkv-table/) instances.
- Treats CSV parsing and ingest as a client-facing convenience layer — the canonical storage and query model remains Arrow `RecordBatch` in [`llkv-table`](../llkv-table/).

## Design Notes

- This crate is intentionally lightweight: it focuses on reading CSV files, optionally synthesizing `row_id` columns, and using [`llkv-table`](../llkv-table/) APIs to persist data. It does not attempt to re-model the batch-oriented storage semantics of the table layer.
- In other words, `llkv-csv` is a frontend to [`llkv-table`](../llkv-table/): data flows in as `RecordBatch` and is appended to tables; queries return `RecordBatch`.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
