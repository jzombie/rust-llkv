# LLKV SQL

`llkv-sql` exposes `SqlEngine`, the primary entry point for executing SQL against the [LLKV](../) database toolkit. It converts SQL text into typed plans, handles dialect quirks, batches insert workloads, and delegates execution to the [runtime](../llkv-runtime/) layer.

This crate is not intended for direct standalone use.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
