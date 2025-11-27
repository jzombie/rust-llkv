# LLKV Executor

`llkv-executor` evaluates `SELECT` plans for the [LLKV](../) database toolkit. It produces streaming Arrow `RecordBatch`es, coordinating with table scans, joins, and aggregation primitives while remaining oblivious to transaction orchestration.

This crate is not intended for direct standalone use.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
