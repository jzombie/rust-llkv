# LLKV Table

`llkv-table` provides the schema-aware table abstraction that sits between SQL plans and the [column store](../llkv-column-map/) in the [LLKV](../) database toolkit. It accepts Arrow `RecordBatch`es, coordinates catalog metadata, and exposes streaming scan APIs used by the [runtime](../llkv-runtime/) and [executor](../llkv-executor/) layers.

This crate is not intended for direct standalone use.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
