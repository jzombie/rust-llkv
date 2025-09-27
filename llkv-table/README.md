# LLKV Table

**Work in Progress**

Columnar table using the [LLKV](https://github.com/jzombie/rust-llkv) toolkit.

## Design Notes

- `scan_stream` emits column batches without performing row-level filtering. Null handling is delegated to the column-map gatherers so higher layers decide whether to preserve or drop null-only rows.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
