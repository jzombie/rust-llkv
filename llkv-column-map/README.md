# LLKV Column Map

`llkv-column-map` implements the `ColumnStore`, the Arrow-native columnar storage engine for the [LLKV](../) stack. It maps logical fields to pager-managed physical chunks, enabling efficient scans, appends, and MVCC bookkeeping.

This crate is not intended for direct standalone use.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
