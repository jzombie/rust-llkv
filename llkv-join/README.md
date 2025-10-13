# LLKV Join

**Work in Progress**

`llkv-join` implements relational join algorithms for the [LLKV](https://github.com/jzombie/rust-llkv) toolkit.

## Purpose

- Provide streaming join operations over [`llkv-table`](../llkv-table/) instances.
- Offer a growable set of join algorithms.
- Expose ergonomic APIs for streaming results via RecordBatches to callers.

## Design Notes

- The crate focuses on correctness and pragmatic performance: a hash-join implementation is provided with specialized fast-paths for single-column primitive joins (e.g., i32/i64/u32/u64).
- Public APIs stream `RecordBatch` results through a callback so callers can process results without allocating large intermediate buffers.
- Implementation uses Arrow `RecordBatch`/`ArrayRef` and integrates with [`llkv-table`](../llkv-table/) scanning primitives.
- Used by [`llkv-executor`](../llkv-executor/) for executing JOIN operations in SELECT queries.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
