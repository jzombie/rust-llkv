# LLKV Join

**Work in Progress**

`llkv-join` implements relational join algorithms for the [LLKV](https://github.com/jzombie/rust-llkv) toolkit.

## Purpose

- Provide streaming join operations over `llkv_table::Table` instances.
- Offer a growable set of join algorithms.
- Expose ergonomic APIs for streaming results via RecordBatches to callers.

## Design Notes

- The crate focuses on correctness and pragmatic performance: a hash-join implementation is provided with specialized fast-paths for single-column primitive joins (e.g., i32/i64/u32/u64).
- Public APIs stream `RecordBatch` results through a callback so callers can process results without allocating large intermediate buffers.
- Implementation uses Arrow `RecordBatch`/`ArrayRef` and integrates with `llkv_table` scanning primitives.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
