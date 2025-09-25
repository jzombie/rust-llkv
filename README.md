# LLKV: A Columnar Storage Layer for Key-Value Stores

**Work in Progress**

`LLKV` is a lightweight toolkit for building logical, columnar layouts and query primitives on top of generic key–value stores.

It is designed for data stores that provide zero-copy reads with SIMD-friendly alignment (vector-width or cacheline aligned), enabling direct use in vectorized kernels (e.g. [SIMD R Drive](https://crates.io/crates/simd-r-drive)).

## Storage Layout: Physical / Logical Key Separation

In `LLKV`, a physical data store key typically maps to a segment of a single column (one logical field) that spans a contiguous range of row IDs. One lookup returns a compact, columnar slice: many rows of one column. This reduces round trips, improves cache locality, and speeds up scans.

Above storage, data is addressed by logical field IDs and row-ID ranges for cataloging, planning, and predicate pushdown. The on-disk layout can split a column into multiple physical keys as data grows or is reorganized. The common path stays simple and efficient: one column segment per key, many rows inside. This separation keeps lookups cheap while preserving clear columnar packing on top of a key-value store.

## License

Licensed under the [Apache-2.0 License](./LICENSE).
