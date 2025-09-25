# LLKV: A Columnar Storage Layer for Key-Value Stores

**Work in Progress**

`LLKV` is a lightweight toolkit for building logical, columnar layouts and query primitives on top of generic key–value stores.

It is designed for data stores that provide zero-copy reads with SIMD-friendly alignment (vector-width or cacheline aligned), enabling direct use in vectorized kernels (e.g. [SIMD R Drive](https://crates.io/crates/simd-r-drive)).

## Storage Layout: Physical vs. Logical Keys

In `LLKV`, data is split into physical keys and logical fields.

- A physical key maps to a chunk of one column (a slice of rows).
- A lookup returns many rows of a single column in one go, using the [Apache Arrow](https://arrow.apache.org/) memory layout for efficient, zero-copy access.
- Logical field IDs describe the higher-level schema, while physical keys handle the actual storage.

As data grows, a column can be split across multiple physical keys, but the main pattern stays simple: one key, one column chunk, many rows inside. This keeps lookups fast while preserving a clean, Arrow-based columnar layout on top of the key-value store.

## License

Licensed under the [Apache-2.0 License](./LICENSE).
