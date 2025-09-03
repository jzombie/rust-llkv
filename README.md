# LLKV: Logical Columnar Storage for Binary Key-Value Storage Systems

**Work in Progress**

A small toolkit for building logical columnar layouts on top of a byte-ordered key-value store (B+Tree, pager, codecs).

Designed to provide columnar storage and expression-based querying to libraries like [SIMD R Drive](https://crates.io/crates/simd-r-drive) and other key-value stores optimized for zero-copy access to binary data.

## License

Licensed under the [Apache-2.0 License](./LICENSE).
