# LLKV: Logical Columnar Storage for Binary Key-Value Storage Systems

**Work in Progress**

LLKV is a lightweight toolkit for building logical, columnar layouts and query primitives on top of generic key–value stores.

It's designed for storage engines that support zero-copy reads (e.g. [SIMD R Drive](https://crates.io/crates/simd-r-drive)) and aims to provide columnar storage with expression-based querying.

## License

Licensed under the [Apache-2.0 License](./LICENSE).
