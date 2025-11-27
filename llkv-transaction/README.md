# LLKV Transaction

`llkv-transaction` implements the MVCC substrate used across the [LLKV](../) database toolkit. It allocates transaction IDs, enforces snapshot isolation, and persists commit metadata so row visibility stays deterministic.

This crate is not intended for direct standalone use.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
