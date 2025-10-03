# LLKV Result

**Work in Progress**

`llkv-result` provides lightweight result and error types for the [LLKV](https://github.com/jzombie/rust-llkv) toolkit.

## Purpose

- Centralize common error/result types so other crates can share a consistent error model.
- Keep the crate intentionally small and focused on types and conversions.

## Design Notes

- The crate exposes thin wrappers and `thiserror`-based error enums to simplify propagation and conversion between layers.
- Consumers should treat this crate as a small utility dependency; it is not intended to hold business logic.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
