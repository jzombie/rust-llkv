# LLKV Result

**Work in Progress**

`llkv-result` provides lightweight result and error types and serves as the base set of result types for the [LLKV](https://github.com/jzombie/rust-llkv) toolkit.

## Purpose

- Provide a common foundation of result and error types that other crates in the LLKV
	workspace should build on or convert to when practical. This crate is intended to be
	the shared base rather than the sole consumer of all result types.
- Keep the crate intentionally small and focused on types and conversions so it remains
	easy to depend on from multiple other crates.

## Design Notes

- The crate exposes thin wrappers and [thiserror](https://crates.io/crates/thiserror)-based error enums to simplify propagation and conversion between layers.
- Other crates should prefer to reuse these base types where it makes sense, but there are
	valid exceptions. Some subsystems (for example, [llkv-runtime](../llkv-runtime/)) may define more explicit
	or specialized result types when the cost of conversion or the need for specialized
	behavior justifies it. In those cases, prefer conversions to the shared base types for
	public APIs.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
