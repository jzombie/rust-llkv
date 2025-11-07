# LLKV Result

[![made-with-rust][rust-logo]][rust-src-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

`llkv-result` provides lightweight result and error types and serves as the base set of result types for the [LLKV](../) toolkit.

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

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
