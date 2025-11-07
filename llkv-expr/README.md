# LLKV Query Expression AST

[![made-with-rust][rust-logo]][rust-src-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

A buffer-level, allocator-friendly predicate AST over borrowed byte slices (`&[u8]`), built for the [LLKV](../) toolkit.

This crate intentionally ships without an evaluator or planner.

Expressions can be used with [llkv-table](../llkv-table/) for querying.

### Currently supports
- **Generic field IDs**: `Expr<'a, F>` and `Filter<'a, F>` work with any `F` (e.g., `u32`, `&'static str`).
- **Logical composition**: `And(Vec<Expr>)`, `Or(Vec<Expr>)`, `Not(Box<Expr>)`.
- **Predicates over bytes**:
  - Equality: `Equals(&[u8])`
  - Ranges with precise bounds: `Range { lower: Bound<&[u8]>, upper: Bound<&[u8]> }`
  - Comparisons: `GreaterThan`, `GreaterThanOrEquals`, `LessThan`, `LessThanOrEquals`
  - Sets & patterns: `In(&[&[u8]])`, `StartsWith`, `EndsWith`, `Contains`
- **Ergonomic builders**: `Expr::all_of`, `Expr::any_of`, `Expr::not`
- **Zero-copy friendly**: operands are borrowed; no implicit allocation or decoding is performed here.

## License

Licensed under the [Apache-2.0 License](../LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
