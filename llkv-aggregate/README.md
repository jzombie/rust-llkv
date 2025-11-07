# LLKV Aggregate

[![made-with-rust][rust-logo]][rust-src-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

`llkv-aggregate` provides aggregate computation functions for the [LLKV](../) toolkit.

## Purpose

- Implement standard SQL aggregate functions (SUM, COUNT, AVG, MIN, MAX, etc.).
- Provide efficient streaming aggregation over Arrow data structures.
- Support grouped aggregations for query execution via [`llkv-executor`](../llkv-executor/).

## Design Notes

- The crate integrates with Arrow's columnar format for efficient computation.
- Aggregates are designed to work with [`llkv-executor`](../llkv-executor/) for pipelined evaluation in SELECT queries.

## License

Licensed under the [Apache-2.0 License](../LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
