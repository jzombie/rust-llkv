# LLKV Storage Adapters

[![made-with-rust][rust-logo]][rust-src-page]
[![rust-docs][rust-docs-badge]][rust-docs-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

`llkv-storage` defines the pager abstraction that underpins persistence for the [LLKV](../) stack. It offers both in-memory and persistent implementations with zero-copy reads for Arrow workloads.

## Pager Trait

- Exposes batch `get`/`put` operations over `PhysicalKey`/`EntryHandle` pairs so callers can read and write blobs efficiently.
- Supports atomic multi-key updates, which the column store relies on when committing append transactions.
- Keeps higher layers agnostic to the backing medium (memory, file-backed, or custom stores).

## Implementations

- `MemPager`: lightweight, heap-backed pager used for tests, temporary namespaces, and staging contexts during transactions.
- `SimdRDrivePager`: wraps `simd_r_drive::DataStore`, enabling zero-copy reads with SIMD-aligned buffers backed by a persistent file.
- Additional pagers can be implemented by consumers if they satisfy the trait contract.

## Integration Points

- [`llkv-column-map`](../llkv-column-map/) uses pagers to persist column chunks, catalogs, and MVCC metadata.
- [`llkv-runtime`](../llkv-runtime/) instantiates both persistent and in-memory pagers to support dual-context transactions.
- Benchmark targets measure pager throughput as part of the Criterion suites consumed by CodSpeed.

## License

Licensed under the [Apache-2.0 License](../LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[rust-docs-page]: https://docs.rs/llkv-storage
[rust-docs-badge]: https://img.shields.io/docsrs/llkv-storage

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
