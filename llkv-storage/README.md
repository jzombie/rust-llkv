# LLKV Storage Adapters

**Work in Progress**

General-purpose storage adapters for the [LLKV](../) toolkit.

## Purpose

- Provide the `Pager` trait abstraction for persistent storage backends.
- Support batch read/write operations for efficient I/O.
- Enable pluggable storage implementations (memory, file-based, etc.).

## Design Notes

- The `Pager` trait is used by [`llkv-column-map`](../llkv-column-map/) for physical data persistence.
- Supports both in-memory (`MemPager`) and persistent storage backends.
- All higher-level crates ([`llkv-table`](../llkv-table/), [`llkv-runtime`](../llkv-runtime/)) are pager-agnostic.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
