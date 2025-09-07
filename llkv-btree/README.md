# LLKV B+Tree

**Note: This B+Tree implementation being replaced by [llkv-column-map](../llkv-column-map/).**

Generic, paged B+Tree with batch-only pager I/O for the [LLKV](https://github.com/jzombie/rust-llkv) toolset.

## Features

- Pluggable storage
- Zero-copy reads: `ValueRef` points directly into `Pager` pages; no value memcpy in the tree. Works best with `Arc<Mmap>` pages; iterator holds one leaf page at a time for streaming.
- Streaming-capable. Entire tree may be much larger than RAM.
- Physical/Logical key separation
- `Graphviz` (`.dot`) visualization generation (see [examples/visualize.rs](examples/visualize.rs)). Integration tests also use this functionality for snapshot testing.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
