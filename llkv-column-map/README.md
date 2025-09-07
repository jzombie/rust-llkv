# LLKV Column Map

**Work in Progress**

Prototype B+Tree replacement with batch-only pager I/O for the [LLKV](https://github.com/jzombie/rust-llkv) toolset.

## Features

- Physical/Logical key separation (batched writes of many logical keys can consume far less physical keys than writing directly to storage keys).
- Logical keys namespaced per field.
- Supports variable and fixed width keys.
- Supports variable and fixed width values.
- Logical key and value segment pruning.
- `Graphviz` (`.dot`) visualization generation (see [examples/visualize.rs](examples/visualize.rs)) for illustrative purposes and debugging.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
