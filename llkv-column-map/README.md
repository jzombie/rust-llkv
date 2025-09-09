# LLKV Column Map

**Work in Progress**

Prototype B+Tree replacement with batch-only pager I/O for the [LLKV](https://github.com/jzombie/rust-llkv) toolkit.

## Features

- Physical/Logical key separation (batched writes of many logical keys can consume far less physical keys than writing directly to storage keys).
- Logical keys namespaced per field.
- Supports variable and fixed width keys.
- Supports variable and fixed width values.
- Logical key and value segment pruning.
- Scanning iterator which can iterate over key or value without maintaining a separate reverse index.
- `Graphviz` (`.dot`) visualization generation (see [examples/visualize.rs](examples/visualize.rs)) for illustrative purposes and debugging.

## Testing

Some large (expensive) tests are marked `#[ignore]` by default.

### Quick Start

```sh
cargo test
```

### Run everything (including ignored)

```sh
cargo test -- --include-ignored
```

## License

Licensed under the [Apache-2.0 License](../LICENSE).
