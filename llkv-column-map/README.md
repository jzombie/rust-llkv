# LLKV Column Map

**Work in Progress**

This crate provides low-level columnar mapping for the [LLKV](../) toolkit.

It's used by [`llkv-table`](../llkv-table/) as an interface into the lower-level [`llkv-storage`](../llkv-storage/) pagers.

## Features

- Physical/Logical key separation (batched writes of many logical keys can consume far less physical keys than writing directly to storage keys).
- Logical keys namespaced per field.
- Supports variable and fixed width keys.
- Supports variable and fixed width values.
- Logical key and value segment pruning.
- Scanning iterator which can iterate over key or value without maintaining a separate reverse index.
- Supports scanning iterator pagination.
- Parallel scan/filter paths honor the `LLKV_MAX_THREADS` environment variable to cap Rayon worker utilization when needed.
- Configurable gather null-handling policies (preserve, error, or drop missing/null rows) for column-oriented consumers.
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
