# LLKV Column Map

**Work in Progress**

This crate provides low-level columnar mapping for the [LLKV](../) toolkit.

It's used by [`llkv-table`](../llkv-table/) as an interface into the lower-level [`llkv-storage`](../llkv-storage/) pagers.

## Key features

- Low-level columnar mapping used by `llkv-table` and backed by `llkv-storage` pagers.
- Efficient physical/logical key separation for batched writes and compact storage layout.
- Scanning and gather APIs with pagination and configurable null-handling for column-oriented consumers.
- Parallel scan/filter paths with Rayon where parallelism is capped by `LLKV_MAX_THREADS`.

## Technical details

- Physical/logical key separation: batched writes of many logical keys can consume far fewer physical keys than writing directly to storage keys.
- Logical keys are namespaced per field.
- Physical keys used by the pager are 64-bit integers (`u64`) — keys are numeric rather than arbitrary variable-width bytes.
- Values may be variable- or fixed-width depending on the chosen encoding.
- Logical key and value segment pruning.
- Scanning iterator can iterate over keys or values without maintaining a separate reverse index.
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
