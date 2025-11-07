# LLKV Column Map

**Work in Progress**

`llkv-column-map` implements the `ColumnStore`, the Arrow-native columnar storage engine for the [LLKV](../) stack. It maps logical fields to pager-managed physical chunks, enabling efficient scans, appends, and MVCC bookkeeping.

## Responsibilities

- Persist Arrow `RecordBatch`es by chunking columns into pager-backed blobs.
- Maintain the column catalog that maps `LogicalFieldId` to the physical keys storing metadata and data pages.
- Provide gather, scan, and append APIs used by [`llkv-table`](../llkv-table/) and the executor layer.

## Logical vs Physical Keys

- `LogicalFieldId` encodes namespace (user data, row-id shadow, MVCC metadata), table ID, and field ID to avoid collisions across tables.
- Physical keys are `u64` identifiers allocated by the pager; catalog entries track which physical keys hold descriptors, data, and row-id segments.
- Namespaces keep MVCC columns (`TxnCreatedBy`, `TxnDeletedBy`) and user columns isolated while still sharing the same physical store.

## Append Pipeline

- `ColumnStore::append` validates `RecordBatch` schemas, ensures `row_id` ordering, and applies last-writer-wins rewrites when incoming data updates existing rows.
- New data and row-id chunks are staged and committed atomically through the pager, guaranteeing crash consistency.
- MVCC metadata columns are appended alongside user columns so visibility checks remain centralized in storage.

## Scan and Gather APIs

- `ColumnStream` supports projection, filtering, and pagination for streaming reads.
- Gather operations offer configurable null-handling policies (preserve, error, drop) to accommodate different executor strategies.
- Parallel scan paths use Rayon; concurrency is bounded by the `LLKV_MAX_THREADS` environment variable when present.

## Pager Integration

- Relies on the `Pager` trait from [`llkv-storage`](../llkv-storage/) for batch get/put operations.
- Works with both `MemPager` (in-memory) and persistent pagers such as `SimdRDrivePager`, which deliver zero-copy, SIMD-aligned reads.
- On open, the store loads its catalog from the pager root key or initializes an empty catalog if none exists.

## Tooling and Testing

- Provides a Graphviz export (`examples/visualize.rs`) for introspecting catalog state.
- Some large or integration-heavy tests are marked `#[ignore]`; run them with `cargo test -- --include-ignored` when needed.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
