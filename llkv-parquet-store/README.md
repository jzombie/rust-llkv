# LLKV Parquet Store

[![made-with-rust][rust-logo]][rust-src-page]

**Work in Progress**

`llkv-parquet-store` provides a Parquet-based columnar storage layer for LLKV. It replaces the custom `llkv-column-map` implementation by storing Parquet files as blobs within the pager abstraction.

## Design Goals

1. **Leverage Parquet**: Use the battle-tested Apache Parquet format instead of custom serialization
2. **Pager Integration**: Store Parquet files as blobs using the existing `llkv-storage` pager trait
3. **Table Management**: Maintain a catalog mapping table names to collections of Parquet files
4. **MVCC Support**: Store transaction visibility metadata (`created_by`, `deleted_by`) as Parquet columns
5. **Simplicity**: Reduce complexity compared to custom columnar storage (~80% less code)

## Architecture

The crate consists of:

- **`ParquetStore`**: Core API for reading/writing Parquet files through the pager
- **`ParquetCatalog`**: Persistent catalog mapping tables to Parquet file collections
- **MVCC Filtering**: Transaction visibility logic for scans
- **Compaction**: Merge-on-read strategy for updates and deletes

See `../dev-docs/PARQUET-STORE-DESIGN.md` for detailed architecture.

## Usage

```rust
use llkv_parquet_store::{ParquetStore, ParquetCatalog};
use llkv_storage::pager::MemPager;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

// Create store with in-memory pager
let pager = Arc::new(MemPager::new(1024 * 1024 * 256)); // 256MB
let store = ParquetStore::open(pager)?;

// Create a table
let table_id = store.create_table("users", schema)?;

// Append data
store.append(table_id, record_batch)?;

// Scan with MVCC filtering
let results = store.scan_visible(table_id, txn_id, None)?;
```

## Comparison to llkv-column-map

| Feature         | llkv-column-map         | llkv-parquet-store                 |
|-----------------|-------------------------|------------------------------------|
| Serialization   | Custom zero-copy format | Apache Parquet                     |
| Compression     | Basic                   | Dictionary, RLE, Snappy/Zstd       |
| Code Complexity | ~5000 LOC               | ~1000 LOC                          |
| Ecosystem       | Custom tools            | Standard Parquet tools             |
| Statistics      | Manual tracking         | Built into Parquet metadata        |
| Maintenance     | High                    | Low (delegates to `parquet` crate) |

## Status

- [x] Initial design document
- [ ] Basic ParquetStore implementation
- [ ] Catalog persistence
- [ ] MVCC filtering
- [ ] Compaction strategy
- [ ] Benchmark vs llkv-column-map
- [ ] Migration guide

## License

Licensed under the [Apache-2.0 License](../LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust
