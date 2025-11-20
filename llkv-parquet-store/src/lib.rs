//! Parquet-based columnar storage for LLKV.
//!
//! This crate provides a storage layer that manages collections of Apache Parquet
//! files stored as blobs within the [`llkv-storage`] pager abstraction. It serves
//! as a simpler alternative to [`llkv-column-map`]'s custom serialization format.
//!
//! # Architecture
//!
//! The storage layer consists of:
//!
//! - [`ParquetStore`]: Core API for reading and writing Parquet files through the pager
//! - [`ParquetCatalog`]: Persistent catalog mapping table names to Parquet file collections
//! - MVCC support via transaction visibility columns (`created_by`, `deleted_by`)
//! - Compaction strategies for merge-on-read semantics
//! - Garbage collection to reclaim unreferenced blobs
//!
//! # Design Philosophy
//!
//! Rather than implementing custom columnar storage with hand-rolled serialization,
//! chunk management, and indexing, this crate leverages Apache Parquet's proven format.
//! Each Parquet file is written to an in-memory buffer and stored as a single blob
//! in the pager, allowing the pager to handle persistence, caching, and memory mapping.
//!
//! Benefits:
//! - **Simpler**: ~80% less code than custom implementation
//! - **Standard**: Uses widely-adopted Parquet format
//! - **Proven**: Leverages battle-tested `parquet` crate
//! - **Tooling**: Standard Parquet tools work directly
//! - **Optimized**: Dictionary encoding, RLE, compression built-in
//!
//! Trade-offs:
//! - **Less control**: Cannot fine-tune layout as precisely
//! - **Merge overhead**: Updates require compaction
//! - **Granularity**: Minimum unit is a Parquet file
//!
//! # Usage Example
//!
//! ```rust,no_run
//! use llkv_parquet_store::{ParquetStore, ParquetCatalog};
//! use llkv_storage::pager::MemPager;
//! use arrow::datatypes::{Schema, Field, DataType};
//! use std::sync::Arc;
//!
//! # fn main() -> llkv_result::Result<()> {
//! // Create store with in-memory pager
//! let pager = Arc::new(MemPager::new());
//! let store = ParquetStore::open(pager)?;
//!
//! // Define schema
//! let schema = Arc::new(Schema::new(vec![
//!     Field::new("id", DataType::UInt64, false),
//!     Field::new("name", DataType::Utf8, false),
//! ]));
//!
//! // Create table
//! let table_id = store.create_table("users", schema)?;
//!
//! // Append data (RecordBatch creation not shown)
//! // store.append(table_id, record_batch)?;
//!
//! // Scan with transaction visibility
//! // let results = store.scan_visible(table_id, txn_id, None)?;
//! # Ok(())
//! # }
//! ```
//!
//! See `../dev-docs/PARQUET-STORE-DESIGN.md` for detailed architecture and design decisions.

mod catalog;
mod compaction;
mod gc;
mod mvcc;
mod reader;
mod store;
mod types;
mod writer;

pub use catalog::{ParquetCatalog, ParquetFileRef, TableMetadata};
pub use compaction::CompactionStrategy;
pub use gc::{collect_reachable_keys, garbage_collect};
pub use mvcc::{add_mvcc_columns, apply_mvcc_filter, deduplicate_by_row_id};
pub use store::ParquetStore;
pub use types::{FileId, TableId};

// Re-export common types for convenience
pub use arrow::datatypes::SchemaRef;
pub use arrow::record_batch::RecordBatch;
