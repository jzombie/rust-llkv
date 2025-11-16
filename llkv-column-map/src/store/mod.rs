//! ColumnStore facade and supporting modules.
//!
//! This module groups the high-level storage concepts exposed by the column map
//! crate: configuration, catalog management, scans, projections, and indexing.
//! Each submodule provides a focused capability while re-exporting the primary
//! entry points (`ColumnStore`, `ScanBuilder`, projection helpers, etc.). The
//! combined surface is what keeps the SQLite and DuckDB `sqllogictest` suites
//! happy—descriptor iterators stream chunk metadata under tight memory budgets
//! while projection and filtering visitors preserve dialect semantics.
//!
//! Updates here should keep the rustdoc narrative aligned with the
//! [workspace README](../../README.md) and accompanying DeepWiki article so the
//! story remains coherent for newcomers evaluating the project.
//!
//! # Layout
//! - [`core`]: ColumnStore implementation, ingestion, and metadata handling
//! - [`catalog`], [`descriptor`], [`layout`]: Persistent metadata structures
//! - [`scan`], projection helpers, [`indexing`]: Execution helpers for readers and indexes
//! - [`rowid`], dtype cache utilities: Supporting utilities for ID management and type lookup
//!
//! Consumers should import from this module rather than the individual files so
//! rustdoc presents a coherent surface.

mod config;
use config::*;

mod constants;
pub use constants::CREATED_BY_COLUMN_NAME;
pub use constants::DELETED_BY_COLUMN_NAME;
pub use constants::FIELD_ID_META_KEY;
pub use constants::ROW_ID_COLUMN_NAME;
use constants::*;

pub mod core;
pub use core::*;

pub mod catalog;
pub mod debug;
pub mod descriptor;

mod ingest;

pub mod layout;
pub use layout::*;

pub mod rowid;
pub use rowid::*;

pub mod scan;
pub use scan::*;

mod projection;
pub use projection::{GatherNullPolicy, MultiGatherContext, Projection};

mod slicing;
use slicing::*;

mod dtype_cache;
use dtype_cache::DTypeCache;

pub mod indexing;
pub use indexing::*;

mod write_hints;
pub use write_hints::ColumnStoreWriteHints;
