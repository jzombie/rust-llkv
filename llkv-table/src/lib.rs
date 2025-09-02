// src/lib.rs
//! Zero-alloc table core: buffer-based predicates, streaming sources,
//! and a minimal table orchestrator.
//!
//! Public API never exposes Vec<u8> or concrete row-id types.
//! Row-ids are &[u8] and sorted/unique within each Source.
//!
//! Modules:
//! - types: FieldId and row-id comparator type.
//! - expr:  Filter and Expr AST with borrowed payloads.
//! - source: streaming Source trait and set-combinators.
//! - btree: btree traits + CursorSource adapter (plug your llvm-btree).
//! - storage: ColumnStorage trait with per-cell callback.
//! - table: Table orchestrator consuming Sources and storage.
//! - index: Index trait returning streaming Sources via GATs.

#![forbid(unsafe_code)]

pub mod expr;
pub mod index;
pub mod source;
pub mod storage;
pub mod table;
pub mod types;

pub use expr::{Expr, Filter, Operator};
pub use index::Index;
pub use source::{Diff2, Intersect2, SliceSource, Source, Union2};
pub use storage::ColumnStorage;
pub use table::{Table, TableCfg};
pub use types::{FieldId, RowIdCmp};
