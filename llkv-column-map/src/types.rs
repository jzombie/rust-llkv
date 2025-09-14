//! Core type definitions for the storage engine.

/// Opaque 64-bit address in the pager namespace.
/// Treated as an opaque handle by higher layers.
pub type PhysicalKey = u64;

/// Logical column/field identifier chosen by the application.
pub type LogicalFieldId = u64;

// Well-known key for the root ColumnCatalog.
pub(crate) const CATALOG_ROOT_PKEY: PhysicalKey = 0;
