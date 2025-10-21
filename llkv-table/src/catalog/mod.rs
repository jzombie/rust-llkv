//! Unified catalog module for table and schema management.
//!
//! This module provides a single interface for all catalog operations:
//! - `TableCatalog`: In-memory name resolution cache (table name ↔ id, field name ↔ id)
//! - `CatalogManager`: High-level service for creating/dropping tables and managing metadata
//!
//! External code should use this `catalog` module instead of accessing individual components.

mod manager;
mod table_catalog;

// Re-export TableCatalog and related types
pub use table_catalog::{
    ColumnResolution, FieldConstraints, FieldDefinition, FieldInfo, FieldResolver,
    FieldResolverState, FieldState, IdentifierContext, IdentifierResolver, QualifiedTableName,
    QualifiedTableNameRef, TableCatalog, TableCatalogSnapshot, TableCatalogState, TableState,
};

// Re-export internal types needed by other modules
pub(crate) use table_catalog::TableMetadataView;

// Re-export CatalogManager (formerly CatalogService)
pub use manager::{CatalogManager, CreateTableResult, MvccColumnBuilder};
