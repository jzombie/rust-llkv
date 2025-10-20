//! Unified catalog module for table and schema management.
//!
//! This module provides a single interface for all catalog operations:
//! - `TableCatalog`: In-memory name resolution cache (table name ↔ id, field name ↔ id)
//! - `CatalogManager`: High-level service for creating/dropping tables and managing metadata
//!
//! External code should use this `catalog` module instead of accessing individual components.

mod table_catalog;
mod manager;

// Re-export TableCatalog and related types
pub use table_catalog::{
    TableCatalog, TableCatalogSnapshot, TableCatalogState, TableState,
    FieldResolver, FieldDefinition, FieldConstraints, FieldInfo,
    ColumnResolution, IdentifierResolver, IdentifierContext,
    QualifiedTableName, QualifiedTableNameRef, FieldResolverState, FieldState,
};

// Re-export internal types needed by other modules
pub(crate) use table_catalog::TableMetadataView;

// Re-export CatalogManager (formerly CatalogService)
pub use manager::{CatalogManager, CreateTableResult};
