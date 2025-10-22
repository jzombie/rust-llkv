//! Unified catalog module for table and schema management.
//!
//! This module provides a single interface for all catalog operations:
//! - `TableCatalog`: In-memory name resolution cache (table name ↔ id, field name ↔ id)
//! - `CatalogManager`: High-level service for creating/dropping tables and managing metadata
//!
//! External code should use this `catalog` module instead of accessing individual components.

mod manager;
mod table_catalog;

pub use table_catalog::{
    ColumnResolution, FieldConstraints, FieldDefinition, FieldInfo, FieldResolver,
    FieldResolverState, FieldState, IdentifierContext, IdentifierResolver, QualifiedTableName,
    QualifiedTableNameRef, TableCatalog, TableCatalogSnapshot, TableCatalogState, TableState,
};

pub(crate) use table_catalog::TableMetadataView;

pub use manager::{
    CatalogManager, CreateTableResult, MvccColumnBuilder, SingleColumnIndexDescriptor,
    SingleColumnIndexRegistration,
};
