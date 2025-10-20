//! Centralized table catalog for table and field name resolution.
//!
//! This module provides thread-safe, performant bidirectional mappings between:
//! - Table names ↔ TableId (case-insensitive)
//! - Field names ↔ FieldId per table (case-insensitive)
//!
//! # Purpose
//!
//! The catalog is an in-memory name resolution layer used during query execution.
//! It does NOT create tables or generate IDs - use `CatalogService` for table creation.
//!
//! # Architecture
//!
//! - **CatalogService**: Creates tables (coordinates metadata + catalog + storage)
//! - **TableCatalog** (this module): Name lookups during query execution
//! - **MetadataManager**: Persistent storage, generates table IDs (source of truth)
//! - **Table**: Data operations (append, scan, update, delete)
//!
//! # Design Goals
//!
//! 1. **Single source of truth**: MetadataManager generates IDs, catalog registers them
//! 2. **Thread-safe**: Concurrent reads via `Arc<RwLock<_>>`
//! 3. **Performance**: Fast lookups with FxHashMap
//! 4. **Transaction isolation**: Immutable snapshots for MVCC
//! 5. **Case-insensitive**: SQL standard compliance, preserves display names

use crate::resolvers::TableNameKey;
use crate::types::TableId;
use llkv_column_map::types::FieldId;
use llkv_result::{Error, Result};
use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};

pub use crate::resolvers::{
    ColumnResolution, FieldConstraints, FieldDefinition, FieldInfo, FieldResolver,
    FieldResolverState, FieldState, IdentifierContext, IdentifierResolver, QualifiedTableName,
    QualifiedTableNameRef,
};

// ============================================================================
// TableCatalog - Table-level resolver
// ============================================================================

/// In-memory table name resolution for query execution.
///
/// **To create tables, use `CatalogService`, not this struct directly.**
///
/// The catalog maps table names to TableIds for fast lookups during query planning
/// and execution. It does not create tables, generate IDs, or persist metadata.
///
/// Table IDs are provided by `MetadataManager` (the persistent layer and source of truth).
/// The catalog simply registers these IDs for in-memory name resolution.
#[derive(Debug, Clone)]
pub struct TableCatalog {
    inner: Arc<RwLock<TableCatalogInner>>,
}

#[derive(Debug)]
struct TableCatalogInner {
    /// Canonical table name (lowercase components) -> TableId
    /// Table names may be schema-qualified (e.g., "test.tbl")
    table_name_to_id: FxHashMap<TableNameKey, TableId>,
    /// TableId -> TableMetadata
    table_id_to_meta: FxHashMap<TableId, TableMetadata>,
    /// Next table ID to assign (monotonically increasing)
    next_table_id: TableId,
    /// Set of registered schema names (canonical lowercase)
    schemas: rustc_hash::FxHashSet<String>,
}

/// Metadata for a registered table
#[derive(Debug, Clone)]
struct TableMetadata {
    /// Display name (preserves original case)
    display_name: QualifiedTableName,
    /// Canonical name (lowercase for case-insensitive lookups)
    canonical_name: TableNameKey,
    /// Field resolver for this table's columns
    field_resolver: FieldResolver,
}

#[derive(Debug, Clone)]
pub(crate) struct TableMetadataView {
    canonical_name: TableNameKey,
}

impl TableMetadata {
    fn to_view(&self) -> TableMetadataView {
        TableMetadataView {
            canonical_name: self.canonical_name.clone(),
        }
    }
}

impl TableMetadataView {
    pub(crate) fn canonical_table(&self) -> &str {
        self.canonical_name.table()
    }

    pub(crate) fn canonical_schema(&self) -> Option<&str> {
        self.canonical_name.schema()
    }
}

impl TableCatalog {
    /// Create a new empty catalog.
    ///
    /// The catalog starts with no registered tables. The first assigned TableId will be 1.
    /// TableId 0 is reserved for special purposes (e.g., system tables).
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(TableCatalogInner {
                table_name_to_id: FxHashMap::default(),
                table_id_to_meta: FxHashMap::default(),
                next_table_id: 1, // Start at 1, reserve 0 for system
                schemas: rustc_hash::FxHashSet::default(),
            })),
        }
    }

    /// Register a table in the catalog with the specified table ID.
    ///
    /// **This does NOT create tables.** It registers an existing table ID from
    /// `MetadataManager` for name lookups. Use `CatalogService` to create tables.
    ///
    /// The table_id must come from metadata (source of truth):
    /// - New tables: `MetadataManager::reserve_table_id()`
    /// - Database restart: `MetadataManager::all_table_metas()`
    ///
    /// # Errors
    ///
    /// Returns error if the name or ID is already registered.
    pub fn register_table(
        &self,
        name: impl Into<QualifiedTableName>,
        table_id: TableId,
    ) -> Result<()> {
        let display_name: QualifiedTableName = name.into();
        let canonical_name = display_name.canonical_key();

        let mut inner = self
            .inner
            .write()
            .map_err(|_| Error::Internal("Failed to acquire catalog write lock".to_string()))?;

        // Check for duplicate name (case-insensitive)
        if inner.table_name_to_id.contains_key(&canonical_name) {
            return Err(Error::CatalogError(format!(
                "Table '{}' already exists in catalog",
                display_name
            )));
        }

        // Check for duplicate ID
        if inner.table_id_to_meta.contains_key(&table_id) {
            return Err(Error::CatalogError(format!(
                "Table ID {} already exists in catalog",
                table_id
            )));
        }

        // Update next_table_id if necessary to avoid conflicts with future allocations
        if table_id >= inner.next_table_id {
            inner.next_table_id = table_id
                .checked_add(1)
                .ok_or_else(|| Error::Internal("TableId overflow".to_string()))?;
        }

        // Create field resolver for this table
        let field_resolver = FieldResolver::new();

        // Store metadata
        let metadata = TableMetadata {
            display_name: display_name.clone(),
            canonical_name: canonical_name.clone(),
            field_resolver,
        };

        inner.table_name_to_id.insert(canonical_name, table_id);
        inner.table_id_to_meta.insert(table_id, metadata);

        Ok(())
    }

    /// Unregister a table from the catalog.
    ///
    /// Removes the table and its associated field resolver from the catalog.
    /// This is typically called when a table is dropped.
    ///
    /// # Arguments
    ///
    /// * `table_id` - The unique table identifier
    ///
    /// # Returns
    ///
    /// `true` if the table was found and removed, `false` if it didn't exist.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let id = catalog.register_table("Users")?;
    /// assert!(catalog.unregister_table(id));
    /// assert_eq!(catalog.table_id("users"), None);
    /// ```
    pub fn unregister_table(&self, table_id: TableId) -> bool {
        let mut inner = match self.inner.write() {
            Ok(guard) => guard,
            Err(_) => return false,
        };

        // Remove from id → metadata map and get the canonical name
        if let Some(meta) = inner.table_id_to_meta.remove(&table_id) {
            // Remove from name → id map using the canonical key
            let canonical_key = meta.display_name.canonical_key();
            inner.table_name_to_id.remove(&canonical_key);
            true
        } else {
            false
        }
    }

    /// Get TableId by table name (case-insensitive lookup).
    ///
    /// # Arguments
    ///
    /// * `name` - Table name (any casing)
    ///
    /// # Returns
    ///
    /// `Some(TableId)` if table exists, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let id = catalog.register_table("Users")?;
    /// assert_eq!(catalog.table_id("users"), Some(id));
    /// assert_eq!(catalog.table_id("USERS"), Some(id));
    /// assert_eq!(catalog.table_id("Users"), Some(id));
    /// ```
    pub fn table_id(&self, name: &str) -> Option<TableId> {
        let canonical = QualifiedTableNameRef::from(name).canonical_key();
        let inner = self.inner.read().ok()?;
        inner.table_name_to_id.get(&canonical).copied()
    }

    /// Get display name for a TableId.
    ///
    /// Returns the original table name with preserved casing.
    ///
    /// # Arguments
    ///
    /// * `id` - The TableId to look up
    ///
    /// # Returns
    ///
    /// `Some(String)` with the display name if table exists, `None` otherwise.
    pub fn table_name(&self, id: TableId) -> Option<QualifiedTableName> {
        let inner = self.inner.read().ok()?;
        inner
            .table_id_to_meta
            .get(&id)
            .map(|meta| meta.display_name.clone())
    }

    pub(crate) fn table_metadata_view(&self, id: TableId) -> Option<TableMetadataView> {
        let inner = self.inner.read().ok()?;
        inner.table_id_to_meta.get(&id).map(TableMetadata::to_view)
    }

    /// Get field resolver for a table.
    ///
    /// # Arguments
    ///
    /// * `table_id` - The TableId to get the resolver for
    ///
    /// # Returns
    ///
    /// `Some(FieldResolver)` if table exists, `None` otherwise.
    pub fn field_resolver(&self, table_id: TableId) -> Option<FieldResolver> {
        let inner = self.inner.read().ok()?;
        inner
            .table_id_to_meta
            .get(&table_id)
            .map(|meta| meta.field_resolver.clone())
    }

    /// List all registered table names.
    ///
    /// Returns display names (with original casing) for all tables.
    /// Useful for `SHOW TABLES` queries.
    ///
    /// # Returns
    ///
    /// Vector of table display names in insertion order (not guaranteed sorted).
    pub fn table_names(&self) -> Vec<String> {
        let inner = match self.inner.read() {
            Ok(inner) => inner,
            Err(_) => return Vec::new(),
        };

        inner
            .table_id_to_meta
            .values()
            .map(|meta| meta.display_name.to_display_string())
            .collect()
    }

    /// Create an immutable snapshot of the catalog for transaction isolation.
    ///
    /// Returns a `TableCatalogSnapshot` containing all current table name→ID mappings.
    /// Snapshots are cheap to create (uses Arc internally).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let snapshot = catalog.snapshot();
    /// // Transaction sees this snapshot even if catalog changes
    /// let table_id = snapshot.table_id("users");
    /// ```
    pub fn snapshot(&self) -> TableCatalogSnapshot {
        let inner = match self.inner.read() {
            Ok(inner) => inner,
            Err(_) => {
                return TableCatalogSnapshot {
                    table_ids: Arc::new(FxHashMap::default()),
                };
            }
        };

        // Clone the mapping into an Arc for cheap snapshot sharing
        let table_ids = Arc::new(inner.table_name_to_id.clone());

        TableCatalogSnapshot { table_ids }
    }

    /// Check if a table exists (case-insensitive).
    ///
    /// # Arguments
    ///
    /// * `name` - Table name to check
    ///
    /// # Returns
    ///
    /// `true` if table exists, `false` otherwise.
    pub fn table_exists(&self, name: &str) -> bool {
        self.table_id(name).is_some()
    }

    /// Get the number of registered tables.
    pub fn table_count(&self) -> usize {
        match self.inner.read() {
            Ok(inner) => inner.table_id_to_meta.len(),
            Err(_) => 0,
        }
    }

    /// Register a new schema in the catalog.
    ///
    /// # Arguments
    ///
    /// * `name` - The schema name (case will be preserved in lookups but stored canonically)
    ///
    /// # Returns
    ///
    /// `Ok(())` if the schema was registered successfully.
    ///
    /// # Errors
    ///
    /// Returns `Error::CatalogError` if a schema with this name already exists (case-insensitive).
    pub fn register_schema(&self, name: impl Into<String>) -> Result<()> {
        let name = name.into();
        let canonical = name.to_ascii_lowercase();

        let mut inner = self
            .inner
            .write()
            .map_err(|_| Error::Internal("Failed to acquire catalog write lock".to_string()))?;

        if inner.schemas.contains(&canonical) {
            return Err(Error::CatalogError(format!(
                "Schema '{}' already exists",
                name
            )));
        }

        inner.schemas.insert(canonical);
        Ok(())
    }

    /// Check if a schema exists (case-insensitive).
    ///
    /// # Arguments
    ///
    /// * `name` - Schema name to check
    ///
    /// # Returns
    ///
    /// `true` if schema exists, `false` otherwise.
    pub fn schema_exists(&self, name: &str) -> bool {
        let canonical = name.to_ascii_lowercase();
        match self.inner.read() {
            Ok(inner) => inner.schemas.contains(&canonical),
            Err(_) => false,
        }
    }

    /// List all registered schema names.
    ///
    /// Returns all schema names in canonical (lowercase) form.
    pub fn schema_names(&self) -> Vec<String> {
        match self.inner.read() {
            Ok(inner) => inner.schemas.iter().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Drop (unregister) a schema from the catalog.
    ///
    /// This does NOT cascade to tables - caller must handle that separately.
    ///
    /// # Returns
    ///
    /// `true` if the schema was found and removed, `false` if it didn't exist.
    pub fn unregister_schema(&self, name: &str) -> bool {
        let canonical = name.to_ascii_lowercase();
        let mut inner = match self.inner.write() {
            Ok(guard) => guard,
            Err(_) => return false,
        };

        inner.schemas.remove(&canonical)
    }

    /// Export catalog state for persistence.
    ///
    /// Returns a serializable representation of the entire catalog state,
    /// including all table and field mappings.
    ///
    /// # Returns
    ///
    /// `TableCatalogState` containing all tables, fields, and next ID counters.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let state = catalog.export_state();
    /// // ... serialize state to disk ...
    /// let restored_catalog = TableCatalog::from_state(state)?;
    /// ```
    pub fn export_state(&self) -> TableCatalogState {
        let inner = match self.inner.read() {
            Ok(inner) => inner,
            Err(_) => {
                return TableCatalogState {
                    tables: Vec::new(),
                    next_table_id: 1,
                    schemas: Vec::new(),
                };
            }
        };

        let mut tables = Vec::new();

        for (&table_id, meta) in &inner.table_id_to_meta {
            let field_state = meta.field_resolver.export_state();
            let display_schema = meta.display_name.schema().map(|s| s.to_string());
            let display_table = meta.display_name.table().to_string();
            let canonical_schema = meta.canonical_name.schema().map(|s| s.to_string());
            let canonical_table = meta.canonical_name.table().to_string();
            tables.push(TableState {
                table_id,
                display_schema,
                display_table,
                canonical_schema,
                canonical_table,
                fields: field_state.fields,
                next_field_id: field_state.next_field_id,
            });
        }

        let schemas: Vec<String> = inner.schemas.iter().cloned().collect();

        TableCatalogState {
            tables,
            next_table_id: inner.next_table_id,
            schemas,
        }
    }

    /// Restore catalog from persisted state.
    ///
    /// Creates a new catalog instance with all table and field mappings
    /// restored from the provided state.
    ///
    /// # Arguments
    ///
    /// * `state` - Previously exported catalog state
    ///
    /// # Returns
    ///
    /// A fully restored `TableCatalog` instance.
    ///
    /// # Errors
    ///
    /// Returns `Error::CatalogError` if state is invalid (e.g., duplicate names/IDs).
    pub fn from_state(state: TableCatalogState) -> Result<Self> {
        let mut table_name_to_id = FxHashMap::default();
        let mut table_id_to_meta = FxHashMap::default();

        for table_state in state.tables {
            // Check for duplicate table IDs
            if table_id_to_meta.contains_key(&table_state.table_id) {
                return Err(Error::CatalogError(format!(
                    "Duplicate table_id {} in catalog state",
                    table_state.table_id
                )));
            }

            // Check for duplicate table names
            let canonical_name = TableNameKey::new(
                table_state.canonical_schema.as_deref(),
                &table_state.canonical_table,
            );

            if table_name_to_id.contains_key(&canonical_name) {
                return Err(Error::CatalogError(format!(
                    "Duplicate table name '{}' in catalog state",
                    QualifiedTableName::new(
                        table_state.display_schema.clone(),
                        table_state.display_table.clone(),
                    )
                )));
            }

            // Restore field resolver
            let field_resolver = FieldResolver::from_state(FieldResolverState {
                fields: table_state.fields,
                next_field_id: table_state.next_field_id,
            })?;

            let metadata = TableMetadata {
                display_name: QualifiedTableName::new(
                    table_state.display_schema.clone(),
                    table_state.display_table.clone(),
                ),
                canonical_name: canonical_name.clone(),
                field_resolver,
            };

            table_name_to_id.insert(canonical_name, table_state.table_id);
            table_id_to_meta.insert(table_state.table_id, metadata);
        }

        Ok(Self {
            inner: Arc::new(RwLock::new(TableCatalogInner {
                table_name_to_id,
                table_id_to_meta,
                next_table_id: state.next_table_id,
                schemas: state.schemas.into_iter().collect(),
            })),
        })
    }
}

impl Default for TableCatalog {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TableCatalogSnapshot - Immutable view for transaction isolation
// ============================================================================

/// Immutable snapshot of table catalog state for transaction isolation.
///
/// Snapshots capture the table name→ID mappings at a point in time.
/// Transactions use snapshots to ensure consistent view of the catalog
/// throughout their execution, even if new tables are created concurrently.
#[derive(Debug, Clone)]
pub struct TableCatalogSnapshot {
    /// Canonical table name -> TableId (immutable)
    table_ids: Arc<FxHashMap<TableNameKey, TableId>>,
}

impl TableCatalogSnapshot {
    /// Get TableId by name from this snapshot (case-insensitive).
    ///
    /// # Arguments
    ///
    /// * `name` - Table name (any casing)
    ///
    /// # Returns
    ///
    /// `Some(TableId)` if table existed in snapshot, `None` otherwise.
    pub fn table_id(&self, name: &str) -> Option<TableId> {
        let canonical = QualifiedTableNameRef::from(name).canonical_key();
        self.table_ids.get(&canonical).copied()
    }

    /// Check if table exists in this snapshot (case-insensitive).
    pub fn table_exists(&self, name: &str) -> bool {
        self.table_id(name).is_some()
    }

    /// Get all table names in this snapshot.
    pub fn table_names(&self) -> Vec<String> {
        self.table_ids.keys().map(TableNameKey::to_string).collect()
    }

    /// Get the number of tables in this snapshot.
    pub fn table_count(&self) -> usize {
        self.table_ids.len()
    }
}

// ============================================================================
// Persistence Types
// ============================================================================

/// Serializable table catalog state for persistence.
///
/// This structure contains all information needed to restore a table catalog
/// from disk, including all table and field mappings.
#[derive(Debug, Clone, bitcode::Encode, bitcode::Decode)]
pub struct TableCatalogState {
    /// All registered tables with their field resolvers
    pub tables: Vec<TableState>,
    /// Next table ID to assign
    pub next_table_id: TableId,
    /// All registered schema names (canonical lowercase)
    pub schemas: Vec<String>,
}

/// Serializable table state.
#[derive(Debug, Clone, bitcode::Encode, bitcode::Decode)]
pub struct TableState {
    /// The table's unique ID
    pub table_id: TableId,
    /// Display schema (preserves original case)
    pub display_schema: Option<String>,
    /// Display table name (preserves original case)
    pub display_table: String,
    /// Canonical schema (lowercase)
    pub canonical_schema: Option<String>,
    /// Canonical table name (lowercase)
    pub canonical_table: String,
    /// All fields in this table
    pub fields: Vec<FieldState>,
    /// Next field ID to assign
    pub next_field_id: FieldId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_basic_operations() {
        let catalog = TableCatalog::new();

        // Register tables with explicit IDs (simulating metadata layer)
        let users_id = 1;
        let orders_id = 2;
        catalog.register_table("Users", users_id).unwrap();
        catalog.register_table("Orders", orders_id).unwrap();

        assert_ne!(users_id, orders_id);

        // Lookup by name (case-insensitive)
        assert_eq!(catalog.table_id("users"), Some(users_id));
        assert_eq!(catalog.table_id("USERS"), Some(users_id));
        assert_eq!(catalog.table_id("Users"), Some(users_id));

        // Reverse lookup preserves display name
        assert_eq!(
            catalog.table_name(users_id),
            Some(QualifiedTableName::from("Users"))
        );
        assert_eq!(
            catalog.table_name(orders_id),
            Some(QualifiedTableName::from("Orders"))
        );

        // Non-existent table
        assert_eq!(catalog.table_id("Products"), None);
        assert_eq!(catalog.table_name(999), None);
    }

    #[test]
    fn test_catalog_duplicate_detection() {
        let catalog = TableCatalog::new();

        catalog.register_table("Users", 1).unwrap();

        // Case-insensitive duplicate detection
        assert!(catalog.register_table("users", 2).is_err());
        assert!(catalog.register_table("USERS", 3).is_err());
        assert!(catalog.register_table("UsErS", 4).is_err());
    }

    #[test]
    fn test_catalog_table_names() {
        let catalog = TableCatalog::new();

        catalog.register_table("Users", 1).unwrap();
        catalog.register_table("Orders", 2).unwrap();
        catalog.register_table("Products", 3).unwrap();

        let names = catalog.table_names();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"Users".to_string()));
        assert!(names.contains(&"Orders".to_string()));
        assert!(names.contains(&"Products".to_string()));
    }

    #[test]
    fn test_catalog_snapshot() {
        let catalog = TableCatalog::new();

        let users_id = 1;
        catalog.register_table("Users", users_id).unwrap();
        let snapshot = catalog.snapshot();

        // Snapshot sees existing table
        assert_eq!(snapshot.table_id("users"), Some(users_id));

        // Add new table after snapshot
        catalog.register_table("Orders", 2).unwrap();

        // Snapshot doesn't see new table (isolation)
        assert_eq!(snapshot.table_id("orders"), None);
        assert_eq!(snapshot.table_count(), 1);

        // But new snapshot does
        let snapshot2 = catalog.snapshot();
        assert!(snapshot2.table_id("orders").is_some());
        assert_eq!(snapshot2.table_count(), 2);
    }

    #[test]
    fn test_field_resolver_basic_operations() {
        let resolver = FieldResolver::new();

        // Register fields
        let name_fid = resolver.register_field("name").unwrap();
        let email_fid = resolver.register_field("Email").unwrap();
        let age_fid = resolver.register_field("AGE").unwrap();

        assert_ne!(name_fid, email_fid);
        assert_ne!(email_fid, age_fid);

        // All should be >= FIRST_USER_FIELD_ID (1)
        // System columns use 0 for row_id, sentinels for MVCC
        assert!(name_fid >= crate::reserved::FIRST_USER_FIELD_ID);
        assert!(email_fid >= crate::reserved::FIRST_USER_FIELD_ID);
        assert!(age_fid >= crate::reserved::FIRST_USER_FIELD_ID);

        // Lookup by name (case-insensitive)
        assert_eq!(resolver.field_id("name"), Some(name_fid));
        assert_eq!(resolver.field_id("NAME"), Some(name_fid));
        assert_eq!(resolver.field_id("Name"), Some(name_fid));

        assert_eq!(resolver.field_id("email"), Some(email_fid));
        assert_eq!(resolver.field_id("EMAIL"), Some(email_fid));

        // Reverse lookup preserves display name
        assert_eq!(resolver.field_name(name_fid), Some("name".to_string()));
        assert_eq!(resolver.field_name(email_fid), Some("Email".to_string()));
        assert_eq!(resolver.field_name(age_fid), Some("AGE".to_string()));

        // Non-existent field
        assert_eq!(resolver.field_id("salary"), None);
        assert_eq!(resolver.field_name(999), None);
    }

    #[test]
    fn test_field_resolver_duplicate_detection() {
        let resolver = FieldResolver::new();

        resolver.register_field("name").unwrap();

        // Case-insensitive duplicate detection
        assert!(resolver.register_field("name").is_err());
        assert!(resolver.register_field("NAME").is_err());
        assert!(resolver.register_field("NaMe").is_err());
    }

    #[test]
    fn test_field_resolver_field_names() {
        let resolver = FieldResolver::new();

        resolver.register_field("name").unwrap();
        resolver.register_field("Email").unwrap();
        resolver.register_field("AGE").unwrap();

        let names = resolver.field_names();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"name".to_string()));
        assert!(names.contains(&"Email".to_string()));
        assert!(names.contains(&"AGE".to_string()));
    }

    #[test]
    fn test_catalog_with_field_resolver() {
        let catalog = TableCatalog::new();

        // Register table
        let table_id = 1;
        catalog.register_table("Users", table_id).unwrap();

        // Get field resolver
        let resolver = catalog.field_resolver(table_id).unwrap();

        // Register fields
        let name_fid = resolver.register_field("name").unwrap();
        let email_fid = resolver.register_field("email").unwrap();

        // Verify fields are registered
        assert_eq!(resolver.field_id("name"), Some(name_fid));
        assert_eq!(resolver.field_id("email"), Some(email_fid));

        // Get resolver again (should be same instance via Arc)
        let resolver2 = catalog.field_resolver(table_id).unwrap();
        assert_eq!(resolver2.field_id("name"), Some(name_fid));
        assert_eq!(resolver2.field_id("email"), Some(email_fid));
    }

    #[test]
    fn test_catalog_exists_helpers() {
        let catalog = TableCatalog::new();
        catalog.register_table("Users", 1).unwrap();

        assert!(catalog.table_exists("users"));
        assert!(catalog.table_exists("USERS"));
        assert!(!catalog.table_exists("orders"));

        assert_eq!(catalog.table_count(), 1);
    }

    #[test]
    fn test_field_resolver_exists_helpers() {
        let resolver = FieldResolver::new();
        resolver.register_field("name").unwrap();

        assert!(resolver.field_exists("name"));
        assert!(resolver.field_exists("NAME"));
        assert!(!resolver.field_exists("email"));

        assert_eq!(resolver.field_count(), 1);
    }

    #[test]
    fn test_catalog_persistence_export_import() {
        let catalog = TableCatalog::new();

        // Register tables with fields
        let users_id = 1;
        let orders_id = 2;
        catalog.register_table("Users", users_id).unwrap();
        catalog.register_table("Orders", orders_id).unwrap();

        let users_resolver = catalog.field_resolver(users_id).unwrap();
        let name_fid = users_resolver.register_field("name").unwrap();
        let email_fid = users_resolver.register_field("Email").unwrap();

        let orders_resolver = catalog.field_resolver(orders_id).unwrap();
        let product_fid = orders_resolver.register_field("product").unwrap();
        let qty_fid = orders_resolver.register_field("quantity").unwrap();

        // Export state
        let state = catalog.export_state();
        assert_eq!(state.tables.len(), 2);
        assert!(state.next_table_id > orders_id);

        // Restore from state
        let restored_catalog = TableCatalog::from_state(state).unwrap();

        // Verify tables
        assert_eq!(restored_catalog.table_id("users"), Some(users_id));
        assert_eq!(restored_catalog.table_id("orders"), Some(orders_id));
        assert_eq!(
            restored_catalog.table_name(users_id),
            Some(QualifiedTableName::from("Users"))
        );
        assert_eq!(
            restored_catalog.table_name(orders_id),
            Some(QualifiedTableName::from("Orders"))
        );

        // Verify fields
        let restored_users_resolver = restored_catalog.field_resolver(users_id).unwrap();
        assert_eq!(restored_users_resolver.field_id("name"), Some(name_fid));
        assert_eq!(restored_users_resolver.field_id("email"), Some(email_fid));
        assert_eq!(
            restored_users_resolver.field_name(name_fid),
            Some("name".to_string())
        );
        assert_eq!(
            restored_users_resolver.field_name(email_fid),
            Some("Email".to_string())
        );

        let restored_orders_resolver = restored_catalog.field_resolver(orders_id).unwrap();
        assert_eq!(
            restored_orders_resolver.field_id("product"),
            Some(product_fid)
        );
        assert_eq!(restored_orders_resolver.field_id("quantity"), Some(qty_fid));
    }

    #[test]
    fn test_catalog_persistence_id_stability() {
        let catalog = TableCatalog::new();

        // Register tables
        let table1_id = 1;
        let table2_id = 2;
        catalog.register_table("Table1", table1_id).unwrap();
        catalog.register_table("Table2", table2_id).unwrap();

        // Export and restore
        let state = catalog.export_state();
        let restored = TableCatalog::from_state(state).unwrap();

        // IDs should be stable
        assert_eq!(restored.table_id("table1"), Some(table1_id));
        assert_eq!(restored.table_id("table2"), Some(table2_id));

        // New registrations should continue from saved counter
        let table3_id = 3;
        restored.register_table("Table3", table3_id).unwrap();
        assert!(table3_id > table2_id);
    }

    #[test]
    fn test_field_resolver_persistence() {
        let resolver = FieldResolver::new();

        let fid1 = resolver.register_field("field1").unwrap();
        let fid2 = resolver.register_field("Field2").unwrap();
        let fid3 = resolver.register_field("FIELD3").unwrap();

        // Export state
        let state = resolver.export_state();
        assert_eq!(state.fields.len(), 3);

        // Restore from state
        let restored = FieldResolver::from_state(state).unwrap();

        // Verify case-insensitive lookups work
        assert_eq!(restored.field_id("field1"), Some(fid1));
        assert_eq!(restored.field_id("FIELD1"), Some(fid1));
        assert_eq!(restored.field_id("field2"), Some(fid2));
        assert_eq!(restored.field_id("field3"), Some(fid3));

        // Verify display names preserved
        assert_eq!(restored.field_name(fid1), Some("field1".to_string()));
        assert_eq!(restored.field_name(fid2), Some("Field2".to_string()));
        assert_eq!(restored.field_name(fid3), Some("FIELD3".to_string()));

        // New registrations should continue from saved counter
        let fid4 = restored.register_field("field4").unwrap();
        assert!(fid4 > fid3);
    }

    #[test]
    fn test_field_constraints_roundtrip() {
        let resolver = FieldResolver::new();

        let fid = resolver
            .register_field(
                FieldDefinition::new("id")
                    .with_primary_key(true)
                    .with_unique(true)
                    .with_check_expr(Some("id > 0".to_string())),
            )
            .unwrap();

        let constraints = resolver.field_constraints(fid).unwrap();
        assert!(constraints.primary_key);
        assert!(constraints.unique);
        assert_eq!(constraints.check_expr.as_deref(), Some("id > 0"));

        let by_name = resolver.field_constraints_by_name("ID").unwrap();
        assert_eq!(by_name, constraints);

        let info = resolver.field_info(fid).unwrap();
        assert_eq!(info.field_id, fid);
        assert_eq!(info.display_name, "id");
        assert!(info.constraints.primary_key);
        assert!(info.constraints.unique);

        let state = resolver.export_state();
        let restored = FieldResolver::from_state(state).unwrap();
        let restored_constraints = restored.field_constraints(fid).unwrap();
        assert_eq!(restored_constraints, constraints);
        let restored_info = restored.field_info_by_name("id").unwrap();
        assert_eq!(restored_info.constraints, constraints);
    }

    #[test]
    fn test_catalog_persistence_error_duplicate_table_id() {
        let state = TableCatalogState {
            tables: vec![
                TableState {
                    table_id: 1,
                    display_schema: None,
                    display_table: "Table1".to_string(),
                    canonical_schema: None,
                    canonical_table: "table1".to_string(),
                    fields: Vec::new(),
                    next_field_id: 3,
                },
                TableState {
                    table_id: 1, // Duplicate!
                    display_schema: None,
                    display_table: "Table2".to_string(),
                    canonical_schema: None,
                    canonical_table: "table2".to_string(),
                    fields: Vec::new(),
                    next_field_id: 3,
                },
            ],
            next_table_id: 3,
            schemas: Vec::new(),
        };

        assert!(TableCatalog::from_state(state).is_err());
    }

    #[test]
    fn test_catalog_persistence_error_duplicate_table_name() {
        let state = TableCatalogState {
            tables: vec![
                TableState {
                    table_id: 1,
                    display_schema: None,
                    display_table: "Table1".to_string(),
                    canonical_schema: None,
                    canonical_table: "table1".to_string(),
                    fields: Vec::new(),
                    next_field_id: 3,
                },
                TableState {
                    table_id: 2,
                    display_schema: None,
                    display_table: "TABLE1".to_string(), // Duplicate (case-insensitive)
                    canonical_schema: None,
                    canonical_table: "table1".to_string(),
                    fields: Vec::new(),
                    next_field_id: 3,
                },
            ],
            next_table_id: 3,
            schemas: Vec::new(),
        };

        assert!(TableCatalog::from_state(state).is_err());
    }
}
