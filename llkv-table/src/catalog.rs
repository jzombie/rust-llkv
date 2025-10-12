//! Centralized catalog for table and field name resolution.
//!
//! This module provides thread-safe, performant bidirectional mappings between:
//! - Table names ↔ TableId
//! - Field names ↔ FieldId (per table)
//!
//! # Design Goals
//!
//! 1. **No logic duplication**: Single source of truth for all name↔ID conversions
//! 2. **Thread-safe**: Concurrent reads via `Arc<RwLock<_>>`
//! 3. **Performance**: Use FxHashMap for fast lookups, numeric ID comparisons in hot paths
//! 4. **Transaction isolation**: Immutable snapshots for MVCC
//! 5. **Case handling**: Case-insensitive lookups, preserve display names
//!
//! # Usage
//!
//! ```rust,ignore
//! use llkv_table::catalog::{Catalog, FieldResolver};
//!
//! // Create catalog
//! let catalog = Catalog::new();
//!
//! // Register table
//! let table_id = catalog.register_table("Users").unwrap();
//!
//! // Get field resolver for table
//! let field_resolver = catalog.field_resolver(table_id).unwrap();
//!
//! // Register fields
//! let name_fid = field_resolver.register_field("name").unwrap();
//! let email_fid = field_resolver.register_field("Email").unwrap();
//!
//! // Case-insensitive lookups
//! assert_eq!(field_resolver.field_id("NAME"), Some(name_fid));
//! assert_eq!(field_resolver.field_id("email"), Some(email_fid));
//! ```

use crate::types::TableId;
use llkv_column_map::types::FieldId;
use llkv_result::{Error, Result};
use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};

// ============================================================================
// Catalog - Table-level resolver
// ============================================================================

/// Centralized catalog for table name resolution and field resolver management.
///
/// The catalog maintains bidirectional mappings between table names and TableIds,
/// and provides access to per-table field resolvers.
///
/// # Thread Safety
///
/// The catalog uses `Arc<RwLock<_>>` for thread-safe access. Multiple readers
/// can access concurrently, while writers have exclusive access.
///
/// # Case Sensitivity
///
/// Table lookups are case-insensitive (SQL standard), but display names preserve
/// original casing for error messages and schema display.
#[derive(Debug, Clone)]
pub struct Catalog {
    inner: Arc<RwLock<CatalogInner>>,
}

#[derive(Debug)]
struct CatalogInner {
    /// Canonical table name (lowercase) -> TableId
    table_name_to_id: FxHashMap<String, TableId>,
    /// TableId -> TableMetadata
    table_id_to_meta: FxHashMap<TableId, TableMetadata>,
    /// Next table ID to assign (monotonically increasing)
    next_table_id: TableId,
}

/// Metadata for a registered table
#[derive(Debug, Clone)]
struct TableMetadata {
    /// Display name (preserves original case)
    display_name: String,
    /// Canonical name (lowercase for case-insensitive lookups)
    canonical_name: String,
    /// Field resolver for this table's columns
    field_resolver: FieldResolver,
}

impl Catalog {
    /// Create a new empty catalog.
    ///
    /// The catalog starts with no registered tables. The first assigned TableId will be 1.
    /// TableId 0 is reserved for special purposes (e.g., system tables).
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(CatalogInner {
                table_name_to_id: FxHashMap::default(),
                table_id_to_meta: FxHashMap::default(),
                next_table_id: 1, // Start at 1, reserve 0 for system
            })),
        }
    }

    /// Register a new table in the catalog.
    ///
    /// # Arguments
    ///
    /// * `display_name` - The table name as provided by user (case preserved)
    ///
    /// # Returns
    ///
    /// The assigned `TableId` for the newly registered table.
    ///
    /// # Errors
    ///
    /// Returns `Error::CatalogError` if:
    /// - A table with this name (case-insensitive) already exists
    /// - TableId overflow (extremely unlikely)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let catalog = Catalog::new();
    /// let users_id = catalog.register_table("Users")?;
    /// let orders_id = catalog.register_table("Orders")?;
    ///
    /// // Case-insensitive duplicate detection
    /// assert!(catalog.register_table("users").is_err());
    /// ```
    pub fn register_table(&self, display_name: impl Into<String>) -> Result<TableId> {
        let display_name = display_name.into();
        let canonical_name = display_name.to_ascii_lowercase();

        let mut inner = self.inner.write().map_err(|_| {
            Error::Internal("Failed to acquire catalog write lock".to_string())
        })?;

        // Check for duplicate (case-insensitive)
        if inner.table_name_to_id.contains_key(&canonical_name) {
            return Err(Error::CatalogError(format!(
                "Table '{}' already exists in catalog",
                display_name
            )));
        }

        // Assign new TableId
        let table_id = inner.next_table_id;
        inner.next_table_id = inner
            .next_table_id
            .checked_add(1)
            .ok_or_else(|| Error::Internal("TableId overflow".to_string()))?;

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

        Ok(table_id)
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
        let canonical = name.to_ascii_lowercase();
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
    pub fn table_name(&self, id: TableId) -> Option<String> {
        let inner = self.inner.read().ok()?;
        inner
            .table_id_to_meta
            .get(&id)
            .map(|meta| meta.display_name.clone())
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
            .map(|meta| meta.display_name.clone())
            .collect()
    }

    /// Create an immutable snapshot of the catalog for transaction isolation.
    ///
    /// Returns a `CatalogSnapshot` containing all current table name→ID mappings.
    /// Snapshots are cheap to create (uses Arc internally).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let snapshot = catalog.snapshot();
    /// // Transaction sees this snapshot even if catalog changes
    /// let table_id = snapshot.table_id("users");
    /// ```
    pub fn snapshot(&self) -> CatalogSnapshot {
        let inner = match self.inner.read() {
            Ok(inner) => inner,
            Err(_) => {
                return CatalogSnapshot {
                    table_ids: Arc::new(FxHashMap::default()),
                }
            }
        };

        // Clone the mapping into an Arc for cheap snapshot sharing
        let table_ids = Arc::new(inner.table_name_to_id.clone());

        CatalogSnapshot { table_ids }
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
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CatalogSnapshot - Immutable view for transaction isolation
// ============================================================================

/// Immutable snapshot of catalog state for transaction isolation.
///
/// Snapshots capture the table name→ID mappings at a point in time.
/// Transactions use snapshots to ensure consistent view of the catalog
/// throughout their execution, even if new tables are created concurrently.
#[derive(Debug, Clone)]
pub struct CatalogSnapshot {
    /// Canonical table name -> TableId (immutable)
    table_ids: Arc<FxHashMap<String, TableId>>,
}

impl CatalogSnapshot {
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
        let canonical = name.to_ascii_lowercase();
        self.table_ids.get(&canonical).copied()
    }

    /// Check if table exists in this snapshot (case-insensitive).
    pub fn table_exists(&self, name: &str) -> bool {
        self.table_id(name).is_some()
    }

    /// Get all table names in this snapshot.
    pub fn table_names(&self) -> Vec<String> {
        self.table_ids.keys().cloned().collect()
    }

    /// Get the number of tables in this snapshot.
    pub fn table_count(&self) -> usize {
        self.table_ids.len()
    }
}

// ============================================================================
// FieldResolver - Per-table field name resolution
// ============================================================================

/// Per-table field name resolver.
///
/// Each table has its own field resolver that maintains bidirectional mappings
/// between field names and FieldIds. Field IDs start at 3 to reserve space for
/// system columns (row_id=0, created_by=1, deleted_by=2).
///
/// # Thread Safety
///
/// Like `Catalog`, the resolver uses `Arc<RwLock<_>>` for safe concurrent access.
///
/// # Case Sensitivity
///
/// Field lookups are case-insensitive (SQL standard), but display names preserve
/// original casing.
#[derive(Debug, Clone)]
pub struct FieldResolver {
    inner: Arc<RwLock<FieldResolverInner>>,
}

#[derive(Debug)]
struct FieldResolverInner {
    /// Canonical field name (lowercase) -> FieldId
    field_name_to_id: FxHashMap<String, FieldId>,
    /// FieldId -> display name
    field_id_to_name: FxHashMap<FieldId, String>,
    /// Next field ID to assign (starts at 3, reserves 0-2 for system columns)
    next_field_id: FieldId,
}

impl FieldResolver {
    /// Create a new empty field resolver.
    ///
    /// The first assigned FieldId will be 3 (reserves 0-2 for system columns).
    pub fn new() -> Self {
        use crate::reserved::FIRST_USER_FIELD_ID;

        Self {
            inner: Arc::new(RwLock::new(FieldResolverInner {
                field_name_to_id: FxHashMap::default(),
                field_id_to_name: FxHashMap::default(),
                next_field_id: FIRST_USER_FIELD_ID,
            })),
        }
    }

    /// Register a new field in this table.
    ///
    /// # Arguments
    ///
    /// * `display_name` - The field name as provided by user (case preserved)
    ///
    /// # Returns
    ///
    /// The assigned `FieldId` for the newly registered field.
    ///
    /// # Errors
    ///
    /// Returns `Error::CatalogError` if:
    /// - A field with this name (case-insensitive) already exists
    /// - FieldId overflow (unlikely)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let resolver = FieldResolver::new();
    /// let name_fid = resolver.register_field("Name")?;
    /// let email_fid = resolver.register_field("email")?;
    ///
    /// // Case-insensitive duplicate detection
    /// assert!(resolver.register_field("NAME").is_err());
    /// ```
    pub fn register_field(&self, display_name: impl Into<String>) -> Result<FieldId> {
        let display_name = display_name.into();
        let canonical_name = display_name.to_ascii_lowercase();

        let mut inner = self.inner.write().map_err(|_| {
            Error::Internal("Failed to acquire field resolver write lock".to_string())
        })?;

        // Check for duplicate (case-insensitive)
        if inner.field_name_to_id.contains_key(&canonical_name) {
            return Err(Error::CatalogError(format!(
                "Field '{}' already exists in table",
                display_name
            )));
        }

        // Assign new FieldId
        let field_id = inner.next_field_id;
        inner.next_field_id = inner
            .next_field_id
            .checked_add(1)
            .ok_or_else(|| Error::Internal("FieldId overflow".to_string()))?;

        // Store mappings
        inner
            .field_name_to_id
            .insert(canonical_name, field_id);
        inner.field_id_to_name.insert(field_id, display_name);

        Ok(field_id)
    }

    /// Get FieldId by field name (case-insensitive lookup).
    ///
    /// # Arguments
    ///
    /// * `name` - Field name (any casing)
    ///
    /// # Returns
    ///
    /// `Some(FieldId)` if field exists, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let fid = resolver.register_field("UserName")?;
    /// assert_eq!(resolver.field_id("username"), Some(fid));
    /// assert_eq!(resolver.field_id("USERNAME"), Some(fid));
    /// assert_eq!(resolver.field_id("UserName"), Some(fid));
    /// ```
    pub fn field_id(&self, name: &str) -> Option<FieldId> {
        let canonical = name.to_ascii_lowercase();
        let inner = self.inner.read().ok()?;
        inner.field_name_to_id.get(&canonical).copied()
    }

    /// Get display name for a FieldId.
    ///
    /// Returns the original field name with preserved casing.
    ///
    /// # Arguments
    ///
    /// * `id` - The FieldId to look up
    ///
    /// # Returns
    ///
    /// `Some(String)` with the display name if field exists, `None` otherwise.
    pub fn field_name(&self, id: FieldId) -> Option<String> {
        let inner = self.inner.read().ok()?;
        inner.field_id_to_name.get(&id).cloned()
    }

    /// Check if a field exists (case-insensitive).
    ///
    /// # Arguments
    ///
    /// * `name` - Field name to check
    ///
    /// # Returns
    ///
    /// `true` if field exists, `false` otherwise.
    pub fn field_exists(&self, name: &str) -> bool {
        self.field_id(name).is_some()
    }

    /// Get the number of registered fields.
    pub fn field_count(&self) -> usize {
        match self.inner.read() {
            Ok(inner) => inner.field_id_to_name.len(),
            Err(_) => 0,
        }
    }

    /// Get all field names in display format.
    pub fn field_names(&self) -> Vec<String> {
        match self.inner.read() {
            Ok(inner) => inner.field_id_to_name.values().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }
}

impl Default for FieldResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_basic_operations() {
        let catalog = Catalog::new();

        // Register tables
        let users_id = catalog.register_table("Users").unwrap();
        let orders_id = catalog.register_table("Orders").unwrap();

        assert_ne!(users_id, orders_id);

        // Lookup by name (case-insensitive)
        assert_eq!(catalog.table_id("users"), Some(users_id));
        assert_eq!(catalog.table_id("USERS"), Some(users_id));
        assert_eq!(catalog.table_id("Users"), Some(users_id));

        // Reverse lookup preserves display name
        assert_eq!(catalog.table_name(users_id), Some("Users".to_string()));
        assert_eq!(catalog.table_name(orders_id), Some("Orders".to_string()));

        // Non-existent table
        assert_eq!(catalog.table_id("Products"), None);
        assert_eq!(catalog.table_name(999), None);
    }

    #[test]
    fn test_catalog_duplicate_detection() {
        let catalog = Catalog::new();

        catalog.register_table("Users").unwrap();

        // Case-insensitive duplicate detection
        assert!(catalog.register_table("users").is_err());
        assert!(catalog.register_table("USERS").is_err());
        assert!(catalog.register_table("UsErS").is_err());
    }

    #[test]
    fn test_catalog_table_names() {
        let catalog = Catalog::new();

        catalog.register_table("Users").unwrap();
        catalog.register_table("Orders").unwrap();
        catalog.register_table("Products").unwrap();

        let names = catalog.table_names();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"Users".to_string()));
        assert!(names.contains(&"Orders".to_string()));
        assert!(names.contains(&"Products".to_string()));
    }

    #[test]
    fn test_catalog_snapshot() {
        let catalog = Catalog::new();

        let users_id = catalog.register_table("Users").unwrap();
        let snapshot = catalog.snapshot();

        // Snapshot sees existing table
        assert_eq!(snapshot.table_id("users"), Some(users_id));

        // Add new table after snapshot
        catalog.register_table("Orders").unwrap();

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
        let catalog = Catalog::new();

        // Register table
        let table_id = catalog.register_table("Users").unwrap();

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
        let catalog = Catalog::new();
        catalog.register_table("Users").unwrap();

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
}
