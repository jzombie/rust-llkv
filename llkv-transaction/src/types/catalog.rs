//! Transaction catalog snapshot trait.

use llkv_column_map::types::TableId;

/// Catalog snapshot interface used by the transaction layer.
///
/// Implementers provide immutable access to table name→ID mappings captured at
/// the start of a transaction. The trait is lightweight so callers can wrap or
/// translate existing catalog views without copying large data structures.
///
/// # Purpose
///
/// This trait allows the transaction layer to work with any catalog implementation
/// that can provide a consistent snapshot of table metadata. The snapshot ensures
/// that table lookups within a transaction remain consistent even if the catalog
/// is modified by other transactions.
pub trait TransactionCatalogSnapshot: Clone + Send + Sync + 'static {
    /// Look up a table ID by name (case-insensitive).
    ///
    /// Returns `None` if the table does not exist in this snapshot.
    fn table_id(&self, name: &str) -> Option<TableId>;

    /// Check whether a table exists in this snapshot.
    ///
    /// This is a convenience method that calls [`table_id`](Self::table_id).
    fn table_exists(&self, name: &str) -> bool {
        self.table_id(name).is_some()
    }

    /// Return all table names captured in this snapshot.
    fn table_names(&self) -> Vec<String>;
}

/// Implement TransactionCatalogSnapshot for llkv_table's TableCatalogSnapshot.
impl TransactionCatalogSnapshot for llkv_table::catalog::TableCatalogSnapshot {
    fn table_id(&self, name: &str) -> Option<TableId> {
        llkv_table::catalog::TableCatalogSnapshot::table_id(self, name)
    }

    fn table_exists(&self, name: &str) -> bool {
        llkv_table::catalog::TableCatalogSnapshot::table_exists(self, name)
    }

    fn table_names(&self) -> Vec<String> {
        llkv_table::catalog::TableCatalogSnapshot::table_names(self)
    }
}
