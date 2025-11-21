use crate::types::PhysicalKey;

// Well-known key for the root ColumnCatalog.
pub const CATALOG_ROOT_PKEY: PhysicalKey = 0;

/// Name of the row ID column.
pub const ROW_ID_COLUMN_NAME: &str = "row_id";

/// Name of the MVCC created_by transaction column.
pub const CREATED_BY_COLUMN_NAME: &str = "created_by_txn";

/// Name of the MVCC deleted_by transaction column.
pub const DELETED_BY_COLUMN_NAME: &str = "deleted_by_txn";

/// Metadata key for storing the field ID.
pub const FIELD_ID_META_KEY: &str = "field_id";
