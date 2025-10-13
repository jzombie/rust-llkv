/// Canonical Arrow field name for physical row-id columns.
pub const ROW_ID_COLUMN_NAME: &str = "rowid";

/// MVCC: Transaction ID that created this row
pub const CREATED_BY_COLUMN_NAME: &str = "_created_by";

/// MVCC: Transaction ID that deleted this row (or TXN_ID_NONE if not deleted)
pub const DELETED_BY_COLUMN_NAME: &str = "_deleted_by";

/// Metadata key used to store the logical field id on Arrow `Field` metadata.
/// Centralized here so the key is authoritative inside the store module.
pub const FIELD_ID_META_KEY: &str = "field_id";

pub(crate) const DESCRIPTOR_ENTRIES_PER_PAGE: usize = 256;

/// Target byte size for descriptor pages. The pager may be mmapped, so page
/// alignment helps. Used to avoid overgrowing tail pages.
pub(crate) const TARGET_DESCRIPTOR_PAGE_BYTES: usize = 4096;

/// Target size for data chunks coalesced by the bounded compactor and used
/// on the ingest path when slicing big appends into multiple chunks.
pub(crate) const TARGET_CHUNK_BYTES: usize = 1024 * 1024; // ~1 MiB

/// Merge only chunks smaller than this threshold.
pub(crate) const MIN_CHUNK_BYTES: usize = TARGET_CHUNK_BYTES / 2; // ~512 KiB

/// Upper bound on bytes coalesced per run, to cap rewrite cost.
pub(crate) const MAX_MERGE_RUN_BYTES: usize = 16 * 1024 * 1024; // ~16 MiB
