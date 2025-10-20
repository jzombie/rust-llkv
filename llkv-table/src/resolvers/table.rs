use crate::catalog::TableCatalog;
use crate::metadata::MetadataManager;
use crate::types::TableId;
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

/// Convert a table name into its display (original) and canonical (lowercase) forms.
pub fn canonical_table_name(name: &str) -> Result<(String, String)> {
    if name.is_empty() {
        return Err(Error::InvalidArgumentError(
            "table name must not be empty".into(),
        ));
    }
    let display = name.to_string();
    let canonical = display.to_ascii_lowercase();
    Ok((display, canonical))
}

/// Resolve a table `TableId` into display + canonical names using the catalog/metadata layers.
pub fn resolve_table_name<P>(
    catalog: &TableCatalog,
    metadata: &MetadataManager<P>,
    table_id: TableId,
) -> Result<(String, String)>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    if let Some(qt) = catalog.table_name(table_id) {
        let display = qt.to_display_string();
        let canonical = display.to_ascii_lowercase();
        return Ok((display, canonical));
    }

    if let Some(meta) = metadata.table_meta(table_id)?
        && let Some(name) = meta.name
    {
        let canonical = name.to_ascii_lowercase();
        return Ok((name, canonical));
    }

    let fallback = table_id.to_string();
    Ok((fallback.clone(), fallback))
}
