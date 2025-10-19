#![forbid(unsafe_code)]

use std::convert::TryFrom;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use llkv_column_map::ColumnStore;
use llkv_plan::ColumnSpec;
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;

use crate::catalog::{FieldDefinition, TableCatalog};
use crate::metadata::MetadataManager;
use crate::table::Table;
use crate::types::{FieldId, TableColumn, TableId};

/// Result of creating a table from column specifications.
pub struct CreateTableResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub table_id: TableId,
    pub catalog_table_id: TableId,
    pub table: Arc<Table<P>>,
    pub table_columns: Vec<TableColumn>,
    pub column_lookup: FxHashMap<String, usize>,
}

/// High-level catalog service for table lifecycle operations.
#[derive(Clone)]
pub struct CatalogService<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    metadata: Arc<MetadataManager<P>>,
    catalog: Arc<TableCatalog>,
    store: Arc<ColumnStore<P>>,
}

impl<P> CatalogService<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(
        metadata: Arc<MetadataManager<P>>,
        catalog: Arc<TableCatalog>,
        store: Arc<ColumnStore<P>>,
    ) -> Self {
        Self {
            metadata,
            catalog,
            store,
        }
    }

    /// Create a new table using the provided column definitions.
    ///
    /// The metadata manager and catalog are updated, and an empty physical table is created.
    /// The caller is responsible for integrating the returned [`Table`] into higher-level
    /// caches and performing any additional constraint registration (e.g. foreign keys).
    pub fn create_table_from_columns(
        &self,
        display_name: &str,
        canonical_name: &str,
        columns: &[ColumnSpec],
    ) -> LlkvResult<CreateTableResult<P>> {
        if columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires at least one column".into(),
            ));
        }

        let mut column_lookup: FxHashMap<String, usize> =
            FxHashMap::with_capacity_and_hasher(columns.len(), Default::default());
        let mut table_columns: Vec<TableColumn> = Vec::with_capacity(columns.len());

        for (idx, column) in columns.iter().enumerate() {
            let normalized = column.name.to_ascii_lowercase();
            if column_lookup.insert(normalized.clone(), idx).is_some() {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in table '{}'",
                    column.name, display_name
                )));
            }

            table_columns.push(TableColumn {
                field_id: FieldId::try_from(idx + 1).map_err(|_| {
                    Error::Internal(format!(
                        "column index {} exceeded supported field id range",
                        idx + 1
                    ))
                })?,
                name: column.name.clone(),
                data_type: column.data_type.clone(),
                nullable: column.nullable,
                primary_key: column.primary_key,
                unique: column.unique,
                check_expr: column.check_expr.clone(),
            });
        }

        let table_id = self.metadata.reserve_table_id()?;
        let timestamp = current_time_micros();
        let table_meta = crate::sys_catalog::TableMeta {
            table_id,
            name: Some(display_name.to_string()),
            created_at_micros: timestamp,
            flags: 0,
            epoch: 0,
        };

        self.metadata.set_table_meta(table_id, table_meta)?;
        self.metadata
            .apply_column_definitions(table_id, &table_columns, timestamp)?;
        self.metadata.flush_table(table_id)?;

        let table = Table::new_with_store(table_id, Arc::clone(&self.store))?;

        let registered_table_id = match self.catalog.register_table(display_name) {
            Ok(id) => id,
            Err(err) => {
                self.metadata.remove_table_state(table_id);
                return Err(err);
            }
        };

        if let Some(field_resolver) = self.catalog.field_resolver(registered_table_id) {
            for column in &table_columns {
                let definition = FieldDefinition::new(&column.name)
                    .with_primary_key(column.primary_key)
                    .with_unique(column.unique)
                    .with_check_expr(column.check_expr.clone());
                if let Err(err) = field_resolver.register_field(definition) {
                    self.catalog.unregister_table(canonical_name);
                    self.metadata.remove_table_state(table_id);
                    return Err(err);
                }
            }
        }

        Ok(CreateTableResult {
            table_id,
            catalog_table_id: registered_table_id,
            table: Arc::new(table),
            table_columns,
            column_lookup,
        })
    }
}

#[allow(clippy::unnecessary_wraps)]
fn current_time_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_micros() as u64)
        .unwrap_or(0)
}
