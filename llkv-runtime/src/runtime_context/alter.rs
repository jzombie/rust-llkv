//! ALTER TABLE operations for RuntimeContext.
//!
//! This module contains all ALTER TABLE operations:
//! - RENAME COLUMN
//! - ALTER COLUMN TYPE
//! - DROP COLUMN

use crate::canonical_table_name;
use llkv_executor::translation::sql_type_to_arrow;
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use super::RuntimeContext;

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Renames a column in a table by delegating to the catalog layer.
    pub fn rename_column(
        &self,
        table_name: &str,
        old_column_name: &str,
        new_column_name: &str,
    ) -> Result<()> {
        let (_, canonical_table) = canonical_table_name(table_name)?;

        // Get table ID
        let table_id = self
            .catalog
            .table_id(&canonical_table)
            .ok_or_else(|| Error::CatalogError(format!("table '{}' not found", table_name)))?;

        // Delegate to catalog layer
        self.catalog_service
            .rename_column(table_id, old_column_name, new_column_name)?;

        // Invalidate cached executor table so subsequent operations see updated schema.
        self.remove_table_entry(&canonical_table);

        Ok(())
    }

    /// Alters the data type of a column by delegating to the catalog layer.
    pub fn alter_column_type(
        &self,
        table_name: &str,
        column_name: &str,
        new_data_type_sql: &str,
    ) -> Result<()> {
        let (_, canonical_table) = canonical_table_name(table_name)?;

        // Get table ID
        let table_id = self
            .catalog
            .table_id(&canonical_table)
            .ok_or_else(|| Error::CatalogError(format!("table '{}' not found", table_name)))?;

        // Parse SQL type string to Arrow DataType
        let arrow_type = sql_type_to_arrow(new_data_type_sql)?;

        // Delegate to catalog layer
        self.catalog_service
            .alter_column_type(table_id, column_name, &arrow_type)?;

        // Invalidate cached executor table so subsequent operations see updated schema.
        self.remove_table_entry(&canonical_table);

        Ok(())
    }

    /// Drops a column from a table by delegating to the catalog layer.
    pub fn drop_column(&self, table_name: &str, column_name: &str) -> Result<()> {
        let (_, canonical_table) = canonical_table_name(table_name)?;

        // Get table ID
        let table_id = self
            .catalog
            .table_id(&canonical_table)
            .ok_or_else(|| Error::CatalogError(format!("table '{}' not found", table_name)))?;

        // Delegate to catalog layer
        self.catalog_service.drop_column(table_id, column_name)?;

        // Invalidate cached executor table so subsequent operations see updated schema.
        self.remove_table_entry(&canonical_table);

        Ok(())
    }
}
