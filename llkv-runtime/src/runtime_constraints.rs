// TODO: Can all of these be moved to `llkv-table` instead (into constraint validation module, or equiv)?

use rustc_hash::FxHashSet;

use llkv_plan::AlterTableOperation;
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_table::{CatalogManager, ConstraintKind, FieldId, TableId, TableView};
use simd_r_drive_entry_handle::EntryHandle;

pub(crate) fn column_in_primary_or_unique(view: &TableView, field_id: FieldId) -> bool {
    view.constraint_records
        .iter()
        .filter(|record| record.is_active())
        .any(|record| match &record.kind {
            ConstraintKind::PrimaryKey(payload) => payload.field_ids.contains(&field_id),
            ConstraintKind::Unique(payload) => payload.field_ids.contains(&field_id),
            _ => false,
        })
}

pub(crate) fn column_in_multi_column_unique(view: &TableView, field_id: FieldId) -> bool {
    view.multi_column_uniques
        .iter()
        .any(|entry| entry.column_ids.contains(&field_id))
}

pub(crate) fn column_in_foreign_keys<PagerType>(
    view: &TableView,
    field_id: FieldId,
    table_id: TableId,
    catalog_service: &CatalogManager<PagerType>,
) -> Result<Option<String>>
where
    PagerType: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    if let Some(fk) = view
        .foreign_keys
        .iter()
        .find(|fk| fk.referencing_field_ids.contains(&field_id))
    {
        return Ok(Some(
            fk.constraint_name
                .as_deref()
                .unwrap_or("unnamed")
                .to_string(),
        ));
    }

    let mut visited: FxHashSet<TableId> = FxHashSet::default();
    for (referencing_table_id, _) in catalog_service.foreign_keys_referencing(table_id)? {
        if !visited.insert(referencing_table_id) {
            continue;
        }

        for fk in catalog_service.foreign_key_views_for_table(referencing_table_id)? {
            if fk.referenced_table_id == table_id && fk.referenced_field_ids.contains(&field_id) {
                return Ok(Some(
                    fk.constraint_name
                        .as_deref()
                        .unwrap_or("unnamed")
                        .to_string(),
                ));
            }
        }
    }

    Ok(None)
}

pub(crate) fn validate_alter_table_operation<PagerType>(
    operation: &AlterTableOperation,
    view: &TableView,
    table_id: TableId,
    catalog_service: &CatalogManager<PagerType>,
) -> Result<()>
where
    PagerType: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    let resolver = catalog_service
        .field_resolver(table_id)
        .ok_or_else(|| Error::Internal("missing field resolver for table".into()))?;

    match operation {
        llkv_plan::AlterTableOperation::RenameColumn {
            old_column_name,
            new_column_name,
        } => {
            let field_id = resolver.field_id(old_column_name).ok_or_else(|| {
                Error::CatalogError(format!(
                    "Catalog Error: column '{}' does not exist",
                    old_column_name
                ))
            })?;

            if resolver.field_id(new_column_name).is_some() {
                return Err(Error::CatalogError(format!(
                    "Catalog Error: column '{}' already exists",
                    new_column_name
                )));
            }

            if let Some(constraint) =
                column_in_foreign_keys(view, field_id, table_id, catalog_service)?
            {
                return Err(Error::CatalogError(format!(
                    "Catalog Error: column '{}' is involved in the foreign key constraint '{}'",
                    old_column_name, constraint
                )));
            }

            Ok(())
        }
        llkv_plan::AlterTableOperation::SetColumnDataType { column_name, .. } => {
            let field_id = resolver.field_id(column_name).ok_or_else(|| {
                Error::CatalogError(format!(
                    "Catalog Error: column '{}' does not exist",
                    column_name
                ))
            })?;

            if column_in_primary_or_unique(view, field_id)
                || column_in_multi_column_unique(view, field_id)
            {
                return Err(Error::InvalidArgumentError(format!(
                    "Binder Error: Cannot change the type of a column that has a UNIQUE or PRIMARY KEY constraint specified (column '{}')",
                    column_name
                )));
            }

            if let Some(constraint) =
                column_in_foreign_keys(view, field_id, table_id, catalog_service)?
            {
                return Err(Error::CatalogError(format!(
                    "Catalog Error: column '{}' is involved in the foreign key constraint '{}'",
                    column_name, constraint
                )));
            }

            Ok(())
        }
        llkv_plan::AlterTableOperation::DropColumn {
            column_name,
            if_exists,
            ..
        } => {
            let field_id = match resolver.field_id(column_name) {
                Some(id) => id,
                None if *if_exists => return Ok(()),
                None => {
                    return Err(Error::CatalogError(format!(
                        "Catalog Error: column '{}' does not exist",
                        column_name
                    )));
                }
            };

            if column_in_primary_or_unique(view, field_id)
                || column_in_multi_column_unique(view, field_id)
            {
                return Err(Error::CatalogError(format!(
                    "Catalog Error: there is a UNIQUE constraint that depends on it (column '{}')",
                    column_name
                )));
            }

            if column_in_foreign_keys(view, field_id, table_id, catalog_service)?.is_some() {
                return Err(Error::CatalogError(format!(
                    "Catalog Error: there is a FOREIGN KEY constraint that depends on it (column '{}')",
                    column_name
                )));
            }

            Ok(())
        }
    }
}
