//! Constraint service that centralises runtime-facing constraint validation helpers.
//! Currently focuses on foreign key enforcement for INSERT operations and will
//! gradually expand to cover additional constraint workflows.

#![forbid(unsafe_code)]

use super::types::ForeignKeyAction;
use super::validation::validate_foreign_key_rows;
use super::validation::{
    ConstraintColumnInfo, UniqueKey, build_composite_unique_key, ensure_multi_column_unique,
    ensure_primary_key, ensure_single_column_unique, validate_check_constraints,
};
use crate::catalog::TableCatalog;
use crate::metadata::MetadataManager;
use crate::types::{FieldId, RowId, TableId};
use crate::view::ForeignKeyView;
use llkv_plan::PlanValue;
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

/// Column metadata required to validate NOT NULL and CHECK constraints during inserts.
#[derive(Clone, Debug)]
pub struct InsertColumnConstraint {
    pub schema_index: usize,
    pub column: ConstraintColumnInfo,
}

/// Descriptor for a single-column UNIQUE constraint.
#[derive(Clone, Debug)]
pub struct InsertUniqueColumn {
    pub schema_index: usize,
    pub field_id: FieldId,
    pub name: String,
}

/// Descriptor for composite UNIQUE or PRIMARY KEY constraints.
#[derive(Clone, Debug)]
pub struct InsertMultiColumnUnique {
    pub schema_indices: Vec<usize>,
    pub field_ids: Vec<FieldId>,
    pub column_names: Vec<String>,
}

/// Callback payload describing what parent rows need to be fetched for validation.
pub struct ForeignKeyRowFetch<'a> {
    pub referenced_table_id: TableId,
    pub referenced_table_canonical: &'a str,
    pub referenced_field_ids: &'a [FieldId],
}

/// Context for collecting parent row values involved in a DELETE operation.
pub struct ForeignKeyParentRowsFetch<'a> {
    pub referenced_table_id: TableId,
    pub referenced_row_ids: &'a [RowId],
    pub referenced_field_ids: &'a [FieldId],
}

/// Context for fetching visible child rows that might reference deleted parents.
pub struct ForeignKeyChildRowsFetch<'a> {
    pub referencing_table_id: TableId,
    pub referencing_table_canonical: &'a str,
    pub referencing_field_ids: &'a [FieldId],
}

/// High-level constraint service API intended for runtime consumers.
#[derive(Clone)]
pub struct ConstraintService<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    metadata: Arc<MetadataManager<P>>,
    catalog: Arc<TableCatalog>,
}

impl<P> ConstraintService<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Create a new constraint validation service.
    pub fn new(metadata: Arc<MetadataManager<P>>, catalog: Arc<TableCatalog>) -> Self {
        Self { metadata, catalog }
    }

    /// Validate that incoming INSERT rows satisfy the table's foreign key constraints.
    pub fn validate_insert_foreign_keys<F>(
        &self,
        referencing_table_id: TableId,
        schema_field_ids: &[FieldId],
        column_order: &[usize],
        rows: &[Vec<PlanValue>],
        mut fetch_parent_rows: F,
    ) -> LlkvResult<()>
    where
        F: FnMut(ForeignKeyRowFetch<'_>) -> LlkvResult<Vec<Vec<PlanValue>>>,
    {
        if rows.is_empty() {
            return Ok(());
        }

        let details = self
            .metadata
            .foreign_key_views(self.catalog.as_ref(), referencing_table_id)?;

        if details.is_empty() {
            return Ok(());
        }

        let field_lookup = build_field_lookup(schema_field_ids);
        let mut table_to_row_index: Vec<Option<usize>> = vec![None; schema_field_ids.len()];
        for (row_pos, &schema_idx) in column_order.iter().enumerate() {
            if let Some(slot) = table_to_row_index.get_mut(schema_idx) {
                *slot = Some(row_pos);
            }
        }

        for detail in &details {
            if detail.referencing_field_ids.is_empty() {
                continue;
            }

            let referencing_positions = referencing_row_positions(
                detail,
                &field_lookup,
                &table_to_row_index,
                referencing_table_id,
            )?;

            let parent_rows = fetch_parent_rows(ForeignKeyRowFetch {
                referenced_table_id: detail.referenced_table_id,
                referenced_table_canonical: &detail.referenced_table_canonical,
                referenced_field_ids: &detail.referenced_field_ids,
            })?;

            let parent_keys = canonical_parent_keys(detail, parent_rows);
            let candidate_keys = candidate_child_keys(&referencing_positions, rows)?;

            validate_foreign_key_rows(
                detail.constraint_name.as_deref(),
                &detail.referencing_table_display,
                &detail.referenced_table_display,
                &detail.referenced_column_names,
                &parent_keys,
                &candidate_keys,
            )?;
        }

        Ok(())
    }

    /// Validate INSERT rows against all table constraints including primary keys, unique constraints,
    /// and CHECK expressions. This is a comprehensive validation that combines uniqueness checks
    /// (both single-column and multi-column) with row-level CHECK constraint evaluation.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_insert_constraints<FSingle, FMulti>(
        &self,
        schema_field_ids: &[FieldId],
        column_constraints: &[InsertColumnConstraint],
        unique_columns: &[InsertUniqueColumn],
        multi_column_uniques: &[InsertMultiColumnUnique],
        primary_key: Option<&InsertMultiColumnUnique>,
        column_order: &[usize],
        rows: &[Vec<PlanValue>],
        mut fetch_column_values: FSingle,
        mut fetch_multi_column_rows: FMulti,
    ) -> LlkvResult<()>
    where
        FSingle: FnMut(FieldId) -> LlkvResult<Vec<PlanValue>>,
        FMulti: FnMut(&[FieldId]) -> LlkvResult<Vec<Vec<PlanValue>>>,
    {
        if rows.is_empty() {
            return Ok(());
        }

        let schema_to_row_index = build_schema_to_row_index(schema_field_ids.len(), column_order)?;
        validate_row_constraints_with_mapping(
            column_constraints,
            rows,
            &schema_to_row_index,
            column_order,
        )?;

        for unique in unique_columns {
            let Some(row_pos) = schema_to_row_index
                .get(unique.schema_index)
                .and_then(|opt| *opt)
            else {
                continue;
            };

            let existing_values = fetch_column_values(unique.field_id)?;
            let mut new_values: Vec<PlanValue> = Vec::with_capacity(rows.len());
            for row in rows {
                let value = row.get(row_pos).cloned().unwrap_or(PlanValue::Null);
                new_values.push(value);
            }

            ensure_single_column_unique(&existing_values, &new_values, &unique.name)?;
        }

        for constraint in multi_column_uniques {
            if constraint.schema_indices.is_empty() {
                continue;
            }

            let existing_rows = fetch_multi_column_rows(&constraint.field_ids)?;
            let new_rows = collect_row_sets(rows, &schema_to_row_index, &constraint.schema_indices);
            ensure_multi_column_unique(&existing_rows, &new_rows, &constraint.column_names)?;
        }

        if let Some(pk) = primary_key
            && !pk.schema_indices.is_empty()
        {
            let existing_rows = fetch_multi_column_rows(&pk.field_ids)?;
            let new_rows = collect_row_sets(rows, &schema_to_row_index, &pk.schema_indices);
            ensure_primary_key(&existing_rows, &new_rows, &pk.column_names)?;
        }

        Ok(())
    }

    /// Validate rows against CHECK constraints. This method evaluates CHECK expressions
    /// for each row, ensuring they satisfy the table's row-level constraint rules.
    pub fn validate_row_level_constraints(
        &self,
        schema_field_ids: &[FieldId],
        column_constraints: &[InsertColumnConstraint],
        column_order: &[usize],
        rows: &[Vec<PlanValue>],
    ) -> LlkvResult<()> {
        if rows.is_empty() {
            return Ok(());
        }

        let schema_to_row_index = build_schema_to_row_index(schema_field_ids.len(), column_order)?;
        validate_row_constraints_with_mapping(
            column_constraints,
            rows,
            &schema_to_row_index,
            column_order,
        )
    }

    /// Validate that INSERT rows satisfy the primary key constraint by checking for duplicates
    /// against both existing rows in the table and within the new batch.
    pub fn validate_primary_key_rows<F>(
        &self,
        schema_field_ids: &[FieldId],
        primary_key: &InsertMultiColumnUnique,
        column_order: &[usize],
        rows: &[Vec<PlanValue>],
        mut fetch_multi_column_rows: F,
    ) -> LlkvResult<()>
    where
        F: FnMut(&[FieldId]) -> LlkvResult<Vec<Vec<PlanValue>>>,
    {
        if rows.is_empty() || primary_key.schema_indices.is_empty() {
            return Ok(());
        }

        let schema_to_row_index = build_schema_to_row_index(schema_field_ids.len(), column_order)?;
        let existing_rows = fetch_multi_column_rows(&primary_key.field_ids)?;
        let new_rows = collect_row_sets(rows, &schema_to_row_index, &primary_key.schema_indices);
        ensure_primary_key(&existing_rows, &new_rows, &primary_key.column_names)
    }

    /// Validate UPDATE operations that modify primary key columns. Ensures that updated
    /// primary key values don't conflict with existing rows (excluding the original row being updated).
    pub fn validate_update_primary_keys<F>(
        &self,
        schema_field_ids: &[FieldId],
        primary_key: &InsertMultiColumnUnique,
        column_order: &[usize],
        rows: &[Vec<PlanValue>],
        original_keys: &[Option<UniqueKey>],
        mut fetch_multi_column_rows: F,
    ) -> LlkvResult<()>
    where
        F: FnMut(&[FieldId]) -> LlkvResult<Vec<Vec<PlanValue>>>,
    {
        if rows.is_empty() || primary_key.schema_indices.is_empty() {
            return Ok(());
        }

        if original_keys.len() != rows.len() {
            return Err(Error::Internal(
                "primary key original value count does not match row count".into(),
            ));
        }

        let schema_to_row_index = build_schema_to_row_index(schema_field_ids.len(), column_order)?;

        let mut existing_rows = fetch_multi_column_rows(&primary_key.field_ids)?;
        let mut existing_keys: FxHashSet<UniqueKey> = FxHashSet::default();
        for row_values in existing_rows.drain(..) {
            if let Some(key) = build_composite_unique_key(&row_values, &primary_key.column_names)? {
                existing_keys.insert(key);
            }
        }

        for key in original_keys.iter().flatten() {
            existing_keys.remove(key);
        }

        let (pk_label, pk_display) = primary_key_context(&primary_key.column_names);
        let mut new_seen: FxHashSet<UniqueKey> = FxHashSet::default();
        let new_row_sets =
            collect_row_sets(rows, &schema_to_row_index, &primary_key.schema_indices);

        for values in new_row_sets {
            let key = build_composite_unique_key(&values, &primary_key.column_names)?;
            let key = key.ok_or_else(|| {
                Error::ConstraintError(format!(
                    "constraint failed: NOT NULL constraint failed for PRIMARY KEY {pk_label} '{pk_display}'"
                ))
            })?;

            if existing_keys.contains(&key) {
                return Err(Error::ConstraintError(format!(
                    "Duplicate key violates primary key constraint on {pk_label} '{}' (PRIMARY KEY or UNIQUE constraint violation)",
                    pk_display
                )));
            }

            if !new_seen.insert(key.clone()) {
                return Err(Error::ConstraintError(format!(
                    "Duplicate key violates primary key constraint on {pk_label} '{}' (PRIMARY KEY or UNIQUE constraint violation)",
                    pk_display
                )));
            }

            existing_keys.insert(key);
        }

        Ok(())
    }

    /// Validate that deleting the given rows will not violate foreign key constraints.
    pub fn validate_delete_foreign_keys<FParents, FChildren>(
        &self,
        referenced_table_id: TableId,
        referenced_row_ids: &[RowId],
        mut fetch_parent_rows: FParents,
        mut fetch_child_rows: FChildren,
    ) -> LlkvResult<()>
    where
        FParents: FnMut(ForeignKeyParentRowsFetch<'_>) -> LlkvResult<Vec<Vec<PlanValue>>>,
        FChildren: FnMut(ForeignKeyChildRowsFetch<'_>) -> LlkvResult<Vec<(RowId, Vec<PlanValue>)>>,
    {
        if referenced_row_ids.is_empty() {
            return Ok(());
        }

        let referencing = self
            .metadata
            .foreign_keys_referencing(referenced_table_id)?;
        if referencing.is_empty() {
            return Ok(());
        }

        let deleting_row_ids: FxHashSet<RowId> = referenced_row_ids.iter().copied().collect();

        for (child_table_id, constraint_id) in referencing {
            let details = self
                .metadata
                .foreign_key_views(self.catalog.as_ref(), child_table_id)?;

            let Some(detail) = details
                .into_iter()
                .find(|detail| detail.constraint_id == constraint_id)
            else {
                continue;
            };

            if detail.referenced_field_ids.is_empty() || detail.referencing_field_ids.is_empty() {
                continue;
            }

            let parent_rows = fetch_parent_rows(ForeignKeyParentRowsFetch {
                referenced_table_id,
                referenced_row_ids,
                referenced_field_ids: &detail.referenced_field_ids,
            })?;

            let parent_keys = canonical_parent_keys(&detail, parent_rows);
            if parent_keys.is_empty() {
                continue;
            }

            let child_rows = fetch_child_rows(ForeignKeyChildRowsFetch {
                referencing_table_id: detail.referencing_table_id,
                referencing_table_canonical: &detail.referencing_table_canonical,
                referencing_field_ids: &detail.referencing_field_ids,
            })?;

            if child_rows.is_empty() {
                continue;
            }

            for (child_row_id, values) in child_rows {
                if values.len() != detail.referencing_field_ids.len() {
                    continue;
                }

                if values.iter().any(|value| matches!(value, PlanValue::Null)) {
                    continue;
                }

                if parent_keys.iter().all(|key| key != &values) {
                    continue;
                }

                if detail.referencing_table_id == detail.referenced_table_id
                    && deleting_row_ids.contains(&child_row_id)
                {
                    continue;
                }

                let constraint_label = detail.constraint_name.as_deref().unwrap_or("FOREIGN KEY");
                match detail.on_delete {
                    ForeignKeyAction::NoAction | ForeignKeyAction::Restrict => {
                        return Err(Error::ConstraintError(format!(
                            "Violates foreign key constraint '{}' on table '{}' referencing '{}' - row is still referenced by a foreign key in a different table",
                            constraint_label,
                            detail.referencing_table_display,
                            detail.referenced_table_display,
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate that updating the given rows will not violate foreign key constraints.
    ///
    /// This checks if any columns being updated are referenced by foreign keys, and whether
    /// the OLD values are still being referenced by child tables.
    pub fn validate_update_foreign_keys<FParents, FChildren>(
        &self,
        referenced_table_id: TableId,
        referenced_row_ids: &[RowId],
        updated_field_ids: &[FieldId],
        mut fetch_parent_rows: FParents,
        mut fetch_child_rows: FChildren,
    ) -> LlkvResult<()>
    where
        FParents: FnMut(ForeignKeyParentRowsFetch<'_>) -> LlkvResult<Vec<Vec<PlanValue>>>,
        FChildren: FnMut(ForeignKeyChildRowsFetch<'_>) -> LlkvResult<Vec<(RowId, Vec<PlanValue>)>>,
    {
        if referenced_row_ids.is_empty() || updated_field_ids.is_empty() {
            return Ok(());
        }

        let referencing = self
            .metadata
            .foreign_keys_referencing(referenced_table_id)?;
        if referencing.is_empty() {
            return Ok(());
        }

        for (child_table_id, constraint_id) in referencing {
            let details = self
                .metadata
                .foreign_key_views(self.catalog.as_ref(), child_table_id)?;

            let Some(detail) = details
                .into_iter()
                .find(|detail| detail.constraint_id == constraint_id)
            else {
                continue;
            };

            if detail.referenced_field_ids.is_empty() || detail.referencing_field_ids.is_empty() {
                continue;
            }

            // Check if any of the columns being updated are part of this foreign key
            let is_referenced_column_updated = detail
                .referenced_field_ids
                .iter()
                .any(|fid| updated_field_ids.contains(fid));

            if !is_referenced_column_updated {
                // This FK doesn't reference any columns being updated, skip
                continue;
            }

            // Fetch the OLD values from the parent table (before update)
            let parent_rows = fetch_parent_rows(ForeignKeyParentRowsFetch {
                referenced_table_id,
                referenced_row_ids,
                referenced_field_ids: &detail.referenced_field_ids,
            })?;

            let parent_keys = canonical_parent_keys(&detail, parent_rows);
            if parent_keys.is_empty() {
                continue;
            }

            // Fetch all rows from child table that reference this parent
            let child_rows = fetch_child_rows(ForeignKeyChildRowsFetch {
                referencing_table_id: detail.referencing_table_id,
                referencing_table_canonical: &detail.referencing_table_canonical,
                referencing_field_ids: &detail.referencing_field_ids,
            })?;

            if child_rows.is_empty() {
                continue;
            }

            // Check if any child rows reference the OLD values
            for (_child_row_id, values) in child_rows {
                if values.len() != detail.referencing_field_ids.len() {
                    continue;
                }

                if values.iter().any(|value| matches!(value, PlanValue::Null)) {
                    continue;
                }

                // If a child row references one of the parent keys being updated, fail
                if parent_keys.iter().any(|key| key == &values) {
                    let constraint_label =
                        detail.constraint_name.as_deref().unwrap_or("FOREIGN KEY");
                    return Err(Error::ConstraintError(format!(
                        "Violates foreign key constraint '{}' on table '{}' referencing '{}' - cannot update referenced column while foreign key exists",
                        constraint_label,
                        detail.referencing_table_display,
                        detail.referenced_table_display,
                    )));
                }
            }
        }

        Ok(())
    }

    /// Return the set of foreign keys referencing the provided table.
    pub fn referencing_foreign_keys(
        &self,
        referenced_table_id: TableId,
    ) -> LlkvResult<Vec<ForeignKeyView>> {
        let referencing = self
            .metadata
            .foreign_keys_referencing(referenced_table_id)?;

        if referencing.is_empty() {
            return Ok(Vec::new());
        }

        let mut details_out = Vec::new();
        for (child_table_id, constraint_id) in referencing {
            let details = match self
                .metadata
                .foreign_key_views(self.catalog.as_ref(), child_table_id)
            {
                Ok(details) => details,
                Err(Error::InvalidArgumentError(_)) | Err(Error::CatalogError(_)) => continue,
                Err(err) => return Err(err),
            };

            if let Some(detail) = details
                .into_iter()
                .find(|detail| detail.constraint_id == constraint_id)
            {
                details_out.push(detail);
            }
        }

        Ok(details_out)
    }
}

fn build_field_lookup(schema_field_ids: &[FieldId]) -> FxHashMap<FieldId, usize> {
    let mut lookup = FxHashMap::default();
    for (idx, field_id) in schema_field_ids.iter().copied().enumerate() {
        lookup.insert(field_id, idx);
    }
    lookup
}

fn validate_row_constraints_with_mapping(
    column_constraints: &[InsertColumnConstraint],
    rows: &[Vec<PlanValue>],
    schema_to_row_index: &[Option<usize>],
    column_order: &[usize],
) -> LlkvResult<()> {
    for constraint in column_constraints {
        if constraint.column.nullable {
            continue;
        }

        let Some(row_pos) = schema_to_row_index
            .get(constraint.schema_index)
            .and_then(|opt| *opt)
        else {
            return Err(Error::ConstraintError(format!(
                "NOT NULL column '{}' missing from INSERT/UPDATE",
                constraint.column.name
            )));
        };

        for row in rows {
            if matches!(row.get(row_pos), Some(PlanValue::Null)) {
                return Err(Error::ConstraintError(format!(
                    "NOT NULL constraint failed for column '{}'",
                    constraint.column.name
                )));
            }
        }
    }

    let check_columns: Vec<ConstraintColumnInfo> = column_constraints
        .iter()
        .map(|constraint| constraint.column.clone())
        .collect();
    validate_check_constraints(check_columns.as_slice(), rows, column_order)?;
    Ok(())
}

fn build_schema_to_row_index(
    schema_len: usize,
    column_order: &[usize],
) -> LlkvResult<Vec<Option<usize>>> {
    let mut schema_to_row_index: Vec<Option<usize>> = vec![None; schema_len];
    for (row_pos, &schema_idx) in column_order.iter().enumerate() {
        if schema_idx >= schema_len {
            return Err(Error::Internal(format!(
                "column index {} out of bounds for schema (len={})",
                schema_idx, schema_len
            )));
        }
        schema_to_row_index[schema_idx] = Some(row_pos);
    }
    Ok(schema_to_row_index)
}

fn primary_key_context(column_names: &[String]) -> (&'static str, String) {
    if column_names.len() == 1 {
        ("column", column_names[0].clone())
    } else {
        ("columns", column_names.join(", "))
    }
}

fn collect_row_sets(
    rows: &[Vec<PlanValue>],
    schema_to_row_index: &[Option<usize>],
    schema_indices: &[usize],
) -> Vec<Vec<PlanValue>> {
    rows.iter()
        .map(|row| {
            schema_indices
                .iter()
                .map(|&schema_idx| {
                    schema_to_row_index
                        .get(schema_idx)
                        .and_then(|opt| {
                            opt.map(|row_pos| row.get(row_pos).cloned().unwrap_or(PlanValue::Null))
                        })
                        .unwrap_or(PlanValue::Null)
                })
                .collect()
        })
        .collect()
}

fn referencing_row_positions(
    detail: &ForeignKeyView,
    lookup: &FxHashMap<FieldId, usize>,
    table_to_row_index: &[Option<usize>],
    table_id: TableId,
) -> LlkvResult<Vec<usize>> {
    let mut positions = Vec::with_capacity(detail.referencing_field_ids.len());

    for (idx, field_id) in detail.referencing_field_ids.iter().cloned().enumerate() {
        let schema_index = lookup.get(&field_id).cloned().ok_or_else(|| {
            Error::Internal(format!(
                "referencing field id {} not found in table '{}' (table_id={})",
                field_id, detail.referencing_table_display, table_id
            ))
        })?;

        let position = table_to_row_index
            .get(schema_index)
            .and_then(|value| *value)
            .ok_or_else(|| {
                let column_name = detail
                    .referencing_column_names
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| schema_index.to_string());
                Error::InvalidArgumentError(format!(
                    "FOREIGN KEY column '{}' missing from INSERT statement",
                    column_name
                ))
            })?;

        positions.push(position);
    }

    Ok(positions)
}

fn canonical_parent_keys(
    detail: &ForeignKeyView,
    parent_rows: Vec<Vec<PlanValue>>,
) -> Vec<Vec<PlanValue>> {
    parent_rows
        .into_iter()
        .filter(|values| values.len() == detail.referenced_field_ids.len())
        .filter(|values| !values.iter().any(|value| matches!(value, PlanValue::Null)))
        .collect()
}

fn candidate_child_keys(
    positions: &[usize],
    rows: &[Vec<PlanValue>],
) -> LlkvResult<Vec<Vec<PlanValue>>> {
    let mut keys = Vec::new();

    for row in rows {
        let mut key: Vec<PlanValue> = Vec::with_capacity(positions.len());
        let mut contains_null = false;

        for &row_pos in positions {
            let value = row.get(row_pos).cloned().ok_or_else(|| {
                Error::InvalidArgumentError("INSERT row is missing a required column value".into())
            })?;

            if matches!(value, PlanValue::Null) {
                contains_null = true;
                break;
            }

            key.push(value);
        }

        if contains_null {
            continue;
        }

        keys.push(key);
    }

    Ok(keys)
}
