//! Constraint service that centralises runtime-facing constraint validation helpers.
//! Currently focuses on foreign key enforcement for INSERT operations and will
//! gradually expand to cover additional constraint workflows.

#![forbid(unsafe_code)]

use crate::catalog::TableCatalog;
use crate::constraint_validation::validate_foreign_key_rows;
use crate::metadata::{ForeignKeyDetail, MetadataManager};
use crate::types::{FieldId, TableId};
use llkv_plan::PlanValue;
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

/// Callback payload describing what parent rows need to be fetched for validation.
pub struct ForeignKeyRowFetch<'a> {
    pub referenced_table_id: TableId,
    pub referenced_table_canonical: &'a str,
    pub referenced_field_ids: &'a [FieldId],
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
            .foreign_key_details(self.catalog.as_ref(), referencing_table_id)?;

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
                None,
                &detail.referencing_table_display,
                &detail.referenced_table_display,
                &detail.referenced_column_names,
                &parent_keys,
                &candidate_keys,
            )?;
        }

        Ok(())
    }
}

fn build_field_lookup(schema_field_ids: &[FieldId]) -> FxHashMap<FieldId, usize> {
    let mut lookup = FxHashMap::default();
    for (idx, field_id) in schema_field_ids.iter().copied().enumerate() {
        lookup.insert(field_id, idx);
    }
    lookup
}

fn referencing_row_positions(
    detail: &ForeignKeyDetail,
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
    detail: &ForeignKeyDetail,
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
