//! Shared metadata manager that consolidates catalog I/O for tables, columns, and constraints.
//!
//! This module offers a single entry point for querying and mutating persisted metadata. It keeps
//! an in-memory snapshot per table, performs diff-aware persistence, and always uses batch catalog
//! operations to minimise I/O.

#![forbid(unsafe_code)]

use crate::constraints::{ConstraintId, ConstraintKind, ConstraintRecord};
use crate::sys_catalog::SysCatalog;
use crate::types::{FieldId, TableId};
use crate::{ColMeta, TableMeta, TableMultiColumnUniqueMeta};
use llkv_column_map::ColumnStore;
use llkv_result::Result as LlkvResult;
use llkv_storage::pager::Pager;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLock};

#[derive(Clone, Debug, Default)]
struct TableSnapshot {
    table_meta: Option<TableMeta>,
    column_metas: FxHashMap<FieldId, ColMeta>,
    constraints: FxHashMap<ConstraintId, ConstraintRecord>,
}

impl TableSnapshot {
    fn new(
        table_meta: Option<TableMeta>,
        column_metas: FxHashMap<FieldId, ColMeta>,
        constraints: FxHashMap<ConstraintId, ConstraintRecord>,
    ) -> Self {
        Self {
            table_meta,
            column_metas,
            constraints,
        }
    }
}

#[derive(Clone, Debug)]
struct TableState {
    current: TableSnapshot,
    persisted: TableSnapshot,
}

impl TableState {
    fn from_snapshot(snapshot: TableSnapshot) -> Self {
        Self {
            current: snapshot.clone(),
            persisted: snapshot,
        }
    }
}

#[derive(Default)]
struct ReferencingIndex {
    parent_to_children: FxHashMap<TableId, FxHashSet<(TableId, ConstraintId)>>,
    child_to_parents: FxHashMap<TableId, FxHashSet<TableId>>,
    initialized: bool,
}

impl ReferencingIndex {
    fn remove_child(&mut self, child_id: TableId) {
        if let Some(parents) = self.child_to_parents.remove(&child_id) {
            for parent_id in parents {
                if let Some(children) = self.parent_to_children.get_mut(&parent_id) {
                    children.retain(|(entry_child, _)| *entry_child != child_id);
                    if children.is_empty() {
                        self.parent_to_children.remove(&parent_id);
                    }
                }
            }
        }
    }

    fn insert(&mut self, parent_id: TableId, child_id: TableId, constraint_id: ConstraintId) {
        self.parent_to_children
            .entry(parent_id)
            .or_default()
            .insert((child_id, constraint_id));
        self.child_to_parents
            .entry(child_id)
            .or_default()
            .insert(parent_id);
        self.initialized = true;
    }

    fn children(&self, parent_id: TableId) -> Vec<(TableId, ConstraintId)> {
        self.parent_to_children
            .get(&parent_id)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    fn mark_initialized(&mut self) {
        self.initialized = true;
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Central metadata facade that hides the raw catalog implementation details.
pub struct MetadataManager<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    store: Arc<ColumnStore<P>>,
    tables: RwLock<FxHashMap<TableId, TableState>>,
    referencing_index: RwLock<ReferencingIndex>,
}

impl<P> MetadataManager<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Create a new metadata manager backed by the provided column store.
    pub fn new(store: Arc<ColumnStore<P>>) -> Self {
        Self {
            store,
            tables: RwLock::new(FxHashMap::default()),
            referencing_index: RwLock::new(ReferencingIndex::default()),
        }
    }

    /// Load metadata for a table from the catalog if not already cached.
    fn ensure_table_state(&self, table_id: TableId) -> LlkvResult<()> {
        if self.tables.read().unwrap().contains_key(&table_id) {
            return Ok(());
        }
        let state = self.load_table_state(table_id)?;
        {
            let mut tables = self.tables.write().unwrap();
            tables.entry(table_id).or_insert(state);
        }
        self.refresh_referencing_index_for_table(table_id);
        Ok(())
    }

    fn load_table_state(&self, table_id: TableId) -> LlkvResult<TableState> {
        let catalog = SysCatalog::new(&self.store);
        let table_meta = catalog.get_table_meta(table_id);
        let constraint_records = catalog.constraint_records_for_table(table_id)?;
        let mut constraints = FxHashMap::default();
        for record in constraint_records {
            constraints.insert(record.constraint_id, record);
        }
        let snapshot = TableSnapshot::new(table_meta, FxHashMap::default(), constraints);
        Ok(TableState::from_snapshot(snapshot))
    }

    fn refresh_referencing_index_for_table(&self, table_id: TableId) {
        let foreign_keys: Vec<(TableId, ConstraintId)> = {
            let tables = self.tables.read().unwrap();
            match tables.get(&table_id) {
                Some(state) => state
                    .current
                    .constraints
                    .iter()
                    .filter(|(_, record)| record.is_active())
                    .filter_map(|(constraint_id, record)| {
                        if let ConstraintKind::ForeignKey(fk) = &record.kind {
                            Some((fk.referenced_table, *constraint_id))
                        } else {
                            None
                        }
                    })
                    .collect(),
                None => Vec::new(),
            }
        };

        let mut index = self.referencing_index.write().unwrap();
        index.remove_child(table_id);
        for (parent_table, constraint_id) in foreign_keys {
            index.insert(parent_table, table_id, constraint_id);
        }
    }

    fn ensure_referencing_index_initialized(&self) -> LlkvResult<()> {
        let needs_init = {
            let index = self.referencing_index.read().unwrap();
            !index.is_initialized()
        };

        if !needs_init {
            return Ok(());
        }

        let metas = self.all_table_metas()?;
        for (table_id, _) in metas {
            self.ensure_table_state(table_id)?;
            self.refresh_referencing_index_for_table(table_id);
        }

        let mut index = self.referencing_index.write().unwrap();
        index.mark_initialized();
        Ok(())
    }

    /// Retrieve the current table metadata snapshot (loaded lazily if required).
    pub fn table_meta(&self, table_id: TableId) -> LlkvResult<Option<TableMeta>> {
        self.ensure_table_state(table_id)?;
        let tables = self.tables.read().unwrap();
        Ok(tables
            .get(&table_id)
            .and_then(|state| state.current.table_meta.clone()))
    }

    /// Return the list of child table + constraint identifiers that reference the provided table.
    pub fn foreign_keys_referencing(
        &self,
        referenced_table: TableId,
    ) -> LlkvResult<Vec<(TableId, ConstraintId)>> {
        self.ensure_referencing_index_initialized()?;
        let index = self.referencing_index.read().unwrap();
        Ok(index.children(referenced_table))
    }

    /// Update the in-memory table metadata. Changes are flushed on demand.
    pub fn set_table_meta(&self, table_id: TableId, meta: TableMeta) -> LlkvResult<()> {
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        state.current.table_meta = Some(meta);
        Ok(())
    }

    /// Fetch column metadata for the requested field identifiers, loading missing entries lazily.
    pub fn column_metas(
        &self,
        table_id: TableId,
        field_ids: &[FieldId],
    ) -> LlkvResult<Vec<Option<ColMeta>>> {
        self.ensure_table_state(table_id)?;

        // Determine which columns still need to be loaded from the catalog.
        let missing_ids = {
            let tables = self.tables.read().unwrap();
            let state = tables.get(&table_id).unwrap();
            field_ids
                .iter()
                .copied()
                .filter(|field_id| !state.current.column_metas.contains_key(field_id))
                .collect::<Vec<_>>()
        };

        if !missing_ids.is_empty() {
            let catalog = SysCatalog::new(&self.store);
            let fetched = catalog.get_cols_meta(table_id, &missing_ids);
            let mut tables = self.tables.write().unwrap();
            let state = tables.get_mut(&table_id).unwrap();
            for (idx, field_id) in missing_ids.iter().enumerate() {
                if let Some(meta) = fetched[idx].clone() {
                    state.current.column_metas.insert(*field_id, meta.clone());
                    state
                        .persisted
                        .column_metas
                        .entry(*field_id)
                        .or_insert(meta);
                }
            }
        }

        let tables = self.tables.read().unwrap();
        let state = tables.get(&table_id).unwrap();
        Ok(field_ids
            .iter()
            .map(|field_id| state.current.column_metas.get(field_id).cloned())
            .collect())
    }

    /// Upsert a single column metadata record in the in-memory snapshot.
    pub fn set_column_meta(&self, table_id: TableId, meta: ColMeta) -> LlkvResult<()> {
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        state.current.column_metas.insert(meta.col_id, meta);
        Ok(())
    }

    /// Prepare the metadata state for dropping a table by clearing cached entries.
    ///
    /// Column metadata is loaded eagerly for the provided field identifiers so deletions
    /// are persisted on the next flush.
    pub fn prepare_table_drop(&self, table_id: TableId, column_ids: &[FieldId]) -> LlkvResult<()> {
        if !column_ids.is_empty() {
            let _ = self.column_metas(table_id, column_ids)?;
        } else {
            self.ensure_table_state(table_id)?;
        }

        let mut tables = self.tables.write().unwrap();
        if let Some(state) = tables.get_mut(&table_id) {
            state.current.table_meta = None;
            state.current.column_metas.clear();
            state.current.constraints.clear();
        }
        drop(tables);
        self.refresh_referencing_index_for_table(table_id);
        Ok(())
    }

    /// Remove any cached snapshots for the specified table.
    pub fn remove_table_state(&self, table_id: TableId) {
        self.tables.write().unwrap().remove(&table_id);
        self.referencing_index
            .write()
            .unwrap()
            .remove_child(table_id);
    }

    /// Delete persisted multi-column UNIQUE metadata for a table.
    pub fn delete_multi_column_uniques(&self, table_id: TableId) -> LlkvResult<()> {
        let catalog = SysCatalog::new(&self.store);
        catalog.delete_multi_column_uniques(table_id)?;
        Ok(())
    }

    /// Return all constraint records currently cached for the table.
    pub fn constraint_records(&self, table_id: TableId) -> LlkvResult<Vec<ConstraintRecord>> {
        self.ensure_table_state(table_id)?;
        let tables = self.tables.read().unwrap();
        let state = tables.get(&table_id).unwrap();
        Ok(state.current.constraints.values().cloned().collect())
    }

    /// Fetch a subset of constraint records by their identifiers.
    pub fn constraint_records_by_id(
        &self,
        table_id: TableId,
        constraint_ids: &[ConstraintId],
    ) -> LlkvResult<Vec<Option<ConstraintRecord>>> {
        self.ensure_table_state(table_id)?;
        let tables = self.tables.read().unwrap();
        let state = tables.get(&table_id).unwrap();
        Ok(constraint_ids
            .iter()
            .map(|constraint_id| state.current.constraints.get(constraint_id).cloned())
            .collect())
    }

    /// Upsert constraint records in the in-memory snapshot.
    pub fn put_constraint_records(
        &self,
        table_id: TableId,
        records: &[ConstraintRecord],
    ) -> LlkvResult<()> {
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        for record in records {
            state
                .current
                .constraints
                .insert(record.constraint_id, record.clone());
        }
        drop(tables);
        self.refresh_referencing_index_for_table(table_id);
        Ok(())
    }

    /// Produce a map of constraint records keyed by identifier.
    pub fn constraint_record_map(
        &self,
        table_id: TableId,
    ) -> LlkvResult<FxHashMap<ConstraintId, ConstraintRecord>> {
        self.ensure_table_state(table_id)?;
        let tables = self.tables.read().unwrap();
        let state = tables.get(&table_id).unwrap();
        Ok(state.current.constraints.clone())
    }

    /// Persist changes for a single table to the underlying catalog, writing only the diffs.
    pub fn flush_table(&self, table_id: TableId) -> LlkvResult<()> {
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();

        let catalog = SysCatalog::new(&self.store);

        match (
            state.current.table_meta.as_ref(),
            state.persisted.table_meta.as_ref(),
        ) {
            (Some(meta), Some(existing)) if meta != existing => {
                catalog.put_table_meta(meta);
                state.persisted.table_meta = Some(meta.clone());
            }
            (Some(meta), None) => {
                catalog.put_table_meta(meta);
                state.persisted.table_meta = Some(meta.clone());
            }
            (None, Some(_)) => {
                catalog.delete_table_meta(table_id)?;
                state.persisted.table_meta = None;
            }
            _ => {}
        }

        let mut dirty_columns: Vec<(FieldId, ColMeta)> = Vec::new();
        for (field_id, meta) in &state.current.column_metas {
            match state.persisted.column_metas.get(field_id) {
                Some(existing) if existing == meta => {}
                _ => dirty_columns.push((*field_id, meta.clone())),
            }
        }
        for (field_id, meta) in dirty_columns.iter() {
            catalog.put_col_meta(table_id, meta);
            state.persisted.column_metas.insert(*field_id, meta.clone());
        }

        let removed_columns: Vec<FieldId> = state
            .persisted
            .column_metas
            .keys()
            .copied()
            .filter(|field_id| !state.current.column_metas.contains_key(field_id))
            .collect();
        if !removed_columns.is_empty() {
            catalog.delete_col_meta(table_id, &removed_columns)?;
            for field_id in removed_columns {
                state.persisted.column_metas.remove(&field_id);
            }
        }

        let mut dirty_constraints: Vec<ConstraintRecord> = Vec::new();
        for (constraint_id, record) in &state.current.constraints {
            match state.persisted.constraints.get(constraint_id) {
                Some(existing) if existing == record => {}
                _ => dirty_constraints.push(record.clone()),
            }
        }
        if !dirty_constraints.is_empty() {
            catalog.put_constraint_records(table_id, &dirty_constraints)?;
            for record in dirty_constraints {
                state
                    .persisted
                    .constraints
                    .insert(record.constraint_id, record);
            }
        }

        let removed_constraints: Vec<ConstraintId> = state
            .persisted
            .constraints
            .keys()
            .copied()
            .filter(|constraint_id| !state.current.constraints.contains_key(constraint_id))
            .collect();
        if !removed_constraints.is_empty() {
            catalog.delete_constraint_records(table_id, &removed_constraints)?;
            for constraint_id in removed_constraints {
                state.persisted.constraints.remove(&constraint_id);
            }
        }

        Ok(())
    }

    /// Persist changes for all tracked tables.
    pub fn flush_all(&self) -> LlkvResult<()> {
        let table_ids: Vec<TableId> = {
            let tables = self.tables.read().unwrap();
            tables.keys().copied().collect()
        };
        for table_id in table_ids {
            self.flush_table(table_id)?;
        }
        Ok(())
    }

    /// Return all persisted table metadata.
    pub fn all_table_metas(&self) -> LlkvResult<Vec<(TableId, TableMeta)>> {
        let catalog = SysCatalog::new(&self.store);
        catalog.all_table_metas()
    }

    /// Return all persisted multi-column unique metadata.
    pub fn all_multi_column_unique_metas(&self) -> LlkvResult<Vec<TableMultiColumnUniqueMeta>> {
        let catalog = SysCatalog::new(&self.store);
        catalog.all_multi_column_unique_metas()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{ConstraintKind, ConstraintState, PrimaryKeyConstraint};
    use llkv_column_map::ColumnStore;
    use llkv_storage::pager::MemPager;
    use std::sync::Arc;

    #[test]
    fn metadata_manager_persists_and_loads() {
        let pager = Arc::new(MemPager::default());
        let store = Arc::new(ColumnStore::open(Arc::clone(&pager)).unwrap());
        let manager = MetadataManager::new(Arc::clone(&store));

        let table_id: TableId = 42;
        let table_meta = TableMeta {
            table_id,
            name: Some("users".into()),
            created_at_micros: 123,
            flags: 0,
            epoch: 1,
        };
        manager
            .set_table_meta(table_id, table_meta.clone())
            .unwrap();

        {
            let tables = manager.tables.read().unwrap();
            let state = tables.get(&table_id).unwrap();
            assert!(state.current.table_meta.is_some());
        }

        let column_meta = ColMeta {
            col_id: 1,
            name: Some("id".into()),
            flags: 0,
            default: None,
        };
        manager
            .set_column_meta(table_id, column_meta.clone())
            .unwrap();

        let constraint = ConstraintRecord {
            constraint_id: 7,
            kind: ConstraintKind::PrimaryKey(PrimaryKeyConstraint {
                field_ids: vec![column_meta.col_id],
            }),
            state: ConstraintState::Active,
            revision: 1,
            last_modified_micros: 456,
        };
        manager
            .put_constraint_records(table_id, std::slice::from_ref(&constraint))
            .unwrap();

        assert_eq!(
            manager.table_meta(table_id).unwrap(),
            Some(table_meta.clone())
        );

        manager.flush_table(table_id).unwrap();

        let verify_catalog = SysCatalog::new(&store);
        let column_roundtrip = verify_catalog.get_cols_meta(table_id, &[column_meta.col_id]);
        assert_eq!(column_roundtrip[0].as_ref(), Some(&column_meta));
        let constraints = verify_catalog
            .constraint_records_for_table(table_id)
            .unwrap();
        assert_eq!(constraints, vec![constraint.clone()]);

        let meta_from_cache = manager.table_meta(table_id).unwrap();
        assert_eq!(meta_from_cache, Some(table_meta.clone()));

        let columns_from_cache = manager
            .column_metas(table_id, &[column_meta.col_id])
            .unwrap();
        assert_eq!(columns_from_cache[0].as_ref(), Some(&column_meta));

        let constraints_from_cache = manager.constraint_records(table_id).unwrap();
        assert_eq!(constraints_from_cache, vec![constraint.clone()]);

        // No additional writes should occur on subsequent flushes without modifications.
        manager.flush_table(table_id).unwrap();
    }

    #[test]
    fn metadata_manager_lazy_loads_columns_and_constraints() {
        let pager = Arc::new(MemPager::default());
        let store = Arc::new(ColumnStore::open(Arc::clone(&pager)).unwrap());
        let manager = MetadataManager::new(Arc::clone(&store));

        let table_id: TableId = 99;
        let column_meta = ColMeta {
            col_id: 3,
            name: Some("value".into()),
            flags: 0,
            default: None,
        };
        let initial_catalog = SysCatalog::new(&store);
        initial_catalog.put_col_meta(table_id, &column_meta);

        let constraint = ConstraintRecord {
            constraint_id: 15,
            kind: ConstraintKind::PrimaryKey(PrimaryKeyConstraint {
                field_ids: vec![column_meta.col_id],
            }),
            state: ConstraintState::Active,
            revision: 1,
            last_modified_micros: 0,
        };
        initial_catalog
            .put_constraint_records(table_id, std::slice::from_ref(&constraint))
            .unwrap();

        let columns = manager
            .column_metas(table_id, &[column_meta.col_id])
            .unwrap();
        assert_eq!(columns[0].as_ref(), Some(&column_meta));

        let constraints = manager.constraint_records(table_id).unwrap();
        assert_eq!(constraints, vec![constraint]);
    }
}
