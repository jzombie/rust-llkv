//! Shared metadata manager that consolidates catalog I/O for tables, columns, and constraints.
//!
//! This module offers a single entry point for querying and mutating persisted metadata. It keeps
//! an in-memory snapshot per table, performs diff-aware persistence, and always uses batch catalog
//! operations to minimise I/O.

#![forbid(unsafe_code)]

use crate::catalog::TableCatalog;
use crate::constraints::{
    ConstraintId, ConstraintKind, ConstraintRecord, ConstraintState, ForeignKeyAction,
    ForeignKeyConstraint, PrimaryKeyConstraint, UniqueConstraint,
};
use crate::constraints::{ForeignKeyTableInfo, ValidatedForeignKey, validate_foreign_keys};
use crate::reserved;
use crate::resolvers::resolve_table_name;
use crate::sys_catalog::{ConstraintNameRecord, SysCatalog};
use crate::sys_catalog::{MultiColumnIndexEntryMeta, SingleColumnIndexEntryMeta, TriggerEntryMeta};
use crate::table::Table;
use crate::types::{FieldId, TableColumn, TableId};
use crate::view::{ForeignKeyView, TableView};
use crate::{ColMeta, TableMeta, TableMultiColumnIndexMeta};
use arrow::datatypes::DataType;
use llkv_column_map::ColumnStore;
use llkv_column_map::store::IndexKind;
use llkv_column_map::types::LogicalFieldId;
use llkv_plan::ForeignKeySpec;
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLock};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SingleColumnIndexEntry {
    pub index_name: String,
    pub canonical_name: String,
    pub column_id: FieldId,
    pub column_name: String,
    pub unique: bool,
    pub ascending: bool,
    pub nulls_first: bool,
}

#[derive(Clone, Debug, Default)]
struct TableSnapshot {
    table_meta: Option<TableMeta>,
    column_metas: FxHashMap<FieldId, ColMeta>,
    constraints: FxHashMap<ConstraintId, ConstraintRecord>,
    constraint_names: FxHashMap<ConstraintId, String>,
    single_indexes: FxHashMap<String, SingleColumnIndexEntry>,
    multi_column_indexes: FxHashMap<String, MultiColumnIndexEntryMeta>,
    sort_indexes: FxHashSet<FieldId>,
    triggers: FxHashMap<String, TriggerEntryMeta>,
}

impl TableSnapshot {
    fn new(
        table_meta: Option<TableMeta>,
        column_metas: FxHashMap<FieldId, ColMeta>,
        constraints: FxHashMap<ConstraintId, ConstraintRecord>,
        constraint_names: FxHashMap<ConstraintId, String>,
        single_indexes: FxHashMap<String, SingleColumnIndexEntry>,
        multi_column_indexes: FxHashMap<String, MultiColumnIndexEntryMeta>,
        sort_indexes: FxHashSet<FieldId>,
        triggers: FxHashMap<String, TriggerEntryMeta>,
    ) -> Self {
        Self {
            table_meta,
            column_metas,
            constraints,
            constraint_names,
            single_indexes,
            multi_column_indexes,
            sort_indexes,
            triggers,
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
        let constraint_ids: Vec<ConstraintId> = constraint_records
            .iter()
            .map(|record| record.constraint_id)
            .collect();
        let constraint_name_entries = if constraint_ids.is_empty() {
            Vec::new()
        } else {
            catalog.get_constraint_names(table_id, &constraint_ids)?
        };
        let multi_uniques = catalog.get_multi_column_indexes(table_id)?;
        let single_index_metas = catalog.get_single_column_indexes(table_id)?;
        let multi_column_index_metas = catalog.get_multi_column_indexes(table_id)?;
        let trigger_metas = catalog.get_triggers(table_id)?;
        let mut constraints = FxHashMap::default();
        let mut constraint_names = FxHashMap::default();
        let mut single_indexes = FxHashMap::default();
        let mut multi_column_indexes = FxHashMap::default();
        let mut sort_indexes = FxHashSet::default();
        for meta in single_index_metas {
            sort_indexes.insert(meta.column_id);
            single_indexes.insert(
                meta.canonical_name.clone(),
                SingleColumnIndexEntry {
                    index_name: meta.index_name,
                    canonical_name: meta.canonical_name,
                    column_id: meta.column_id,
                    column_name: meta.column_name,
                    unique: meta.unique,
                    ascending: meta.ascending,
                    nulls_first: meta.nulls_first,
                },
            );
        }
        for meta in multi_uniques {
            multi_column_indexes.insert(meta.canonical_name.clone(), meta);
        }
        for meta in multi_column_index_metas {
            multi_column_indexes.insert(meta.canonical_name.clone(), meta);
        }
        let mut triggers = FxHashMap::default();
        for meta in trigger_metas {
            triggers.insert(meta.canonical_name.clone(), meta);
        }
        for (record, name) in constraint_records
            .into_iter()
            .zip(constraint_name_entries.into_iter())
        {
            if let Some(name) = name {
                constraint_names.insert(record.constraint_id, name);
            }
            constraints.insert(record.constraint_id, record);
        }
        let snapshot = TableSnapshot::new(
            table_meta,
            FxHashMap::default(),
            constraints,
            constraint_names,
            single_indexes,
            multi_column_indexes,
            sort_indexes,
            triggers,
        );
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

    fn constraint_name_for(
        &self,
        table_id: TableId,
        constraint_id: ConstraintId,
    ) -> LlkvResult<Option<String>> {
        self.ensure_table_state(table_id)?;
        let tables = self.tables.read().unwrap();
        let state = tables.get(&table_id).unwrap();
        Ok(state.current.constraint_names.get(&constraint_id).cloned())
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

    /// Return the multi-column UNIQUE definitions cached for the table.
    pub fn multi_column_uniques(
        &self,
        table_id: TableId,
    ) -> LlkvResult<Vec<MultiColumnIndexEntryMeta>> {
        self.ensure_table_state(table_id)?;
        let tables = self.tables.read().unwrap();
        let state = tables.get(&table_id).unwrap();
        Ok(state
            .current
            .multi_column_indexes
            .values()
            .filter(|entry| entry.unique)
            .cloned()
            .collect())
    }

    /// Replace the cached multi-column UNIQUE definitions for the table.
    pub fn set_multi_column_uniques(
        &self,
        table_id: TableId,
        uniques: Vec<MultiColumnIndexEntryMeta>,
    ) -> LlkvResult<()> {
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        // Remove all existing unique multi-column indexes
        state
            .current
            .multi_column_indexes
            .retain(|_, entry| !entry.unique);
        // Add the new unique indexes
        for unique in uniques {
            state
                .current
                .multi_column_indexes
                .insert(unique.canonical_name.clone(), unique);
        }
        Ok(())
    }

    /// Return the named single-column indexes registered for a table.
    pub fn single_column_indexes(
        &self,
        table_id: TableId,
    ) -> LlkvResult<Vec<SingleColumnIndexEntry>> {
        self.ensure_table_state(table_id)?;
        let tables = self.tables.read().unwrap();
        let state = tables.get(&table_id).unwrap();
        Ok(state.current.single_indexes.values().cloned().collect())
    }

    /// Lookup a single-column index by canonical name.
    pub fn single_column_index(
        &self,
        table_id: TableId,
        canonical_index_name: &str,
    ) -> LlkvResult<Option<SingleColumnIndexEntry>> {
        self.ensure_table_state(table_id)?;
        let tables = self.tables.read().unwrap();
        let state = tables.get(&table_id).unwrap();
        Ok(state
            .current
            .single_indexes
            .get(canonical_index_name)
            .cloned())
    }

    /// Register or replace a single-column index metadata entry in the cached snapshot.
    pub fn put_single_column_index(
        &self,
        table_id: TableId,
        entry: SingleColumnIndexEntry,
    ) -> LlkvResult<()> {
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        let column_id = entry.column_id;
        state
            .current
            .single_indexes
            .insert(entry.canonical_name.clone(), entry);
        state.current.sort_indexes.insert(column_id);
        Ok(())
    }

    /// Remove a single-column index metadata entry from the cached snapshot.
    pub fn remove_single_column_index(
        &self,
        table_id: TableId,
        canonical_index_name: &str,
    ) -> LlkvResult<Option<SingleColumnIndexEntry>> {
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        let removed = state.current.single_indexes.remove(canonical_index_name);
        if let Some(ref entry) = removed {
            let still_indexed = state
                .current
                .single_indexes
                .values()
                .any(|existing| existing.column_id == entry.column_id);
            if !still_indexed {
                state.current.sort_indexes.remove(&entry.column_id);
            }
        }
        Ok(removed)
    }

    /// Register or replace a multi-column index metadata entry in the cached snapshot.
    pub fn put_multi_column_index(
        &self,
        table_id: TableId,
        entry: MultiColumnIndexEntryMeta,
    ) -> LlkvResult<()> {
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        state
            .current
            .multi_column_indexes
            .insert(entry.canonical_name.clone(), entry);
        Ok(())
    }

    /// Remove a multi-column index metadata entry from the cached snapshot.
    pub fn remove_multi_column_index(
        &self,
        table_id: TableId,
        canonical_index_name: &str,
    ) -> LlkvResult<Option<MultiColumnIndexEntryMeta>> {
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        Ok(state
            .current
            .multi_column_indexes
            .remove(canonical_index_name))
    }

    /// Retrieve a multi-column index metadata entry by canonical name.
    pub fn get_multi_column_index(
        &self,
        table_id: TableId,
        canonical_index_name: &str,
    ) -> LlkvResult<Option<MultiColumnIndexEntryMeta>> {
        self.ensure_table_state(table_id)?;
        let tables = self.tables.read().unwrap();
        Ok(tables.get(&table_id).and_then(|state| {
            state
                .current
                .multi_column_indexes
                .get(canonical_index_name)
                .cloned()
        }))
    }

    /// Register a sort index for a column at the metadata level, staging the change for the next flush.
    pub fn register_sort_index(&self, table_id: TableId, field_id: FieldId) -> LlkvResult<()> {
        self.ensure_table_state(table_id)?;

        {
            let mut tables = self.tables.write().unwrap();
            let state = tables.get_mut(&table_id).unwrap();
            if state.persisted.sort_indexes.contains(&field_id)
                || state.current.sort_indexes.contains(&field_id)
            {
                state.current.sort_indexes.insert(field_id);
                return Ok(());
            }
        }

        if self.field_has_sort_index(table_id, field_id)? {
            let mut tables = self.tables.write().unwrap();
            let state = tables.get_mut(&table_id).unwrap();
            state.persisted.sort_indexes.insert(field_id);
            state.current.sort_indexes.insert(field_id);
            return Ok(());
        }

        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        state.current.sort_indexes.insert(field_id);
        Ok(())
    }

    /// Unregister a sort index for a column, staging removal for the next flush.
    pub fn unregister_sort_index(&self, table_id: TableId, field_id: FieldId) -> LlkvResult<()> {
        self.ensure_table_state(table_id)?;

        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        state.current.sort_indexes.remove(&field_id);

        if !state.persisted.sort_indexes.contains(&field_id) {
            drop(tables);
            if self.field_has_sort_index(table_id, field_id)? {
                let mut tables = self.tables.write().unwrap();
                let state = tables.get_mut(&table_id).unwrap();
                state.persisted.sort_indexes.insert(field_id);
            }
        }

        Ok(())
    }

    /// Mutate the cached multi-column UNIQUE definitions for a table in-place.
    pub fn update_multi_column_uniques<F, T>(&self, table_id: TableId, f: F) -> LlkvResult<T>
    where
        F: FnOnce(&mut Vec<MultiColumnIndexEntryMeta>) -> T,
    {
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        let mut uniques: Vec<MultiColumnIndexEntryMeta> = state
            .current
            .multi_column_indexes
            .values()
            .filter(|entry| entry.unique)
            .cloned()
            .collect();
        let result = f(&mut uniques);
        // Remove all existing unique multi-column indexes
        state
            .current
            .multi_column_indexes
            .retain(|_, entry| !entry.unique);
        // Add back the modified unique indexes
        for unique in uniques {
            state
                .current
                .multi_column_indexes
                .insert(unique.canonical_name.clone(), unique);
        }
        Ok(result)
    }

    /// Return the trigger definitions cached for the table.
    pub fn triggers(&self, table_id: TableId) -> LlkvResult<Vec<TriggerEntryMeta>> {
        self.ensure_table_state(table_id)?;
        let tables = self.tables.read().unwrap();
        let state = tables.get(&table_id).unwrap();
        Ok(state.current.triggers.values().cloned().collect())
    }

    /// Fetch a trigger definition by its canonical name.
    pub fn trigger(
        &self,
        table_id: TableId,
        canonical_name: &str,
    ) -> LlkvResult<Option<TriggerEntryMeta>> {
        self.ensure_table_state(table_id)?;
        let tables = self.tables.read().unwrap();
        let state = tables.get(&table_id).unwrap();
        Ok(state
            .current
            .triggers
            .get(&canonical_name.to_ascii_lowercase())
            .cloned())
    }

    /// Insert or replace a trigger definition in the cached snapshot.
    pub fn insert_trigger(&self, table_id: TableId, trigger: TriggerEntryMeta) -> LlkvResult<()> {
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        state
            .current
            .triggers
            .insert(trigger.canonical_name.clone(), trigger);
        Ok(())
    }

    /// Remove a trigger definition by canonical name. Returns true when the trigger existed.
    pub fn remove_trigger(&self, table_id: TableId, canonical_name: &str) -> LlkvResult<bool> {
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        let state = tables.get_mut(&table_id).unwrap();
        Ok(state
            .current
            .triggers
            .remove(&canonical_name.to_ascii_lowercase())
            .is_some())
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
            state.current.constraint_names.clear();
            state.current.single_indexes.clear();
            state.current.multi_column_indexes.clear();
            state.current.sort_indexes.clear();
            state.current.triggers.clear();
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

    /// Upsert constraint names in the in-memory snapshot.
    pub fn put_constraint_names(
        &self,
        table_id: TableId,
        names: &[(ConstraintId, Option<String>)],
    ) -> LlkvResult<()> {
        if names.is_empty() {
            return Ok(());
        }
        self.ensure_table_state(table_id)?;
        let mut tables = self.tables.write().unwrap();
        if let Some(state) = tables.get_mut(&table_id) {
            for (constraint_id, name) in names {
                if let Some(name) = name {
                    state
                        .current
                        .constraint_names
                        .insert(*constraint_id, name.clone());
                } else {
                    state.current.constraint_names.remove(constraint_id);
                }
            }
        }
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

        let mut dirty_constraint_names: Vec<(ConstraintId, String)> = Vec::new();
        for (constraint_id, name) in &state.current.constraint_names {
            match state.persisted.constraint_names.get(constraint_id) {
                Some(existing) if existing == name => {}
                _ => dirty_constraint_names.push((*constraint_id, name.clone())),
            }
        }
        if !dirty_constraint_names.is_empty() {
            let records: Vec<ConstraintNameRecord> = dirty_constraint_names
                .iter()
                .map(|(constraint_id, name)| ConstraintNameRecord {
                    constraint_id: *constraint_id,
                    name: Some(name.clone()),
                })
                .collect();
            catalog.put_constraint_names(table_id, &records)?;
            for (constraint_id, name) in dirty_constraint_names {
                state.persisted.constraint_names.insert(constraint_id, name);
            }
        }

        let removed_constraint_names: Vec<ConstraintId> = state
            .persisted
            .constraint_names
            .keys()
            .copied()
            .filter(|constraint_id| !state.current.constraint_names.contains_key(constraint_id))
            .collect();
        if !removed_constraint_names.is_empty() {
            catalog.delete_constraint_names(table_id, &removed_constraint_names)?;
            for constraint_id in removed_constraint_names {
                state.persisted.constraint_names.remove(&constraint_id);
            }
        }

        // Flush all multi-column indexes (both unique and non-unique) to the catalog
        if state.current.multi_column_indexes != state.persisted.multi_column_indexes {
            if state.current.multi_column_indexes.is_empty() {
                catalog.delete_multi_column_indexes(table_id)?;
            } else {
                let mut entries: Vec<MultiColumnIndexEntryMeta> = state
                    .current
                    .multi_column_indexes
                    .values()
                    .cloned()
                    .collect();
                entries.sort_by(|a, b| a.canonical_name.cmp(&b.canonical_name));
                catalog.put_multi_column_indexes(table_id, &entries)?;
            }
            state.persisted.multi_column_indexes = state.current.multi_column_indexes.clone();
        }

        if state.current.single_indexes != state.persisted.single_indexes {
            if state.current.single_indexes.is_empty() {
                catalog.delete_single_column_indexes(table_id)?;
                state.persisted.single_indexes.clear();
            } else {
                let mut entries: Vec<SingleColumnIndexEntryMeta> = state
                    .current
                    .single_indexes
                    .values()
                    .cloned()
                    .map(|entry| SingleColumnIndexEntryMeta {
                        index_name: entry.index_name,
                        canonical_name: entry.canonical_name,
                        column_id: entry.column_id,
                        column_name: entry.column_name,
                        unique: entry.unique,
                        ascending: entry.ascending,
                        nulls_first: entry.nulls_first,
                    })
                    .collect();
                entries.sort_by(|a, b| a.canonical_name.cmp(&b.canonical_name));
                catalog.put_single_column_indexes(table_id, &entries)?;
                state.persisted.single_indexes = state.current.single_indexes.clone();
            }
        }

        if state.current.triggers != state.persisted.triggers {
            if state.current.triggers.is_empty() {
                if !state.persisted.triggers.is_empty() {
                    catalog.delete_triggers(table_id)?;
                }
                state.persisted.triggers.clear();
            } else {
                let mut entries: Vec<TriggerEntryMeta> =
                    state.current.triggers.values().cloned().collect();
                entries.sort_by(|a, b| a.canonical_name.cmp(&b.canonical_name));
                catalog.put_triggers(table_id, &entries)?;
                state.persisted.triggers = state.current.triggers.clone();
            }
        }

        let sort_adds: Vec<FieldId> = state
            .current
            .sort_indexes
            .iter()
            .copied()
            .filter(|field_id| !state.persisted.sort_indexes.contains(field_id))
            .collect();
        let sort_removes: Vec<FieldId> = state
            .persisted
            .sort_indexes
            .iter()
            .copied()
            .filter(|field_id| !state.current.sort_indexes.contains(field_id))
            .collect();
        if !sort_adds.is_empty() || !sort_removes.is_empty() {
            let table = Table::from_id_and_store(table_id, Arc::clone(&self.store))?;
            for field_id in &sort_adds {
                table.register_sort_index(*field_id)?;
                state.persisted.sort_indexes.insert(*field_id);
            }
            for field_id in &sort_removes {
                table.unregister_sort_index(*field_id)?;
                state.persisted.sort_indexes.remove(field_id);
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
    pub fn all_multi_column_unique_metas(&self) -> LlkvResult<Vec<TableMultiColumnIndexMeta>> {
        let catalog = SysCatalog::new(&self.store);
        let all = catalog.all_multi_column_index_metas()?;
        // Filter to unique indexes only
        Ok(all
            .into_iter()
            .map(|mut meta| {
                meta.indexes.retain(|idx| idx.unique);
                meta
            })
            .filter(|meta| !meta.indexes.is_empty())
            .collect())
    }

    /// Assemble foreign key descriptors for the table using cached metadata.
    pub fn foreign_key_descriptors(
        &self,
        table_id: TableId,
    ) -> LlkvResult<Vec<ForeignKeyDescriptor>> {
        let records = self.constraint_records(table_id)?;
        let mut descriptors = Vec::new();

        for record in records {
            if !record.is_active() {
                continue;
            }

            let ConstraintKind::ForeignKey(fk) = record.kind else {
                continue;
            };

            descriptors.push(ForeignKeyDescriptor {
                constraint_id: record.constraint_id,
                referencing_table_id: table_id,
                referencing_field_ids: fk.referencing_field_ids.clone(),
                referenced_table_id: fk.referenced_table,
                referenced_field_ids: fk.referenced_field_ids.clone(),
                on_delete: fk.on_delete,
                on_update: fk.on_update,
            });
        }

        Ok(descriptors)
    }

    /// Resolve foreign key descriptors into names suitable for runtime consumers.
    pub fn foreign_key_views(
        &self,
        catalog: &TableCatalog,
        table_id: TableId,
    ) -> LlkvResult<Vec<ForeignKeyView>> {
        let descriptors = self.foreign_key_descriptors(table_id)?;

        if descriptors.is_empty() {
            return Ok(Vec::new());
        }

        let (referencing_display, referencing_canonical) =
            resolve_table_name(catalog, self, table_id)?;

        let mut details = Vec::with_capacity(descriptors.len());
        for descriptor in descriptors {
            let referenced_table_id = descriptor.referenced_table_id;
            let (referenced_display, referenced_canonical) =
                resolve_table_name(catalog, self, referenced_table_id)?;

            let referencing_column_names =
                self.column_names(table_id, &descriptor.referencing_field_ids)?;
            let referenced_column_names =
                self.column_names(referenced_table_id, &descriptor.referenced_field_ids)?;
            let constraint_name = self.constraint_name_for(table_id, descriptor.constraint_id)?;

            details.push(ForeignKeyView {
                constraint_id: descriptor.constraint_id,
                constraint_name,
                referencing_table_id: descriptor.referencing_table_id,
                referencing_table_display: referencing_display.clone(),
                referencing_table_canonical: referencing_canonical.clone(),
                referencing_field_ids: descriptor.referencing_field_ids.clone(),
                referencing_column_names,
                referenced_table_id,
                referenced_table_display: referenced_display.clone(),
                referenced_table_canonical: referenced_canonical.clone(),
                referenced_field_ids: descriptor.referenced_field_ids.clone(),
                referenced_column_names,
                on_delete: descriptor.on_delete,
                on_update: descriptor.on_update,
            });
        }

        Ok(details)
    }

    /// Assemble a consolidated read-only view of table metadata.
    pub fn table_view(
        &self,
        catalog: &TableCatalog,
        table_id: TableId,
        field_ids: &[FieldId],
    ) -> LlkvResult<TableView> {
        let table_meta = self.table_meta(table_id)?;
        let column_metas = self.column_metas(table_id, field_ids)?;
        let constraint_records = self.constraint_records(table_id)?;
        let multi_column_uniques = self.multi_column_uniques(table_id)?;
        let foreign_keys = self.foreign_key_views(catalog, table_id)?;

        Ok(TableView {
            table_meta,
            column_metas,
            constraint_records,
            multi_column_uniques,
            foreign_keys,
        })
    }

    /// Validate foreign key specifications and persist them for the referencing table.
    pub fn validate_and_register_foreign_keys<F>(
        &self,
        referencing_table: &ForeignKeyTableInfo,
        specs: &[ForeignKeySpec],
        lookup_table: F,
        timestamp_micros: u64,
    ) -> LlkvResult<Vec<ValidatedForeignKey>>
    where
        F: FnMut(&str) -> LlkvResult<ForeignKeyTableInfo>,
    {
        let validated = validate_foreign_keys(referencing_table, specs, lookup_table)?;
        self.register_foreign_keys(referencing_table.table_id, &validated, timestamp_micros)?;
        Ok(validated)
    }

    /// Register validated foreign key definitions for a table.
    pub fn register_foreign_keys(
        &self,
        table_id: TableId,
        foreign_keys: &[ValidatedForeignKey],
        timestamp_micros: u64,
    ) -> LlkvResult<()> {
        if foreign_keys.is_empty() {
            return Ok(());
        }

        let existing_constraints = self.constraint_record_map(table_id)?;
        let mut next_constraint_id = existing_constraints
            .keys()
            .copied()
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        let mut constraint_records = Vec::with_capacity(foreign_keys.len());
        let mut constraint_names: Vec<(ConstraintId, Option<String>)> =
            Vec::with_capacity(foreign_keys.len());

        for fk in foreign_keys {
            let constraint_id = next_constraint_id;
            constraint_records.push(ConstraintRecord {
                constraint_id,
                kind: ConstraintKind::ForeignKey(ForeignKeyConstraint {
                    referencing_field_ids: fk.referencing_field_ids.clone(),
                    referenced_table: fk.referenced_table_id,
                    referenced_field_ids: fk.referenced_field_ids.clone(),
                    on_delete: fk.on_delete,
                    on_update: fk.on_update,
                }),
                state: ConstraintState::Active,
                revision: 1,
                last_modified_micros: timestamp_micros,
            });
            constraint_names.push((constraint_id, fk.name.clone()));
            next_constraint_id = next_constraint_id.saturating_add(1);
        }

        self.put_constraint_records(table_id, &constraint_records)?;
        self.put_constraint_names(table_id, &constraint_names)?;
        self.flush_table(table_id)?;

        Ok(())
    }

    /// Register column metadata, physical storage columns, and primary/unique constraints.
    pub fn apply_column_definitions(
        &self,
        table_id: TableId,
        columns: &[TableColumn],
        timestamp_micros: u64,
    ) -> LlkvResult<()> {
        if columns.is_empty() {
            return Ok(());
        }

        self.ensure_table_state(table_id)?;

        for column in columns {
            let column_meta = ColMeta {
                col_id: column.field_id,
                name: Some(column.name.clone()),
                flags: 0,
                default: None,
            };
            self.set_column_meta(table_id, column_meta)?;
        }

        let table = Table::from_id_and_store(table_id, Arc::clone(&self.store))?;
        let store = table.store();

        for column in columns {
            let logical_field_id = LogicalFieldId::for_user(table_id, column.field_id);
            store.ensure_column_registered(logical_field_id, &column.data_type)?;
            store.data_type(logical_field_id)?;
        }

        let created_by_lfid = LogicalFieldId::for_mvcc_created_by(table_id);
        store.ensure_column_registered(created_by_lfid, &DataType::UInt64)?;

        let deleted_by_lfid = LogicalFieldId::for_mvcc_deleted_by(table_id);
        store.ensure_column_registered(deleted_by_lfid, &DataType::UInt64)?;

        let existing = self.constraint_record_map(table_id)?;
        let mut next_constraint_id = existing
            .keys()
            .copied()
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        let mut constraints = Vec::new();

        let primary_key_fields: Vec<FieldId> = columns
            .iter()
            .filter(|col| col.primary_key)
            .map(|col| col.field_id)
            .collect();
        if !primary_key_fields.is_empty() {
            constraints.push(ConstraintRecord {
                constraint_id: next_constraint_id,
                kind: ConstraintKind::PrimaryKey(PrimaryKeyConstraint {
                    field_ids: primary_key_fields,
                }),
                state: ConstraintState::Active,
                revision: 1,
                last_modified_micros: timestamp_micros,
            });
            next_constraint_id = next_constraint_id.saturating_add(1);
        }

        for column in columns.iter().filter(|col| col.unique && !col.primary_key) {
            constraints.push(ConstraintRecord {
                constraint_id: next_constraint_id,
                kind: ConstraintKind::Unique(UniqueConstraint {
                    field_ids: vec![column.field_id],
                }),
                state: ConstraintState::Active,
                revision: 1,
                last_modified_micros: timestamp_micros,
            });
            next_constraint_id = next_constraint_id.saturating_add(1);
        }

        if !constraints.is_empty() {
            self.put_constraint_records(table_id, &constraints)?;
        }

        Ok(())
    }

    pub fn column_data_type(&self, table_id: TableId, field_id: FieldId) -> LlkvResult<DataType> {
        let table = Table::from_id_and_store(table_id, Arc::clone(&self.store))?;
        let store = table.store();
        let logical_field_id = LogicalFieldId::for_user(table_id, field_id);
        store.data_type(logical_field_id)
    }

    /// Register a multi-column UNIQUE definition for a table.
    pub fn register_multi_column_unique(
        &self,
        table_id: TableId,
        column_ids: &[FieldId],
        index_name: Option<String>,
    ) -> LlkvResult<MultiColumnUniqueRegistration> {
        let mut created = false;
        let mut existing_name: Option<Option<String>> = None;
        let column_vec: Vec<FieldId> = column_ids.to_vec();

        // Generate canonical name from column IDs
        let canonical_name = format!(
            "__unique_{}_{}",
            table_id,
            column_vec
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join("_")
        );

        self.update_multi_column_uniques(table_id, |entries| {
            if let Some(existing) = entries.iter().find(|entry| entry.column_ids == column_vec) {
                existing_name = Some(existing.index_name.clone());
            } else {
                entries.push(MultiColumnIndexEntryMeta {
                    index_name: index_name.clone(),
                    canonical_name: canonical_name.clone(),
                    column_ids: column_vec.clone(),
                    unique: true,
                });
                created = true;
            }
        })?;

        if created {
            Ok(MultiColumnUniqueRegistration::Created)
        } else {
            Ok(MultiColumnUniqueRegistration::AlreadyExists {
                index_name: existing_name.unwrap_or(None),
            })
        }
    }

    fn column_names(&self, table_id: TableId, field_ids: &[FieldId]) -> LlkvResult<Vec<String>> {
        if field_ids.is_empty() {
            return Ok(Vec::new());
        }

        let metas = self.column_metas(table_id, field_ids)?;
        let mut names = Vec::with_capacity(field_ids.len());
        for (idx, field_id) in field_ids.iter().enumerate() {
            let name = metas
                .get(idx)
                .and_then(|meta| meta.as_ref())
                .and_then(|meta| meta.name.clone())
                .unwrap_or_else(|| format!("col_{}", field_id));
            names.push(name);
        }
        Ok(names)
    }

    /// Reserve and return the next available table id.
    pub fn reserve_table_id(&self) -> LlkvResult<TableId> {
        let catalog = SysCatalog::new(&self.store);

        let mut next = match catalog.get_next_table_id()? {
            Some(value) => value,
            None => {
                let seed = catalog
                    .max_table_id()?
                    .unwrap_or(reserved::CATALOG_TABLE_ID);
                let initial = seed.checked_add(1).ok_or_else(|| {
                    Error::InvalidArgumentError("exhausted available table ids".into())
                })?;
                catalog.put_next_table_id(initial)?;
                initial
            }
        };

        while reserved::is_reserved_table_id(next) {
            next = next.checked_add(1).ok_or_else(|| {
                Error::InvalidArgumentError("exhausted available table ids".into())
            })?;
        }

        let mut following = next
            .checked_add(1)
            .ok_or_else(|| Error::InvalidArgumentError("exhausted available table ids".into()))?;

        while reserved::is_reserved_table_id(following) {
            following = following.checked_add(1).ok_or_else(|| {
                Error::InvalidArgumentError("exhausted available table ids".into())
            })?;
        }

        catalog.put_next_table_id(following)?;
        Ok(next)
    }

    /// Ensure the catalog's next_table_id counter is at least `minimum`.
    ///
    /// This is primarily used by in-memory namespaces (e.g., temporary tables)
    /// that share a catalog handle with persistent storage. By seeding their
    /// own metadata store with a higher starting point we avoid collisions on
    /// table IDs already allocated by the persistent catalog.
    pub fn ensure_next_table_id_at_least(&self, minimum: TableId) -> LlkvResult<()> {
        if reserved::is_reserved_table_id(minimum) {
            return Err(Error::InvalidArgumentError(
                reserved::reserved_table_id_message(minimum),
            ));
        }

        let catalog = SysCatalog::new(&self.store);
        match catalog.get_next_table_id()? {
            Some(current) if current >= minimum => Ok(()),
            _ => catalog.put_next_table_id(minimum),
        }
    }

    /// Check if a field has a sort index in the underlying store.
    ///
    /// Note: Creates a temporary Table instance to access index metadata.
    /// This is acceptable since Table::from_id_and_store is lightweight (just wraps
    /// table_id + `Arc<ColumnStore>`) and this method is only called during index
    /// registration/unregistration, not in query hot paths.
    fn field_has_sort_index(&self, table_id: TableId, field_id: FieldId) -> LlkvResult<bool> {
        let table = Table::from_id_and_store(table_id, Arc::clone(&self.store))?;
        let indexes = table.list_registered_indexes(field_id)?;
        Ok(indexes.contains(&IndexKind::Sort))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{ConstraintKind, ConstraintState, PrimaryKeyConstraint};
    use crate::{MultiColumnIndexEntryMeta, Table};
    use llkv_column_map::ColumnStore;
    use llkv_column_map::store::IndexKind;
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
            view_definition: None,
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

        let logical_field_id =
            llkv_column_map::types::LogicalFieldId::for_user(table_id, column_meta.col_id);
        store
            .ensure_column_registered(logical_field_id, &arrow::datatypes::DataType::Utf8)
            .unwrap();

        manager
            .register_sort_index(table_id, column_meta.col_id)
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

        let multi_unique = MultiColumnIndexEntryMeta {
            index_name: Some("uniq_users_name".into()),
            canonical_name: "uniq_users_name".to_lowercase(),
            column_ids: vec![column_meta.col_id],
            unique: true,
        };
        manager
            .set_multi_column_uniques(table_id, vec![multi_unique.clone()])
            .unwrap();

        assert_eq!(
            manager.table_meta(table_id).unwrap(),
            Some(table_meta.clone())
        );

        manager.flush_table(table_id).unwrap();

        let table = Table::from_id_and_store(table_id, Arc::clone(&store)).unwrap();
        let indexes = table.list_registered_indexes(column_meta.col_id).unwrap();
        assert!(indexes.contains(&IndexKind::Sort));

        let verify_catalog = SysCatalog::new(&store);
        let column_roundtrip = verify_catalog.get_cols_meta(table_id, &[column_meta.col_id]);
        assert_eq!(column_roundtrip[0].as_ref(), Some(&column_meta));
        let constraints = verify_catalog
            .constraint_records_for_table(table_id)
            .unwrap();
        assert_eq!(constraints, vec![constraint.clone()]);
        let unique_roundtrip = verify_catalog.get_multi_column_indexes(table_id).unwrap();
        assert_eq!(unique_roundtrip, vec![multi_unique.clone()]);

        let meta_from_cache = manager.table_meta(table_id).unwrap();
        assert_eq!(meta_from_cache, Some(table_meta.clone()));

        let columns_from_cache = manager
            .column_metas(table_id, &[column_meta.col_id])
            .unwrap();
        assert_eq!(columns_from_cache[0].as_ref(), Some(&column_meta));

        let constraints_from_cache = manager.constraint_records(table_id).unwrap();
        assert_eq!(constraints_from_cache, vec![constraint.clone()]);

        let uniques_from_cache = manager.multi_column_uniques(table_id).unwrap();
        assert_eq!(uniques_from_cache, vec![multi_unique]);

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
        let multi_unique = MultiColumnIndexEntryMeta {
            index_name: Some("uniq_value".into()),
            canonical_name: "uniq_value".to_lowercase(),
            column_ids: vec![column_meta.col_id],
            unique: true,
        };
        initial_catalog
            .put_multi_column_indexes(table_id, std::slice::from_ref(&multi_unique))
            .unwrap();

        let columns = manager
            .column_metas(table_id, &[column_meta.col_id])
            .unwrap();
        assert_eq!(columns[0].as_ref(), Some(&column_meta));

        let constraints = manager.constraint_records(table_id).unwrap();
        assert_eq!(constraints, vec![constraint]);

        let uniques = manager.multi_column_uniques(table_id).unwrap();
        assert_eq!(uniques, vec![multi_unique]);
    }
}

/// Descriptor describing a foreign key constraint scoped to field identifiers.
#[derive(Clone, Debug)]
pub struct ForeignKeyDescriptor {
    pub constraint_id: ConstraintId,
    pub referencing_table_id: TableId,
    pub referencing_field_ids: Vec<FieldId>,
    pub referenced_table_id: TableId,
    pub referenced_field_ids: Vec<FieldId>,
    pub on_delete: ForeignKeyAction,
    pub on_update: ForeignKeyAction,
}

/// Result of attempting to register a multi-column unique definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiColumnUniqueRegistration {
    Created,
    AlreadyExists { index_name: Option<String> },
}
