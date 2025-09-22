//!
//! Centralized, extensible indexing for the ColumnStore.

pub mod presence;
pub mod sort;
pub mod traits;

use crate::error::{Error, Result};
use crate::types::LogicalFieldId;
use arrow::array::ArrayRef;
use rustc_hash::FxHashMap;
use traits::{BackfillContext, Index, IndexPlan};

/// A concrete implementation of an index manager that plans work synchronously.
pub struct DefaultIndexManager {
    /// Maps an index name to an instance of that index.
    indexes: FxHashMap<String, Box<dyn Index>>,
    /// Maps a field_id to the list of index names active on it.
    pub(crate) field_indexes: FxHashMap<LogicalFieldId, Vec<String>>,
}

impl Default for DefaultIndexManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultIndexManager {
    pub fn new() -> Self {
        Self {
            indexes: FxHashMap::default(),
            field_indexes: FxHashMap::default(),
        }
    }

    /// Registers a new index type.
    pub fn register_index(&mut self, index: Box<dyn Index>) {
        self.indexes.insert(index.name().to_string(), index);
    }

    /// Activates an index for a specific field.
    pub fn enable_index_for_field(&mut self, field_id: LogicalFieldId, index_name: &str) {
        // Avoid duplicates
        let entry = self.field_indexes.entry(field_id).or_default();
        if !entry.iter().any(|n| n == index_name) {
            entry.push(index_name.to_string());
        }
    }

    /// For a given field and new data, returns a list of plans for all active indexes.
    /// The plans contain the raw bytes and metadata for the new index segments.
    pub fn plan_append_for_field(
        &self,
        field_id: LogicalFieldId,
        data_for_chunk: &ArrayRef,
        rids_for_chunk: &ArrayRef,
    ) -> Result<Vec<(&str, IndexPlan)>> {
        let mut plans = Vec::new();
        if let Some(active_indexes) = self.field_indexes.get(&field_id) {
            for index_name in active_indexes {
                if let Some(index) = self.indexes.get(index_name.as_str()) {
                    if let Some(plan) = index.plan_append(data_for_chunk, rids_for_chunk)? {
                        plans.push((index.name(), plan));
                    }
                }
            }
        }
        Ok(plans)
    }

    /// Dispatches a backfill operation to a specific index implementation.
    pub fn backfill_index(
        &self,
        context: &mut BackfillContext<'_>,
        field_id: LogicalFieldId,
        index_name: &str,
    ) -> Result<()> {
        if let Some(index) = self.indexes.get(index_name) {
            index.backfill(context, field_id)
        } else {
            Err(Error::InvalidArgumentError(format!(
                "Index '{}' not registered",
                index_name
            )))
        }
    }

    pub fn registered_indexes(&self) -> impl Iterator<Item = &Box<dyn Index>> {
        self.indexes.values()
    }
}
