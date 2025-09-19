//! Centralized, extensible indexing for the ColumnStore.

pub mod presence;
pub mod traits;

use crate::error::{Error, Result};
use crate::storage::pager::Pager;
use crate::store::descriptor::ChunkMetadata;
use crate::types::LogicalFieldId;
use arrow::array::ArrayRef;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use traits::{Index, IndexManager, IndexOpContext};

/// A concrete implementation of `IndexManager`.
pub struct DefaultIndexManager {
    /// Maps an index name to an instance of that index.
    indexes: FxHashMap<String, Box<dyn Index>>,
    /// Maps a field_id to the list of index names active on it.
    field_indexes: FxHashMap<LogicalFieldId, Vec<String>>,
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

    /// Activates an index for a specific field.
    pub fn enable_index_for_field(
        &mut self,
        field_id: LogicalFieldId,
        index_name: &str,
    ) -> Result<()> {
        if !self.indexes.contains_key(index_name) {
            return Err(Error::Internal(format!(
                "Index '{}' not registered",
                index_name
            )));
        }
        self.field_indexes
            .entry(field_id)
            .or_default()
            .push(index_name.to_string());
        Ok(())
    }
}

impl IndexManager for DefaultIndexManager {
    fn register_index(&mut self, index: Box<dyn Index>) {
        self.indexes.insert(index.name().to_string(), index);
    }

    fn dispatch_append(
        &mut self,
        context: &mut IndexOpContext<'_>,
        field_id: LogicalFieldId,
        new_chunks: &[(ChunkMetadata, ArrayRef)],
    ) -> Result<()> {
        if let Some(active_indexes) = self.field_indexes.get(&field_id) {
            for index_name in active_indexes {
                if let Some(index) = self.indexes.get_mut(index_name) {
                    index.on_append(context, field_id, new_chunks)?;
                }
            }
        }
        Ok(())
    }

    fn dispatch_rewrite(
        &mut self,
        context: &mut IndexOpContext<'_>,
        field_id: LogicalFieldId,
        rewritten_chunks: &[(ChunkMetadata, ArrayRef)],
    ) -> Result<()> {
        if let Some(active_indexes) = self.field_indexes.get(&field_id) {
            for index_name in active_indexes {
                if let Some(index) = self.indexes.get_mut(index_name) {
                    index.on_rewrite(context, field_id, rewritten_chunks)?;
                }
            }
        }
        Ok(())
    }

    fn dispatch_compact(
        &mut self,
        context: &mut IndexOpContext<'_>,
        field_id: LogicalFieldId,
        new_metas: &[ChunkMetadata],
    ) -> Result<()> {
        if let Some(active_indexes) = self.field_indexes.get(&field_id) {
            for index_name in active_indexes {
                if let Some(index) = self.indexes.get_mut(index_name) {
                    index.on_compact(context, field_id, new_metas)?;
                }
            }
        }
        Ok(())
    }
}
