//! The main module for managing column indexes.

use crate::error::{Error, Result};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::ColumnDescriptor;
use crate::types::LogicalFieldId;
use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};

/// A trait for column indexes.
pub trait Index {
    /// The name of the index.
    fn name(&self) -> &'static str;
    // In the future, this trait will include methods for building, querying,
    // and maintaining the index.
}

/// A no-op index used for discovery of legacy sort permutations.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct SortIndex;

impl SortIndex {
    pub fn new() -> Self {
        Self
    }
}

impl Index for SortIndex {
    fn name(&self) -> &'static str {
        "sort"
    }
}

/// An index for fast row presence checks.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct PresenceIndex;

impl PresenceIndex {
    pub fn new() -> Self {
        Self
    }
}

impl Index for PresenceIndex {
    fn name(&self) -> &'static str {
        "presence"
    }
}

/// Manages the registration and discovery of indexes on columns.
pub struct IndexManager<P: Pager> {
    pager: Arc<P>,
    catalog: Arc<RwLock<ColumnCatalog>>,
}

impl<P: Pager> IndexManager<P> {
    /// Creates a new `IndexManager`.
    pub fn new(pager: Arc<P>, catalog: Arc<RwLock<ColumnCatalog>>) -> Self {
        Self { pager, catalog }
    }

    /// Registers an index for a given column.
    pub fn register_index(&self, field_id: LogicalFieldId, index_name: &str) -> Result<()> {
        self.update_indexes(field_id, |indexes| {
            if indexes.contains(&index_name.to_string()) {
                return Err(Error::InvalidArgumentError(format!(
                    "Index '{}' already exists for this column.",
                    index_name
                )));
            }
            indexes.push(index_name.to_string());
            Ok(())
        })
    }

    /// Unregisters an index from a given column.
    pub fn unregister_index(&self, field_id: LogicalFieldId, index_name: &str) -> Result<()> {
        self.update_indexes(field_id, |indexes| {
            let original_len = indexes.len();
            indexes.retain(|name| name != index_name);
            if indexes.len() == original_len {
                return Err(Error::InvalidArgumentError(format!(
                    "Index '{}' not found for this column.",
                    index_name
                )));
            }
            Ok(())
        })
    }

    /// A helper function to load, modify, and save column descriptor indexes.
    fn update_indexes<F>(&self, field_id: LogicalFieldId, mut modifier: F) -> Result<()>
    where
        F: FnMut(&mut Vec<String>) -> Result<()>,
    {
        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;

        let desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;

        let mut descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
        let mut indexes = descriptor.get_indexes()?;
        modifier(&mut indexes)?;
        descriptor.set_indexes(&indexes)?;

        self.pager.batch_put(&[BatchPut::Raw {
            key: descriptor_pk,
            bytes: descriptor.to_le_bytes(),
        }])
    }

    /// Retrieves the names of registered indexes for a set of columns.
    pub fn get_column_indexes(
        &self,
        field_ids: &[LogicalFieldId],
    ) -> Result<FxHashMap<LogicalFieldId, Vec<String>>> {
        let catalog = self.catalog.read().unwrap();
        let mut gets = Vec::new();
        let mut pk_to_fid = FxHashMap::default();

        for &field_id in field_ids {
            if let Some(&pk) = catalog.map.get(&field_id) {
                gets.push(BatchGet::Raw { key: pk });
                pk_to_fid.insert(pk, field_id);
            }
        }

        if gets.is_empty() {
            return Ok(FxHashMap::default());
        }

        let results = self.pager.batch_get(&gets)?;
        let mut all_indexes = FxHashMap::default();

        for result in results {
            if let GetResult::Raw { key, bytes } = result {
                if let Some(&field_id) = pk_to_fid.get(&key) {
                    let descriptor = ColumnDescriptor::from_le_bytes(bytes.as_ref());
                    all_indexes.insert(field_id, descriptor.get_indexes()?);
                }
            }
        }

        Ok(all_indexes)
    }
}
