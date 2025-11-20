//! Core ParquetStore implementation.

use crate::catalog::{ParquetCatalog, ParquetFileRef};
use crate::types::TableId;
use crate::writer::WriterConfig;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use llkv_result::{Error, Result};
use llkv_storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use llkv_storage::types::PhysicalKey;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLock};

/// Reserved pager key for storing the catalog blob.
/// Uses the same key as the main LLKV catalog to avoid conflicts.
const PARQUET_CATALOG_ROOT_KEY: PhysicalKey = 0;

/// Main interface for Parquet-based columnar storage.
///
/// `ParquetStore` manages collections of Parquet files stored as blobs
/// within the pager. It provides table creation, data appends, and scans
/// with transaction visibility.
pub struct ParquetStore<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pager: Arc<P>,
    catalog: Arc<RwLock<ParquetCatalog>>,
    writer_config: WriterConfig,
}

impl<P> Clone for ParquetStore<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            pager: Arc::clone(&self.pager),
            catalog: Arc::clone(&self.catalog),
            writer_config: self.writer_config.clone(),
        }
    }
}

impl<P> ParquetStore<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Open or create a ParquetStore using the provided pager with default configuration.
    ///
    /// Loads the catalog from the pager's reserved key, or initializes
    /// an empty catalog if none exists.
    pub fn open(pager: Arc<P>) -> Result<Self> {
        Self::open_with_config(pager, WriterConfig::default())
    }

    /// Open or create a ParquetStore with custom writer configuration.
    ///
    /// Loads the catalog from the pager's reserved key, or initializes
    /// an empty catalog if none exists.
    pub fn open_with_config(pager: Arc<P>, writer_config: WriterConfig) -> Result<Self> {
        let catalog = match pager
            .batch_get(&[BatchGet::Raw {
                key: PARQUET_CATALOG_ROOT_KEY,
            }])?
            .pop()
        {
            Some(GetResult::Raw { bytes, .. }) => ParquetCatalog::from_bytes(bytes.as_ref())?,
            _ => {
                // Initialize empty catalog and reserve its key by writing it immediately
                let catalog = ParquetCatalog::default();
                let bytes = catalog.to_bytes()?;
                pager.batch_put(&[BatchPut::Raw {
                    key: PARQUET_CATALOG_ROOT_KEY,
                    bytes: bytes.into(),
                }])?;
                catalog
            }
        };

        Ok(Self {
            pager,
            catalog: Arc::new(RwLock::new(catalog)),
            writer_config,
        })
    }

    /// Get the current writer configuration.
    pub fn writer_config(&self) -> &WriterConfig {
        &self.writer_config
    }

    /// Set a new writer configuration.
    ///
    /// This affects all subsequent `append_many()` operations.
    pub fn set_writer_config(&mut self, config: WriterConfig) {
        self.writer_config = config;
    }

    /// Persist the catalog to the pager.
    fn save_catalog(&self) -> Result<()> {
        let catalog = self.catalog.read().unwrap();
        let bytes = catalog.to_bytes()?;

        // Store at reserved catalog key
        self.pager.batch_put(&[BatchPut::Raw {
            key: PARQUET_CATALOG_ROOT_KEY,
            bytes: bytes.into(),
        }])?;

        Ok(())
    }

    /// Create a new table in the catalog.
    ///
    /// Returns the assigned `TableId`.
    pub fn create_table(&self, name: impl Into<String>, schema: SchemaRef) -> Result<TableId> {
        let name = name.into();
        let table_id = {
            let mut catalog = self.catalog.write().unwrap();
            catalog.create_table(name, schema)?
        };

        self.save_catalog()?;
        Ok(table_id)
    }

    /// Append multiple RecordBatches to a table in a single transaction.
    ///
    /// This is more efficient than calling `append()` multiple times because
    /// it only saves the catalog once at the end. Parquet encoding is parallelized
    /// across batches using Rayon.
    pub fn append_many(&self, table_id: TableId, batches: Vec<RecordBatch>) -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }

        // Allocate all keys upfront
        let file_ids = self.pager.alloc_many(batches.len())?;

        // Parallelize Parquet encoding across all batches using Rayon
        use rayon::prelude::*;
        let results: Vec<_> = batches
            .par_iter()
            .zip(file_ids.par_iter())
            .map(|(batch, file_id)| {
                // Write Parquet to memory using configured compression
                let parquet_bytes =
                    crate::writer::write_parquet_to_memory_with_config(batch, &self.writer_config)?;

                // Extract row ID range from batch
                let (min_row_id, max_row_id) = extract_row_id_range(batch)?;

                Ok::<_, Error>((
                    *file_id,
                    parquet_bytes,
                    batch.num_rows(),
                    min_row_id,
                    max_row_id,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        // Build puts and file refs from results
        let mut puts = Vec::with_capacity(results.len());
        let mut file_refs = Vec::with_capacity(results.len());

        for (file_id, parquet_bytes, row_count, min_row_id, max_row_id) in results {
            puts.push(BatchPut::Raw {
                key: file_id,
                bytes: parquet_bytes.into(),
            });

            file_refs.push(ParquetFileRef {
                physical_key: file_id,
                row_count: row_count as u64,
                min_row_id,
                max_row_id,
                column_stats: None, // TODO: Extract from Parquet metadata
            });
        }

        // Batch write all Parquet files
        self.pager.batch_put(&puts)?;

        // Update catalog with all file refs
        {
            let mut catalog = self.catalog.write().unwrap();
            for file_ref in file_refs {
                catalog.add_file_to_table(table_id, file_ref)?;
            }
        }

        // Save catalog once
        self.save_catalog()?;
        Ok(())
    }

    /// Read all Parquet files for a table.
    ///
    /// Returns a vector of RecordBatches. Caller is responsible for
    /// applying MVCC filtering.
    pub fn read_table_files(
        &self,
        table_id: TableId,
        projection: Option<Vec<usize>>,
    ) -> Result<Vec<RecordBatch>> {
        // Get file list from catalog
        let file_keys = {
            let catalog = self.catalog.read().unwrap();
            let (_, metadata) = catalog.get_table_by_id(table_id)?;
            metadata
                .parquet_files
                .iter()
                .map(|f| f.physical_key)
                .collect::<Vec<_>>()
        };

        if file_keys.is_empty() {
            return Ok(Vec::new());
        }

        // Batch fetch from pager
        let gets: Vec<BatchGet> = file_keys.iter().map(|&key| BatchGet::Raw { key }).collect();
        let blobs = self.pager.batch_get(&gets)?;

        // Parse each Parquet blob
        let mut batches = Vec::new();
        for blob in blobs {
            if let GetResult::Raw { bytes, .. } = blob {
                let file_batches =
                    crate::reader::read_parquet_from_memory(bytes.as_ref(), projection.clone())?;
                batches.extend(file_batches);
            }
        }

        // Apply LWW deduplication: keep only latest version of each row_id
        crate::mvcc::deduplicate_by_row_id(batches)
    }

    /// Scan a table with transaction visibility filtering.
    ///
    /// Only returns rows visible to the given transaction ID.
    pub fn scan_visible(
        &self,
        table_id: TableId,
        txn_id: u64,
        projection: Option<Vec<usize>>,
    ) -> Result<Vec<RecordBatch>> {
        let batches = self.read_table_files(table_id, projection)?;

        batches
            .into_iter()
            .map(|batch| crate::mvcc::apply_mvcc_filter(batch, txn_id))
            .collect()
    }

    /// List all tables in the catalog.
    pub fn list_tables(&self) -> Vec<String> {
        let catalog = self.catalog.read().unwrap();
        catalog.list_tables().into_iter().cloned().collect()
    }

    /// Drop a table and free all associated Parquet file keys.
    ///
    /// This removes the table from the catalog and calls `free_many` on the
    /// pager to deallocate all Parquet file blobs.
    pub fn drop_table(&self, name: &str) -> Result<()> {
        let metadata = {
            let mut catalog = self.catalog.write().unwrap();
            catalog.drop_table(name)?
        };

        // Free all Parquet file keys from the pager
        let keys: Vec<PhysicalKey> = metadata
            .parquet_files
            .iter()
            .map(|f| f.physical_key)
            .collect();

        if !keys.is_empty() {
            self.pager.free_many(&keys)?;
        }

        self.save_catalog()?;
        Ok(())
    }

    /// Collect all physical keys currently referenced by the catalog.
    ///
    /// Returns a set containing:
    /// - Key 0 (catalog root)
    /// - All Parquet file keys from all tables
    pub fn collect_reachable_keys(&self) -> rustc_hash::FxHashSet<PhysicalKey> {
        let catalog = self.catalog.read().unwrap();
        crate::gc::collect_reachable_keys(&catalog)
    }

    /// Identify and free all unreferenced blobs.
    ///
    /// This performs garbage collection by:
    /// 1. Enumerating all keys in the pager
    /// 2. Collecting all keys referenced by the catalog
    /// 3. Freeing any keys that exist but are not reachable
    ///
    /// Returns the number of keys freed.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use llkv_parquet_store::ParquetStore;
    /// # use llkv_storage::pager::MemPager;
    /// # use std::sync::Arc;
    /// # fn main() -> llkv_result::Result<()> {
    /// let pager = Arc::new(MemPager::new());
    /// let store = ParquetStore::open(pager)?;
    ///
    /// // ... perform operations, including drop_table() ...
    ///
    /// // Clean up any leaked keys
    /// let freed_count = store.garbage_collect()?;
    /// println!("Freed {} unreferenced blobs", freed_count);
    /// # Ok(())
    /// # }
    /// ```
    pub fn garbage_collect(&self) -> Result<usize> {
        let catalog = self.catalog.read().unwrap();
        crate::gc::garbage_collect(self.pager.as_ref(), &catalog)
    }

    // Note on garbage collection:
    //
    // The garbage_collect() method above performs automatic GC using the
    // pager's enumerate_keys() capability. For manual inspection, you can:
    //   1. Get all keys: pager.enumerate_keys()?
    //   2. Get reachable keys: store.collect_reachable_keys()
    //   3. Find difference: all_keys.difference(&reachable)
    //   4. Free: pager.free_many(&unreferenced)?
}

/// Extract min/max row_id from a RecordBatch.
///
/// Assumes the batch has a "row_id" column of type UInt64.
fn extract_row_id_range(batch: &RecordBatch) -> Result<(u64, u64)> {
    use arrow::array::UInt64Array;
    use arrow::datatypes::DataType;

    let row_id_col = batch
        .column_by_name("row_id")
        .ok_or_else(|| Error::InvalidArgumentError("batch missing row_id column".into()))?;

    if !matches!(row_id_col.data_type(), DataType::UInt64) {
        return Err(Error::InvalidArgumentError(
            "row_id column must be UInt64".into(),
        ));
    }

    let row_ids = row_id_col.as_any().downcast_ref::<UInt64Array>().unwrap();

    if row_ids.is_empty() {
        return Err(Error::InvalidArgumentError("empty batch".into()));
    }

    let min_row_id = (0..row_ids.len()).map(|i| row_ids.value(i)).min().unwrap();

    let max_row_id = (0..row_ids.len()).map(|i| row_ids.value(i)).max().unwrap();

    Ok((min_row_id, max_row_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use llkv_storage::pager::MemPager;

    #[test]
    fn test_create_and_list_tables() {
        let pager = Arc::new(MemPager::new());
        let store = ParquetStore::open(pager).unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("id", DataType::UInt64, false),
        ]));

        store.create_table("test_table", schema).unwrap();

        let tables = store.list_tables();
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0], "test_table");
    }

    #[test]
    fn test_append_and_read() {
        let pager = Arc::new(MemPager::new());
        let store = ParquetStore::open(pager).unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let table_id = store.create_table("users", schema.clone()).unwrap();

        // Create batch with MVCC columns
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
            ],
        )
        .unwrap();

        // Add MVCC columns
        let batch_with_mvcc = crate::mvcc::add_mvcc_columns(batch, 1).unwrap();

        store.append_many(table_id, vec![batch_with_mvcc]).unwrap();

        // Read back
        let batches = store.read_table_files(table_id, None).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);
    }
}
