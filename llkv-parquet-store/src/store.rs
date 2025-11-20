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

                // Extract column statistics from Parquet metadata
                let column_stats = if self.writer_config.enable_statistics {
                    crate::statistics::extract_statistics(&parquet_bytes).ok()
                } else {
                    None
                };

                Ok::<_, Error>((
                    *file_id,
                    parquet_bytes,
                    batch.num_rows(),
                    min_row_id,
                    max_row_id,
                    column_stats,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        // Build puts and file refs from results
        let mut puts = Vec::with_capacity(results.len());
        let mut file_refs = Vec::with_capacity(results.len());

        for (file_id, parquet_bytes, row_count, min_row_id, max_row_id, column_stats) in results {
            puts.push(BatchPut::Raw {
                key: file_id,
                bytes: parquet_bytes.into(),
            });

            file_refs.push(ParquetFileRef {
                physical_key: file_id,
                row_count: row_count as u64,
                min_row_id,
                max_row_id,
                column_stats,
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

    /// Scan a table with optional filtering, projection, and limit.
    ///
    /// This is the primary API for reading data, optimized using:
    /// - **File pruning**: Skips files based on column statistics and DataFusion predicates
    /// - **Column projection**: Only reads requested columns from Parquet
    /// - **Streaming**: Returns an iterator to avoid loading everything into memory
    /// - **Limit pushdown**: Stops reading files early when limit is reached
    /// - **MVCC deduplication**: Applies last-write-wins semantics per file
    ///
    /// # Arguments
    /// - `table_id`: Which table to scan
    /// - `filters`: DataFusion expressions for predicate pushdown
    /// - `projection`: Optional column indices to read (None = all columns)
    /// - `limit`: Optional maximum number of rows to return
    ///
    /// # Example
    /// ```rust,no_run
    /// # use llkv_parquet_store::{ParquetStore, TableId};
    /// # use llkv_storage::pager::MemPager;
    /// # use std::sync::Arc;
    /// # fn main() -> llkv_result::Result<()> {
    /// let pager = Arc::new(MemPager::new());
    /// let store = ParquetStore::open(pager)?;
    /// # let table_id = TableId(0);
    ///
    /// // Scan with filters and limit
    /// for batch in store.scan(table_id, &[], Some(vec![0, 2]), Some(1000))? {
    ///     let batch = batch?;
    ///     println!("Read {} rows", batch.num_rows());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn scan(
        &self,
        table_id: TableId,
        filters: &[datafusion_expr::Expr],
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
    ) -> Result<impl Iterator<Item = Result<RecordBatch>> + '_> {
        // Get file list from catalog and prune based on filters
        let (file_keys, schema) = {
            let catalog = self.catalog.read().unwrap();
            let (_, metadata) = catalog.get_table_by_id(table_id)?;
            let schema = metadata.schema()?;

            let mut file_keys = Vec::new();
            for file_ref in &metadata.parquet_files {
                // Apply file pruning based on column statistics and predicates
                if filters.is_empty() || should_scan_file(file_ref, filters, &schema)? {
                    file_keys.push(file_ref.physical_key);
                }
            }

            (file_keys, schema)
        };

        Ok(ParquetScanIterator {
            store: self,
            file_keys: file_keys.into_iter(),
            projection,
            limit,
            rows_returned: 0,
            _schema: schema,
        })
    }

    // TODO: Make this iterator based
    /// Scan a table with transaction visibility filtering.
    ///
    /// Only returns rows visible to the given transaction ID.
    pub fn scan_visible(
        &self,
        table_id: TableId,
        txn_id: u64,
        projection: Option<Vec<usize>>,
    ) -> Result<Vec<RecordBatch>> {
        let batches = self
            .scan(table_id, &[], projection, None)?
            .collect::<Result<Vec<_>>>()?;

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

/// Streaming iterator for scanning Parquet files one at a time.
///
/// This avoids loading all data into memory by fetching and decoding
/// one file at a time from the pager.
struct ParquetScanIterator<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    store: &'a ParquetStore<P>,
    file_keys: std::vec::IntoIter<PhysicalKey>,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    rows_returned: usize,
    _schema: SchemaRef,
}

impl<'a, P> Iterator for ParquetScanIterator<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've hit the limit
        if let Some(limit) = self.limit {
            if self.rows_returned >= limit {
                return None;
            }
        }

        let key = self.file_keys.next()?;

        // Fetch single file from pager
        let bytes = match self.store.pager.batch_get(&[BatchGet::Raw { key }]) {
            Ok(mut results) => match results.pop() {
                Some(GetResult::Raw { bytes, .. }) => bytes,
                _ => return Some(Err(Error::Internal(format!("file key {} not found", key)))),
            },
            Err(e) => return Some(Err(e)),
        };

        // Decode Parquet file with projection
        match crate::reader::read_parquet_from_memory(bytes.as_ref(), self.projection.clone()) {
            Ok(batches) => {
                // Apply LWW deduplication within this file
                match crate::mvcc::deduplicate_by_row_id(batches) {
                    Ok(deduped) => {
                        // Return first deduplicated batch (typically 1 batch per file)
                        if let Some(batch) = deduped.into_iter().next() {
                            // Apply limit to this batch if needed
                            let batch = if let Some(limit) = self.limit {
                                let remaining = limit.saturating_sub(self.rows_returned);
                                if batch.num_rows() > remaining {
                                    // Slice the batch to only return up to limit
                                    match batch.slice(0, remaining) {
                                        sliced => {
                                            self.rows_returned += sliced.num_rows();
                                            sliced
                                        }
                                    }
                                } else {
                                    self.rows_returned += batch.num_rows();
                                    batch
                                }
                            } else {
                                batch
                            };
                            Some(Ok(batch))
                        } else {
                            None
                        }
                    }
                    Err(e) => Some(Err(e)),
                }
            }
            Err(e) => Some(Err(e)),
        }
    }
}

/// Determine if a file should be scanned based on column statistics and filter predicates.
///
/// This evaluates DataFusion expressions against the file's column statistics to
/// determine if the file could possibly contain matching rows.
fn should_scan_file(
    file_ref: &ParquetFileRef,
    filters: &[datafusion_expr::Expr],
    schema: &Arc<arrow::datatypes::Schema>,
) -> Result<bool> {
    // If no statistics available, must scan conservatively
    let Some(ref column_stats) = file_ref.column_stats else {
        return Ok(true);
    };

    // Evaluate each filter predicate
    for filter in filters {
        if !evaluate_predicate(filter, column_stats, schema)? {
            // Predicate definitively excludes this file
            return Ok(false);
        }
    }

    // All predicates could potentially match
    Ok(true)
}

/// Evaluate a single predicate against column statistics.
///
/// Returns:
/// - `true` if the file might contain matching rows (conservative)
/// - `false` if the file definitely cannot contain matching rows
fn evaluate_predicate(
    expr: &datafusion_expr::Expr,
    stats: &std::collections::HashMap<String, crate::types::ColumnStats>,
    schema: &Arc<arrow::datatypes::Schema>,
) -> Result<bool> {
    use datafusion_expr::{Expr, Operator};

    match expr {
        // Binary comparisons: col op literal (excluding AND/OR)
        Expr::BinaryExpr(binary) if !matches!(binary.op, Operator::And | Operator::Or) => {
            let col_name = match binary.left.as_ref() {
                Expr::Column(col) => col.name.as_str(),
                _ => return Ok(true), // Conservative: can't evaluate
            };

            let literal_value = match binary.right.as_ref() {
                Expr::Literal(val, ..) => val,
                _ => return Ok(true), // Conservative: can't evaluate
            };

            // Get column statistics
            let col_stats = match stats.get(col_name) {
                Some(s) => s,
                None => return Ok(true), // No stats, must scan
            };

            // Get field data type
            let field = match schema.field_with_name(col_name) {
                Ok(f) => f,
                Err(_) => return Ok(true), // Column not found, conservative
            };

            // Evaluate based on operator
            match binary.op {
                Operator::Eq => evaluate_eq(col_stats, literal_value, field.data_type()),
                Operator::NotEq => Ok(true), // Can't prune on != without bloom filters
                Operator::Lt => evaluate_lt(col_stats, literal_value, field.data_type()),
                Operator::LtEq => evaluate_lteq(col_stats, literal_value, field.data_type()),
                Operator::Gt => evaluate_gt(col_stats, literal_value, field.data_type()),
                Operator::GtEq => evaluate_gteq(col_stats, literal_value, field.data_type()),
                _ => Ok(true), // Conservative for other operators
            }
        }

        // BETWEEN: col BETWEEN low AND high
        Expr::Between(between) => {
            let col_name = match between.expr.as_ref() {
                Expr::Column(col) => col.name.as_str(),
                _ => return Ok(true),
            };

            let low_value = match between.low.as_ref() {
                Expr::Literal(val, ..) => val,
                _ => return Ok(true),
            };

            let high_value = match between.high.as_ref() {
                Expr::Literal(val, ..) => val,
                _ => return Ok(true),
            };

            let col_stats = match stats.get(col_name) {
                Some(s) => s,
                None => return Ok(true),
            };

            let field = match schema.field_with_name(col_name) {
                Ok(f) => f,
                Err(_) => return Ok(true),
            };

            let data_type = field.data_type();

            // File matches if: max >= low AND min <= high
            let max_gte_low = if between.negated {
                true // Can't prune NOT BETWEEN
            } else {
                evaluate_gteq(col_stats, low_value, data_type)?
            };

            let min_lte_high = if between.negated {
                true
            } else {
                evaluate_lteq(col_stats, high_value, data_type)?
            };

            Ok(max_gte_low && min_lte_high)
        }

        // AND: all conditions must potentially match
        Expr::BinaryExpr(binary) if matches!(binary.op, Operator::And) => {
            let left_match = evaluate_predicate(binary.left.as_ref(), stats, schema)?;
            let right_match = evaluate_predicate(binary.right.as_ref(), stats, schema)?;
            Ok(left_match && right_match)
        }

        // OR: any condition can potentially match
        Expr::BinaryExpr(binary) if matches!(binary.op, Operator::Or) => {
            let left_match = evaluate_predicate(binary.left.as_ref(), stats, schema)?;
            let right_match = evaluate_predicate(binary.right.as_ref(), stats, schema)?;
            Ok(left_match || right_match)
        }

        // Conservative for other expression types
        _ => Ok(true),
    }
}

/// Evaluate: col = value
fn evaluate_eq(
    stats: &crate::types::ColumnStats,
    value: &datafusion_common::ScalarValue,
    data_type: &arrow::datatypes::DataType,
) -> Result<bool> {
    // File matches if: min <= value <= max
    let min_lte = compare_stat_to_scalar(stats.min.as_deref(), value, data_type, |cmp| {
        matches!(cmp, std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    })?;
    let max_gte = compare_stat_to_scalar(stats.max.as_deref(), value, data_type, |cmp| {
        matches!(cmp, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
    })?;
    Ok(min_lte && max_gte)
}

/// Evaluate: col < value
fn evaluate_lt(
    stats: &crate::types::ColumnStats,
    value: &datafusion_common::ScalarValue,
    data_type: &arrow::datatypes::DataType,
) -> Result<bool> {
    // File matches if: min < value
    compare_stat_to_scalar(stats.min.as_deref(), value, data_type, |cmp| {
        matches!(cmp, std::cmp::Ordering::Less)
    })
}

/// Evaluate: col <= value
fn evaluate_lteq(
    stats: &crate::types::ColumnStats,
    value: &datafusion_common::ScalarValue,
    data_type: &arrow::datatypes::DataType,
) -> Result<bool> {
    // File matches if: min <= value
    compare_stat_to_scalar(stats.min.as_deref(), value, data_type, |cmp| {
        matches!(cmp, std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    })
}

/// Evaluate: col > value
fn evaluate_gt(
    stats: &crate::types::ColumnStats,
    value: &datafusion_common::ScalarValue,
    data_type: &arrow::datatypes::DataType,
) -> Result<bool> {
    // File matches if: max > value
    compare_stat_to_scalar(stats.max.as_deref(), value, data_type, |cmp| {
        matches!(cmp, std::cmp::Ordering::Greater)
    })
}

/// Evaluate: col >= value
fn evaluate_gteq(
    stats: &crate::types::ColumnStats,
    value: &datafusion_common::ScalarValue,
    data_type: &arrow::datatypes::DataType,
) -> Result<bool> {
    // File matches if: max >= value
    compare_stat_to_scalar(stats.max.as_deref(), value, data_type, |cmp| {
        matches!(cmp, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
    })
}

/// Compare a statistics byte array to a scalar value using proper type-aware comparison.
fn compare_stat_to_scalar<F>(
    stat_bytes: Option<&[u8]>,
    scalar: &datafusion_common::ScalarValue,
    data_type: &arrow::datatypes::DataType,
    cmp_fn: F,
) -> Result<bool>
where
    F: Fn(std::cmp::Ordering) -> bool,
{
    use arrow::datatypes::DataType;
    use datafusion_common::ScalarValue;

    let stat_bytes = match stat_bytes {
        Some(b) => b,
        None => return Ok(true), // No stat, must scan conservatively
    };

    // Type-specific comparison
    match (data_type, scalar) {
        // UInt64 comparison
        (DataType::UInt64, ScalarValue::UInt64(Some(val))) => {
            if stat_bytes.len() != 8 {
                return Ok(true);
            }
            let stat_val = u64::from_le_bytes(stat_bytes.try_into().unwrap());
            Ok(cmp_fn(stat_val.cmp(val)))
        }

        // Int64 comparison
        (DataType::Int64, ScalarValue::Int64(Some(val))) => {
            if stat_bytes.len() != 8 {
                return Ok(true);
            }
            let stat_val = i64::from_le_bytes(stat_bytes.try_into().unwrap());
            Ok(cmp_fn(stat_val.cmp(val)))
        }

        // Float64 comparison
        (DataType::Float64, ScalarValue::Float64(Some(val))) => {
            if stat_bytes.len() != 8 {
                return Ok(true);
            }
            let stat_val = f64::from_le_bytes(stat_bytes.try_into().unwrap());
            // Use total_cmp for proper NaN/Inf handling
            Ok(cmp_fn(stat_val.total_cmp(val)))
        }

        // String comparison (UTF-8)
        (DataType::Utf8, ScalarValue::Utf8(Some(val))) => {
            match std::str::from_utf8(stat_bytes) {
                Ok(stat_str) => Ok(cmp_fn(stat_str.cmp(val.as_str()))),
                Err(_) => Ok(true), // Invalid UTF-8, must scan
            }
        }

        // Add more types as needed...
        _ => Ok(true), // Conservative: unsupported type comparison
    }
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
        let batches: Vec<_> = store
            .scan(table_id, &[], None, None)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);
    }

    #[test]
    fn test_scan_with_pruning() {
        let pager = Arc::new(MemPager::new());
        let store = ParquetStore::open(pager).unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("value", DataType::UInt64, false),
        ]));

        let table_id = store.create_table("data", schema.clone()).unwrap();

        // Insert 3 batches with different row_id ranges
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3])),
                Arc::new(UInt64Array::from(vec![100, 200, 300])),
            ],
        )
        .unwrap();

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![100, 101, 102])),
                Arc::new(UInt64Array::from(vec![1000, 1010, 1020])),
            ],
        )
        .unwrap();

        let batch3 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![200, 201, 202])),
                Arc::new(UInt64Array::from(vec![2000, 2010, 2020])),
            ],
        )
        .unwrap();

        let batch1_mvcc = crate::mvcc::add_mvcc_columns(batch1, 1).unwrap();
        let batch2_mvcc = crate::mvcc::add_mvcc_columns(batch2, 1).unwrap();
        let batch3_mvcc = crate::mvcc::add_mvcc_columns(batch3, 1).unwrap();

        store
            .append_many(table_id, vec![batch1_mvcc, batch2_mvcc, batch3_mvcc])
            .unwrap();

        // Scan with predicate that should only hit batch2 (row_id BETWEEN 100 AND 102)
        use datafusion_expr::{col, lit, Expr};
        let pred = col("row_id").between(lit(100u64), lit(102u64));
        let batches: Vec<_> = store
            .scan(table_id, &[pred], None, None)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        // Should get exactly 1 batch (batch2) with 3 rows
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);

        // Scan with predicate that hits batch1 and batch2 (row_id BETWEEN 1 AND 101)
        let pred = col("row_id").between(lit(1u64), lit(101u64));
        let batches: Vec<_> = store
            .scan(table_id, &[pred], None, None)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        // Should get 2 batches
        assert_eq!(batches.len(), 2);

        // Scan all
        let batches: Vec<_> = store
            .scan(table_id, &[], None, None)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        // Should get all 3 batches
        assert_eq!(batches.len(), 3);
    }
}
