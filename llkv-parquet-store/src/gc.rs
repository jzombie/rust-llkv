//! Garbage collection utilities for ParquetStore.

use crate::catalog::ParquetCatalog;
use arrow::array::Array;
use llkv_result::Result;
use llkv_storage::pager::{BatchGet, GetResult, Pager};
use llkv_storage::types::PhysicalKey;
use rustc_hash::FxHashSet;
use simd_r_drive_entry_handle::EntryHandle;

/// Collect all physical keys referenced by the catalog.
///
/// This includes:
/// - The catalog root key itself (key 0)
/// - All Parquet file keys from all tables
/// - All external blob keys referenced in external storage columns
pub fn collect_reachable_keys<P>(
    catalog: &ParquetCatalog,
    pager: &P,
) -> Result<FxHashSet<PhysicalKey>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut reachable = FxHashSet::default();

    // The catalog itself occupies key 0
    reachable.insert(0);

    // Collect all Parquet file keys from all tables
    for table_meta in catalog.tables.values() {
        let schema = table_meta.schema()?;

        // Check if any columns use external storage
        let external_col_indices: Vec<usize> = schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| crate::external::is_external_field(field))
            .map(|(idx, _)| idx)
            .collect();

        for file_ref in &table_meta.parquet_files {
            // Add the Parquet file key itself
            reachable.insert(file_ref.physical_key);

            // If this table has external columns, read the file to collect blob keys
            if !external_col_indices.is_empty() {
                if let Ok(blob_keys) = collect_external_keys_from_file(
                    pager,
                    file_ref.physical_key,
                    &external_col_indices,
                ) {
                    reachable.extend(blob_keys);
                }
            }
        }
    }

    Ok(reachable)
}

/// Extract external blob keys from a Parquet file.
///
/// Reads the specified columns (which should be FixedSizeBinary(8) key columns)
/// and extracts the u64 keys they contain.
fn collect_external_keys_from_file<P>(
    pager: &P,
    file_key: PhysicalKey,
    external_col_indices: &[usize],
) -> Result<Vec<PhysicalKey>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    // Fetch the Parquet file
    let bytes = match pager.batch_get(&[BatchGet::Raw { key: file_key }])? {
        mut results if results.len() == 1 => match results.pop() {
            Some(GetResult::Raw { bytes, .. }) => bytes.into_bytes(),
            _ => return Ok(Vec::new()),
        },
        _ => return Ok(Vec::new()),
    };

    // Read the Parquet file with projection to only external columns
    let batches =
        crate::reader::read_parquet_from_memory(bytes, Some(external_col_indices.to_vec()))?;

    let mut blob_keys = Vec::new();

    // Extract keys from each batch
    // Note: After projection, column indices are renumbered (0, 1, 2...)
    for batch in batches {
        for projected_col_idx in 0..batch.num_columns() {
            let array = batch.column(projected_col_idx);

            // The column should be FixedSizeBinary containing physical keys
            // For column-level storage: FixedSizeBinary(12) = [physical_key(8), row_count(4)]
            // For legacy row-level storage: FixedSizeBinary(8) = [physical_key(8)]
            if let Some(fsb_array) = array
                .as_any()
                .downcast_ref::<arrow::array::FixedSizeBinaryArray>()
            {
                let key_size = fsb_array.value_length();

                if key_size == 12 {
                    // Column-level storage: extract physical_key from first row (all rows identical)
                    if !fsb_array.is_empty() {
                        let key_bytes = fsb_array.value(0);
                        let key = u64::from_le_bytes([
                            key_bytes[0],
                            key_bytes[1],
                            key_bytes[2],
                            key_bytes[3],
                            key_bytes[4],
                            key_bytes[5],
                            key_bytes[6],
                            key_bytes[7],
                        ]);
                        blob_keys.push(key);
                    }
                } else if key_size == 8 {
                    // Legacy row-level storage: extract key from each row
                    for row_idx in 0..array.len() {
                        let key_bytes = fsb_array.value(row_idx);
                        let key = u64::from_le_bytes([
                            key_bytes[0],
                            key_bytes[1],
                            key_bytes[2],
                            key_bytes[3],
                            key_bytes[4],
                            key_bytes[5],
                            key_bytes[6],
                            key_bytes[7],
                        ]);
                        blob_keys.push(key);
                    }
                }
            }
        }
    }

    Ok(blob_keys)
}

/// Identify and free all unreferenced blobs in a pager.
///
/// This performs garbage collection by:
/// 1. Enumerating all keys in the pager
/// 2. Identifying which keys are reachable from the catalog
/// 3. Freeing any keys that exist but are not reachable
///
/// Returns the number of keys freed.
///
/// # Arguments
///
/// * `pager` - The pager to garbage collect
/// * `catalog` - The catalog defining reachable keys
pub fn garbage_collect<P>(pager: &P, catalog: &ParquetCatalog) -> Result<usize>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    // Get all keys in the pager
    let all_keys: FxHashSet<PhysicalKey> = pager.enumerate_keys()?.into_iter().collect();

    // Get all reachable keys from catalog (including external blob keys)
    let reachable = collect_reachable_keys(catalog, pager)?;

    // Find unreferenced keys
    let unreferenced: Vec<PhysicalKey> = all_keys.difference(&reachable).copied().collect();

    let count = unreferenced.len();

    // Free them
    if !unreferenced.is_empty() {
        pager.free_many(&unreferenced)?;
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{ParquetCatalog, ParquetFileRef};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_collect_reachable_keys_empty() {
        use llkv_storage::pager::MemPager;

        let catalog = ParquetCatalog::default();
        let pager = MemPager::new();
        let reachable = collect_reachable_keys(&catalog, &pager).unwrap();

        // Only catalog key should be present
        assert_eq!(reachable.len(), 1);
        assert!(reachable.contains(&0));
    }

    #[test]
    fn test_collect_reachable_keys_with_tables() {
        use llkv_storage::pager::MemPager;

        let mut catalog = ParquetCatalog::default();

        // Create a table with some files
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("value", DataType::Utf8, true),
        ]));

        let _table_id = catalog
            .create_table("test_table".to_string(), schema.clone())
            .unwrap();

        let table_meta = catalog.get_table_mut("test_table").unwrap();

        // Add some Parquet files with physical keys 10, 20, 30
        table_meta.parquet_files.push(ParquetFileRef {
            physical_key: 10,
            row_count: 100,
            min_row_id: 0,
            max_row_id: 99,
            column_stats: None,
        });
        table_meta.parquet_files.push(ParquetFileRef {
            physical_key: 20,
            row_count: 50,
            min_row_id: 100,
            max_row_id: 149,
            column_stats: None,
        });
        table_meta.parquet_files.push(ParquetFileRef {
            physical_key: 30,
            row_count: 75,
            min_row_id: 150,
            max_row_id: 224,
            column_stats: None,
        });

        let pager = MemPager::new();
        let reachable = collect_reachable_keys(&catalog, &pager).unwrap();

        // Should have catalog + 3 file keys
        assert_eq!(reachable.len(), 4);
        assert!(reachable.contains(&0)); // catalog
        assert!(reachable.contains(&10));
        assert!(reachable.contains(&20));
        assert!(reachable.contains(&30));
    }

    #[test]
    fn test_collect_reachable_keys_multiple_tables() {
        use llkv_storage::pager::MemPager;

        let mut catalog = ParquetCatalog::default();

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));

        catalog
            .create_table("table1".to_string(), schema.clone())
            .unwrap();
        catalog
            .create_table("table2".to_string(), schema.clone())
            .unwrap();

        let t1 = catalog.get_table_mut("table1").unwrap();
        t1.parquet_files.push(ParquetFileRef {
            physical_key: 100,
            row_count: 10,
            min_row_id: 0,
            max_row_id: 9,
            column_stats: None,
        });

        let t2 = catalog.get_table_mut("table2").unwrap();
        t2.parquet_files.push(ParquetFileRef {
            physical_key: 200,
            row_count: 20,
            min_row_id: 0,
            max_row_id: 19,
            column_stats: None,
        });
        t2.parquet_files.push(ParquetFileRef {
            physical_key: 201,
            row_count: 15,
            min_row_id: 20,
            max_row_id: 34,
            column_stats: None,
        });

        let pager = MemPager::new();
        let reachable = collect_reachable_keys(&catalog, &pager).unwrap();

        // catalog + 1 file from table1 + 2 files from table2
        assert_eq!(reachable.len(), 4);
        assert!(reachable.contains(&0));
        assert!(reachable.contains(&100));
        assert!(reachable.contains(&200));
        assert!(reachable.contains(&201));
    }
}
