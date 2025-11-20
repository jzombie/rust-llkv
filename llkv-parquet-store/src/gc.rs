//! Garbage collection utilities for ParquetStore.

use crate::catalog::ParquetCatalog;
use llkv_result::Result;
use llkv_storage::pager::Pager;
use llkv_storage::types::PhysicalKey;
use rustc_hash::FxHashSet;
use simd_r_drive_entry_handle::EntryHandle;

/// Collect all physical keys referenced by the catalog.
///
/// This includes:
/// - The catalog root key itself (key 0)
/// - All Parquet file keys from all tables
pub fn collect_reachable_keys(catalog: &ParquetCatalog) -> FxHashSet<PhysicalKey> {
    let mut reachable = FxHashSet::default();

    // The catalog itself occupies key 0
    reachable.insert(0);

    // Collect all Parquet file keys from all tables
    for table_meta in catalog.tables.values() {
        for file_ref in &table_meta.parquet_files {
            reachable.insert(file_ref.physical_key);
        }
    }

    reachable
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

    // Get all reachable keys from catalog
    let reachable = collect_reachable_keys(catalog);

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
        let catalog = ParquetCatalog::default();
        let reachable = collect_reachable_keys(&catalog);

        // Only catalog key should be present
        assert_eq!(reachable.len(), 1);
        assert!(reachable.contains(&0));
    }

    #[test]
    fn test_collect_reachable_keys_with_tables() {
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

        let reachable = collect_reachable_keys(&catalog);

        // Should have catalog + 3 file keys
        assert_eq!(reachable.len(), 4);
        assert!(reachable.contains(&0)); // catalog
        assert!(reachable.contains(&10));
        assert!(reachable.contains(&20));
        assert!(reachable.contains(&30));
    }

    #[test]
    fn test_collect_reachable_keys_multiple_tables() {
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

        let reachable = collect_reachable_keys(&catalog);

        // catalog + 1 file from table1 + 2 files from table2
        assert_eq!(reachable.len(), 4);
        assert!(reachable.contains(&0));
        assert!(reachable.contains(&100));
        assert!(reachable.contains(&200));
        assert!(reachable.contains(&201));
    }
}
