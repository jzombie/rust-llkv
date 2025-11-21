//! Demonstration of configurable compression in ParquetStore.

use arrow::array::{StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_parquet_store::{ParquetStore, WriterConfig};
use llkv_storage::pager::{MemPager, Pager};
use parquet::basic::Compression;
use std::sync::Arc;

fn main() -> llkv_result::Result<()> {
    println!("=== Parquet Compression Demo ===\n");

    // Create test data (100 rows with repetitive data - good for compression)
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from_iter(0..100)),
            Arc::new(StringArray::from_iter(
                (0..100).map(|i| Some(format!("User_{}", i))),
            )),
            Arc::new(StringArray::from(vec!["Premium"; 100])),
        ],
    )?;

    // Add MVCC columns
    let batch_with_mvcc = llkv_parquet_store::add_mvcc_columns(batch, 1)?;

    // Test different compression configurations
    let configs = vec![
        ("Uncompressed", WriterConfig::uncompressed()),
        ("Snappy (default)", WriterConfig::snappy()),
        (
            "Custom (Snappy, large row groups)",
            WriterConfig::snappy().with_max_row_group_size(100_000),
        ),
    ];

    for (name, config) in configs {
        let pager = Arc::new(MemPager::new());
        let store = ParquetStore::open_with_config(Arc::clone(&pager), config)?;

        // Create table and append data
        let table_id = store.create_table("test", schema.clone())?;
        store.append_many(table_id, vec![batch_with_mvcc.clone()])?;

        // Get all keys and calculate total size
        let keys = pager.enumerate_keys()?;
        let mut total_bytes = 0;
        for &key in &keys {
            if let Some(size) = get_blob_size(pager.as_ref(), key)? {
                total_bytes += size;
            }
        }

        println!("{:30} -> {} bytes", name, total_bytes);
    }

    println!("\n=== Builder Pattern Example ===\n");

    // Custom configuration using builder pattern
    let custom_config = WriterConfig::default()
        .with_compression(Compression::SNAPPY)
        .with_max_row_group_size(50_000)
        .with_statistics(true)
        .with_dictionary(true);

    let pager = Arc::new(MemPager::new());
    let mut store = ParquetStore::open_with_config(pager, custom_config)?;

    println!("Initial config: {:?}", store.writer_config());

    // You can also change configuration at runtime
    store.set_writer_config(WriterConfig::uncompressed());
    println!("Updated config: {:?}", store.writer_config());

    println!("\n=== Compression Recommendations ===\n");
    println!("• Uncompressed:  Fastest writes, largest storage");
    println!("• Snappy:        Good balance (recommended)");
    println!("• Zstd:          Best compression (requires feature flag)");
    println!("• LZ4:           Fast compression (requires feature flag)");
    println!("• Gzip:          Good compatibility (requires feature flag)");

    Ok(())
}

fn get_blob_size<P: Pager>(pager: &P, key: u64) -> llkv_result::Result<Option<usize>> {
    use llkv_storage::pager::{BatchGet, GetResult};

    let results = pager.batch_get(&[BatchGet::Raw { key }])?;
    match results.first() {
        Some(GetResult::Raw { bytes, .. }) => Ok(Some(bytes.as_ref().len())),
        _ => Ok(None),
    }
}
