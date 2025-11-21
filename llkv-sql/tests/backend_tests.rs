use llkv_column_map::store::ColumnStore;
use llkv_parquet_store::ParquetStore;
use llkv_sql::{SqlEngine, SqlStatementResult};
use llkv_storage::pager::{BoxedPager, MemPager};
use llkv_table::catalog::TableCatalog;
use llkv_table::providers::column_map::ColumnStoreBackend;
use llkv_table::providers::parquet::ParquetStoreBackend;
use llkv_table::traits::CatalogBackend;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;

#[test]
fn test_backend_selection() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let rt = Runtime::new().expect("runtime");

    // Initialize stores once
    let column_store = Arc::new(ColumnStore::open(Arc::clone(&pager)).expect("column store"));
    let parquet_store = Arc::new(ParquetStore::open(Arc::clone(&pager)).expect("parquet store"));

    // Create Engine 1 with default=parquet
    let engine_parquet = {
        let metadata_backend = Box::new(ColumnStoreBackend::new(Arc::clone(&column_store)));
        let parquet_backend = Box::new(ParquetStoreBackend::new(Arc::clone(&parquet_store)));
        let column_backend = Box::new(ColumnStoreBackend::new(Arc::clone(&column_store)));

        let mut data_backends: HashMap<String, Box<dyn CatalogBackend>> = HashMap::new();
        data_backends.insert("columnstore".to_string(), column_backend);
        data_backends.insert("parquet".to_string(), parquet_backend);

        let catalog = TableCatalog::new(metadata_backend, data_backends, "parquet".to_string())
            .expect("catalog");
        SqlEngine::new_with_catalog(catalog).expect("engine parquet")
    };

    rt.block_on(async {
        // 1. Create table with default backend (Parquet)
        engine_parquet
            .execute("CREATE TABLE t_parquet(id INT, val TEXT);")
            .await
            .expect("create table parquet");

        engine_parquet
            .execute("INSERT INTO t_parquet(id, val) VALUES (1, 'a');")
            .await
            .expect("insert parquet");
    });

    // Create Engine 2 with default=columnstore, sharing the same stores
    let engine_columnstore = {
        let metadata_backend = Box::new(ColumnStoreBackend::new(Arc::clone(&column_store)));
        let parquet_backend = Box::new(ParquetStoreBackend::new(Arc::clone(&parquet_store)));
        let column_backend = Box::new(ColumnStoreBackend::new(Arc::clone(&column_store)));

        let mut data_backends: HashMap<String, Box<dyn CatalogBackend>> = HashMap::new();
        data_backends.insert("columnstore".to_string(), column_backend);
        data_backends.insert("parquet".to_string(), parquet_backend);

        let catalog = TableCatalog::new(metadata_backend, data_backends, "columnstore".to_string())
            .expect("catalog");
        SqlEngine::new_with_catalog(catalog).expect("engine columnstore")
    };

    rt.block_on(async {
        // 2. Create table with default backend (ColumnStore)
        engine_columnstore
            .execute("CREATE TABLE t_columnstore(id INT);")
            .await
            .expect("create table columnstore");

        engine_columnstore
            .execute("INSERT INTO t_columnstore(id) VALUES (1);")
            .await
            .expect("insert columnstore");

        // 3. Verify persistence and visibility
        // engine_columnstore should see t_parquet (loaded from metadata)
        let res = engine_columnstore
            .execute("SELECT id, val FROM t_parquet")
            .await
            .expect("select parquet");
        match res.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => assert_eq!(batches[0].num_rows(), 1),
            _ => panic!("expected query"),
        }

        let res = engine_columnstore
            .execute("SELECT id FROM t_columnstore")
            .await
            .expect("select columnstore");
        match res.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => assert_eq!(batches[0].num_rows(), 1),
            _ => panic!("expected query"),
        }
    });
}
