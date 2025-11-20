//! DataFusion integration helpers for LLKV storage.
//!
//! This crate wires [`llkv-column-map`] and [`llkv-storage`] into a
//! [`datafusion`] [`datafusion::datasource::TableProvider`] so DataFusion's
//! query engine can operate directly on LLKV's persisted columnar data. The
//! integration is intentionally minimal—writes flow through
//! [`llkv_column_map::store::ColumnStore::append`], while reads rely on the
//! store's projection utilities to materialize
//! [`arrow::record_batch::RecordBatch`]es for DataFusion.
//!
//! The primary entry points are:
//! - [`LlkvTableBuilder`]: registers logical columns inside a
//!   [`llkv_column_map::store::ColumnStore`],
//!   appends Arrow batches (with automatic `rowid` management), and tracks the
//!   row IDs that back each insert.
//! - [`LlkvTableProvider`]: implements DataFusion's
//!   [`datafusion::datasource::TableProvider`] by gathering LLKV rows in
//!   scan-sized chunks and surfacing them through a
//!   [`datafusion_datasource::memory::MemorySourceConfig`].

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::context::QueryPlanner;
use datafusion::logical_expr::{DmlStatement, LogicalPlan};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use llkv_storage::pager::BoxedPager;
use llkv_table::catalog::TableCatalog;

pub use llkv_table::{
    ColumnMapTableBuilder, ColumnMapTableProvider, LlkvTableBuilder, LlkvTableProvider,
};

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use arrow::util::pretty::pretty_format_batches;
    use datafusion::prelude::SessionContext;
    use llkv_column_map::store::ColumnStore;
    use llkv_storage::pager::MemPager;

    fn build_demo_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::UInt64, false),
            Field::new("score", DataType::Int32, false),
            Field::new("status", DataType::Utf8, true),
        ]));

        let user_ids = Arc::new(UInt64Array::from(vec![1_u64, 2, 3]));
        let scores = Arc::new(Int32Array::from(vec![42, 7, 99]));
        let statuses = Arc::new(StringArray::from(vec![Some("active"), None, Some("vip")]));

        RecordBatch::try_new(schema, vec![user_ids, scores, statuses]).expect("batch")
    }

    #[tokio::test]
    async fn datafusion_joins_llkv_tables() {
        let pager = Arc::new(MemPager::default());
        let store = Arc::new(ColumnStore::open(Arc::clone(&pager)).expect("store"));

        // Create users table
        let users_schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::UInt64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let users_batch = RecordBatch::try_new(
            users_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1_u64, 2, 3, 4])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie", "Diana"])),
            ],
        )
        .expect("users batch");

        let mut users_builder = ColumnMapTableBuilder::new(Arc::clone(&store), 1, users_schema);
        users_builder
            .append_batch(&users_batch)
            .expect("users append");
        let users_provider = users_builder.finish().expect("users provider");

        // Create orders table
        let orders_schema = Arc::new(Schema::new(vec![
            Field::new("order_id", DataType::UInt64, false),
            Field::new("user_id", DataType::UInt64, false),
            Field::new("amount", DataType::Int32, false),
        ]));
        let orders_batch = RecordBatch::try_new(
            orders_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![101_u64, 102, 103, 104, 105])),
                Arc::new(UInt64Array::from(vec![1_u64, 2, 1, 3, 2])),
                Arc::new(Int32Array::from(vec![50, 75, 120, 30, 90])),
            ],
        )
        .expect("orders batch");

        let mut orders_builder = ColumnMapTableBuilder::new(Arc::clone(&store), 2, orders_schema);
        orders_builder
            .append_batch(&orders_batch)
            .expect("orders append");
        let orders_provider = orders_builder.finish().expect("orders provider");

        // Register tables and perform join
        let ctx = SessionContext::new();
        ctx.register_table("users", Arc::new(users_provider))
            .expect("register users");
        ctx.register_table("orders", Arc::new(orders_provider))
            .expect("register orders");

        let df = ctx
            .sql(
                "SELECT u.name, o.order_id, o.amount \
                 FROM users u \
                 INNER JOIN orders o ON u.user_id = o.user_id \
                 ORDER BY u.name, o.order_id",
            )
            .await
            .expect("sql");
        let results = df.collect().await.expect("collect");
        let formatted = pretty_format_batches(&results).expect("format").to_string();

        let expected = vec![
            "+---------+----------+--------+",
            "| name    | order_id | amount |",
            "+---------+----------+--------+",
            "| Alice   | 101      | 50     |",
            "| Alice   | 103      | 120    |",
            "| Bob     | 102      | 75     |",
            "| Bob     | 105      | 90     |",
            "| Charlie | 104      | 30     |",
            "+---------+----------+--------+",
        ]
        .join("\n");

        assert_eq!(formatted.trim(), expected.trim());
    }

    #[tokio::test]
    async fn datafusion_catalog_schema_joins() {
        use datafusion::catalog::{CatalogProvider, SchemaProvider};
        use datafusion_catalog::{MemoryCatalogProvider, MemorySchemaProvider};

        let pager = Arc::new(MemPager::default());
        let store = Arc::new(ColumnStore::open(Arc::clone(&pager)).expect("store"));

        // Create products table in schema1
        let products_schema = Arc::new(Schema::new(vec![
            Field::new("product_id", DataType::UInt64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("price", DataType::Int32, false),
        ]));
        let products_batch = RecordBatch::try_new(
            products_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![10_u64, 20, 30])),
                Arc::new(StringArray::from(vec!["Widget", "Gadget", "Doohickey"])),
                Arc::new(Int32Array::from(vec![100, 200, 150])),
            ],
        )
        .expect("products batch");

        let mut products_builder =
            ColumnMapTableBuilder::new(Arc::clone(&store), 1, products_schema);
        products_builder
            .append_batch(&products_batch)
            .expect("products append");
        let products_provider = products_builder.finish().expect("products provider");

        // Create inventory table in schema2
        let inventory_schema = Arc::new(Schema::new(vec![
            Field::new("product_id", DataType::UInt64, false),
            Field::new("warehouse", DataType::Utf8, false),
            Field::new("quantity", DataType::Int32, false),
        ]));
        let inventory_batch = RecordBatch::try_new(
            inventory_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![10_u64, 20, 10, 30])),
                Arc::new(StringArray::from(vec!["East", "West", "West", "East"])),
                Arc::new(Int32Array::from(vec![50, 75, 25, 100])),
            ],
        )
        .expect("inventory batch");

        let mut inventory_builder =
            ColumnMapTableBuilder::new(Arc::clone(&store), 2, inventory_schema);
        inventory_builder
            .append_batch(&inventory_batch)
            .expect("inventory append");
        let inventory_provider = inventory_builder.finish().expect("inventory provider");

        // Set up catalog hierarchy: my_catalog -> schema1, schema2
        let ctx = SessionContext::new();

        let catalog = Arc::new(MemoryCatalogProvider::new());
        let schema1 = Arc::new(MemorySchemaProvider::new());
        let schema2 = Arc::new(MemorySchemaProvider::new());

        schema1
            .register_table("products".to_string(), Arc::new(products_provider))
            .expect("register products");
        schema2
            .register_table("inventory".to_string(), Arc::new(inventory_provider))
            .expect("register inventory");

        catalog
            .register_schema("schema1", schema1)
            .expect("register schema1");
        catalog
            .register_schema("schema2", schema2)
            .expect("register schema2");

        ctx.register_catalog("my_catalog", catalog);

        // Set default catalog and query with schema qualification
        ctx.sql("SET datafusion.catalog.default_catalog = 'my_catalog'")
            .await
            .expect("set catalog");

        // First test: simple query from one schema
        let df_simple = ctx
            .sql("SELECT name, price FROM schema1.products ORDER BY name")
            .await
            .expect("simple sql");
        let simple_results = df_simple.collect().await.expect("simple collect");
        let simple_formatted = pretty_format_batches(&simple_results)
            .expect("format")
            .to_string();

        let simple_expected = vec![
            "+-----------+-------+",
            "| name      | price |",
            "+-----------+-------+",
            "| Doohickey | 150   |",
            "| Gadget    | 200   |",
            "| Widget    | 100   |",
            "+-----------+-------+",
        ]
        .join("\n");

        assert_eq!(simple_formatted.trim(), simple_expected.trim());

        // Second test: join across schemas
        let df = ctx
            .sql(
                "SELECT p.name, i.warehouse, i.quantity, p.price \
                 FROM schema1.products p \
                 INNER JOIN schema2.inventory i ON p.product_id = i.product_id \
                 ORDER BY p.name, i.warehouse",
            )
            .await
            .expect("sql");
        let results = df.collect().await.expect("collect");
        let formatted = pretty_format_batches(&results).expect("format").to_string();

        let expected = vec![
            "+-----------+-----------+----------+-------+",
            "| name      | warehouse | quantity | price |",
            "+-----------+-----------+----------+-------+",
            "| Doohickey | East      | 100      | 150   |",
            "| Gadget    | West      | 75       | 200   |",
            "| Widget    | East      | 50       | 100   |",
            "| Widget    | West      | 25       | 100   |",
            "+-----------+-----------+----------+-------+",
        ]
        .join("\n");

        assert_eq!(formatted.trim(), expected.trim());

        // Test fully qualified table names
        let df2 = ctx
            .sql(
                "SELECT COUNT(*) as total \
                 FROM my_catalog.schema1.products",
            )
            .await
            .expect("fully qualified sql");
        let results2 = df2.collect().await.expect("collect");
        let formatted2 = pretty_format_batches(&results2)
            .expect("format")
            .to_string();

        let expected2 = vec![
            "+-------+",
            "| total |",
            "+-------+",
            "| 3     |",
            "+-------+",
        ]
        .join("\n");

        assert_eq!(formatted2.trim(), expected2.trim());
    }

    #[tokio::test]
    #[should_panic(expected = "DELETE FROM llkv_demo not yet implemented")]
    async fn datafusion_delete_statement_is_rejected() {
        use datafusion::execution::session_state::SessionStateBuilder;
        use llkv_storage::pager::BoxedPager;
        use llkv_table::catalog::TableCatalog;
        use llkv_table::providers::column_map::ColumnStoreBackend;

        let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
        let store = Arc::new(ColumnStore::open(Arc::clone(&pager)).expect("store"));
        let backend = Box::new(ColumnStoreBackend::new(Arc::clone(&store)));
        let catalog = TableCatalog::new(backend).expect("catalog");

        let schema = build_demo_batch().schema();
        let mut builder = ColumnMapTableBuilder::new(Arc::clone(&store), 1, schema);
        builder.append_batch(&build_demo_batch()).expect("append");
        let provider = builder.finish().expect("provider");

        // Create context with custom query planner that intercepts DELETE
        let session_state = SessionStateBuilder::new()
            .with_default_features()
            .with_query_planner(Arc::new(LlkvQueryPlanner::new(catalog)))
            .build();
        let ctx = SessionContext::new_with_state(session_state);
        ctx.register_table("llkv_demo", Arc::new(provider))
            .expect("register");

        let delete_df = ctx
            .sql("DELETE FROM llkv_demo WHERE user_id = 2")
            .await
            .expect("DELETE planning");

        // This will panic with todo!() - demonstrating interception works
        let _results = delete_df.collect().await.unwrap();
    }
}

/// Custom physical planner that intercepts DDL and DML operations.
pub struct LlkvQueryPlanner {
    catalog: Arc<TableCatalog>,
    fallback: Arc<dyn PhysicalPlanner>,
}

impl fmt::Debug for LlkvQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlkvQueryPlanner")
            .field("catalog", &"<TableCatalog>")
            .finish()
    }
}

impl LlkvQueryPlanner {
    /// Construct a new planner with a catalog reference.
    pub fn new(catalog: Arc<TableCatalog>) -> Self {
        Self {
            catalog,
            fallback: Arc::new(DefaultPhysicalPlanner::default()),
        }
    }
}

#[async_trait]
impl QueryPlanner for LlkvQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &datafusion::execution::context::SessionState,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Intercept DELETE
        if let LogicalPlan::Dml(DmlStatement {
            table_name,
            op: datafusion::logical_expr::WriteOp::Delete,
            ..
        }) = logical_plan
        {
            return Err(DataFusionError::NotImplemented(format!(
                "DELETE FROM {} not yet implemented",
                table_name
            )));
        }

        // Delegate everything else to the default planner
        self.fallback
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

#[cfg(test)]
mod persistence_tests {
    use arrow::array::{Int32Array, StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use arrow::util::pretty::pretty_format_batches;
    use datafusion::prelude::SessionContext;
    use llkv_storage::pager::MemPager;
    use std::sync::Arc;

    #[tokio::test]
    async fn table_persists_across_scopes() {
        use llkv_column_map::store::ColumnStore;
        use llkv_table::catalog::TableCatalog;
        use llkv_table::providers::column_map::ColumnStoreBackend;

        // Outer scope: create a persistent pager that will outlive the table creation
        let pager = Arc::new(MemPager::default());

        // Inner scope 1: Create table schema, insert data, then drop everything except pager
        {
            let store = Arc::new(ColumnStore::open(Arc::clone(&pager)).expect("store"));
            let backend = Box::new(ColumnStoreBackend::new(store));
            let catalog = TableCatalog::new(backend).expect("open catalog");

            let users_schema = Arc::new(Schema::new(vec![
                Field::new("user_id", DataType::UInt64, false),
                Field::new("username", DataType::Utf8, false),
                Field::new("age", DataType::Int32, false),
            ]));

            let users_batch = RecordBatch::try_new(
                users_schema.clone(),
                vec![
                    Arc::new(UInt64Array::from(vec![1_u64, 2, 3, 4])),
                    Arc::new(StringArray::from(vec!["alice", "bob", "carol", "dave"])),
                    Arc::new(Int32Array::from(vec![30, 25, 35, 28])),
                ],
            )
            .expect("users batch");

            let mut users_builder = catalog
                .create_table("users", users_schema)
                .expect("create users table");

            users_builder
                .append_batch(&users_batch)
                .expect("users append");

            let users_provider = users_builder.finish().expect("users provider");

            // Update catalog with the new row IDs (1, 2, 3, 4)
            let new_row_ids: Vec<u64> = (1..=users_batch.num_rows() as u64).collect();
            catalog
                .update_table_rows("users", new_row_ids)
                .expect("update row ids");

            // Verify the data is immediately queryable
            let ctx = SessionContext::new();
            ctx.register_table("users", users_provider)
                .expect("register users");

            let df = ctx
                .sql("SELECT COUNT(*) as cnt FROM users")
                .await
                .expect("count sql");

            let results = df.collect().await.expect("count collect");
            let formatted = pretty_format_batches(&results).expect("format").to_string();
            assert!(formatted.contains("| 4"));

            // Everything except `pager` drops here - catalog metadata is persisted
        }

        // Inner scope 2: Reopen the catalog with the same pager and verify data persists
        {
            let store = Arc::new(ColumnStore::open(Arc::clone(&pager)).expect("reopen store"));
            let backend = Box::new(ColumnStoreBackend::new(store));
            let catalog = TableCatalog::new(backend).expect("reopen catalog");

            // List tables to verify it was persisted
            let tables = catalog.list_tables();
            assert!(
                tables.contains(&"users".to_string()),
                "users table should be in catalog"
            );

            // Get the table provider - this reconstructs it from persisted metadata
            let users_provider = catalog
                .get_table("users")
                .expect("get users table")
                .expect("users table should exist");

            // Query the table through DataFusion
            let ctx = SessionContext::new();
            ctx.register_table("users", users_provider)
                .expect("register users");

            let df = ctx
                .sql("SELECT username, age FROM users WHERE age > 27 ORDER BY username")
                .await
                .expect("sql");

            let results = df.collect().await.expect("collect");
            let formatted = pretty_format_batches(&results).expect("format").to_string();

            let expected = vec![
                "+----------+-----+",
                "| username | age |",
                "+----------+-----+",
                "| alice    | 30  |",
                "| carol    | 35  |",
                "| dave     | 28  |",
                "+----------+-----+",
            ]
            .join("\n");

            assert_eq!(formatted.trim(), expected.trim());
        }
    }

    #[tokio::test]
    async fn federated_query_across_pagers() {
        use llkv_column_map::store::ColumnStore;
        use llkv_storage::pager::SimdRDrivePager;
        use llkv_table::catalog::TableCatalog;
        use llkv_table::providers::column_map::ColumnStoreBackend;
        use tempfile::TempDir;

        // Create temporary directories for the databases
        let temp_dir = TempDir::new().expect("create temp dir");
        let sales_path = temp_dir.path().join("sales.db");
        let inventory_path = temp_dir.path().join("inventory.db");

        // Pager 1 - sales database with orders table
        let sales_pager = Arc::new(SimdRDrivePager::open(&sales_path).expect("open sales pager"));
        let sales_store =
            Arc::new(ColumnStore::open(Arc::clone(&sales_pager)).expect("open sales store"));
        let sales_backend = Box::new(ColumnStoreBackend::new(sales_store));
        let sales_catalog = TableCatalog::new(sales_backend).expect("open sales catalog");

        let orders_schema = Arc::new(Schema::new(vec![
            Field::new("order_id", DataType::UInt64, false),
            Field::new("product_id", DataType::UInt64, false),
            Field::new("customer", DataType::Utf8, false),
            Field::new("quantity", DataType::Int32, false),
        ]));

        let orders_batch = RecordBatch::try_new(
            orders_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1001_u64, 1002, 1003, 1004])),
                Arc::new(UInt64Array::from(vec![10_u64, 20, 10, 30])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol", "Dave"])),
                Arc::new(Int32Array::from(vec![5, 3, 2, 10])),
            ],
        )
        .expect("orders batch");

        let mut orders_builder = sales_catalog
            .create_table("orders", orders_schema)
            .expect("create orders table");

        orders_builder
            .append_batch(&orders_batch)
            .expect("append orders");

        let _orders_provider = orders_builder.finish().expect("finish orders");

        // Update catalog with row IDs
        sales_catalog
            .update_table_rows("orders", (1..=4).collect())
            .expect("update orders rows");

        // Pager 2 - inventory database with products table
        let inventory_pager =
            Arc::new(SimdRDrivePager::open(&inventory_path).expect("open inventory pager"));
        let inventory_store = Arc::new(
            ColumnStore::open(Arc::clone(&inventory_pager)).expect("open inventory store"),
        );
        let inventory_backend = Box::new(ColumnStoreBackend::new(inventory_store));
        let inventory_catalog =
            TableCatalog::new(inventory_backend).expect("open inventory catalog");

        let products_schema = Arc::new(Schema::new(vec![
            Field::new("product_id", DataType::UInt64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("price", DataType::Int32, false),
        ]));

        let products_batch = RecordBatch::try_new(
            products_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![10_u64, 20, 30])),
                Arc::new(StringArray::from(vec!["Widget", "Gadget", "Doohickey"])),
                Arc::new(Int32Array::from(vec![100, 200, 150])),
            ],
        )
        .expect("products batch");

        let mut products_builder = inventory_catalog
            .create_table("products", products_schema)
            .expect("create products table");

        products_builder
            .append_batch(&products_batch)
            .expect("append products");

        let _products_provider = products_builder.finish().expect("finish products");

        // Update catalog with row IDs
        inventory_catalog
            .update_table_rows("products", (1..=3).collect())
            .expect("update products rows");

        // Register tables from both catalogs into DataFusion
        let ctx = SessionContext::new();

        let orders_table = sales_catalog
            .get_table("orders")
            .expect("get orders")
            .expect("orders should exist");

        let products_table = inventory_catalog
            .get_table("products")
            .expect("get products")
            .expect("products should exist");

        ctx.register_table("orders", orders_table)
            .expect("register orders");
        ctx.register_table("products", products_table)
            .expect("register products");

        // Test 1: Query each table independently
        let orders_df = ctx
            .sql("SELECT customer, quantity FROM orders ORDER BY customer")
            .await
            .expect("query orders");

        let orders_results = orders_df.collect().await.expect("collect orders");
        let orders_formatted = pretty_format_batches(&orders_results)
            .expect("format orders")
            .to_string();

        let expected_orders = vec![
            "+----------+----------+",
            "| customer | quantity |",
            "+----------+----------+",
            "| Alice    | 5        |",
            "| Bob      | 3        |",
            "| Carol    | 2        |",
            "| Dave     | 10       |",
            "+----------+----------+",
        ]
        .join("\n");

        assert_eq!(orders_formatted.trim(), expected_orders.trim());

        let products_df = ctx
            .sql("SELECT name, price FROM products ORDER BY name")
            .await
            .expect("query products");

        let products_results = products_df.collect().await.expect("collect products");
        let products_formatted = pretty_format_batches(&products_results)
            .expect("format products")
            .to_string();

        let expected_products = vec![
            "+-----------+-------+",
            "| name      | price |",
            "+-----------+-------+",
            "| Doohickey | 150   |",
            "| Gadget    | 200   |",
            "| Widget    | 100   |",
            "+-----------+-------+",
        ]
        .join("\n");

        assert_eq!(products_formatted.trim(), expected_products.trim());

        // Test 2: Federated query - join across pagers!
        let join_df = ctx
            .sql(
                "SELECT o.customer, p.name, o.quantity, p.price, (o.quantity * p.price) as total \
                 FROM orders o \
                 INNER JOIN products p ON o.product_id = p.product_id \
                 ORDER BY o.customer",
            )
            .await
            .expect("federated join query");

        let join_results = join_df.collect().await.expect("collect join");
        let join_formatted = pretty_format_batches(&join_results)
            .expect("format join")
            .to_string();

        let expected_join = vec![
            "+----------+-----------+----------+-------+-------+",
            "| customer | name      | quantity | price | total |",
            "+----------+-----------+----------+-------+-------+",
            "| Alice    | Widget    | 5        | 100   | 500   |",
            "| Bob      | Gadget    | 3        | 200   | 600   |",
            "| Carol    | Widget    | 2        | 100   | 200   |",
            "| Dave     | Doohickey | 10       | 150   | 1500  |",
            "+----------+-----------+----------+-------+-------+",
        ]
        .join("\n");

        assert_eq!(join_formatted.trim(), expected_join.trim());

        // Test 3: Verify data persists - reopen catalogs and query again
        drop(sales_catalog);
        drop(inventory_catalog);

        let sales_store = Arc::new(ColumnStore::open(sales_pager).expect("reopen sales store"));
        let sales_backend = Box::new(ColumnStoreBackend::new(sales_store));
        let sales_catalog_reopened =
            TableCatalog::new(sales_backend).expect("reopen sales catalog");

        let inventory_store =
            Arc::new(ColumnStore::open(inventory_pager).expect("reopen inventory store"));
        let inventory_backend = Box::new(ColumnStoreBackend::new(inventory_store));
        let inventory_catalog_reopened =
            TableCatalog::new(inventory_backend).expect("reopen inventory catalog");

        // Verify tables are still in the catalogs
        let sales_tables = sales_catalog_reopened.list_tables();
        let inventory_tables = inventory_catalog_reopened.list_tables();

        assert!(sales_tables.contains(&"orders".to_string()));
        assert!(inventory_tables.contains(&"products".to_string()));

        // Re-register and query
        let ctx2 = SessionContext::new();

        let orders_table2 = sales_catalog_reopened
            .get_table("orders")
            .expect("get orders 2")
            .expect("orders should exist 2");

        let products_table2 = inventory_catalog_reopened
            .get_table("products")
            .expect("get products 2")
            .expect("products should exist 2");

        ctx2.register_table("orders", orders_table2)
            .expect("register orders 2");
        ctx2.register_table("products", products_table2)
            .expect("register products 2");

        // Run the same join query - should work with persisted data
        let join_df2 = ctx2
            .sql(
                "SELECT o.customer, p.name, o.quantity * p.price as total \
                 FROM orders o \
                 INNER JOIN products p ON o.product_id = p.product_id \
                 WHERE o.quantity > 2 \
                 ORDER BY total DESC",
            )
            .await
            .expect("federated join query 2");

        let join_results2 = join_df2.collect().await.expect("collect join 2");
        let join_formatted2 = pretty_format_batches(&join_results2)
            .expect("format join 2")
            .to_string();

        let expected_join2 = vec![
            "+----------+-----------+-------+",
            "| customer | name      | total |",
            "+----------+-----------+-------+",
            "| Dave     | Doohickey | 1500  |",
            "| Bob      | Gadget    | 600   |",
            "| Alice    | Widget    | 500   |",
            "+----------+-----------+-------+",
        ]
        .join("\n");

        assert_eq!(join_formatted2.trim(), expected_join2.trim());
    }
}
