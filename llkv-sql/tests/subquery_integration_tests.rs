use std::sync::Arc;

use arrow::array::{Array, Int64Array, StringArray};
use llkv_runtime::RuntimeStatementResult;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;

#[test]
fn scalar_subquery_with_nested_filter_exists() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT)")
        .expect("create customers table");
    engine
        .execute("CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER)")
        .expect("create orders table");
    engine
        .execute(
            "CREATE TABLE order_items(id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER, qty INTEGER)",
        )
        .expect("create order_items table");
    engine
        .execute("CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, price INTEGER)")
        .expect("create products table");
    engine
        .execute("CREATE TABLE reviews(id INTEGER PRIMARY KEY, product_id INTEGER, rating INTEGER)")
        .expect("create reviews table");

    engine
        .execute("INSERT INTO customers VALUES (1, 'alice'), (2, 'bob')")
        .expect("insert customers");
    engine
        .execute("INSERT INTO orders VALUES (201, 1), (202, 1), (203, 2)")
        .expect("insert orders");
    engine
        .execute(
            "INSERT INTO order_items VALUES (301, 201, 101, 1), (302, 201, 102, 2), (303, 202, 101, 1), (304, 203, 101, 1)",
        )
        .expect("insert order_items");
    engine
        .execute("INSERT INTO products VALUES (101, 'widget', 20), (102, 'gadget', 8)")
        .expect("insert products");
    engine
        .execute("INSERT INTO reviews VALUES (401, 101, 5), (402, 102, 3)")
        .expect("insert reviews");

    let query = r#"
        SELECT
            c.name,
            (
                SELECT price
                FROM products p
                WHERE p.id = (
                    SELECT MIN(product_id)
                    FROM order_items
                    WHERE order_id = (
                        SELECT MIN(id)
                        FROM orders
                    )
                )
            ) AS nested_price,
            (
                SELECT COUNT(*)
                FROM orders
            ) AS total_orders
        FROM customers c
        WHERE EXISTS (
            SELECT 1
            FROM reviews r
            WHERE r.rating >= 4
        )
        ORDER BY name
    "#;

    let mut results = engine.execute(query).expect("execute nested subquery");
    assert_eq!(results.len(), 1, "expected single statement result");
    let select_result = results.remove(0);

    let batches = match select_result {
        RuntimeStatementResult::Select { execution, .. } => {
            execution.collect().expect("collect query batches")
        }
        other => panic!("expected select result, got {other:?}"),
    };
    assert_eq!(batches.len(), 1, "expected single record batch");
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 2, "expected two result rows");

    let names = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("downcast names column to StringArray");
    assert_eq!(names.value(0), "alice", "expected first customer");
    assert_eq!(names.value(1), "bob", "expected second customer");

    let nested_prices = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("downcast nested_price column to Int64Array");
    assert_eq!(nested_prices.value(0), 20, "expected alice nested price");
    assert_eq!(nested_prices.value(1), 20, "expected bob nested price");

    let total_orders = batch
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("downcast total_orders column to Int64Array");
    assert_eq!(total_orders.value(0), 3, "expected alice total orders");
    assert_eq!(total_orders.value(1), 3, "expected bob total orders");
}
