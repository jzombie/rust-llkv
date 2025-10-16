# LLKV SQL

**Work in Progress**

`llkv-sql` is the SQL interface for the [LLKV](https://github.com/jzombie/rust-llkv) toolkit.

## Purpose

- Parse SQL statements and convert them to execution plans.
- Provide the primary user-facing interface for database operations.
- Execute SQL with full transaction support via [`llkv-runtime`](../llkv-runtime/).

## Design Notes

- Uses [sqlparser](https://crates.io/crates/sqlparser) for SQL parsing and AST generation.
- Uses [sqllogictest](https://crates.io/crates/sqllogictest) for SQL testing.
- Converts AST to execution plans and delegates to [`llkv-runtime`](../llkv-runtime/) for execution.
- Returns results as Arrow `RecordBatch` instances for SELECT queries.
- The runtime handles all operation types and coordinates with [`llkv-transaction`](../llkv-transaction/) for MVCC support.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
