# LLKV SQL

**Work in Progress**

`llkv-sql` is the SQL interface for the [LLKV](../) toolkit.

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
- Correlated subqueries reuse the shared [`llkv_plan::correlation`] utilities so that placeholder generation and column mapping logic remain centralized in the planning layer.
- Scalar subqueries materialize into `SelectPlan::scalar_subqueries`, allowing the executor to bind correlated inputs without re-parsing SQL.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
