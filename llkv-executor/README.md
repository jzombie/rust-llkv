# LLKV Executor

**Work in Progress**

`llkv-executor` is the query execution engine for the [LLKV](../) toolkit.

## Purpose

- Execute **SELECT queries only** over table data with projection, filtering, and ordering.
- Provide streaming query results via RecordBatch iterators.
- Integrate with [`llkv-runtime`](../llkv-runtime/) for transaction-aware query execution.

## Executor vs Runtime

The **executor** ([llkv-executor](../llkv-executor/)) and **runtime** ([llkv-runtime](../llkv-runtime/)) serve different purposes:

- **Executor**: Low-level query evaluation engine that **only handles SELECT queries**. It takes a SELECT plan and produces streaming Arrow RecordBatch results. The executor knows nothing about transactions, MVCC metadata, or other SQL operations (INSERT, UPDATE, DELETE, CREATE TABLE).

- **Runtime**: High-level orchestration layer that handles **all SQL operations**, manages transactions, injects MVCC metadata, and coordinates between storage and execution layers. The runtime invokes the executor for SELECT operations.

In short: **Executor = SELECT-only query engine** | **Runtime = Full SQL coordinator**

## Design Notes

- The executor works with Arrow `RecordBatch` data and integrates with [`llkv-table`](../llkv-table/) scanning primitives.
- Query execution is designed for streaming results to avoid materializing entire result sets in memory.
- Invoked by [`llkv-runtime`](../llkv-runtime/) which provides the transaction context for row visibility filtering.
- Scalar subqueries run through the same evaluation pipeline as other scalar expressions, using planner-supplied correlated column placeholders to pull outer values during execution.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
