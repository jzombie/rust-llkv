# LLKV Runtime

**Work in Progress**

`llkv-runtime` provides the execution runtime for the [LLKV](https://github.com/jzombie/rust-llkv) toolkit.

## Purpose

- Coordinate between the [transaction layer](../llkv-transaction/), [storage layer](../llkv-table/), and [query executor](../llkv-executor/).
- Execute SQL operations (CREATE TABLE, INSERT, UPDATE, DELETE, SELECT) with full transaction support.
- Manage MVCC metadata injection for all data modifications.
- Provide session-level transaction management (auto-commit and multi-statement transactions).

## Runtime vs Executor

The **runtime** ([llkv-runtime](../llkv-runtime/)) and **executor** ([llkv-executor](../llkv-executor/)) serve different purposes:

- **Runtime**: High-level orchestration layer that handles **all SQL operations**, manages transactions, injects MVCC metadata, and coordinates between storage and execution layers. Used by [`llkv-sql`](../llkv-sql/) to execute complete SQL statements.

- **Executor**: Low-level query evaluation engine that **only handles SELECT queries**. It takes a SELECT plan and produces streaming Arrow `RecordBatch` results. The executor is invoked by the runtime for SELECT operations but knows nothing about transactions, MVCC metadata, or other SQL operations.

In short: **Runtime = Full SQL coordinator** | **Executor = SELECT-only query engine**

## Design Notes

- The runtime automatically injects MVCC columns (`row_id`, `created_by`, `deleted_by`) for all data operations.
- Supports both auto-commit (single-statement) and explicit BEGIN/COMMIT/ROLLBACK transactions.
- Integrates with [`llkv-transaction`](../llkv-transaction/) for snapshot isolation and visibility filtering.
- Delegates SELECT query evaluation to [`llkv-executor`](../llkv-executor/) while handling transaction context.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
