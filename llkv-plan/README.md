# LLKV Plan

**Work in Progress**

`llkv-plan` defines the typed plan surface that bridges parsed SQL and LLKV runtime execution. It organizes statement plans, expression programs, and subquery metadata so lower layers can execute without re-parsing SQL.

## Responsibilities

- Translate `sqlparser` ASTs into strongly typed plan enums for DDL, DML, and queries.
- Encode correlated subqueries, scalar programs, and parameter bindings used during execution.
- Provide canonical representations (e.g., `CanonicalRow`, `CanonicalScalar`) for hashing and deduplicating expression inputs.

## Select Plan Structure

- `SelectPlan` carries projection lists, filters, sort clauses, limits, and expression programs compiled from [`llkv-expr`](../llkv-expr/).
- Correlated subqueries are captured as `ScalarSubquery` or `FilterSubquery` entries, pairing placeholder metadata with outer column bindings.
- Join plans describe join type, keys, and required columns so [`llkv-executor`](../llkv-executor/) can assemble row streams efficiently.

## DML and DDL Plans

- `InsertPlan` encapsulates Arrow `RecordBatch` sources, conflict policies, and column mappings consumed by [`llkv-runtime`](../llkv-runtime/).
- Update/Delete plans carry predicate programs plus column projections for the runtime to apply changes with MVCC bookkeeping.
- Catalog operations (create table, create index, etc.) surface schema definitions used by [`llkv-table`](../llkv-table/) and the system catalog.

## Correlation Utilities

- `llkv_plan::correlation` centralizes placeholder generation and column mapping for correlated subqueries, ensuring planner decisions remain consistent across statements.
- Shared helpers prevent duplication and keep higher layers focused on execution logic rather than subquery plumbing.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
