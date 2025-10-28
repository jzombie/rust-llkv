# LLKV Plan

**Work in Progress**

`llkv-plan` is the query planner crate for the [LLKV](../) toolkit.

## Purpose

- Define execution plan structures for SQL operations (SELECT, INSERT, UPDATE, DELETE, CREATE TABLE).
- Convert SQL AST from [sqlparser](https://crates.io/crates/sqlparser) into typed execution plans.
- Provide plan data structures used by [`llkv-runtime`](../llkv-runtime/) and [`llkv-executor`](../llkv-executor/).

## Design Notes

- Plans are consumed by [`llkv-sql`](../llkv-sql/) which parses SQL, creates plans, and delegates to [`llkv-runtime`](../llkv-runtime/) for execution.
- The crate focuses on plan representation and does not perform query optimization or execution.
- Shared correlation helpers under `llkv_plan::correlation` manage placeholder assignment for scalar and EXISTS subqueries so higher layers do not duplicate tracking logic.
- `SelectPlan` now carries structured metadata (`SelectFilter`, `ScalarSubquery`, and `FilterSubquery`) to surface correlated subqueries alongside their captured column maps.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
