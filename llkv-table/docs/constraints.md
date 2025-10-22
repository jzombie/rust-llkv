# LLKV Constraint System

This document explains how LLKV stores, enforces, and exposes table constraints. It is
intended both for users who want to understand the feature set and for contributors
who need architectural context before extending the system.

## Supported Constraint Types

LLKV currently supports the following constraint categories:

- **Primary key** – ensures uniqueness and non-nullability for one or more columns.
  Represented by `ConstraintKind::PrimaryKey` with ordered field IDs.
- **Unique** – single-column or composite uniqueness without enforcing NOT NULL.
- **Foreign key** – references another table and optionally specifies `ON DELETE` /
  `ON UPDATE` actions (currently `NO ACTION` or `RESTRICT`).
- **Check** – arbitrary boolean expressions stored as serialized SQL strings.

The concrete payloads live in `llkv-table/src/constraints.rs`. Each constraint persisted
in the catalog receives a stable `ConstraintId` that is scoped per table.

## Metadata Persistence

Constraint metadata is managed by `MetadataManager` inside the `llkv-table` crate. The
manager is responsible for:

1. Serialising constraint records into the system catalog (table 0) using bitcode.
2. Resolving human-friendly names via `TableCatalog` when producing read views.
3. Emitting table-scoped snapshots that combine table metadata, column metadata,
   constraint records, foreign keys, and multi-column unique descriptors.

Two public snapshots are relevant:

- `TableView` – includes everything (constraints + foreign keys). Runtime uses this for
  operations that need FK information.
- `TableConstraintSummary` – introduced for read-only constraint metadata (table meta,
  column meta, constraint records, multi-column uniques) without foreign keys. Runtime
  lazy-loading calls this through `CatalogService::table_constraint_summary`.

Constraint records are stored as individual rows keyed by `(table_id, constraint_id)`,
which allows batch retrieval and diff-based updates.

## CatalogService Entry Points

`CatalogService` in `llkv-table` exposes higher-level APIs that orchestrate metadata,
catalog registration, and storage work:

- `create_table_from_columns` / `create_table_from_schema` – write table & column metadata,
  register LogicalFieldId descriptors, and hand back `CreateTableResult`.
- `register_foreign_keys_for_new_table` – validates FK specs (via `ConstraintService`),
  persists them, and returns friendly view structs.
- `table_column_specs` – rebuilds restart-stable `PlanColumnSpec` instances with PK/unique/check
  flags resolved from metadata.
- `table_view` – returns the full `TableView` snapshot.
- `table_constraint_summary` – returns constraint-only metadata snapshots.
- `foreign_key_views` – returns hydrated `ForeignKeyView` structs with display/canonical names.

Runtime and SQL layers should **always** use these helpers instead of touching
`MetadataManager` directly.

## Enforcement

Constraint enforcement is centralised in `ConstraintService`:

- Insert/update flows call `validate_insert_foreign_keys`, `validate_check_constraints`,
  and uniqueness helpers.
- Delete flows call `validate_delete_foreign_keys`.

These APIs accept executor schemas and row batches so that higher layers stay free of
constraint-specific logic.

## Runtime Integration

`RuntimeContext` now treats `CatalogService` as the canonical interface:

- `table_column_specs`, `table_view`, and `foreign_key_views` forward to catalog service.
- Lazy table loading (`lookup_table`) uses `table_constraint_summary` to rebuild executor
  schemas, ensuring constraint flags are derived from persisted records.
- `RuntimeCreateTableBuilder` / `apply_create_table_plan` route CREATE TABLE + FK registration
  through catalog service, keeping metadata writes inside `llkv-table`.

This split lets the runtime focus on execution and transaction orchestration.

## Foreign Key Lifecycle

1. During `CREATE TABLE`, SQL planning gathers FK clauses and serialises them into
   `ForeignKeySpec`.
2. Runtime hands the specs to `CatalogService::register_foreign_keys_for_new_table`.
3. The service validates the references (resolving table IDs and column names), persists
   constraint records, and returns `ForeignKeyView` structures.
4. Reads use `catalog_service.foreign_key_views(...)` to obtain the same view after restart.

The FK metadata is therefore durable, restart-safe, and shared across runtime instances.

## Testing and Validation

Key test suites that exercise constraint behaviour:

- `llkv-runtime/tests/constraint_metadata_tests.rs` – verifies restart persistence for PK/UNIQUE
  metadata and foreign key views.
- SLT suites under `llkv-sql/tests/slt/duckdb/constraints` – re-run regularly to validate
  planner/runtime integration.
- Targeted unit tests inside `llkv-table` (e.g., `constraint_service.rs`, `metadata.rs`) that
  cover registration and enforcement logic.

Running `cargo test --workspace` plus the SLT harness is recommended after modifying constraint
code.

## Extending the System

When adding new constraint features or metadata:

1. Extend the data model in `llkv-table/src/constraints.rs`.
2. Teach `MetadataManager` how to persist/load the new payload.
3. Expose the necessary read helpers from `CatalogService`.
4. Route runtime and SQL layers through those helpers rather than introducing new ad-hoc
   metadata access.
5. Extend `ConstraintService` if runtime enforcement logic is required.
6. Update this document so future contributors understand the current architecture.
