# API Refactor Opportunities

This note captures follow-on cleanup items that surfaced while centralising constraint
metadata. Each item points out logic currently implemented in higher layers that would
benefit from moving closer to the storage/metadata crates, along with why the change
would help.

## RuntimeContext hotspots

- ✅ **Uniqueness helpers** (`llkv-table/src/constraint_validation.rs:77-137`): shared
  functions now convert `PlanValue` rows into reusable `UniqueKey`s and perform multi-column
  uniqueness checks, with the runtime acting as a thin caller.
- ✅ **CHECK expression evaluation** (`llkv-table/src/constraint_validation.rs:31-207`):
  CHECK parsing and evaluation moved to the table layer, eliminating the bespoke SQL AST walker
  in the runtime.
- ✅ **Foreign-key metadata assembly** (`llkv-table/src/metadata.rs:532-574`):
  `MetadataManager::foreign_key_descriptors` materialises referencing/referenced names so the
  runtime only resolves schema indices.
- ✅ **Index registration/unregistration** (`llkv-table/src/metadata.rs:318-371`):
  staging sort-index adds/removals in the metadata snapshot lets `flush_table` batch the catalog
  work, and the runtime now routes single-column CREATE INDEX through the helper path.
- ✅ **Schema/constraint batching** (`llkv-runtime/src/lib.rs:2346-2414`): CREATE TABLE now stages
  table and column metadata entirely through `MetadataManager`, so multi-column definitions persist
  in a single `flush_table` call with no per-column catalog writes.
- ✅ **Multi-column UNIQUE registration**: `MetadataManager::register_multi_column_unique` now owns
  catalog updates for composite unique keys. Runtime simply validates existing data and delegates the
  metadata write, avoiding the bespoke closure previously used with `update_multi_column_uniques`.

## SQL planning duplication

The SQL engine in `llkv-sql/src/sql_engine.rs` has internal duplication that should be addressed:

- **Column extraction from SQL AST**: The pattern of extracting column names from `OrderBy` expressions and validating them (checking for operator classes, sort options, compound identifiers) appears 3+ times in `handle_create_table` (for PRIMARY KEY and UNIQUE constraints) and `handle_create_index`. This could be consolidated into a helper function within the SQL engine module.

- **Column validation already shared**: The `llkv-plan::validation` module successfully provides `ensure_non_empty`, `ensure_unique_case_insensitive`, and `ensure_known_columns_case_insensitive` utilities that are already used by the SQL engine to eliminate duplicate validation logic.

- **SQL-specific validations**: Most of the remaining validation logic in `handle_create_table` and `handle_create_index` is inherently SQL AST-specific (checking unsupported SQL features like `CONCURRENTLY`, `USING`, `NULLS DISTINCT`, etc.) and cannot easily move to `llkv-plan` since that crate doesn't depend on `sqlparser`. These validations serve as a translation layer from SQL syntax to plan structures and appropriately belong in the SQL engine.

## Status Snapshot

- ✅ `ConstraintService` now enforces foreign keys for INSERT/DELETE; runtime simply streams rows/snapshots.
- ✅ `ConstraintService` owns PK/UNIQUE/CHECK enforcement for INSERT/UPDATE/commit flows; runtime just forwards executor schemas + row batches.
- ✅ `MetadataManager::table_view` + runtime lazy-load: executor schemas now come from a single consolidated snapshot.
- ✅ SQL `CREATE INDEX` column validation and FOREIGN KEY checks in `CREATE TABLE` now reuse `TableView` data.
- ✅ `CatalogService::create_table_from_columns` handles metadata + catalog staging for regular CREATE TABLE; runtime only materialises executor caches and optional FK wiring.
- ✅ `CatalogService::create_table_from_schema` backs CTAS flows; runtime simply wires executor caches after the service returns metadata.
- ✅ `CatalogService::append_batches_with_mvcc` handles CTAS data staging (injecting MVCC columns and appending batches) with transactional rollback if the runtime sees an error.
- ✅ `CatalogService::create_single_column_index` performs index-existence checks, metadata registration, and catalog updates for single-column CREATE INDEX; runtime validates data uniqueness (when needed) and updates executor caches.
- ✅ `CatalogService::register_multi_column_unique_index` performs the metadata/catalog work for multi-column UNIQUE indexes; runtime validates existing data and updates executor caches.
- ✅ `CatalogService::register_foreign_keys_for_new_table` wraps FK registration during CREATE TABLE so the runtime only supplies closures that resolve referenced tables.
- ✅ `CatalogService::drop_table` centralises metadata teardown and catalog unregister so runtime drop only enforces FK safety and cache updates.
- ⚙️ Runtime now only stages executor caches for CREATE TABLE/CTAS; metadata, catalog writes, and MVCC batch seeding are handled by `CatalogService`.
- ✅ `CatalogService::table_column_specs` rebuilds column specs from persisted metadata and constraint records, giving runtime callers restart-stable constraint flags without touching executor caches.
- 🚧 Remaining runtime helpers (catalog-derived read views, executor cache rebuilds) still live in `llkv-runtime`.
- ⚙️ SQL planner/CREATE flows now reuse shared FK validation helpers, but further consolidation (e.g., UNIQUE checks) remains.
- ✅ Catalog read APIs (column specs, table view, foreign-key views, constraint summaries) now route through the catalog service instead of touching metadata snapshots directly.
- ✅ Shared validation utilities (`llkv_plan::validation`) centralise column/FK/PK/UNIQUE shape checks so frontends can reuse them without SQL-specific logic.
- ✅ SQL planner column-resolution (`collect_known_columns`) now sources metadata via `RuntimeContext::table_column_specs`, avoiding ad-hoc `table_view` + `lookup_table` fallbacks.
- ✅ SQL engine internal cleanup: consolidated the repeated "extract column name from OrderBy expression" pattern into `extract_index_column_name` helper within `sql_engine.rs`, eliminating duplicate AST validation code across PRIMARY KEY, UNIQUE, and CREATE INDEX handlers.

### In-flight
- Persistence + change detection polish:
  - Add diff-aware metadata writes so `flush_table` skips no-op updates.
  - Expose snapshot invalidation semantics for constraint updates across threads.
  - Complete migration of multi-column unique, PK, CHECK, and FK metadata into the unified constraint store and remove the legacy catalog blobs once reads are redirected.
- Validation coverage:
  - Expand restart/persistence tests that exercise constraint snapshots.
  - Keep SLT suites under `llkv-sql/tests/slt/duckdb/constraints` and related transaction tests in the loop during refactors.

### Catalog service lift – runtime responsibilities inventory

The `CatalogService` now handles most CREATE/DROP/index metadata and catalog operations. The runtime's remaining responsibilities for these operations are:

- `RuntimeContext::create_table_from_columns`: delegates metadata + catalog registration to `CatalogService::create_table_from_columns`, then constructs executor caches; FK wiring is a thin call into `CatalogService::register_foreign_keys_for_new_table` with runtime closures.
- `RuntimeContext::create_table_from_batches`: delegates metadata + CTAS batch appends to `CatalogService::create_table_from_schema` and `CatalogService::append_batches_with_mvcc`; runtime just updates executor caches and tracks transactions.
- `RuntimeContext::create_index`: delegates index-existence checks and metadata registration to `CatalogService::create_single_column_index` or `CatalogService::register_multi_column_unique_index`; runtime resolves columns, enforces uniqueness against live data, and refreshes executor caches.
- `RuntimeContext::drop_table_immediate`: delegates metadata removal + catalog unregister to `CatalogService::drop_table`; runtime enforces FK safety and tracks dropped-table cache locally.
- Supporting helpers such as `remove_table_entry`, `rebuild_executor_table_with_unique` remain in the runtime for executor cache management.

The catalog service successfully centralizes metadata/catalog orchestration while the runtime focuses on transaction orchestration and executor caching.

### Up Next

> **Current checkpoint:** CREATE/DROP/index responsibilities now delegated to CatalogService. Runtime focuses on transaction orchestration, data validation with live snapshots, and executor cache management. SQL engine internal cleanup completed—column extraction from SQL AST consolidated into reusable helper.

1. **Persistence polish** – add diff-aware metadata writes and snapshot invalidation semantics for constraint updates.
2. **Module cleanup** – align naming (`types.rs`, service modules) once the API surface settles.

---

## Runtime API debt (detailed)

The runtime is still packed with thin wrappers that merely forward to `TableCatalog`,
`MetadataManager`, or the constraint helpers. These wrappers make it impossible to know
which layer owns what. The goal is to relocate *all* catalog / constraint / metadata
operations into `llkv-table`, leaving the runtime responsible only for:

- orchestrating transactions (begin/commit/rollback)
- planning/executing operations via the executor
- marshalling rows and plan statements

Everything else should move down. The main offenders are listed below.

### 1. Table creation / deletion pipeline

| Runtime Method                                                    | What it currently does                                                                                    | Desired destination                                                                                                      |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `create_table_from_columns` (implicit inside `create_table_plan`) | Allocates table id, writes table/column metadata, registers catalog entries, sets up MVCC columns         | `MetadataManager::create_table_with_columns` (new helper) plus a `TableBuilder` that returns `ExecutorTable` metadata    |
| `drop_table_immediate`                                            | Looks up referencing tables, checks FK constraints, calls `prepare_table_drop`, unregisters catalog entry | `MetadataManager::drop_table` should own referencing lookups + FK enforcement (with runtime just passing canonical name) |
| `reserve_table_id` (now removed but historically here)            | Reserved ids out of SysCatalog                                                                            | Already moved to `MetadataManager::reserve_table_id`, but more context needed (doc + dedicated API)                      |

### 2. Foreign key lifecycle

Current state:
- Plan-level specs validated and persisted via `MetadataManager::validate_and_register_foreign_keys` (✅ done)
- Runtime still computes column metadata, resolves referenced table schemas.

Next steps:
- Provide `MetadataManager::prepare_foreign_key_tables` that takes referencing table id and returns a struct containing:
  - referencing column metadata (name/type/nullable/unique/pk)
  - closure/callback for resolving referenced table metadata
- Move `foreign_keys_referencing` / `foreign_keys_for_table` into `llkv-table` so runtime just receives ready-to-use metadata for constraint enforcement and error messages.

### 3. Unique / index metadata

Wrappers that should move:
- `MetadataManager::register_sort_index` and `flush_table` already exist, but runtime still stages new `ExecutorMultiColumnUnique` entries.
- Create a table-layer helper that:
  1. Validates requested unique indices.
  2. Persists metadata via `MetadataManager::update_multi_column_uniques`.
  3. Returns the executor-facing `ExecutorMultiColumnUnique` list.
- Runtime then updates its in-memory table cache with the precomputed structures, no custom logic.

### 4. Constraint checks on INSERT/UPDATE/DELETE

Remaining work:
- `ensure_existing_rows_unique`, `ensure_existing_rows_unique_multi`, `validate_primary_keys_for_commit`, and the CHECK / FK validations in insert/update should be packaged as `ConstraintValidator` APIs in `llkv-table`.
- Pass the executor schema + new rows + operation type into the table crate, get back either `Ok` or a descriptive constraint error.
- Runtime should not parse expressions or inspect plan rows directly.

### 5. Catalog snapshots / lookups

`RuntimeContext::table_column_specs`, `export_table_rows`, `lookup_table`, etc., are
light wrappers around `TableCatalog` or `MetadataManager`. Preferred approach:

1. Expose read-only catalog views from `llkv-table` (e.g. `CatalogReader`).
2. Runtime depends only on that interface rather than touching metadata directly.

This gives a single, well-documented surface layer for all schema-related reads.

### 6. Naming / module structure

While refactoring, align the structure across crates:

- `types.rs` contains public structs/enums.
- `services/` (or similar) holds the catalog/constraint helpers.
- `runtime` should only see high-level builders/services without diving into individual modules.

---

## Refactor roadmap (high level)

1. ✅ **Constraint Service**: consolidate FK/PK/UNIQUE/CHECK validation into a table-layer
   service. Runtime supplies input rows + executor schema; service returns detailed errors.
2. **Catalog Service**: move table create/drop/index operations into `llkv-table`. Runtime
   just calls `catalog_service.create_table(...)`.
3. **Metadata Views**: expose read-only views for column metadata / table metadata / FK details.
   Runtime reads from those views instead of manipulating metadata snapshots directly.
4. **Executor Integration**: provide helpers to build `ExecutorTable` and `ExecutorSchema`
   from catalog metadata, making lazy loading symmetrical with initial creation.
5. **SQL Planning**: after lifting the runtime wrappers, revisit `llkv-sql` to reuse the same
   validation helpers and further minimize duplicate logic.

This re-architecture will finally give us a single authoritative API in `llkv-table`, with
`llkv-runtime` focusing solely on orchestration and execution.

## Open Questions

- How do we safely retire `CATALOG_FIELD_CATALOG_STATE` while keeping mixed-version upgrades smooth?
- Are the current locking primitives (`RwLock` snapshots, etc.) sufficient once constraint diffing lands, or do we need a more explicit cache invalidation layer?
