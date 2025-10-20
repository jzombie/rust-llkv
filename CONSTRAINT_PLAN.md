# Constraint System Redesign Plan (Issue #124)

## Assessment Snapshot

- Foreign key metadata currently lives only in `RuntimeContext::foreign_keys` and is lost on restart; enforcement depends on string lookups instead of stable IDs.
- Column-level constraints (primary key, unique, check) are tracked in `TableCatalog` state that is serialized wholesale (`CATALOG_FIELD_CATALOG_STATE`), preventing granular persistence or batch queries.
- Multi-column unique metadata is persisted separately but re-hydrated into ad-hoc per-table vectors guarded by locks, leading to yet another constraint path.
- Table and column metadata reads/writes are spread across bespoke pager calls in `llkv-table` and `llkv-runtime`, complicating thread-safety and making change detection difficult.
- Current APIs expose mutable vectors and string-heavy structs, which blocks zero-copy sharing and encourages per-call cloning.

## Architectural Direction

- Introduce a `MetadataManager` in `llkv-table` that encapsulates all system catalog access for table, column, and constraint metadata. Writers get diff-aware commit helpers; readers obtain thread-safe snapshots.
- Define unified constraint descriptors (`ConstraintId`, `ConstraintKind`, `ConstraintScope`) that store only table/field IDs plus action enums. Display names are resolved lazily through the catalog when building error messages.
- Store every constraint as an individual bitcode row under a dedicated catalog field (e.g. `CATALOG_FIELD_CONSTRAINT_META_ID`), using `(table_id, constraint_id)` derived row IDs so multiple tables can be fetched in a single batch.
- Replace the legacy global blobs (`CATALOG_FIELD_CATALOG_STATE`, `CATALOG_FIELD_MULTI_COLUMN_UNIQUE_META_ID`, etc.) with managed, per-constraint rows; the manager will keep reading the old blobs temporarily but all new writes land in the unified store.
- Provide batch-oriented APIs that hand back shared slices (`Arc<[ConstraintRecord]>`) to consumers, avoiding per-call allocations while keeping the interface zero-copy friendly while still supporting streaming iteration so large catalogs never require full materialization.
- Replace ad-hoc registries in `llkv-runtime` (foreign keys, multi-column unique caches, scattered column metadata loads) with manager-backed views that can be shared across threads.
- Unify column-level and table-level constraint registration in `llkv-sql`/`llkv-runtime`, emitting canonical descriptors once and delegating persistence to the manager.

## Execution Phases

1. **Foundations**
   - Add the new constraint data model and serialization primitives to `llkv-table`.
   - Extend `SysCatalog` with constraint read/write helpers and batch loaders.
   - Stand up the shared `MetadataManager`, covering table metadata, column metadata, multi-uniques, and constraints.
   - Route `RuntimeContext` table creation/CTAS flows and foreign-key persistence through the manager (no more ad-hoc `SysCatalog` writes in those paths).
2. **Runtime Integration**
   - Wire `RuntimeContext` to consume the manager for table registration, constraint loading, and persistence.
   - Retire `ForeignKeyRegistry` and similar ad-hoc maps in favor of manager snapshots built on precomputed `Arc` views that refresh under write locks.
   - ✅ Lazy table loads now rebuild executor schemas from `MetadataManager` snapshots, eliminating the remaining direct `SysCatalog` reads and ensuring column/constraint metadata comes from a single source. Empty-table type resolution was hardened by extending the column store dtype cache so descriptors expose their Arrow types immediately after registration.
   - ✅ Drop flows now route through the manager: `RuntimeContext::drop_table_immediate` collects the registered columns, clears table/column/constraint snapshots via `MetadataManager::prepare_table_drop`, flushes the diffs, and nulls out persisted multi-column unique rows. Catalog state and foreign key caches are pruned in the same pass so restarts see the table as fully removed.
   - ✅ `RuntimeContext::table_column_specs` now delegates to `CatalogService::table_column_specs`, synthesizes specs from persisted column metadata and constraint records while falling back to resolver state only when no catalog records exist.
   - ✅ `RuntimeContext::table_view` now calls into `CatalogService::table_view`, keeping read paths aligned with the catalog service and avoiding direct metadata access from the runtime layer.
   - ✅ `CatalogService::foreign_key_views` (with `RuntimeContext::foreign_key_views`) surfaces persisted FK metadata without touching metadata internals, keeping read access aligned with the table-layer helpers.
   - ✅ `CatalogService::table_constraint_summary` now feeds runtime table loading, so constraint and multi-unique metadata no longer come directly from `MetadataManager::table_view`.
3. **SQL Frontend Cleanup**
   - Update `llkv-sql` planning/execution to emit constraint descriptors that reference IDs instead of strings.
   - Ensure column and table constraint syntax follow the same plumbing path, eliminating one-off handlers.
4. **Persistence & Change Detection**
   - Introduce diffing so metadata writes only occur when content changes.
   - Guarantee snapshot refresh + invalidation semantics for constraint updates across threads.
   - Migrate multi-column unique metadata, primary keys, checks, and foreign keys into the shared constraint store; remove the redundant catalog blobs once reads are redirected.
5. **Validation**
   - Extend coverage with restart/persistence tests and concurrent read checks.
   - Regularly run `cargo test --workspace` plus the SLT suites under `llkv-sql/tests/slt/duckdb/constraints` and related transaction tests during the refactor.

## Open Questions

- Define the cutover plan for retiring `CATALOG_FIELD_CATALOG_STATE` while maintaining compatibility during the transition; the goal is to drop the blob entirely once the manager becomes the sole writer.
- Confirm the concurrency primitives (e.g. `RwLock` plus snapshot caching or other standard-library approaches) after profiling contention paths in the new manager.
