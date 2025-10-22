# Runtime Simplification Plan (Updated October 22, 2025)

## Executive Summary
- The autocommit helper refactor is in place for all DML and DDL paths; `RuntimeSession` now fans out through shared helpers instead of duplicating logic across the session/context boundary.
- `RuntimeTransactionContext` has been hardened (including NoOp propagation) so transactional replay accepts the new helper flow.
- Runtime namespace adapters have been split into `runtime_storage_namespace::{mod.rs, namespace.rs}`, renamed to `RuntimeStorageNamespace`, and backed by a central registry that tracks table ownership; temporary cleanup now drains through the registry.
- The largest remaining risks live inside `RuntimeContext`, where storage, catalog, and MVCC responsibilities are still interleaved. Next focus: finish the surface audit and migrate catalog-oriented helpers.

## Current Progress Snapshot
- ✅ DML autocommit helpers (`insert`/`update`/`delete`/`select`) centralized in `RuntimeSession`.
- ✅ DDL helpers (`create/drop table`, `rename/alter table`, `create/drop index`) now share the same execution path and emit supported `StatementResult`s.
- ✅ `RuntimeTransactionContext` conversion map accepts `RuntimeStatementResult::NoOp`, unblocking sqllogictests that exercise DROP paths.
- ✅ Runtime namespace adapters trimmed: registry-backed `RuntimeStorageNamespace` module keeps persistent/temp routing thin and documented.
- ⚠️ `RuntimeContext` still exposes catalog utilities, view/type helpers, and MVCC plumbing in one giant surface area.
- ⚠️ Full workspace regression pass (slt + unit suites) still outstanding after the namespace refactor.

## Reality Check
Even after the helper refactor, the storage layer owns concerns that belong in the catalog crate:
- Type registry, view management, and catalog snapshot access can move into `llkv-table` so they are reusable by tooling and keep the runtime facade thin.
- Table caching, MVCC visibility, and transaction replay must stay in `llkv-runtime`; these are the pieces that coordinate with `llkv-transaction` and the namespace adapters.
- The architectural target remains a clean pipeline: `Engine → RuntimeSession → RuntimeTransactionContext → RuntimeContext → llkv-table`.

### Target Architecture
```
User / SQL Frontend
    ↓
RuntimeSession (public API, transaction lifecycle, namespace routing)
    ↓
RuntimeTransactionContext (snapshot + replay adapter)
    ↓
RuntimeContext (storage + MVCC primitives only)
    ↓
llkv-table (catalog, metadata, column store)
```

`RuntimeContext` should be usable without any session scaffolding when given a snapshot. Everything else—temporary namespaces, locking, autocommit decisions—belongs above it.

## Roadmap & Status
| Phase | Focus                           | Status        | Notes                                                                                                   |
| ----- | ------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------- |
| 0     | Emergency assessment            | ✅ Complete    | Root cause documented; helper refactor kicked off the cleanup.                                          |
| 1     | Session-side helper unification | ✅ Complete    | All CRUD/DDL entry points in `RuntimeSession` delegate through shared autocommit helpers.               |
| 2     | Transaction context alignment   | ✅ Complete    | `RuntimeTransactionContext` translation table updated; sqllogictests green.                             |
| 3     | RuntimeContext surface audit    | 🟡 In progress | Catalog/type helpers still interleaved; classification list underway. Rescope with namespace learnings. |
| 4     | Namespace adapter trim          | ✅ Complete    | Registry-backed `RuntimeStorageNamespace` split landed; persistent/temp cleanup flows through registry. |
| 5     | SQL frontend consolidation      | ⏳ Not started | Force all statements through the session API; no direct context calls.                                  |
| 6     | Documentation + guides          | ⏳ Not started | Update feature placement guide and add diagrams after code stabilizes.                                  |

## Immediate Objectives (Next 24 Hours)
1. **Finish RuntimeContext surface classification**
    - Complete the method inventory started in Phase 3 and decide destination (`llkv-table`, runtime-only, or private helpers).
    - Sketch the `llkv-table` API additions needed for type registry + view support so downstream crates can adopt them.
2. **Backfill validation on namespace refactor**
    - Run targeted `llkv-runtime` + `llkv-sql` test suites and the sqllogictest harness to confirm registry plumbing.
    - Add regression tests covering temp+persistent lifecycles that rely on the new drain/registry behavior.
3. **Stage migration work tickets**
    - Break the remaining `RuntimeContext` moves into reviewable PRs (catalog helpers → llkv-table → runtime cleanup).
    - Plan deprecation shims or feature flags for any APIs that will change shape during the migration.

## Opportunities to Push Into `llkv-table`
- **Type management**: move `register_type`, `drop_type`, and `resolve_type` into a catalog-facing API that persists through `SysCatalog`.
- **View creation**: expose `CatalogManager::create_view` / `is_view` so runtime becomes a thin proxy.
- **Metadata lookups**: surface `table_view`, `table_column_specs`, and `foreign_key_views` directly from `llkv-table`.
- **Column mutations**: keep table-ID resolution in the runtime but delegate rename/alter/drop logic to the catalog manager.

## Non-Negotiables
- `runtime_storage_namespace` remains the namespace routing layer. Goal: keep it slim, but **do not delete**—it prevents Persistent/Temporary duplication from leaking back into the session layer.
- MVCC snapshot handling, table caching, and transaction replay stay in `RuntimeContext` + `RuntimeTransactionContext`; moving them would entangle `llkv-table` with transaction specifics.

## Risks & Mitigations
| Risk                                                  | Impact | Mitigation                                                                                        |
| ----------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------- |
| API churn in `RuntimeContext` breaks dependent crates | High   | Stage work: add new APIs, migrate call sites, then remove old signatures with deprecation window. |
| Namespace cleanup regresses temp-table semantics      | Medium | Add targeted tests covering temp + persistent mixes before refactoring adapters.                  |
| Time pressure (finish plan by tomorrow)               | Medium | Lock scope to documented phases; anything outside plan goes to backlog section.                   |

## Validation Strategy
- `cargo fmt` + `cargo check` per touched crate (`llkv-runtime`, `llkv-table`, `llkv-sql`).
- Targeted unit tests: runtime transaction replay, namespace adapters, catalog helpers.
- sqllogictests: run foreign key suites plus temp-table focused runs after each structural change.
- Smoke run for end-to-end SQL queries that mix temp + persistent tables to verify namespace routing.

## Backlog & Follow-Ups
- Consider introducing a `StorageEngine` trait once the surface area stabilizes, making it explicit what the runtime exposes.
- Revisit documentation in `dev-docs` (feature placement guide, high-level crate linkage) to mirror the new call chain.
- Evaluate whether utility crates (e.g., tooling, migration helpers) can consume the thinner runtime surface directly.

---
**Reminder:** every new feature should enter through `RuntimeSession`. If you catch yourself wiring up the same signature in both the session and context, stop—route it through the shared helper instead.
