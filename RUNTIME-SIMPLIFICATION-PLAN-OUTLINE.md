# Runtime Simplification Plan

## Goals
- Apply the feature placement guide across the runtime: single, well-named entry points per layer, zero duplicate helpers, and clear separation of concerns.
- Simplify the SQL execution path for *all* DDL/DML statements (CREATE/ALTER/DROP TABLE, CREATE/DROP INDEX, INSERT/UPDATE/DELETE, SELECT) so higher layers never bypass the plan pipeline.
- Keep behavior and performance intact while reducing debugging overhead and future maintenance cost.

## Scope Overview
- **Session Layer (`RuntimeSession`)**: expose one public method per statement that translates planner intents into namespace/context calls.
- **Namespace Layer (`storage_namespace`)**: act as thin adapters that forward to context-level implementations without additional logic duplication.
- **Context Layer (`RuntimeContext`, `TransactionContext`)**: concentrate catalog + storage operations behind clearly named, scoped helpers.
- **SQL Frontend (`llkv-sql`)**: ensure every statement path builds plans and routes through the session API; remove remaining shortcuts.
- **Documentation & Tests**: align comments with the comment style guide and keep the full test suite passing.

## Deliverables
1. Naming/visibility overhaul for all session-level statement handlers.
2. Consistent namespace adapters covering create/drop/alter/index operations.
3. Catalog-side helpers renamed and restricted (`drop_*`, `create_*`, `alter_*`) with explicit contracts.
4. SQL engine updated to rely exclusively on plan execution, including schema-level commands.
5. Doc comments, design notes, and guide references refreshed to match the new structure.
6. Green builds via `cargo fmt`, `cargo check`, targeted `cargo test`, and sqllogictest runs.

## Work Breakdown

### 1. Session Layer Rationalization
- Audit every public method on `RuntimeSession` (create/drop/alter table, create/drop index, insert/update/delete/select, transaction controls).
- Standardize naming (e.g., `execute_create_table_plan`, `execute_drop_index_plan`, `execute_insert_plan`) or adopt a consistent alternative prefix (`handle_*`).
- Remove legacy convenience helpers (`drop_table`, direct namespace calls) and ensure `RuntimeEngine::execute_statement` covers all statements through the unified set.
- Document the call chain (Engine → Session → Namespace → Context) at the top of the module.

### 2. Namespace Adapter Cleanup
- Update `PersistentNamespace` and `TemporaryNamespace` to offer only the trait-mandated methods.
- Ensure each method forwards directly to context helpers (created/renamed in Step 3) without extra branching.
- Revisit temporary namespace behavior for unsupported features (e.g., DROP INDEX inside transactions) and surface meaningful errors via the session layer.

### 3. Context-Level Consolidation
- Rename and scope catalog/storage helpers:
	- `drop_table_immediate` → `drop_table_catalog_side` (or similar) with `pub(crate)` visibility.
	- Inspect other helpers (`create_table_plan`, `rename_table_immediate`, index methods) and align naming/visibility.
- Centralize shared validation (FK checks, cache eviction, lazy reload) into reusable utilities where appropriate.
- Update `TransactionContext` replay logic to depend on the renamed helpers.

### 4. SQL Frontend Alignment
- Route all `DROP`, `CREATE`, `ALTER`, `INSERT`, `UPDATE`, `DELETE`, and `SELECT` commands through `execute_plan_statement`.
- Remove direct namespace/context calls (e.g., schema cascade path, temporary table shortcuts).
- Confirm plan construction covers options (IF EXISTS, CASCADE, temp namespaces) and surface consistent error messaging.

### 5. Documentation & Guide Synchronization
- Refresh doc comments in session/context/namespace modules per `dev-docs/comment-style-guide.md`.
- Update `dev-docs/feature-placement-guide.md` examples if the layering adjustments warrant it.
- Consider adding a short `docs/runtime-drop-flow.md` (or similar) if inline documentation becomes too dense.

### 6. Validation & Regression Safety
- `cargo fmt` across touched crates.
- `cargo check -p llkv-runtime`, `cargo check -p llkv-sql`, and any affected crates (e.g., `llkv-plan` if signatures change).
- `cargo test -p llkv-runtime --lib` plus targeted module tests (drop metadata, transaction replay, namespace behavior).
- Run sqllogictests focusing on DDL-heavy suites to confirm no behavioral regressions.

## Pending Questions / Assumptions
- Naming convention: settle on `execute_*` vs `handle_*` vs `apply_*` after inspecting all statement handlers for consistency.
- Temporary namespace support: validate which statements should be rejected vs rerouted.
- Transaction limitations (e.g., DROP INDEX) should remain enforced but with clearer messaging.
- Any follow-up refactors (e.g., builder patterns for plans) will be logged as separate tasks to avoid scope creep.

## Next Steps
1. Apply Steps 1–3 within `llkv-runtime`, adjusting APIs and visibility.
2. Update `llkv-sql` and other crates (`llkv-transaction`, tests) to match the new call chain.
3. Refresh documentation and run the validation matrix.
4. Summarize outcomes, capture follow-up items, and proceed to implementation phases.
