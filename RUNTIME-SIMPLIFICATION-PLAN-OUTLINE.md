# Runtime Simplification Plan

## CRITICAL STATUS UPDATE (October 21, 2025)

**Current State: Architecture in Crisis**

The modularization effort has exposed a fundamental architecture problem that's been making the codebase progressively worse. Despite good intentions and the Feature Placement Guide, we have:

1. **Massive Role Duplication**: `RuntimeContext` and `RuntimeSession` both expose nearly identical APIs for all CRUD operations (insert, update, delete, select)
2. **Violated Separation of Concerns**: `RuntimeContext` has grown session-level responsibilities (`has_active_transaction()`, transaction-aware methods) when it should only handle storage
3. **Confused Call Chains**: RuntimeSession calls RuntimeContext, but RuntimeContext also has transaction awareness, creating circular dependencies in responsibilities
4. **Guide Violation**: Feature Placement Guide explicitly says: *"If you catch yourself wiring up the same method signature twice in different files or crates, stop and reassess (the architecture is likely wrong and needs refactoring rather than another copy)."* - This is exactly what we have.

**The Core Problem**: RuntimeContext grew beyond its intended scope as a storage engine layer and now competes with RuntimeSession instead of serving it.

## Goals (REVISED)
- **PRIMARY**: Fix the fundamental layer separation between RuntimeContext (storage) and RuntimeSession (session/transaction management)
- **SECONDARY**: Apply the feature placement guide across the runtime: single, well-named entry points per layer, zero duplicate helpers, and clear separation of concerns
- Simplify the SQL execution path for *all* DDL/DML statements (CREATE/ALTER/DROP TABLE, CREATE/DROP INDEX, INSERT/UPDATE/DELETE, SELECT) so higher layers never bypass the plan pipeline
- Keep behavior and performance intact while reducing debugging overhead and future maintenance cost

## Scope Overview (UPDATED)

### Intended Architecture (What We Need):
```
User/SQL Frontend
    ↓
RuntimeSession (Transaction/Session Management)
    ↓
RuntimeTransactionContext (Transaction State)
    ↓
RuntimeContext (Storage Engine - NO transaction awareness)
    ↓
Tables/Catalog/Storage
```

### Current Broken Architecture (What We Have):
```
User/SQL Frontend
    ↓
RuntimeSession (execute_insert_plan, execute_select_plan, etc.)
    ↓                    ↘
RuntimeContext ←→ Transaction State (WRONG!)
    ↓ (has insert(), select(), has_active_transaction() - TOO MUCH!)
    ↓
Tables/Catalog/Storage
```

### Layer Responsibilities (CORRECTED):

**RuntimeSession** (Session/Transaction Layer):
- Transaction lifecycle (begin, commit, rollback, abort)
- Multi-namespace coordination (persistent + temporary)
- ALL user-facing execute_* methods
- Routing operations to correct namespace and context
- Transaction state management

**RuntimeContext** (Storage Engine Layer - MUST BE SIMPLIFIED):
- ❌ REMOVE: `has_active_transaction()` (belongs in session)
- ❌ REMOVE: Transaction-aware method variants (`insert()` vs `insert_with_snapshot()`)
- ✅ KEEP: Direct table access with explicit snapshot parameter
- ✅ KEEP: Catalog operations (create/drop/alter tables)
- ✅ KEEP: MVCC primitives (snapshot creation, row filtering)
- ✅ KEEP: Table metadata and schema management

**Current Duplication Examples:**
- `RuntimeContext::insert()` vs `RuntimeSession::execute_insert_plan()`
- `RuntimeContext::update()` vs `RuntimeSession::execute_update_plan()`
- `RuntimeContext::delete()` vs `RuntimeSession::execute_delete_plan()`
- `RuntimeContext::execute_select()` vs `RuntimeSession::execute_select_plan()`

All RuntimeContext methods should take explicit snapshot parameters and have NO transaction awareness.

## Deliverables
1. Naming/visibility overhaul for all session-level statement handlers.
2. Consistent namespace adapters covering create/drop/alter/index operations.
3. Catalog-side helpers renamed and restricted (`drop_*`, `create_*`, `alter_*`) with explicit contracts.
4. SQL engine updated to rely exclusively on plan execution, including schema-level commands.
5. Doc comments, design notes, and guide references refreshed to match the new structure.
6. Green builds via `cargo fmt`, `cargo check`, targeted `cargo test`, and sqllogictest runs.

## Work Breakdown (REVISED & PRIORITIZED)

### PHASE 0: Emergency Assessment (DONE)
- ✅ Identified fundamental architecture violation
- ✅ Documented RuntimeContext/RuntimeSession role confusion
- ✅ Confirmed duplication across all CRUD operations
- ✅ Recognized modularization made problem more visible (good!) but didn't fix root cause

### PHASE 1: RuntimeContext Surgery (CRITICAL - DO FIRST)
**Goal**: Remove all session/transaction concerns from RuntimeContext

**Step 1a: Remove Transaction Awareness from RuntimeContext**
- Remove `has_active_transaction()` method
- Remove dual-method pattern (e.g., `insert()` + `insert_with_snapshot()`)
- Make ALL data operations require explicit `TransactionSnapshot` parameter
- Example transformations:
  ```rust
  // BEFORE (WRONG - transaction-aware)
  pub fn insert(&self, plan: InsertPlan) -> Result<...>
  pub fn insert_with_snapshot(&self, plan: InsertPlan, snapshot: TransactionSnapshot) -> Result<...>
  
  // AFTER (CORRECT - snapshot required)
  pub fn insert(&self, plan: InsertPlan, snapshot: TransactionSnapshot) -> Result<...>
  ```

**Step 1b: Update RuntimeSession to Manage Snapshots**
- RuntimeSession creates/manages snapshots based on transaction state
- RuntimeSession passes snapshots to all RuntimeContext calls
- No RuntimeContext method should check transaction state internally

**Step 1c: Update RuntimeTransactionContext**
- Ensure it properly manages snapshot lifecycle
- Verify it passes snapshots through to RuntimeContext

**Validation**: After Phase 1, RuntimeContext should be compilable as a standalone storage engine with ZERO knowledge of transactions.

### PHASE 2: Session Layer Rationalization (DEPENDS ON PHASE 1)
### PHASE 2: Session Layer Rationalization (DEPENDS ON PHASE 1)
**Goal**: Make RuntimeSession the ONLY user-facing API for SQL operations

- Audit every public method on `RuntimeSession` (create/drop/alter table, create/drop index, insert/update/delete/select, transaction controls)
- Standardize naming (e.g., `execute_create_table_plan`, `execute_drop_index_plan`, `execute_insert_plan`) or adopt a consistent alternative prefix (`handle_*`)
- Remove legacy convenience helpers (`drop_table`, direct namespace calls) and ensure `RuntimeEngine::execute_statement` covers all statements through the unified set
- Document the call chain (Engine → Session → Namespace → Context) at the top of the module
- **Verify**: RuntimeSession has ALL the execute_* methods, RuntimeContext has NONE

### PHASE 3: Namespace Adapter Cleanup (DEPENDS ON PHASE 2)
### PHASE 3: Namespace Adapter Cleanup (DEPENDS ON PHASE 2)
**Goal**: Thin adapters that route to context, nothing more

- Update `PersistentNamespace` and `TemporaryNamespace` to offer only the trait-mandated methods
- Ensure each method forwards directly to context helpers (created/renamed in Phase 4) without extra branching
- Revisit temporary namespace behavior for unsupported features (e.g., DROP INDEX inside transactions) and surface meaningful errors via the session layer

### PHASE 4: Context-Level Consolidation (DEPENDS ON PHASE 3)
### PHASE 4: Context-Level Consolidation (DEPENDS ON PHASE 3)
**Goal**: Clean storage-only APIs with consistent naming

- Rename and scope catalog/storage helpers:
	- `drop_table_catalog` (formerly `apply_drop_table` / `drop_table_immediate`) – revisit visibility once integration tests stop depending on it
	- Inspect other helpers (`apply_create_table_plan`, `apply_rename_table`, index methods) and align naming/visibility
- Centralize shared validation (FK checks, cache eviction, lazy reload) into reusable utilities where appropriate
- Update `TransactionContext` replay logic to depend on the renamed helpers
- **All methods must accept explicit snapshot parameters - NO implicit transaction state**

### PHASE 5: SQL Frontend Alignment (DEPENDS ON PHASE 4)
### PHASE 5: SQL Frontend Alignment (DEPENDS ON PHASE 4)
**Goal**: All SQL goes through session layer

- Route all `DROP`, `CREATE`, `ALTER`, `INSERT`, `UPDATE`, `DELETE`, and `SELECT` commands through `execute_plan_statement`
- Remove direct namespace/context calls (e.g., schema cascade path, temporary table shortcuts)
- Confirm plan construction covers options (IF EXISTS, CASCADE, temp namespaces) and surface consistent error messaging

### PHASE 6: Documentation & Guide Synchronization (DEPENDS ON PHASE 5)
### PHASE 6: Documentation & Guide Synchronization (DEPENDS ON PHASE 5)
**Goal**: Documentation matches reality

- Refresh doc comments in session/context/namespace modules per `dev-docs/comment-style-guide.md`
- Update `dev-docs/feature-placement-guide.md` examples if the layering adjustments warrant it
- Add architecture diagram showing clear layer separation
- Document the "no transaction awareness in storage layer" principle

### PHASE 7: Validation & Regression Safety (AFTER EACH PHASE)
### PHASE 7: Validation & Regression Safety (AFTER EACH PHASE)
**Goal**: Continuous verification

- `cargo fmt` across touched crates
- `cargo check -p llkv-runtime`, `cargo check -p llkv-sql`, and any affected crates (e.g., `llkv-plan` if signatures change)
- `cargo test -p llkv-runtime --lib` plus targeted module tests (drop metadata, transaction replay, namespace behavior)
- Run sqllogictests focusing on DDL-heavy suites to confirm no behavioral regressions
- **CRITICAL**: Run full test suite after Phase 1 before proceeding

## Pending Questions / Assumptions (UPDATED)

**RESOLVED:**
- ✅ Architecture problem identified: RuntimeContext has session responsibilities it shouldn't have
- ✅ Duplication root cause: RuntimeContext exposes same API as RuntimeSession

**NEW QUESTIONS:**
- Should RuntimeContext::create_table and DDL operations also require snapshot parameters?
- How should catalog-only operations (type registration, view creation) be handled?
- Should we introduce a separate StorageEngine trait to formalize RuntimeContext's actual role?
- What's the migration path for external code depending on RuntimeContext methods?

**ASSUMPTIONS:**
- MVCC snapshot is required for ALL data access, even auto-commit operations
- Session layer is responsible for creating "default" snapshot for auto-commit
- RuntimeContext should be usable in non-transactional contexts (e.g., bulk loading tools)
- Transaction limitations (e.g., DROP INDEX) should remain enforced but with clearer messaging

## Risk Assessment

**HIGH RISK:**
- Phase 1 will break significant external code if RuntimeContext is used directly
- Large API surface area to change
- Many call sites to update

**MITIGATION:**
- Keep deprecated methods temporarily with clear warnings
- Update internal code first
- Comprehensive testing after Phase 1
- Consider feature flags for gradual rollout

## Success Criteria

**Phase 1 Complete When:**
- [ ] RuntimeContext has NO `has_active_transaction()` method
- [ ] ALL RuntimeContext data operations require explicit snapshot parameter
- [ ] RuntimeContext can be instantiated and used WITHOUT any transaction manager
- [ ] Full test suite passes with new API

**Final Success When:**
- [ ] Zero method signature duplication between RuntimeContext and RuntimeSession
- [ ] Clear layer separation: Session → Transaction → Storage
- [ ] Feature Placement Guide violations resolved
- [ ] All tests pass
- [ ] Documentation reflects actual architecture

## Next Steps (REVISED)

**IMMEDIATE (Start Here):**
1. Create detailed API change specification for Phase 1
2. Identify all RuntimeContext methods that need snapshot parameters
3. Create deprecation plan for old methods
4. Begin Phase 1 implementation with comprehensive tests

**DO NOT:**
- Try to "clean up" further without fixing Phase 1 first
- Add new features until architecture is corrected
- Continue modularization without addressing role separation

**Remember**: "If you catch yourself wiring up the same method signature twice... stop and reassess (the architecture is likely wrong)." - The architecture IS wrong. Fix it before proceeding.
