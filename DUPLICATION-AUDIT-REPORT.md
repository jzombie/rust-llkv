# Duplication Audit Report
**Date:** October 22, 2025  
**Scope:** llkv-runtime, llkv-table, llkv-executor, llkv-transaction, llkv-plan

## Executive Summary

This audit identifies **CRITICAL ARCHITECTURAL DUPLICATIONS** that are causing the "clusterfuck mess" preventing project completion. The core problem: **RuntimeContext has become a god object** that owns responsibilities belonging to llkv-table and llkv-executor.

### Severity Breakdown
- 🔴 **CRITICAL (Must Fix)**: 5 major duplications
- 🟡 **MODERATE (Should Fix)**: 3 organizational issues  
- 🟢 **MINOR (Can Defer)**: 2 design improvements

---

## 🔴 CRITICAL DUPLICATIONS

### 1. Type Registry System (RuntimeContext vs. llkv-table)
**Location:** `llkv-runtime/src/runtime_context/mod.rs` lines 314-421  
**Problem:** RuntimeContext implements a complete custom type registry that should be in llkv-table's catalog system.

**Current Implementation:**
```rust
// llkv-runtime/src/runtime_context/mod.rs:314
pub fn register_type(&self, name: String, data_type: sqlparser::ast::DataType) {
    self.catalog_service.register_type(name, data_type);
}

pub fn drop_type(&self, name: &str) -> Result<()> {
    self.catalog_service.drop_type(name)?;
    Ok(())
}

pub fn resolve_type(&self, data_type: &sqlparser::ast::DataType) -> sqlparser::ast::DataType {
    // 50+ lines of complex type resolution logic
}
```

**Evidence of Duplication:**
- `CatalogManager<P>` in llkv-table already has type registry capabilities
- RuntimeContext just proxies to `catalog_service.register_type()` and `catalog_service.drop_type()`
- BUT `resolve_type()` has 50+ lines of custom logic that should be in CatalogManager

**Impact:** 
- Type management scattered across two crates
- Cannot use catalog features without runtime overhead
- Violates single responsibility principle

**Recommended Fix:**
1. Move `resolve_type()` implementation to `CatalogManager` in llkv-table
2. Keep only thin proxies in RuntimeContext if needed for convenience
3. Make CatalogManager's type registry the single source of truth

**Affected Files:**
- `llkv-runtime/src/runtime_context/mod.rs` (lines 314-421)
- `llkv-table/src/catalog/manager.rs` (needs new method)

---

### 2. View Management (RuntimeContext vs. llkv-table CatalogManager)
**Location:** `llkv-runtime/src/runtime_context/mod.rs` lines 333-368  
**Problem:** View creation and management duplicated between runtime and table layers.

**Current State:**
```rust
// RuntimeContext wraps CatalogManager but adds no value
pub fn create_view(&self, display_name: &str, view_definition: String) -> Result<TableId> {
    // Just calls catalog_service.create_view() with extra validation
}

pub fn is_view(&self, table_id: TableId) -> Result<bool> {
    // Just calls catalog_service.is_view()
}
```

**Evidence:**
- `CatalogManager` in llkv-table already has full view support
- RuntimeContext methods are pure pass-throughs with minimal added value
- Creates unnecessary dependency chain: caller → RuntimeContext → CatalogManager

**Impact:**
- Unnecessary indirection slows development
- Two APIs for the same functionality confuses users
- Makes testing harder (must mock entire runtime instead of just catalog)

**Recommended Fix:**
1. **Remove** view methods from RuntimeContext entirely
2. Expose CatalogManager directly: `pub fn catalog_manager(&self) -> &CatalogManager<P>`
3. Update all callers to use CatalogManager directly

**Affected Files:**
- `llkv-runtime/src/runtime_context/mod.rs` (lines 333-368) - DELETE
- All callsites in llkv-sql and tests - UPDATE to use catalog_manager()

---

### 3. SQL Type Parsing (RuntimeContext vs. llkv-executor)
**Location:** `llkv-runtime/src/lib.rs` line 518 - `sql_type_to_arrow()`  
**Problem:** SQL type to Arrow DataType conversion is in runtime, but executor needs it too.

**Current Code:**
```rust
// llkv-runtime/src/lib.rs:518
pub(crate) fn sql_type_to_arrow(type_str: &str) -> Result<DataType> {
    let normalized = type_str.trim().to_uppercase();
    let base_type = if let Some(paren_pos) = normalized.find('(') {
        &normalized[..paren_pos]
    } else {
        &normalized
    };

    match base_type {
        "TEXT" | "VARCHAR" | "CHAR" | "STRING" => Ok(DataType::Utf8),
        "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => Ok(DataType::Int64),
        // ... 10 more type mappings
    }
}
```

**Evidence of Duplication:**
- This is a pure translation function with NO runtime-specific logic
- Used during table creation, which happens in executor context
- Similar logic likely exists or will be needed in insert value coercion

**Impact:**
- Cannot parse SQL types without importing entire runtime
- Forces runtime dependency on llkv-executor when it should be opposite
- Type mapping is a translation concern, not a runtime concern

**Recommended Fix:**
1. Move `sql_type_to_arrow()` to `llkv-executor/src/translation/types.rs` (new file)
2. Re-export from llkv-executor root
3. Update runtime imports to use executor version

**Affected Files:**
- `llkv-runtime/src/lib.rs` (line 518) - MOVE to executor
- `llkv-executor/src/translation/types.rs` (NEW FILE)
- All callsites - UPDATE imports

---

### 4. Table Metadata Access Duplication (RuntimeContext proxy methods)
**Location:** `llkv-runtime/src/runtime_context/mod.rs` lines 581-677  
**Problem:** RuntimeContext has 6+ methods that just call CatalogManager/CatalogService.

**Duplicated Methods:**
```rust
pub fn table_view(&self, canonical_name: &str) -> Result<TableView> {
    self.catalog_service.table_view(canonical_name)  // PURE PASS-THROUGH
}

pub fn table_column_specs(&self, name: &str) -> Result<Vec<PlanColumnSpec>> {
    self.catalog_service.table_column_specs(&canonical_name)  // PURE PASS-THROUGH
}

pub fn rename_column(&self, ...) -> Result<()> {
    self.catalog_service.rename_column(...)  // PURE PASS-THROUGH
}

pub fn alter_column_type(&self, ...) -> Result<()> {
    self.catalog_service.alter_column_type(...)  // PURE PASS-THROUGH
}

pub fn drop_column(&self, ...) -> Result<()> {
    self.catalog_service.drop_column(...)  // PURE PASS-THROUGH
}

pub fn foreign_key_views(&self, name: &str) -> Result<Vec<ForeignKeyView>> {
    self.catalog_service.foreign_key_views(&canonical_name)  // PURE PASS-THROUGH
}
```

**Evidence:**
- ALL 6 methods are single-line pass-throughs
- Add ZERO validation, transformation, or business logic
- Force callers to depend on runtime when they only need catalog

**Impact:**
- Bloats RuntimeContext API surface (currently 3993 lines!)
- Creates false sense of "runtime owns metadata" when it doesn't
- Makes it impossible to use catalog independently

**Recommended Fix:**
1. **DELETE** all 6 pass-through methods from RuntimeContext
2. Expose CatalogManager: `pub fn catalog(&self) -> &CatalogManager<P>`
3. Update callers to: `context.catalog().table_view(name)` instead of `context.table_view(name)`

**Affected Files:**
- `llkv-runtime/src/runtime_context/mod.rs` (lines 581-677) - DELETE methods
- `llkv-runtime/src/runtime_session.rs` - UPDATE 10+ callsites
- `llkv-sql/src/sql_engine.rs` - UPDATE callsites
- Tests across multiple crates - UPDATE

---

### 5. Query Translation Logic in RuntimeContext
**Location:** `llkv-runtime/src/runtime_context/mod.rs` lines 1500-2500 (estimated)  
**Problem:** RuntimeContext has massive query translation/execution logic that belongs in llkv-executor.

**Evidence:**
```rust
// RuntimeContext implements entire SELECT execution pipeline
fn execute_select_plan_inner(...) -> Result<SelectExecution<P>> {
    // 200+ lines of:
    // - Predicate translation
    // - Projection building  
    // - Aggregate setup
    // - JOIN coordination
    // This is ALL executor responsibility!
}
```

**Current Structure:**
- RuntimeContext: ~3993 lines
- Includes query planning, translation, and execution
- QueryExecutor in llkv-executor should own this

**Impact:**
- RuntimeContext cannot be used without entire query engine
- Impossible to swap out query executor implementations
- Violates separation of concerns: runtime ≠ executor

**Recommended Fix:**
1. Extract all query translation logic to llkv-executor
2. RuntimeContext should call `QueryExecutor::execute()` not implement it
3. Move `execute_select`, `execute_insert`, etc. to QueryExecutor
4. RuntimeContext becomes thin coordination layer only

**Affected Files:**
- `llkv-runtime/src/runtime_context/mod.rs` (lines 1500-2500~) - EXTRACT to executor
- `llkv-executor/src/lib.rs` - ADD new query execution methods

---

## 🟡 MODERATE ISSUES

### 6. Range SELECT Parsing in Runtime
**Location:** `llkv-runtime/src/lib.rs` lines 179-466  
**Problem:** `extract_rows_from_range()` is SQL parsing logic in runtime layer.

**Current:**
```rust
pub fn extract_rows_from_range(select: &Select) -> Result<Option<RuntimeRangeSelectRows>>
```

**Issue:** This is query planning/parsing, not runtime execution. Should be in llkv-plan or llkv-sql.

**Fix:** Move to `llkv-plan/src/range.rs` as plan-time transformation.

---

### 7. Plan Value Conversion in Runtime
**Location:** `llkv-runtime/src/lib.rs` lines 112-165  
**Problem:** `plan_value_from_sql_expr()` and `plan_value_from_sql_value()` in wrong crate.

**Current:**
```rust
fn plan_value_from_sql_expr(expr: &SqlExpr) -> Result<PlanValue>
fn plan_value_from_sql_value(value: &ValueWithSpan) -> Result<PlanValue>
```

**Issue:** These are SQL → Plan conversions, not runtime operations. Belong in llkv-plan or llkv-sql.

**Fix:** Move to `llkv-plan/src/conversion.rs`.

---

### 8. Constraint Service in RuntimeContext
**Location:** `llkv-runtime/src/runtime_context/mod.rs` - `constraint_service` field  
**Problem:** RuntimeContext holds ConstraintService but it's a pure llkv-table concern.

**Current:**
```rust
pub struct RuntimeContext<P> {
    constraint_service: ConstraintService<P>,  // Why is this here?
}
```

**Issue:** Constraint validation is table-level logic, not runtime orchestration.

**Fix:** Move ConstraintService to CatalogManager in llkv-table. Runtime should call `catalog.validate_constraints()`.

---

## 🟢 MINOR IMPROVEMENTS

### 9. Transaction Manager Placement
**Location:** `llkv-runtime/src/runtime_context/mod.rs` - `transaction_manager` field  
**Observation:** RuntimeContext owns TransactionManager, but llkv-transaction is already separate crate.

**Current State:** Acceptable - RuntimeContext coordinates transactions, this is reasonable.

**Optional Improvement:** Consider if RuntimeSession should own TransactionManager instead.

---

### 10. MVCC Helpers Duplication
**Location:** `llkv-runtime/src/runtime_context/mvcc_helpers.rs`  
**Status:** ✅ RECENTLY FIXED - moved to llkv-transaction

**Evidence:** 
- `filter_row_ids_for_snapshot()` moved to llkv-transaction/helpers
- Now properly re-exported from transaction crate
- No longer duplicated

---

## PROPOSED ARCHITECTURE

### Current (Broken) Flow:
```
SQL Frontend
    ↓
RuntimeSession
    ↓
RuntimeContext (GOD OBJECT)
    ├─ Owns: CatalogManager (should be independent)
    ├─ Owns: ConstraintService (should be in catalog)
    ├─ Implements: Query execution (should be in executor)
    ├─ Implements: Type registry (should be in catalog)
    ├─ Implements: View management (should be in catalog)
    └─ Implements: SQL parsing (should be in plan/sql)
```

### Proposed (Clean) Architecture:
```
SQL Frontend
    ↓
RuntimeSession (transaction lifecycle, namespace routing)
    ↓
RuntimeContext (THIN: MVCC coordination, storage access only)
    ├─ Uses: QueryExecutor<P> (query execution)
    ├─ Uses: CatalogManager (table/field/type/view management)
    │   └─ Contains: ConstraintService (constraint validation)
    └─ Uses: TransactionManager (transaction coordination)

Horizontal Dependencies:
- QueryExecutor uses CatalogManager for schema lookups
- CatalogManager uses storage pager directly
- RuntimeContext coordinates but doesn't own everything
```

---

## CONSOLIDATION PLAN

### Phase 1: Extract Type & View Management (High Priority)
**Goal:** Remove type/view methods from RuntimeContext

**Steps:**
1. Move `resolve_type()` implementation to `CatalogManager::resolve_type()`
2. Delete `register_type()`, `drop_type()`, `create_view()`, `is_view()` from RuntimeContext
3. Add `pub fn catalog(&self) -> &CatalogManager<P>` to RuntimeContext
4. Update all callers to use `context.catalog().create_view()` pattern

**Impact:** Reduces RuntimeContext by ~150 lines, establishes catalog as authoritative

---

### Phase 2: Remove Metadata Pass-Throughs (High Priority)
**Goal:** Stop RuntimeContext from wrapping CatalogManager methods

**Steps:**
1. Delete 6 pass-through methods: `table_view()`, `table_column_specs()`, `rename_column()`, `alter_column_type()`, `drop_column()`, `foreign_key_views()`
2. Update RuntimeSession to access `context.catalog()` directly
3. Update llkv-sql to use catalog directly

**Impact:** Reduces RuntimeContext by ~100 lines, clarifies responsibility

---

### Phase 3: Extract Query Execution Logic (Critical)
**Goal:** Move query planning/translation out of RuntimeContext

**Steps:**
1. Create `llkv-executor/src/query_planner.rs`
2. Move SELECT/INSERT/UPDATE/DELETE execution logic from RuntimeContext to QueryExecutor
3. RuntimeContext becomes caller of QueryExecutor, not implementer
4. Move SQL type parsing to llkv-executor/translation

**Impact:** Reduces RuntimeContext by ~1000+ lines, enables executor swapping

---

### Phase 4: Move SQL Parsing Helpers (Medium Priority)
**Goal:** Relocate SQL-to-Plan conversions

**Steps:**
1. Move `extract_rows_from_range()` to llkv-plan
2. Move `plan_value_from_sql_*()` to llkv-plan  
3. Move `sql_type_to_arrow()` to llkv-executor

**Impact:** Cleans up llkv-runtime/lib.rs, proper concern separation

---

### Phase 5: Reorganize Constraint Validation (Medium Priority)
**Goal:** Move ConstraintService into CatalogManager

**Steps:**
1. Make ConstraintService a field of CatalogManager (not RuntimeContext)
2. Expose via `CatalogManager::validate_constraints()`
3. Update RuntimeContext to call through catalog

**Impact:** Constraint validation becomes table concern, not runtime concern

---

## VALIDATION STRATEGY

### Pre-Flight Checks:
- [ ] Run full test suite baseline: `cargo test --all`
- [ ] Document all current RuntimeContext public methods
- [ ] Identify all external callers of RuntimeContext methods

### Per-Phase Validation:
- [ ] `cargo check --all` after each file move
- [ ] Run affected crate tests after method deletions
- [ ] Run full sqllogictest suite after Phase 3
- [ ] Verify no new `pub(crate)` leaks across crate boundaries

### Acceptance Criteria:
- All 68 sqllogictests passing
- RuntimeContext < 2000 lines (currently 3993)
- Zero pass-through methods in RuntimeContext
- CatalogManager usable without RuntimeContext
- QueryExecutor usable without RuntimeContext

---

## RISK ASSESSMENT

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking public API | HIGH | HIGH | Use `#[deprecated]` shims during migration |
| Test failures | MEDIUM | MEDIUM | Phase work, validate per phase |
| Performance regression | LOW | LOW | No algorithmic changes, just reorganization |
| Merge conflicts | HIGH | HIGH | Do work in dedicated branch, frequent rebases |

---

## IMMEDIATE NEXT STEPS

1. **STOP adding features** until duplication is fixed
2. **Create migration branch:** `git checkout -b refactor/consolidate-duplication`
3. **Start with Phase 1** (Type & View Management) - smallest, highest value
4. **Validate Phase 1** completely before starting Phase 2
5. **Document every breaking change** in migration guide

---

## CONCLUSION

The "clusterfuck" stems from **RuntimeContext being a 3993-line god object** that owns responsibilities of 3 other crates:

1. **llkv-table** (catalog, types, views, constraints)
2. **llkv-executor** (query execution, SQL parsing, type translation)  
3. **llkv-transaction** (MVCC helpers - recently fixed ✅)

**Critical Path:** Phases 1-3 must be completed before any new features. This will reduce RuntimeContext to its true purpose: **thin coordination layer** between transaction isolation, storage access, and query execution.

**Estimated Effort:**
- Phase 1: 4-6 hours
- Phase 2: 4-6 hours  
- Phase 3: 16-24 hours (most complex)
- Phase 4: 6-8 hours
- Phase 5: 6-8 hours

**Total:** ~40-50 hours of focused refactoring to establish clean architecture.

**Success Metric:** RuntimeContext < 2000 lines, zero pass-throughs, clear separation of concerns.
