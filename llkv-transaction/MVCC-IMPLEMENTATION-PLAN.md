# MVCC Implementation Plan

## Current Status
✅ **Phase 1: Infrastructure** - COMPLETE
- Created `mvcc.rs` with `TxnIdManager`, `RowVersion`, and visibility rules
- Integrated `TxnId` and `TxnIdManager` into transaction system
- All 18 SLT tests passing

## Architecture Overview

### Current (Snapshot-at-BEGIN)
```
BEGIN → Copy entire tables to MemPager staging
  ↓
Read from staging (isolated snapshot)
  ↓
Write to staging (isolated changes)
  ↓
COMMIT → Copy staging changes back to base
```
**Problem**: Copies entire tables, doesn't scale

### Target (MVCC with Transaction IDs)
```
BEGIN → Allocate transaction ID (txn_id)
  ↓
Read from base, filter by visibility:
  - Show row if: created_by <= my_txn_id AND (deleted_by > my_txn_id OR deleted_by = MAX)
  ↓
Write to base with transaction IDs:
  - INSERT: Set created_by = my_txn_id, deleted_by = MAX
  - DELETE: Set deleted_by = my_txn_id (soft delete)
  - UPDATE: DELETE old version + INSERT new version
  ↓
COMMIT → No copying needed, just mark transaction as committed
```

## Implementation Phases

### Phase 2: Schema Extension (NEXT)
**Goal**: Add transaction ID columns to all tables

#### 2.1 Define System Columns
Location: `llkv-column-map/src/store/mod.rs` or new `mvcc_columns.rs`

```rust
// System column names (similar to ROW_ID_COLUMN_NAME)
pub const CREATED_BY_COLUMN_NAME: &str = "_created_by";
pub const DELETED_BY_COLUMN_NAME: &str = "_deleted_by";
```

#### 2.2 Update Schema Creation
Location: `llkv-runtime/src/lib.rs` - `create_table()` function

Add two new UInt64 columns to every table schema:
- `_created_by`: Transaction ID that created this row
- `_deleted_by`: Transaction ID that deleted this row (or TXN_ID_NONE/u64::MAX)

#### 2.3 Update RecordBatch Construction
Location: `llkv-runtime/src/lib.rs` - `insert_rows()` and `insert_batches()`

When building RecordBatches, include:
1. `row_id` (existing)
2. `_created_by` (NEW - set to current txn_id)
3. `_deleted_by` (NEW - set to TXN_ID_NONE)
4. User columns (existing)

### Phase 3: INSERT with Transaction IDs
**Goal**: Tag new rows with creating transaction ID

#### 3.1 Modify INSERT Logic
Location: `llkv-runtime/src/lib.rs` - `insert_rows()`

```rust
// Build arrays with transaction columns
let created_by_builder = UInt64Builder::with_capacity(row_count);
let deleted_by_builder = UInt64Builder::with_capacity(row_count);

for _ in 0..row_count {
    created_by_builder.append_value(current_txn_id); // From SessionTransaction
    deleted_by_builder.append_value(TXN_ID_NONE);
}

// Add to arrays vector
arrays.push(Arc::new(created_by_builder.finish()));
arrays.push(Arc::new(deleted_by_builder.finish()));

// Add to fields vector
fields.push(Field::new(CREATED_BY_COLUMN_NAME, DataType::UInt64, false));
fields.push(Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, false));
```

### Phase 4: SELECT with Visibility Filtering
**Goal**: Only show rows visible to current transaction

#### 4.1 Add Visibility Filter to Scans
Location: `llkv-executor/src/lib.rs` or `llkv-table/src/lib.rs`

Apply visibility rules during scan:
```rust
// For each row, check visibility
let visible = row_version.is_visible(snapshot_txn_id);
```

#### 4.2 Options:
**Option A**: Filter in scan iteration (most efficient)
- Modify `ScanBuilder` to filter by transaction visibility
- Requires plumbing txn_id through scan operations

**Option B**: Filter in executor layer
- Get all rows, filter in `SelectExecution`
- Simpler but less efficient

**Recommendation**: Start with Option B for correctness, optimize to Option A later

### Phase 5: DELETE with Soft Deletes
**Goal**: Mark rows as deleted instead of removing them

#### 5.1 Replace Physical Delete with Soft Delete
Location: `llkv-runtime/src/lib.rs` - `delete()` function

Instead of `table.delete_rows()`:
```rust
// Find matching rows
let row_ids = /* existing logic */;

// For each matching row, UPDATE deleted_by column
for row_id in row_ids {
    update_deleted_by_column(table, row_id, current_txn_id)?;
}
```

#### 5.2 Implement Soft Delete Operation
```rust
fn update_deleted_by_column(
    table: &Table,
    row_id: RowId,
    txn_id: TxnId
) -> Result<()> {
    // Read existing row
    // Create new version with deleted_by = txn_id
    // Write back
}
```

### Phase 6: UPDATE Implementation
**Goal**: Implement UPDATE as DELETE + INSERT

```rust
fn update_rows(
    &self,
    table: &ExecutorTable<P>,
    plan: UpdatePlan,
) -> Result<StatementResult<P>> {
    // 1. Find matching rows (existing logic)
    let matching_rows = /* ... */;
    
    // 2. Soft-delete old versions
    for row_id in &matching_rows {
        update_deleted_by_column(table, *row_id, current_txn_id)?;
    }
    
    // 3. Insert new versions with updated values
    insert_rows(table, new_rows_with_updates)?;
    
    Ok(StatementResult::Update { ... })
}
```

### Phase 7: Remove Table Copying
**Goal**: Eliminate inefficient snapshotting

#### 7.1 Remove Staging Logic
Location: `llkv-transaction/src/lib.rs`

Remove or simplify:
- `copy_tables_to_staging()`
- `snapshotted_tables` tracking
- Staging context entirely (or repurpose for write buffering)

#### 7.2 Direct Base Context Access
All operations go directly to base context, filtered by visibility rules

### Phase 8: Garbage Collection (Future)
**Goal**: Clean up old row versions

```rust
fn vacuum_table(
    table: &Table,
    oldest_active_txn: TxnId
) -> Result<()> {
    // Find rows where deleted_by < oldest_active_txn
    // Physically remove these rows
    // They're invisible to all active transactions
}
```

## Testing Strategy

1. **Unit Tests**: Test visibility rules in isolation (✅ DONE in mvcc.rs)
2. **Integration Tests**: Modify existing SLT tests to verify MVCC behavior
3. **Concurrent Tests**: Test multiple transactions reading/writing simultaneously
4. **Rollback Tests**: Ensure aborted transactions don't affect other transactions

## Migration Path

### For Existing Data
Tables created before MVCC need migration:
1. Add `_created_by` column (set all to 0 = "before MVCC")
2. Add `_deleted_by` column (set all to TXN_ID_NONE)
3. Mark table as "migrated to MVCC"

### Backward Compatibility
- Tables without MVCC columns: Treat as "before MVCC" (txn_id = 0)
- Gradually migrate tables on first write after upgrade

## Performance Considerations

### Pros
- No table copying overhead
- Concurrent readers don't block writers
- Snapshot isolation without locks

### Cons
- Extra storage for transaction columns (16 bytes per row)
- Soft deletes accumulate until VACUUM
- Visibility filtering overhead on reads

### Optimizations
- Batch visibility checks in SIMD
- Index on `deleted_by` for efficient vacuum
- Periodic auto-vacuum in background

## Next Steps

1. ✅ Create this implementation plan
2. → Implement Phase 2: Schema Extension
3. → Implement Phase 3: INSERT with Transaction IDs
4. → Run tests, verify correctness
5. → Continue with remaining phases

## Success Criteria

- ✅ All 18 SLT tests pass
- ✅ No table copying during transactions
- ✅ Proper isolation between concurrent transactions
- ✅ Correct rollback behavior
