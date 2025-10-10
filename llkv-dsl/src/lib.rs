#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::mem;
use std::ops::Bound;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{
    Array, ArrayRef, Date32Array, Date32Builder, Float64Array, Float64Builder, Int64Array,
    Int64Builder, StringArray, StringBuilder, UInt64Array, UInt64Builder, new_null_array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ColumnStore;
use llkv_column_map::store::{Projection as StoreProjection, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::expr::{Expr as LlkvExpr, Filter, Operator, ScalarExpr};
use llkv_expr::literal::Literal;
use llkv_result::Error;
use llkv_storage::pager::{MemPager, Pager};
use llkv_table::table::{
    ScanOrderDirection, ScanOrderSpec, ScanOrderTransform, ScanProjection, ScanStreamOptions, Table,
};
use llkv_table::types::{FieldId, TableId};
use llkv_table::{CATALOG_TABLE_ID, ColMeta, SysCatalog, TableMeta};
use simd_r_drive_entry_handle::EntryHandle;
use sqlparser::ast::{
    Expr as SqlExpr, FunctionArg, FunctionArgExpr, GroupByExpr, ObjectName, ObjectNamePart, Select,
    SelectItem, SelectItemQualifiedWildcardKind, TableAlias, TableFactor, UnaryOperator, Value,
    ValueWithSpan,
};
use time::{Date, Month};

pub type DslResult<T> = llkv_result::Result<T>;

// Re-export plan structures from llkv-plan

pub use llkv_plan::{
    AggregateExpr, AggregateFunction, AssignmentValue, ColumnAssignment, ColumnNullability,
    ColumnSpec, CreateTablePlan, CreateTableSource, DeletePlan, DslOperation, DslValue, InsertPlan,
    InsertSource, IntoColumnSpec, NotNull, Nullable, OrderByPlan, OrderSortType, OrderTarget,
    SelectPlan, SelectProjection, UpdatePlan,
};

/// Result of running a DSL statement.
#[derive(Clone)]
pub enum StatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    CreateTable {
        table_name: String,
    },
    NoOp,
    Insert {
        table_name: String,
        rows_inserted: usize,
    },
    Update {
        table_name: String,
        rows_updated: usize,
    },
    Delete {
        table_name: String,
        rows_deleted: usize,
    },
    Select {
        table_name: String,
        schema: Arc<Schema>,
        execution: SelectExecution<P>,
    },
    Transaction {
        kind: TransactionKind,
    },
}

impl<P> fmt::Debug for StatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatementResult::CreateTable { table_name } => f
                .debug_struct("CreateTable")
                .field("table_name", table_name)
                .finish(),
            StatementResult::NoOp => f.debug_struct("NoOp").finish(),
            StatementResult::Insert {
                table_name,
                rows_inserted,
            } => f
                .debug_struct("Insert")
                .field("table_name", table_name)
                .field("rows_inserted", rows_inserted)
                .finish(),
            StatementResult::Update {
                table_name,
                rows_updated,
            } => f
                .debug_struct("Update")
                .field("table_name", table_name)
                .field("rows_updated", rows_updated)
                .finish(),
            StatementResult::Delete {
                table_name,
                rows_deleted,
            } => f
                .debug_struct("Delete")
                .field("table_name", table_name)
                .field("rows_deleted", rows_deleted)
                .finish(),
            StatementResult::Select {
                table_name, schema, ..
            } => f
                .debug_struct("Select")
                .field("table_name", table_name)
                .field("schema", schema)
                .finish(),
            StatementResult::Transaction { kind } => {
                f.debug_struct("Transaction").field("kind", kind).finish()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionKind {
    Begin,
    Commit,
    Rollback,
}

impl<P> StatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Convert a StatementResult from one pager type to another.
    /// Only works for non-SELECT results (CreateTable, Insert, Update, Delete, NoOp, Transaction).
    pub(crate) fn convert_pager_type<Q>(self) -> DslResult<StatementResult<Q>>
    where
        Q: Pager<Blob = EntryHandle> + Send + Sync,
    {
        match self {
            StatementResult::CreateTable { table_name } => {
                Ok(StatementResult::CreateTable { table_name })
            }
            StatementResult::NoOp => Ok(StatementResult::NoOp),
            StatementResult::Insert {
                table_name,
                rows_inserted,
            } => Ok(StatementResult::Insert {
                table_name,
                rows_inserted,
            }),
            StatementResult::Update {
                table_name,
                rows_updated,
            } => Ok(StatementResult::Update {
                table_name,
                rows_updated,
            }),
            StatementResult::Delete {
                table_name,
                rows_deleted,
            } => Ok(StatementResult::Delete {
                table_name,
                rows_deleted,
            }),
            StatementResult::Transaction { kind } => Ok(StatementResult::Transaction { kind }),
            StatementResult::Select { .. } => Err(Error::Internal(
                "Cannot convert SELECT result between pager types in transaction".into(),
            )),
        }
    }
}

// ============================================================================
// Plan Structures (now in llkv-plan and re-exported above)
// ============================================================================
//
// The following types are defined in llkv-plan and re-exported:
// - DslValue, CreateTablePlan, ColumnSpec, IntoColumnSpec
// - InsertPlan, InsertSource, UpdatePlan, DeletePlan
// - SelectPlan, SelectProjection, AggregateExpr, AggregateFunction
// - OrderByPlan, OrderSortType, OrderTarget
// - DslOperation
//
// This separation allows plans to be used independently of execution logic.
// ============================================================================

/// Delta state for tracking table modifications within a transaction.
#[derive(Clone, Debug)]
pub struct TableDeltaState {
    /// Filters that have been seeded (snapshot copied) into staging.
    pub seeded_filters: HashSet<String>,
    /// Exclusion predicates from UPDATE/DELETE operations.
    pub exclusion_predicates: Vec<LlkvExpr<'static, String>>,
    /// If true, all rows from base table should be excluded.
    pub exclude_all_rows: bool,
}

impl TableDeltaState {
    fn new() -> Self {
        Self {
            seeded_filters: HashSet::new(),
            exclusion_predicates: Vec::new(),
            exclude_all_rows: false,
        }
    }
}

/// Transaction state for the DSL context.
pub struct DslTransaction<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Staging context with MemPager for isolation.
    staging: Arc<DslContext<MemPager>>,
    /// Operations to replay on commit.
    operations: Vec<DslOperation>,
    /// Tables that have been staged (created/snapshotted in staging).
    staged_tables: HashSet<String>,
    /// Per-table delta tracking for merging reads.
    table_deltas: HashMap<String, TableDeltaState>,
    /// Tables created within this transaction.
    new_tables: HashSet<String>,
    /// Tables that have been snapshotted into staging.
    snapshotted_tables: HashSet<String>,
    /// Tables known to be missing.
    missing_tables: HashSet<String>,
    /// Snapshot of catalog at transaction start.
    catalog_snapshot: HashSet<String>,
    /// Base context for table snapshotting.
    base_context: Arc<DslContext<P>>,
}

impl<P> DslTransaction<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn new(base_context: Arc<DslContext<P>>) -> Self {
        let staging = Arc::new(DslContext::new(Arc::new(MemPager::default())));
        let catalog_snapshot = base_context.table_names().into_iter().collect();

        Self {
            staging,
            operations: Vec::new(),
            staged_tables: HashSet::new(),
            table_deltas: HashMap::new(),
            new_tables: HashSet::new(),
            snapshotted_tables: HashSet::new(),
            missing_tables: HashSet::new(),
            catalog_snapshot,
            base_context,
        }
    }

    /// Ensure a table exists in the staging context, snapshotting from base if needed.
    fn ensure_table_in_delta(&mut self, table_name: &str) -> DslResult<()> {
        if self.staged_tables.contains(table_name) {
            return Ok(());
        }

        let canonical_name = table_name.to_ascii_lowercase();

        if !self.catalog_snapshot.contains(&canonical_name) && !self.new_tables.contains(table_name)
        {
            self.missing_tables.insert(table_name.to_string());
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{table_name}' does not exist"
            )));
        }

        if self.missing_tables.contains(table_name) {
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{table_name}' does not exist"
            )));
        }

        // Try to get specs from base context
        match self.base_context.table_column_specs(table_name) {
            Ok(specs) => {
                self.missing_tables.remove(table_name);
                let mut plan = CreateTablePlan::new(table_name.to_string());
                plan.if_not_exists = true;
                plan.columns = specs;
                self.staging.create_table_plan(plan)?;

                // Snapshot rows from base
                match self.base_context.export_table_rows(table_name) {
                    Ok(snapshot) => {
                        if !snapshot.rows.is_empty() {
                            let insert_plan = InsertPlan {
                                table: table_name.to_string(),
                                columns: snapshot.columns.clone(),
                                source: InsertSource::Rows(snapshot.rows),
                            };
                            self.staging.insert(insert_plan)?;
                        }
                        self.snapshotted_tables.insert(table_name.to_string());
                    }
                    Err(Error::NotFound) => {
                        self.snapshotted_tables.insert(table_name.to_string());
                    }
                    Err(other) => return Err(other),
                }
            }
            Err(Error::NotFound) => {
                // Check if it exists in staging only
                match self.staging.table_column_specs(table_name) {
                    Ok(_) => {
                        self.staged_tables.insert(table_name.to_string());
                        return Ok(());
                    }
                    Err(_) => {
                        self.missing_tables.insert(table_name.to_string());
                        return Err(Error::CatalogError(format!(
                            "Catalog Error: Table '{table_name}' does not exist"
                        )));
                    }
                }
            }
            Err(other) => return Err(other),
        }

        self.staged_tables.insert(table_name.to_string());
        Ok(())
    }

    /// Seed rows from base table for UPDATE/DELETE operations.
    fn seed_rows_for_update(
        &mut self,
        table_name: &str,
        filter: Option<&LlkvExpr<'static, String>>,
    ) -> DslResult<()> {
        let filter_key = filter
            .map(|expr| format!("{:?}", expr))
            .unwrap_or_else(|| "__all__".to_string());

        if self
            .table_deltas
            .get(table_name)
            .map(|delta| delta.seeded_filters.contains(&filter_key))
            .unwrap_or(false)
        {
            return Ok(());
        }

        self.ensure_table_in_delta(table_name)?;

        let batches = match self
            .base_context
            .get_batches_with_row_ids(table_name, filter.cloned())
        {
            Ok(batches) => batches,
            Err(Error::NotFound) => {
                return Ok(());
            }
            Err(other) => return Err(other),
        };

        {
            let delta = self
                .table_deltas
                .entry(table_name.to_string())
                .or_insert_with(TableDeltaState::new);
            delta.seeded_filters.insert(filter_key.clone());
        }

        if batches.is_empty() || batches.iter().all(|b| b.num_rows() == 0) {
            return Ok(());
        }

        self.staging
            .append_batches_with_row_ids(table_name, batches)
            .map(|_| ())
    }

    /// Record an exclusion predicate for UPDATE/DELETE.
    fn record_update_exclusion(
        &mut self,
        table_name: &str,
        filter: Option<&LlkvExpr<'static, String>>,
    ) {
        let delta = self
            .table_deltas
            .entry(table_name.to_string())
            .or_insert_with(TableDeltaState::new);
        match filter {
            Some(expr) => {
                if !delta.exclude_all_rows {
                    delta.exclusion_predicates.push(expr.clone());
                }
            }
            None => {
                delta.exclude_all_rows = true;
                delta.exclusion_predicates.clear();
            }
        }
    }

    /// Execute a SELECT query with transaction merge logic.
    fn execute_select(&mut self, plan: SelectPlan) -> DslResult<SelectExecution<P>> {
        // Ensure table exists in staging
        self.ensure_table_in_delta(&plan.table)?;

        // Check if all data is in staging (new table or fully snapshotted)
        let all_local =
            self.new_tables.contains(&plan.table) || self.snapshotted_tables.contains(&plan.table);

        // Get delta from staging context
        let delta_result = self.staging.execute_select(plan.clone())?;

        // If table has exclusion predicates, we need to apply filters to base query
        let skip_base = if let Some(delta) = self.table_deltas.get(&plan.table) {
            delta.exclude_all_rows
        } else {
            false
        };

        // Execute on base if needed
        let base_result = if skip_base || all_local {
            None
        } else {
            // Apply exclusion filters if needed
            let mut modified_plan = plan.clone();
            if let Some(delta) = self.table_deltas.get(&plan.table) {
                if !delta.exclusion_predicates.is_empty() {
                    // Combine existing filter with NOT (exclusion predicates)
                    let mut filters = Vec::new();

                    // Add existing filter if present
                    if let Some(existing) = modified_plan.filter.clone() {
                        filters.push(existing);
                    }

                    // Add NOT (exclusion) for each exclusion predicate
                    for exclusion in &delta.exclusion_predicates {
                        filters.push(LlkvExpr::Not(Box::new(exclusion.clone())));
                    }

                    // Combine all filters with AND
                    if filters.len() == 1 {
                        modified_plan.filter = Some(filters.into_iter().next().unwrap());
                    } else if filters.len() > 1 {
                        modified_plan.filter = Some(LlkvExpr::And(filters));
                    }
                }
            }

            match self.base_context.execute_select(modified_plan) {
                Ok(result) => Some(result),
                Err(Error::NotFound) => None,
                Err(other) => return Err(other),
            }
        };

        // Merge results
        Self::merge_select_results(base_result, delta_result, &plan)
    }

    /// Merge SELECT results from base and delta contexts.
    fn merge_select_results(
        base: Option<SelectExecution<P>>,
        delta: SelectExecution<MemPager>,
        plan: &SelectPlan,
    ) -> DslResult<SelectExecution<P>> {
        let delta_batches = match delta.collect() {
            Ok(batches) => batches,
            Err(Error::NotFound) => Vec::new(),
            Err(other) => return Err(other),
        };

        let (base_batches, schema) = match base {
            Some(execution) => {
                let schema = execution.schema();
                let batches = match execution.collect() {
                    Ok(batches) => batches,
                    Err(Error::NotFound) => Vec::new(),
                    Err(other) => return Err(other),
                };
                (batches, schema)
            }
            None => {
                // No base data, use delta schema
                if delta_batches.is_empty() {
                    // Create empty result from plan
                    let table_name = plan.table.clone();
                    let schema = Arc::new(Schema::new(Vec::<Field>::new()));
                    return Ok(SelectExecution::from_batch(
                        table_name,
                        schema.clone(),
                        RecordBatch::new_empty(schema),
                    ));
                }
                (Vec::new(), delta_batches[0].schema())
            }
        };

        // Handle aggregates specially
        if !plan.aggregates.is_empty() {
            return Self::merge_aggregate_results(base_batches, delta_batches, schema, plan);
        }

        // Concatenate batches
        let mut all_batches: Vec<RecordBatch> = Vec::new();
        all_batches.extend(base_batches);
        all_batches.extend(delta_batches);

        let combined = if all_batches.is_empty() {
            RecordBatch::new_empty(Arc::clone(&schema))
        } else if all_batches.len() == 1 {
            all_batches.remove(0)
        } else {
            let refs: Vec<&RecordBatch> = all_batches.iter().collect();
            arrow::compute::concat_batches(&schema, refs).map_err(|err| {
                Error::Internal(format!("failed to merge transaction select batches: {err}"))
            })?
        };

        // Apply ORDER BY if needed
        let final_batch = if let Some(order_by) = &plan.order_by {
            if combined.num_rows() > 1 {
                Self::resort_record_batch(combined, order_by, &schema)?
            } else {
                combined
            }
        } else {
            combined
        };

        Ok(SelectExecution::from_batch(
            plan.table.clone(),
            Arc::clone(&schema),
            final_batch,
        ))
    }

    /// Merge aggregate results from base and delta.
    fn merge_aggregate_results(
        base_batches: Vec<RecordBatch>,
        delta_batches: Vec<RecordBatch>,
        schema: Arc<Schema>,
        plan: &SelectPlan,
    ) -> DslResult<SelectExecution<P>> {
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(plan.aggregates.len());

        for (idx, spec) in plan.aggregates.iter().enumerate() {
            let base_value = Self::extract_aggregate_value(&base_batches, idx)?;
            let delta_value = Self::extract_aggregate_value(&delta_batches, idx)?;

            let combined = match spec {
                AggregateExpr::CountStar { .. } => {
                    Some(base_value.unwrap_or(0) + delta_value.unwrap_or(0))
                }
                AggregateExpr::Column { function, .. } => match function {
                    AggregateFunction::Count | AggregateFunction::CountNulls => {
                        Some(base_value.unwrap_or(0) + delta_value.unwrap_or(0))
                    }
                    AggregateFunction::SumInt64 => {
                        if base_value.is_none() && delta_value.is_none() {
                            None
                        } else {
                            Some(base_value.unwrap_or(0) + delta_value.unwrap_or(0))
                        }
                    }
                    AggregateFunction::MinInt64 => match (base_value, delta_value) {
                        (Some(b), Some(d)) => Some(std::cmp::min(b, d)),
                        (Some(b), None) => Some(b),
                        (None, Some(d)) => Some(d),
                        (None, None) => None,
                    },
                    AggregateFunction::MaxInt64 => match (base_value, delta_value) {
                        (Some(b), Some(d)) => Some(std::cmp::max(b, d)),
                        (Some(b), None) => Some(b),
                        (None, Some(d)) => Some(d),
                        (None, None) => None,
                    },
                },
            };

            let mut builder = Int64Builder::with_capacity(1);
            if let Some(value) = combined {
                builder.append_value(value);
            } else {
                builder.append_null();
            }
            columns.push(Arc::new(builder.finish()) as ArrayRef);
        }

        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).map_err(|err| {
            Error::Internal(format!("failed to build merged aggregate batch: {err}"))
        })?;

        Ok(SelectExecution::from_batch(
            plan.table.clone(),
            Arc::clone(&schema),
            batch,
        ))
    }

    /// Extract aggregate value from batches at given column index.
    fn extract_aggregate_value(
        batches: &[RecordBatch],
        column_idx: usize,
    ) -> DslResult<Option<i64>> {
        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }
            let array = batch
                .column(column_idx)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    Error::Internal("aggregate output column is not INT64 as expected".into())
                })?;
            if array.is_empty() {
                continue;
            }
            if array.is_null(0) {
                return Ok(None);
            }
            return Ok(Some(array.value(0)));
        }
        Ok(None)
    }

    /// Resort a record batch according to ORDER BY specification.
    fn resort_record_batch(
        batch: RecordBatch,
        order_by: &OrderByPlan,
        schema: &Arc<Schema>,
    ) -> DslResult<RecordBatch> {
        use arrow::compute::{SortColumn, SortOptions, TakeOptions, lexsort_to_indices, take};

        let column_index = match &order_by.target {
            OrderTarget::Column(name) => {
                schema
                    .column_with_name(name)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "ORDER BY column '{}' not found in projection",
                            name
                        ))
                    })?
                    .0
            }
            OrderTarget::Index(idx) => *idx,
        };

        let sort_column = SortColumn {
            values: batch.column(column_index).clone(),
            options: Some(SortOptions {
                descending: !order_by.ascending,
                nulls_first: order_by.nulls_first,
            }),
        };

        let indices: arrow::array::UInt32Array = lexsort_to_indices(&[sort_column], None)
            .map_err(|err| Error::Internal(format!("failed to sort transaction result: {err}")))?;

        let mut sorted_columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
        for column_index in 0..schema.fields().len() {
            let column = batch.column(column_index);
            let taken = take(
                column.as_ref(),
                &indices,
                Some(TakeOptions { check_bounds: true }),
            )
            .map_err(|err| Error::Internal(format!("failed to reorder rows: {err}")))?;
            sorted_columns.push(taken);
        }

        RecordBatch::try_new(Arc::clone(schema), sorted_columns).map_err(|err| {
            Error::Internal(format!("failed to build sorted transaction batch: {err}"))
        })
    }
}

/// In-memory execution context shared by DSL queries.
pub struct DslContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pager: Arc<P>,
    tables: RwLock<HashMap<String, Arc<DslTable<P>>>>,
    // Session-based transactions: each session ID maps to its own transaction state
    transactions: Mutex<HashMap<u64, DslTransaction<P>>>,
    next_session_id: AtomicU64,
}

/// A session handle for transaction management.
/// When dropped, automatically rolls back any active transaction.
pub struct DslSession<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<DslContext<P>>,
    session_id: u64,
}

impl<P> DslSession<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Get the session ID.
    pub fn session_id(&self) -> u64 {
        self.session_id
    }

    /// Get the underlying context (for advanced use).
    pub fn context(&self) -> &Arc<DslContext<P>> {
        &self.context
    }

    /// Check if this session has an active transaction.
    pub fn has_active_transaction(&self) -> bool {
        self.context
            .transactions
            .lock()
            .expect("transactions lock poisoned")
            .contains_key(&self.session_id)
    }

    /// Begin a transaction in this session.
    pub fn begin_transaction(&self) -> DslResult<StatementResult<P>> {
        let mut guard = self
            .context
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        if guard.contains_key(&self.session_id) {
            return Err(Error::InvalidArgumentError(
                "a transaction is already in progress in this session".into(),
            ));
        }
        guard.insert(
            self.session_id,
            DslTransaction::new(Arc::clone(&self.context)),
        );
        Ok(StatementResult::Transaction {
            kind: TransactionKind::Begin,
        })
    }

    /// Commit the transaction in this session.
    pub fn commit_transaction(&self) -> DslResult<StatementResult<P>> {
        let mut guard = self
            .context
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        let tx = guard.remove(&self.session_id).ok_or_else(|| {
            Error::InvalidArgumentError(
                "no transaction is currently in progress in this session".into(),
            )
        })?;
        drop(guard);

        // Replay all operations on the base context
        for operation in tx.operations {
            match operation {
                DslOperation::CreateTable(plan) => {
                    self.context.create_table_plan(plan)?;
                }
                DslOperation::Insert(plan) => {
                    self.context.insert(plan)?;
                }
                DslOperation::Update(plan) => {
                    self.context.update(plan)?;
                }
                DslOperation::Delete(plan) => {
                    self.context.delete(plan)?;
                }
                DslOperation::Select(_) => {
                    // SELECT is read-only, no replay needed
                }
            }
        }

        Ok(StatementResult::Transaction {
            kind: TransactionKind::Commit,
        })
    }

    /// Rollback the transaction in this session.
    pub fn rollback_transaction(&self) -> DslResult<StatementResult<P>> {
        let mut guard = self
            .context
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        if guard.remove(&self.session_id).is_none() {
            return Err(Error::InvalidArgumentError(
                "no transaction is currently in progress in this session".into(),
            ));
        }
        Ok(StatementResult::Transaction {
            kind: TransactionKind::Rollback,
        })
    }

    /// Execute an operation in this session's transaction, or directly if no transaction is active.
    pub fn execute_operation(&self, operation: DslOperation) -> DslResult<StatementResult<P>> {
        if !self.has_active_transaction() {
            // No transaction - execute directly on base context
            return match operation {
                DslOperation::CreateTable(plan) => self.context.create_table_plan(plan),
                DslOperation::Insert(plan) => self.context.insert(plan),
                DslOperation::Update(plan) => self.context.update(plan),
                DslOperation::Delete(plan) => self.context.delete(plan),
                DslOperation::Select(plan) => {
                    let table_name = plan.table.clone();
                    let execution = self.context.execute_select(plan)?;
                    let schema = execution.schema();
                    Ok(StatementResult::Select {
                        table_name,
                        schema,
                        execution,
                    })
                }
            };
        }

        // In transaction - add to transaction and execute on staging context
        let mut guard = self
            .context
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        let tx = guard
            .get_mut(&self.session_id)
            .ok_or_else(|| Error::Internal("transaction disappeared during execution".into()))?;

        // Execute the operation and update transaction state
        let result = match operation {
            DslOperation::CreateTable(ref plan) => {
                let result = tx.staging.create_table_plan(plan.clone())?;
                // Track new table so it's visible to subsequent operations in this transaction
                tx.new_tables.insert(plan.name.clone());
                tx.missing_tables.remove(&plan.name);
                tx.staged_tables.insert(plan.name.clone());
                // Track for commit replay
                tx.operations.push(DslOperation::CreateTable(plan.clone()));
                result.convert_pager_type()?
            }
            DslOperation::Insert(ref plan) => {
                tx.ensure_table_in_delta(&plan.table)?;
                let result = tx.staging.insert(plan.clone())?;
                // Track for commit replay
                tx.operations.push(DslOperation::Insert(plan.clone()));
                result.convert_pager_type()?
            }
            DslOperation::Update(ref plan) => {
                tx.ensure_table_in_delta(&plan.table)?;
                let result = tx.staging.update(plan.clone())?;
                // Track for commit replay
                tx.operations.push(DslOperation::Update(plan.clone()));
                result.convert_pager_type()?
            }
            DslOperation::Delete(ref plan) => {
                tx.ensure_table_in_delta(&plan.table)?;
                let result = tx.staging.delete(plan.clone())?;
                // Track for commit replay
                tx.operations.push(DslOperation::Delete(plan.clone()));
                result.convert_pager_type()?
            }
            DslOperation::Select(ref plan) => {
                // SELECT is read-only, not tracked for replay
                let table_name = plan.table.clone();
                let execution = tx.execute_select(plan.clone())?;
                let schema = execution.schema();
                StatementResult::Select {
                    table_name,
                    schema,
                    execution,
                }
            }
        };

        Ok(result)
    }

    /// Pass-through methods to context for convenience
    pub fn create_table_plan(&self, plan: CreateTablePlan) -> DslResult<StatementResult<P>> {
        self.execute_operation(DslOperation::CreateTable(plan))
    }

    pub fn insert(&self, plan: InsertPlan) -> DslResult<StatementResult<P>> {
        self.execute_operation(DslOperation::Insert(plan))
    }

    pub fn update(&self, plan: UpdatePlan) -> DslResult<StatementResult<P>> {
        self.execute_operation(DslOperation::Update(plan))
    }

    pub fn delete(&self, plan: DeletePlan) -> DslResult<StatementResult<P>> {
        self.execute_operation(DslOperation::Delete(plan))
    }

    pub fn select(&self, plan: SelectPlan) -> DslResult<StatementResult<P>> {
        self.execute_operation(DslOperation::Select(plan))
    }

    /// Get table names from the context.
    pub fn table_names(&self) -> Vec<String> {
        self.context.table_names()
    }

    /// Get table column specs from the context.
    pub fn table_column_specs(&self, table_name: &str) -> DslResult<Vec<ColumnSpec>> {
        self.context.table_column_specs(table_name)
    }
}

impl<P> Drop for DslSession<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn drop(&mut self) {
        // Auto-rollback on drop if transaction is active
        let mut guard = self
            .context
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        if guard.remove(&self.session_id).is_some() {
            eprintln!("Warning: DslSession dropped with active transaction - auto-rolling back");
        }
    }
}

impl<P> DslContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(pager: Arc<P>) -> Self {
        Self {
            pager,
            tables: RwLock::new(HashMap::new()),
            transactions: Mutex::new(HashMap::new()),
            next_session_id: AtomicU64::new(1),
        }
    }

    /// Create a new session for transaction management.
    /// Each session can have its own independent transaction.
    pub fn create_session(self: &Arc<Self>) -> DslSession<P> {
        let session_id = self.next_session_id.fetch_add(1, Ordering::SeqCst);
        DslSession {
            context: Arc::clone(self),
            session_id,
        }
    }

    /// Check if there's an active transaction (legacy - checks if ANY session has a transaction).
    #[deprecated(note = "Use session-based transactions instead")]
    pub fn has_active_transaction(&self) -> bool {
        !self
            .transactions
            .lock()
            .expect("transactions lock poisoned")
            .is_empty()
    }

    pub fn create_table<C, I>(self: &Arc<Self>, name: &str, columns: I) -> DslResult<TableHandle<P>>
    where
        C: IntoColumnSpec,
        I: IntoIterator<Item = C>,
    {
        self.create_table_with_options(name, columns, false)
    }

    pub fn create_table_if_not_exists<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
    ) -> DslResult<TableHandle<P>>
    where
        C: IntoColumnSpec,
        I: IntoIterator<Item = C>,
    {
        self.create_table_with_options(name, columns, true)
    }

    pub fn create_table_plan(&self, plan: CreateTablePlan) -> DslResult<StatementResult<P>> {
        if plan.columns.is_empty() && plan.source.is_none() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires explicit columns or a source".into(),
            ));
        }

        let (display_name, canonical_name) = canonical_table_name(&plan.name)?;
        let exists = {
            let tables = self.tables.read().unwrap();
            tables.contains_key(&canonical_name)
        };
        if exists {
            if plan.if_not_exists {
                return Ok(StatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' already exists",
                display_name
            )));
        }

        match plan.source {
            Some(CreateTableSource::Batches { schema, batches }) => self.create_table_from_batches(
                display_name,
                canonical_name,
                schema,
                batches,
                plan.if_not_exists,
            ),
            None => self.create_table_from_columns(
                display_name,
                canonical_name,
                plan.columns,
                plan.if_not_exists,
            ),
        }
    }

    pub fn table_names(self: &Arc<Self>) -> Vec<String> {
        let tables = self.tables.read().unwrap();
        tables.keys().cloned().collect()
    }

    pub fn create_table_builder(&self, name: &str) -> CreateTableBuilder<'_, P> {
        CreateTableBuilder {
            ctx: self,
            plan: CreateTablePlan::new(name),
        }
    }

    pub fn table_column_specs(self: &Arc<Self>, name: &str) -> DslResult<Vec<ColumnSpec>> {
        let (_, canonical_name) = canonical_table_name(name)?;
        let table = self.lookup_table(&canonical_name)?;
        Ok(table
            .schema
            .columns
            .iter()
            .map(|column| {
                ColumnSpec::new(
                    column.name.clone(),
                    column.data_type.clone(),
                    column.nullable,
                )
            })
            .collect())
    }

    pub fn export_table_rows(self: &Arc<Self>, name: &str) -> DslResult<RowBatch> {
        let handle = TableHandle::new(Arc::clone(self), name)?;
        handle.lazy()?.collect_rows()
    }

    fn execute_create_table(&self, plan: CreateTablePlan) -> DslResult<StatementResult<P>> {
        self.create_table_plan(plan)
    }

    fn create_table_with_options<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
        if_not_exists: bool,
    ) -> DslResult<TableHandle<P>>
    where
        C: IntoColumnSpec,
        I: IntoIterator<Item = C>,
    {
        let mut plan = CreateTablePlan::new(name);
        plan.if_not_exists = if_not_exists;
        plan.columns = columns
            .into_iter()
            .map(|column| column.into_column_spec())
            .collect();
        let result = self.create_table_plan(plan)?;
        match result {
            StatementResult::CreateTable { .. } => TableHandle::new(Arc::clone(self), name),
            other => Err(Error::InvalidArgumentError(format!(
                "unexpected statement result {other:?} when creating table"
            ))),
        }
    }

    pub fn insert(&self, plan: InsertPlan) -> DslResult<StatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;
        match plan.source {
            InsertSource::Rows(rows) => {
                self.insert_rows(table.as_ref(), display_name, rows, plan.columns)
            }
            InsertSource::Batches(batches) => {
                self.insert_batches(table.as_ref(), display_name, batches, plan.columns)
            }
        }
    }

    /// Get raw batches from a table including row_ids, optionally filtered.
    /// This is used for transaction seeding where we need to preserve existing row_ids.
    pub fn get_batches_with_row_ids(
        &self,
        table_name: &str,
        filter: Option<LlkvExpr<'static, String>>,
    ) -> DslResult<Vec<RecordBatch>> {
        let (_, canonical_name) = canonical_table_name(table_name)?;
        let table = self.lookup_table(&canonical_name)?;

        let filter_expr = match filter {
            Some(expr) => translate_predicate(expr, table.schema.as_ref())?,
            None => {
                let field_id = table.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform wildcard scan".into(),
                    )
                })?;
                full_table_scan_filter(field_id)
            }
        };

        // First, get the row_ids that match the filter
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Scan to get the column data
        let mut batches_without_rowid = Vec::new();
        let table_id = table.table.table_id();
        let projections: Vec<ScanProjection> = table
            .schema
            .columns
            .iter()
            .map(|col| {
                let logical_field_id = LogicalFieldId::for_user(table_id, col.field_id);
                ScanProjection::column((logical_field_id, col.name.clone()))
            })
            .collect();

        let options = ScanStreamOptions {
            include_nulls: true,
            order: None,
        };

        table
            .table
            .scan_stream_with_exprs(&projections, &filter_expr, options, |batch| {
                batches_without_rowid.push(batch.clone());
            })?;

        // Now add row_id column to each batch
        let mut batches_with_rowid = Vec::new();
        let mut row_id_offset = 0usize;

        for batch in batches_without_rowid {
            let batch_size = batch.num_rows();
            let batch_row_ids: Vec<u64> =
                row_ids[row_id_offset..row_id_offset + batch_size].to_vec();
            row_id_offset += batch_size;

            // Create a new batch with row_id as the first column
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns() + 1);
            arrays.push(Arc::new(UInt64Array::from(batch_row_ids)));
            for i in 0..batch.num_columns() {
                arrays.push(batch.column(i).clone());
            }

            let mut fields: Vec<Field> = Vec::with_capacity(batch.schema().fields().len() + 1);
            fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

            // Reconstruct fields with proper field_id metadata
            for (idx, _field) in batch.schema().fields().iter().enumerate() {
                let col = &table.schema.columns[idx];
                let mut metadata = HashMap::new();
                metadata.insert(
                    llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                    col.field_id.to_string(),
                );
                let field_with_metadata =
                    Field::new(&col.name, col.data_type.clone(), col.nullable)
                        .with_metadata(metadata);
                fields.push(field_with_metadata);
            }

            let schema_with_rowid = Arc::new(Schema::new(fields));
            let batch_with_rowid = RecordBatch::try_new(schema_with_rowid, arrays)?;
            batches_with_rowid.push(batch_with_rowid);
        }

        Ok(batches_with_rowid)
    }

    /// Append batches directly to a table, preserving row_ids from the batches.
    /// This is used for transaction seeding where we need to preserve existing row_ids.
    pub fn append_batches_with_row_ids(
        &self,
        table_name: &str,
        batches: Vec<RecordBatch>,
    ) -> DslResult<usize> {
        let (_, canonical_name) = canonical_table_name(table_name)?;
        let table = self.lookup_table(&canonical_name)?;

        let mut total_rows = 0;
        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            // Verify the batch has a row_id column
            let _row_id_idx = batch.schema().index_of(ROW_ID_COLUMN_NAME).map_err(|_| {
                Error::InvalidArgumentError(
                    "batch must contain row_id column for direct append".into(),
                )
            })?;

            // Append the batch directly to the underlying table
            table.table.append(&batch)?;
            total_rows += batch.num_rows();
        }

        Ok(total_rows)
    }

    pub fn update(&self, plan: UpdatePlan) -> DslResult<StatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;
        match plan.filter {
            Some(filter) => {
                self.update_filtered_rows(table.as_ref(), display_name, plan.assignments, filter)
            }
            None => self.update_all_rows(table.as_ref(), display_name, plan.assignments),
        }
    }

    pub fn delete(&self, plan: DeletePlan) -> DslResult<StatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;
        match plan.filter {
            Some(filter) => self.delete_filtered_rows(table.as_ref(), display_name, filter),
            None => self.delete_all_rows(table.as_ref(), display_name),
        }
    }

    pub fn table_handle(self: &Arc<Self>, name: &str) -> DslResult<TableHandle<P>> {
        TableHandle::new(Arc::clone(self), name)
    }

    pub fn execute_select(&self, plan: SelectPlan) -> DslResult<SelectExecution<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;
        if !plan.aggregates.is_empty() {
            self.execute_aggregates(Arc::clone(&table), display_name, plan)
        } else {
            self.execute_projection(table, display_name, plan)
        }
    }

    fn create_table_from_columns(
        &self,
        display_name: String,
        canonical_name: String,
        columns: Vec<ColumnSpec>,
        if_not_exists: bool,
    ) -> DslResult<StatementResult<P>> {
        if columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires at least one column".into(),
            ));
        }

        let mut column_defs: Vec<DslColumn> = Vec::with_capacity(columns.len());
        let mut lookup: HashMap<String, usize> = HashMap::new();
        for (idx, column) in columns.iter().enumerate() {
            let normalized = column.name.to_ascii_lowercase();
            if lookup.insert(normalized.clone(), idx).is_some() {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in table '{}'",
                    column.name, display_name
                )));
            }
            column_defs.push(DslColumn {
                name: column.name.clone(),
                data_type: column.data_type.clone(),
                nullable: column.nullable,
                field_id: (idx + 1) as FieldId,
            });
        }

        let table_id = self.reserve_table_id()?;
        let table = Table::new(table_id, Arc::clone(&self.pager))?;
        table.put_table_meta(&TableMeta {
            table_id,
            name: Some(display_name.clone()),
            created_at_micros: current_time_micros(),
            flags: 0,
            epoch: 0,
        });

        for column in &column_defs {
            table.put_col_meta(&ColMeta {
                col_id: column.field_id,
                name: Some(column.name.clone()),
                flags: 0,
                default: None,
            });
        }

        let schema = Arc::new(DslSchema {
            columns: column_defs,
            lookup,
        });
        let table_entry = Arc::new(DslTable {
            table: Arc::new(table),
            schema,
            next_row_id: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
        });

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            if if_not_exists {
                return Ok(StatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' already exists",
                display_name
            )));
        }
        tables.insert(canonical_name, table_entry);
        Ok(StatementResult::CreateTable {
            table_name: display_name,
        })
    }

    fn create_table_from_batches(
        &self,
        display_name: String,
        canonical_name: String,
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
        if_not_exists: bool,
    ) -> DslResult<StatementResult<P>> {
        if schema.fields().is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE AS SELECT requires at least one column".into(),
            ));
        }
        let mut column_defs: Vec<DslColumn> = Vec::with_capacity(schema.fields().len());
        let mut lookup: HashMap<String, usize> = HashMap::new();
        for (idx, field) in schema.fields().iter().enumerate() {
            let data_type = match field.data_type() {
                DataType::Int64 | DataType::Float64 | DataType::Utf8 | DataType::Date32 => {
                    field.data_type().clone()
                }
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported column type in CTAS result: {other:?}"
                    )));
                }
            };
            let normalized = field.name().to_ascii_lowercase();
            if lookup.insert(normalized.clone(), idx).is_some() {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in CTAS result",
                    field.name()
                )));
            }
            column_defs.push(DslColumn {
                name: field.name().to_string(),
                data_type,
                nullable: field.is_nullable(),
                field_id: (idx + 1) as FieldId,
            });
        }

        let table_id = self.reserve_table_id()?;
        let table = Table::new(table_id, Arc::clone(&self.pager))?;
        table.put_table_meta(&TableMeta {
            table_id,
            name: Some(display_name.clone()),
            created_at_micros: current_time_micros(),
            flags: 0,
            epoch: 0,
        });

        for column in &column_defs {
            table.put_col_meta(&ColMeta {
                col_id: column.field_id,
                name: Some(column.name.clone()),
                flags: 0,
                default: None,
            });
        }

        let schema_arc = Arc::new(DslSchema {
            columns: column_defs.clone(),
            lookup,
        });
        let table_entry = Arc::new(DslTable {
            table: Arc::new(table),
            schema: schema_arc,
            next_row_id: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
        });

        let mut next_row_id: u64 = 0;
        let mut total_rows: u64 = 0;
        for batch in batches {
            let row_count = batch.num_rows();
            if row_count == 0 {
                continue;
            }
            if batch.num_columns() != column_defs.len() {
                return Err(Error::InvalidArgumentError(
                    "CTAS query returned unexpected column count".into(),
                ));
            }
            let start_row = next_row_id;
            next_row_id += row_count as u64;
            total_rows += row_count as u64;

            let mut row_builder = UInt64Builder::with_capacity(row_count);
            for offset in 0..row_count {
                row_builder.append_value(start_row + offset as u64);
            }

            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(row_count + 1);
            arrays.push(Arc::new(row_builder.finish()) as ArrayRef);

            let mut fields: Vec<Field> = Vec::with_capacity(column_defs.len() + 1);
            fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

            for (idx, column) in column_defs.iter().enumerate() {
                let mut metadata = HashMap::new();
                metadata.insert(
                    llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                    column.field_id.to_string(),
                );
                let field = Field::new(&column.name, column.data_type.clone(), column.nullable)
                    .with_metadata(metadata);
                fields.push(field);
                arrays.push(batch.column(idx).clone());
            }

            let append_schema = Arc::new(Schema::new(fields));
            let append_batch = RecordBatch::try_new(append_schema, arrays)?;
            table_entry.table.append(&append_batch)?;
        }

        table_entry.next_row_id.store(next_row_id, Ordering::SeqCst);
        table_entry.total_rows.store(total_rows, Ordering::SeqCst);

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            if if_not_exists {
                return Ok(StatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' already exists",
                display_name
            )));
        }
        tables.insert(canonical_name, table_entry);
        Ok(StatementResult::CreateTable {
            table_name: display_name,
        })
    }

    fn insert_rows(
        &self,
        table: &DslTable<P>,
        display_name: String,
        rows: Vec<Vec<DslValue>>,
        columns: Vec<String>,
    ) -> DslResult<StatementResult<P>> {
        if rows.is_empty() {
            return Err(Error::InvalidArgumentError(
                "INSERT requires at least one row".into(),
            ));
        }

        let column_order = resolve_insert_columns(&columns, table.schema.as_ref())?;
        let expected_len = column_order.len();
        for row in &rows {
            if row.len() != expected_len {
                return Err(Error::InvalidArgumentError(format!(
                    "expected {} values in INSERT row, found {}",
                    expected_len,
                    row.len()
                )));
            }
        }

        let row_count = rows.len();
        let mut column_values: Vec<Vec<DslValue>> =
            vec![Vec::with_capacity(row_count); table.schema.columns.len()];
        for row in rows {
            for (idx, value) in row.into_iter().enumerate() {
                let dest_index = column_order[idx];
                column_values[dest_index].push(value);
            }
        }

        let mut row_id_builder = UInt64Builder::with_capacity(row_count);
        let start_row = table.next_row_id.load(Ordering::SeqCst);
        for offset in 0..row_count {
            row_id_builder.append_value(start_row + offset as u64);
        }

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_values.len() + 1);
        arrays.push(Arc::new(row_id_builder.finish()));

        let mut fields: Vec<Field> = Vec::with_capacity(column_values.len() + 1);
        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        for (column, values) in table.schema.columns.iter().zip(column_values.into_iter()) {
            let array = build_array_for_column(&column.data_type, &values)?;
            let mut metadata = HashMap::new();
            metadata.insert(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                column.field_id.to_string(),
            );
            let field = Field::new(&column.name, column.data_type.clone(), column.nullable)
                .with_metadata(metadata);
            arrays.push(array);
            fields.push(field);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        table.table.append(&batch)?;
        table
            .next_row_id
            .store(start_row + row_count as u64, Ordering::SeqCst);
        table
            .total_rows
            .fetch_add(row_count as u64, Ordering::SeqCst);

        Ok(StatementResult::Insert {
            table_name: display_name,
            rows_inserted: row_count,
        })
    }

    fn insert_batches(
        &self,
        table: &DslTable<P>,
        display_name: String,
        batches: Vec<RecordBatch>,
        columns: Vec<String>,
    ) -> DslResult<StatementResult<P>> {
        if batches.is_empty() {
            return Ok(StatementResult::Insert {
                table_name: display_name,
                rows_inserted: 0,
            });
        }

        let expected_len = if columns.is_empty() {
            table.schema.columns.len()
        } else {
            columns.len()
        };
        let mut total_rows_inserted = 0usize;

        for batch in batches {
            if batch.num_columns() != expected_len {
                return Err(Error::InvalidArgumentError(format!(
                    "expected {} columns in INSERT batch, found {}",
                    expected_len,
                    batch.num_columns()
                )));
            }
            let row_count = batch.num_rows();
            if row_count == 0 {
                continue;
            }
            let mut rows: Vec<Vec<DslValue>> = Vec::with_capacity(row_count);
            for row_idx in 0..row_count {
                let mut row: Vec<DslValue> = Vec::with_capacity(expected_len);
                for col_idx in 0..expected_len {
                    let array = batch.column(col_idx);
                    row.push(dsl_value_from_array(array, row_idx)?);
                }
                rows.push(row);
            }

            match self.insert_rows(table, display_name.clone(), rows, columns.clone())? {
                StatementResult::Insert { rows_inserted, .. } => {
                    total_rows_inserted += rows_inserted;
                }
                _ => unreachable!("insert_rows must return Insert result"),
            }
        }

        Ok(StatementResult::Insert {
            table_name: display_name,
            rows_inserted: total_rows_inserted,
        })
    }

    fn update_filtered_rows(
        &self,
        table: &DslTable<P>,
        display_name: String,
        assignments: Vec<ColumnAssignment>,
        filter: LlkvExpr<'static, String>,
    ) -> DslResult<StatementResult<P>> {
        if assignments.is_empty() {
            return Err(Error::InvalidArgumentError(
                "UPDATE requires at least one assignment".into(),
            ));
        }

        let schema = table.schema.as_ref();
        let filter_expr = translate_predicate(filter, schema)?;

        enum PreparedValue {
            Literal(DslValue),
            Expression { expr_index: usize },
        }

        let mut seen_columns: HashSet<String> = HashSet::new();
        let mut prepared: Vec<(DslColumn, PreparedValue)> = Vec::with_capacity(assignments.len());
        let mut scalar_exprs: Vec<ScalarExpr<FieldId>> = Vec::new();

        for assignment in assignments {
            let normalized = assignment.column.to_ascii_lowercase();
            if !seen_columns.insert(normalized.clone()) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in UPDATE assignments",
                    assignment.column
                )));
            }
            let column = table.schema.resolve(&assignment.column).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column '{}' in UPDATE",
                    assignment.column
                ))
            })?;

            match assignment.value {
                AssignmentValue::Literal(value) => {
                    prepared.push((column.clone(), PreparedValue::Literal(value)));
                }
                AssignmentValue::Expression(expr) => {
                    let translated = translate_scalar(&expr, schema)?;
                    let expr_index = scalar_exprs.len();
                    scalar_exprs.push(translated);
                    prepared.push((column.clone(), PreparedValue::Expression { expr_index }));
                }
            }
        }

        let (row_ids, mut expr_values) =
            self.collect_update_rows(table, &filter_expr, &scalar_exprs)?;

        if row_ids.is_empty() {
            return Ok(StatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let row_count = row_ids.len();
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(prepared.len() + 1);
        arrays.push(Arc::new(UInt64Array::from(row_ids.clone())));

        let mut fields: Vec<Field> = Vec::with_capacity(prepared.len() + 1);
        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        for (column, value) in prepared {
            let values = match value {
                PreparedValue::Literal(lit) => vec![lit; row_count],
                PreparedValue::Expression { expr_index } => {
                    let column_values = expr_values.get_mut(expr_index).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expression assignment value missing during UPDATE".into(),
                        )
                    })?;
                    if column_values.len() != row_count {
                        return Err(Error::InvalidArgumentError(
                            "expression result count did not match targeted row count".into(),
                        ));
                    }
                    mem::take(column_values)
                }
            };
            let array = build_array_for_column(&column.data_type, &values)?;
            arrays.push(array);

            let mut metadata = HashMap::new();
            metadata.insert(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                column.field_id.to_string(),
            );
            let field = Field::new(&column.name, column.data_type.clone(), column.nullable)
                .with_metadata(metadata);
            fields.push(field);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        table.table.append(&batch)?;

        Ok(StatementResult::Update {
            table_name: display_name,
            rows_updated: row_count,
        })
    }

    fn update_all_rows(
        &self,
        table: &DslTable<P>,
        display_name: String,
        assignments: Vec<ColumnAssignment>,
    ) -> DslResult<StatementResult<P>> {
        if assignments.is_empty() {
            return Err(Error::InvalidArgumentError(
                "UPDATE requires at least one assignment".into(),
            ));
        }

        let total_rows = table.total_rows.load(Ordering::SeqCst);
        let total_rows_usize = usize::try_from(total_rows).map_err(|_| {
            Error::InvalidArgumentError("table row count exceeds supported range".into())
        })?;
        if total_rows_usize == 0 {
            return Ok(StatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let schema = table.schema.as_ref();

        enum PreparedValue {
            Literal(DslValue),
            Expression { expr_index: usize },
        }

        let mut seen_columns: HashSet<String> = HashSet::new();
        let mut prepared: Vec<(DslColumn, PreparedValue)> = Vec::with_capacity(assignments.len());
        let mut scalar_exprs: Vec<ScalarExpr<FieldId>> = Vec::new();
        let mut first_field_id: Option<FieldId> = None;

        for assignment in assignments {
            let normalized = assignment.column.to_ascii_lowercase();
            if !seen_columns.insert(normalized.clone()) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in UPDATE assignments",
                    assignment.column
                )));
            }
            let column = table.schema.resolve(&assignment.column).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column '{}' in UPDATE",
                    assignment.column
                ))
            })?;
            if first_field_id.is_none() {
                first_field_id = Some(column.field_id);
            }

            match assignment.value {
                AssignmentValue::Literal(value) => {
                    prepared.push((column.clone(), PreparedValue::Literal(value)));
                }
                AssignmentValue::Expression(expr) => {
                    let translated = translate_scalar(&expr, schema)?;
                    let expr_index = scalar_exprs.len();
                    scalar_exprs.push(translated);
                    prepared.push((column.clone(), PreparedValue::Expression { expr_index }));
                }
            }
        }

        let anchor_field = first_field_id.ok_or_else(|| {
            Error::InvalidArgumentError("UPDATE requires at least one target column".into())
        })?;

        let filter_expr = full_table_scan_filter(anchor_field);
        let (row_ids, mut expr_values) =
            self.collect_update_rows(table, &filter_expr, &scalar_exprs)?;

        if row_ids.is_empty() {
            return Ok(StatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let row_count = row_ids.len();
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(prepared.len() + 1);
        arrays.push(Arc::new(UInt64Array::from(row_ids.clone())));

        let mut fields: Vec<Field> = Vec::with_capacity(prepared.len() + 1);
        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        for (column, value) in prepared {
            let values = match value {
                PreparedValue::Literal(lit) => vec![lit; row_count],
                PreparedValue::Expression { expr_index } => {
                    let column_values = expr_values.get_mut(expr_index).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expression assignment value missing during UPDATE".into(),
                        )
                    })?;
                    if column_values.len() != row_count {
                        return Err(Error::InvalidArgumentError(
                            "expression result count did not match targeted row count".into(),
                        ));
                    }
                    mem::take(column_values)
                }
            };
            let array = build_array_for_column(&column.data_type, &values)?;
            arrays.push(array);

            let mut metadata = HashMap::new();
            metadata.insert(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                column.field_id.to_string(),
            );
            let field = Field::new(&column.name, column.data_type.clone(), column.nullable)
                .with_metadata(metadata);
            fields.push(field);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        table.table.append(&batch)?;

        Ok(StatementResult::Update {
            table_name: display_name,
            rows_updated: row_count,
        })
    }

    fn delete_filtered_rows(
        &self,
        table: &DslTable<P>,
        display_name: String,
        filter: LlkvExpr<'static, String>,
    ) -> DslResult<StatementResult<P>> {
        let schema = table.schema.as_ref();
        let filter_expr = translate_predicate(filter, schema)?;
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        self.apply_delete(table, display_name, row_ids)
    }

    fn delete_all_rows(
        &self,
        table: &DslTable<P>,
        display_name: String,
    ) -> DslResult<StatementResult<P>> {
        let total_rows = table.total_rows.load(Ordering::SeqCst);
        if total_rows == 0 {
            return Ok(StatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        let anchor_field = table.schema.first_field_id().ok_or_else(|| {
            Error::InvalidArgumentError("DELETE requires a table with at least one column".into())
        })?;
        let filter_expr = full_table_scan_filter(anchor_field);
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        self.apply_delete(table, display_name, row_ids)
    }

    fn apply_delete(
        &self,
        table: &DslTable<P>,
        display_name: String,
        row_ids: Vec<u64>,
    ) -> DslResult<StatementResult<P>> {
        if row_ids.is_empty() {
            return Ok(StatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        let logical_fields: Vec<LogicalFieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| LogicalFieldId::for_user(table.table.table_id(), column.field_id))
            .collect();

        if logical_fields.is_empty() {
            return Err(Error::InvalidArgumentError(
                "DELETE requires a table with at least one column".into(),
            ));
        }

        table.table.store().delete_rows(&logical_fields, &row_ids)?;

        let removed = row_ids.len();
        let removed_u64 = u64::try_from(removed)
            .map_err(|_| Error::InvalidArgumentError("row count exceeds supported range".into()))?;
        table.total_rows.fetch_sub(removed_u64, Ordering::SeqCst);

        Ok(StatementResult::Delete {
            table_name: display_name,
            rows_deleted: removed,
        })
    }

    fn collect_update_rows(
        &self,
        table: &DslTable<P>,
        filter_expr: &LlkvExpr<'static, FieldId>,
        expressions: &[ScalarExpr<FieldId>],
    ) -> DslResult<(Vec<u64>, Vec<Vec<DslValue>>)> {
        let row_ids = table.table.filter_row_ids(filter_expr)?;
        if row_ids.is_empty() {
            return Ok((row_ids, vec![Vec::new(); expressions.len()]));
        }

        if expressions.is_empty() {
            return Ok((row_ids, Vec::new()));
        }

        let mut projections: Vec<ScanProjection> = Vec::with_capacity(expressions.len());
        for (idx, expr) in expressions.iter().enumerate() {
            let alias = format!("__expr_{idx}");
            projections.push(ScanProjection::computed(expr.clone(), alias));
        }

        let mut expr_values: Vec<Vec<DslValue>> =
            vec![Vec::with_capacity(row_ids.len()); expressions.len()];
        let mut error: Option<Error> = None;
        let options = ScanStreamOptions {
            include_nulls: true,
            order: None,
        };

        table
            .table
            .scan_stream_with_exprs(&projections, filter_expr, options, |batch| {
                if error.is_some() {
                    return;
                }
                if let Err(err) = Self::collect_expression_values(&mut expr_values, batch) {
                    error = Some(err);
                }
            })?;

        if let Some(err) = error {
            return Err(err);
        }

        for values in &expr_values {
            if values.len() != row_ids.len() {
                return Err(Error::InvalidArgumentError(
                    "expression result count did not match targeted row count".into(),
                ));
            }
        }

        Ok((row_ids, expr_values))
    }

    fn collect_expression_values(
        expr_values: &mut [Vec<DslValue>],
        batch: RecordBatch,
    ) -> DslResult<()> {
        for row_idx in 0..batch.num_rows() {
            for (expr_index, values) in expr_values.iter_mut().enumerate() {
                let value = dsl_value_from_array(batch.column(expr_index), row_idx)?;
                values.push(value);
            }
        }

        Ok(())
    }

    fn execute_projection(
        &self,
        table: Arc<DslTable<P>>,
        display_name: String,
        plan: SelectPlan,
    ) -> DslResult<SelectExecution<P>> {
        let table_ref = table.as_ref();
        let projections = if plan.projections.is_empty() {
            build_wildcard_projections(table_ref)
        } else {
            build_projected_columns(table_ref, &plan.projections)?
        };
        let schema = schema_for_projections(table_ref, &projections)?;

        let (filter_expr, full_table_scan) = match plan.filter {
            Some(expr) => (translate_predicate(expr, table_ref.schema.as_ref())?, false),
            None => {
                let field_id = table_ref.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform wildcard scan".into(),
                    )
                })?;
                (full_table_scan_filter(field_id), true)
            }
        };

        let options = if let Some(order_plan) = &plan.order_by {
            let order_spec = resolve_scan_order(table_ref, &projections, order_plan)?;
            ScanStreamOptions {
                include_nulls: true,
                order: Some(order_spec),
            }
        } else {
            ScanStreamOptions {
                include_nulls: true,
                order: None,
            }
        };

        Ok(SelectExecution::new_projection(
            display_name,
            schema,
            table,
            projections,
            filter_expr,
            options,
            full_table_scan,
        ))
    }

    fn execute_aggregates(
        &self,
        table: Arc<DslTable<P>>,
        display_name: String,
        plan: SelectPlan,
    ) -> DslResult<SelectExecution<P>> {
        let table_ref = table.as_ref();
        let mut specs: Vec<AggregateSpec> = Vec::with_capacity(plan.aggregates.len());
        for aggregate in plan.aggregates {
            match aggregate {
                AggregateExpr::CountStar { alias } => {
                    specs.push(AggregateSpec {
                        alias,
                        kind: AggregateKind::CountStar,
                    });
                }
                AggregateExpr::Column {
                    column,
                    alias,
                    function,
                } => {
                    let col = table_ref.schema.resolve(&column).ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "unknown column '{}' in aggregate",
                            column
                        ))
                    })?;
                    let kind = match function {
                        AggregateFunction::Count => AggregateKind::CountField {
                            field_id: col.field_id,
                        },
                        AggregateFunction::SumInt64 => {
                            if col.data_type != DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "SUM currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::SumInt64 {
                                field_id: col.field_id,
                            }
                        }
                        AggregateFunction::MinInt64 => {
                            if col.data_type != DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "MIN currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::MinInt64 {
                                field_id: col.field_id,
                            }
                        }
                        AggregateFunction::MaxInt64 => {
                            if col.data_type != DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "MAX currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::MaxInt64 {
                                field_id: col.field_id,
                            }
                        }
                        AggregateFunction::CountNulls => AggregateKind::CountNulls {
                            field_id: col.field_id,
                        },
                    };
                    specs.push(AggregateSpec { alias, kind });
                }
            }
        }

        if specs.is_empty() {
            return Err(Error::InvalidArgumentError(
                "aggregate query requires at least one aggregate expression".into(),
            ));
        }

        let had_filter = plan.filter.is_some();
        let filter_expr = match plan.filter {
            Some(expr) => translate_predicate(expr, table.schema.as_ref())?,
            None => {
                let field_id = table.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform aggregate scan".into(),
                    )
                })?;
                full_table_scan_filter(field_id)
            }
        };

        let mut projections = Vec::new();
        for spec in &specs {
            if let Some(field_id) = spec.kind.field_id() {
                projections.push(ScanProjection::from(StoreProjection::with_alias(
                    LogicalFieldId::for_user(table.table.table_id(), field_id),
                    table
                        .schema
                        .column_by_field_id(field_id)
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| format!("col{field_id}")),
                )));
            }
        }

        if projections.is_empty() {
            let field_id = table.schema.first_field_id().ok_or_else(|| {
                Error::InvalidArgumentError(
                    "table has no columns; cannot perform aggregate scan".into(),
                )
            })?;
            projections.push(ScanProjection::from(StoreProjection::with_alias(
                LogicalFieldId::for_user(table.table.table_id(), field_id),
                table
                    .schema
                    .column_by_field_id(field_id)
                    .map(|c| c.name.clone())
                    .unwrap_or_else(|| format!("col{field_id}")),
            )));
        }

        let options = ScanStreamOptions {
            include_nulls: true,
            ..Default::default()
        };

        let mut states: Vec<AggregateState> = Vec::with_capacity(specs.len());
        let mut count_star_override: Option<i64> = None;
        let total_rows = table.total_rows.load(Ordering::SeqCst);
        if !had_filter {
            if total_rows > i64::MAX as u64 {
                return Err(Error::InvalidArgumentError(
                    "COUNT(*) result exceeds supported range".into(),
                ));
            }
            count_star_override = Some(total_rows as i64);
        }

        for spec in &specs {
            states.push(AggregateState {
                alias: spec.alias.clone(),
                accumulator: AggregateAccumulator::new(
                    table.schema.as_ref(),
                    spec,
                    count_star_override,
                )?,
                override_value: match spec.kind {
                    AggregateKind::CountStar => count_star_override,
                    _ => None,
                },
            });
        }

        let mut error: Option<Error> = None;
        match table
            .table
            .scan_stream(projections, &filter_expr, options, |batch| {
                if error.is_some() {
                    return;
                }
                for state in &mut states {
                    if let Err(err) = state.update(&batch) {
                        error = Some(err);
                        return;
                    }
                }
            }) {
            Ok(()) => {}
            Err(llkv_result::Error::NotFound) => {
                // Treat missing storage keys as an empty result set. This occurs
                // for freshly created tables that have no persisted chunks yet.
            }
            Err(err) => return Err(err),
        }
        if let Some(err) = error {
            return Err(err);
        }

        let mut fields = Vec::with_capacity(states.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(states.len());
        for state in states {
            let (field, array) = state.finalize()?;
            fields.push(field);
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
        Ok(SelectExecution::new_single_batch(
            display_name,
            schema,
            batch,
        ))
    }

    fn lookup_table(&self, canonical_name: &str) -> DslResult<Arc<DslTable<P>>> {
        let tables = self.tables.read().unwrap();
        tables
            .get(canonical_name)
            .cloned()
            .ok_or_else(|| Error::InvalidArgumentError(format!("unknown table '{canonical_name}'")))
    }

    fn reserve_table_id(&self) -> DslResult<TableId> {
        let store = ColumnStore::open(Arc::clone(&self.pager))?;
        let catalog = SysCatalog::new(&store);

        let mut next = match catalog.get_next_table_id()? {
            Some(value) => value,
            None => {
                let seed = catalog.max_table_id()?.unwrap_or(CATALOG_TABLE_ID);
                let initial = seed.checked_add(1).ok_or_else(|| {
                    Error::InvalidArgumentError("exhausted available table ids".into())
                })?;
                catalog.put_next_table_id(initial)?;
                initial
            }
        };

        if next == CATALOG_TABLE_ID {
            next = next.checked_add(1).ok_or_else(|| {
                Error::InvalidArgumentError("exhausted available table ids".into())
            })?;
        }

        let mut following = next
            .checked_add(1)
            .ok_or_else(|| Error::InvalidArgumentError("exhausted available table ids".into()))?;
        if following == CATALOG_TABLE_ID {
            following = following.checked_add(1).ok_or_else(|| {
                Error::InvalidArgumentError("exhausted available table ids".into())
            })?;
        }
        catalog.put_next_table_id(following)?;
        Ok(next)
    }
}

fn resolve_scan_order<P>(
    table: &DslTable<P>,
    projections: &[ScanProjection],
    order_plan: &OrderByPlan,
) -> DslResult<ScanOrderSpec>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let (column, field_id) = match &order_plan.target {
        OrderTarget::Column(name) => {
            let column = table.schema.resolve(name).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{}' in ORDER BY", name))
            })?;
            (column, column.field_id)
        }
        OrderTarget::Index(position) => {
            let projection = projections.get(*position).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "ORDER BY position {} is out of range",
                    position + 1
                ))
            })?;
            match projection {
                ScanProjection::Column(store_projection) => {
                    let field_id = store_projection.logical_field_id.field_id();
                    let column = table.schema.column_by_field_id(field_id).ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "unknown column with field id {field_id} in ORDER BY"
                        ))
                    })?;
                    (column, field_id)
                }
                ScanProjection::Computed { .. } => {
                    return Err(Error::InvalidArgumentError(
                        "ORDER BY position referring to computed projection is not supported"
                            .into(),
                    ));
                }
            }
        }
    };

    let transform = match order_plan.sort_type {
        OrderSortType::Native => match column.data_type {
            DataType::Int64 => ScanOrderTransform::IdentityInteger,
            DataType::Utf8 => ScanOrderTransform::IdentityUtf8,
            ref other => {
                return Err(Error::InvalidArgumentError(format!(
                    "ORDER BY on column type {:?} is not supported",
                    other
                )));
            }
        },
        OrderSortType::CastTextToInteger => {
            if column.data_type != DataType::Utf8 {
                return Err(Error::InvalidArgumentError(
                    "ORDER BY CAST expects a text column".into(),
                ));
            }
            ScanOrderTransform::CastUtf8ToInteger
        }
    };

    let direction = if order_plan.ascending {
        ScanOrderDirection::Ascending
    } else {
        ScanOrderDirection::Descending
    };

    Ok(ScanOrderSpec {
        field_id,
        direction,
        nulls_first: order_plan.nulls_first,
        transform,
    })
}

/// Lazily built logical plan.
pub struct LazyFrame<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<DslContext<P>>,
    plan: SelectPlan,
}

impl<P> LazyFrame<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn scan(context: Arc<DslContext<P>>, table: &str) -> DslResult<Self> {
        let (display, canonical) = canonical_table_name(table)?;
        context.lookup_table(&canonical)?;
        Ok(Self {
            context,
            plan: SelectPlan::new(display),
        })
    }

    pub fn filter(mut self, predicate: LlkvExpr<'static, String>) -> Self {
        self.plan.filter = Some(predicate);
        self
    }

    pub fn select(mut self, projections: Vec<SelectProjection>) -> Self {
        self.plan.projections = projections;
        self
    }

    pub fn aggregate(mut self, aggregates: Vec<AggregateExpr>) -> Self {
        self.plan.aggregates = aggregates;
        self
    }

    pub fn collect(self) -> DslResult<SelectExecution<P>> {
        self.context.execute_select(self.plan)
    }

    pub fn collect_rows(self) -> DslResult<RowBatch> {
        let execution = self.context.execute_select(self.plan)?;
        execution.collect_rows()
    }
}

/// Streaming execution handle for SELECT queries.
#[derive(Clone)]
pub struct SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table_name: String,
    schema: Arc<Schema>,
    stream: SelectStream<P>,
}

#[derive(Clone)]
enum SelectStream<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    Projection {
        table: Arc<DslTable<P>>,
        projections: Vec<ScanProjection>,
        filter_expr: LlkvExpr<'static, FieldId>,
        options: ScanStreamOptions,
        full_table_scan: bool,
    },
    Aggregation {
        batch: RecordBatch,
    },
}

impl<P> SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn new_projection(
        table_name: String,
        schema: Arc<Schema>,
        table: Arc<DslTable<P>>,
        projections: Vec<ScanProjection>,
        filter_expr: LlkvExpr<'static, FieldId>,
        options: ScanStreamOptions,
        full_table_scan: bool,
    ) -> Self {
        Self {
            table_name,
            schema,
            stream: SelectStream::Projection {
                table,
                projections,
                filter_expr,
                options,
                full_table_scan,
            },
        }
    }

    fn new_single_batch(table_name: String, schema: Arc<Schema>, batch: RecordBatch) -> Self {
        Self {
            table_name,
            schema,
            stream: SelectStream::Aggregation { batch },
        }
    }

    pub fn from_batch(table_name: String, schema: Arc<Schema>, batch: RecordBatch) -> Self {
        Self::new_single_batch(table_name, schema, batch)
    }

    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    pub fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    pub fn stream(self, mut on_batch: impl FnMut(RecordBatch) -> DslResult<()>) -> DslResult<()> {
        let schema = Arc::clone(&self.schema);
        match self.stream {
            SelectStream::Projection {
                table,
                projections,
                filter_expr,
                options,
                full_table_scan,
            } => {
                // Early return for empty tables to avoid ColumnStore data_type() errors
                let total_rows = table.total_rows.load(Ordering::SeqCst);
                if total_rows == 0 {
                    // Empty table - return empty result with correct schema
                    return Ok(());
                }

                let mut error: Option<Error> = None;
                let mut produced = false;
                let mut produced_rows: u64 = 0;
                let capture_nulls_first = matches!(options.order, Some(spec) if spec.nulls_first);
                let mut buffered_batches: Vec<RecordBatch> = Vec::new();
                table
                    .table
                    .scan_stream(projections, &filter_expr, options, |batch| {
                        if error.is_some() {
                            return;
                        }
                        produced = true;
                        produced_rows = produced_rows.saturating_add(batch.num_rows() as u64);
                        if capture_nulls_first {
                            buffered_batches.push(batch);
                        } else if let Err(err) = on_batch(batch) {
                            error = Some(err);
                        }
                    })?;
                if let Some(err) = error {
                    return Err(err);
                }
                if !produced {
                    if total_rows > 0 {
                        for batch in synthesize_null_scan(Arc::clone(&schema), total_rows)? {
                            on_batch(batch)?;
                        }
                    }
                    return Ok(());
                }
                let mut null_batches: Vec<RecordBatch> = Vec::new();
                if options.include_nulls && full_table_scan && produced_rows < total_rows {
                    let missing = total_rows - produced_rows;
                    if missing > 0 {
                        null_batches = synthesize_null_scan(Arc::clone(&schema), missing)?;
                    }
                }

                if capture_nulls_first {
                    for batch in null_batches {
                        on_batch(batch)?;
                    }
                    for batch in buffered_batches {
                        on_batch(batch)?;
                    }
                } else if !null_batches.is_empty() {
                    for batch in null_batches {
                        on_batch(batch)?;
                    }
                }
                Ok(())
            }
            SelectStream::Aggregation { batch } => on_batch(batch),
        }
    }

    pub fn collect(self) -> DslResult<Vec<RecordBatch>> {
        let mut batches = Vec::new();
        self.stream(|batch| {
            batches.push(batch);
            Ok(())
        })?;
        Ok(batches)
    }

    pub fn collect_rows(self) -> DslResult<RowBatch> {
        let schema = self.schema();
        let mut rows: Vec<Vec<DslValue>> = Vec::new();
        self.stream(|batch| {
            for row_idx in 0..batch.num_rows() {
                let mut row: Vec<DslValue> = Vec::with_capacity(batch.num_columns());
                for col_idx in 0..batch.num_columns() {
                    let value = dsl_value_from_array(batch.column(col_idx), row_idx)?;
                    row.push(value);
                }
                rows.push(row);
            }
            Ok(())
        })?;
        let columns = schema
            .fields()
            .iter()
            .map(|field| field.name().to_string())
            .collect();
        Ok(RowBatch { columns, rows })
    }

    pub fn into_rows(self) -> DslResult<Vec<Vec<DslValue>>> {
        Ok(self.collect_rows()?.rows)
    }
}

impl<P> fmt::Debug for SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectExecution")
            .field("table_name", &self.table_name)
            .field("schema", &self.schema)
            .finish()
    }
}

struct DslTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: Arc<Table<P>>,
    schema: Arc<DslSchema>,
    next_row_id: AtomicU64,
    total_rows: AtomicU64,
}

struct DslSchema {
    columns: Vec<DslColumn>,
    lookup: HashMap<String, usize>,
}

impl DslSchema {
    fn resolve(&self, name: &str) -> Option<&DslColumn> {
        let normalized = name.to_ascii_lowercase();
        self.lookup
            .get(&normalized)
            .and_then(|idx| self.columns.get(*idx))
    }

    fn first_field_id(&self) -> Option<FieldId> {
        self.columns.first().map(|col| col.field_id)
    }

    fn column_by_field_id(&self, field_id: FieldId) -> Option<&DslColumn> {
        self.columns.iter().find(|col| col.field_id == field_id)
    }
}

#[derive(Clone)]
struct DslColumn {
    name: String,
    data_type: DataType,
    nullable: bool,
    field_id: FieldId,
}

#[derive(Clone)]
struct AggregateSpec {
    alias: String,
    kind: AggregateKind,
}

#[derive(Clone)]
enum AggregateKind {
    CountStar,
    CountField { field_id: FieldId },
    SumInt64 { field_id: FieldId },
    MinInt64 { field_id: FieldId },
    MaxInt64 { field_id: FieldId },
    CountNulls { field_id: FieldId },
}

impl AggregateKind {
    fn field_id(&self) -> Option<FieldId> {
        match self {
            AggregateKind::CountStar => None,
            AggregateKind::CountField { field_id }
            | AggregateKind::SumInt64 { field_id }
            | AggregateKind::MinInt64 { field_id }
            | AggregateKind::MaxInt64 { field_id }
            | AggregateKind::CountNulls { field_id } => Some(*field_id),
        }
    }
}

struct AggregateState {
    alias: String,
    accumulator: AggregateAccumulator,
    override_value: Option<i64>,
}

enum AggregateAccumulator {
    CountStar {
        value: i64,
    },
    CountColumn {
        column_index: usize,
        value: i64,
    },
    SumInt64 {
        column_index: usize,
        value: i64,
        saw_value: bool,
    },
    MinInt64 {
        column_index: usize,
        value: Option<i64>,
    },
    MaxInt64 {
        column_index: usize,
        value: Option<i64>,
    },
    CountNulls {
        column_index: usize,
        non_null_rows: i64,
        total_rows: i64,
    },
}

impl AggregateAccumulator {
    fn new(
        schema: &DslSchema,
        spec: &AggregateSpec,
        total_rows_hint: Option<i64>,
    ) -> DslResult<Self> {
        match spec.kind {
            AggregateKind::CountStar => Ok(AggregateAccumulator::CountStar { value: 0 }),
            AggregateKind::CountField { field_id } => {
                let position = schema
                    .columns
                    .iter()
                    .position(|c| c.field_id == field_id)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "aggregate referenced unknown column field id".into(),
                        )
                    })?;
                Ok(AggregateAccumulator::CountColumn {
                    column_index: position,
                    value: 0,
                })
            }
            AggregateKind::SumInt64 { field_id } => {
                let position = schema
                    .columns
                    .iter()
                    .position(|c| c.field_id == field_id)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "aggregate referenced unknown column field id".into(),
                        )
                    })?;
                Ok(AggregateAccumulator::SumInt64 {
                    column_index: position,
                    value: 0,
                    saw_value: false,
                })
            }
            AggregateKind::MinInt64 { field_id } => {
                let position = schema
                    .columns
                    .iter()
                    .position(|c| c.field_id == field_id)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "aggregate referenced unknown column field id".into(),
                        )
                    })?;
                Ok(AggregateAccumulator::MinInt64 {
                    column_index: position,
                    value: None,
                })
            }
            AggregateKind::MaxInt64 { field_id } => {
                let position = schema
                    .columns
                    .iter()
                    .position(|c| c.field_id == field_id)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "aggregate referenced unknown column field id".into(),
                        )
                    })?;
                Ok(AggregateAccumulator::MaxInt64 {
                    column_index: position,
                    value: None,
                })
            }
            AggregateKind::CountNulls { field_id } => {
                let position = schema
                    .columns
                    .iter()
                    .position(|c| c.field_id == field_id)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "aggregate referenced unknown column field id".into(),
                        )
                    })?;
                let total_rows = total_rows_hint.ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "SUM(CASE WHEN ... IS NULL ...) with WHERE clauses is not supported yet"
                            .into(),
                    )
                })?;
                Ok(AggregateAccumulator::CountNulls {
                    column_index: position,
                    non_null_rows: 0,
                    total_rows,
                })
            }
        }
    }

    fn update(&mut self, batch: &RecordBatch) -> DslResult<()> {
        match self {
            AggregateAccumulator::CountStar { value } => {
                let rows = i64::try_from(batch.num_rows()).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *value = value.checked_add(rows).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
            }
            AggregateAccumulator::CountColumn {
                column_index,
                value,
            } => {
                let array = batch.column(*column_index);
                let non_null = (0..array.len()).filter(|idx| array.is_valid(*idx)).count();
                let non_null = i64::try_from(non_null).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *value = value.checked_add(non_null).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
            }
            AggregateAccumulator::SumInt64 {
                column_index,
                value,
                saw_value,
            } => {
                let array = batch.column(*column_index);
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "SUM aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let v = array.value(i);
                        *value = value.checked_add(v).ok_or_else(|| {
                            Error::InvalidArgumentError(
                                "SUM aggregate result exceeds i64 range".into(),
                            )
                        })?;
                        *saw_value = true;
                    }
                }
            }
            AggregateAccumulator::MinInt64 {
                column_index,
                value,
            } => {
                let array = batch.column(*column_index);
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "MIN aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let v = array.value(i);
                        *value = Some(match *value {
                            Some(current) => current.min(v),
                            None => v,
                        });
                    }
                }
            }
            AggregateAccumulator::MaxInt64 {
                column_index,
                value,
            } => {
                let array = batch.column(*column_index);
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "MAX aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let v = array.value(i);
                        *value = Some(match *value {
                            Some(current) => current.max(v),
                            None => v,
                        });
                    }
                }
            }
            AggregateAccumulator::CountNulls {
                column_index,
                non_null_rows,
                total_rows: _,
            } => {
                let array = batch.column(*column_index);
                let non_null = (0..array.len()).filter(|idx| array.is_valid(*idx)).count();
                let non_null = i64::try_from(non_null).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *non_null_rows = non_null_rows.checked_add(non_null).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
            }
        }
        Ok(())
    }

    fn finalize(self) -> DslResult<(Field, ArrayRef)> {
        match self {
            AggregateAccumulator::CountStar { value } => {
                let mut builder = Int64Builder::with_capacity(1);
                builder.append_value(value);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("count", DataType::Int64, false), array))
            }
            AggregateAccumulator::CountColumn { value, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                builder.append_value(value);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("count", DataType::Int64, false), array))
            }
            AggregateAccumulator::SumInt64 {
                value, saw_value, ..
            } => {
                let mut builder = Int64Builder::with_capacity(1);
                if saw_value {
                    builder.append_value(value);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("sum", DataType::Int64, true), array))
            }
            AggregateAccumulator::MinInt64 { value, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                if let Some(v) = value {
                    builder.append_value(v);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("min", DataType::Int64, true), array))
            }
            AggregateAccumulator::MaxInt64 { value, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                if let Some(v) = value {
                    builder.append_value(v);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("max", DataType::Int64, true), array))
            }
            AggregateAccumulator::CountNulls {
                non_null_rows,
                total_rows,
                ..
            } => {
                let nulls = total_rows.checked_sub(non_null_rows).ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "NULL-count aggregate observed more non-null rows than total rows".into(),
                    )
                })?;
                let mut builder = Int64Builder::with_capacity(1);
                builder.append_value(nulls);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("count_nulls", DataType::Int64, false), array))
            }
        }
    }
}

impl AggregateState {
    fn update(&mut self, batch: &RecordBatch) -> DslResult<()> {
        self.accumulator.update(batch)
    }

    fn finalize(self) -> DslResult<(Field, ArrayRef)> {
        let (mut field, array) = self.accumulator.finalize()?;
        field = field.with_name(self.alias);
        if let Some(value) = self.override_value {
            let mut builder = Int64Builder::with_capacity(1);
            builder.append_value(value);
            let array = Arc::new(builder.finish()) as ArrayRef;
            return Ok((field, array));
        }
        Ok((field, array))
    }
}

fn canonical_table_name(name: &str) -> DslResult<(String, String)> {
    if name.is_empty() {
        return Err(Error::InvalidArgumentError(
            "table name must not be empty".into(),
        ));
    }
    let display = name.to_string();
    let canonical = display.to_ascii_lowercase();
    Ok((display, canonical))
}

fn current_time_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

fn resolve_insert_columns(columns: &[String], schema: &DslSchema) -> DslResult<Vec<usize>> {
    if columns.is_empty() {
        return Ok((0..schema.columns.len()).collect());
    }
    let mut resolved = Vec::with_capacity(columns.len());
    for column in columns {
        let normalized = column.to_ascii_lowercase();
        let index = schema
            .lookup
            .get(&normalized)
            .ok_or_else(|| Error::InvalidArgumentError(format!("unknown column '{}'", column)))?;
        resolved.push(*index);
    }
    Ok(resolved)
}

fn build_array_for_column(dtype: &DataType, values: &[DslValue]) -> DslResult<ArrayRef> {
    match dtype {
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    DslValue::Null => builder.append_null(),
                    DslValue::Integer(v) => builder.append_value(*v),
                    DslValue::Float(v) => builder.append_value(*v as i64),
                    DslValue::String(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert string into INT column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    DslValue::Null => builder.append_null(),
                    DslValue::Integer(v) => builder.append_value(*v as f64),
                    DslValue::Float(v) => builder.append_value(*v),
                    DslValue::String(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert string into DOUBLE column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Utf8 => {
            let mut builder = StringBuilder::with_capacity(values.len(), values.len() * 8);
            for value in values {
                match value {
                    DslValue::Null => builder.append_null(),
                    DslValue::Integer(v) => builder.append_value(v.to_string()),
                    DslValue::Float(v) => builder.append_value(v.to_string()),
                    DslValue::String(s) => builder.append_value(s),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Date32 => {
            let mut builder = Date32Builder::with_capacity(values.len());
            for value in values {
                match value {
                    DslValue::Null => builder.append_null(),
                    DslValue::Integer(days) => {
                        let casted = i32::try_from(*days).map_err(|_| {
                            Error::InvalidArgumentError(
                                "integer literal out of range for DATE column".into(),
                            )
                        })?;
                        builder.append_value(casted);
                    }
                    DslValue::Float(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert float into DATE column".into(),
                        ));
                    }
                    DslValue::String(text) => {
                        let days = parse_date32_literal(text)?;
                        builder.append_value(days);
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported Arrow data type for INSERT: {other:?}"
        ))),
    }
}

fn parse_date32_literal(text: &str) -> DslResult<i32> {
    let mut parts = text.split('-');
    let year_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    let month_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    let day_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError(format!(
            "invalid DATE literal '{text}'"
        )));
    }

    let year = year_str.parse::<i32>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid year in DATE literal '{text}'"))
    })?;
    let month_num = month_str.parse::<u8>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid month in DATE literal '{text}'"))
    })?;
    let day = day_str.parse::<u8>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid day in DATE literal '{text}'"))
    })?;

    let month = Month::try_from(month_num).map_err(|_| {
        Error::InvalidArgumentError(format!("invalid month in DATE literal '{text}'"))
    })?;

    let date = Date::from_calendar_date(year, month, day).map_err(|err| {
        Error::InvalidArgumentError(format!("invalid DATE literal '{text}': {err}"))
    })?;
    let days = date.to_julian_day() - epoch_julian_day();
    Ok(days)
}

fn epoch_julian_day() -> i32 {
    Date::from_calendar_date(1970, Month::January, 1)
        .expect("1970-01-01 is a valid date")
        .to_julian_day()
}

fn dsl_value_from_array(array: &ArrayRef, index: usize) -> DslResult<DslValue> {
    if array.is_null(index) {
        return Ok(DslValue::Null);
    }
    match array.data_type() {
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                Error::InvalidArgumentError("expected Int64 array in INSERT SELECT".into())
            })?;
            Ok(DslValue::Integer(values.value(index)))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Float64 array in INSERT SELECT".into())
                })?;
            Ok(DslValue::Float(values.value(index)))
        }
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Utf8 array in INSERT SELECT".into())
                })?;
            Ok(DslValue::String(values.value(index).to_string()))
        }
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Date32 array in INSERT SELECT".into())
                })?;
            Ok(DslValue::Integer(values.value(index) as i64))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported data type in INSERT SELECT: {other:?}"
        ))),
    }
}

fn build_wildcard_projections<P>(table: &DslTable<P>) -> Vec<ScanProjection>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table
        .schema
        .columns
        .iter()
        .map(|column| {
            ScanProjection::from(StoreProjection::with_alias(
                LogicalFieldId::for_user(table.table.table_id(), column.field_id),
                column.name.clone(),
            ))
        })
        .collect()
}

fn build_projected_columns<P>(
    table: &DslTable<P>,
    projections: &[SelectProjection],
) -> DslResult<Vec<ScanProjection>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut result = Vec::with_capacity(projections.len());
    for projection in projections.iter() {
        match projection {
            SelectProjection::AllColumns => {
                result.extend(build_wildcard_projections(table));
            }
            SelectProjection::Column { name, alias } => {
                let column = table.schema.resolve(name).ok_or_else(|| {
                    Error::InvalidArgumentError(format!("unknown column '{}' in projection", name))
                })?;
                let alias = alias.clone().unwrap_or_else(|| column.name.clone());
                result.push(ScanProjection::from(StoreProjection::with_alias(
                    LogicalFieldId::for_user(table.table.table_id(), column.field_id),
                    alias,
                )));
            }
            SelectProjection::Computed { expr, alias } => {
                let scalar = translate_scalar(expr, table.schema.as_ref())?;
                result.push(ScanProjection::computed(scalar, alias.clone()));
            }
        }
    }
    if result.is_empty() {
        return Err(Error::InvalidArgumentError(
            "projection must include at least one column".into(),
        ));
    }
    Ok(result)
}

fn schema_for_projections<P>(
    table: &DslTable<P>,
    projections: &[ScanProjection],
) -> DslResult<Arc<Schema>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut fields: Vec<Field> = Vec::with_capacity(projections.len());
    for projection in projections {
        match projection {
            ScanProjection::Column(proj) => {
                let field_id = proj.logical_field_id.field_id();
                let column = table.schema.column_by_field_id(field_id).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unknown column with field id {} in projection",
                        field_id
                    ))
                })?;
                let name = proj.alias.clone().unwrap_or_else(|| column.name.clone());
                let mut metadata = HashMap::new();
                metadata.insert(
                    llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                    column.field_id.to_string(),
                );
                let field = Field::new(&name, column.data_type.clone(), column.nullable)
                    .with_metadata(metadata);
                fields.push(field);
            }
            ScanProjection::Computed { alias, expr } => {
                let dtype = match expr {
                    ScalarExpr::Literal(Literal::Integer(_)) => DataType::Int64,
                    ScalarExpr::Literal(Literal::Float(_)) => DataType::Float64,
                    ScalarExpr::Literal(Literal::String(_)) => DataType::Utf8,
                    ScalarExpr::Column(field_id) => {
                        let column =
                            table.schema.column_by_field_id(*field_id).ok_or_else(|| {
                                Error::InvalidArgumentError(format!(
                                    "unknown column with field id {} in computed projection",
                                    field_id
                                ))
                            })?;
                        column.data_type.clone()
                    }
                    ScalarExpr::Binary { .. } => DataType::Float64,
                };
                let field = Field::new(alias, dtype, true);
                fields.push(field);
            }
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn full_table_scan_filter(field_id: FieldId) -> LlkvExpr<'static, FieldId> {
    LlkvExpr::Pred(Filter {
        field_id,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    })
}

fn synthesize_null_scan(schema: Arc<Schema>, total_rows: u64) -> DslResult<Vec<RecordBatch>> {
    let row_count = usize::try_from(total_rows).map_err(|_| {
        Error::InvalidArgumentError("table row count exceeds supported in-memory batch size".into())
    })?;

    if row_count == 0 {
        return Ok(Vec::new());
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        columns.push(new_null_array(field.data_type(), row_count));
    }

    let batch = RecordBatch::try_new(schema, columns)?;
    Ok(vec![batch])
}

fn translate_predicate(
    expr: LlkvExpr<'static, String>,
    schema: &DslSchema,
) -> DslResult<LlkvExpr<'static, FieldId>> {
    match expr {
        LlkvExpr::And(list) => {
            let mut converted = Vec::with_capacity(list.len());
            for item in list {
                converted.push(translate_predicate(item, schema)?);
            }
            Ok(LlkvExpr::And(converted))
        }
        LlkvExpr::Or(list) => {
            let mut converted = Vec::with_capacity(list.len());
            for item in list {
                converted.push(translate_predicate(item, schema)?);
            }
            Ok(LlkvExpr::Or(converted))
        }
        LlkvExpr::Not(inner) => Ok(LlkvExpr::Not(Box::new(translate_predicate(
            *inner, schema,
        )?))),
        LlkvExpr::Pred(Filter { field_id, op }) => {
            let column = schema.resolve(&field_id).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{field_id}' in filter"))
            })?;
            Ok(LlkvExpr::Pred(Filter {
                field_id: column.field_id,
                op,
            }))
        }
        LlkvExpr::Compare { left, op, right } => {
            let left = translate_scalar(&left, schema)?;
            let right = translate_scalar(&right, schema)?;
            Ok(LlkvExpr::Compare { left, op, right })
        }
    }
}

fn translate_scalar(
    expr: &ScalarExpr<String>,
    schema: &DslSchema,
) -> DslResult<ScalarExpr<FieldId>> {
    match expr {
        ScalarExpr::Column(name) => {
            let column = schema.resolve(name).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{name}' in expression"))
            })?;
            Ok(ScalarExpr::column(column.field_id))
        }
        ScalarExpr::Literal(lit) => Ok(ScalarExpr::Literal(lit.clone())),
        ScalarExpr::Binary { left, op, right } => {
            let left_expr = translate_scalar(left, schema)?;
            let right_expr = translate_scalar(right, schema)?;
            Ok(ScalarExpr::Binary {
                left: Box::new(left_expr),
                op: *op,
                right: Box::new(right_expr),
            })
        }
    }
}

fn dsl_value_from_sql_expr(expr: &SqlExpr) -> DslResult<DslValue> {
    match expr {
        SqlExpr::Value(value) => dsl_value_from_sql_value(value),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match dsl_value_from_sql_expr(expr)? {
            DslValue::Integer(v) => Ok(DslValue::Integer(-v)),
            DslValue::Float(v) => Ok(DslValue::Float(-v)),
            DslValue::Null | DslValue::String(_) => Err(Error::InvalidArgumentError(
                "cannot negate non-numeric literal".into(),
            )),
        },
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => dsl_value_from_sql_expr(expr),
        SqlExpr::Nested(inner) => dsl_value_from_sql_expr(inner),
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported literal expression: {other:?}"
        ))),
    }
}

fn dsl_value_from_sql_value(value: &ValueWithSpan) -> DslResult<DslValue> {
    match &value.value {
        Value::Null => Ok(DslValue::Null),
        Value::Number(text, _) => {
            if text.contains(['.', 'e', 'E']) {
                let parsed = text.parse::<f64>().map_err(|err| {
                    Error::InvalidArgumentError(format!("invalid float literal: {err}"))
                })?;
                Ok(DslValue::Float(parsed))
            } else {
                let parsed = text.parse::<i64>().map_err(|err| {
                    Error::InvalidArgumentError(format!("invalid integer literal: {err}"))
                })?;
                Ok(DslValue::Integer(parsed))
            }
        }
        Value::Boolean(_) => Err(Error::InvalidArgumentError(
            "BOOLEAN literals are not supported yet".into(),
        )),
        other => {
            if let Some(text) = other.clone().into_string() {
                Ok(DslValue::String(text))
            } else {
                Err(Error::InvalidArgumentError(format!(
                    "unsupported literal: {other:?}"
                )))
            }
        }
    }
}

fn group_by_is_empty(expr: &GroupByExpr) -> bool {
    matches!(
        expr,
        GroupByExpr::Expressions(exprs, modifiers)
            if exprs.is_empty() && modifiers.is_empty()
    )
}

#[derive(Clone)]
pub struct RangeSelectRows {
    rows: Vec<Vec<DslValue>>,
}

impl RangeSelectRows {
    pub fn into_rows(self) -> Vec<Vec<DslValue>> {
        self.rows
    }
}

#[derive(Clone)]
enum RangeProjection {
    Column,
    Literal(DslValue),
}

#[derive(Clone)]
pub struct RangeSpec {
    row_count: usize,
    column_name_lower: String,
    table_alias_lower: Option<String>,
}

impl RangeSpec {
    fn matches_identifier(&self, ident: &str) -> bool {
        let lower = ident.to_ascii_lowercase();
        lower == self.column_name_lower || lower == "range"
    }

    fn matches_table_alias(&self, ident: &str) -> bool {
        let lower = ident.to_ascii_lowercase();
        match &self.table_alias_lower {
            Some(alias) => lower == *alias,
            None => lower == "range",
        }
    }

    fn matches_object_name(&self, name: &ObjectName) -> bool {
        if name.0.len() != 1 {
            return false;
        }
        match &name.0[0] {
            ObjectNamePart::Identifier(ident) => self.matches_table_alias(&ident.value),
            _ => false,
        }
    }
}

pub fn extract_rows_from_range(select: &Select) -> DslResult<Option<RangeSelectRows>> {
    let spec = match parse_range_spec(select)? {
        Some(spec) => spec,
        None => return Ok(None),
    };

    if select.selection.is_some() {
        return Err(Error::InvalidArgumentError(
            "WHERE clauses are not supported for range() SELECT statements".into(),
        ));
    }
    if select.having.is_some()
        || !select.named_window.is_empty()
        || select.qualify.is_some()
        || select.distinct.is_some()
        || select.top.is_some()
        || select.into.is_some()
        || select.prewhere.is_some()
        || !select.lateral_views.is_empty()
        || select.value_table_mode.is_some()
        || !group_by_is_empty(&select.group_by)
    {
        return Err(Error::InvalidArgumentError(
            "advanced SELECT clauses are not supported for range() SELECT statements".into(),
        ));
    }

    let mut projections: Vec<RangeProjection> = Vec::with_capacity(select.projection.len());
    for item in &select.projection {
        let projection = match item {
            SelectItem::Wildcard(_) => RangeProjection::Column,
            SelectItem::QualifiedWildcard(kind, _) => match kind {
                SelectItemQualifiedWildcardKind::ObjectName(object_name) => {
                    if spec.matches_object_name(object_name) {
                        RangeProjection::Column
                    } else {
                        return Err(Error::InvalidArgumentError(
                            "qualified wildcard must reference the range() source".into(),
                        ));
                    }
                }
                SelectItemQualifiedWildcardKind::Expr(_) => {
                    return Err(Error::InvalidArgumentError(
                        "expression-qualified wildcards are not supported for range() SELECT statements".into(),
                    ));
                }
            },
            SelectItem::UnnamedExpr(expr) => build_range_projection_expr(expr, &spec)?,
            SelectItem::ExprWithAlias { expr, .. } => build_range_projection_expr(expr, &spec)?,
        };
        projections.push(projection);
    }

    if projections.is_empty() {
        return Err(Error::InvalidArgumentError(
            "SELECT projection must include at least one column".into(),
        ));
    }

    let mut rows: Vec<Vec<DslValue>> = Vec::with_capacity(spec.row_count);
    for idx in 0..spec.row_count {
        let mut row: Vec<DslValue> = Vec::with_capacity(projections.len());
        for projection in &projections {
            match projection {
                RangeProjection::Column => row.push(DslValue::Integer(idx as i64)),
                RangeProjection::Literal(value) => row.push(value.clone()),
            }
        }
        rows.push(row);
    }

    Ok(Some(RangeSelectRows { rows }))
}

fn build_range_projection_expr(expr: &SqlExpr, spec: &RangeSpec) -> DslResult<RangeProjection> {
    match expr {
        SqlExpr::Identifier(ident) => {
            if spec.matches_identifier(&ident.value) {
                Ok(RangeProjection::Column)
            } else {
                Err(Error::InvalidArgumentError(format!(
                    "unknown column '{}' in range() SELECT",
                    ident.value
                )))
            }
        }
        SqlExpr::CompoundIdentifier(parts) => {
            if parts.len() == 2
                && spec.matches_table_alias(&parts[0].value)
                && spec.matches_identifier(&parts[1].value)
            {
                Ok(RangeProjection::Column)
            } else {
                Err(Error::InvalidArgumentError(
                    "compound identifiers must reference the range() source".into(),
                ))
            }
        }
        SqlExpr::Wildcard(_) | SqlExpr::QualifiedWildcard(_, _) => unreachable!(),
        other => Ok(RangeProjection::Literal(dsl_value_from_sql_expr(other)?)),
    }
}

fn parse_range_spec(select: &Select) -> DslResult<Option<RangeSpec>> {
    if select.from.len() != 1 {
        return Ok(None);
    }
    let item = &select.from[0];
    if !item.joins.is_empty() {
        return Err(Error::InvalidArgumentError(
            "JOIN clauses are not supported for range() SELECT statements".into(),
        ));
    }

    match &item.relation {
        TableFactor::Function {
            lateral,
            name,
            args,
            alias,
        } => {
            if *lateral {
                return Err(Error::InvalidArgumentError(
                    "LATERAL range() is not supported".into(),
                ));
            }
            parse_range_spec_from_args(name, args, alias)
        }
        TableFactor::Table {
            name,
            alias,
            args: Some(table_args),
            with_ordinality,
            ..
        } => {
            if *with_ordinality {
                return Err(Error::InvalidArgumentError(
                    "WITH ORDINALITY is not supported for range()".into(),
                ));
            }
            if table_args.settings.is_some() {
                return Err(Error::InvalidArgumentError(
                    "range() SETTINGS clause is not supported".into(),
                ));
            }
            parse_range_spec_from_args(name, &table_args.args, alias)
        }
        _ => Ok(None),
    }
}

fn parse_range_spec_from_args(
    name: &ObjectName,
    args: &[FunctionArg],
    alias: &Option<TableAlias>,
) -> DslResult<Option<RangeSpec>> {
    if name.0.len() != 1 {
        return Ok(None);
    }
    let func_name = match &name.0[0] {
        ObjectNamePart::Identifier(ident) => ident.value.to_ascii_lowercase(),
        _ => return Ok(None),
    };
    if func_name != "range" {
        return Ok(None);
    }

    if args.len() != 1 {
        return Err(Error::InvalidArgumentError(
            "range() requires exactly one argument".into(),
        ));
    }

    let arg_expr = match &args[0] {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
        FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_))
        | FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {
            return Err(Error::InvalidArgumentError(
                "range() argument must be an integer literal".into(),
            ));
        }
        FunctionArg::Named { .. } | FunctionArg::ExprNamed { .. } => {
            return Err(Error::InvalidArgumentError(
                "named arguments are not supported for range()".into(),
            ));
        }
    };

    let value = dsl_value_from_sql_expr(arg_expr)?;
    let row_count = match value {
        DslValue::Integer(v) if v >= 0 => v as usize,
        DslValue::Integer(_) => {
            return Err(Error::InvalidArgumentError(
                "range() argument must be non-negative".into(),
            ));
        }
        _ => {
            return Err(Error::InvalidArgumentError(
                "range() argument must be an integer literal".into(),
            ));
        }
    };

    let column_name_lower = alias
        .as_ref()
        .and_then(|a| {
            a.columns
                .first()
                .map(|col| col.name.value.to_ascii_lowercase())
        })
        .unwrap_or_else(|| "range".to_string());
    let table_alias_lower = alias.as_ref().map(|a| a.name.value.to_ascii_lowercase());

    Ok(Some(RangeSpec {
        row_count,
        column_name_lower,
        table_alias_lower,
    }))
}

#[derive(Clone, Debug)]
pub struct RowBatch {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<DslValue>>,
}

pub struct CreateTableBuilder<'ctx, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    ctx: &'ctx DslContext<P>,
    plan: CreateTablePlan,
}

impl<'ctx, P> CreateTableBuilder<'ctx, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn if_not_exists(mut self) -> Self {
        self.plan.if_not_exists = true;
        self
    }

    pub fn with_column(mut self, name: impl Into<String>, data_type: DataType) -> Self {
        self.plan
            .columns
            .push(ColumnSpec::new(name.into(), data_type, true));
        self
    }

    pub fn with_not_null_column(mut self, name: impl Into<String>, data_type: DataType) -> Self {
        self.plan
            .columns
            .push(ColumnSpec::new(name.into(), data_type, false));
        self
    }

    pub fn with_column_spec(mut self, spec: ColumnSpec) -> Self {
        self.plan.columns.push(spec);
        self
    }

    pub fn finish(self) -> DslResult<StatementResult<P>> {
        self.ctx.execute_create_table(self.plan)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Row {
    values: Vec<(String, DslValue)>,
}

impl Row {
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    pub fn with(mut self, name: impl Into<String>, value: impl Into<DslValue>) -> Self {
        self.set(name, value);
        self
    }

    pub fn set(&mut self, name: impl Into<String>, value: impl Into<DslValue>) -> &mut Self {
        let name = name.into();
        let value = value.into();
        if let Some((_, existing)) = self.values.iter_mut().find(|(n, _)| *n == name) {
            *existing = value;
        } else {
            self.values.push((name, value));
        }
        self
    }

    fn columns(&self) -> Vec<String> {
        self.values.iter().map(|(n, _)| n.clone()).collect()
    }

    fn values_for_columns(&self, columns: &[String]) -> DslResult<Vec<DslValue>> {
        let mut out = Vec::with_capacity(columns.len());
        for column in columns {
            let value = self
                .values
                .iter()
                .find(|(name, _)| name == column)
                .ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "insert row missing value for column '{}'",
                        column
                    ))
                })?;
            out.push(value.1.clone());
        }
        Ok(out)
    }
}

pub fn row() -> Row {
    Row::new()
}

#[doc(hidden)]
pub enum InsertRowKind {
    Named {
        columns: Vec<String>,
        values: Vec<DslValue>,
    },
    Positional(Vec<DslValue>),
}

pub trait IntoInsertRow {
    fn into_insert_row(self) -> DslResult<InsertRowKind>;
}

impl IntoInsertRow for Row {
    fn into_insert_row(self) -> DslResult<InsertRowKind> {
        let row = self;
        if row.values.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        let columns = row.columns();
        let values = row.values_for_columns(&columns)?;
        Ok(InsertRowKind::Named { columns, values })
    }
}

impl<T> IntoInsertRow for &T
where
    T: Clone + IntoInsertRow,
{
    fn into_insert_row(self) -> DslResult<InsertRowKind> {
        self.clone().into_insert_row()
    }
}

impl<T> IntoInsertRow for Vec<T>
where
    T: Into<DslValue>,
{
    fn into_insert_row(self) -> DslResult<InsertRowKind> {
        if self.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        Ok(InsertRowKind::Positional(
            self.into_iter().map(Into::into).collect(),
        ))
    }
}

impl<T, const N: usize> IntoInsertRow for [T; N]
where
    T: Into<DslValue>,
{
    fn into_insert_row(self) -> DslResult<InsertRowKind> {
        if N == 0 {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        Ok(InsertRowKind::Positional(
            self.into_iter().map(Into::into).collect(),
        ))
    }
}

macro_rules! impl_into_insert_row_tuple {
    ($($type:ident => $value:ident),+) => {
        impl<$($type,)+> IntoInsertRow for ($($type,)+)
        where
            $($type: Into<DslValue>,)+
        {
            fn into_insert_row(self) -> DslResult<InsertRowKind> {
                let ($($value,)+) = self;
                Ok(InsertRowKind::Positional(vec![$($value.into(),)+]))
            }
        }
    };
}

impl_into_insert_row_tuple!(T1 => v1);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2, T3 => v3);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2, T3 => v3, T4 => v4);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2, T3 => v3, T4 => v4, T5 => v5);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2, T3 => v3, T4 => v4, T5 => v5, T6 => v6);
impl_into_insert_row_tuple!(
    T1 => v1,
    T2 => v2,
    T3 => v3,
    T4 => v4,
    T5 => v5,
    T6 => v6,
    T7 => v7
);
impl_into_insert_row_tuple!(
    T1 => v1,
    T2 => v2,
    T3 => v3,
    T4 => v4,
    T5 => v5,
    T6 => v6,
    T7 => v7,
    T8 => v8
);

pub struct TableHandle<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<DslContext<P>>,
    display_name: String,
    _canonical_name: String,
}

impl<P> TableHandle<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(context: Arc<DslContext<P>>, name: &str) -> DslResult<Self> {
        let (display_name, canonical_name) = canonical_table_name(name)?;
        context.lookup_table(&canonical_name)?;
        Ok(Self {
            context,
            display_name,
            _canonical_name: canonical_name,
        })
    }

    pub fn lazy(&self) -> DslResult<LazyFrame<P>> {
        LazyFrame::scan(Arc::clone(&self.context), &self.display_name)
    }

    pub fn insert_rows<R>(&self, rows: impl IntoIterator<Item = R>) -> DslResult<StatementResult<P>>
    where
        R: IntoInsertRow,
    {
        enum InsertMode {
            Named,
            Positional,
        }

        let table = self.context.lookup_table(&self._canonical_name)?;
        let schema = table.schema.as_ref();
        let schema_column_names: Vec<String> =
            schema.columns.iter().map(|col| col.name.clone()).collect();
        let mut normalized_rows: Vec<Vec<DslValue>> = Vec::new();
        let mut mode: Option<InsertMode> = None;
        let mut column_names: Option<Vec<String>> = None;
        let mut row_count = 0usize;

        for row in rows.into_iter() {
            row_count += 1;
            match row.into_insert_row()? {
                InsertRowKind::Named { columns, values } => {
                    if let Some(existing) = &mode {
                        if !matches!(existing, InsertMode::Named) {
                            return Err(Error::InvalidArgumentError(
                                "cannot mix positional and named insert rows".into(),
                            ));
                        }
                    } else {
                        mode = Some(InsertMode::Named);
                        let mut seen = HashSet::new();
                        for column in &columns {
                            if !seen.insert(column.clone()) {
                                return Err(Error::InvalidArgumentError(format!(
                                    "duplicate column '{}' in insert row",
                                    column
                                )));
                            }
                        }
                        column_names = Some(columns.clone());
                    }

                    let expected = column_names
                        .as_ref()
                        .expect("column names must be initialized for named insert");
                    if columns != *expected {
                        return Err(Error::InvalidArgumentError(
                            "insert rows must specify the same columns".into(),
                        ));
                    }
                    if values.len() != expected.len() {
                        return Err(Error::InvalidArgumentError(format!(
                            "insert row expected {} values, found {}",
                            expected.len(),
                            values.len()
                        )));
                    }
                    normalized_rows.push(values);
                }
                InsertRowKind::Positional(values) => {
                    if let Some(existing) = &mode {
                        if !matches!(existing, InsertMode::Positional) {
                            return Err(Error::InvalidArgumentError(
                                "cannot mix positional and named insert rows".into(),
                            ));
                        }
                    } else {
                        mode = Some(InsertMode::Positional);
                        column_names = Some(schema_column_names.clone());
                    }

                    if values.len() != schema.columns.len() {
                        return Err(Error::InvalidArgumentError(format!(
                            "insert row expected {} values, found {}",
                            schema.columns.len(),
                            values.len()
                        )));
                    }
                    normalized_rows.push(values);
                }
            }
        }

        if row_count == 0 {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one row".into(),
            ));
        }

        let columns = column_names.unwrap_or_else(|| schema_column_names.clone());
        self.insert_row_batch(RowBatch {
            columns,
            rows: normalized_rows,
        })
    }

    pub fn insert_row_batch(&self, batch: RowBatch) -> DslResult<StatementResult<P>> {
        if batch.rows.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one row".into(),
            ));
        }
        if batch.columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        for row in &batch.rows {
            if row.len() != batch.columns.len() {
                return Err(Error::InvalidArgumentError(
                    "insert rows must have values for every column".into(),
                ));
            }
        }

        let plan = InsertPlan {
            table: self.display_name.clone(),
            columns: batch.columns,
            source: InsertSource::Rows(batch.rows),
        };
        self.context.insert(plan)
    }

    pub fn insert_batches(&self, batches: Vec<RecordBatch>) -> DslResult<StatementResult<P>> {
        let plan = InsertPlan {
            table: self.display_name.clone(),
            columns: Vec::new(),
            source: InsertSource::Batches(batches),
        };
        self.context.insert(plan)
    }

    pub fn insert_lazy(&self, frame: LazyFrame<P>) -> DslResult<StatementResult<P>> {
        let RowBatch { columns, rows } = frame.collect_rows()?;
        self.insert_row_batch(RowBatch { columns, rows })
    }

    pub fn name(&self) -> &str {
        &self.display_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, StringArray};
    use llkv_storage::pager::MemPager;
    use std::sync::Arc;

    #[test]
    fn create_insert_select_roundtrip() {
        let pager = Arc::new(MemPager::default());
        let context = Arc::new(DslContext::new(pager));

        let table = context
            .create_table(
                "people",
                [
                    ("id", DataType::Int64, NotNull),
                    ("name", DataType::Utf8, Nullable),
                ],
            )
            .expect("create table");
        table
            .insert_rows([(1_i64, "alice"), (2_i64, "bob")])
            .expect("insert rows");

        let execution = table.lazy().expect("lazy scan");
        let select = execution.collect().expect("build select execution");
        let batches = select.collect().expect("collect batches");
        assert_eq!(batches.len(), 1);
        let column = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string column");
        assert_eq!(column.len(), 2);
    }

    #[test]
    fn aggregate_count_nulls() {
        let pager = Arc::new(MemPager::default());
        let context = Arc::new(DslContext::new(pager));

        let table = context
            .create_table("ints", [("i", DataType::Int64)])
            .expect("create table");
        table
            .insert_rows([
                (DslValue::Null,),
                (DslValue::Integer(1),),
                (DslValue::Null,),
            ])
            .expect("insert rows");

        let plan =
            SelectPlan::new("ints").with_aggregates(vec![AggregateExpr::count_nulls("i", "nulls")]);
        let execution = context.execute_select(plan).expect("select");
        let batches = execution.collect().expect("collect batches");
        let column = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int column");
        assert_eq!(column.value(0), 2);
    }
}
