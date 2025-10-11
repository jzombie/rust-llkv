use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ArrayRef, Int64Array, Int64Builder, RecordBatch};
use arrow::datatypes::{Field, Schema};

use llkv_expr::expr::Expr as LlkvExpr;
use llkv_plan::plans::{
    AggregateExpr, AggregateFunction, ColumnSpec, CreateTablePlan, DeletePlan, DslOperation,
    InsertPlan, InsertSource, OrderByPlan, OrderTarget, PlanValue, SelectPlan, UpdatePlan,
};
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::{MemPager, Pager};
use simd_r_drive_entry_handle::EntryHandle;

use llkv_executor::SelectExecution;

// ============================================================================
// Type Definitions
// ============================================================================

/// Simplified row batch for export/import
pub struct RowBatch {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<PlanValue>>,
}

/// Transaction kind enum
#[derive(Clone, Debug)]
pub enum TransactionKind {
    Begin,
    Commit,
    Rollback,
}

/// Transaction result enum (simplified version for transaction module)
#[derive(Clone, Debug)]
pub enum TransactionResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    CreateTable {
        table_name: String,
    },
    Insert {
        rows_inserted: usize,
    },
    Update {
        rows_matched: usize,
        rows_updated: usize,
    },
    Delete {
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

impl<P> TransactionResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Convert pager type for compatibility
    pub fn convert_pager_type<P2>(self) -> LlkvResult<TransactionResult<P2>>
    where
        P2: Pager<Blob = EntryHandle> + Send + Sync + 'static,
    {
        match self {
            TransactionResult::CreateTable { table_name } => {
                Ok(TransactionResult::CreateTable { table_name })
            }
            TransactionResult::Insert { rows_inserted } => {
                Ok(TransactionResult::Insert { rows_inserted })
            }
            TransactionResult::Update {
                rows_matched,
                rows_updated,
            } => Ok(TransactionResult::Update {
                rows_matched,
                rows_updated,
            }),
            TransactionResult::Delete { rows_deleted } => {
                Ok(TransactionResult::Delete { rows_deleted })
            }
            TransactionResult::Transaction { kind } => Ok(TransactionResult::Transaction { kind }),
            TransactionResult::Select { .. } => Err(Error::Internal(
                "cannot convert SELECT TransactionResult between pager types".into(),
            )),
        }
    }
}

// ============================================================================
// Transaction Management Types
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
    // This constructor is used by or_insert_with(TableDeltaState::new)
    // in several places but may be flagged as unused by clippy in some
    // build configurations. Allow dead_code here to silence that warning
    // while keeping the constructor available for potential future use.
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            seeded_filters: HashSet::new(),
            exclusion_predicates: Vec::new(),
            exclude_all_rows: false,
        }
    }
}

/// A trait for transaction context operations.
/// This allows DslTransaction to work with any context that implements these operations.
/// The associated type P specifies the pager type this context uses.
pub trait TransactionContext: Send + Sync {
    /// The pager type used by this context
    type Pager: Pager<Blob = EntryHandle> + Send + Sync + 'static;

    /// Get table column specifications
    fn table_column_specs(&self, table_name: &str) -> LlkvResult<Vec<ColumnSpec>>;

    /// Export table rows for snapshotting
    fn export_table_rows(&self, table_name: &str) -> LlkvResult<RowBatch>;

    /// Get batches with row IDs for seeding updates
    fn get_batches_with_row_ids(
        &self,
        table_name: &str,
        filter: Option<LlkvExpr<'static, String>>,
    ) -> LlkvResult<Vec<RecordBatch>>;

    /// Execute a SELECT plan with this context's pager type
    fn execute_select(&self, plan: SelectPlan) -> LlkvResult<SelectExecution<Self::Pager>>;

    /// Create a table from plan
    fn create_table_plan(&self, plan: CreateTablePlan) -> LlkvResult<TransactionResult<MemPager>>;

    /// Insert rows
    fn insert(&self, plan: InsertPlan) -> LlkvResult<TransactionResult<MemPager>>;

    /// Update rows
    fn update(&self, plan: UpdatePlan) -> LlkvResult<TransactionResult<MemPager>>;

    /// Delete rows
    fn delete(&self, plan: DeletePlan) -> LlkvResult<TransactionResult<MemPager>>;

    /// Append batches with row IDs
    fn append_batches_with_row_ids(
        &self,
        table_name: &str,
        batches: Vec<RecordBatch>,
    ) -> LlkvResult<usize>;

    /// Get table names for catalog snapshot
    fn table_names(&self) -> Vec<String>;
}

/// Transaction state for the DSL context.
pub struct DslTransaction<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    /// Staging context with MemPager for isolation.
    staging: Arc<StagingCtx>,
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
    base_context: Arc<BaseCtx>,
    /// Whether this transaction has been aborted due to an error.
    is_aborted: bool,
}

impl<BaseCtx, StagingCtx> DslTransaction<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    pub fn new(base_context: Arc<BaseCtx>, staging: Arc<StagingCtx>) -> Self {
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
            is_aborted: false,
        }
    }

    /// Ensure a table exists in the staging context, snapshotting from base if needed.
    fn ensure_table_in_delta(&mut self, table_name: &str) -> LlkvResult<()> {
        tracing::trace!(
            "[ENSURE] ensure_table_in_delta called for table='{}'",
            table_name
        );
        tracing::trace!(
            "[ENSURE]   staged_tables.contains={}",
            self.staged_tables.contains(table_name)
        );
        if self.staged_tables.contains(table_name) {
            tracing::trace!("[ENSURE]   returning early - table already staged");
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
        tracing::trace!("[ENSURE] About to check base_context.table_column_specs");
        match self.base_context.table_column_specs(table_name) {
            Ok(specs) => {
                tracing::trace!(
                    "[ENSURE] Got {} specs from BASE context for table '{}'",
                    specs.len(),
                    table_name
                );
                for spec in &specs {
                    tracing::trace!(
                        "[ENSURE]   spec: name='{}' primary_key={}",
                        spec.name,
                        spec.primary_key
                    );
                }
                self.missing_tables.remove(table_name);
                let mut plan = CreateTablePlan::new(table_name.to_string());
                plan.if_not_exists = true;
                plan.columns = specs;
                tracing::trace!(
                    "[ENSURE] About to create table in staging with {} columns",
                    plan.columns.len()
                );
                self.staging.create_table_plan(plan)?;
                tracing::trace!("[ENSURE] Created table in staging successfully");

                // Snapshot rows from base
                tracing::trace!("[ENSURE] About to snapshot rows from base context");
                match self.base_context.export_table_rows(table_name) {
                    Ok(snapshot) => {
                        tracing::trace!("[ENSURE] Got {} rows from base", snapshot.rows.len());
                        if !snapshot.rows.is_empty() {
                            tracing::trace!(
                                "[ENSURE] About to INSERT {} rows into staging",
                                snapshot.rows.len()
                            );
                            let insert_plan = InsertPlan {
                                table: table_name.to_string(),
                                columns: snapshot.columns.clone(),
                                source: InsertSource::Rows(snapshot.rows),
                            };
                            self.staging.insert(insert_plan)?;
                            tracing::trace!(
                                "[ENSURE] Successfully inserted snapshot rows into staging"
                            );
                        }
                        self.snapshotted_tables.insert(table_name.to_string());
                        tracing::trace!("[ENSURE] Marked table as snapshotted");
                    }
                    Err(Error::NotFound) => {
                        tracing::trace!(
                            "[ENSURE] No rows found in base, marking as snapshotted anyway"
                        );
                        self.snapshotted_tables.insert(table_name.to_string());
                    }
                    Err(other) => return Err(other),
                }
            }
            Err(Error::NotFound) => {
                tracing::trace!("[ENSURE] Table not found in BASE context, checking STAGING");
                // Check if it exists in staging only
                match self.staging.table_column_specs(table_name) {
                    Ok(_) => {
                        tracing::trace!("[ENSURE] Found in STAGING only, marking as staged");
                        self.staged_tables.insert(table_name.to_string());
                        return Ok(());
                    }
                    Err(_) => {
                        tracing::trace!("[ENSURE] Not found in STAGING either, returning error");
                        self.missing_tables.insert(table_name.to_string());
                        return Err(Error::CatalogError(format!(
                            "Catalog Error: Table '{table_name}' does not exist"
                        )));
                    }
                }
            }
            Err(other) => {
                tracing::trace!(
                    "[ENSURE] base_context.table_column_specs returned error: {:?}",
                    other
                );
                return Err(other);
            }
        }

        self.staged_tables.insert(table_name.to_string());
        Ok(())
    }

    /// Seed rows from base table for UPDATE/DELETE operations.
    // May be used by UPDATE/DELETE code paths in the future; keep it
    // but allow dead_code so CI (clippy -D warnings) won't fail when it's
    // not referenced in all build configurations.
    #[allow(dead_code)]
    fn seed_rows_for_update(
        &mut self,
        table_name: &str,
        filter: Option<&LlkvExpr<'static, String>>,
    ) -> LlkvResult<()> {
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
    // Similar rationale as seed_rows_for_update: keep the helper but
    // allow dead_code until it's wired into all code paths.
    #[allow(dead_code)]
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
    pub fn execute_select(
        &mut self,
        plan: SelectPlan,
    ) -> LlkvResult<SelectExecution<BaseCtx::Pager>> {
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
            if let Some(delta) = self
                .table_deltas
                .get(&plan.table)
                .filter(|d| !d.exclusion_predicates.is_empty())
            {
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
        base: Option<SelectExecution<BaseCtx::Pager>>,
        delta: SelectExecution<StagingCtx::Pager>,
        plan: &SelectPlan,
    ) -> LlkvResult<SelectExecution<BaseCtx::Pager>> {
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
    ) -> LlkvResult<SelectExecution<BaseCtx::Pager>> {
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
    ) -> LlkvResult<Option<i64>> {
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
    ) -> LlkvResult<RecordBatch> {
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

    /// Execute an operation in the transaction staging context
    pub fn execute_operation(
        &mut self,
        operation: DslOperation,
    ) -> LlkvResult<TransactionResult<BaseCtx::Pager>> {
        tracing::trace!(
            "[TX] DslTransaction::execute_operation called, operation={:?}",
            match &operation {
                DslOperation::Insert(p) => format!("INSERT({})", p.table),
                DslOperation::Update(p) => format!("UPDATE({})", p.table),
                DslOperation::Delete(p) => format!("DELETE({})", p.table),
                DslOperation::CreateTable(p) => format!("CREATE_TABLE({})", p.name),
                _ => "OTHER".to_string(),
            }
        );
        // Check if transaction is aborted
        if self.is_aborted {
            return Err(Error::TransactionContextError(
                "TransactionContext Error: transaction is aborted".into(),
            ));
        }

        // Execute operation and catch errors to mark transaction as aborted
        let result = match operation {
            DslOperation::CreateTable(ref plan) => {
                match self.staging.create_table_plan(plan.clone()) {
                    Ok(result) => {
                        // Track new table so it's visible to subsequent operations in this transaction
                        self.new_tables.insert(plan.name.clone());
                        self.missing_tables.remove(&plan.name);
                        self.staged_tables.insert(plan.name.clone());
                        // Track for commit replay
                        self.operations
                            .push(DslOperation::CreateTable(plan.clone()));
                        result.convert_pager_type()?
                    }
                    Err(e) => {
                        self.is_aborted = true;
                        return Err(e);
                    }
                }
            }
            DslOperation::Insert(ref plan) => {
                tracing::trace!(
                    "[TX] DslTransaction::execute_operation INSERT for table='{}'",
                    plan.table
                );
                // First ensure table exists
                if let Err(e) = self.ensure_table_in_delta(&plan.table) {
                    self.is_aborted = true;
                    return Err(e);
                }
                tracing::trace!("[TX] About to call self.staging.insert...");
                match self.staging.insert(plan.clone()) {
                    Ok(result) => {
                        tracing::trace!("[TX] INSERT succeeded, pushing operation to replay queue");
                        // Track for commit replay
                        self.operations.push(DslOperation::Insert(plan.clone()));
                        result.convert_pager_type()?
                    }
                    Err(e) => {
                        tracing::trace!(
                            "DEBUG DslTransaction::execute_operation INSERT failed: {:?}",
                            e
                        );
                        tracing::trace!("DEBUG setting is_aborted=true");
                        self.is_aborted = true;
                        return Err(e);
                    }
                }
            }
            DslOperation::Update(ref plan) => {
                if let Err(e) = self.ensure_table_in_delta(&plan.table) {
                    self.is_aborted = true;
                    return Err(e);
                }
                match self.staging.update(plan.clone()) {
                    Ok(result) => {
                        // Track for commit replay
                        self.operations.push(DslOperation::Update(plan.clone()));
                        result.convert_pager_type()?
                    }
                    Err(e) => {
                        self.is_aborted = true;
                        return Err(e);
                    }
                }
            }
            DslOperation::Delete(ref plan) => {
                if let Err(e) = self.ensure_table_in_delta(&plan.table) {
                    self.is_aborted = true;
                    return Err(e);
                }
                match self.staging.delete(plan.clone()) {
                    Ok(result) => {
                        // Track for commit replay
                        self.operations.push(DslOperation::Delete(plan.clone()));
                        result.convert_pager_type()?
                    }
                    Err(e) => {
                        self.is_aborted = true;
                        return Err(e);
                    }
                }
            }
            DslOperation::Select(ref plan) => {
                // SELECT is read-only, not tracked for replay
                // But still fails if transaction is aborted (already checked above)
                let table_name = plan.table.clone();
                match self.execute_select(plan.clone()) {
                    Ok(execution) => {
                        let schema = execution.schema();
                        TransactionResult::Select {
                            table_name,
                            schema,
                            execution,
                        }
                    }
                    Err(e) => {
                        // Don't abort on SELECT errors (like table not found)
                        // Only Session layer aborts on constraint violations
                        return Err(e);
                    }
                }
            }
        };

        Ok(result)
    }

    /// Get the operations queued for commit
    pub fn operations(&self) -> &[DslOperation] {
        &self.operations
    }
}

/// A session handle for transaction management.
/// When dropped, automatically rolls back any active transaction.
pub struct TransactionSession<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    context: Arc<BaseCtx>,
    session_id: u64,
    transactions: Arc<Mutex<HashMap<u64, DslTransaction<BaseCtx, StagingCtx>>>>,
}

impl<BaseCtx, StagingCtx> TransactionSession<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    pub fn new(
        context: Arc<BaseCtx>,
        session_id: u64,
        transactions: Arc<Mutex<HashMap<u64, DslTransaction<BaseCtx, StagingCtx>>>>,
    ) -> Self {
        Self {
            context,
            session_id,
            transactions,
        }
    }

    /// Get the session ID.
    pub fn session_id(&self) -> u64 {
        self.session_id
    }

    /// Get the underlying context (for advanced use).
    pub fn context(&self) -> &Arc<BaseCtx> {
        &self.context
    }

    /// Check if this session has an active transaction.
    pub fn has_active_transaction(&self) -> bool {
        self.transactions
            .lock()
            .expect("transactions lock poisoned")
            .contains_key(&self.session_id)
    }

    /// Check if the current transaction has been aborted due to an error.
    pub fn is_aborted(&self) -> bool {
        self.transactions
            .lock()
            .expect("transactions lock poisoned")
            .get(&self.session_id)
            .map(|tx| tx.is_aborted)
            .unwrap_or(false)
    }

    /// Mark the current transaction as aborted due to an error.
    /// This should be called when any error occurs during a transaction.
    pub fn abort_transaction(&self) {
        let mut guard = self
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        if let Some(tx) = guard.get_mut(&self.session_id) {
            tx.is_aborted = true;
        }
    }

    /// Begin a transaction in this session.
    pub fn begin_transaction(
        &self,
        staging: Arc<StagingCtx>,
    ) -> LlkvResult<TransactionResult<BaseCtx::Pager>> {
        let mut guard = self
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
            DslTransaction::new(Arc::clone(&self.context), staging),
        );
        Ok(TransactionResult::Transaction {
            kind: TransactionKind::Begin,
        })
    }

    /// Commit the transaction in this session.
    /// If the transaction is aborted, this acts as a ROLLBACK instead.
    pub fn commit_transaction(
        &self,
    ) -> LlkvResult<(TransactionResult<BaseCtx::Pager>, Vec<DslOperation>)> {
        tracing::trace!(
            "[COMMIT] commit_transaction called for session {:?}",
            self.session_id
        );
        let mut guard = self
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        tracing::trace!("[COMMIT] commit_transaction got lock, checking for transaction...");
        let tx_opt = guard.remove(&self.session_id);
        tracing::trace!(
            "[COMMIT] commit_transaction remove returned: {}",
            tx_opt.is_some()
        );
        let tx = tx_opt.ok_or_else(|| {
            tracing::trace!("[COMMIT] commit_transaction: no transaction found!");
            Error::InvalidArgumentError(
                "no transaction is currently in progress in this session".into(),
            )
        })?;
        tracing::trace!("DEBUG commit_transaction: is_aborted={}", tx.is_aborted);

        // If transaction is aborted, commit becomes a rollback (no operations to replay)
        if tx.is_aborted {
            tracing::trace!("DEBUG commit_transaction: returning Rollback with 0 operations");
            return Ok((
                TransactionResult::Transaction {
                    kind: TransactionKind::Rollback,
                },
                Vec::new(),
            ));
        }

        let operations = tx.operations;
        tracing::trace!(
            "DEBUG commit_transaction: returning Commit with {} operations",
            operations.len()
        );

        Ok((
            TransactionResult::Transaction {
                kind: TransactionKind::Commit,
            },
            operations,
        ))
    }

    /// Rollback the transaction in this session.
    pub fn rollback_transaction(&self) -> LlkvResult<TransactionResult<BaseCtx::Pager>> {
        let mut guard = self
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        if guard.remove(&self.session_id).is_none() {
            return Err(Error::InvalidArgumentError(
                "no transaction is currently in progress in this session".into(),
            ));
        }
        Ok(TransactionResult::Transaction {
            kind: TransactionKind::Rollback,
        })
    }

    /// Execute an operation in this session's transaction, or directly if no transaction is active.
    pub fn execute_operation(
        &self,
        operation: DslOperation,
    ) -> LlkvResult<TransactionResult<BaseCtx::Pager>> {
        if !self.has_active_transaction() {
            // No transaction - caller must handle direct execution
            return Err(Error::InvalidArgumentError(
                "execute_operation called without active transaction".into(),
            ));
        }

        // In transaction - add to transaction and execute on staging context
        let mut guard = self
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        let tx = guard
            .get_mut(&self.session_id)
            .ok_or_else(|| Error::Internal("transaction disappeared during execution".into()))?;

        let result = tx.execute_operation(operation);
        if let Err(ref e) = result {
            tracing::trace!("DEBUG TransactionSession::execute_operation error: {:?}", e);
            tracing::trace!("DEBUG Transaction is_aborted={}", tx.is_aborted);
        }
        result
    }
}

impl<BaseCtx, StagingCtx> Drop for TransactionSession<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext,
    StagingCtx: TransactionContext,
{
    fn drop(&mut self) {
        // Auto-rollback on drop if transaction is active
        // Handle poisoned mutex gracefully to avoid panic during cleanup
        match self.transactions.lock() {
            Ok(mut guard) => {
                if guard.remove(&self.session_id).is_some() {
                    eprintln!(
                        "Warning: TransactionSession dropped with active transaction - auto-rolling back"
                    );
                }
            }
            Err(_) => {
                // Mutex is poisoned, likely due to a panic elsewhere
                // Don't panic again during cleanup
                tracing::trace!(
                    "Warning: TransactionSession dropped with poisoned transaction mutex"
                );
            }
        }
    }
}

/// Transaction manager for coordinating sessions
pub struct TransactionManager<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    transactions: Arc<Mutex<HashMap<u64, DslTransaction<BaseCtx, StagingCtx>>>>,
    next_session_id: AtomicU64,
}

impl<BaseCtx, StagingCtx> TransactionManager<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    pub fn new() -> Self {
        Self {
            transactions: Arc::new(Mutex::new(HashMap::new())),
            next_session_id: AtomicU64::new(1),
        }
    }

    /// Create a new session for transaction management.
    pub fn create_session(&self, context: Arc<BaseCtx>) -> TransactionSession<BaseCtx, StagingCtx> {
        let session_id = self.next_session_id.fetch_add(1, Ordering::SeqCst);
        TransactionSession::new(context, session_id, Arc::clone(&self.transactions))
    }

    /// Check if there's an active transaction (checks if ANY session has a transaction).
    pub fn has_active_transaction(&self) -> bool {
        !self
            .transactions
            .lock()
            .expect("transactions lock poisoned")
            .is_empty()
    }
}

impl<BaseCtx, StagingCtx> Default for TransactionManager<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}
