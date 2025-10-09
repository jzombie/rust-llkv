use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

use crate::SqlResult;
use crate::SqlValue;

use arrow::array::{Array, ArrayRef, Int64Array, Int64Builder, UInt32Array};
use arrow::compute::{
    SortColumn, SortOptions, TakeOptions, concat_batches, lexsort_to_indices, take,
};
use arrow::record_batch::RecordBatch;

use llkv_dsl::TransactionKind;
use llkv_dsl::{
    AggregateExpr, AggregateFunction, AssignmentValue, ColumnAssignment, ColumnSpec,
    CreateTablePlan, CreateTableSource, DeletePlan, DslContext, DslValue, InsertPlan, InsertSource,
    OrderByPlan, OrderSortType, OrderTarget, SelectExecution, SelectPlan, SelectProjection,
    StatementResult, UpdatePlan, extract_rows_from_range,
};
use llkv_expr::literal::Literal;
use llkv_result::Error;
use llkv_storage::pager::{MemPager, Pager};
use simd_r_drive_entry_handle::EntryHandle;
use sqlparser::ast::{
    Assignment, AssignmentTarget, BeginTransactionKind, BinaryOperator, ColumnOption,
    ColumnOptionDef, DataType as SqlDataType, Delete, ExceptionWhen, Expr as SqlExpr, FromTable,
    FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Ident, LimitClause, ObjectName,
    ObjectNamePart, OrderBy, OrderByExpr, OrderByKind, Query, Select, SelectItem,
    SelectItemQualifiedWildcardKind, Set, SetExpr, Statement, TableFactor, TableObject,
    TableWithJoins, TransactionMode, TransactionModifier, UnaryOperator, UpdateTableFromKind,
    Value, ValueWithSpan,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

struct TableDeltaState {
    seeded_filters: HashSet<String>,
    exclusion_predicates: Vec<SqlExpr>,
    exclude_all_rows: bool,
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

struct TransactionState {
    staging: Box<SqlEngine<MemPager>>,
    statements: Vec<String>,
    staged_tables: HashSet<String>,
    table_deltas: HashMap<String, TableDeltaState>,
    new_tables: HashSet<String>,
    snapshotted_tables: HashSet<String>,
    missing_tables: HashSet<String>,
    catalog_snapshot: HashSet<String>,
}

pub struct SqlEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<DslContext<P>>,
    default_nulls_first: AtomicBool,
    transaction: Mutex<Option<TransactionState>>,
}

impl<P> Clone for SqlEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            default_nulls_first: AtomicBool::new(
                self.default_nulls_first.load(AtomicOrdering::Relaxed),
            ),
            transaction: Mutex::new(None),
        }
    }
}

impl<P> SqlEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn map_table_error(table_name: &str, err: Error) -> Error {
        match err {
            Error::NotFound => Self::table_not_found_error(table_name),
            Error::InvalidArgumentError(msg) if msg.contains("unknown table") => {
                Self::table_not_found_error(table_name)
            }
            other => other,
        }
    }

    fn table_not_found_error(table_name: &str) -> Error {
        Error::CatalogError(format!(
            "Catalog Error: Table '{table_name}' does not exist"
        ))
    }

    fn is_table_missing_error(err: &Error) -> bool {
        match err {
            Error::NotFound => true,
            Error::CatalogError(msg) => {
                msg.contains("Catalog Error: Table") || msg.contains("unknown table")
            }
            Error::InvalidArgumentError(msg) => {
                msg.contains("Catalog Error: Table") || msg.contains("unknown table")
            }
            _ => false,
        }
    }

    pub fn new(pager: Arc<P>) -> Self {
        Self {
            context: Arc::new(DslContext::new(pager)),
            default_nulls_first: AtomicBool::new(false),
            transaction: Mutex::new(None),
        }
    }

    pub(crate) fn context_arc(&self) -> Arc<DslContext<P>> {
        Arc::clone(&self.context)
    }

    pub fn with_context(context: Arc<DslContext<P>>, default_nulls_first: bool) -> Self {
        Self {
            context,
            default_nulls_first: AtomicBool::new(default_nulls_first),
            transaction: Mutex::new(None),
        }
    }

    #[cfg(test)]
    fn default_nulls_first_for_tests(&self) -> bool {
        self.default_nulls_first.load(AtomicOrdering::Relaxed)
    }

    fn has_active_transaction(&self) -> bool {
        self.transaction
            .lock()
            .expect("transaction lock poisoned")
            .is_some()
    }

    pub fn execute(&self, sql: &str) -> SqlResult<Vec<StatementResult<P>>> {
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(&dialect, sql)
            .map_err(|err| Error::InvalidArgumentError(format!("failed to parse SQL: {err}")))?;

        let mut results = Vec::with_capacity(statements.len());
        for statement in statements {
            results.push(self.execute_statement(statement)?);
        }
        Ok(results)
    }

    fn execute_statement(&self, statement: Statement) -> SqlResult<StatementResult<P>> {
        match statement {
            Statement::StartTransaction {
                modes,
                begin,
                transaction,
                modifier,
                statements,
                exception,
                has_end_keyword,
            } => self.handle_start_transaction(
                modes,
                begin,
                transaction,
                modifier,
                statements,
                exception,
                has_end_keyword,
            ),
            Statement::Commit {
                chain,
                end,
                modifier,
            } => self.handle_commit(chain, end, modifier),
            Statement::Rollback { chain, savepoint } => self.handle_rollback(chain, savepoint),
            other => {
                if self.has_active_transaction() {
                    self.execute_statement_in_transaction(other)
                } else {
                    self.execute_statement_non_transactional(other)
                }
            }
        }
    }

    fn execute_statement_non_transactional(
        &self,
        statement: Statement,
    ) -> SqlResult<StatementResult<P>> {
        match statement {
            Statement::CreateTable(stmt) => self.handle_create_table(stmt),
            Statement::Insert(stmt) => self.handle_insert(stmt),
            Statement::Query(query) => self.handle_query(*query),
            Statement::Update {
                table,
                assignments,
                from,
                selection,
                returning,
                ..
            } => self.handle_update(table, assignments, from, selection, returning),
            Statement::Delete(delete) => self.handle_delete(delete),
            Statement::Set(set_stmt) => self.handle_set(set_stmt),
            Statement::Pragma { name, value, is_eq } => self.handle_pragma(name, value, is_eq),
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported SQL statement: {other:?}"
            ))),
        }
    }

    fn execute_statement_in_transaction(
        &self,
        statement: Statement,
    ) -> SqlResult<StatementResult<P>> {
        let sql_text = statement.to_string();
        let mut guard = self.transaction.lock().expect("transaction lock poisoned");
        let tx = guard.as_mut().ok_or_else(|| {
            Error::InvalidArgumentError("no transaction is currently in progress".into())
        })?;

        match &statement {
            Statement::Set(set_stmt) => {
                let result = self.handle_set(set_stmt.clone())?;
                tx.staging.execute(&sql_text)?;
                return Ok(result);
            }
            Statement::Pragma { name, value, is_eq } => {
                let result = self.handle_pragma(name.clone(), value.clone(), *is_eq)?;
                tx.staging.execute(&sql_text)?;
                return Ok(result);
            }
            _ => {}
        }

        match statement {
            Statement::Query(query) => {
                let tables = Self::tables_in_query(&query)?;
                for table in &tables {
                    self.ensure_table_in_delta(tx, table)?;
                }
                let all_tables_local = tables.iter().all(|table| {
                    tx.new_tables.contains(table) || tx.snapshotted_tables.contains(table)
                });

                let aggregate_specs = if let SetExpr::Select(select) = query.body.as_ref() {
                    self.detect_simple_aggregates(&select.projection)?
                } else {
                    None
                };

                let mut base_query = *query.clone();
                let order_by_exprs: Vec<OrderByExpr> = base_query
                    .order_by
                    .as_ref()
                    .and_then(|order| match &order.kind {
                        OrderByKind::Expressions(exprs) => Some(exprs.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let skip_base = Self::apply_transaction_filters(tx, &mut base_query)?;
                let base_result = if skip_base || all_tables_local {
                    None
                } else {
                    match self.handle_query(base_query) {
                        Ok(result) => Some(result),
                        Err(err) if Self::is_table_missing_error(&err) => None,
                        Err(err) => return Err(err),
                    }
                };
                let mut delta_results = tx.staging.execute(&sql_text)?;
                if delta_results.len() != 1 {
                    return Err(Error::Internal(
                        "transaction statements must yield exactly one result".into(),
                    ));
                }
                let delta_raw = delta_results.remove(0);
                let delta_converted = Self::convert_statement_result(delta_raw)?;
                let default_nulls_first = self.default_nulls_first.load(AtomicOrdering::Relaxed);
                Self::merge_select_results(
                    base_result,
                    delta_converted,
                    aggregate_specs,
                    &order_by_exprs,
                    default_nulls_first,
                )
            }
            Statement::CreateTable(stmt) => {
                let table_name = Self::object_name_to_string(&stmt.name)?;
                let mut delta_results = match tx.staging.execute(&sql_text) {
                    Ok(results) => results,
                    Err(err) => {
                        println!("delta create table error: {err:?}");
                        return Err(err);
                    }
                };
                if delta_results.len() != 1 {
                    return Err(Error::Internal(
                        "transaction statements must yield exactly one result".into(),
                    ));
                }
                let delta_raw = delta_results.remove(0);
                tx.new_tables.insert(table_name.clone());
                tx.missing_tables.remove(table_name.as_str());
                tx.statements.push(sql_text);
                Self::convert_statement_result(delta_raw)
            }
            Statement::Insert(insert) => {
                let table_name = Self::table_name_from_insert(&insert)?;
                self.ensure_table_in_delta(tx, &table_name)?;
                let mut delta_results = match tx.staging.execute(&sql_text) {
                    Ok(results) => results,
                    Err(err) => {
                        println!("delta insert error: {err:?}");
                        return Err(err);
                    }
                };
                if delta_results.len() != 1 {
                    return Err(Error::Internal(
                        "transaction statements must yield exactly one result".into(),
                    ));
                }
                let delta_raw = delta_results.remove(0);
                tx.statements.push(sql_text);
                Self::convert_statement_result(delta_raw)
            }
            Statement::Update {
                table, selection, ..
            } => {
                if let Some(name) = Self::table_name_from_update(&table)? {
                    self.ensure_table_in_delta(tx, &name)?;
                    self.seed_rows_for_update(tx, &name, selection.as_ref())?;
                }
                let mut delta_results = match tx.staging.execute(&sql_text) {
                    Ok(results) => results,
                    Err(err) => {
                        println!("delta update error: {err:?}");
                        return Err(err);
                    }
                };
                if delta_results.len() != 1 {
                    return Err(Error::Internal(
                        "transaction statements must yield exactly one result".into(),
                    ));
                }
                let delta_raw = delta_results.remove(0);
                if let Some(name) = Self::table_name_from_update(&table)? {
                    Self::record_update_exclusion(tx, &name, selection.as_ref());
                }
                tx.statements.push(sql_text);
                Self::convert_statement_result(delta_raw)
            }
            Statement::Delete(delete) => {
                let selection_ref = delete.selection.as_ref();
                let table_name = Self::table_name_from_delete(&delete)?;
                if let Some(name) = table_name.as_ref() {
                    self.ensure_table_in_delta(tx, name)?;
                    self.seed_rows_for_update(tx, name, selection_ref)?;
                }
                let mut delta_results = match tx.staging.execute(&sql_text) {
                    Ok(results) => results,
                    Err(err) => {
                        println!("delta delete error: {err:?}");
                        return Err(err);
                    }
                };
                if delta_results.len() != 1 {
                    return Err(Error::Internal(
                        "transaction statements must yield exactly one result".into(),
                    ));
                }
                let delta_raw = delta_results.remove(0);
                if let Some(name) = table_name.as_ref() {
                    Self::record_update_exclusion(tx, name, selection_ref);
                }
                tx.statements.push(sql_text);
                Self::convert_statement_result(delta_raw)
            }
            other => {
                let mut delta_results = match tx.staging.execute(&sql_text) {
                    Ok(results) => results,
                    Err(err) => {
                        println!("delta misc error: {err:?}");
                        return Err(err);
                    }
                };
                if delta_results.len() != 1 {
                    return Err(Error::Internal(
                        "transaction statements must yield exactly one result".into(),
                    ));
                }
                let delta_raw = delta_results.remove(0);
                if Self::is_mutating_statement(&other) {
                    tx.statements.push(sql_text);
                }
                Self::convert_statement_result(delta_raw)
            }
        }
    }

    fn ensure_table_in_delta(&self, tx: &mut TransactionState, table_name: &str) -> SqlResult<()> {
        if tx.staged_tables.contains(table_name) {
            return Ok(());
        }

        let canonical_name = table_name.to_ascii_lowercase();
        if !tx.catalog_snapshot.contains(&canonical_name) && !tx.new_tables.contains(table_name) {
            tx.missing_tables.insert(table_name.to_string());
            return Err(Self::table_not_found_error(table_name));
        }

        if tx.missing_tables.contains(table_name) {
            return Err(Self::table_not_found_error(table_name));
        }

        match self.context.table_column_specs(table_name) {
            Ok(specs) => {
                tx.missing_tables.remove(table_name);
                let mut plan = CreateTablePlan::new(table_name.to_string());
                plan.if_not_exists = true;
                plan.columns = specs;
                tx.staging.context_arc().create_table_plan(plan)?;

                match self.context.export_table_rows(table_name) {
                    Ok(snapshot) => {
                        if !snapshot.rows.is_empty() {
                            let insert_plan = InsertPlan {
                                table: table_name.to_string(),
                                columns: snapshot.columns.clone(),
                                source: InsertSource::Rows(snapshot.rows),
                            };
                            tx.staging.context_arc().insert(insert_plan)?;
                        }
                        tx.snapshotted_tables.insert(table_name.to_string());
                    }
                    Err(Error::NotFound) => {
                        tx.snapshotted_tables.insert(table_name.to_string());
                    }
                    Err(other) => return Err(other),
                }
            }
            Err(Error::NotFound) => match tx.staging.context_arc().table_column_specs(table_name) {
                Ok(_) => {
                    tx.staged_tables.insert(table_name.to_string());
                    return Ok(());
                }
                Err(Error::NotFound) => {
                    tx.missing_tables.insert(table_name.to_string());
                    return Err(Self::table_not_found_error(table_name));
                }
                Err(Error::InvalidArgumentError(msg)) if msg.contains("unknown table") => {
                    tx.missing_tables.insert(table_name.to_string());
                    return Err(Self::table_not_found_error(table_name));
                }
                Err(other) => {
                    return Err(other);
                }
            },
            Err(Error::InvalidArgumentError(msg)) if msg.contains("unknown table") => {
                match tx.staging.context_arc().table_column_specs(table_name) {
                    Ok(_) => {
                        tx.staged_tables.insert(table_name.to_string());
                        return Ok(());
                    }
                    Err(Error::NotFound) => {
                        tx.missing_tables.insert(table_name.to_string());
                        return Err(Self::table_not_found_error(table_name));
                    }
                    Err(Error::InvalidArgumentError(inner)) if inner.contains("unknown table") => {
                        tx.missing_tables.insert(table_name.to_string());
                        return Err(Self::table_not_found_error(table_name));
                    }
                    Err(other) => {
                        return Err(other);
                    }
                }
            }
            Err(other) => {
                return Err(other);
            }
        }

        tx.staged_tables.insert(table_name.to_string());
        Ok(())
    }

    fn seed_rows_for_update(
        &self,
        tx: &mut TransactionState,
        table_name: &str,
        selection: Option<&SqlExpr>,
    ) -> SqlResult<()> {
        let filter_key = selection
            .map(|expr| expr.to_string())
            .unwrap_or_else(|| "__all__".to_string());

        if tx
            .table_deltas
            .get(table_name)
            .map(|delta| delta.seeded_filters.contains(&filter_key))
            .unwrap_or(false)
        {
            return Ok(());
        }

        self.ensure_table_in_delta(tx, table_name)?;

        // Convert SQL WHERE expression to LlkvExpr if present
        let filter = selection
            .map(|expr| translate_condition(expr))
            .transpose()?;

        // Get batches from the base table with row_ids preserved
        let batches = match self.context.get_batches_with_row_ids(table_name, filter) {
            Ok(batches) => batches,
            Err(err) if Self::is_table_missing_error(&err) => {
                return Ok(());
            }
            Err(other) => return Err(other),
        };

        {
            let delta = tx
                .table_deltas
                .entry(table_name.to_string())
                .or_insert_with(TableDeltaState::new);
            delta.seeded_filters.insert(filter_key.clone());
        }

        if batches.is_empty() || batches.iter().all(|b| b.num_rows() == 0) {
            return Ok(());
        }

        // Use append_batches_with_row_ids to preserve row_ids when seeding
        tx.staging
            .context_arc()
            .append_batches_with_row_ids(table_name, batches)
            .map(|_| ())
    }

    fn record_update_exclusion(
        tx: &mut TransactionState,
        table_name: &str,
        selection: Option<&SqlExpr>,
    ) {
        let delta = tx
            .table_deltas
            .entry(table_name.to_string())
            .or_insert_with(TableDeltaState::new);
        match selection {
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

    fn apply_transaction_filters(tx: &TransactionState, query: &mut Query) -> SqlResult<bool> {
        let set_expr = query.body.as_mut();
        let SetExpr::Select(select) = set_expr else {
            return Ok(false);
        };

        if select.from.is_empty() {
            return Ok(false);
        }

        let Ok((table_name, _)) = extract_single_table(&select.from) else {
            return Ok(false);
        };
        let Some(delta) = tx.table_deltas.get(&table_name) else {
            return Ok(false);
        };

        if delta.exclude_all_rows {
            return Ok(true);
        }

        if delta.exclusion_predicates.is_empty() {
            return Ok(false);
        }

        let mut combined: Option<SqlExpr> = None;
        for predicate in &delta.exclusion_predicates {
            let not_expr = SqlExpr::UnaryOp {
                op: UnaryOperator::Not,
                expr: Box::new(predicate.clone()),
            };
            combined = Some(match combined {
                Some(existing) => SqlExpr::BinaryOp {
                    left: Box::new(existing),
                    op: BinaryOperator::And,
                    right: Box::new(not_expr),
                },
                None => not_expr,
            });
        }

        if let Some(extra_filter) = combined {
            let merged = Self::combine_with_and(select.selection.take(), extra_filter);
            select.selection = Some(merged);
        }
        Ok(false)
    }

    fn combine_with_and(existing: Option<SqlExpr>, new_expr: SqlExpr) -> SqlExpr {
        match existing {
            Some(expr) => SqlExpr::BinaryOp {
                left: Box::new(expr),
                op: BinaryOperator::And,
                right: Box::new(new_expr),
            },
            None => new_expr,
        }
    }

    fn is_mutating_statement(statement: &Statement) -> bool {
        matches!(
            statement,
            Statement::CreateTable(_)
                | Statement::Insert(_)
                | Statement::Update { .. }
                | Statement::Delete(_)
        )
    }

    fn convert_statement_result(
        result: StatementResult<MemPager>,
    ) -> SqlResult<StatementResult<P>> {
        match result {
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
            StatementResult::Select {
                table_name,
                schema,
                execution,
            } => {
                let mut batches = match execution.collect() {
                    Ok(batches) => batches,
                    Err(Error::NotFound) => Vec::new(),
                    Err(other) => return Err(other),
                };
                let combined = if batches.is_empty() {
                    RecordBatch::new_empty(schema.clone())
                } else if batches.len() == 1 {
                    batches.remove(0)
                } else {
                    let refs: Vec<&RecordBatch> = batches.iter().collect();
                    concat_batches(&schema, refs).map_err(|err| {
                        Error::Internal(format!(
                            "failed to concatenate select batches in transaction: {err}"
                        ))
                    })?
                };
                let execution =
                    SelectExecution::from_batch(table_name.clone(), schema.clone(), combined);
                Ok(StatementResult::Select {
                    table_name,
                    schema,
                    execution,
                })
            }
            StatementResult::Transaction { kind } => Ok(StatementResult::Transaction { kind }),
        }
    }

    fn merge_select_results(
        base: Option<StatementResult<P>>,
        delta: StatementResult<P>,
        aggregates: Option<Vec<AggregateExpr>>,
        order_by: &[OrderByExpr],
        default_nulls_first: bool,
    ) -> SqlResult<StatementResult<P>> {
        let (delta_table_name, delta_schema, delta_execution) = match delta {
            StatementResult::Select {
                table_name,
                schema,
                execution,
            } => (table_name, schema, execution),
            other => {
                return Err(Error::Internal(format!(
                    "expected SELECT result from transaction delta, found {other:?}"
                )));
            }
        };

        let (base_table_name_opt, base_schema_opt, base_batches) = match base {
            Some(StatementResult::Select {
                table_name,
                schema,
                execution,
            }) => {
                let batches = match execution.collect() {
                    Ok(batches) => batches,
                    Err(Error::NotFound) => Vec::new(),
                    Err(other) => return Err(other),
                };
                (Some(table_name), Some(schema), batches)
            }
            Some(other) => {
                return Err(Error::Internal(format!(
                    "expected SELECT result from base context, found {other:?}"
                )));
            }
            None => (None, None, Vec::new()),
        };

        let delta_batches = match delta_execution.collect() {
            Ok(batches) => batches,
            Err(Error::NotFound) => Vec::new(),
            Err(other) => return Err(other),
        };

        if let Some(base_schema) = &base_schema_opt {
            if base_schema.fields().len() != delta_schema.fields().len() {
                return Err(Error::Internal(
                    "mismatched schemas when merging transaction select results".into(),
                ));
            }
        }

        if let Some(base_table_name) = &base_table_name_opt {
            if base_table_name != &delta_table_name {
                return Err(Error::Internal(
                    "mismatched table names when merging transaction select results".into(),
                ));
            }
        }

        if let Some(specs) = aggregates.clone() {
            if !specs.is_empty() {
                let schema = base_schema_opt
                    .clone()
                    .unwrap_or_else(|| Arc::clone(&delta_schema));
                let mut columns: Vec<ArrayRef> = Vec::with_capacity(specs.len());

                for (idx, spec) in specs.iter().enumerate() {
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
                let execution = SelectExecution::from_batch(
                    delta_table_name.clone(),
                    Arc::clone(&schema),
                    batch,
                );
                return Ok(StatementResult::Select {
                    table_name: delta_table_name,
                    schema,
                    execution,
                });
            }
        }

        let mut all_batches: Vec<RecordBatch> = Vec::new();
        all_batches.extend(base_batches);
        all_batches.extend(delta_batches);

        let schema = base_schema_opt
            .clone()
            .unwrap_or_else(|| Arc::clone(&delta_schema));
        let mut combined = if all_batches.is_empty() {
            RecordBatch::new_empty(Arc::clone(&schema))
        } else if all_batches.len() == 1 {
            all_batches.remove(0)
        } else {
            let refs: Vec<&RecordBatch> = all_batches.iter().collect();
            concat_batches(&schema, refs).map_err(|err| {
                Error::Internal(format!("failed to merge transaction select batches: {err}"))
            })?
        };

        if !order_by.is_empty() && combined.num_rows() > 1 {
            combined = Self::resort_record_batch(combined, order_by, default_nulls_first)?;
        }

        let execution =
            SelectExecution::from_batch(delta_table_name.clone(), Arc::clone(&schema), combined);
        Ok(StatementResult::Select {
            table_name: delta_table_name,
            schema,
            execution,
        })
    }

    fn extract_aggregate_value(
        batches: &[RecordBatch],
        column_idx: usize,
    ) -> SqlResult<Option<i64>> {
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
            if array.len() == 0 {
                continue;
            }
            if array.is_null(0) {
                return Ok(None);
            }
            return Ok(Some(array.value(0)));
        }
        Ok(None)
    }

    fn resort_record_batch(
        batch: RecordBatch,
        order_by: &[OrderByExpr],
        default_nulls_first: bool,
    ) -> SqlResult<RecordBatch> {
        let schema = batch.schema();
        let mut sort_columns: Vec<SortColumn> = Vec::with_capacity(order_by.len());

        for order_expr in order_by {
            let SqlExpr::Identifier(ident) = &order_expr.expr else {
                // Fall back to the existing order if we cannot interpret the expression.
                return Ok(batch);
            };

            let (index, _) = schema.column_with_name(&ident.value).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "ORDER BY column '{}' not found in projection",
                    ident.value
                ))
            })?;

            let descending = order_expr.options.asc.map(|asc| !asc).unwrap_or(false);
            let nulls_first = order_expr
                .options
                .nulls_first
                .unwrap_or(default_nulls_first);

            sort_columns.push(SortColumn {
                values: batch.column(index).clone(),
                options: Some(SortOptions {
                    descending,
                    nulls_first,
                }),
            });
        }

        if sort_columns.is_empty() {
            return Ok(batch);
        }

        let indices: UInt32Array = lexsort_to_indices(&sort_columns, None)
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

        RecordBatch::try_new(schema, sorted_columns).map_err(|err| {
            Error::Internal(format!("failed to build sorted transaction batch: {err}"))
        })
    }

    fn table_name_from_insert(insert: &sqlparser::ast::Insert) -> SqlResult<String> {
        match &insert.table {
            TableObject::TableName(name) => Self::object_name_to_string(name),
            _ => Err(Error::InvalidArgumentError(
                "INSERT requires a plain table name".into(),
            )),
        }
    }

    fn table_name_from_update(table: &TableWithJoins) -> SqlResult<Option<String>> {
        if !table.joins.is_empty() {
            return Err(Error::InvalidArgumentError(
                "UPDATE with JOIN targets is not supported yet".into(),
            ));
        }
        Self::table_with_joins_name(table)
    }

    fn table_name_from_delete(delete: &Delete) -> SqlResult<Option<String>> {
        if !delete.tables.is_empty() {
            return Err(Error::InvalidArgumentError(
                "multi-table DELETE is not supported yet".into(),
            ));
        }
        let from_tables = match &delete.from {
            FromTable::WithFromKeyword(tables) | FromTable::WithoutKeyword(tables) => tables,
        };
        if from_tables.is_empty() {
            return Ok(None);
        }
        if from_tables.len() != 1 {
            return Err(Error::InvalidArgumentError(
                "DELETE over multiple tables is not supported yet".into(),
            ));
        }
        Self::table_with_joins_name(&from_tables[0])
    }

    fn object_name_to_string(name: &ObjectName) -> SqlResult<String> {
        let (display, _) = canonical_object_name(name)?;
        Ok(display)
    }

    fn table_object_to_name(table: &TableObject) -> SqlResult<Option<String>> {
        match table {
            TableObject::TableName(name) => Ok(Some(Self::object_name_to_string(name)?)),
            TableObject::TableFunction(_) => Ok(None),
        }
    }

    fn table_with_joins_name(table: &TableWithJoins) -> SqlResult<Option<String>> {
        match &table.relation {
            TableFactor::Table { name, .. } => Ok(Some(Self::object_name_to_string(name)?)),
            _ => Ok(None),
        }
    }

    fn tables_in_query(query: &Query) -> SqlResult<Vec<String>> {
        let mut tables = Vec::new();
        if let sqlparser::ast::SetExpr::Select(select) = query.body.as_ref() {
            for table in &select.from {
                if let TableFactor::Table { name, .. } = &table.relation {
                    tables.push(Self::object_name_to_string(name)?);
                }
            }
        }
        Ok(tables)
    }

    fn handle_create_table(
        &self,
        mut stmt: sqlparser::ast::CreateTable,
    ) -> SqlResult<StatementResult<P>> {
        validate_create_table_common(&stmt)?;

        let (display_name, canonical_name) = canonical_object_name(&stmt.name)?;
        if display_name.is_empty() {
            return Err(Error::InvalidArgumentError(
                "table name must not be empty".into(),
            ));
        }

        if let Some(query) = stmt.query.take() {
            validate_create_table_as(&stmt)?;
            return self.handle_create_table_as(
                display_name,
                canonical_name,
                *query,
                stmt.if_not_exists,
            );
        }

        if stmt.columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires at least one column".into(),
            ));
        }

        validate_create_table_definition(&stmt)?;

        let mut columns: Vec<ColumnSpec> = Vec::with_capacity(stmt.columns.len());
        let mut names: HashMap<String, ()> = HashMap::new();
        for column_def in stmt.columns {
            let column = ColumnSpec::new(
                column_def.name.value.clone(),
                arrow_type_from_sql(&column_def.data_type)?,
                column_def
                    .options
                    .iter()
                    .all(|opt| !matches!(opt.option, ColumnOption::NotNull)),
            );
            let normalized = column.name.to_ascii_lowercase();
            if names.insert(normalized, ()).is_some() {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in table '{}'",
                    column.name, display_name
                )));
            }
            columns.push(column);
        }

        let plan = CreateTablePlan {
            name: display_name,
            if_not_exists: stmt.if_not_exists,
            columns,
            source: None,
        };
        self.context.create_table_plan(plan)
    }

    fn handle_create_table_as(
        &self,
        display_name: String,
        _canonical_name: String,
        query: Query,
        if_not_exists: bool,
    ) -> SqlResult<StatementResult<P>> {
        let execution = self.execute_query_collect(query)?;
        let schema = execution.schema();
        let batches = execution.collect()?;

        if schema.fields().is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE AS SELECT requires at least one projected column".into(),
            ));
        }

        let plan = CreateTablePlan {
            name: display_name,
            if_not_exists,
            columns: Vec::new(),
            source: Some(CreateTableSource::Batches { schema, batches }),
        };
        self.context.create_table_plan(plan)
    }

    fn handle_insert(&self, stmt: sqlparser::ast::Insert) -> SqlResult<StatementResult<P>> {
        if stmt.replace_into || stmt.ignore || stmt.or.is_some() {
            return Err(Error::InvalidArgumentError(
                "non-standard INSERT forms are not supported".into(),
            ));
        }
        if stmt.overwrite {
            return Err(Error::InvalidArgumentError(
                "INSERT OVERWRITE is not supported".into(),
            ));
        }
        if !stmt.assignments.is_empty() {
            return Err(Error::InvalidArgumentError(
                "INSERT ... SET is not supported".into(),
            ));
        }
        if stmt.partitioned.is_some() || !stmt.after_columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "partitioned INSERT is not supported".into(),
            ));
        }
        if stmt.returning.is_some() {
            return Err(Error::InvalidArgumentError(
                "INSERT ... RETURNING is not supported".into(),
            ));
        }
        if stmt.format_clause.is_some() || stmt.settings.is_some() {
            return Err(Error::InvalidArgumentError(
                "INSERT with FORMAT or SETTINGS is not supported".into(),
            ));
        }

        let (display_name, _canonical_name) = match &stmt.table {
            TableObject::TableName(name) => canonical_object_name(name)?,
            _ => {
                return Err(Error::InvalidArgumentError(
                    "INSERT requires a plain table name".into(),
                ));
            }
        };

        let columns: Vec<String> = stmt
            .columns
            .iter()
            .map(|ident| ident.value.clone())
            .collect();
        let source_expr = stmt
            .source
            .as_ref()
            .ok_or_else(|| Error::InvalidArgumentError("INSERT requires a VALUES clause".into()))?;
        validate_simple_query(source_expr)?;

        let insert_source = match source_expr.body.as_ref() {
            SetExpr::Values(values) => {
                if values.rows.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "INSERT VALUES list must contain at least one row".into(),
                    ));
                }
                let mut rows: Vec<Vec<SqlValue>> = Vec::with_capacity(values.rows.len());
                for row in &values.rows {
                    let mut converted = Vec::with_capacity(row.len());
                    for expr in row {
                        converted.push(SqlValue::try_from_expr(expr)?);
                    }
                    rows.push(converted);
                }
                InsertSource::Rows(
                    rows.into_iter()
                        .map(|row| row.into_iter().map(DslValue::from).collect())
                        .collect(),
                )
            }
            SetExpr::Select(select) => {
                if let Some(rows) = extract_constant_select_rows(select.as_ref())? {
                    InsertSource::Rows(rows)
                } else if let Some(range_rows) = extract_rows_from_range(select.as_ref())? {
                    InsertSource::Rows(range_rows.into_rows())
                } else {
                    let execution = self.execute_query_collect((**source_expr).clone())?;
                    let rows = execution.into_rows()?;
                    InsertSource::Rows(rows)
                }
            }
            _ => {
                return Err(Error::InvalidArgumentError(
                    "unsupported INSERT source".into(),
                ));
            }
        };

        let plan = InsertPlan {
            table: display_name.clone(),
            columns,
            source: insert_source,
        };
        self.context
            .insert(plan)
            .map_err(|err| Self::map_table_error(&display_name, err))
    }

    fn handle_update(
        &self,
        table: TableWithJoins,
        assignments: Vec<Assignment>,
        from: Option<UpdateTableFromKind>,
        selection: Option<SqlExpr>,
        returning: Option<Vec<SelectItem>>,
    ) -> SqlResult<StatementResult<P>> {
        if from.is_some() {
            return Err(Error::InvalidArgumentError(
                "UPDATE ... FROM is not supported yet".into(),
            ));
        }
        if returning.is_some() {
            return Err(Error::InvalidArgumentError(
                "UPDATE ... RETURNING is not supported".into(),
            ));
        }
        if assignments.is_empty() {
            return Err(Error::InvalidArgumentError(
                "UPDATE requires at least one assignment".into(),
            ));
        }

        let (display_name, _) = extract_single_table(std::slice::from_ref(&table))?;

        let mut column_assignments = Vec::with_capacity(assignments.len());
        let mut seen: HashMap<String, ()> = HashMap::new();
        for assignment in assignments {
            let column_name = resolve_assignment_column_name(&assignment.target)?;
            let normalized = column_name.to_ascii_lowercase();
            if seen.insert(normalized, ()).is_some() {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in UPDATE assignments",
                    column_name
                )));
            }
            let value = match SqlValue::try_from_expr(&assignment.value) {
                Ok(literal) => AssignmentValue::Literal(DslValue::from(literal)),
                Err(Error::InvalidArgumentError(msg))
                    if msg.contains("unsupported literal expression") =>
                {
                    let translated = translate_scalar(&assignment.value)?;
                    AssignmentValue::Expression(translated)
                }
                Err(err) => return Err(err),
            };
            column_assignments.push(ColumnAssignment {
                column: column_name,
                value,
            });
        }

        let filter = match selection {
            Some(expr) => Some(translate_condition(&expr)?),
            None => None,
        };

        let plan = UpdatePlan {
            table: display_name.clone(),
            assignments: column_assignments,
            filter,
        };
        self.context
            .update(plan)
            .map_err(|err| Self::map_table_error(&display_name, err))
    }

    fn handle_delete(&self, delete: Delete) -> SqlResult<StatementResult<P>> {
        let Delete {
            tables,
            from,
            using,
            selection,
            returning,
            order_by,
            limit,
        } = delete;

        if !tables.is_empty() {
            return Err(Error::InvalidArgumentError(
                "multi-table DELETE is not supported yet".into(),
            ));
        }
        if let Some(using_tables) = using {
            if !using_tables.is_empty() {
                return Err(Error::InvalidArgumentError(
                    "DELETE ... USING is not supported yet".into(),
                ));
            }
        }
        if returning.is_some() {
            return Err(Error::InvalidArgumentError(
                "DELETE ... RETURNING is not supported".into(),
            ));
        }
        if !order_by.is_empty() {
            return Err(Error::InvalidArgumentError(
                "DELETE ... ORDER BY is not supported yet".into(),
            ));
        }
        if limit.is_some() {
            return Err(Error::InvalidArgumentError(
                "DELETE ... LIMIT is not supported yet".into(),
            ));
        }

        let from_tables = match from {
            FromTable::WithFromKeyword(tables) | FromTable::WithoutKeyword(tables) => tables,
        };
        let (display_name, _) = extract_single_table(&from_tables)?;

        let filter = selection
            .map(|expr| translate_condition(&expr))
            .transpose()?;

        let plan = DeletePlan {
            table: display_name.clone(),
            filter,
        };
        self.context
            .delete(plan)
            .map_err(|err| Self::map_table_error(&display_name, err))
    }

    fn handle_query(&self, query: Query) -> SqlResult<StatementResult<P>> {
        let execution = self.execute_query_collect(query)?;
        let table_name = execution.table_name().to_string();
        let schema = execution.schema();
        Ok(StatementResult::Select {
            table_name,
            schema,
            execution,
        })
    }

    fn execute_query_collect(&self, query: Query) -> SqlResult<SelectExecution<P>> {
        validate_simple_query(&query)?;
        let mut select_plan = match query.body.as_ref() {
            SetExpr::Select(select) => self.translate_select(select.as_ref())?,
            other => {
                return Err(Error::InvalidArgumentError(format!(
                    "unsupported query expression: {other:?}"
                )));
            }
        };
        if let Some(order_by) = &query.order_by {
            if !select_plan.aggregates.is_empty() {
                return Err(Error::InvalidArgumentError(
                    "ORDER BY is not supported for aggregate queries".into(),
                ));
            }
            let order_plan = self.translate_order_by(order_by)?;
            select_plan = select_plan.with_order_by(Some(order_plan));
        }
        let table_name = select_plan.table.clone();
        self.context
            .execute_select(select_plan)
            .map_err(|err| Self::map_table_error(&table_name, err))
    }

    fn translate_select(&self, select: &Select) -> SqlResult<SelectPlan> {
        if select.distinct.is_some() {
            return Err(Error::InvalidArgumentError(
                "SELECT DISTINCT is not supported".into(),
            ));
        }
        if select.top.is_some() {
            return Err(Error::InvalidArgumentError(
                "SELECT TOP is not supported".into(),
            ));
        }
        if select.exclude.is_some() {
            return Err(Error::InvalidArgumentError(
                "SELECT EXCLUDE is not supported".into(),
            ));
        }
        if select.into.is_some() {
            return Err(Error::InvalidArgumentError(
                "SELECT INTO is not supported".into(),
            ));
        }
        if !select.lateral_views.is_empty() {
            return Err(Error::InvalidArgumentError(
                "LATERAL VIEW is not supported".into(),
            ));
        }
        if select.prewhere.is_some() {
            return Err(Error::InvalidArgumentError(
                "PREWHERE is not supported".into(),
            ));
        }
        if !group_by_is_empty(&select.group_by) || select.value_table_mode.is_some() {
            return Err(Error::InvalidArgumentError(
                "GROUP BY and SELECT AS VALUE/STRUCT are not supported".into(),
            ));
        }
        if !select.cluster_by.is_empty()
            || !select.distribute_by.is_empty()
            || !select.sort_by.is_empty()
        {
            return Err(Error::InvalidArgumentError(
                "CLUSTER/DISTRIBUTE/SORT BY clauses are not supported".into(),
            ));
        }
        if select.having.is_some()
            || !select.named_window.is_empty()
            || select.qualify.is_some()
            || select.connect_by.is_some()
        {
            return Err(Error::InvalidArgumentError(
                "advanced SELECT clauses are not supported".into(),
            ));
        }

        let (display_name, _canonical_name) = extract_single_table(&select.from)?;
        let mut plan = SelectPlan::new(display_name);

        if let Some(aggregates) = self.detect_simple_aggregates(&select.projection)? {
            plan = plan.with_aggregates(aggregates);
        } else {
            let projections = self.build_projection_list(&select.projection)?;
            plan = plan.with_projections(projections);
        }

        let filter_expr = match &select.selection {
            Some(expr) => Some(translate_condition(expr)?),
            None => None,
        };
        plan = plan.with_filter(filter_expr);
        Ok(plan)
    }

    fn translate_order_by(&self, order_by: &OrderBy) -> SqlResult<OrderByPlan> {
        let exprs = match &order_by.kind {
            OrderByKind::Expressions(exprs) => exprs,
            _ => {
                return Err(Error::InvalidArgumentError(
                    "unsupported ORDER BY clause".into(),
                ));
            }
        };

        if exprs.len() != 1 {
            return Err(Error::InvalidArgumentError(
                "ORDER BY currently supports a single expression".into(),
            ));
        }

        let order_expr: &OrderByExpr = &exprs[0];
        let ascending = order_expr.options.asc.unwrap_or(true);
        let base_nulls_first = self.default_nulls_first.load(AtomicOrdering::Relaxed);
        let default_nulls_first_for_direction = if ascending {
            base_nulls_first
        } else {
            !base_nulls_first
        };
        let nulls_first = order_expr
            .options
            .nulls_first
            .unwrap_or(default_nulls_first_for_direction);

        let (target, sort_type) = match &order_expr.expr {
            SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => (
                OrderTarget::Column(resolve_column_name(&order_expr.expr)?),
                OrderSortType::Native,
            ),
            SqlExpr::Cast {
                expr,
                data_type:
                    SqlDataType::Int(_)
                    | SqlDataType::Integer(_)
                    | SqlDataType::BigInt(_)
                    | SqlDataType::SmallInt(_)
                    | SqlDataType::TinyInt(_),
                ..
            } => (
                OrderTarget::Column(resolve_column_name(expr)?),
                OrderSortType::CastTextToInteger,
            ),
            SqlExpr::Cast { data_type, .. } => {
                return Err(Error::InvalidArgumentError(format!(
                    "ORDER BY CAST target type {:?} is not supported",
                    data_type
                )));
            }
            SqlExpr::Value(value_with_span) => match &value_with_span.value {
                Value::Number(raw, _) => {
                    let position: usize = raw.parse().map_err(|_| {
                        Error::InvalidArgumentError(format!(
                            "ORDER BY position '{}' is not a valid positive integer",
                            raw
                        ))
                    })?;
                    if position == 0 {
                        return Err(Error::InvalidArgumentError(
                            "ORDER BY position must be at least 1".into(),
                        ));
                    }
                    (OrderTarget::Index(position - 1), OrderSortType::Native)
                }
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported ORDER BY literal expression: {other:?}"
                    )));
                }
            },
            other => {
                return Err(Error::InvalidArgumentError(format!(
                    "unsupported ORDER BY expression: {other:?}"
                )));
            }
        };

        Ok(OrderByPlan {
            target,
            sort_type,
            ascending,
            nulls_first,
        })
    }

    fn detect_simple_aggregates(
        &self,
        projection_items: &[SelectItem],
    ) -> SqlResult<Option<Vec<AggregateExpr>>> {
        if projection_items.is_empty() {
            return Ok(None);
        }

        let mut specs: Vec<AggregateExpr> = Vec::with_capacity(projection_items.len());
        for (idx, item) in projection_items.iter().enumerate() {
            let (expr, alias_opt) = match item {
                SelectItem::UnnamedExpr(expr) => (expr, None),
                SelectItem::ExprWithAlias { expr, alias } => (expr, Some(alias.value.clone())),
                _ => return Ok(None),
            };

            let alias = alias_opt.unwrap_or_else(|| format!("col{}", idx + 1));
            let SqlExpr::Function(func) = expr else {
                return Ok(None);
            };

            if func.uses_odbc_syntax {
                return Err(Error::InvalidArgumentError(
                    "ODBC function syntax is not supported in aggregate queries".into(),
                ));
            }
            if !matches!(func.parameters, FunctionArguments::None) {
                return Err(Error::InvalidArgumentError(
                    "parameterized aggregate functions are not supported".into(),
                ));
            }
            if func.filter.is_some()
                || func.null_treatment.is_some()
                || func.over.is_some()
                || !func.within_group.is_empty()
            {
                return Err(Error::InvalidArgumentError(
                    "advanced aggregate clauses are not supported".into(),
                ));
            }

            let args_slice: &[FunctionArg] = match &func.args {
                FunctionArguments::List(list) => {
                    if list.duplicate_treatment.is_some() {
                        return Err(Error::InvalidArgumentError(
                            "DISTINCT aggregates are not supported".into(),
                        ));
                    }
                    if !list.clauses.is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "aggregate argument clauses are not supported".into(),
                        ));
                    }
                    &list.args
                }
                FunctionArguments::None => &[],
                FunctionArguments::Subquery(_) => {
                    return Err(Error::InvalidArgumentError(
                        "aggregate subquery arguments are not supported".into(),
                    ));
                }
            };

            let func_name = if func.name.0.len() == 1 {
                match &func.name.0[0] {
                    ObjectNamePart::Identifier(ident) => ident.value.to_ascii_lowercase(),
                    _ => {
                        return Err(Error::InvalidArgumentError(
                            "unsupported aggregate function name".into(),
                        ));
                    }
                }
            } else {
                return Err(Error::InvalidArgumentError(
                    "qualified aggregate function names are not supported".into(),
                ));
            };

            let aggregate = match func_name.as_str() {
                "count" => {
                    if args_slice.len() != 1 {
                        return Err(Error::InvalidArgumentError(
                            "COUNT accepts exactly one argument".into(),
                        ));
                    }
                    match &args_slice[0] {
                        FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {
                            AggregateExpr::count_star(alias)
                        }
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(arg_expr)) => {
                            let column = resolve_column_name(arg_expr)?;
                            AggregateExpr::count_column(column, alias)
                        }
                        FunctionArg::Named { .. } | FunctionArg::ExprNamed { .. } => {
                            return Err(Error::InvalidArgumentError(
                                "named COUNT arguments are not supported".into(),
                            ));
                        }
                        FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_)) => {
                            return Err(Error::InvalidArgumentError(
                                "COUNT does not support qualified wildcards".into(),
                            ));
                        }
                    }
                }
                "sum" | "min" | "max" => {
                    if args_slice.len() != 1 {
                        return Err(Error::InvalidArgumentError(format!(
                            "{} accepts exactly one argument",
                            func_name.to_uppercase()
                        )));
                    }
                    let arg_expr = match &args_slice[0] {
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(arg_expr)) => arg_expr,
                        FunctionArg::Unnamed(FunctionArgExpr::Wildcard)
                        | FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_)) => {
                            return Err(Error::InvalidArgumentError(format!(
                                "{} does not support wildcard arguments",
                                func_name.to_uppercase()
                            )));
                        }
                        FunctionArg::Named { .. } | FunctionArg::ExprNamed { .. } => {
                            return Err(Error::InvalidArgumentError(format!(
                                "{} arguments must be column references",
                                func_name.to_uppercase()
                            )));
                        }
                    };

                    if func_name == "sum" {
                        if let Some(column) = parse_count_nulls_case(arg_expr)? {
                            AggregateExpr::count_nulls(column, alias)
                        } else {
                            let column = resolve_column_name(arg_expr)?;
                            AggregateExpr::sum_int64(column, alias)
                        }
                    } else {
                        let column = resolve_column_name(arg_expr)?;
                        if func_name == "min" {
                            AggregateExpr::min_int64(column, alias)
                        } else {
                            AggregateExpr::max_int64(column, alias)
                        }
                    }
                }
                _ => return Ok(None),
            };

            specs.push(aggregate);
        }

        if specs.is_empty() {
            return Ok(None);
        }
        Ok(Some(specs))
    }

    fn build_projection_list(
        &self,
        projection_items: &[SelectItem],
    ) -> SqlResult<Vec<SelectProjection>> {
        if projection_items.is_empty() {
            return Err(Error::InvalidArgumentError(
                "SELECT projection must include at least one column".into(),
            ));
        }
        let mut projections = Vec::with_capacity(projection_items.len());
        for (idx, item) in projection_items.iter().enumerate() {
            match item {
                SelectItem::Wildcard(_) => {
                    projections.push(SelectProjection::AllColumns);
                }
                SelectItem::QualifiedWildcard(kind, _) => match kind {
                    SelectItemQualifiedWildcardKind::ObjectName(name) => {
                        projections.push(SelectProjection::Column {
                            name: name.to_string(),
                            alias: None,
                        });
                    }
                    SelectItemQualifiedWildcardKind::Expr(_) => {
                        return Err(Error::InvalidArgumentError(
                            "expression-qualified wildcards are not supported".into(),
                        ));
                    }
                },
                SelectItem::UnnamedExpr(expr) => {
                    let scalar = translate_scalar(expr)?;
                    let alias = format!("col{}", idx + 1);
                    projections.push(SelectProjection::Computed {
                        expr: scalar,
                        alias,
                    });
                }
                SelectItem::ExprWithAlias { expr, alias } => {
                    let scalar = translate_scalar(expr)?;
                    projections.push(SelectProjection::Computed {
                        expr: scalar,
                        alias: alias.value.clone(),
                    });
                }
            }
        }
        Ok(projections)
    }

    #[allow(clippy::too_many_arguments)] // TODO: Refactor using struct for arg
    fn handle_start_transaction(
        &self,
        modes: Vec<TransactionMode>,
        begin: bool,
        transaction: Option<BeginTransactionKind>,
        modifier: Option<TransactionModifier>,
        statements: Vec<Statement>,
        exception: Option<Vec<ExceptionWhen>>,
        has_end_keyword: bool,
    ) -> SqlResult<StatementResult<P>> {
        if !modes.is_empty() {
            return Err(Error::InvalidArgumentError(
                "transaction modes are not supported".into(),
            ));
        }
        if modifier.is_some() {
            return Err(Error::InvalidArgumentError(
                "transaction modifiers are not supported".into(),
            ));
        }
        if !statements.is_empty() || exception.is_some() || has_end_keyword {
            return Err(Error::InvalidArgumentError(
                "BEGIN blocks with inline statements or exceptions are not supported".into(),
            ));
        }
        if let Some(kind) = transaction {
            match kind {
                BeginTransactionKind::Transaction | BeginTransactionKind::Work => {}
            }
        }
        if !begin {
            // Currently treat START TRANSACTION same as BEGIN
            tracing::warn!("Currently treat `START TRANSACTION` same as `BEGIN`")
        }
        let mut guard = self.transaction.lock().expect("transaction lock poisoned");
        if guard.is_some() {
            return Err(Error::InvalidArgumentError(
                "a transaction is already in progress".into(),
            ));
        }

        let staging_engine = SqlEngine::new(Arc::new(MemPager::default()));
        let catalog_snapshot = self.context.table_names().into_iter().collect();
        let state = TransactionState {
            staging: Box::new(staging_engine),
            statements: Vec::new(),
            staged_tables: HashSet::new(),
            table_deltas: HashMap::new(),
            new_tables: HashSet::new(),
            snapshotted_tables: HashSet::new(),
            missing_tables: HashSet::new(),
            catalog_snapshot,
        };

        *guard = Some(state);
        Ok(StatementResult::Transaction {
            kind: TransactionKind::Begin,
        })
    }

    fn handle_commit(
        &self,
        chain: bool,
        end: bool,
        modifier: Option<TransactionModifier>,
    ) -> SqlResult<StatementResult<P>> {
        if chain {
            return Err(Error::InvalidArgumentError(
                "COMMIT AND [NO] CHAIN is not supported".into(),
            ));
        }
        if end {
            return Err(Error::InvalidArgumentError(
                "END blocks are not supported".into(),
            ));
        }
        if modifier.is_some() {
            return Err(Error::InvalidArgumentError(
                "transaction modifiers are not supported".into(),
            ));
        }
        let mut guard = self.transaction.lock().expect("transaction lock poisoned");
        let tx = guard.take().ok_or_else(|| {
            Error::InvalidArgumentError("no transaction is currently in progress".into())
        })?;
        drop(guard);

        for stmt in tx.statements {
            let _ = self.execute(&stmt)?;
        }

        Ok(StatementResult::Transaction {
            kind: TransactionKind::Commit,
        })
    }

    fn handle_rollback(
        &self,
        chain: bool,
        savepoint: Option<Ident>,
    ) -> SqlResult<StatementResult<P>> {
        if chain {
            return Err(Error::InvalidArgumentError(
                "ROLLBACK AND [NO] CHAIN is not supported".into(),
            ));
        }
        if savepoint.is_some() {
            return Err(Error::InvalidArgumentError(
                "ROLLBACK TO SAVEPOINT is not supported".into(),
            ));
        }
        let mut guard = self.transaction.lock().expect("transaction lock poisoned");
        if guard.take().is_none() {
            return Err(Error::InvalidArgumentError(
                "no transaction is currently in progress".into(),
            ));
        }
        Ok(StatementResult::Transaction {
            kind: TransactionKind::Rollback,
        })
    }

    fn handle_set(&self, set_stmt: Set) -> SqlResult<StatementResult<P>> {
        match set_stmt {
            Set::SingleAssignment {
                scope,
                hivevar,
                variable,
                values,
            } => {
                if scope.is_some() || hivevar {
                    return Err(Error::InvalidArgumentError(
                        "SET modifiers are not supported".into(),
                    ));
                }

                let variable_name = variable.to_string();
                if !variable_name.eq_ignore_ascii_case("default_null_order") {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported SET variable: {variable_name}"
                    )));
                }

                if values.len() != 1 {
                    return Err(Error::InvalidArgumentError(
                        "SET default_null_order expects exactly one value".into(),
                    ));
                }

                let value_expr = &values[0];
                let normalized = match value_expr {
                    SqlExpr::Value(value_with_span) => value_with_span
                        .value
                        .clone()
                        .into_string()
                        .map(|s| s.to_ascii_lowercase()),
                    SqlExpr::Identifier(ident) => Some(ident.value.to_ascii_lowercase()),
                    _ => None,
                };

                if !matches!(normalized.as_deref(), Some("nulls_first" | "nulls_last")) {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported value for SET default_null_order: {value_expr:?}"
                    )));
                }

                let use_nulls_first = matches!(normalized.as_deref(), Some("nulls_first"));
                self.default_nulls_first
                    .store(use_nulls_first, AtomicOrdering::Relaxed);

                Ok(StatementResult::NoOp)
            }
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported SQL SET statement: {other:?}",
            ))),
        }
    }

    fn handle_pragma(
        &self,
        name: ObjectName,
        value: Option<Value>,
        is_eq: bool,
    ) -> SqlResult<StatementResult<P>> {
        let (display, canonical) = canonical_object_name(&name)?;
        if value.is_some() || is_eq {
            return Err(Error::InvalidArgumentError(format!(
                "PRAGMA '{display}' does not accept a value"
            )));
        }

        match canonical.as_str() {
            "enable_verification" | "disable_verification" => Ok(StatementResult::NoOp),
            _ => Err(Error::InvalidArgumentError(format!(
                "unsupported PRAGMA '{}'",
                display
            ))),
        }
    }
}

fn canonical_object_name(name: &ObjectName) -> SqlResult<(String, String)> {
    if name.0.is_empty() {
        return Err(Error::InvalidArgumentError(
            "object name must not be empty".into(),
        ));
    }
    let mut parts: Vec<String> = Vec::with_capacity(name.0.len());
    for part in &name.0 {
        let ident = match part {
            ObjectNamePart::Identifier(ident) => ident,
            _ => {
                return Err(Error::InvalidArgumentError(
                    "object names using functions are not supported".into(),
                ));
            }
        };
        parts.push(ident.value.clone());
    }
    let display = parts.join(".");
    let canonical = display.to_ascii_lowercase();
    Ok((display, canonical))
}

fn validate_create_table_common(stmt: &sqlparser::ast::CreateTable) -> SqlResult<()> {
    if stmt.clone.is_some() || stmt.like.is_some() {
        return Err(Error::InvalidArgumentError(
            "CREATE TABLE LIKE/CLONE is not supported".into(),
        ));
    }
    if stmt.or_replace {
        return Err(Error::InvalidArgumentError(
            "CREATE OR REPLACE TABLE is not supported".into(),
        ));
    }
    if !stmt.constraints.is_empty() {
        return Err(Error::InvalidArgumentError(
            "table-level constraints are not supported".into(),
        ));
    }
    Ok(())
}

fn validate_create_table_definition(stmt: &sqlparser::ast::CreateTable) -> SqlResult<()> {
    for column in &stmt.columns {
        for ColumnOptionDef { option, .. } in &column.options {
            match option {
                ColumnOption::Null | ColumnOption::NotNull | ColumnOption::Unique { .. } => {}
                ColumnOption::Default(_) => {
                    return Err(Error::InvalidArgumentError(format!(
                        "DEFAULT values are not supported for column '{}'",
                        column.name
                    )));
                }
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported column option {:?} on '{}'",
                        other, column.name
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_create_table_as(stmt: &sqlparser::ast::CreateTable) -> SqlResult<()> {
    if !stmt.columns.is_empty() {
        return Err(Error::InvalidArgumentError(
            "CREATE TABLE AS SELECT does not support column definitions yet".into(),
        ));
    }
    Ok(())
}

fn validate_simple_query(query: &Query) -> SqlResult<()> {
    if query.with.is_some() {
        return Err(Error::InvalidArgumentError(
            "WITH clauses are not supported".into(),
        ));
    }
    if let Some(limit_clause) = &query.limit_clause {
        match limit_clause {
            LimitClause::LimitOffset {
                offset: Some(_), ..
            }
            | LimitClause::OffsetCommaLimit { .. } => {
                return Err(Error::InvalidArgumentError(
                    "OFFSET clauses are not supported".into(),
                ));
            }
            LimitClause::LimitOffset { limit_by, .. } if !limit_by.is_empty() => {
                return Err(Error::InvalidArgumentError(
                    "LIMIT BY clauses are not supported".into(),
                ));
            }
            _ => {}
        }
    }
    if query.fetch.is_some() {
        return Err(Error::InvalidArgumentError(
            "FETCH clauses are not supported".into(),
        ));
    }
    Ok(())
}

fn resolve_column_name(expr: &SqlExpr) -> SqlResult<String> {
    match expr {
        SqlExpr::Identifier(ident) => Ok(ident.value.clone()),
        SqlExpr::CompoundIdentifier(parts) => {
            if let Some(last) = parts.last() {
                Ok(last.value.clone())
            } else {
                Err(Error::InvalidArgumentError(
                    "empty column identifier".into(),
                ))
            }
        }
        _ => Err(Error::InvalidArgumentError(
            "aggregate arguments must be plain column identifiers".into(),
        )),
    }
}

fn parse_count_nulls_case(expr: &SqlExpr) -> SqlResult<Option<String>> {
    let SqlExpr::Case {
        operand,
        conditions,
        else_result,
        ..
    } = expr
    else {
        return Ok(None);
    };

    if operand.is_some() || conditions.len() != 1 {
        return Ok(None);
    }

    let case_when = &conditions[0];
    if !is_integer_literal(&case_when.result, 1) {
        return Ok(None);
    }

    let else_expr = match else_result {
        Some(expr) => expr.as_ref(),
        None => return Ok(None),
    };
    if !is_integer_literal(else_expr, 0) {
        return Ok(None);
    }

    let inner = match &case_when.condition {
        SqlExpr::IsNull(inner) => inner.as_ref(),
        _ => return Ok(None),
    };

    resolve_column_name(inner).map(Some)
}

fn is_integer_literal(expr: &SqlExpr, expected: i64) -> bool {
    match expr {
        SqlExpr::Value(ValueWithSpan {
            value: Value::Number(text, _),
            ..
        }) => text.parse::<i64>() == Ok(expected),
        _ => false,
    }
}

fn translate_condition(expr: &SqlExpr) -> SqlResult<llkv_expr::expr::Expr<'static, String>> {
    match expr {
        SqlExpr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => Ok(llkv_expr::expr::Expr::And(vec![
                translate_condition(left)?,
                translate_condition(right)?,
            ])),
            BinaryOperator::Or => Ok(llkv_expr::expr::Expr::Or(vec![
                translate_condition(left)?,
                translate_condition(right)?,
            ])),
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Lt
            | BinaryOperator::LtEq
            | BinaryOperator::Gt
            | BinaryOperator::GtEq => translate_comparison(left, op.clone(), right),
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported binary operator in WHERE clause: {other:?}"
            ))),
        },
        SqlExpr::UnaryOp {
            op: UnaryOperator::Not,
            expr,
        } => Ok(llkv_expr::expr::Expr::not(translate_condition(expr)?)),
        SqlExpr::Nested(inner) => translate_condition(inner),
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported WHERE clause: {other:?}"
        ))),
    }
}

fn translate_comparison(
    left: &SqlExpr,
    op: BinaryOperator,
    right: &SqlExpr,
) -> SqlResult<llkv_expr::expr::Expr<'static, String>> {
    let left_scalar = translate_scalar(left)?;
    let right_scalar = translate_scalar(right)?;
    let compare_op = match op {
        BinaryOperator::Eq => llkv_expr::expr::CompareOp::Eq,
        BinaryOperator::NotEq => llkv_expr::expr::CompareOp::NotEq,
        BinaryOperator::Lt => llkv_expr::expr::CompareOp::Lt,
        BinaryOperator::LtEq => llkv_expr::expr::CompareOp::LtEq,
        BinaryOperator::Gt => llkv_expr::expr::CompareOp::Gt,
        BinaryOperator::GtEq => llkv_expr::expr::CompareOp::GtEq,
        other => {
            return Err(Error::InvalidArgumentError(format!(
                "unsupported comparison operator: {other:?}"
            )));
        }
    };

    if let (
        llkv_expr::expr::ScalarExpr::Column(column),
        llkv_expr::expr::ScalarExpr::Literal(literal),
    ) = (&left_scalar, &right_scalar)
        && let Some(op) = compare_op_to_filter_operator(compare_op, literal)
    {
        return Ok(llkv_expr::expr::Expr::Pred(llkv_expr::expr::Filter {
            field_id: column.clone(),
            op,
        }));
    }

    if let (
        llkv_expr::expr::ScalarExpr::Literal(literal),
        llkv_expr::expr::ScalarExpr::Column(column),
    ) = (&left_scalar, &right_scalar)
        && let Some(flipped) = flip_compare_op(compare_op)
        && let Some(op) = compare_op_to_filter_operator(flipped, literal)
    {
        return Ok(llkv_expr::expr::Expr::Pred(llkv_expr::expr::Filter {
            field_id: column.clone(),
            op,
        }));
    }

    Ok(llkv_expr::expr::Expr::Compare {
        left: left_scalar,
        op: compare_op,
        right: right_scalar,
    })
}

fn compare_op_to_filter_operator(
    op: llkv_expr::expr::CompareOp,
    literal: &Literal,
) -> Option<llkv_expr::expr::Operator<'static>> {
    let lit = literal.clone();
    match op {
        llkv_expr::expr::CompareOp::Eq => Some(llkv_expr::expr::Operator::Equals(lit)),
        llkv_expr::expr::CompareOp::Lt => Some(llkv_expr::expr::Operator::LessThan(lit)),
        llkv_expr::expr::CompareOp::LtEq => Some(llkv_expr::expr::Operator::LessThanOrEquals(lit)),
        llkv_expr::expr::CompareOp::Gt => Some(llkv_expr::expr::Operator::GreaterThan(lit)),
        llkv_expr::expr::CompareOp::GtEq => {
            Some(llkv_expr::expr::Operator::GreaterThanOrEquals(lit))
        }
        llkv_expr::expr::CompareOp::NotEq => None,
    }
}

fn flip_compare_op(op: llkv_expr::expr::CompareOp) -> Option<llkv_expr::expr::CompareOp> {
    match op {
        llkv_expr::expr::CompareOp::Eq => Some(llkv_expr::expr::CompareOp::Eq),
        llkv_expr::expr::CompareOp::Lt => Some(llkv_expr::expr::CompareOp::Gt),
        llkv_expr::expr::CompareOp::LtEq => Some(llkv_expr::expr::CompareOp::GtEq),
        llkv_expr::expr::CompareOp::Gt => Some(llkv_expr::expr::CompareOp::Lt),
        llkv_expr::expr::CompareOp::GtEq => Some(llkv_expr::expr::CompareOp::LtEq),
        llkv_expr::expr::CompareOp::NotEq => None,
    }
}

fn translate_scalar(expr: &SqlExpr) -> SqlResult<llkv_expr::expr::ScalarExpr<String>> {
    match expr {
        SqlExpr::Identifier(ident) => Ok(llkv_expr::expr::ScalarExpr::column(ident.value.clone())),
        SqlExpr::CompoundIdentifier(idents) => {
            if let Some(last) = idents.last() {
                translate_scalar(&SqlExpr::Identifier(last.clone()))
            } else {
                Err(Error::InvalidArgumentError(
                    "invalid compound identifier".into(),
                ))
            }
        }
        SqlExpr::Value(value) => literal_from_value(value),
        SqlExpr::BinaryOp { left, op, right } => {
            let left_expr = translate_scalar(left)?;
            let right_expr = translate_scalar(right)?;
            let op = match op {
                BinaryOperator::Plus => llkv_expr::expr::BinaryOp::Add,
                BinaryOperator::Minus => llkv_expr::expr::BinaryOp::Subtract,
                BinaryOperator::Multiply => llkv_expr::expr::BinaryOp::Multiply,
                BinaryOperator::Divide => llkv_expr::expr::BinaryOp::Divide,
                BinaryOperator::Modulo => llkv_expr::expr::BinaryOp::Modulo,
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported scalar binary operator: {other:?}"
                    )));
                }
            };
            Ok(llkv_expr::expr::ScalarExpr::binary(
                left_expr, op, right_expr,
            ))
        }
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match translate_scalar(expr)? {
            llkv_expr::expr::ScalarExpr::Literal(lit) => match lit {
                Literal::Integer(v) => {
                    Ok(llkv_expr::expr::ScalarExpr::literal(Literal::Integer(-v)))
                }
                Literal::Float(v) => Ok(llkv_expr::expr::ScalarExpr::literal(Literal::Float(-v))),
                Literal::String(_) => Err(Error::InvalidArgumentError(
                    "cannot negate string literal".into(),
                )),
            },
            _ => Err(Error::InvalidArgumentError(
                "cannot negate non-literal expression".into(),
            )),
        },
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => translate_scalar(expr),
        SqlExpr::Nested(inner) => translate_scalar(inner),
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported scalar expression: {other:?}"
        ))),
    }
}

fn literal_from_value(value: &ValueWithSpan) -> SqlResult<llkv_expr::expr::ScalarExpr<String>> {
    match &value.value {
        Value::Number(text, _) => {
            if text.contains(['.', 'e', 'E']) {
                let parsed = text.parse::<f64>().map_err(|err| {
                    Error::InvalidArgumentError(format!("invalid float literal: {err}"))
                })?;
                Ok(llkv_expr::expr::ScalarExpr::literal(Literal::Float(parsed)))
            } else {
                let parsed = text.parse::<i128>().map_err(|err| {
                    Error::InvalidArgumentError(format!("invalid integer literal: {err}"))
                })?;
                Ok(llkv_expr::expr::ScalarExpr::literal(Literal::Integer(
                    parsed,
                )))
            }
        }
        Value::Boolean(_) => Err(Error::InvalidArgumentError(
            "BOOLEAN literals are not supported yet".into(),
        )),
        Value::Null => Err(Error::InvalidArgumentError(
            "NULL literal is not supported in comparisons; use IS NULL".into(),
        )),
        other => {
            if let Some(text) = other.clone().into_string() {
                Ok(llkv_expr::expr::ScalarExpr::literal(Literal::String(text)))
            } else {
                Err(Error::InvalidArgumentError(format!(
                    "unsupported literal: {other:?}"
                )))
            }
        }
    }
}

fn resolve_assignment_column_name(target: &AssignmentTarget) -> SqlResult<String> {
    match target {
        AssignmentTarget::ColumnName(name) => {
            if name.0.len() != 1 {
                return Err(Error::InvalidArgumentError(
                    "qualified column names in UPDATE assignments are not supported yet".into(),
                ));
            }
            match &name.0[0] {
                ObjectNamePart::Identifier(ident) => Ok(ident.value.clone()),
                other => Err(Error::InvalidArgumentError(format!(
                    "unsupported column reference in UPDATE assignment: {other:?}"
                ))),
            }
        }
        AssignmentTarget::Tuple(_) => Err(Error::InvalidArgumentError(
            "tuple assignments are not supported yet".into(),
        )),
    }
}

fn arrow_type_from_sql(data_type: &SqlDataType) -> SqlResult<arrow::datatypes::DataType> {
    use arrow::datatypes::DataType;
    match data_type {
        SqlDataType::Int(_)
        | SqlDataType::Integer(_)
        | SqlDataType::BigInt(_)
        | SqlDataType::SmallInt(_)
        | SqlDataType::TinyInt(_) => Ok(DataType::Int64),
        SqlDataType::Float(_)
        | SqlDataType::Real
        | SqlDataType::Double(_)
        | SqlDataType::DoublePrecision => Ok(DataType::Float64),
        SqlDataType::Text
        | SqlDataType::String(_)
        | SqlDataType::Varchar(_)
        | SqlDataType::Char(_)
        | SqlDataType::Uuid => Ok(DataType::Utf8),
        SqlDataType::Date => Ok(DataType::Date32),
        SqlDataType::Decimal(_) | SqlDataType::Numeric(_) => Ok(DataType::Float64),
        SqlDataType::Boolean => Err(Error::InvalidArgumentError(
            "BOOLEAN columns are not supported yet".into(),
        )),
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported SQL data type: {other:?}"
        ))),
    }
}

fn extract_constant_select_rows(select: &Select) -> SqlResult<Option<Vec<Vec<DslValue>>>> {
    if !select.from.is_empty() {
        return Ok(None);
    }

    if select.selection.is_some()
        || select.having.is_some()
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
            "constant SELECT statements do not support advanced clauses".into(),
        ));
    }

    if select.projection.is_empty() {
        return Err(Error::InvalidArgumentError(
            "constant SELECT requires at least one projection".into(),
        ));
    }

    let mut row: Vec<DslValue> = Vec::with_capacity(select.projection.len());
    for item in &select.projection {
        let expr = match item {
            SelectItem::UnnamedExpr(expr) => expr,
            SelectItem::ExprWithAlias { expr, .. } => expr,
            other => {
                return Err(Error::InvalidArgumentError(format!(
                    "unsupported projection in constant SELECT: {other:?}"
                )));
            }
        };

        let value = SqlValue::try_from_expr(expr)?;
        row.push(DslValue::from(value));
    }

    Ok(Some(vec![row]))
}

fn extract_single_table(from: &[TableWithJoins]) -> SqlResult<(String, String)> {
    if from.len() != 1 {
        return Err(Error::InvalidArgumentError(
            "queries over multiple tables are not supported yet".into(),
        ));
    }
    let item = &from[0];
    if !item.joins.is_empty() {
        return Err(Error::InvalidArgumentError(
            "JOIN clauses are not supported yet".into(),
        ));
    }
    match &item.relation {
        TableFactor::Table { name, .. } => canonical_object_name(name),
        _ => Err(Error::InvalidArgumentError(
            "queries require a plain table name".into(),
        )),
    }
}

fn group_by_is_empty(expr: &GroupByExpr) -> bool {
    matches!(
        expr,
        GroupByExpr::Expressions(exprs, modifiers)
            if exprs.is_empty() && modifiers.is_empty()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, StringArray};
    use arrow::record_batch::RecordBatch;
    use llkv_storage::pager::MemPager;

    fn extract_string_options(batches: &[RecordBatch]) -> Vec<Option<String>> {
        let mut values: Vec<Option<String>> = Vec::new();
        for batch in batches {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("string column");
            for idx in 0..column.len() {
                if column.is_null(idx) {
                    values.push(None);
                } else {
                    values.push(Some(column.value(idx).to_string()));
                }
            }
        }
        values
    }

    #[test]
    fn create_insert_select_roundtrip() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        let result = engine
            .execute("CREATE TABLE people (id INT NOT NULL, name TEXT NOT NULL)")
            .expect("create table");
        assert!(matches!(result[0], StatementResult::CreateTable { .. }));

        let result = engine
            .execute("INSERT INTO people (id, name) VALUES (1, 'alice'), (2, 'bob')")
            .expect("insert rows");
        assert!(matches!(
            result[0],
            StatementResult::Insert {
                rows_inserted: 2,
                ..
            }
        ));

        let mut result = engine
            .execute("SELECT name FROM people WHERE id = 2")
            .expect("select rows");
        let select_result = result.remove(0);
        let batches = match select_result {
            StatementResult::Select { execution, .. } => {
                execution.collect().expect("collect batches")
            }
            _ => panic!("expected select result"),
        };
        assert_eq!(batches.len(), 1);
        let column = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string column");
        assert_eq!(column.len(), 1);
        assert_eq!(column.value(0), "bob");
    }

    #[test]
    fn insert_select_constant_including_null() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE integers(i INTEGER)")
            .expect("create table");

        let result = engine
            .execute("INSERT INTO integers SELECT 42")
            .expect("insert literal");
        assert!(matches!(
            result[0],
            StatementResult::Insert {
                rows_inserted: 1,
                ..
            }
        ));

        let result = engine
            .execute("INSERT INTO integers SELECT CAST(NULL AS VARCHAR)")
            .expect("insert null literal");
        assert!(matches!(
            result[0],
            StatementResult::Insert {
                rows_inserted: 1,
                ..
            }
        ));

        let mut result = engine
            .execute("SELECT * FROM integers")
            .expect("select rows");
        let select_result = result.remove(0);
        let batches = match select_result {
            StatementResult::Select { execution, .. } => {
                execution.collect().expect("collect batches")
            }
            _ => panic!("expected select result"),
        };

        let mut values: Vec<Option<i64>> = Vec::new();
        for batch in &batches {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int column");
            for idx in 0..column.len() {
                if column.is_null(idx) {
                    values.push(None);
                } else {
                    values.push(Some(column.value(idx)));
                }
            }
        }

        assert_eq!(values, vec![Some(42), None]);
    }

    #[test]
    fn update_with_where_clause_filters_rows() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("SET default_null_order='nulls_first'")
            .expect("set default null order");

        engine
            .execute("CREATE TABLE strings(a VARCHAR)")
            .expect("create table");

        engine
            .execute("INSERT INTO strings VALUES ('3'), ('4'), (NULL)")
            .expect("insert seed rows");

        let result = engine
            .execute("UPDATE strings SET a = 13 WHERE a = '3'")
            .expect("update rows");
        assert!(matches!(
            result[0],
            StatementResult::Update {
                rows_updated: 1,
                ..
            }
        ));

        let mut result = engine
            .execute("SELECT * FROM strings ORDER BY cast(a AS INTEGER)")
            .expect("select rows");
        let select_result = result.remove(0);
        let batches = match select_result {
            StatementResult::Select { execution, .. } => {
                execution.collect().expect("collect batches")
            }
            _ => panic!("expected select result"),
        };

        let mut values: Vec<Option<String>> = Vec::new();
        for batch in &batches {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("string column");
            for idx in 0..column.len() {
                if column.is_null(idx) {
                    values.push(None);
                } else {
                    values.push(Some(column.value(idx).to_string()));
                }
            }
        }

        values.sort_by(|a, b| match (a, b) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (Some(av), Some(bv)) => {
                let a_val = av.parse::<i64>().unwrap_or_default();
                let b_val = bv.parse::<i64>().unwrap_or_default();
                a_val.cmp(&b_val)
            }
        });

        assert_eq!(
            values,
            vec![None, Some("4".to_string()), Some("13".to_string())]
        );
    }

    #[test]
    fn order_by_honors_configured_default_null_order() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE strings(a VARCHAR)")
            .expect("create table");
        engine
            .execute("INSERT INTO strings VALUES ('3'), ('4'), (NULL)")
            .expect("insert values");
        engine
            .execute("UPDATE strings SET a = 13 WHERE a = '3'")
            .expect("update value");

        let mut result = engine
            .execute("SELECT * FROM strings ORDER BY cast(a AS INTEGER)")
            .expect("select rows");
        let select_result = result.remove(0);
        let batches = match select_result {
            StatementResult::Select { execution, .. } => {
                execution.collect().expect("collect batches")
            }
            _ => panic!("expected select result"),
        };

        let values = extract_string_options(&batches);
        assert_eq!(
            values,
            vec![Some("4".to_string()), Some("13".to_string()), None]
        );

        assert!(!engine.default_nulls_first_for_tests());

        engine
            .execute("SET default_null_order='nulls_first'")
            .expect("set default null order");

        assert!(engine.default_nulls_first_for_tests());

        let mut result = engine
            .execute("SELECT * FROM strings ORDER BY cast(a AS INTEGER)")
            .expect("select rows");
        let select_result = result.remove(0);
        let batches = match select_result {
            StatementResult::Select { execution, .. } => {
                execution.collect().expect("collect batches")
            }
            _ => panic!("expected select result"),
        };

        let values = extract_string_options(&batches);
        assert_eq!(
            values,
            vec![None, Some("4".to_string()), Some("13".to_string())]
        );
    }
}
