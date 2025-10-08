use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::ops::Bound;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::SqlResult;
use crate::value::SqlValue;
use arrow::array::{
    Array, ArrayRef, Date32Array, Date32Builder, Float64Array, Float64Builder, Int64Array,
    Int64Builder, StringArray, StringBuilder, UInt64Builder, new_null_array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ColumnStore;
use llkv_column_map::store::{Projection, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::expr::{BinaryOp, CompareOp, Expr as LlkvExpr, Filter, Operator, ScalarExpr};
use llkv_expr::literal::Literal;
use llkv_result::Error;
use llkv_storage::pager::Pager;
use llkv_table::table::{ScanProjection, ScanStreamOptions, Table};
use llkv_table::types::{FieldId, TableId};
use llkv_table::{CATALOG_TABLE_ID, SysCatalog};
use llkv_table::{ColMeta, TableMeta};
use simd_r_drive_entry_handle::EntryHandle;
use sqlparser::ast::{
    Assignment, AssignmentTarget, BeginTransactionKind, BinaryOperator, ColumnDef, ColumnOption,
    ColumnOptionDef, DataType as SqlDataType, ExceptionWhen, Expr as SqlExpr, FunctionArg,
    FunctionArgExpr, FunctionArguments, GroupByExpr, Ident, ObjectName, ObjectNamePart, Query,
    Select, SelectItem, SelectItemQualifiedWildcardKind, SetExpr, Statement, TableFactor,
    TableObject, TableWithJoins, TransactionMode, TransactionModifier, UnaryOperator,
    UpdateTableFromKind, Value, ValueWithSpan,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use time::{Date, Month};

/// Executes SQL statements against `llkv-table`.
pub struct SqlEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pager: Arc<P>,
    tables: RwLock<HashMap<String, Arc<SqlTable<P>>>>,
    transaction_active: Mutex<bool>,
}

impl<P> SqlEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Create a new SQL engine over the provided pager.
    pub fn new(pager: Arc<P>) -> Self {
        Self {
            pager,
            tables: RwLock::new(HashMap::new()),
            transaction_active: Mutex::new(false),
        }
    }

    /// Execute one or more SQL statements contained in `sql`.
    pub fn execute(&self, sql: &str) -> SqlResult<Vec<SqlStatementResult>> {
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(&dialect, sql)
            .map_err(|err| Error::InvalidArgumentError(format!("failed to parse SQL: {err}")))?;

        let mut results = Vec::with_capacity(statements.len());
        for stmt in statements {
            results.push(self.execute_statement(stmt)?);
        }
        Ok(results)
    }

    fn execute_statement(&self, statement: Statement) -> SqlResult<SqlStatementResult> {
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
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported SQL statement: {other:?}"
            ))),
        }
    }

    fn parse_count_nulls_case(
        &self,
        table: &SqlTable<P>,
        expr: &SqlExpr,
    ) -> SqlResult<Option<FieldId>> {
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

        let column = self.resolve_aggregate_column(table, inner)?;
        Ok(Some(column.field_id))
    }

    fn handle_create_table(
        &self,
        stmt: sqlparser::ast::CreateTable,
    ) -> SqlResult<SqlStatementResult> {
        validate_create_table_common(&stmt)?;

        let (display_name, canonical_name) = canonical_object_name(&stmt.name)?;
        if display_name.is_empty() {
            return Err(Error::InvalidArgumentError(
                "table name must not be empty".into(),
            ));
        }

        if stmt.query.is_some() {
            return self.handle_create_table_as(stmt, display_name, canonical_name);
        }

        if stmt.columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires at least one column".into(),
            ));
        }

        validate_create_table_definition(&stmt)?;

        let exists = {
            let tables = self.tables.read().unwrap();
            tables.contains_key(&canonical_name)
        };
        if exists {
            if stmt.if_not_exists {
                return Ok(SqlStatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' already exists",
                display_name
            )));
        }

        let mut columns: Vec<SqlColumn> = Vec::with_capacity(stmt.columns.len());
        let mut lookup: HashMap<String, usize> = HashMap::new();
        for (idx, column_def) in stmt.columns.iter().enumerate() {
            let column = SqlColumn::try_from_ast(column_def, (idx + 1) as FieldId)?;
            if lookup
                .insert(column.normalized_name.clone(), columns.len())
                .is_some()
            {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in table '{}'",
                    column.name, display_name
                )));
            }
            columns.push(column);
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

        for column in &columns {
            table.put_col_meta(&ColMeta {
                col_id: column.field_id,
                name: Some(column.name.clone()),
                flags: 0,
                default: None,
            });
        }

        let schema = Arc::new(SqlSchema { columns, lookup });
        let sql_table = Arc::new(SqlTable {
            table: Arc::new(table),
            schema,
            next_row_id: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
        });

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            if stmt.if_not_exists {
                return Ok(SqlStatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' already exists",
                display_name
            )));
        }
        let stored_table = Arc::clone(&sql_table);
        tables.insert(canonical_name, sql_table);

        let field_ids = stored_table
            .table
            .store()
            .user_field_ids_for_table(stored_table.table.table_id());
        tracing::debug!(table = %display_name, ?field_ids, "CTAS registered field ids for table");

        Ok(SqlStatementResult::CreateTable {
            table_name: display_name,
        })
    }

    fn handle_create_table_as(
        &self,
        mut stmt: sqlparser::ast::CreateTable,
        display_name: String,
        canonical_name: String,
    ) -> SqlResult<SqlStatementResult> {
        validate_create_table_as(&stmt)?;

        if !stmt.columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE AS SELECT does not support column definitions yet".into(),
            ));
        }

        let exists = {
            let tables = self.tables.read().unwrap();
            tables.contains_key(&canonical_name)
        };
        if exists {
            if stmt.if_not_exists {
                return Ok(SqlStatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' already exists",
                display_name
            )));
        }

        let query = stmt
            .query
            .take()
            .expect("CTAS path verified query is present");
        let SelectExecution {
            table_name: _,
            schema: select_schema,
            batches,
        } = self.execute_query_collect(*query)?;
        tracing::info!(batch_count = batches.len(), "CTAS source produced batches");

        if select_schema.fields().is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE AS SELECT requires at least one projected column".into(),
            ));
        }

        let mut columns: Vec<SqlColumn> = Vec::with_capacity(select_schema.fields().len());
        let mut lookup: HashMap<String, usize> = HashMap::new();
        for (idx, field) in select_schema.fields().iter().enumerate() {
            let column = SqlColumn::from_field(field, (idx + 1) as FieldId)?;
            if lookup
                .insert(column.normalized_name.clone(), columns.len())
                .is_some()
            {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in CTAS result",
                    column.name
                )));
            }
            columns.push(column);
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

        for column in &columns {
            table.put_col_meta(&ColMeta {
                col_id: column.field_id,
                name: Some(column.name.clone()),
                flags: 0,
                default: None,
            });
        }

        let column_defs = columns.clone();
        let sql_schema = Arc::new(SqlSchema { columns, lookup });
        let sql_table = Arc::new(SqlTable {
            table: Arc::new(table),
            schema: sql_schema,
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
            sql_table.table.append(&append_batch)?;
        }

        sql_table.next_row_id.store(next_row_id, Ordering::SeqCst);
        sql_table.total_rows.store(total_rows, Ordering::SeqCst);

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            if stmt.if_not_exists {
                return Ok(SqlStatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' already exists",
                display_name
            )));
        }
        tables.insert(canonical_name, sql_table);

        Ok(SqlStatementResult::CreateTable {
            table_name: display_name,
        })
    }

    fn handle_insert(&self, stmt: sqlparser::ast::Insert) -> SqlResult<SqlStatementResult> {
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

        let (display_name, canonical_name) = match &stmt.table {
            TableObject::TableName(name) => canonical_object_name(name)?,
            _ => {
                return Err(Error::InvalidArgumentError(
                    "INSERT requires a plain table name".into(),
                ));
            }
        };

        let table = self.lookup_table(&canonical_name)?;
        let query = stmt
            .source
            .as_ref()
            .ok_or_else(|| Error::InvalidArgumentError("INSERT requires a VALUES clause".into()))?;
        validate_simple_query(query)?;

        let column_order = resolve_insert_columns(&stmt.columns, table.schema.as_ref())?;
        let expected_len = column_order.len();

        let rows: Vec<Vec<SqlValue>> = match query.body.as_ref() {
            SetExpr::Values(values) => {
                if values.rows.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "INSERT VALUES list must contain at least one row".into(),
                    ));
                }
                let mut out: Vec<Vec<SqlValue>> = Vec::with_capacity(values.rows.len());
                for row in &values.rows {
                    if row.len() != expected_len {
                        return Err(Error::InvalidArgumentError(format!(
                            "expected {} values in INSERT row, found {}",
                            expected_len,
                            row.len()
                        )));
                    }
                    let mut converted = Vec::with_capacity(row.len());
                    for expr in row {
                        converted.push(SqlValue::try_from_expr(expr)?);
                    }
                    out.push(converted);
                }
                out
            }
            SetExpr::Select(select) => {
                if let Some(range_rows) = extract_rows_from_range(select.as_ref())? {
                    if range_rows.column_count != expected_len {
                        return Err(Error::InvalidArgumentError(format!(
                            "expected {} values in INSERT SELECT, found {}",
                            expected_len, range_rows.column_count
                        )));
                    }
                    range_rows.rows
                } else {
                    let SelectExecution {
                        batches, schema, ..
                    } = self.execute_query_collect((**query).clone())?;
                    if schema.fields().len() != expected_len {
                        return Err(Error::InvalidArgumentError(format!(
                            "expected {} values in INSERT SELECT, found {}",
                            expected_len,
                            schema.fields().len()
                        )));
                    }
                    let mut out: Vec<Vec<SqlValue>> = Vec::new();
                    for batch in batches {
                        if batch.num_columns() != expected_len {
                            return Err(Error::InvalidArgumentError(
                                "INSERT SELECT produced unexpected column count".into(),
                            ));
                        }
                        let row_count = batch.num_rows();
                        for row_idx in 0..row_count {
                            let mut row: Vec<SqlValue> = Vec::with_capacity(expected_len);
                            for col_idx in 0..expected_len {
                                row.push(sql_value_from_array(batch.column(col_idx), row_idx)?);
                            }
                            out.push(row);
                        }
                    }
                    out
                }
            }
            _ => {
                return Err(Error::InvalidArgumentError(
                    "INSERT currently supports only VALUES lists or SELECT statements over range()"
                        .into(),
                ));
            }
        };

        let row_count = rows.len();
        tracing::info!(row_count = row_count, "INSERT row_count");

        let mut column_values: Vec<Vec<SqlValue>> = table
            .schema
            .columns
            .iter()
            .map(|_| Vec::with_capacity(row_count))
            .collect();

        let mut provided = vec![false; table.schema.columns.len()];
        for &idx in &column_order {
            provided[idx] = true;
        }

        for row in rows.into_iter() {
            if row.len() != expected_len {
                return Err(Error::InvalidArgumentError(format!(
                    "expected {} values in INSERT row, found {}",
                    expected_len,
                    row.len()
                )));
            }
            for (pos, value) in row.into_iter().enumerate() {
                let target_index = column_order[pos];
                column_values[target_index].push(value);
            }
            for (idx, column) in table.schema.columns.iter().enumerate() {
                if provided[idx] {
                    continue;
                }
                if !column.nullable {
                    return Err(Error::InvalidArgumentError(format!(
                        "column '{}' requires a value in INSERT",
                        column.name
                    )));
                }
                column_values[idx].push(SqlValue::Null);
            }
        }

        let row_start = table
            .next_row_id
            .fetch_add(row_count as u64, Ordering::SeqCst);
        let mut row_id_builder = UInt64Builder::with_capacity(row_count);
        for pos in 0..row_count {
            row_id_builder.append_value(row_start + pos as u64);
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
            .total_rows
            .fetch_add(row_count as u64, Ordering::SeqCst);

        Ok(SqlStatementResult::Insert {
            table_name: display_name,
            rows_inserted: row_count,
        })
    }
    fn handle_update(
        &self,
        table: TableWithJoins,
        assignments: Vec<Assignment>,
        from: Option<UpdateTableFromKind>,
        selection: Option<SqlExpr>,
        returning: Option<Vec<SelectItem>>,
    ) -> SqlResult<SqlStatementResult> {
        if from.is_some() {
            return Err(Error::InvalidArgumentError(
                "UPDATE ... FROM is not supported yet".into(),
            ));
        }
        if selection.is_some() {
            return Err(Error::InvalidArgumentError(
                "UPDATE with WHERE clauses is not supported yet".into(),
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

        let (display_name, canonical_name) = extract_single_table(std::slice::from_ref(&table))?;
        let table_entry = self.lookup_table(&canonical_name)?;

        let total_rows = table_entry.total_rows.load(Ordering::SeqCst);
        let total_rows_usize = usize::try_from(total_rows).map_err(|_| {
            Error::InvalidArgumentError("table row count exceeds supported range".into())
        })?;

        if total_rows_usize == 0 {
            return Ok(SqlStatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let mut row_id_builder = UInt64Builder::with_capacity(total_rows_usize);
        for row_id in 0..total_rows {
            row_id_builder.append_value(row_id);
        }

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(assignments.len() + 1);
        arrays.push(Arc::new(row_id_builder.finish()));

        let mut fields: Vec<Field> = Vec::with_capacity(assignments.len() + 1);
        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        let mut seen_columns: HashSet<String> = HashSet::with_capacity(assignments.len());

        for assignment in assignments {
            let column_name = resolve_assignment_column_name(&assignment.target)?;
            let normalized_name = column_name.to_ascii_lowercase();
            if !seen_columns.insert(normalized_name) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in UPDATE assignments",
                    column_name
                )));
            }
            let column = table_entry.schema.resolve(&column_name).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{}' in UPDATE", column_name))
            })?;

            let literal = SqlValue::try_from_expr(&assignment.value)?;
            let mut column_values: Vec<SqlValue> = Vec::with_capacity(total_rows_usize);
            for _ in 0..total_rows_usize {
                column_values.push(literal.clone());
            }

            let array = build_array_for_column(&column.data_type, &column_values)?;
            arrays.push(array);

            let mut metadata = HashMap::new();
            metadata.insert(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                column.field_id.to_string(),
            );
            fields.push(
                Field::new(&column.name, column.data_type.clone(), column.nullable)
                    .with_metadata(metadata),
            );
        }

        let update_batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        table_entry.table.append(&update_batch)?;

        Ok(SqlStatementResult::Update {
            table_name: display_name,
            rows_updated: total_rows_usize,
        })
    }

    fn handle_query(&self, query: Query) -> SqlResult<SqlStatementResult> {
        let SelectExecution {
            table_name,
            batches,
            ..
        } = self.execute_query_collect(query)?;
        Ok(SqlStatementResult::Select {
            table_name,
            batches,
        })
    }

    fn execute_query_collect(&self, query: Query) -> SqlResult<SelectExecution> {
        validate_simple_query(&query)?;

        let execution = match query.body.as_ref() {
            SetExpr::Select(select) => self.execute_select(select.as_ref())?,
            other => {
                return Err(Error::InvalidArgumentError(format!(
                    "unsupported query expression: {other:?}"
                )));
            }
        };

        if query.order_by.is_some() {
            // TODO: support ORDER BY clauses
        }

        Ok(execution)
    }

    fn try_parse_simple_aggregates(
        &self,
        table: &SqlTable<P>,
        projection_items: &[SelectItem],
    ) -> SqlResult<Option<Vec<AggregateSpec>>> {
        if projection_items.is_empty() {
            return Ok(None);
        }

        let mut specs: Vec<AggregateSpec> = Vec::with_capacity(projection_items.len());
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

            // TODO: Use enum for function
            let kind = match func_name.as_str() {
                "count" => {
                    if args_slice.len() != 1 {
                        return Err(Error::InvalidArgumentError(
                            "COUNT accepts exactly one argument".into(),
                        ));
                    }
                    match &args_slice[0] {
                        FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => AggregateKind::CountStar,
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(arg_expr)) => {
                            let column = self.resolve_aggregate_column(table, arg_expr)?;
                            AggregateKind::CountField {
                                field_id: column.field_id,
                            }
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
                        if let Some(field_id) = self.parse_count_nulls_case(table, arg_expr)? {
                            AggregateKind::CountNulls { field_id }
                        } else {
                            let column = self.resolve_aggregate_column(table, arg_expr)?;
                            if column.data_type != DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "SUM currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::SumInt64 {
                                field_id: column.field_id,
                            }
                        }
                    } else {
                        let column = self.resolve_aggregate_column(table, arg_expr)?;
                        if column.data_type != DataType::Int64 {
                            return Err(Error::InvalidArgumentError(format!(
                                "{} currently supports only INTEGER columns",
                                func_name.to_uppercase()
                            )));
                        }
                        if func_name == "min" {
                            AggregateKind::MinInt64 {
                                field_id: column.field_id,
                            }
                        } else {
                            AggregateKind::MaxInt64 {
                                field_id: column.field_id,
                            }
                        }
                    }
                }
                _ => return Ok(None),
            };

            specs.push(AggregateSpec { alias, kind });
        }

        if specs.is_empty() {
            return Ok(None);
        }
        Ok(Some(specs))
    }

    fn execute_simple_aggregates(
        &self,
        table: &SqlTable<P>,
        select: &Select,
        table_name: &str,
        aggregates: Vec<AggregateSpec>,
    ) -> SqlResult<SelectExecution> {
        let filter_expr = if let Some(expr) = &select.selection {
            translate_condition(expr, table.schema.as_ref())?
        } else {
            let field_id = table.schema.first_field_id().ok_or_else(|| {
                Error::InvalidArgumentError("table has no selectable columns".into())
            })?;
            full_table_scan_filter(field_id)
        };

        let mut required_fields: Vec<FieldId> = Vec::new();
        for spec in &aggregates {
            if let Some(field_id) = spec.kind.field_id()
                && !required_fields.contains(&field_id)
            {
                required_fields.push(field_id);
            }
        }

        let mut projection_specs: Vec<(FieldId, Projection)> = Vec::new();
        for field_id in &required_fields {
            let column = table
                .schema
                .column_by_field_id(*field_id)
                .ok_or_else(|| Error::InvalidArgumentError("unknown column in aggregate".into()))?;
            let projection = Projection::with_alias(
                LogicalFieldId::for_user(table.table.table_id(), column.field_id),
                column.name.clone(),
            );
            projection_specs.push((*field_id, projection));
        }

        if projection_specs.is_empty()
            && let Some(first_column) = table.schema.columns.first()
        {
            let projection = Projection::with_alias(
                LogicalFieldId::for_user(table.table.table_id(), first_column.field_id),
                first_column.name.clone(),
            );
            projection_specs.push((first_column.field_id, projection));
        }

        if projection_specs.is_empty() {
            return Err(Error::InvalidArgumentError(
                "aggregate queries require at least one column".into(),
            ));
        }

        let mut column_index: HashMap<FieldId, usize> = HashMap::new();
        for (idx, (field_id, _)) in projection_specs.iter().enumerate() {
            column_index.insert(*field_id, idx);
        }

        let scan_projections: Vec<ScanProjection> = projection_specs
            .iter()
            .map(|(_, proj)| ScanProjection::from(proj.clone()))
            .collect();

        let mut states: Vec<AggregateState> = Vec::with_capacity(aggregates.len());
        let count_star_override = if select.selection.is_none() {
            let total_rows = table.total_rows.load(Ordering::SeqCst);
            Some(i64::try_from(total_rows).map_err(|_| {
                Error::InvalidArgumentError("table row count exceeds i64 range".into())
            })?)
        } else {
            None
        };
        for spec in aggregates {
            let accumulator = match spec.kind {
                AggregateKind::CountStar => AggregateAccumulator::CountStar { value: 0 },
                AggregateKind::CountField { field_id } => {
                    let idx = *column_index.get(&field_id).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "internal aggregate error: missing column projection".into(),
                        )
                    })?;
                    AggregateAccumulator::CountColumn {
                        column_index: idx,
                        value: 0,
                    }
                }
                AggregateKind::SumInt64 { field_id } => {
                    let idx = *column_index.get(&field_id).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "internal aggregate error: missing column projection".into(),
                        )
                    })?;
                    AggregateAccumulator::SumInt64 {
                        column_index: idx,
                        value: 0,
                        saw_value: false,
                    }
                }
                AggregateKind::MinInt64 { field_id } => {
                    let idx = *column_index.get(&field_id).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "internal aggregate error: missing column projection".into(),
                        )
                    })?;
                    AggregateAccumulator::MinInt64 {
                        column_index: idx,
                        value: None,
                    }
                }
                AggregateKind::MaxInt64 { field_id } => {
                    let idx = *column_index.get(&field_id).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "internal aggregate error: missing column projection".into(),
                        )
                    })?;
                    AggregateAccumulator::MaxInt64 {
                        column_index: idx,
                        value: None,
                    }
                }
                AggregateKind::CountNulls { field_id: _ } => {
                    let total_rows = count_star_override.ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "SUM(CASE WHEN ... IS NULL ...) with WHERE clauses is not supported yet"
                                .into(),
                        )
                    })?;
                    AggregateAccumulator::CountNulls {
                        non_null_rows: 0,
                        total_rows,
                    }
                }
            };
            states.push(AggregateState {
                alias: spec.alias,
                accumulator,
                override_value: match spec.kind {
                    AggregateKind::CountStar => count_star_override,
                    _ => None,
                },
            });
        }

        let options = ScanStreamOptions {
            include_nulls: true,
        };

        let mut error: Option<Error> = None;
        table
            .table
            .scan_stream(scan_projections, &filter_expr, options, |batch| {
                if error.is_some() {
                    return;
                }
                for state in &mut states {
                    if let Err(err) = state.update(&batch) {
                        error = Some(err);
                        return;
                    }
                }
            })?;

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
        Ok(SelectExecution {
            table_name: table_name.to_string(),
            schema,
            batches: vec![batch],
        })
    }

    fn resolve_aggregate_column<'a>(
        &self,
        table: &'a SqlTable<P>,
        expr: &SqlExpr,
    ) -> SqlResult<&'a SqlColumn> {
        let column_name = match expr {
            SqlExpr::Identifier(ident) => ident.value.clone(),
            SqlExpr::CompoundIdentifier(parts) => {
                if let Some(last) = parts.last() {
                    last.value.clone()
                } else {
                    return Err(Error::InvalidArgumentError(
                        "empty column identifier".into(),
                    ));
                }
            }
            _ => {
                return Err(Error::InvalidArgumentError(
                    "aggregate arguments must be plain column identifiers".into(),
                ));
            }
        };

        table.schema.resolve(&column_name).ok_or_else(|| {
            Error::InvalidArgumentError(format!(
                "unknown column '{}' in aggregate expression",
                column_name
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_start_transaction(
        &self,
        modes: Vec<TransactionMode>,
        begin: bool,
        transaction: Option<BeginTransactionKind>,
        modifier: Option<TransactionModifier>,
        statements: Vec<Statement>,
        exception: Option<Vec<ExceptionWhen>>,
        has_end_keyword: bool,
    ) -> SqlResult<SqlStatementResult> {
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
            // We currently treat START TRANSACTION the same as BEGIN, but keep
            // a guard in case dialects set `begin` to false for unsupported forms.
        }

        let mut active = self.transaction_active.lock().unwrap();
        if *active {
            return Err(Error::InvalidArgumentError(
                "a transaction is already in progress".into(),
            ));
        }
        *active = true;
        Ok(SqlStatementResult::Transaction {
            kind: TransactionKind::Begin,
        })
    }

    fn handle_commit(
        &self,
        chain: bool,
        end: bool,
        modifier: Option<TransactionModifier>,
    ) -> SqlResult<SqlStatementResult> {
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

        let mut active = self.transaction_active.lock().unwrap();
        if !*active {
            return Err(Error::InvalidArgumentError(
                "no transaction is currently in progress".into(),
            ));
        }
        *active = false;
        Ok(SqlStatementResult::Transaction {
            kind: TransactionKind::Commit,
        })
    }

    fn handle_rollback(
        &self,
        chain: bool,
        savepoint: Option<Ident>,
    ) -> SqlResult<SqlStatementResult> {
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

        let mut active = self.transaction_active.lock().unwrap();
        if !*active {
            return Err(Error::InvalidArgumentError(
                "no transaction is currently in progress".into(),
            ));
        }
        *active = false;
        // NOTE: The current engine does not implement MVCC/undo. Rolling back will
        // not revert previously applied changes, but we accept the statement to keep
        // sqllogictest scripts moving forward.
        Ok(SqlStatementResult::Transaction {
            kind: TransactionKind::Rollback,
        })
    }

    fn execute_select(&self, select: &Select) -> SqlResult<SelectExecution> {
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

        let (display_name, canonical_name) = extract_single_table(&select.from)?;
        let table = self.lookup_table(&canonical_name)?;

        if let Some(aggregates) =
            self.try_parse_simple_aggregates(table.as_ref(), &select.projection)?
        {
            return self.execute_simple_aggregates(
                table.as_ref(),
                select,
                &display_name,
                aggregates,
            );
        }

        let projections = build_projections(&select.projection, table.as_ref())?;
        let schema = schema_for_projections(table.as_ref(), &projections)?;
        let filter_expr = if let Some(expr) = &select.selection {
            translate_condition(expr, table.schema.as_ref())?
        } else {
            let field_id = table.schema.first_field_id().ok_or_else(|| {
                Error::InvalidArgumentError("table has no selectable columns".into())
            })?;
            full_table_scan_filter(field_id)
        };

        let mut batches: Vec<RecordBatch> = Vec::new();
        let scan_options = ScanStreamOptions {
            include_nulls: true,
        };

        let scan_result =
            table
                .table
                .scan_stream(&projections, &filter_expr, scan_options, |batch| {
                    batches.push(batch)
                });

        match scan_result {
            Ok(()) => {}
            Err(Error::NotFound) if select.selection.is_none() => {
                tracing::debug!(
                    "scan_stream returned NotFound; synthesizing null batch for full table scan"
                );
                let total_rows = table.total_rows.load(Ordering::SeqCst);
                let fallback_batches = synthesize_null_scan(Arc::clone(&schema), total_rows)?;
                batches = fallback_batches;
            }
            Err(err) => {
                tracing::error!(error = ?err, "scan_stream failed");
                return Err(err);
            }
        }

        if batches.is_empty() && select.selection.is_none() {
            let total_rows = table.total_rows.load(Ordering::SeqCst);
            if total_rows > 0 {
                tracing::debug!(
                    "scan_stream produced no batches for full table scan; synthesizing null batch"
                );
            }
            if !select.projection.is_empty()
                && select
                    .projection
                    .iter()
                    .all(|item| matches!(item, SelectItem::ExprWithAlias { .. }))
            {
                batches.push(RecordBatch::try_new(Arc::clone(&schema), vec![])?);
            } else {
                let fallback_batches = synthesize_null_scan(Arc::clone(&schema), total_rows)?;
                batches = fallback_batches;
            }
        }

        Ok(SelectExecution {
            table_name: display_name,
            schema,
            batches,
        })
    }

    fn lookup_table(&self, canonical_name: &str) -> SqlResult<Arc<SqlTable<P>>> {
        let tables = self.tables.read().unwrap();
        tables
            .get(canonical_name)
            .cloned()
            .ok_or_else(|| Error::InvalidArgumentError(format!("unknown table '{canonical_name}'")))
    }

    fn reserve_table_id(&self) -> SqlResult<TableId> {
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

/// Result of executing a SQL statement.
#[derive(Debug)]
pub enum SqlStatementResult {
    CreateTable {
        table_name: String,
    },
    Insert {
        table_name: String,
        rows_inserted: usize,
    },
    Update {
        table_name: String,
        rows_updated: usize,
    },
    Select {
        table_name: String,
        batches: Vec<RecordBatch>,
    },
    Transaction {
        kind: TransactionKind,
    },
}

#[derive(Debug)]
pub enum TransactionKind {
    Begin,
    Commit,
    Rollback,
}

struct SelectExecution {
    table_name: String,
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
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

#[derive(Clone)]
struct AggregateSpec {
    alias: String,
    kind: AggregateKind,
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
        non_null_rows: i64,
        total_rows: i64,
    },
}

struct AggregateState {
    alias: String,
    accumulator: AggregateAccumulator,
    override_value: Option<i64>,
}

impl AggregateState {
    fn update(&mut self, batch: &RecordBatch) -> SqlResult<()> {
        match &mut self.accumulator {
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
                let array = batch.column(*column_index).as_ref();
                let non_null = array.len() - array.null_count();
                let increment = i64::try_from(non_null).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *value = value.checked_add(increment).ok_or_else(|| {
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
            AggregateAccumulator::CountNulls { non_null_rows, .. } => {
                let rows = i64::try_from(batch.num_rows()).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *non_null_rows = non_null_rows.checked_add(rows).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
            }
        }
        Ok(())
    }

    fn finalize(self) -> SqlResult<(Field, ArrayRef)> {
        let AggregateState {
            alias,
            accumulator,
            override_value,
        } = self;
        match accumulator {
            AggregateAccumulator::CountStar { value } => {
                let mut builder = Int64Builder::with_capacity(1);
                let result = override_value.unwrap_or(value);
                builder.append_value(result);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new(&alias, DataType::Int64, false), array))
            }
            AggregateAccumulator::CountColumn { value, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                builder.append_value(value);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new(&alias, DataType::Int64, false), array))
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
                Ok((Field::new(&alias, DataType::Int64, true), array))
            }
            AggregateAccumulator::MinInt64 { value, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                if let Some(v) = value {
                    builder.append_value(v);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new(&alias, DataType::Int64, true), array))
            }
            AggregateAccumulator::MaxInt64 { value, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                if let Some(v) = value {
                    builder.append_value(v);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new(&alias, DataType::Int64, true), array))
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
                Ok((Field::new(&alias, DataType::Int64, false), array))
            }
        }
    }
}

struct SqlTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: Arc<Table<P>>,
    schema: Arc<SqlSchema>,
    next_row_id: AtomicU64,
    total_rows: AtomicU64,
}

struct SqlSchema {
    columns: Vec<SqlColumn>,
    lookup: HashMap<String, usize>,
}

impl SqlSchema {
    fn resolve(&self, name: &str) -> Option<&SqlColumn> {
        let normalized = name.to_ascii_lowercase();
        self.lookup
            .get(&normalized)
            .and_then(|idx| self.columns.get(*idx))
    }

    fn first_field_id(&self) -> Option<FieldId> {
        self.columns.first().map(|col| col.field_id)
    }

    fn column_by_field_id(&self, field_id: FieldId) -> Option<&SqlColumn> {
        self.columns.iter().find(|col| col.field_id == field_id)
    }
}

#[derive(Clone)]
struct SqlColumn {
    name: String,
    normalized_name: String,
    data_type: DataType,
    nullable: bool,
    field_id: FieldId,
}

impl SqlColumn {
    fn try_from_ast(def: &ColumnDef, field_id: FieldId) -> SqlResult<Self> {
        let data_type = arrow_type_from_sql(&def.data_type)?;
        let mut nullable = true;
        for ColumnOptionDef { option, .. } in &def.options {
            match option {
                ColumnOption::Null => nullable = true,
                ColumnOption::NotNull => nullable = false,
                ColumnOption::Unique { .. } => {
                    // Uniqueness constraints are accepted but not enforced yet.
                }
                ColumnOption::Default(_) => {
                    return Err(Error::InvalidArgumentError(format!(
                        "DEFAULT values are not supported for column '{}'",
                        def.name
                    )));
                }
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported column option {:?} on '{}'",
                        other, def.name
                    )));
                }
            }
        }

        Ok(Self {
            normalized_name: def.name.value.to_ascii_lowercase(),
            name: def.name.value.clone(),
            data_type,
            nullable,
            field_id,
        })
    }

    fn from_field(field: &Field, field_id: FieldId) -> SqlResult<Self> {
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

        Ok(Self {
            normalized_name: field.name().to_ascii_lowercase(),
            name: field.name().to_string(),
            data_type,
            nullable: field.is_nullable(),
            field_id,
        })
    }
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
    if stmt.query.is_some() {
        return Err(Error::InvalidArgumentError(
            "CREATE TABLE AS SELECT must use the CTAS path".into(),
        ));
    }
    Ok(())
}

fn validate_create_table_as(stmt: &sqlparser::ast::CreateTable) -> SqlResult<()> {
    if stmt.query.is_none() {
        return Err(Error::InvalidArgumentError(
            "CREATE TABLE AS SELECT requires a query".into(),
        ));
    }
    Ok(())
}

fn validate_simple_query(query: &Query) -> SqlResult<()> {
    if query.with.is_some()
        || query.limit_clause.is_some()
        || query.fetch.is_some()
        || !query.locks.is_empty()
        || query.for_clause.is_some()
        || query.settings.is_some()
        || query.format_clause.is_some()
        || !query.pipe_operators.is_empty()
    {
        return Err(Error::InvalidArgumentError(
            "complex query features are not supported yet".into(),
        ));
    }
    Ok(())
}

fn group_by_is_empty(expr: &GroupByExpr) -> bool {
    match expr {
        GroupByExpr::All(_) => false,
        GroupByExpr::Expressions(columns, modifiers) => columns.is_empty() && modifiers.is_empty(),
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

fn current_time_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

fn arrow_type_from_sql(data_type: &SqlDataType) -> SqlResult<DataType> {
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

fn resolve_insert_columns(columns: &[Ident], schema: &SqlSchema) -> SqlResult<Vec<usize>> {
    if columns.is_empty() {
        return Ok((0..schema.columns.len()).collect());
    }
    let mut resolved = Vec::with_capacity(columns.len());
    for ident in columns {
        let normalized = ident.value.to_ascii_lowercase();
        let index = schema.lookup.get(&normalized).ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown column '{}'", ident.value))
        })?;
        resolved.push(*index);
    }
    Ok(resolved)
}

fn build_array_for_column(dtype: &DataType, values: &[SqlValue]) -> SqlResult<ArrayRef> {
    match dtype {
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    SqlValue::Null => builder.append_null(),
                    SqlValue::Integer(v) => builder.append_value(*v),
                    SqlValue::Float(v) => builder.append_value(*v as i64),
                    SqlValue::String(_) => {
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
                    SqlValue::Null => builder.append_null(),
                    SqlValue::Integer(v) => builder.append_value(*v as f64),
                    SqlValue::Float(v) => builder.append_value(*v),
                    SqlValue::String(_) => {
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
                    SqlValue::Null => builder.append_null(),
                    SqlValue::Integer(v) => builder.append_value(v.to_string()),
                    SqlValue::Float(v) => builder.append_value(v.to_string()),
                    SqlValue::String(s) => builder.append_value(s),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Date32 => {
            let mut builder = Date32Builder::with_capacity(values.len());
            for value in values {
                match value {
                    SqlValue::Null => builder.append_null(),
                    SqlValue::Integer(days) => {
                        let casted = i32::try_from(*days).map_err(|_| {
                            Error::InvalidArgumentError(
                                "integer literal out of range for DATE column".into(),
                            )
                        })?;
                        builder.append_value(casted);
                    }
                    SqlValue::Float(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert float into DATE column".into(),
                        ));
                    }
                    SqlValue::String(text) => {
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

fn parse_date32_literal(text: &str) -> SqlResult<i32> {
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

fn format_date32(days: i32) -> SqlResult<String> {
    let base = epoch_julian_day() as i64;
    let julian = base + days as i64;
    if !(i32::MIN as i64..=i32::MAX as i64).contains(&julian) {
        return Err(Error::InvalidArgumentError(
            "DATE value out of supported range".into(),
        ));
    }
    let date = Date::from_julian_day(julian as i32).map_err(|err| {
        Error::InvalidArgumentError(format!("DATE value out of supported range: {err}"))
    })?;
    Ok(date.to_string())
}

fn sql_value_from_array(array: &ArrayRef, index: usize) -> SqlResult<SqlValue> {
    if array.is_null(index) {
        return Ok(SqlValue::Null);
    }
    match array.data_type() {
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                Error::InvalidArgumentError("expected Int64 array in INSERT SELECT".into())
            })?;
            Ok(SqlValue::Integer(values.value(index)))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Float64 array in INSERT SELECT".into())
                })?;
            Ok(SqlValue::Float(values.value(index)))
        }
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Utf8 array in INSERT SELECT".into())
                })?;
            Ok(SqlValue::String(values.value(index).to_string()))
        }
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Date32 array in INSERT SELECT".into())
                })?;
            let text = format_date32(values.value(index))?;
            Ok(SqlValue::String(text))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported column type in INSERT SELECT: {other:?}"
        ))),
    }
}
struct RangeSelectRows {
    rows: Vec<Vec<SqlValue>>,
    column_count: usize,
}

struct RangeSpec {
    row_count: usize,
    column_name_lower: String,
    table_alias_lower: Option<String>,
}

enum RangeProjection {
    Column,
    Literal(SqlValue),
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

fn extract_rows_from_range(select: &Select) -> SqlResult<Option<RangeSelectRows>> {
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

    let mut rows: Vec<Vec<SqlValue>> = Vec::with_capacity(spec.row_count);
    for idx in 0..spec.row_count {
        let mut row: Vec<SqlValue> = Vec::with_capacity(projections.len());
        for projection in &projections {
            match projection {
                RangeProjection::Column => row.push(SqlValue::Integer(idx as i64)),
                RangeProjection::Literal(value) => row.push(value.clone()),
            }
        }
        rows.push(row);
    }

    Ok(Some(RangeSelectRows {
        rows,
        column_count: projections.len(),
    }))
}

fn parse_range_spec(select: &Select) -> SqlResult<Option<RangeSpec>> {
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
    alias: &Option<sqlparser::ast::TableAlias>,
) -> SqlResult<Option<RangeSpec>> {
    let func_name = if name.0.len() == 1 {
        match &name.0[0] {
            ObjectNamePart::Identifier(ident) => ident.value.to_ascii_lowercase(),
            _ => return Ok(None),
        }
    } else {
        return Ok(None);
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

    let value = SqlValue::try_from_expr(arg_expr)?;
    let row_count = match value {
        SqlValue::Integer(v) if v >= 0 => v as usize,
        SqlValue::Integer(_) => {
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

fn build_range_projection_expr(expr: &SqlExpr, spec: &RangeSpec) -> SqlResult<RangeProjection> {
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
        other => {
            let value = SqlValue::try_from_expr(other)?;
            Ok(RangeProjection::Literal(value))
        }
    }
}

fn extract_single_table(from: &[TableWithJoins]) -> SqlResult<(String, String)> {
    if from.len() != 1 {
        return Err(Error::InvalidArgumentError(
            "only single-table SELECT statements are supported".into(),
        ));
    }
    let item = &from[0];
    if !item.joins.is_empty() {
        return Err(Error::InvalidArgumentError(
            "JOIN clauses are not supported".into(),
        ));
    }
    match &item.relation {
        TableFactor::Table { name, .. } => canonical_object_name(name),
        _ => Err(Error::InvalidArgumentError(
            "SELECT requires a plain table name in FROM".into(),
        )),
    }
}

fn build_projections<P>(
    projection_items: &[SelectItem],
    table: &SqlTable<P>,
) -> SqlResult<Vec<ScanProjection>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut projections = Vec::new();
    for (idx, item) in projection_items.iter().enumerate() {
        match item {
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(_, _) => {
                for column in &table.schema.columns {
                    projections.push(ScanProjection::from(Projection::with_alias(
                        LogicalFieldId::for_user(table.table.table_id(), column.field_id),
                        column.name.clone(),
                    )));
                }
            }
            SelectItem::UnnamedExpr(SqlExpr::Identifier(ident)) => {
                let column = table.schema.resolve(&ident.value).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unknown column '{}' in projection",
                        ident.value
                    ))
                })?;
                projections.push(ScanProjection::from(Projection::with_alias(
                    LogicalFieldId::for_user(table.table.table_id(), column.field_id),
                    column.name.clone(),
                )));
            }
            SelectItem::ExprWithAlias {
                expr: SqlExpr::Identifier(ident),
                alias,
            } => {
                let column = table.schema.resolve(&ident.value).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unknown column '{}' in projection",
                        ident.value
                    ))
                })?;
                projections.push(ScanProjection::from(Projection::with_alias(
                    LogicalFieldId::for_user(table.table.table_id(), column.field_id),
                    alias.value.clone(),
                )));
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                let scalar = translate_scalar(expr, table.schema.as_ref())?;
                projections.push(ScanProjection::computed(scalar, alias.value.clone()));
            }
            SelectItem::UnnamedExpr(expr) => {
                let scalar = translate_scalar(expr, table.schema.as_ref())?;
                let alias = format!("col{}", idx + 1);
                projections.push(ScanProjection::computed(scalar, alias));
            }
        }
    }
    if projections.is_empty() {
        return Err(Error::InvalidArgumentError(
            "SELECT projection must include at least one column".into(),
        ));
    }
    Ok(projections)
}

fn schema_for_projections<P>(
    table: &SqlTable<P>,
    projections: &[ScanProjection],
) -> SqlResult<Arc<Schema>>
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
                    ScalarExpr::Column(_) | ScalarExpr::Binary { .. } => DataType::Float64,
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

fn synthesize_null_scan(schema: Arc<Schema>, total_rows: u64) -> SqlResult<Vec<RecordBatch>> {
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

fn translate_condition(
    expr: &SqlExpr,
    schema: &SqlSchema,
) -> SqlResult<LlkvExpr<'static, FieldId>> {
    match expr {
        SqlExpr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => Ok(LlkvExpr::And(vec![
                translate_condition(left, schema)?,
                translate_condition(right, schema)?,
            ])),
            BinaryOperator::Or => Ok(LlkvExpr::Or(vec![
                translate_condition(left, schema)?,
                translate_condition(right, schema)?,
            ])),
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Lt
            | BinaryOperator::LtEq
            | BinaryOperator::Gt
            | BinaryOperator::GtEq => translate_comparison(left, op.clone(), right, schema),
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported binary operator in WHERE clause: {other:?}"
            ))),
        },
        SqlExpr::UnaryOp {
            op: UnaryOperator::Not,
            expr,
        } => Ok(LlkvExpr::not(translate_condition(expr, schema)?)),
        SqlExpr::Nested(inner) => translate_condition(inner, schema),
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported WHERE clause: {other:?}"
        ))),
    }
}

fn translate_comparison(
    left: &SqlExpr,
    op: BinaryOperator,
    right: &SqlExpr,
    schema: &SqlSchema,
) -> SqlResult<LlkvExpr<'static, FieldId>> {
    let left_scalar = translate_scalar(left, schema)?;
    let right_scalar = translate_scalar(right, schema)?;
    let compare_op = match op {
        BinaryOperator::Eq => CompareOp::Eq,
        BinaryOperator::NotEq => CompareOp::NotEq,
        BinaryOperator::Lt => CompareOp::Lt,
        BinaryOperator::LtEq => CompareOp::LtEq,
        BinaryOperator::Gt => CompareOp::Gt,
        BinaryOperator::GtEq => CompareOp::GtEq,
        other => {
            return Err(Error::InvalidArgumentError(format!(
                "unsupported comparison operator: {other:?}"
            )));
        }
    };
    Ok(LlkvExpr::Compare {
        left: left_scalar,
        op: compare_op,
        right: right_scalar,
    })
}

fn translate_scalar(expr: &SqlExpr, schema: &SqlSchema) -> SqlResult<ScalarExpr<FieldId>> {
    match expr {
        SqlExpr::Identifier(ident) => {
            let column = schema.resolve(&ident.value).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{}'", ident.value))
            })?;
            Ok(ScalarExpr::column(column.field_id))
        }
        SqlExpr::CompoundIdentifier(idents) => {
            if let Some(last) = idents.last() {
                translate_scalar(&SqlExpr::Identifier(last.clone()), schema)
            } else {
                Err(Error::InvalidArgumentError(
                    "invalid compound identifier".into(),
                ))
            }
        }
        SqlExpr::Value(value) => literal_from_value(value),
        SqlExpr::BinaryOp { left, op, right } => {
            let left_expr = translate_scalar(left, schema)?;
            let right_expr = translate_scalar(right, schema)?;
            let op = match op {
                BinaryOperator::Plus => BinaryOp::Add,
                BinaryOperator::Minus => BinaryOp::Subtract,
                BinaryOperator::Multiply => BinaryOp::Multiply,
                BinaryOperator::Divide => BinaryOp::Divide,
                BinaryOperator::Modulo => BinaryOp::Modulo,
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported scalar binary operator: {other:?}"
                    )));
                }
            };
            Ok(ScalarExpr::binary(left_expr, op, right_expr))
        }
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match translate_scalar(expr, schema)? {
            ScalarExpr::Literal(lit) => match lit {
                Literal::Integer(v) => Ok(ScalarExpr::literal(Literal::Integer(-v))),
                Literal::Float(v) => Ok(ScalarExpr::literal(Literal::Float(-v))),
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
        } => translate_scalar(expr, schema),
        SqlExpr::Nested(inner) => translate_scalar(inner, schema),
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported scalar expression: {other:?}"
        ))),
    }
}

fn literal_from_value(value: &ValueWithSpan) -> SqlResult<ScalarExpr<FieldId>> {
    match &value.value {
        Value::Number(text, _) => {
            if text.contains(['.', 'e', 'E']) {
                let parsed = text.parse::<f64>().map_err(|err| {
                    Error::InvalidArgumentError(format!("invalid float literal: {err}"))
                })?;
                Ok(ScalarExpr::literal(Literal::Float(parsed)))
            } else {
                let parsed = text.parse::<i128>().map_err(|err| {
                    Error::InvalidArgumentError(format!("invalid integer literal: {err}"))
                })?;
                Ok(ScalarExpr::literal(Literal::Integer(parsed)))
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
                Ok(ScalarExpr::literal(Literal::String(text)))
            } else {
                Err(Error::InvalidArgumentError(format!(
                    "unsupported literal: {other:?}"
                )))
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, StringArray};
    use llkv_storage::pager::MemPager;

    #[test]
    fn create_insert_select_roundtrip() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        let result = engine
            .execute("CREATE TABLE people (id INT NOT NULL, name TEXT NOT NULL)")
            .expect("create table");
        assert!(matches!(result[0], SqlStatementResult::CreateTable { .. }));

        let result = engine
            .execute("INSERT INTO people (id, name) VALUES (1, 'alice'), (2, 'bob')")
            .expect("insert rows");
        assert!(matches!(
            result[0],
            SqlStatementResult::Insert {
                rows_inserted: 2,
                ..
            }
        ));

        let result = engine
            .execute("SELECT name FROM people WHERE id = 2")
            .expect("select rows");
        let batches = match &result[0] {
            SqlStatementResult::Select { batches, .. } => batches,
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
    fn select_wildcard_returns_all_columns() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);
        engine
            .execute("CREATE TABLE numbers (id INT NOT NULL, label TEXT)")
            .expect("create table");
        engine
            .execute("INSERT INTO numbers (id, label) VALUES (5, 'five')")
            .expect("insert row");

        let result = engine.execute("SELECT * FROM numbers").expect("select");
        let batches = match &result[0] {
            SqlStatementResult::Select { batches, .. } => batches,
            _ => panic!("expected select result"),
        };
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_columns(), 2);
    }

    #[test]
    fn modulo_predicate_filters_rows() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);
        engine
            .execute("CREATE TABLE seq (val INT NOT NULL)")
            .expect("create table");
        engine
            .execute("INSERT INTO seq (val) VALUES (1), (2), (3), (4)")
            .expect("insert rows");

        let result = engine
            .execute("SELECT val FROM seq WHERE val % 2 <> 0")
            .expect("select odds");

        let batches = match &result[0] {
            SqlStatementResult::Select { batches, .. } => batches,
            _ => panic!("expected select result"),
        };
        assert_eq!(batches.len(), 1);
        let column = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .expect("int column");
        assert_eq!(column.len(), 2);
        assert_eq!(column.value(0), 1);
        assert_eq!(column.value(1), 3);
    }

    #[test]
    fn ctas_select_constant_from_filtered_table() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);
        engine
            .execute("CREATE TABLE integers (i INT NOT NULL)")
            .expect("create source table");
        engine
            .execute("INSERT INTO integers VALUES (1), (2), (3), (4), (5)")
            .expect("seed source data");

        let integers_table = engine.lookup_table("integers").expect("lookup integers");
        let stored_ids = integers_table
            .table
            .store()
            .user_field_ids_for_table(integers_table.table.table_id());
        tracing::info!(?stored_ids, "integers stored field ids after insert");
        let total_rows = integers_table.table.total_rows().expect("table rows");
        tracing::info!(total_rows = total_rows, "integers total_rows after insert");

        let source_check = engine
            .execute("SELECT * FROM integers ORDER BY 1")
            .expect("select integers");
        let source_batches = match &source_check[0] {
            SqlStatementResult::Select { batches, .. } => batches,
            _ => panic!("expected select"),
        };
        assert_eq!(source_batches.len(), 1);
        let source_column = source_batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .expect("int column");
        assert_eq!(source_column.len(), 5);

        engine
            .execute("CREATE TABLE i2 AS SELECT 1 AS i FROM integers WHERE i % 2 <> 0")
            .expect("create ctas table");

        let result = engine
            .execute("SELECT * FROM i2 ORDER BY 1")
            .expect("select from ctas table");
        let batches = match &result[0] {
            SqlStatementResult::Select { batches, .. } => batches,
            _ => panic!("expected select"),
        };
        assert_eq!(batches.len(), 1);
        let column = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .expect("int column");
        assert_eq!(column.len(), 3);
        for idx in 0..column.len() {
            assert_eq!(column.value(idx), 1);
        }
    }
    // TODO: This should more finely check max id to ensure it's incremented per table
    #[test]
    fn table_id_allocation_persists_across_engines() {
        let pager = Arc::new(MemPager::default());
        {
            let engine = SqlEngine::new(Arc::clone(&pager));
            engine
                .execute("CREATE TABLE alpha (id INT NOT NULL)")
                .expect("create table alpha");
        }
        {
            let engine = SqlEngine::new(Arc::clone(&pager));
            engine
                .execute("CREATE TABLE beta (id INT NOT NULL)")
                .expect("create table beta");
        }

        let store = ColumnStore::open(Arc::clone(&pager)).expect("open store");
        let catalog = SysCatalog::new(&store);
        let max_id = catalog
            .max_table_id()
            .expect("max table id")
            .expect("some table id");
        assert!(
            max_id > CATALOG_TABLE_ID,
            "expected user tables to be registered in catalog"
        );
        let next = catalog
            .get_next_table_id()
            .expect("read next table id")
            .expect("next table id value");
        assert!(next > max_id, "next id should advance past existing tables");
    }
}
