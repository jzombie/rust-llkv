use std::collections::HashMap;
use std::ops::Bound;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::SqlResult;
use crate::value::SqlValue;
use arrow::array::{ArrayRef, Float64Builder, Int64Builder, StringBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ColumnStore;
use llkv_column_map::store::{Projection, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::expr::{CompareOp, Expr as LlkvExpr, Filter, Operator, ScalarExpr};
use llkv_expr::literal::Literal;
use llkv_result::Error;
use llkv_storage::pager::Pager;
use llkv_table::table::{ScanProjection, ScanStreamOptions, Table};
use llkv_table::types::{FieldId, TableId};
use llkv_table::{CATALOG_TABLE_ID, SysCatalog};
use llkv_table::{ColMeta, TableMeta};
use simd_r_drive_entry_handle::EntryHandle;
use sqlparser::ast::{
    BinaryOperator, ColumnDef, ColumnOption, ColumnOptionDef, DataType as SqlDataType,
    Expr as SqlExpr, GroupByExpr, Ident, ObjectName, ObjectNamePart, Query, Select, SelectItem,
    SetExpr, Statement, TableFactor, TableObject, TableWithJoins, UnaryOperator, Value,
    ValueWithSpan,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

/// Executes SQL statements against `llkv-table`.
pub struct SqlEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pager: Arc<P>,
    tables: RwLock<HashMap<String, Arc<SqlTable<P>>>>,
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
        }
    }

    /// Execute one or more SQL statements contained in `sql`.
    pub fn execute(&self, sql: &str) -> SqlResult<Vec<SqlStatementResult>> {
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(&dialect, sql)
            .map_err(|err| Error::InvalidArgumentError(err.to_string()))?;

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
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported SQL statement: {other:?}"
            ))),
        }
    }

    fn handle_create_table(
        &self,
        stmt: sqlparser::ast::CreateTable,
    ) -> SqlResult<SqlStatementResult> {
        validate_simple_create_table(&stmt)?;

        let (display_name, canonical_name) = canonical_object_name(&stmt.name)?;
        if display_name.is_empty() {
            return Err(Error::InvalidArgumentError(
                "table name must not be empty".into(),
            ));
        }
        if stmt.columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires at least one column".into(),
            ));
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

        let tables = &mut *self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' already exists",
                display_name
            )));
        }

        let schema = Arc::new(SqlSchema { columns, lookup });
        let sql_table = SqlTable {
            table: Arc::new(table),
            schema,
            next_row_id: AtomicU64::new(0),
        };

        tables.insert(canonical_name, Arc::new(sql_table));

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

        let values = match query.body.as_ref() {
            SetExpr::Values(v) => v,
            _ => {
                return Err(Error::InvalidArgumentError(
                    "INSERT currently supports only VALUES lists".into(),
                ));
            }
        };
        if values.rows.is_empty() {
            return Err(Error::InvalidArgumentError(
                "INSERT VALUES list must contain at least one row".into(),
            ));
        }

        let column_order = resolve_insert_columns(&stmt.columns, table.schema.as_ref())?;
        let row_count = values.rows.len();

        let mut column_values: Vec<Vec<SqlValue>> = table
            .schema
            .columns
            .iter()
            .map(|_| Vec::with_capacity(row_count))
            .collect();

        for row in &values.rows {
            if row.len() != column_order.len() {
                return Err(Error::InvalidArgumentError(format!(
                    "expected {} values in INSERT row, found {}",
                    column_order.len(),
                    row.len()
                )));
            }
            for (idx, expr) in row.iter().enumerate() {
                let target_index = column_order[idx];
                let value = SqlValue::try_from_expr(expr)?;
                column_values[target_index].push(value);
            }
            for (idx, column) in table.schema.columns.iter().enumerate() {
                if column_order.contains(&idx) {
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
            metadata.insert("field_id".to_string(), column.field_id.to_string());
            let field = Field::new(&column.name, column.data_type.clone(), column.nullable)
                .with_metadata(metadata);
            arrays.push(array);
            fields.push(field);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        table.table.append(&batch)?;

        Ok(SqlStatementResult::Insert {
            table_name: display_name,
            rows_inserted: row_count,
        })
    }

    fn handle_query(&self, query: Query) -> SqlResult<SqlStatementResult> {
        validate_simple_query(&query)?;
        let select = match query.body.as_ref() {
            SetExpr::Select(select) => select.as_ref(),
            _ => {
                return Err(Error::InvalidArgumentError(
                    "only simple SELECT statements are supported".into(),
                ));
            }
        };
        self.execute_select(select)
    }

    fn execute_select(&self, select: &Select) -> SqlResult<SqlStatementResult> {
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

        let projections = build_projections(&select.projection, table.as_ref())?;
        let filter_expr = if let Some(expr) = &select.selection {
            translate_condition(expr, table.schema.as_ref())?
        } else {
            let field_id = table.schema.first_field_id().ok_or_else(|| {
                Error::InvalidArgumentError("table has no selectable columns".into())
            })?;
            full_table_scan_filter(field_id)
        };

        let mut batches: Vec<RecordBatch> = Vec::new();
        table
            .table
            .scan_stream(
                &projections,
                &filter_expr,
                ScanStreamOptions::default(),
                |batch| batches.push(batch),
            )
            .map_err(|err| {
                eprintln!("scan_stream failed: {err:?}");
                err
            })?;

        Ok(SqlStatementResult::Select {
            table_name: display_name,
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
pub enum SqlStatementResult {
    CreateTable {
        table_name: String,
    },
    Insert {
        table_name: String,
        rows_inserted: usize,
    },
    Select {
        table_name: String,
        batches: Vec<RecordBatch>,
    },
}

struct SqlTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: Arc<Table<P>>,
    schema: Arc<SqlSchema>,
    next_row_id: AtomicU64,
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
}

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
}

fn validate_simple_create_table(stmt: &sqlparser::ast::CreateTable) -> SqlResult<()> {
    if stmt.query.is_some() || stmt.clone.is_some() || stmt.like.is_some() {
        return Err(Error::InvalidArgumentError(
            "CREATE TABLE AS/LIKE/CLONE is not supported".into(),
        ));
    }
    if !stmt.constraints.is_empty() {
        return Err(Error::InvalidArgumentError(
            "table-level constraints are not supported".into(),
        ));
    }
    Ok(())
}

fn validate_simple_query(query: &Query) -> SqlResult<()> {
    if query.with.is_some()
        || query.order_by.is_some()
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
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported Arrow data type for INSERT: {other:?}"
        ))),
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
    for item in projection_items {
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
            other => {
                return Err(Error::InvalidArgumentError(format!(
                    "unsupported SELECT item: {other:?}"
                )));
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

fn full_table_scan_filter(field_id: FieldId) -> LlkvExpr<'static, FieldId> {
    LlkvExpr::Pred(Filter {
        field_id,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    })
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
