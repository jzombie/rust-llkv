use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use crate::SqlResult;
use crate::SqlValue;

use llkv_expr::literal::Literal;
use llkv_result::Error;
use llkv_runtime::{
    AggregateExpr, AssignmentValue, ColumnAssignment, ColumnSpec, Context, CreateTablePlan,
    CreateTableSource, DeletePlan, Engine, InsertPlan, InsertSource, OrderByPlan, OrderSortType,
    OrderTarget, PlanStatement, PlanValue, SelectPlan, SelectProjection, Session, StatementResult,
    UpdatePlan, extract_rows_from_range,
};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use sqlparser::ast::{
    Assignment, AssignmentTarget, BeginTransactionKind, BinaryOperator, ColumnOption,
    ColumnOptionDef, DataType as SqlDataType, Delete, ExceptionWhen, Expr as SqlExpr, FromTable,
    FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Ident, LimitClause, ObjectName,
    ObjectNamePart, ObjectType, OrderBy, OrderByExpr, OrderByKind, Query, Select, SelectItem,
    SelectItemQualifiedWildcardKind, Set, SetExpr, Statement, TableFactor, TableObject,
    TableWithJoins, TransactionMode, TransactionModifier, UnaryOperator, UpdateTableFromKind,
    Value, ValueWithSpan,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

pub struct SqlEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    engine: Engine<P>,
    default_nulls_first: AtomicBool,
}

const DROPPED_TABLE_TRANSACTION_ERR: &str = "another transaction has dropped this table";

impl<P> Clone for SqlEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        tracing::warn!(
            "[SQL_ENGINE] SqlEngine::clone() called - will create new Engine with new session!"
        );
        // Create a new session from the same context
        Self {
            engine: self.engine.clone(),
            default_nulls_first: AtomicBool::new(
                self.default_nulls_first.load(AtomicOrdering::Relaxed),
            ),
        }
    }
}

#[allow(dead_code)]
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

    // `statement_table_name` is provided by llkv-runtime; use it to avoid
    // duplicating plan-level logic here.

    fn execute_plan_statement(&self, statement: PlanStatement) -> SqlResult<StatementResult<P>> {
        let table = llkv_runtime::statement_table_name(&statement).map(str::to_string);
        self.engine.execute_statement(statement).map_err(|err| {
            if let Some(table_name) = table {
                Self::map_table_error(&table_name, err)
            } else {
                err
            }
        })
    }

    pub fn new(pager: Arc<P>) -> Self {
        let engine = Engine::new(pager);
        Self {
            engine,
            default_nulls_first: AtomicBool::new(false),
        }
    }

    pub(crate) fn context_arc(&self) -> Arc<Context<P>> {
        self.engine.context()
    }

    pub fn with_context(context: Arc<Context<P>>, default_nulls_first: bool) -> Self {
        Self {
            engine: Engine::from_context(context),
            default_nulls_first: AtomicBool::new(default_nulls_first),
        }
    }

    #[cfg(test)]
    fn default_nulls_first_for_tests(&self) -> bool {
        self.default_nulls_first.load(AtomicOrdering::Relaxed)
    }

    fn has_active_transaction(&self) -> bool {
        self.engine.session().has_active_transaction()
    }

    /// Get a reference to the underlying session (for advanced use like error handling in test harnesses).
    pub fn session(&self) -> &Session<P> {
        self.engine.session()
    }

    pub fn execute(&self, sql: &str) -> SqlResult<Vec<StatementResult<P>>> {
        tracing::trace!("DEBUG SQL execute: {}", sql);
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(&dialect, sql)
            .map_err(|err| Error::InvalidArgumentError(format!("failed to parse SQL: {err}")))?;
        tracing::trace!("DEBUG SQL execute: parsed {} statements", statements.len());

        let mut results = Vec::with_capacity(statements.len());
        for (i, statement) in statements.iter().enumerate() {
            tracing::trace!("DEBUG SQL execute: processing statement {}", i);
            results.push(self.execute_statement(statement.clone())?);
            tracing::trace!("DEBUG SQL execute: statement {} completed", i);
        }
        tracing::trace!("DEBUG SQL execute completed successfully");
        Ok(results)
    }

    fn execute_statement(&self, statement: Statement) -> SqlResult<StatementResult<P>> {
        tracing::trace!(
            "DEBUG SQL execute_statement: {:?}",
            match &statement {
                Statement::Insert(insert) =>
                    format!("Insert(table={:?})", Self::table_name_from_insert(insert)),
                Statement::Query(_) => "Query".to_string(),
                Statement::StartTransaction { .. } => "StartTransaction".to_string(),
                Statement::Commit { .. } => "Commit".to_string(),
                Statement::Rollback { .. } => "Rollback".to_string(),
                Statement::CreateTable(_) => "CreateTable".to_string(),
                Statement::Update { .. } => "Update".to_string(),
                Statement::Delete(_) => "Delete".to_string(),
                other => format!("Other({:?})", other),
            }
        );
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
            other => self.execute_statement_non_transactional(other),
        }
    }

    fn execute_statement_non_transactional(
        &self,
        statement: Statement,
    ) -> SqlResult<StatementResult<P>> {
        tracing::trace!("DEBUG SQL execute_statement_non_transactional called");
        match statement {
            Statement::CreateTable(stmt) => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: CreateTable");
                self.handle_create_table(stmt)
            }
            Statement::Insert(stmt) => {
                let table_name =
                    Self::table_name_from_insert(&stmt).unwrap_or_else(|_| "unknown".to_string());
                tracing::trace!(
                    "DEBUG SQL execute_statement_non_transactional: Insert(table={})",
                    table_name
                );
                self.handle_insert(stmt)
            }
            Statement::Query(query) => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: Query");
                self.handle_query(*query)
            }
            Statement::Update {
                table,
                assignments,
                from,
                selection,
                returning,
                ..
            } => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: Update");
                self.handle_update(table, assignments, from, selection, returning)
            }
            Statement::Delete(delete) => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: Delete");
                self.handle_delete(delete)
            }
            Statement::Drop {
                object_type,
                if_exists,
                names,
                cascade,
                restrict,
                purge,
                temporary,
                ..
            } => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: Drop");
                self.handle_drop(
                    object_type,
                    if_exists,
                    names,
                    cascade,
                    restrict,
                    purge,
                    temporary,
                )
            }
            Statement::Set(set_stmt) => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: Set");
                self.handle_set(set_stmt)
            }
            Statement::Pragma { name, value, is_eq } => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: Pragma");
                self.handle_pragma(name, value, is_eq)
            }
            other => {
                tracing::trace!(
                    "DEBUG SQL execute_statement_non_transactional: Other({:?})",
                    other
                );
                Err(Error::InvalidArgumentError(format!(
                    "unsupported SQL statement: {other:?}"
                )))
            }
        }
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

    #[allow(dead_code)]
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

    fn is_table_marked_dropped(&self, table_name: &str) -> SqlResult<bool> {
        let canonical = table_name.to_ascii_lowercase();
        Ok(self.engine.context().is_table_marked_dropped(&canonical))
    }

    fn handle_create_table(
        &self,
        mut stmt: sqlparser::ast::CreateTable,
    ) -> SqlResult<StatementResult<P>> {
        validate_create_table_common(&stmt)?;

        let (display_name, canonical_name) = canonical_object_name(&stmt.name)?;
        tracing::trace!(
            "\n=== HANDLE_CREATE_TABLE: table='{}' columns={} ===",
            display_name,
            stmt.columns.len()
        );
        if display_name.is_empty() {
            return Err(Error::InvalidArgumentError(
                "table name must not be empty".into(),
            ));
        }

        if let Some(query) = stmt.query.take() {
            validate_create_table_as(&stmt)?;
            if let Some(result) = self.try_handle_range_ctas(
                &display_name,
                &canonical_name,
                &query,
                stmt.if_not_exists,
                stmt.or_replace,
            )? {
                return Ok(result);
            }
            return self.handle_create_table_as(
                display_name,
                canonical_name,
                *query,
                stmt.if_not_exists,
                stmt.or_replace,
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
            let is_nullable = column_def
                .options
                .iter()
                .all(|opt| !matches!(opt.option, ColumnOption::NotNull));

            let is_primary_key = column_def.options.iter().any(|opt| {
                matches!(
                    opt.option,
                    ColumnOption::Unique {
                        is_primary: true,
                        characteristics: _
                    }
                )
            });

            tracing::trace!(
                "DEBUG CREATE TABLE column '{}' is_primary_key={}",
                column_def.name.value,
                is_primary_key
            );

            let mut column = ColumnSpec::new(
                column_def.name.value.clone(),
                arrow_type_from_sql(&column_def.data_type)?,
                is_nullable,
            );
            tracing::trace!(
                "DEBUG ColumnSpec after new(): primary_key={}",
                column.primary_key
            );

            column = column.with_primary_key(is_primary_key);
            tracing::trace!(
                "DEBUG ColumnSpec after with_primary_key({}): primary_key={}",
                is_primary_key,
                column.primary_key
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
            or_replace: stmt.or_replace,
            columns,
            source: None,
        };
        self.execute_plan_statement(PlanStatement::CreateTable(plan))
    }

    fn try_handle_range_ctas(
        &self,
        display_name: &str,
        _canonical_name: &str,
        query: &Query,
        if_not_exists: bool,
        or_replace: bool,
    ) -> SqlResult<Option<StatementResult<P>>> {
        let select = match query.body.as_ref() {
            SetExpr::Select(select) => select,
            _ => return Ok(None),
        };
        if select.from.len() != 1 {
            return Ok(None);
        }
        let table_with_joins = &select.from[0];
        if !table_with_joins.joins.is_empty() {
            return Ok(None);
        }
        let (range_size, range_alias) = match &table_with_joins.relation {
            TableFactor::Table {
                name,
                args: Some(args),
                alias,
                ..
            } => {
                let func_name = name.to_string().to_ascii_lowercase();
                if func_name != "range" {
                    return Ok(None);
                }
                if args.args.len() != 1 {
                    return Err(Error::InvalidArgumentError(
                        "range table function expects a single argument".into(),
                    ));
                }
                let size_expr = &args.args[0];
                let range_size = match size_expr {
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Value(value))) => {
                        match &value.value {
                            Value::Number(raw, _) => raw.parse::<i64>().map_err(|e| {
                                Error::InvalidArgumentError(format!(
                                    "invalid range size literal {}: {}",
                                    raw, e
                                ))
                            })?,
                            other => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "unsupported range size value: {:?}",
                                    other
                                )));
                            }
                        }
                    }
                    _ => {
                        return Err(Error::InvalidArgumentError(
                            "unsupported range argument".into(),
                        ));
                    }
                };
                (range_size, alias.as_ref().map(|a| a.name.value.clone()))
            }
            _ => return Ok(None),
        };

        if range_size < 0 {
            return Err(Error::InvalidArgumentError(
                "range size must be non-negative".into(),
            ));
        }

        if select.projection.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE AS SELECT requires at least one projected column".into(),
            ));
        }

        let mut column_specs = Vec::with_capacity(select.projection.len());
        let mut column_names = Vec::with_capacity(select.projection.len());
        let mut row_template = Vec::with_capacity(select.projection.len());
        for item in &select.projection {
            match item {
                SelectItem::ExprWithAlias { expr, alias } => {
                    let (value, data_type) = match expr {
                        SqlExpr::Value(value_with_span) => match &value_with_span.value {
                            Value::Number(raw, _) => {
                                let parsed = raw.parse::<i64>().map_err(|e| {
                                    Error::InvalidArgumentError(format!(
                                        "invalid numeric literal {}: {}",
                                        raw, e
                                    ))
                                })?;
                                (
                                    PlanValue::Integer(parsed),
                                    arrow::datatypes::DataType::Int64,
                                )
                            }
                            Value::SingleQuotedString(s) => (
                                PlanValue::String(s.clone()),
                                arrow::datatypes::DataType::Utf8,
                            ),
                            other => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "unsupported SELECT expression in range CTAS: {:?}",
                                    other
                                )));
                            }
                        },
                        SqlExpr::Identifier(ident) => {
                            let ident_lower = ident.value.to_ascii_lowercase();
                            if range_alias
                                .as_ref()
                                .map(|a| a.eq_ignore_ascii_case(&ident_lower))
                                .unwrap_or(false)
                                || ident_lower == "range"
                            {
                                return Err(Error::InvalidArgumentError(
                                    "range() table function columns are not supported yet".into(),
                                ));
                            }
                            return Err(Error::InvalidArgumentError(format!(
                                "unsupported identifier '{}' in range CTAS projection",
                                ident.value
                            )));
                        }
                        other => {
                            return Err(Error::InvalidArgumentError(format!(
                                "unsupported SELECT expression in range CTAS: {:?}",
                                other
                            )));
                        }
                    };
                    let column_name = alias.value.clone();
                    column_specs.push(ColumnSpec::new(column_name.clone(), data_type, true));
                    column_names.push(column_name);
                    row_template.push(value);
                }
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported projection {:?} in range CTAS",
                        other
                    )));
                }
            }
        }

        let plan = CreateTablePlan {
            name: display_name.to_string(),
            if_not_exists,
            or_replace,
            columns: column_specs,
            source: None,
        };
        let create_result = self.execute_plan_statement(PlanStatement::CreateTable(plan))?;

        let row_count = range_size
            .try_into()
            .map_err(|_| Error::InvalidArgumentError("range size exceeds usize".into()))?;
        if row_count > 0 {
            let rows = vec![row_template; row_count];
            let insert_plan = InsertPlan {
                table: display_name.to_string(),
                columns: column_names,
                source: InsertSource::Rows(rows),
            };
            self.execute_plan_statement(PlanStatement::Insert(insert_plan))?;
        }

        Ok(Some(create_result))
    }

    fn handle_create_table_as(
        &self,
        display_name: String,
        _canonical_name: String,
        query: Query,
        if_not_exists: bool,
        or_replace: bool,
    ) -> SqlResult<StatementResult<P>> {
        let select_plan = self.build_select_plan(query)?;

        if select_plan.projections.is_empty() && select_plan.aggregates.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE AS SELECT requires at least one projected column".into(),
            ));
        }

        let plan = CreateTablePlan {
            name: display_name,
            if_not_exists,
            or_replace,
            columns: Vec::new(),
            source: Some(CreateTableSource::Select {
                plan: Box::new(select_plan),
            }),
        };
        self.execute_plan_statement(PlanStatement::CreateTable(plan))
    }

    fn handle_insert(&self, stmt: sqlparser::ast::Insert) -> SqlResult<StatementResult<P>> {
        let table_name_debug =
            Self::table_name_from_insert(&stmt).unwrap_or_else(|_| "unknown".to_string());
        tracing::trace!(
            "DEBUG SQL handle_insert called for table={}",
            table_name_debug
        );
        if !self.engine.session().has_active_transaction()
            && self.is_table_marked_dropped(&table_name_debug)?
        {
            return Err(Error::TransactionContextError(
                DROPPED_TABLE_TRANSACTION_ERR.into(),
            ));
        }
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
                        .map(|row| row.into_iter().map(PlanValue::from).collect())
                        .collect(),
                )
            }
            SetExpr::Select(select) => {
                if let Some(rows) = extract_constant_select_rows(select.as_ref())? {
                    InsertSource::Rows(rows)
                } else if let Some(range_rows) = extract_rows_from_range(select.as_ref())? {
                    InsertSource::Rows(range_rows.into_rows())
                } else {
                    let select_plan = self.build_select_plan((**source_expr).clone())?;
                    InsertSource::Select {
                        plan: Box::new(select_plan),
                    }
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
        tracing::trace!(
            "DEBUG SQL handle_insert: about to execute insert for table={}",
            display_name
        );
        self.execute_plan_statement(PlanStatement::Insert(plan))
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

        let (display_name, canonical_name) = extract_single_table(std::slice::from_ref(&table))?;

        if !self.engine.session().has_active_transaction()
            && self
                .engine
                .context()
                .is_table_marked_dropped(&canonical_name)
        {
            return Err(Error::TransactionContextError(
                DROPPED_TABLE_TRANSACTION_ERR.into(),
            ));
        }

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
                Ok(literal) => AssignmentValue::Literal(PlanValue::from(literal)),
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
        self.execute_plan_statement(PlanStatement::Update(plan))
    }

    #[allow(clippy::collapsible_if)]
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
        let (display_name, canonical_name) = extract_single_table(&from_tables)?;

        if !self.engine.session().has_active_transaction()
            && self
                .engine
                .context()
                .is_table_marked_dropped(&canonical_name)
        {
            return Err(Error::TransactionContextError(
                DROPPED_TABLE_TRANSACTION_ERR.into(),
            ));
        }

        let filter = selection
            .map(|expr| translate_condition(&expr))
            .transpose()?;

        let plan = DeletePlan {
            table: display_name.clone(),
            filter,
        };
        self.execute_plan_statement(PlanStatement::Delete(plan))
    }

    #[allow(clippy::too_many_arguments)] // TODO: Consider refactor
    fn handle_drop(
        &self,
        object_type: ObjectType,
        if_exists: bool,
        names: Vec<ObjectName>,
        cascade: bool,
        restrict: bool,
        purge: bool,
        temporary: bool,
    ) -> SqlResult<StatementResult<P>> {
        if cascade || restrict || purge || temporary {
            return Err(Error::InvalidArgumentError(
                "DROP TABLE cascade/restrict/purge/temporary options are not supported".into(),
            ));
        }

        if object_type != ObjectType::Table {
            return Err(Error::InvalidArgumentError(
                "only DROP TABLE is supported".into(),
            ));
        }

        let ctx = self.engine.context();
        for name in names {
            let table_name = Self::object_name_to_string(&name)?;
            ctx.drop_table_immediate(&table_name, if_exists)
                .map_err(|err| Self::map_table_error(&table_name, err))?;
        }

        Ok(StatementResult::NoOp)
    }

    fn handle_query(&self, query: Query) -> SqlResult<StatementResult<P>> {
        let select_plan = self.build_select_plan(query)?;
        self.execute_plan_statement(PlanStatement::Select(select_plan))
    }

    fn build_select_plan(&self, query: Query) -> SqlResult<SelectPlan> {
        if self.engine.session().has_active_transaction() && self.engine.session().is_aborted() {
            return Err(Error::TransactionContextError(
                "TransactionContext Error: transaction is aborted".into(),
            ));
        }

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
        Ok(select_plan)
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

        self.execute_plan_statement(PlanStatement::BeginTransaction)
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

        self.execute_plan_statement(PlanStatement::CommitTransaction)
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

        self.execute_plan_statement(PlanStatement::RollbackTransaction)
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

                let variable_name_raw = variable.to_string();
                let variable_name = variable_name_raw.to_ascii_lowercase();

                match variable_name.as_str() {
                    "default_null_order" => {
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
                    "immediate_transaction_mode" => {
                        if values.len() != 1 {
                            return Err(Error::InvalidArgumentError(
                                "SET immediate_transaction_mode expects exactly one value".into(),
                            ));
                        }
                        let normalized = values[0].to_string().to_ascii_lowercase();
                        let enabled = match normalized.as_str() {
                            "true" | "on" | "1" => true,
                            "false" | "off" | "0" => false,
                            _ => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "unsupported value for SET immediate_transaction_mode: {}",
                                    values[0]
                                )));
                            }
                        };
                        if !enabled {
                            tracing::warn!(
                                "SET immediate_transaction_mode=false has no effect; continuing with auto mode"
                            );
                        }
                        Ok(StatementResult::NoOp)
                    }
                    _ => Err(Error::InvalidArgumentError(format!(
                        "unsupported SET variable: {variable_name_raw}"
                    ))),
                }
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
    if stmt.or_replace && stmt.if_not_exists {
        return Err(Error::InvalidArgumentError(
            "CREATE TABLE cannot combine OR REPLACE with IF NOT EXISTS".into(),
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

/// Try to parse a function as an aggregate call for use in scalar expressions
/// Check if a scalar expression contains any aggregate functions
#[allow(dead_code)] // Utility function for future use
fn expr_contains_aggregate(expr: &llkv_expr::expr::ScalarExpr<String>) -> bool {
    match expr {
        llkv_expr::expr::ScalarExpr::Aggregate(_) => true,
        llkv_expr::expr::ScalarExpr::Binary { left, right, .. } => {
            expr_contains_aggregate(left) || expr_contains_aggregate(right)
        }
        llkv_expr::expr::ScalarExpr::Column(_) | llkv_expr::expr::ScalarExpr::Literal(_) => false,
    }
}

fn try_parse_aggregate_function(
    func: &sqlparser::ast::Function,
) -> SqlResult<Option<llkv_expr::expr::AggregateCall<String>>> {
    use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments, ObjectNamePart};

    if func.uses_odbc_syntax {
        return Ok(None);
    }
    if !matches!(func.parameters, FunctionArguments::None) {
        return Ok(None);
    }
    if func.filter.is_some()
        || func.null_treatment.is_some()
        || func.over.is_some()
        || !func.within_group.is_empty()
    {
        return Ok(None);
    }

    let func_name = if func.name.0.len() == 1 {
        match &func.name.0[0] {
            ObjectNamePart::Identifier(ident) => ident.value.to_ascii_lowercase(),
            _ => return Ok(None),
        }
    } else {
        return Ok(None);
    };

    let args_slice: &[FunctionArg] = match &func.args {
        FunctionArguments::List(list) => {
            if list.duplicate_treatment.is_some() || !list.clauses.is_empty() {
                return Ok(None);
            }
            &list.args
        }
        FunctionArguments::None => &[],
        FunctionArguments::Subquery(_) => return Ok(None),
    };

    let agg_call = match func_name.as_str() {
        "count" => {
            if args_slice.len() != 1 {
                return Err(Error::InvalidArgumentError(
                    "COUNT accepts exactly one argument".into(),
                ));
            }
            match &args_slice[0] {
                FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {
                    llkv_expr::expr::AggregateCall::CountStar
                }
                FunctionArg::Unnamed(FunctionArgExpr::Expr(arg_expr)) => {
                    let column = resolve_column_name(arg_expr)?;
                    llkv_expr::expr::AggregateCall::Count(column)
                }
                _ => {
                    return Err(Error::InvalidArgumentError(
                        "unsupported COUNT argument".into(),
                    ));
                }
            }
        }
        "sum" => {
            if args_slice.len() != 1 {
                return Err(Error::InvalidArgumentError(
                    "SUM accepts exactly one argument".into(),
                ));
            }
            let arg_expr = match &args_slice[0] {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
                _ => {
                    return Err(Error::InvalidArgumentError(
                        "SUM requires a column argument".into(),
                    ));
                }
            };

            // Check for COUNT(CASE ...) pattern
            if let Some(column) = parse_count_nulls_case(arg_expr)? {
                llkv_expr::expr::AggregateCall::CountNulls(column)
            } else {
                let column = resolve_column_name(arg_expr)?;
                llkv_expr::expr::AggregateCall::Sum(column)
            }
        }
        "min" => {
            if args_slice.len() != 1 {
                return Err(Error::InvalidArgumentError(
                    "MIN accepts exactly one argument".into(),
                ));
            }
            let arg_expr = match &args_slice[0] {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
                _ => {
                    return Err(Error::InvalidArgumentError(
                        "MIN requires a column argument".into(),
                    ));
                }
            };
            let column = resolve_column_name(arg_expr)?;
            llkv_expr::expr::AggregateCall::Min(column)
        }
        "max" => {
            if args_slice.len() != 1 {
                return Err(Error::InvalidArgumentError(
                    "MAX accepts exactly one argument".into(),
                ));
            }
            let arg_expr = match &args_slice[0] {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
                _ => {
                    return Err(Error::InvalidArgumentError(
                        "MAX requires a column argument".into(),
                    ));
                }
            };
            let column = resolve_column_name(arg_expr)?;
            llkv_expr::expr::AggregateCall::Max(column)
        }
        _ => return Ok(None),
    };

    Ok(Some(agg_call))
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
        SqlExpr::Function(func) => {
            // Try to parse as an aggregate function
            if let Some(agg_call) = try_parse_aggregate_function(func)? {
                Ok(llkv_expr::expr::ScalarExpr::aggregate(agg_call))
            } else {
                Err(Error::InvalidArgumentError(format!(
                    "unsupported function in scalar expression: {:?}",
                    func.name
                )))
            }
        }
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

fn extract_constant_select_rows(select: &Select) -> SqlResult<Option<Vec<Vec<PlanValue>>>> {
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

    let mut row: Vec<PlanValue> = Vec::with_capacity(select.projection.len());
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
        row.push(PlanValue::from(value));
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
