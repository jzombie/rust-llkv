use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use crate::SqlResult;
use crate::SqlValue;

use llkv_dsl::{
    AggregateExpr, AssignmentValue, ColumnAssignment, ColumnSpec, CreateTablePlan,
    CreateTableSource, DeletePlan, DslContext, DslValue, InsertPlan, InsertSource, OrderByPlan,
    OrderSortType, OrderTarget, SelectExecution, SelectPlan, SelectProjection, Session,
    StatementResult, UpdatePlan, extract_rows_from_range,
};
use llkv_expr::literal::Literal;
use llkv_result::Error;
use llkv_storage::pager::Pager;
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

pub struct SqlEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<DslContext<P>>,
    session: Session<P>,
    default_nulls_first: AtomicBool,
}

impl<P> Clone for SqlEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        // Create a new session from the same context
        Self {
            context: Arc::clone(&self.context),
            session: self.context.create_session(),
            default_nulls_first: AtomicBool::new(
                self.default_nulls_first.load(AtomicOrdering::Relaxed),
            ),
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
        let context = Arc::new(DslContext::new(pager));
        let session = context.create_session();
        Self {
            context: Arc::clone(&context),
            session,
            default_nulls_first: AtomicBool::new(false),
        }
    }

    pub(crate) fn context_arc(&self) -> Arc<DslContext<P>> {
        Arc::clone(&self.context)
    }

    pub fn with_context(context: Arc<DslContext<P>>, default_nulls_first: bool) -> Self {
        let session = context.create_session();
        Self {
            context: Arc::clone(&context),
            session,
            default_nulls_first: AtomicBool::new(default_nulls_first),
        }
    }

    #[cfg(test)]
    fn default_nulls_first_for_tests(&self) -> bool {
        self.default_nulls_first.load(AtomicOrdering::Relaxed)
    }

    fn has_active_transaction(&self) -> bool {
        self.session.has_active_transaction()
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
            other => self.execute_statement_non_transactional(other),
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
        self.session.create_table_plan(plan)
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
        self.session.create_table_plan(plan)
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
        self.session
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
        self.session
            .update(plan)
            .map_err(|err| Self::map_table_error(&display_name, err))
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
        let (display_name, _) = extract_single_table(&from_tables)?;

        let filter = selection
            .map(|expr| translate_condition(&expr))
            .transpose()?;

        let plan = DeletePlan {
            table: display_name.clone(),
            filter,
        };
        self.session
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
        // Use session.select() to go through transaction system if active
        match self.session.select(select_plan) {
            Ok(StatementResult::Select { execution, .. }) => Ok(execution),
            Ok(other) => Err(Error::Internal(format!(
                "Expected Select result, got {:?}",
                other
            ))),
            Err(err) => Err(Self::map_table_error(&table_name, err)),
        }
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

        self.session.begin_transaction()
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

        self.session.commit_transaction()
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

        self.session.rollback_transaction()
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
