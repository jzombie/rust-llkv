use std::collections::{HashMap, HashSet};
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering as AtomicOrdering},
};

use crate::SqlResult;
use crate::SqlValue;
use arrow::record_batch::RecordBatch;

use llkv_executor::SelectExecution;
use llkv_expr::literal::Literal;
use llkv_plan::TransformFrame;
use llkv_plan::validation::{
    ensure_known_columns_case_insensitive, ensure_non_empty, ensure_unique_case_insensitive,
};
use llkv_result::Error;
use llkv_runtime::TEMPORARY_NAMESPACE_ID;
use llkv_runtime::{
    AggregateExpr, AssignmentValue, ColumnAssignment, CreateIndexPlan, CreateTablePlan,
    CreateTableSource, DeletePlan, ForeignKeyAction, ForeignKeySpec, IndexColumnPlan, InsertPlan,
    InsertSource, MultiColumnUniqueSpec, OrderByPlan, OrderSortType, OrderTarget, PlanColumnSpec,
    PlanStatement, PlanValue, RenameTablePlan, RuntimeContext, RuntimeEngine, RuntimeSession,
    RuntimeStatementResult, SelectPlan, SelectProjection, TruncatePlan, UpdatePlan,
    extract_rows_from_range,
};
use llkv_storage::pager::Pager;
use llkv_table::CatalogDdl;
use llkv_table::catalog::{IdentifierContext, IdentifierResolver};
use regex::Regex;
use simd_r_drive_entry_handle::EntryHandle;
use sqlparser::ast::{
    AlterColumnOperation, AlterTableOperation, Assignment, AssignmentTarget, BeginTransactionKind,
    BinaryOperator, ColumnOption, ColumnOptionDef, ConstraintCharacteristics,
    DataType as SqlDataType, Delete, Distinct, ExceptionWhen, Expr as SqlExpr, FromTable,
    FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Ident, JoinConstraint,
    JoinOperator, LimitClause, NullsDistinctOption, ObjectName, ObjectNamePart, ObjectType,
    OrderBy, OrderByKind, Query, ReferentialAction, SchemaName, Select, SelectItem,
    SelectItemQualifiedWildcardKind, Set, SetExpr, SqlOption, Statement, TableConstraint,
    TableFactor, TableObject, TableWithJoins, TransactionMode, TransactionModifier, UnaryOperator,
    UpdateTableFromKind, Value, ValueWithSpan,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::Span;

// TODO: Extract to constants.rs
// TODO: Rename to SQL_PARSER_RECURSION_LIMIT
/// Maximum recursion depth for SQL parser.
///
/// The default in sqlparser is 50, which can be exceeded by deeply nested queries
/// (e.g., SQLite test suite). This value allows for more complex expressions while
/// still preventing stack overflows.
const PARSER_RECURSION_LIMIT: usize = 200;

/// SQL execution engine built on top of the LLKV runtime.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use arrow::array::StringArray;
/// use llkv_sql::{RuntimeStatementResult, SqlEngine};
/// use llkv_storage::pager::MemPager;
///
/// let engine = SqlEngine::new(Arc::new(MemPager::default()));
///
/// let setup = r#"
///     CREATE TABLE users (id INT PRIMARY KEY, name TEXT);
///     INSERT INTO users (id, name) VALUES (1, 'Ada');
/// "#;
///
/// let results = engine.execute(setup).unwrap();
/// assert_eq!(results.len(), 2);
///
/// assert!(matches!(
///     results[0],
///     RuntimeStatementResult::CreateTable { ref table_name } if table_name == "users"
/// ));
/// assert!(matches!(
///     results[1],
///     RuntimeStatementResult::Insert { rows_inserted, .. } if rows_inserted == 1
/// ));
///
/// let batches = engine.sql("SELECT id, name FROM users ORDER BY id;").unwrap();
/// assert_eq!(batches.len(), 1);
///
/// let batch = &batches[0];
/// assert_eq!(batch.num_rows(), 1);
/// assert_eq!(batch.schema().field(1).name(), "name");
///
/// let names = batch
///     .column(1)
///     .as_any()
///     .downcast_ref::<StringArray>()
///     .unwrap();
///
/// assert_eq!(names.value(0), "Ada");
/// ```
pub struct SqlEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    engine: RuntimeEngine<P>,
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

    fn execute_plan_statement(
        &self,
        statement: PlanStatement,
    ) -> SqlResult<RuntimeStatementResult<P>> {
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
        let engine = RuntimeEngine::new(pager);
        Self {
            engine,
            default_nulls_first: AtomicBool::new(false),
        }
    }

    /// Preprocess SQL to handle qualified names in EXCLUDE clauses
    /// Converts EXCLUDE (schema.table.col) to EXCLUDE ("schema.table.col")
    /// Preprocess CREATE TYPE to CREATE DOMAIN for sqlparser compatibility.
    ///
    /// DuckDB uses `CREATE TYPE name AS basetype` for type aliases, but sqlparser
    /// only supports the SQL standard `CREATE DOMAIN name AS basetype` syntax.
    /// This method converts the DuckDB syntax to the standard syntax.
    fn preprocess_create_type_syntax(sql: &str) -> String {
        static CREATE_TYPE_REGEX: OnceLock<Regex> = OnceLock::new();
        static DROP_TYPE_REGEX: OnceLock<Regex> = OnceLock::new();

        // Match: CREATE TYPE name AS datatype
        let create_re = CREATE_TYPE_REGEX.get_or_init(|| {
            Regex::new(r"(?i)\bCREATE\s+TYPE\s+").expect("valid CREATE TYPE regex")
        });

        // Match: DROP TYPE [IF EXISTS] name
        let drop_re = DROP_TYPE_REGEX
            .get_or_init(|| Regex::new(r"(?i)\bDROP\s+TYPE\s+").expect("valid DROP TYPE regex"));

        // First replace CREATE TYPE with CREATE DOMAIN
        let sql = create_re.replace_all(sql, "CREATE DOMAIN ").to_string();

        // Then replace DROP TYPE with DROP DOMAIN
        drop_re.replace_all(&sql, "DROP DOMAIN ").to_string()
    }

    fn preprocess_exclude_syntax(sql: &str) -> String {
        static EXCLUDE_REGEX: OnceLock<Regex> = OnceLock::new();

        // Pattern to match EXCLUDE followed by qualified identifiers
        // Matches: EXCLUDE (identifier.identifier.identifier)
        let re = EXCLUDE_REGEX.get_or_init(|| {
            Regex::new(
                r"(?i)EXCLUDE\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\s*\)",
            )
            .expect("valid EXCLUDE qualifier regex")
        });

        re.replace_all(sql, |caps: &regex::Captures| {
            let qualified_name = &caps[1];
            format!("EXCLUDE (\"{}\")", qualified_name)
        })
        .to_string()
    }

    /// Preprocess SQL to remove trailing commas in VALUES clauses.
    /// DuckDB allows trailing commas like VALUES ('v2',) but sqlparser does not.
    fn preprocess_trailing_commas_in_values(sql: &str) -> String {
        static TRAILING_COMMA_REGEX: OnceLock<Regex> = OnceLock::new();

        // Pattern to match trailing comma before closing paren in VALUES
        // Matches: , followed by optional whitespace and )
        let re = TRAILING_COMMA_REGEX
            .get_or_init(|| Regex::new(r",(\s*)\)").expect("valid trailing comma regex"));

        re.replace_all(sql, "$1)").to_string()
    }

    pub(crate) fn context_arc(&self) -> Arc<RuntimeContext<P>> {
        self.engine.context()
    }

    pub fn with_context(context: Arc<RuntimeContext<P>>, default_nulls_first: bool) -> Self {
        Self {
            engine: RuntimeEngine::from_context(context),
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
    pub fn session(&self) -> &RuntimeSession<P> {
        self.engine.session()
    }

    /// Execute one or more SQL statements and return their raw [`RuntimeStatementResult`]s.
    ///
    /// This method is the general-purpose entry point for running SQL against the engine when
    /// you need to mix statement types (e.g. `CREATE TABLE`, `INSERT`, `UPDATE`, `SELECT`) or
    /// when you care about per-statement status information. Statements are executed in the order
    /// they appear in the input string, and the results vector mirrors that ordering.
    ///
    /// For ad-hoc read queries where you only care about the resulting Arrow [`RecordBatch`]es,
    /// prefer [`SqlEngine::sql`], which enforces a single `SELECT` statement and collects its
    /// output for you. `execute` remains the right tool for schema migrations, transactional
    /// scripts, or workflows that need to inspect the specific runtime response for each
    /// statement.
    pub fn execute(&self, sql: &str) -> SqlResult<Vec<RuntimeStatementResult<P>>> {
        tracing::trace!("DEBUG SQL execute: {}", sql);

        // Preprocess SQL to handle CREATE TYPE / DROP TYPE (DuckDB syntax)
        let processed_sql = Self::preprocess_create_type_syntax(sql);

        // Preprocess SQL to handle qualified names in EXCLUDE
        // Replace EXCLUDE (schema.table.col) with EXCLUDE ("schema.table.col")
        let processed_sql = Self::preprocess_exclude_syntax(&processed_sql);

        // Preprocess SQL to remove trailing commas in VALUES clauses
        // DuckDB allows VALUES ('v2',) but sqlparser does not
        let processed_sql = Self::preprocess_trailing_commas_in_values(&processed_sql);

        let dialect = GenericDialect {};
        let statements = parse_sql_with_recursion_limit(&dialect, &processed_sql)
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

    /// Execute a single SELECT statement and return its results as Arrow [`RecordBatch`]es.
    ///
    /// The SQL passed to this method must contain exactly one statement, and that statement must
    /// be a `SELECT`. Statements that modify data (e.g. `INSERT`) should be executed up front
    /// using [`SqlEngine::execute`] before calling this helper.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use arrow::array::StringArray;
    /// use llkv_sql::SqlEngine;
    /// use llkv_storage::pager::MemPager;
    ///
    /// let engine = SqlEngine::new(Arc::new(MemPager::default()));
    /// let _ = engine
    ///     .execute(
    ///         "CREATE TABLE users (id INT PRIMARY KEY, name TEXT);\n         \
    ///          INSERT INTO users (id, name) VALUES (1, 'Ada');",
    ///     )
    ///     .unwrap();
    ///
    /// let batches = engine.sql("SELECT id, name FROM users ORDER BY id;").unwrap();
    /// assert_eq!(batches.len(), 1);
    ///
    /// let batch = &batches[0];
    /// assert_eq!(batch.num_rows(), 1);
    ///
    /// let names = batch
    ///     .column(1)
    ///     .as_any()
    ///     .downcast_ref::<StringArray>()
    ///     .unwrap();
    /// assert_eq!(names.value(0), "Ada");
    /// ```
    pub fn sql(&self, sql: &str) -> SqlResult<Vec<RecordBatch>> {
        let mut results = self.execute(sql)?;
        if results.len() != 1 {
            return Err(Error::InvalidArgumentError(
                "SqlEngine::sql expects exactly one SQL statement".into(),
            ));
        }

        match results.pop().expect("checked length above") {
            RuntimeStatementResult::Select { execution, .. } => execution.collect(),
            other => Err(Error::InvalidArgumentError(format!(
                "SqlEngine::sql requires a SELECT statement, got {other:?}",
            ))),
        }
    }

    fn execute_statement(&self, statement: Statement) -> SqlResult<RuntimeStatementResult<P>> {
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
    ) -> SqlResult<RuntimeStatementResult<P>> {
        tracing::trace!("DEBUG SQL execute_statement_non_transactional called");
        match statement {
            Statement::CreateTable(stmt) => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: CreateTable");
                self.handle_create_table(stmt)
            }
            Statement::CreateIndex(stmt) => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: CreateIndex");
                self.handle_create_index(stmt)
            }
            Statement::CreateSchema {
                schema_name,
                if_not_exists,
                with,
                options,
                default_collate_spec,
                clone,
            } => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: CreateSchema");
                self.handle_create_schema(
                    schema_name,
                    if_not_exists,
                    with,
                    options,
                    default_collate_spec,
                    clone,
                )
            }
            Statement::CreateView {
                name,
                columns,
                query,
                materialized,
                or_replace,
                or_alter,
                options,
                cluster_by,
                comment,
                with_no_schema_binding,
                if_not_exists,
                temporary,
                to,
                params,
                secure,
                name_before_not_exists,
            } => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: CreateView");
                self.handle_create_view(
                    name,
                    columns,
                    query,
                    materialized,
                    or_replace,
                    or_alter,
                    options,
                    cluster_by,
                    comment,
                    with_no_schema_binding,
                    if_not_exists,
                    temporary,
                    to,
                    params,
                    secure,
                    name_before_not_exists,
                )
            }
            Statement::CreateDomain(create_domain) => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: CreateDomain");
                self.handle_create_domain(create_domain)
            }
            Statement::DropDomain(drop_domain) => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: DropDomain");
                self.handle_drop_domain(drop_domain)
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
            Statement::Truncate {
                ref table_names,
                ref partitions,
                table,
                ref identity,
                cascade,
                ref on_cluster,
            } => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: Truncate");
                self.handle_truncate(
                    table_names,
                    partitions,
                    table,
                    identity,
                    cascade,
                    on_cluster,
                )
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
            Statement::AlterTable {
                name,
                if_exists,
                only,
                operations,
                ..
            } => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: AlterTable");
                self.handle_alter_table(name, if_exists, only, operations)
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

    fn collect_known_columns(
        &self,
        display_name: &str,
        canonical_name: &str,
    ) -> SqlResult<HashSet<String>> {
        let context = self.engine.context();

        if context.is_table_marked_dropped(canonical_name) {
            return Err(Self::table_not_found_error(display_name));
        }

        // First, check if the table was created in the current transaction
        if let Some(specs) = self
            .engine
            .session()
            .table_column_specs_from_transaction(canonical_name)
        {
            return Ok(specs
                .into_iter()
                .map(|spec| spec.name.to_ascii_lowercase())
                .collect());
        }

        // Otherwise, look it up in the committed catalog
        let (_, canonical_name) = llkv_table::canonical_table_name(display_name)
            .map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))?;
        match context.catalog().table_column_specs(&canonical_name) {
            Ok(specs) => Ok(specs
                .into_iter()
                .map(|spec| spec.name.to_ascii_lowercase())
                .collect()),
            Err(err) => {
                if !Self::is_table_missing_error(&err) {
                    return Err(Self::map_table_error(display_name, err));
                }

                Ok(HashSet::new())
            }
        }
    }

    fn is_table_marked_dropped(&self, table_name: &str) -> SqlResult<bool> {
        let canonical = table_name.to_ascii_lowercase();
        Ok(self.engine.context().is_table_marked_dropped(&canonical))
    }

    fn handle_create_table(
        &self,
        mut stmt: sqlparser::ast::CreateTable,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        validate_create_table_common(&stmt)?;

        let (mut schema_name, table_name) = parse_schema_qualified_name(&stmt.name)?;

        let namespace = if stmt.temporary {
            if schema_name.is_some() {
                return Err(Error::InvalidArgumentError(
                    "temporary tables cannot specify an explicit schema".into(),
                ));
            }
            schema_name = None;
            Some(TEMPORARY_NAMESPACE_ID.to_string())
        } else {
            None
        };

        // Validate schema exists if specified
        if let Some(ref schema) = schema_name {
            let catalog = self.engine.context().table_catalog();
            if !catalog.schema_exists(schema) {
                return Err(Error::CatalogError(format!(
                    "Schema '{}' does not exist",
                    schema
                )));
            }
        }

        // Use full qualified name (schema.table or just table)
        let display_name = match &schema_name {
            Some(schema) => format!("{}.{}", schema, table_name),
            None => table_name.clone(),
        };
        let canonical_name = display_name.to_ascii_lowercase();
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
                namespace.clone(),
            )? {
                return Ok(result);
            }
            return self.handle_create_table_as(
                display_name,
                canonical_name,
                *query,
                stmt.if_not_exists,
                stmt.or_replace,
                namespace.clone(),
            );
        }

        if stmt.columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires at least one column".into(),
            ));
        }

        validate_create_table_definition(&stmt)?;

        let column_defs_ast = std::mem::take(&mut stmt.columns);
        let constraints = std::mem::take(&mut stmt.constraints);

        let column_names: Vec<String> = column_defs_ast
            .iter()
            .map(|column_def| column_def.name.value.clone())
            .collect();
        ensure_unique_case_insensitive(column_names.iter().map(|name| name.as_str()), |dup| {
            format!(
                "duplicate column name '{}' in table '{}'",
                dup, display_name
            )
        })?;
        let column_names_lower: HashSet<String> = column_names
            .iter()
            .map(|name| name.to_ascii_lowercase())
            .collect();

        let mut columns: Vec<PlanColumnSpec> = Vec::with_capacity(column_defs_ast.len());
        let mut primary_key_columns: HashSet<String> = HashSet::new();
        let mut foreign_keys: Vec<ForeignKeySpec> = Vec::new();
        let mut multi_column_uniques: Vec<MultiColumnUniqueSpec> = Vec::new();

        // Second pass: process columns including CHECK validation and column-level FKs
        for column_def in column_defs_ast {
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

            let has_unique_constraint = column_def
                .options
                .iter()
                .any(|opt| matches!(opt.option, ColumnOption::Unique { .. }));

            // Extract CHECK constraint if present and validate it
            let check_expr = column_def.options.iter().find_map(|opt| {
                if let ColumnOption::Check(expr) = &opt.option {
                    Some(expr)
                } else {
                    None
                }
            });

            // Validate CHECK constraint if present (now we have all column names)
            if let Some(check_expr) = check_expr {
                let all_col_refs: Vec<&str> = column_names.iter().map(|s| s.as_str()).collect();
                validate_check_constraint(check_expr, &display_name, &all_col_refs)?;
            }

            let check_expr_str = check_expr.map(|e| e.to_string());

            // Extract column-level FOREIGN KEY (REFERENCES clause)
            for opt in &column_def.options {
                if let ColumnOption::ForeignKey {
                    foreign_table,
                    referred_columns,
                    on_delete,
                    on_update,
                    characteristics,
                } = &opt.option
                {
                    let spec = self.build_foreign_key_spec(
                        &display_name,
                        &canonical_name,
                        vec![column_def.name.value.clone()],
                        foreign_table,
                        referred_columns,
                        *on_delete,
                        *on_update,
                        characteristics,
                        &column_names_lower,
                        None,
                    )?;
                    foreign_keys.push(spec);
                }
            }

            tracing::trace!(
                "DEBUG CREATE TABLE column '{}' is_primary_key={} has_unique={} check_expr={:?}",
                column_def.name.value,
                is_primary_key,
                has_unique_constraint,
                check_expr_str
            );

            // Resolve custom type aliases to their base types
            let resolved_data_type = self.engine.context().resolve_type(&column_def.data_type);

            let mut column = PlanColumnSpec::new(
                column_def.name.value.clone(),
                arrow_type_from_sql(&resolved_data_type)?,
                is_nullable,
            );
            tracing::trace!(
                "DEBUG PlanColumnSpec after new(): primary_key={} unique={}",
                column.primary_key,
                column.unique
            );

            column = column
                .with_primary_key(is_primary_key)
                .with_unique(has_unique_constraint)
                .with_check(check_expr_str);

            if is_primary_key {
                column.nullable = false;
                primary_key_columns.insert(column.name.to_ascii_lowercase());
            }
            tracing::trace!(
                "DEBUG PlanColumnSpec after with_primary_key({})/with_unique({}): primary_key={} unique={} check_expr={:?}",
                is_primary_key,
                has_unique_constraint,
                column.primary_key,
                column.unique,
                column.check_expr
            );

            columns.push(column);
        }

        // Apply supported table-level constraints (e.g., PRIMARY KEY)
        if !constraints.is_empty() {
            let mut column_lookup: HashMap<String, usize> = HashMap::with_capacity(columns.len());
            for (idx, column) in columns.iter().enumerate() {
                column_lookup.insert(column.name.to_ascii_lowercase(), idx);
            }

            for constraint in constraints {
                match constraint {
                    TableConstraint::PrimaryKey {
                        columns: constraint_columns,
                        ..
                    } => {
                        if !primary_key_columns.is_empty() {
                            return Err(Error::InvalidArgumentError(
                                "multiple PRIMARY KEY constraints are not supported".into(),
                            ));
                        }

                        ensure_non_empty(&constraint_columns, || {
                            "PRIMARY KEY requires at least one column".into()
                        })?;

                        let mut pk_column_names: Vec<String> =
                            Vec::with_capacity(constraint_columns.len());

                        for index_col in &constraint_columns {
                            let column_ident = extract_index_column_name(
                                index_col,
                                "PRIMARY KEY",
                                false, // no sort options allowed
                                false, // only simple identifiers
                            )?;
                            pk_column_names.push(column_ident);
                        }

                        ensure_unique_case_insensitive(
                            pk_column_names.iter().map(|name| name.as_str()),
                            |dup| format!("duplicate column '{}' in PRIMARY KEY constraint", dup),
                        )?;

                        ensure_known_columns_case_insensitive(
                            pk_column_names.iter().map(|name| name.as_str()),
                            &column_names_lower,
                            |unknown| {
                                format!("unknown column '{}' in PRIMARY KEY constraint", unknown)
                            },
                        )?;

                        for column_ident in pk_column_names {
                            let normalized = column_ident.to_ascii_lowercase();
                            let idx = column_lookup.get(&normalized).copied().ok_or_else(|| {
                                Error::InvalidArgumentError(format!(
                                    "unknown column '{}' in PRIMARY KEY constraint",
                                    column_ident
                                ))
                            })?;

                            let column = columns.get_mut(idx).expect("column index valid");
                            column.primary_key = true;
                            column.unique = true;
                            column.nullable = false;

                            primary_key_columns.insert(normalized);
                        }
                    }
                    TableConstraint::Unique {
                        columns: constraint_columns,
                        index_type,
                        index_options,
                        characteristics,
                        nulls_distinct,
                        name,
                        ..
                    } => {
                        if !matches!(nulls_distinct, NullsDistinctOption::None) {
                            return Err(Error::InvalidArgumentError(
                                "UNIQUE constraints with NULLS DISTINCT/NOT DISTINCT are not supported yet".into(),
                            ));
                        }

                        if index_type.is_some() {
                            return Err(Error::InvalidArgumentError(
                                "UNIQUE constraints with index types are not supported yet".into(),
                            ));
                        }

                        if !index_options.is_empty() {
                            return Err(Error::InvalidArgumentError(
                                "UNIQUE constraints with index options are not supported yet"
                                    .into(),
                            ));
                        }

                        if characteristics.is_some() {
                            return Err(Error::InvalidArgumentError(
                                "UNIQUE constraint characteristics are not supported yet".into(),
                            ));
                        }

                        ensure_non_empty(&constraint_columns, || {
                            "UNIQUE constraint requires at least one column".into()
                        })?;

                        let mut unique_column_names: Vec<String> =
                            Vec::with_capacity(constraint_columns.len());

                        for index_column in &constraint_columns {
                            let column_ident = extract_index_column_name(
                                index_column,
                                "UNIQUE constraint",
                                false, // no sort options allowed
                                false, // only simple identifiers
                            )?;
                            unique_column_names.push(column_ident);
                        }

                        ensure_unique_case_insensitive(
                            unique_column_names.iter().map(|name| name.as_str()),
                            |dup| format!("duplicate column '{}' in UNIQUE constraint", dup),
                        )?;

                        ensure_known_columns_case_insensitive(
                            unique_column_names.iter().map(|name| name.as_str()),
                            &column_names_lower,
                            |unknown| format!("unknown column '{}' in UNIQUE constraint", unknown),
                        )?;

                        if unique_column_names.len() > 1 {
                            // Multi-column UNIQUE constraint
                            multi_column_uniques.push(MultiColumnUniqueSpec {
                                name: name.map(|n| n.value),
                                columns: unique_column_names,
                            });
                        } else {
                            // Single-column UNIQUE constraint
                            let column_ident = unique_column_names
                                .into_iter()
                                .next()
                                .expect("unique constraint checked for emptiness");
                            let normalized = column_ident.to_ascii_lowercase();
                            let idx = column_lookup.get(&normalized).copied().ok_or_else(|| {
                                Error::InvalidArgumentError(format!(
                                    "unknown column '{}' in UNIQUE constraint",
                                    column_ident
                                ))
                            })?;

                            let column = columns
                                .get_mut(idx)
                                .expect("column index from lookup must be valid");
                            column.unique = true;
                        }
                    }
                    TableConstraint::ForeignKey {
                        name,
                        index_name,
                        columns: fk_columns,
                        foreign_table,
                        referred_columns,
                        on_delete,
                        on_update,
                        characteristics,
                        ..
                    } => {
                        if index_name.is_some() {
                            return Err(Error::InvalidArgumentError(
                                "FOREIGN KEY index clauses are not supported yet".into(),
                            ));
                        }

                        let referencing_columns: Vec<String> =
                            fk_columns.into_iter().map(|ident| ident.value).collect();
                        let spec = self.build_foreign_key_spec(
                            &display_name,
                            &canonical_name,
                            referencing_columns,
                            &foreign_table,
                            &referred_columns,
                            on_delete,
                            on_update,
                            &characteristics,
                            &column_names_lower,
                            name.map(|ident| ident.value),
                        )?;

                        foreign_keys.push(spec);
                    }
                    unsupported => {
                        return Err(Error::InvalidArgumentError(format!(
                            "table-level constraint {:?} is not supported",
                            unsupported
                        )));
                    }
                }
            }
        }

        let plan = CreateTablePlan {
            name: display_name,
            if_not_exists: stmt.if_not_exists,
            or_replace: stmt.or_replace,
            columns,
            source: None,
            namespace,
            foreign_keys,
            multi_column_uniques,
        };
        self.execute_plan_statement(PlanStatement::CreateTable(plan))
    }

    fn handle_create_index(
        &self,
        stmt: sqlparser::ast::CreateIndex,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        let sqlparser::ast::CreateIndex {
            name,
            table_name,
            using,
            columns,
            unique,
            concurrently,
            if_not_exists,
            include,
            nulls_distinct,
            with,
            predicate,
            index_options,
            alter_options,
            ..
        } = stmt;

        if concurrently {
            return Err(Error::InvalidArgumentError(
                "CREATE INDEX CONCURRENTLY is not supported".into(),
            ));
        }
        if using.is_some() {
            return Err(Error::InvalidArgumentError(
                "CREATE INDEX USING clauses are not supported".into(),
            ));
        }
        if !include.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE INDEX INCLUDE columns are not supported".into(),
            ));
        }
        if nulls_distinct.is_some() {
            return Err(Error::InvalidArgumentError(
                "CREATE INDEX NULLS DISTINCT is not supported".into(),
            ));
        }
        if !with.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE INDEX WITH options are not supported".into(),
            ));
        }
        if predicate.is_some() {
            return Err(Error::InvalidArgumentError(
                "partial CREATE INDEX is not supported".into(),
            ));
        }
        if !index_options.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE INDEX options are not supported".into(),
            ));
        }
        if !alter_options.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE INDEX ALTER options are not supported".into(),
            ));
        }
        if columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE INDEX requires at least one column".into(),
            ));
        }

        let (schema_name, base_table_name) = parse_schema_qualified_name(&table_name)?;
        if let Some(ref schema) = schema_name {
            let catalog = self.engine.context().table_catalog();
            if !catalog.schema_exists(schema) {
                return Err(Error::CatalogError(format!(
                    "Schema '{}' does not exist",
                    schema
                )));
            }
        }

        let display_table_name = schema_name
            .as_ref()
            .map(|schema| format!("{}.{}", schema, base_table_name))
            .unwrap_or_else(|| base_table_name.clone());
        let canonical_table_name = display_table_name.to_ascii_lowercase();

        let known_columns =
            self.collect_known_columns(&display_table_name, &canonical_table_name)?;
        let enforce_known_columns = !known_columns.is_empty();

        let index_name = match name {
            Some(name_obj) => Some(Self::object_name_to_string(&name_obj)?),
            None => None,
        };

        let mut index_columns: Vec<IndexColumnPlan> = Vec::with_capacity(columns.len());
        let mut seen_column_names: HashSet<String> = HashSet::new();
        for item in columns {
            // Check WITH FILL before calling helper (not part of standard validation)
            if item.column.with_fill.is_some() {
                return Err(Error::InvalidArgumentError(
                    "CREATE INDEX column WITH FILL is not supported".into(),
                ));
            }

            let column_name = extract_index_column_name(
                &item,
                "CREATE INDEX",
                true, // allow and validate sort options
                true, // allow compound identifiers
            )?;

            // Get sort options (already validated by helper)
            let order_expr = &item.column;
            let ascending = order_expr.options.asc.unwrap_or(true);
            let nulls_first = order_expr.options.nulls_first.unwrap_or(false);

            let normalized = column_name.to_ascii_lowercase();
            if !seen_column_names.insert(normalized.clone()) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in CREATE INDEX",
                    column_name
                )));
            }

            if enforce_known_columns && !known_columns.contains(&normalized) {
                return Err(Error::InvalidArgumentError(format!(
                    "column '{}' does not exist in table '{}'",
                    column_name, display_table_name
                )));
            }

            let column_plan = IndexColumnPlan::new(column_name).with_sort(ascending, nulls_first);
            index_columns.push(column_plan);
        }

        let plan = CreateIndexPlan::new(display_table_name)
            .with_name(index_name)
            .with_unique(unique)
            .with_if_not_exists(if_not_exists)
            .with_columns(index_columns);

        self.execute_plan_statement(PlanStatement::CreateIndex(plan))
    }

    fn map_referential_action(
        action: Option<ReferentialAction>,
        kind: &str,
    ) -> SqlResult<ForeignKeyAction> {
        match action {
            None | Some(ReferentialAction::NoAction) => Ok(ForeignKeyAction::NoAction),
            Some(ReferentialAction::Restrict) => Ok(ForeignKeyAction::Restrict),
            Some(other) => Err(Error::InvalidArgumentError(format!(
                "FOREIGN KEY ON {kind} {:?} is not supported yet",
                other
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_foreign_key_spec(
        &self,
        _referencing_display: &str,
        referencing_canonical: &str,
        referencing_columns: Vec<String>,
        foreign_table: &ObjectName,
        referenced_columns: &[Ident],
        on_delete: Option<ReferentialAction>,
        on_update: Option<ReferentialAction>,
        characteristics: &Option<ConstraintCharacteristics>,
        known_columns_lower: &HashSet<String>,
        name: Option<String>,
    ) -> SqlResult<ForeignKeySpec> {
        if characteristics.is_some() {
            return Err(Error::InvalidArgumentError(
                "FOREIGN KEY constraint characteristics are not supported yet".into(),
            ));
        }

        ensure_non_empty(&referencing_columns, || {
            "FOREIGN KEY constraint requires at least one referencing column".into()
        })?;
        ensure_unique_case_insensitive(
            referencing_columns.iter().map(|name| name.as_str()),
            |dup| format!("duplicate column '{}' in FOREIGN KEY constraint", dup),
        )?;
        ensure_known_columns_case_insensitive(
            referencing_columns.iter().map(|name| name.as_str()),
            known_columns_lower,
            |unknown| format!("unknown column '{}' in FOREIGN KEY constraint", unknown),
        )?;

        let referenced_columns_vec: Vec<String> = referenced_columns
            .iter()
            .map(|ident| ident.value.clone())
            .collect();
        ensure_unique_case_insensitive(
            referenced_columns_vec.iter().map(|name| name.as_str()),
            |dup| {
                format!(
                    "duplicate referenced column '{}' in FOREIGN KEY constraint",
                    dup
                )
            },
        )?;

        if !referenced_columns_vec.is_empty()
            && referenced_columns_vec.len() != referencing_columns.len()
        {
            return Err(Error::InvalidArgumentError(
                "FOREIGN KEY referencing and referenced column counts must match".into(),
            ));
        }

        let (referenced_display, referenced_canonical) = canonical_object_name(foreign_table)?;

        // Check if the referenced table is a VIEW - VIEWs cannot be referenced by foreign keys
        let catalog = self.engine.context().table_catalog();
        if let Some(table_id) = catalog.table_id(&referenced_canonical) {
            let context = self.engine.context();
            if context.is_view(table_id)? {
                return Err(Error::CatalogError(format!(
                    "Binder Error: cannot reference a VIEW with a FOREIGN KEY: {}",
                    referenced_display
                )));
            }
        }

        if referenced_canonical == referencing_canonical {
            ensure_known_columns_case_insensitive(
                referenced_columns_vec.iter().map(|name| name.as_str()),
                known_columns_lower,
                |unknown| {
                    format!(
                        "Binder Error: table '{}' does not have a column named '{}'",
                        referenced_display, unknown
                    )
                },
            )?;
        } else {
            let known_columns =
                self.collect_known_columns(&referenced_display, &referenced_canonical)?;
            if !known_columns.is_empty() {
                ensure_known_columns_case_insensitive(
                    referenced_columns_vec.iter().map(|name| name.as_str()),
                    &known_columns,
                    |unknown| {
                        format!(
                            "Binder Error: table '{}' does not have a column named '{}'",
                            referenced_display, unknown
                        )
                    },
                )?;
            }
        }

        let on_delete_action = Self::map_referential_action(on_delete, "DELETE")?;
        let on_update_action = Self::map_referential_action(on_update, "UPDATE")?;

        Ok(ForeignKeySpec {
            name,
            columns: referencing_columns,
            referenced_table: referenced_display,
            referenced_columns: referenced_columns_vec,
            on_delete: on_delete_action,
            on_update: on_update_action,
        })
    }

    fn handle_create_schema(
        &self,
        schema_name: SchemaName,
        _if_not_exists: bool,
        with: Option<Vec<SqlOption>>,
        options: Option<Vec<SqlOption>>,
        default_collate_spec: Option<SqlExpr>,
        clone: Option<ObjectName>,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        if clone.is_some() {
            return Err(Error::InvalidArgumentError(
                "CREATE SCHEMA ... CLONE is not supported".into(),
            ));
        }
        if with.as_ref().is_some_and(|opts| !opts.is_empty()) {
            return Err(Error::InvalidArgumentError(
                "CREATE SCHEMA ... WITH options are not supported".into(),
            ));
        }
        if options.as_ref().is_some_and(|opts| !opts.is_empty()) {
            return Err(Error::InvalidArgumentError(
                "CREATE SCHEMA options are not supported".into(),
            ));
        }
        if default_collate_spec.is_some() {
            return Err(Error::InvalidArgumentError(
                "CREATE SCHEMA DEFAULT COLLATE is not supported".into(),
            ));
        }

        let schema_name = match schema_name {
            SchemaName::Simple(name) => name,
            _ => {
                return Err(Error::InvalidArgumentError(
                    "CREATE SCHEMA authorization is not supported".into(),
                ));
            }
        };

        let (display_name, canonical) = canonical_object_name(&schema_name)?;
        if display_name.is_empty() {
            return Err(Error::InvalidArgumentError(
                "schema name must not be empty".into(),
            ));
        }

        // Register schema in the catalog
        let catalog = self.engine.context().table_catalog();

        if _if_not_exists && catalog.schema_exists(&canonical) {
            return Ok(RuntimeStatementResult::NoOp);
        }

        catalog.register_schema(&canonical).map_err(|err| {
            Error::CatalogError(format!(
                "Failed to create schema '{}': {}",
                display_name, err
            ))
        })?;

        Ok(RuntimeStatementResult::NoOp)
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_create_view(
        &self,
        name: ObjectName,
        _columns: Vec<sqlparser::ast::ViewColumnDef>,
        query: Box<sqlparser::ast::Query>,
        materialized: bool,
        or_replace: bool,
        or_alter: bool,
        _options: sqlparser::ast::CreateTableOptions,
        _cluster_by: Vec<sqlparser::ast::Ident>,
        _comment: Option<String>,
        _with_no_schema_binding: bool,
        if_not_exists: bool,
        temporary: bool,
        _to: Option<ObjectName>,
        _params: Option<sqlparser::ast::CreateViewParams>,
        _secure: bool,
        _name_before_not_exists: bool,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        // Validate unsupported features
        if materialized {
            return Err(Error::InvalidArgumentError(
                "MATERIALIZED VIEWS are not supported".into(),
            ));
        }
        if or_replace {
            return Err(Error::InvalidArgumentError(
                "CREATE OR REPLACE VIEW is not supported".into(),
            ));
        }
        if or_alter {
            return Err(Error::InvalidArgumentError(
                "CREATE OR ALTER VIEW is not supported".into(),
            ));
        }
        if temporary {
            return Err(Error::InvalidArgumentError(
                "TEMPORARY VIEWS are not supported".into(),
            ));
        }

        // Parse view name (same as table parsing)
        let (schema_name, view_name) = parse_schema_qualified_name(&name)?;

        // Validate schema exists if specified
        if let Some(ref schema) = schema_name {
            let catalog = self.engine.context().table_catalog();
            if !catalog.schema_exists(schema) {
                return Err(Error::CatalogError(format!(
                    "Schema '{}' does not exist",
                    schema
                )));
            }
        }

        // Use full qualified name (schema.view or just view)
        let display_name = match &schema_name {
            Some(schema) => format!("{}.{}", schema, view_name),
            None => view_name.clone(),
        };
        let canonical_name = display_name.to_ascii_lowercase();

        // Check if view already exists
        let catalog = self.engine.context().table_catalog();
        if catalog.table_exists(&canonical_name) {
            if if_not_exists {
                return Ok(RuntimeStatementResult::NoOp);
            }
            return Err(Error::CatalogError(format!(
                "Table or view '{}' already exists",
                display_name
            )));
        }

        // Convert query to SQL string for storage
        let view_definition = query.to_string();

        // Create the view using the runtime context helper
        let context = self.engine.context();
        context.create_view(&display_name, view_definition)?;

        tracing::debug!("Created view: {}", display_name);
        Ok(RuntimeStatementResult::NoOp)
    }

    fn handle_create_domain(
        &self,
        create_domain: sqlparser::ast::CreateDomain,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        use llkv_table::CustomTypeMeta;
        use std::time::{SystemTime, UNIX_EPOCH};

        // Extract the type alias name
        let type_name = create_domain.name.to_string();

        // Convert the data type to SQL string for persistence
        let base_type_sql = create_domain.data_type.to_string();

        // Register the type alias in the runtime context (in-memory)
        self.engine
            .context()
            .register_type(type_name.clone(), create_domain.data_type.clone());

        // Persist to catalog
        let context = self.engine.context();
        let catalog = llkv_table::SysCatalog::new(context.store());

        let created_at_micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let meta = CustomTypeMeta {
            name: type_name.clone(),
            base_type_sql,
            created_at_micros,
        };

        catalog.put_custom_type_meta(&meta)?;

        tracing::debug!("Created and persisted type alias: {}", type_name);
        Ok(RuntimeStatementResult::NoOp)
    }

    fn handle_drop_domain(
        &self,
        drop_domain: sqlparser::ast::DropDomain,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        let if_exists = drop_domain.if_exists;
        let type_name = drop_domain.name.to_string();

        // Drop the type from the registry (in-memory)
        let result = self.engine.context().drop_type(&type_name);

        if let Err(err) = result {
            if !if_exists {
                return Err(err);
            }
            // if_exists = true, so ignore the error
        } else {
            // Persist deletion to catalog
            let context = self.engine.context();
            let catalog = llkv_table::SysCatalog::new(context.store());
            catalog.delete_custom_type_meta(&type_name)?;

            tracing::debug!("Dropped and removed from catalog type alias: {}", type_name);
        }

        Ok(RuntimeStatementResult::NoOp)
    }

    fn try_handle_range_ctas(
        &self,
        display_name: &str,
        _canonical_name: &str,
        query: &Query,
        if_not_exists: bool,
        or_replace: bool,
        namespace: Option<String>,
    ) -> SqlResult<Option<RuntimeStatementResult<P>>> {
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
                    column_specs.push(PlanColumnSpec::new(column_name.clone(), data_type, true));
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
            namespace,
            foreign_keys: Vec::new(),
            multi_column_uniques: Vec::new(),
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

    // TODO: Refactor into runtime or executor layer?
    // NOTE: PRAGMA handling lives in the SQL layer until a shared runtime hook exists.
    /// Try to handle pragma_table_info('table_name') table function
    fn try_handle_pragma_table_info(
        &self,
        query: &Query,
    ) -> SqlResult<Option<RuntimeStatementResult<P>>> {
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

        // Check if this is pragma_table_info function call
        let table_name = match &table_with_joins.relation {
            TableFactor::Table {
                name,
                args: Some(args),
                ..
            } => {
                let func_name = name.to_string().to_ascii_lowercase();
                if func_name != "pragma_table_info" {
                    return Ok(None);
                }

                // Extract table name from argument
                if args.args.len() != 1 {
                    return Err(Error::InvalidArgumentError(
                        "pragma_table_info expects exactly one argument".into(),
                    ));
                }

                match &args.args[0] {
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Value(value))) => {
                        match &value.value {
                            Value::SingleQuotedString(s) => s.clone(),
                            Value::DoubleQuotedString(s) => s.clone(),
                            _ => {
                                return Err(Error::InvalidArgumentError(
                                    "pragma_table_info argument must be a string".into(),
                                ));
                            }
                        }
                    }
                    _ => {
                        return Err(Error::InvalidArgumentError(
                            "pragma_table_info argument must be a string literal".into(),
                        ));
                    }
                }
            }
            _ => return Ok(None),
        };

        // Get table column specs from runtime context
        let context = self.engine.context();
        let (_, canonical_name) = llkv_table::canonical_table_name(&table_name)?;
        let columns = context.catalog().table_column_specs(&canonical_name)?;

        // Build RecordBatch with table column information
        use arrow::array::{BooleanArray, Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};

        let mut cid_values = Vec::new();
        let mut name_values = Vec::new();
        let mut type_values = Vec::new();
        let mut notnull_values = Vec::new();
        let mut dflt_value_values: Vec<Option<String>> = Vec::new();
        let mut pk_values = Vec::new();

        for (idx, col) in columns.iter().enumerate() {
            cid_values.push(idx as i32);
            name_values.push(col.name.clone());
            type_values.push(format!("{:?}", col.data_type)); // Simple type representation
            notnull_values.push(!col.nullable);
            dflt_value_values.push(None); // We don't track default values yet
            pk_values.push(col.primary_key);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("cid", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("type", DataType::Utf8, false),
            Field::new("notnull", DataType::Boolean, false),
            Field::new("dflt_value", DataType::Utf8, true),
            Field::new("pk", DataType::Boolean, false),
        ]));

        use arrow::array::ArrayRef;
        let mut batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(cid_values)) as ArrayRef,
                Arc::new(StringArray::from(name_values)) as ArrayRef,
                Arc::new(StringArray::from(type_values)) as ArrayRef,
                Arc::new(BooleanArray::from(notnull_values)) as ArrayRef,
                Arc::new(StringArray::from(dflt_value_values)) as ArrayRef,
                Arc::new(BooleanArray::from(pk_values)) as ArrayRef,
            ],
        )
        .map_err(|e| Error::Internal(format!("failed to create pragma_table_info batch: {}", e)))?;

        // Apply SELECT projections: extract only requested columns
        let projection_indices: Vec<usize> = select
            .projection
            .iter()
            .filter_map(|item| {
                match item {
                    SelectItem::UnnamedExpr(SqlExpr::Identifier(ident)) => {
                        schema.index_of(&ident.value).ok()
                    }
                    SelectItem::ExprWithAlias { expr, .. } => {
                        if let SqlExpr::Identifier(ident) = expr {
                            schema.index_of(&ident.value).ok()
                        } else {
                            None
                        }
                    }
                    SelectItem::Wildcard(_) => None, // Handle * separately
                    _ => None,
                }
            })
            .collect();

        // Apply projections if not SELECT *
        let projected_schema;
        if !projection_indices.is_empty() {
            let projected_fields: Vec<Field> = projection_indices
                .iter()
                .map(|&idx| schema.field(idx).clone())
                .collect();
            projected_schema = Arc::new(Schema::new(projected_fields));

            let projected_columns: Vec<ArrayRef> = projection_indices
                .iter()
                .map(|&idx| Arc::clone(batch.column(idx)))
                .collect();

            batch = RecordBatch::try_new(Arc::clone(&projected_schema), projected_columns)
                .map_err(|e| Error::Internal(format!("failed to project columns: {}", e)))?;
        } else {
            // SELECT * or complex projections - use original schema
            projected_schema = schema;
        }

        // Apply ORDER BY using Arrow compute kernels
        if let Some(order_by) = &query.order_by {
            use arrow::compute::SortColumn;
            use arrow::compute::lexsort_to_indices;
            use sqlparser::ast::OrderByKind;

            let exprs = match &order_by.kind {
                OrderByKind::Expressions(exprs) => exprs,
                _ => {
                    return Err(Error::InvalidArgumentError(
                        "unsupported ORDER BY clause".into(),
                    ));
                }
            };

            let mut sort_columns = Vec::new();
            for order_expr in exprs {
                if let SqlExpr::Identifier(ident) = &order_expr.expr
                    && let Ok(col_idx) = projected_schema.index_of(&ident.value)
                {
                    let options = arrow::compute::SortOptions {
                        descending: !order_expr.options.asc.unwrap_or(true),
                        nulls_first: order_expr.options.nulls_first.unwrap_or(false),
                    };
                    sort_columns.push(SortColumn {
                        values: Arc::clone(batch.column(col_idx)),
                        options: Some(options),
                    });
                }
            }

            if !sort_columns.is_empty() {
                let indices = lexsort_to_indices(&sort_columns, None)
                    .map_err(|e| Error::Internal(format!("failed to sort: {}", e)))?;

                use arrow::compute::take;
                let sorted_columns: Result<Vec<ArrayRef>, _> = batch
                    .columns()
                    .iter()
                    .map(|col| take(col.as_ref(), &indices, None))
                    .collect();

                batch = RecordBatch::try_new(
                    Arc::clone(&projected_schema),
                    sorted_columns
                        .map_err(|e| Error::Internal(format!("failed to apply sort: {}", e)))?,
                )
                .map_err(|e| Error::Internal(format!("failed to create sorted batch: {}", e)))?;
            }
        }

        let execution = SelectExecution::new_single_batch(
            table_name.clone(),
            Arc::clone(&projected_schema),
            batch,
        );

        Ok(Some(RuntimeStatementResult::Select {
            table_name,
            schema: projected_schema,
            execution: Box::new(execution),
        }))
    }

    fn handle_create_table_as(
        &self,
        display_name: String,
        _canonical_name: String,
        query: Query,
        if_not_exists: bool,
        or_replace: bool,
        namespace: Option<String>,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        // Check if this is a SELECT from VALUES in a derived table
        // Pattern: SELECT * FROM (VALUES ...) alias(col1, col2, ...)
        if let SetExpr::Select(select) = query.body.as_ref()
            && let Some((rows, column_names)) = extract_values_from_derived_table(&select.from)?
        {
            // Convert VALUES rows to Arrow RecordBatches
            return self.handle_create_table_from_values(
                display_name,
                rows,
                column_names,
                if_not_exists,
                or_replace,
                namespace,
            );
        }

        // Regular CTAS with SELECT from existing tables
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
            namespace,
            foreign_keys: Vec::new(),
            multi_column_uniques: Vec::new(),
        };
        self.execute_plan_statement(PlanStatement::CreateTable(plan))
    }

    fn handle_create_table_from_values(
        &self,
        display_name: String,
        rows: Vec<Vec<PlanValue>>,
        column_names: Vec<String>,
        if_not_exists: bool,
        or_replace: bool,
        namespace: Option<String>,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        use arrow::array::{ArrayRef, Float64Builder, Int64Builder, StringBuilder};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        if rows.is_empty() {
            return Err(Error::InvalidArgumentError(
                "VALUES must have at least one row".into(),
            ));
        }

        let num_cols = column_names.len();

        // Infer schema from first row
        let first_row = &rows[0];
        if first_row.len() != num_cols {
            return Err(Error::InvalidArgumentError(
                "VALUES row column count mismatch".into(),
            ));
        }

        let mut fields = Vec::with_capacity(num_cols);
        let mut column_types = Vec::with_capacity(num_cols);

        for (idx, value) in first_row.iter().enumerate() {
            let (data_type, nullable) = match value {
                PlanValue::Integer(_) => (DataType::Int64, false),
                PlanValue::Float(_) => (DataType::Float64, false),
                PlanValue::String(_) => (DataType::Utf8, false),
                PlanValue::Null => (DataType::Utf8, true), // Default NULL to string type
                _ => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported value type in VALUES for column '{}'",
                        column_names.get(idx).unwrap_or(&format!("column{}", idx))
                    )));
                }
            };

            column_types.push(data_type.clone());
            fields.push(Field::new(&column_names[idx], data_type, nullable));
        }

        let schema = Arc::new(Schema::new(fields));

        // Build Arrow arrays for each column
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(num_cols);

        for col_idx in 0..num_cols {
            let col_type = &column_types[col_idx];

            match col_type {
                DataType::Int64 => {
                    let mut builder = Int64Builder::with_capacity(rows.len());
                    for row in &rows {
                        match &row[col_idx] {
                            PlanValue::Integer(v) => builder.append_value(*v),
                            PlanValue::Null => builder.append_null(),
                            other => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "type mismatch in VALUES: expected Integer, got {:?}",
                                    other
                                )));
                            }
                        }
                    }
                    arrays.push(Arc::new(builder.finish()) as ArrayRef);
                }
                DataType::Float64 => {
                    let mut builder = Float64Builder::with_capacity(rows.len());
                    for row in &rows {
                        match &row[col_idx] {
                            PlanValue::Float(v) => builder.append_value(*v),
                            PlanValue::Null => builder.append_null(),
                            other => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "type mismatch in VALUES: expected Float, got {:?}",
                                    other
                                )));
                            }
                        }
                    }
                    arrays.push(Arc::new(builder.finish()) as ArrayRef);
                }
                DataType::Utf8 => {
                    let mut builder = StringBuilder::with_capacity(rows.len(), 1024);
                    for row in &rows {
                        match &row[col_idx] {
                            PlanValue::String(v) => builder.append_value(v),
                            PlanValue::Null => builder.append_null(),
                            other => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "type mismatch in VALUES: expected String, got {:?}",
                                    other
                                )));
                            }
                        }
                    }
                    arrays.push(Arc::new(builder.finish()) as ArrayRef);
                }
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported column type in VALUES: {:?}",
                        other
                    )));
                }
            }
        }

        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).map_err(|e| {
            Error::Internal(format!("failed to create RecordBatch from VALUES: {}", e))
        })?;

        let plan = CreateTablePlan {
            name: display_name.clone(),
            if_not_exists,
            or_replace,
            columns: Vec::new(),
            source: Some(CreateTableSource::Batches {
                schema: Arc::clone(&schema),
                batches: vec![batch],
            }),
            namespace,
            foreign_keys: Vec::new(),
            multi_column_uniques: Vec::new(),
        };

        self.execute_plan_statement(PlanStatement::CreateTable(plan))
    }

    fn handle_insert(&self, stmt: sqlparser::ast::Insert) -> SqlResult<RuntimeStatementResult<P>> {
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
    ) -> SqlResult<RuntimeStatementResult<P>> {
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

        let catalog = self.engine.context().table_catalog();
        let resolver = catalog.identifier_resolver();
        let table_id = catalog.table_id(&canonical_name);

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
                    let translated = translate_scalar_with_context(
                        &resolver,
                        IdentifierContext::new(table_id),
                        &assignment.value,
                    )?;
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
            Some(expr) => {
                let materialized_expr = self.materialize_in_subquery(expr)?;
                Some(translate_condition_with_context(
                    &resolver,
                    IdentifierContext::new(table_id),
                    &materialized_expr,
                )?)
            }
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
    fn handle_delete(&self, delete: Delete) -> SqlResult<RuntimeStatementResult<P>> {
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

        let catalog = self.engine.context().table_catalog();
        let resolver = catalog.identifier_resolver();
        let table_id = catalog.table_id(&canonical_name);

        let filter = selection
            .map(|expr| {
                let materialized_expr = self.materialize_in_subquery(expr)?;
                translate_condition_with_context(
                    &resolver,
                    IdentifierContext::new(table_id),
                    &materialized_expr,
                )
            })
            .transpose()?;

        let plan = DeletePlan {
            table: display_name.clone(),
            filter,
        };
        self.execute_plan_statement(PlanStatement::Delete(plan))
    }

    fn handle_truncate(
        &self,
        table_names: &[sqlparser::ast::TruncateTableTarget],
        partitions: &Option<Vec<SqlExpr>>,
        _table: bool, // boolean field in sqlparser, not the table name
        identity: &Option<sqlparser::ast::TruncateIdentityOption>,
        cascade: Option<sqlparser::ast::CascadeOption>,
        on_cluster: &Option<Ident>,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        // Validate unsupported features
        if table_names.len() > 1 {
            return Err(Error::InvalidArgumentError(
                "TRUNCATE with multiple tables is not supported yet".into(),
            ));
        }
        if partitions.is_some() {
            return Err(Error::InvalidArgumentError(
                "TRUNCATE ... PARTITION is not supported".into(),
            ));
        }
        if identity.is_some() {
            return Err(Error::InvalidArgumentError(
                "TRUNCATE ... RESTART/CONTINUE IDENTITY is not supported".into(),
            ));
        }
        use sqlparser::ast::CascadeOption;
        if matches!(cascade, Some(CascadeOption::Cascade)) {
            return Err(Error::InvalidArgumentError(
                "TRUNCATE ... CASCADE is not supported".into(),
            ));
        }
        if on_cluster.is_some() {
            return Err(Error::InvalidArgumentError(
                "TRUNCATE ... ON CLUSTER is not supported".into(),
            ));
        }

        // Extract table name from table_names
        let table_name = if let Some(target) = table_names.first() {
            // TruncateTableTarget has a name field
            let table_obj = &target.name;
            let display_name = table_obj.to_string();
            let canonical_name = display_name.to_ascii_lowercase();

            // Check if table is dropped in transaction
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

            display_name
        } else {
            return Err(Error::InvalidArgumentError(
                "TRUNCATE requires a table name".into(),
            ));
        };

        let plan = TruncatePlan {
            table: table_name.clone(),
        };
        self.execute_plan_statement(PlanStatement::Truncate(plan))
    }

    #[allow(clippy::too_many_arguments)] // NOTE: Signature mirrors SQL grammar; keep grouped until command builder is introduced.
    fn handle_drop(
        &self,
        object_type: ObjectType,
        if_exists: bool,
        names: Vec<ObjectName>,
        cascade: bool,
        restrict: bool,
        purge: bool,
        temporary: bool,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        if purge || temporary {
            return Err(Error::InvalidArgumentError(
                "DROP purge/temporary options are not supported".into(),
            ));
        }

        match object_type {
            ObjectType::Table => {
                if cascade || restrict {
                    return Err(Error::InvalidArgumentError(
                        "DROP TABLE CASCADE/RESTRICT is not supported".into(),
                    ));
                }

                for name in names {
                    let table_name = Self::object_name_to_string(&name)?;
                    let mut plan = llkv_plan::DropTablePlan::new(table_name.clone());
                    plan.if_exists = if_exists;

                    self.execute_plan_statement(llkv_plan::PlanStatement::DropTable(plan))
                        .map_err(|err| Self::map_table_error(&table_name, err))?;
                }

                Ok(RuntimeStatementResult::NoOp)
            }
            ObjectType::Index => {
                if cascade || restrict {
                    return Err(Error::InvalidArgumentError(
                        "DROP INDEX CASCADE/RESTRICT is not supported".into(),
                    ));
                }

                for name in names {
                    let index_name = Self::object_name_to_string(&name)?;
                    let plan = llkv_plan::DropIndexPlan::new(index_name).if_exists(if_exists);
                    self.execute_plan_statement(llkv_plan::PlanStatement::DropIndex(plan))?;
                }

                Ok(RuntimeStatementResult::NoOp)
            }
            ObjectType::Schema => {
                if restrict {
                    return Err(Error::InvalidArgumentError(
                        "DROP SCHEMA RESTRICT is not supported".into(),
                    ));
                }

                let catalog = self.engine.context().table_catalog();

                for name in names {
                    let (display_name, canonical_name) = canonical_object_name(&name)?;

                    if !catalog.schema_exists(&canonical_name) {
                        if if_exists {
                            continue;
                        }
                        return Err(Error::CatalogError(format!(
                            "Schema '{}' does not exist",
                            display_name
                        )));
                    }

                    if cascade {
                        // Drop all tables in this schema
                        let all_tables = catalog.table_names();
                        let schema_prefix = format!("{}.", canonical_name);

                        for table in all_tables {
                            if table.to_ascii_lowercase().starts_with(&schema_prefix) {
                                let mut plan = llkv_plan::DropTablePlan::new(table.clone());
                                plan.if_exists = false;
                                self.execute_plan_statement(llkv_plan::PlanStatement::DropTable(
                                    plan,
                                ))?;
                            }
                        }
                    } else {
                        // Check if schema has any tables
                        let all_tables = catalog.table_names();
                        let schema_prefix = format!("{}.", canonical_name);
                        let has_tables = all_tables
                            .iter()
                            .any(|t| t.to_ascii_lowercase().starts_with(&schema_prefix));

                        if has_tables {
                            return Err(Error::CatalogError(format!(
                                "Schema '{}' is not empty. Use CASCADE to drop schema and all its tables",
                                display_name
                            )));
                        }
                    }

                    // Drop the schema
                    if !catalog.unregister_schema(&canonical_name) && !if_exists {
                        return Err(Error::CatalogError(format!(
                            "Schema '{}' does not exist",
                            display_name
                        )));
                    }
                }

                Ok(RuntimeStatementResult::NoOp)
            }
            _ => Err(Error::InvalidArgumentError(format!(
                "DROP {} is not supported",
                object_type
            ))),
        }
    }

    fn handle_alter_table(
        &self,
        name: ObjectName,
        if_exists: bool,
        only: bool,
        operations: Vec<AlterTableOperation>,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        if only {
            return Err(Error::InvalidArgumentError(
                "ALTER TABLE ONLY is not supported yet".into(),
            ));
        }

        if operations.is_empty() {
            return Ok(RuntimeStatementResult::NoOp);
        }

        if operations.len() != 1 {
            return Err(Error::InvalidArgumentError(
                "ALTER TABLE currently supports exactly one operation".into(),
            ));
        }

        let operation = operations.into_iter().next().expect("checked length");
        match operation {
            AlterTableOperation::RenameTable { table_name } => {
                let new_name = table_name.to_string();
                self.handle_alter_table_rename(name, new_name, if_exists)
            }
            AlterTableOperation::RenameColumn {
                old_column_name,
                new_column_name,
            } => {
                let plan = llkv_plan::AlterTablePlan {
                    table_name: name.to_string(),
                    if_exists,
                    operation: llkv_plan::AlterTableOperation::RenameColumn {
                        old_column_name: old_column_name.to_string(),
                        new_column_name: new_column_name.to_string(),
                    },
                };
                self.execute_plan_statement(PlanStatement::AlterTable(plan))
            }
            AlterTableOperation::AlterColumn { column_name, op } => {
                // Only support SET DATA TYPE for now
                if let AlterColumnOperation::SetDataType {
                    data_type,
                    using,
                    had_set: _,
                } = op
                {
                    if using.is_some() {
                        return Err(Error::InvalidArgumentError(
                            "ALTER COLUMN SET DATA TYPE USING clause is not yet supported".into(),
                        ));
                    }

                    let plan = llkv_plan::AlterTablePlan {
                        table_name: name.to_string(),
                        if_exists,
                        operation: llkv_plan::AlterTableOperation::SetColumnDataType {
                            column_name: column_name.to_string(),
                            new_data_type: data_type.to_string(),
                        },
                    };
                    self.execute_plan_statement(PlanStatement::AlterTable(plan))
                } else {
                    Err(Error::InvalidArgumentError(format!(
                        "unsupported ALTER COLUMN operation: {:?}",
                        op
                    )))
                }
            }
            AlterTableOperation::DropColumn {
                has_column_keyword: _,
                column_names,
                if_exists: column_if_exists,
                drop_behavior,
            } => {
                if column_names.len() != 1 {
                    return Err(Error::InvalidArgumentError(
                        "DROP COLUMN currently supports dropping one column at a time".into(),
                    ));
                }

                let column_name = column_names.into_iter().next().unwrap().to_string();
                let cascade = matches!(drop_behavior, Some(sqlparser::ast::DropBehavior::Cascade));

                let plan = llkv_plan::AlterTablePlan {
                    table_name: name.to_string(),
                    if_exists,
                    operation: llkv_plan::AlterTableOperation::DropColumn {
                        column_name,
                        if_exists: column_if_exists,
                        cascade,
                    },
                };
                self.execute_plan_statement(PlanStatement::AlterTable(plan))
            }
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported ALTER TABLE operation: {:?}",
                other
            ))),
        }
    }

    fn handle_alter_table_rename(
        &self,
        original_name: ObjectName,
        new_table_name: String,
        if_exists: bool,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        let (schema_opt, table_name) = parse_schema_qualified_name(&original_name)?;

        let new_table_name_clean = new_table_name.trim();

        if new_table_name_clean.is_empty() {
            return Err(Error::InvalidArgumentError(
                "ALTER TABLE RENAME requires a non-empty table name".into(),
            ));
        }

        let (raw_new_schema_opt, raw_new_table) =
            if let Some((schema_part, table_part)) = new_table_name_clean.split_once('.') {
                (
                    Some(schema_part.trim().to_string()),
                    table_part.trim().to_string(),
                )
            } else {
                (None, new_table_name_clean.to_string())
            };

        if schema_opt.is_none() && raw_new_schema_opt.is_some() {
            return Err(Error::InvalidArgumentError(
                "ALTER TABLE RENAME cannot add a schema qualifier".into(),
            ));
        }

        let new_table_trimmed = raw_new_table.trim_matches('"');
        if new_table_trimmed.is_empty() {
            return Err(Error::InvalidArgumentError(
                "ALTER TABLE RENAME requires a non-empty table name".into(),
            ));
        }

        if let (Some(existing_schema), Some(new_schema_raw)) =
            (schema_opt.as_ref(), raw_new_schema_opt.as_ref())
        {
            let new_schema_trimmed = new_schema_raw.trim_matches('"');
            if !existing_schema.eq_ignore_ascii_case(new_schema_trimmed) {
                return Err(Error::InvalidArgumentError(
                    "ALTER TABLE RENAME cannot change table schema".into(),
                ));
            }
        }

        let new_table_display = raw_new_table;
        let new_schema_opt = raw_new_schema_opt;

        fn join_schema_table(schema: &str, table: &str) -> String {
            let mut qualified = String::with_capacity(schema.len() + table.len() + 1);
            qualified.push_str(schema);
            qualified.push('.');
            qualified.push_str(table);
            qualified
        }

        let current_display = schema_opt
            .as_ref()
            .map(|schema| join_schema_table(schema, &table_name))
            .unwrap_or_else(|| table_name.clone());

        let new_display = if let Some(new_schema_raw) = new_schema_opt.clone() {
            join_schema_table(&new_schema_raw, &new_table_display)
        } else if let Some(schema) = schema_opt.as_ref() {
            join_schema_table(schema, &new_table_display)
        } else {
            new_table_display.clone()
        };

        let plan = RenameTablePlan::new(&current_display, &new_display).if_exists(if_exists);

        match CatalogDdl::rename_table(self.engine.session(), plan) {
            Ok(()) => Ok(RuntimeStatementResult::NoOp),
            Err(err) => Err(Self::map_table_error(&current_display, err)),
        }
    }

    /// Materialize IN (SELECT ...) subqueries by executing them and converting to IN lists.
    ///
    /// This preprocesses SQL expressions to execute subqueries and replace them with
    /// their materialized results before translation to the execution plan.
    ///
    /// Uses iterative traversal with an explicit work stack to handle deeply nested
    /// expressions without stack overflow.
    fn materialize_in_subquery(&self, root_expr: SqlExpr) -> SqlResult<SqlExpr> {
        // Stack-based iterative traversal to avoid recursion
        enum WorkItem {
            Process(Box<SqlExpr>),
            BuildBinaryOp {
                op: BinaryOperator,
                left: Box<SqlExpr>,
                right_done: bool,
            },
            BuildUnaryOp {
                op: UnaryOperator,
            },
            BuildNested,
            BuildIsNull,
            BuildIsNotNull,
            FinishBetween {
                negated: bool,
            },
        }

        let mut work_stack: Vec<WorkItem> = vec![WorkItem::Process(Box::new(root_expr))];
        let mut result_stack: Vec<SqlExpr> = Vec::new();

        while let Some(item) = work_stack.pop() {
            match item {
                WorkItem::Process(expr) => {
                    match *expr {
                        SqlExpr::InSubquery {
                            expr: left_expr,
                            subquery,
                            negated,
                        } => {
                            // Execute the subquery
                            let result = self.handle_query(*subquery)?;

                            // Extract values from first column
                            let values = match result {
                                RuntimeStatementResult::Select { execution, .. } => {
                                    let batches = execution.collect()?;
                                    let mut collected_values = Vec::new();

                                    for batch in batches {
                                        if batch.num_columns() == 0 {
                                            continue;
                                        }
                                        let column = batch.column(0);

                                        for row_idx in 0..column.len() {
                                            use arrow::datatypes::DataType;
                                            let value = if column.is_null(row_idx) {
                                                Value::Null
                                            } else {
                                                match column.data_type() {
                                                    DataType::Int64 => {
                                                        let arr = column
                                                        .as_any()
                                                        .downcast_ref::<arrow::array::Int64Array>()
                                                        .unwrap();
                                                        Value::Number(
                                                            arr.value(row_idx).to_string(),
                                                            false,
                                                        )
                                                    }
                                                    DataType::Float64 => {
                                                        let arr = column
                                                        .as_any()
                                                        .downcast_ref::<arrow::array::Float64Array>()
                                                        .unwrap();
                                                        Value::Number(
                                                            arr.value(row_idx).to_string(),
                                                            false,
                                                        )
                                                    }
                                                    DataType::Utf8 => {
                                                        let arr = column
                                                        .as_any()
                                                        .downcast_ref::<arrow::array::StringArray>()
                                                        .unwrap();
                                                        Value::SingleQuotedString(
                                                            arr.value(row_idx).to_string(),
                                                        )
                                                    }
                                                    DataType::Boolean => {
                                                        let arr = column
                                                        .as_any()
                                                        .downcast_ref::<arrow::array::BooleanArray>()
                                                        .unwrap();
                                                        Value::Boolean(arr.value(row_idx))
                                                    }
                                                    other => {
                                                        return Err(Error::InvalidArgumentError(
                                                            format!(
                                                                "unsupported data type in IN subquery: {other:?}"
                                                            ),
                                                        ));
                                                    }
                                                }
                                            };
                                            collected_values.push(ValueWithSpan {
                                                value,
                                                span: Span::empty(),
                                            });
                                        }
                                    }

                                    collected_values
                                }
                                _ => {
                                    return Err(Error::InvalidArgumentError(
                                        "IN subquery must be a SELECT statement".into(),
                                    ));
                                }
                            };

                            // Convert to IN list with materialized values
                            result_stack.push(SqlExpr::InList {
                                expr: left_expr,
                                list: values.into_iter().map(SqlExpr::Value).collect(),
                                negated,
                            });
                        }
                        SqlExpr::BinaryOp { left, op, right } => {
                            // Push builder, then schedule the left operand followed by the right.
                            // Because the work stack is LIFO, the right-hand side executes first,
                            // leaving the left result on top for the builder to consume without
                            // swapping operand order.
                            work_stack.push(WorkItem::BuildBinaryOp {
                                op,
                                left: left.clone(),
                                right_done: false,
                            });
                            // Evaluate the right-hand side first so the left result remains on top
                            // of the result stack when the builder consumes it. This preserves the
                            // original operand ordering when reconstructing the expression tree.
                            work_stack.push(WorkItem::Process(left));
                            work_stack.push(WorkItem::Process(right));
                        }
                        SqlExpr::UnaryOp { op, expr } => {
                            work_stack.push(WorkItem::BuildUnaryOp { op });
                            work_stack.push(WorkItem::Process(expr));
                        }
                        SqlExpr::Nested(inner) => {
                            work_stack.push(WorkItem::BuildNested);
                            work_stack.push(WorkItem::Process(inner));
                        }
                        SqlExpr::IsNull(inner) => {
                            work_stack.push(WorkItem::BuildIsNull);
                            work_stack.push(WorkItem::Process(inner));
                        }
                        SqlExpr::IsNotNull(inner) => {
                            work_stack.push(WorkItem::BuildIsNotNull);
                            work_stack.push(WorkItem::Process(inner));
                        }
                        SqlExpr::Between {
                            expr,
                            negated,
                            low,
                            high,
                        } => {
                            work_stack.push(WorkItem::FinishBetween { negated });
                            work_stack.push(WorkItem::Process(high));
                            work_stack.push(WorkItem::Process(low));
                            work_stack.push(WorkItem::Process(expr));
                        }
                        // All other expressions: push as-is to result stack
                        other => {
                            result_stack.push(other);
                        }
                    }
                }
                WorkItem::BuildBinaryOp {
                    op,
                    left,
                    right_done,
                } => {
                    if !right_done {
                        // Left done, now mark that right will be done next
                        let left_result = result_stack.pop().unwrap();
                        work_stack.push(WorkItem::BuildBinaryOp {
                            op,
                            left: Box::new(left_result),
                            right_done: true,
                        });
                    } else {
                        // Both done, build the BinaryOp
                        let right_result = result_stack.pop().unwrap();
                        let left_result = *left;
                        result_stack.push(SqlExpr::BinaryOp {
                            left: Box::new(left_result),
                            op,
                            right: Box::new(right_result),
                        });
                    }
                }
                WorkItem::BuildUnaryOp { op } => {
                    let inner = result_stack.pop().unwrap();
                    result_stack.push(SqlExpr::UnaryOp {
                        op,
                        expr: Box::new(inner),
                    });
                }
                WorkItem::BuildNested => {
                    let inner = result_stack.pop().unwrap();
                    result_stack.push(SqlExpr::Nested(Box::new(inner)));
                }
                WorkItem::BuildIsNull => {
                    let inner = result_stack.pop().unwrap();
                    result_stack.push(SqlExpr::IsNull(Box::new(inner)));
                }
                WorkItem::BuildIsNotNull => {
                    let inner = result_stack.pop().unwrap();
                    result_stack.push(SqlExpr::IsNotNull(Box::new(inner)));
                }
                WorkItem::FinishBetween { negated } => {
                    let high_result = result_stack.pop().unwrap();
                    let low_result = result_stack.pop().unwrap();
                    let expr_result = result_stack.pop().unwrap();
                    result_stack.push(SqlExpr::Between {
                        expr: Box::new(expr_result),
                        negated,
                        low: Box::new(low_result),
                        high: Box::new(high_result),
                    });
                }
            }
        }

        // Final result should be the only item on the result stack
        Ok(result_stack
            .pop()
            .expect("result stack should have exactly one item"))
    }

    fn handle_query(&self, query: Query) -> SqlResult<RuntimeStatementResult<P>> {
        // Check for pragma_table_info() table function first
        if let Some(result) = self.try_handle_pragma_table_info(&query)? {
            return Ok(result);
        }

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
        let catalog = self.engine.context().table_catalog();
        let resolver = catalog.identifier_resolver();

        let (mut select_plan, select_context) = match query.body.as_ref() {
            SetExpr::Select(select) => self.translate_select(select.as_ref(), &resolver)?,
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
            let order_plan = self.translate_order_by(&resolver, select_context, order_by)?;
            select_plan = select_plan.with_order_by(order_plan);
        }
        Ok(select_plan)
    }

    fn translate_select(
        &self,
        select: &Select,
        resolver: &IdentifierResolver<'_>,
    ) -> SqlResult<(SelectPlan, IdentifierContext)> {
        let distinct = match &select.distinct {
            None => false,
            Some(Distinct::Distinct) => true,
            Some(Distinct::On(_)) => {
                return Err(Error::InvalidArgumentError(
                    "SELECT DISTINCT ON is not supported".into(),
                ));
            }
        };
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

        let table_alias = select
            .from
            .first()
            .and_then(|table_with_joins| match &table_with_joins.relation {
                TableFactor::Table { alias, .. } => alias.as_ref().map(|a| a.name.value.clone()),
                _ => None,
            });

        if let Some(alias) = table_alias.as_ref() {
            validate_projection_alias_qualifiers(&select.projection, alias)?;
        }
        let has_joins = select
            .from
            .iter()
            .any(|table_with_joins| !table_with_joins.joins.is_empty());
        // Handle different FROM clause scenarios
        let catalog = self.engine.context().table_catalog();
        let (mut plan, id_context) = if select.from.is_empty() {
            // No FROM clause - use empty string for table context (e.g., SELECT 42, SELECT {'a': 1} AS x)
            let mut p = SelectPlan::new("");
            let projections = self.build_projection_list(
                resolver,
                IdentifierContext::new(None),
                &select.projection,
            )?;
            p = p.with_projections(projections);
            (p, IdentifierContext::new(None))
        } else if select.from.len() == 1 && !has_joins {
            // Single table query
            let (display_name, canonical_name) = extract_single_table(&select.from)?;
            let table_id = catalog.table_id(&canonical_name);
            let mut p = SelectPlan::new(display_name.clone());
            let single_table_context =
                IdentifierContext::new(table_id).with_table_alias(table_alias.clone());
            if let Some(aggregates) = self.detect_simple_aggregates(&select.projection)? {
                p = p.with_aggregates(aggregates);
            } else {
                let projections = self.build_projection_list(
                    resolver,
                    single_table_context.clone(),
                    &select.projection,
                )?;
                p = p.with_projections(projections);
            }
            (p, single_table_context)
        } else {
            // Multiple tables or explicit joins - treat as cross product for now
            let tables = extract_tables(&select.from)?;
            let mut p = SelectPlan::with_tables(tables);
            // For multi-table queries, we'll build projections differently
            // For now, just handle simple column references
            let projections = self.build_projection_list(
                resolver,
                IdentifierContext::new(None),
                &select.projection,
            )?;
            p = p.with_projections(projections);
            (p, IdentifierContext::new(None))
        };

        let filter_expr = match &select.selection {
            Some(expr) => {
                let materialized_expr = self.materialize_in_subquery(expr.clone())?;
                Some(translate_condition_with_context(
                    resolver,
                    id_context.clone(),
                    &materialized_expr,
                )?)
            }
            None => None,
        };
        plan = plan.with_filter(filter_expr);
        plan = plan.with_distinct(distinct);
        Ok((plan, id_context))
    }

    fn translate_order_by(
        &self,
        resolver: &IdentifierResolver<'_>,
        id_context: IdentifierContext,
        order_by: &OrderBy,
    ) -> SqlResult<Vec<OrderByPlan>> {
        let exprs = match &order_by.kind {
            OrderByKind::Expressions(exprs) => exprs,
            _ => {
                return Err(Error::InvalidArgumentError(
                    "unsupported ORDER BY clause".into(),
                ));
            }
        };

        let base_nulls_first = self.default_nulls_first.load(AtomicOrdering::Relaxed);

        let mut plans = Vec::with_capacity(exprs.len());
        for order_expr in exprs {
            let ascending = order_expr.options.asc.unwrap_or(true);
            let default_nulls_first_for_direction = if ascending {
                base_nulls_first
            } else {
                !base_nulls_first
            };
            let nulls_first = order_expr
                .options
                .nulls_first
                .unwrap_or(default_nulls_first_for_direction);

            if let SqlExpr::Identifier(ident) = &order_expr.expr
                && ident.value.eq_ignore_ascii_case("ALL")
                && ident.quote_style.is_none()
            {
                plans.push(OrderByPlan {
                    target: OrderTarget::All,
                    sort_type: OrderSortType::Native,
                    ascending,
                    nulls_first,
                });
                continue;
            }

            let (target, sort_type) = match &order_expr.expr {
                SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => (
                    OrderTarget::Column(Self::resolve_simple_column_expr(
                        resolver,
                        id_context.clone(),
                        &order_expr.expr,
                    )?),
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
                    OrderTarget::Column(Self::resolve_simple_column_expr(
                        resolver,
                        id_context.clone(),
                        expr,
                    )?),
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

            plans.push(OrderByPlan {
                target,
                sort_type,
                ascending,
                nulls_first,
            });
        }

        Ok(plans)
    }

    fn resolve_simple_column_expr(
        resolver: &IdentifierResolver<'_>,
        context: IdentifierContext,
        expr: &SqlExpr,
    ) -> SqlResult<String> {
        let scalar = translate_scalar_with_context(resolver, context, expr)?;
        match scalar {
            llkv_expr::expr::ScalarExpr::Column(column) => Ok(column),
            other => Err(Error::InvalidArgumentError(format!(
                "ORDER BY expression must reference a simple column, found {other:?}"
            ))),
        }
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

            let mut is_distinct = false;
            let args_slice: &[FunctionArg] = match &func.args {
                FunctionArguments::List(list) => {
                    if let Some(dup) = &list.duplicate_treatment {
                        use sqlparser::ast::DuplicateTreatment;
                        match dup {
                            DuplicateTreatment::All => {}
                            DuplicateTreatment::Distinct => is_distinct = true,
                        }
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
                            if is_distinct {
                                return Err(Error::InvalidArgumentError(
                                    "COUNT(DISTINCT *) is not supported".into(),
                                ));
                            }
                            AggregateExpr::count_star(alias)
                        }
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(arg_expr)) => {
                            let column = resolve_column_name(arg_expr)?;
                            if is_distinct {
                                AggregateExpr::count_distinct_column(column, alias)
                            } else {
                                AggregateExpr::count_column(column, alias)
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
                    if is_distinct {
                        return Err(Error::InvalidArgumentError(
                            "DISTINCT is not supported for this aggregate".into(),
                        ));
                    }
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
        resolver: &IdentifierResolver<'_>,
        id_context: IdentifierContext,
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
                SelectItem::Wildcard(options) => {
                    if let Some(exclude) = &options.opt_exclude {
                        use sqlparser::ast::ExcludeSelectItem;
                        let exclude_cols = match exclude {
                            ExcludeSelectItem::Single(ident) => vec![ident.value.clone()],
                            ExcludeSelectItem::Multiple(idents) => {
                                idents.iter().map(|id| id.value.clone()).collect()
                            }
                        };
                        projections.push(SelectProjection::AllColumnsExcept {
                            exclude: exclude_cols,
                        });
                    } else {
                        projections.push(SelectProjection::AllColumns);
                    }
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
                SelectItem::UnnamedExpr(expr) => match expr {
                    SqlExpr::Identifier(ident) => {
                        let parts = vec![ident.value.clone()];
                        let resolution = resolver.resolve(&parts, id_context.clone())?;
                        if resolution.is_simple() {
                            projections.push(SelectProjection::Column {
                                name: resolution.column().to_string(),
                                alias: None,
                            });
                        } else {
                            let alias = format!("col{}", idx + 1);
                            projections.push(SelectProjection::Computed {
                                expr: resolution.into_scalar_expr(),
                                alias,
                            });
                        }
                    }
                    SqlExpr::CompoundIdentifier(parts) => {
                        let name_parts: Vec<String> =
                            parts.iter().map(|part| part.value.clone()).collect();
                        let resolution = resolver.resolve(&name_parts, id_context.clone())?;
                        if resolution.is_simple() {
                            projections.push(SelectProjection::Column {
                                name: resolution.column().to_string(),
                                alias: None,
                            });
                        } else {
                            let alias = format!("col{}", idx + 1);
                            projections.push(SelectProjection::Computed {
                                expr: resolution.into_scalar_expr(),
                                alias,
                            });
                        }
                    }
                    _ => {
                        let alias = format!("col{}", idx + 1);
                        let scalar =
                            translate_scalar_with_context(resolver, id_context.clone(), expr)?;
                        projections.push(SelectProjection::Computed {
                            expr: scalar,
                            alias,
                        });
                    }
                },
                SelectItem::ExprWithAlias { expr, alias } => match expr {
                    SqlExpr::Identifier(ident) => {
                        let parts = vec![ident.value.clone()];
                        let resolution = resolver.resolve(&parts, id_context.clone())?;
                        if resolution.is_simple() {
                            projections.push(SelectProjection::Column {
                                name: resolution.column().to_string(),
                                alias: Some(alias.value.clone()),
                            });
                        } else {
                            projections.push(SelectProjection::Computed {
                                expr: resolution.into_scalar_expr(),
                                alias: alias.value.clone(),
                            });
                        }
                    }
                    SqlExpr::CompoundIdentifier(parts) => {
                        let name_parts: Vec<String> =
                            parts.iter().map(|part| part.value.clone()).collect();
                        let resolution = resolver.resolve(&name_parts, id_context.clone())?;
                        if resolution.is_simple() {
                            projections.push(SelectProjection::Column {
                                name: resolution.column().to_string(),
                                alias: Some(alias.value.clone()),
                            });
                        } else {
                            projections.push(SelectProjection::Computed {
                                expr: resolution.into_scalar_expr(),
                                alias: alias.value.clone(),
                            });
                        }
                    }
                    _ => {
                        let scalar =
                            translate_scalar_with_context(resolver, id_context.clone(), expr)?;
                        projections.push(SelectProjection::Computed {
                            expr: scalar,
                            alias: alias.value.clone(),
                        });
                    }
                },
            }
        }
        Ok(projections)
    }

    #[allow(clippy::too_many_arguments)] // NOTE: Keeps parity with SQL START TRANSACTION grammar; revisit when options expand.
    fn handle_start_transaction(
        &self,
        modes: Vec<TransactionMode>,
        begin: bool,
        transaction: Option<BeginTransactionKind>,
        modifier: Option<TransactionModifier>,
        statements: Vec<Statement>,
        exception: Option<Vec<ExceptionWhen>>,
        has_end_keyword: bool,
    ) -> SqlResult<RuntimeStatementResult<P>> {
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
    ) -> SqlResult<RuntimeStatementResult<P>> {
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
    ) -> SqlResult<RuntimeStatementResult<P>> {
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

    fn handle_set(&self, set_stmt: Set) -> SqlResult<RuntimeStatementResult<P>> {
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

                        Ok(RuntimeStatementResult::NoOp)
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
                        Ok(RuntimeStatementResult::NoOp)
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
    ) -> SqlResult<RuntimeStatementResult<P>> {
        let (display, canonical) = canonical_object_name(&name)?;
        if value.is_some() || is_eq {
            return Err(Error::InvalidArgumentError(format!(
                "PRAGMA '{display}' does not accept a value"
            )));
        }

        match canonical.as_str() {
            "enable_verification" | "disable_verification" => Ok(RuntimeStatementResult::NoOp),
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

/// Parse an object name into optional schema and table name components.
///
/// Returns (schema_name, table_name) where schema_name is None if not qualified.
///
/// Examples:
/// - "users" -> (None, "users")
/// - "test.users" -> (Some("test"), "users")
/// - "catalog.test.users" -> Error (too many parts)
fn parse_schema_qualified_name(name: &ObjectName) -> SqlResult<(Option<String>, String)> {
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

    match parts.len() {
        1 => Ok((None, parts[0].clone())),
        2 => Ok((Some(parts[0].clone()), parts[1].clone())),
        _ => Err(Error::InvalidArgumentError(format!(
            "table name has too many parts: {}",
            name
        ))),
    }
}

/// Extract column name from an index column specification (OrderBy expression).
///
/// This handles the common pattern of validating and extracting column names from
/// SQL index column definitions (PRIMARY KEY, UNIQUE, CREATE INDEX).
///
/// # Parameters
/// - `index_col`: The index column from sqlparser AST
/// - `context`: Description of where this column appears (e.g., "PRIMARY KEY", "UNIQUE constraint")
/// - `allow_sort_options`: If false, errors if any sort options are present; if true, validates them
/// - `allow_compound`: If true, allows compound identifiers and takes the last part; if false, only allows simple identifiers
///
/// # Returns
/// Column name as a String
fn extract_index_column_name(
    index_col: &sqlparser::ast::IndexColumn,
    context: &str,
    allow_sort_options: bool,
    allow_compound: bool,
) -> SqlResult<String> {
    use sqlparser::ast::Expr as SqlExpr;

    // Check operator class
    if index_col.operator_class.is_some() {
        return Err(Error::InvalidArgumentError(format!(
            "{} operator classes are not supported",
            context
        )));
    }

    let order_expr = &index_col.column;

    // Validate sort options
    if allow_sort_options {
        // For CREATE INDEX: extract and validate sort options
        let _ascending = order_expr.options.asc.unwrap_or(true);
        let _nulls_first = order_expr.options.nulls_first.unwrap_or(false);
        // DESC and NULLS FIRST are now supported
    } else {
        // For constraints: no sort options allowed
        if order_expr.options.asc.is_some()
            || order_expr.options.nulls_first.is_some()
            || order_expr.with_fill.is_some()
        {
            return Err(Error::InvalidArgumentError(format!(
                "{} columns must be simple identifiers",
                context
            )));
        }
    }

    // Extract column name from expression
    let column_name = match &order_expr.expr {
        SqlExpr::Identifier(ident) => ident.value.clone(),
        SqlExpr::CompoundIdentifier(parts) => {
            if allow_compound {
                // For CREATE INDEX: allow qualified names, take last part
                parts
                    .last()
                    .map(|ident| ident.value.clone())
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "invalid column reference in {}",
                            context
                        ))
                    })?
            } else if parts.len() == 1 {
                // For constraints: only allow single-part compound identifiers
                parts[0].value.clone()
            } else {
                return Err(Error::InvalidArgumentError(format!(
                    "{} columns must be column identifiers",
                    context
                )));
            }
        }
        other => {
            return Err(Error::InvalidArgumentError(format!(
                "{} only supports column references, found {:?}",
                context, other
            )));
        }
    };

    Ok(column_name)
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
    use sqlparser::ast::TableConstraint;

    let mut seen_primary_key = false;
    for constraint in &stmt.constraints {
        match constraint {
            TableConstraint::PrimaryKey { .. } => {
                if seen_primary_key {
                    return Err(Error::InvalidArgumentError(
                        "multiple PRIMARY KEY constraints are not supported".into(),
                    ));
                }
                seen_primary_key = true;
            }
            TableConstraint::Unique { .. } => {
                // Detailed validation is performed later during plan construction.
            }
            TableConstraint::ForeignKey { .. } => {
                // Detailed validation is performed later during plan construction.
            }
            other => {
                return Err(Error::InvalidArgumentError(format!(
                    "table-level constraint {:?} is not supported",
                    other
                )));
            }
        }
    }
    Ok(())
}

fn validate_check_constraint(
    check_expr: &sqlparser::ast::Expr,
    table_name: &str,
    column_names: &[&str],
) -> SqlResult<()> {
    use sqlparser::ast::Expr as SqlExpr;

    let column_names_lower: HashSet<String> = column_names
        .iter()
        .map(|name| name.to_ascii_lowercase())
        .collect();

    let mut stack: Vec<&SqlExpr> = vec![check_expr];

    while let Some(expr) = stack.pop() {
        match expr {
            SqlExpr::Subquery(_) => {
                return Err(Error::InvalidArgumentError(
                    "Subqueries are not allowed in CHECK constraints".into(),
                ));
            }
            SqlExpr::Function(func) => {
                let func_name = func.name.to_string().to_uppercase();
                if matches!(func_name.as_str(), "SUM" | "AVG" | "COUNT" | "MIN" | "MAX") {
                    return Err(Error::InvalidArgumentError(
                        "Aggregate functions are not allowed in CHECK constraints".into(),
                    ));
                }

                if let sqlparser::ast::FunctionArguments::List(list) = &func.args {
                    for arg in &list.args {
                        if let sqlparser::ast::FunctionArg::Unnamed(
                            sqlparser::ast::FunctionArgExpr::Expr(expr),
                        ) = arg
                        {
                            stack.push(expr);
                        }
                    }
                }
            }
            SqlExpr::Identifier(ident) => {
                if !column_names_lower.contains(&ident.value.to_ascii_lowercase()) {
                    return Err(Error::InvalidArgumentError(format!(
                        "Column '{}' referenced in CHECK constraint does not exist",
                        ident.value
                    )));
                }
            }
            SqlExpr::CompoundIdentifier(idents) => {
                if idents.len() == 2 {
                    let first = idents[0].value.as_str();
                    let second = &idents[1].value;

                    if column_names_lower.contains(&first.to_ascii_lowercase()) {
                        continue;
                    }

                    if !first.eq_ignore_ascii_case(table_name) {
                        return Err(Error::InvalidArgumentError(format!(
                            "CHECK constraint references column from different table '{}'",
                            first
                        )));
                    }

                    if !column_names_lower.contains(&second.to_ascii_lowercase()) {
                        return Err(Error::InvalidArgumentError(format!(
                            "Column '{}' referenced in CHECK constraint does not exist",
                            second
                        )));
                    }
                } else if idents.len() == 3 {
                    let first = &idents[0].value;
                    let second = &idents[1].value;
                    let third = &idents[2].value;

                    if first.eq_ignore_ascii_case(table_name) {
                        if !column_names_lower.contains(&second.to_ascii_lowercase()) {
                            return Err(Error::InvalidArgumentError(format!(
                                "Column '{}' referenced in CHECK constraint does not exist",
                                second
                            )));
                        }
                    } else if second.eq_ignore_ascii_case(table_name) {
                        if !column_names_lower.contains(&third.to_ascii_lowercase()) {
                            return Err(Error::InvalidArgumentError(format!(
                                "Column '{}' referenced in CHECK constraint does not exist",
                                third
                            )));
                        }
                    } else {
                        return Err(Error::InvalidArgumentError(format!(
                            "CHECK constraint references column from different table '{}'",
                            second
                        )));
                    }
                }
            }
            SqlExpr::BinaryOp { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            SqlExpr::UnaryOp { expr, .. } | SqlExpr::Nested(expr) => {
                stack.push(expr);
            }
            SqlExpr::Value(_) | SqlExpr::TypedString { .. } => {}
            _ => {}
        }
    }

    Ok(())
}

fn validate_create_table_definition(stmt: &sqlparser::ast::CreateTable) -> SqlResult<()> {
    for column in &stmt.columns {
        for ColumnOptionDef { option, .. } in &column.options {
            match option {
                ColumnOption::Null
                | ColumnOption::NotNull
                | ColumnOption::Unique { .. }
                | ColumnOption::Check(_)
                | ColumnOption::ForeignKey { .. } => {}
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

fn validate_projection_alias_qualifiers(
    projection_items: &[SelectItem],
    alias: &str,
) -> SqlResult<()> {
    let alias_lower = alias.to_ascii_lowercase();
    for item in projection_items {
        match item {
            SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => {
                if let SqlExpr::CompoundIdentifier(parts) = expr
                    && parts.len() >= 2
                    && let Some(first) = parts.first()
                    && !first.value.eq_ignore_ascii_case(&alias_lower)
                {
                    return Err(Error::InvalidArgumentError(format!(
                        "Binder Error: table '{}' not found",
                        first.value
                    )));
                }
            }
            _ => {}
        }
    }
    Ok(())
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
        llkv_expr::expr::ScalarExpr::GetField { base, .. } => expr_contains_aggregate(base),
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

fn translate_condition_with_context(
    resolver: &IdentifierResolver<'_>,
    context: IdentifierContext,
    expr: &SqlExpr,
) -> SqlResult<llkv_expr::expr::Expr<'static, String>> {
    // Iterative postorder traversal using the TransformFrame pattern.
    // See llkv-plan::TransformFrame documentation for pattern details.
    //
    // This avoids stack overflow on deeply nested expressions (50k+ nodes) by using
    // explicit work_stack and result_stack instead of recursion.

    enum ConditionExitContext {
        And,
        Or,
        Not,
        Nested,
    }

    type ConditionFrame<'a> = llkv_plan::TransformFrame<
        'a,
        SqlExpr,
        llkv_expr::expr::Expr<'static, String>,
        ConditionExitContext,
    >;

    let mut work_stack: Vec<ConditionFrame> = vec![ConditionFrame::Enter(expr)];
    let mut result_stack: Vec<llkv_expr::expr::Expr<'static, String>> = Vec::new();

    while let Some(frame) = work_stack.pop() {
        match frame {
            ConditionFrame::Enter(node) => match node {
                SqlExpr::BinaryOp { left, op, right } => match op {
                    BinaryOperator::And => {
                        work_stack.push(ConditionFrame::Exit(ConditionExitContext::And));
                        work_stack.push(ConditionFrame::Enter(right));
                        work_stack.push(ConditionFrame::Enter(left));
                    }
                    BinaryOperator::Or => {
                        work_stack.push(ConditionFrame::Exit(ConditionExitContext::Or));
                        work_stack.push(ConditionFrame::Enter(right));
                        work_stack.push(ConditionFrame::Enter(left));
                    }
                    BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Lt
                    | BinaryOperator::LtEq
                    | BinaryOperator::Gt
                    | BinaryOperator::GtEq => {
                        let result = translate_comparison_with_context(
                            resolver,
                            context.clone(),
                            left,
                            op.clone(),
                            right,
                        )?;
                        work_stack.push(ConditionFrame::Leaf(result));
                    }
                    other => {
                        return Err(Error::InvalidArgumentError(format!(
                            "unsupported binary operator in WHERE clause: {other:?}"
                        )));
                    }
                },
                SqlExpr::UnaryOp {
                    op: UnaryOperator::Not,
                    expr: inner,
                } => {
                    work_stack.push(ConditionFrame::Exit(ConditionExitContext::Not));
                    work_stack.push(ConditionFrame::Enter(inner));
                }
                SqlExpr::Nested(inner) => {
                    work_stack.push(ConditionFrame::Exit(ConditionExitContext::Nested));
                    work_stack.push(ConditionFrame::Enter(inner));
                }
                SqlExpr::IsNull(inner) => {
                    let scalar = translate_scalar_with_context(resolver, context.clone(), inner)?;
                    match scalar {
                        llkv_expr::expr::ScalarExpr::Column(column) => {
                            work_stack.push(ConditionFrame::Leaf(llkv_expr::expr::Expr::Pred(
                                llkv_expr::expr::Filter {
                                    field_id: column,
                                    op: llkv_expr::expr::Operator::IsNull,
                                },
                            )));
                        }
                        _ => {
                            return Err(Error::InvalidArgumentError(
                                "IS NULL predicates currently support column references only"
                                    .into(),
                            ));
                        }
                    }
                }
                SqlExpr::IsNotNull(inner) => {
                    let scalar = translate_scalar_with_context(resolver, context.clone(), inner)?;
                    match scalar {
                        llkv_expr::expr::ScalarExpr::Column(column) => {
                            work_stack.push(ConditionFrame::Leaf(llkv_expr::expr::Expr::Pred(
                                llkv_expr::expr::Filter {
                                    field_id: column,
                                    op: llkv_expr::expr::Operator::IsNotNull,
                                },
                            )));
                        }
                        _ => {
                            return Err(Error::InvalidArgumentError(
                                "IS NOT NULL predicates currently support column references only"
                                    .into(),
                            ));
                        }
                    }
                }
                SqlExpr::InList {
                    expr: in_expr,
                    list,
                    negated,
                } => {
                    if list.is_empty() {
                        let result = if *negated {
                            llkv_expr::expr::Expr::Literal(true)
                        } else {
                            llkv_expr::expr::Expr::Literal(false)
                        };
                        work_stack.push(ConditionFrame::Leaf(result));
                    } else {
                        let target =
                            translate_scalar_with_context(resolver, context.clone(), in_expr)?;
                        let mut values = Vec::with_capacity(list.len());
                        for value_expr in list {
                            let scalar = translate_scalar_with_context(
                                resolver,
                                context.clone(),
                                value_expr,
                            )?;
                            values.push(scalar);
                        }

                        work_stack.push(ConditionFrame::Leaf(llkv_expr::expr::Expr::InList {
                            expr: target,
                            list: values,
                            negated: *negated,
                        }));
                    }
                }
                SqlExpr::InSubquery { .. } => {
                    return Err(Error::InvalidArgumentError(
                        "IN (SELECT ...) subqueries must be materialized before translation".into(),
                    ));
                }
                SqlExpr::Between {
                    expr: between_expr,
                    negated,
                    low,
                    high,
                } => {
                    let lower_bound = translate_comparison_with_context(
                        resolver,
                        context.clone(),
                        between_expr,
                        BinaryOperator::GtEq,
                        low,
                    )?;
                    let upper_bound = translate_comparison_with_context(
                        resolver,
                        context.clone(),
                        between_expr,
                        BinaryOperator::LtEq,
                        high,
                    )?;

                    let between_expr_result =
                        llkv_expr::expr::Expr::And(vec![lower_bound, upper_bound]);

                    let result = if *negated {
                        llkv_expr::expr::Expr::not(between_expr_result)
                    } else {
                        between_expr_result
                    };
                    work_stack.push(ConditionFrame::Leaf(result));
                }
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported WHERE clause: {other:?}"
                    )));
                }
            },
            ConditionFrame::Leaf(translated) => {
                result_stack.push(translated);
            }
            ConditionFrame::Exit(exit_context) => match exit_context {
                ConditionExitContext::And => {
                    let right = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_condition: result stack underflow for And right".into(),
                        )
                    })?;
                    let left = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_condition: result stack underflow for And left".into(),
                        )
                    })?;
                    result_stack.push(flatten_and(left, right));
                }
                ConditionExitContext::Or => {
                    let right = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_condition: result stack underflow for Or right".into(),
                        )
                    })?;
                    let left = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_condition: result stack underflow for Or left".into(),
                        )
                    })?;
                    result_stack.push(flatten_or(left, right));
                }
                ConditionExitContext::Not => {
                    let inner = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_condition: result stack underflow for Not".into(),
                        )
                    })?;
                    result_stack.push(llkv_expr::expr::Expr::not(inner));
                }
                ConditionExitContext::Nested => {
                    // Nested is a no-op - just pass through the inner expression
                }
            },
        }
    }

    result_stack.pop().ok_or_else(|| {
        Error::Internal("translate_condition_with_context: empty result stack".into())
    })
}

fn flatten_and(
    left: llkv_expr::expr::Expr<'static, String>,
    right: llkv_expr::expr::Expr<'static, String>,
) -> llkv_expr::expr::Expr<'static, String> {
    let mut children: Vec<llkv_expr::expr::Expr<'static, String>> = Vec::new();
    match left {
        llkv_expr::expr::Expr::And(mut left_children) => children.append(&mut left_children),
        other => children.push(other),
    }
    match right {
        llkv_expr::expr::Expr::And(mut right_children) => children.append(&mut right_children),
        other => children.push(other),
    }
    if children.len() == 1 {
        children.into_iter().next().unwrap()
    } else {
        llkv_expr::expr::Expr::And(children)
    }
}

fn flatten_or(
    left: llkv_expr::expr::Expr<'static, String>,
    right: llkv_expr::expr::Expr<'static, String>,
) -> llkv_expr::expr::Expr<'static, String> {
    let mut children: Vec<llkv_expr::expr::Expr<'static, String>> = Vec::new();
    match left {
        llkv_expr::expr::Expr::Or(mut left_children) => children.append(&mut left_children),
        other => children.push(other),
    }
    match right {
        llkv_expr::expr::Expr::Or(mut right_children) => children.append(&mut right_children),
        other => children.push(other),
    }
    if children.len() == 1 {
        children.into_iter().next().unwrap()
    } else {
        llkv_expr::expr::Expr::Or(children)
    }
}

fn translate_comparison_with_context(
    resolver: &IdentifierResolver<'_>,
    context: IdentifierContext,
    left: &SqlExpr,
    op: BinaryOperator,
    right: &SqlExpr,
) -> SqlResult<llkv_expr::expr::Expr<'static, String>> {
    let left_scalar = translate_scalar_with_context(resolver, context.clone(), left)?;
    let right_scalar = translate_scalar_with_context(resolver, context, right)?;
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
        tracing::debug!(
            column = ?column,
            literal = ?literal,
            ?compare_op,
            "translate_comparison direct"
        );
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
        tracing::debug!(
            column = ?column,
            literal = ?literal,
            original_op = ?compare_op,
            flipped_op = ?flipped,
            "translate_comparison flipped"
        );
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
    tracing::debug!(?op, literal = ?literal, "compare_op_to_filter_operator input");
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
/// Translate scalar expression with knowledge of the FROM table context.
/// This allows us to properly distinguish schema.table.column from column.field.field.
fn translate_scalar_with_context(
    resolver: &IdentifierResolver<'_>,
    context: IdentifierContext,
    expr: &SqlExpr,
) -> SqlResult<llkv_expr::expr::ScalarExpr<String>> {
    translate_scalar_internal(expr, Some(resolver), Some(&context))
}

#[allow(dead_code)]
fn translate_scalar(expr: &SqlExpr) -> SqlResult<llkv_expr::expr::ScalarExpr<String>> {
    translate_scalar_internal(expr, None, None)
}

fn translate_scalar_internal(
    expr: &SqlExpr,
    resolver: Option<&IdentifierResolver<'_>>,
    context: Option<&IdentifierContext>,
) -> SqlResult<llkv_expr::expr::ScalarExpr<String>> {
    // Iterative postorder traversal using the TransformFrame pattern.
    // See llkv-plan::traversal module documentation for pattern details.
    //
    // This avoids stack overflow on deeply nested expressions (50k+ nodes) by using
    // explicit work_stack and result_stack instead of recursion.

    /// Context passed through Exit frames during scalar expression translation
    enum ScalarExitContext {
        BinaryOp { op: BinaryOperator },
        UnaryMinus,
        UnaryPlus,
        Nested,
    }

    type ScalarFrame<'a> =
        TransformFrame<'a, SqlExpr, llkv_expr::expr::ScalarExpr<String>, ScalarExitContext>;

    let mut work_stack: Vec<ScalarFrame> = vec![ScalarFrame::Enter(expr)];
    let mut result_stack: Vec<llkv_expr::expr::ScalarExpr<String>> = Vec::new();

    while let Some(frame) = work_stack.pop() {
        match frame {
            ScalarFrame::Enter(node) => match node {
                SqlExpr::Identifier(ident) => {
                    if let (Some(resolver), Some(ctx)) = (resolver, context) {
                        let parts = vec![ident.value.clone()];
                        let resolution = resolver.resolve(&parts, (*ctx).clone())?;
                        work_stack.push(ScalarFrame::Leaf(resolution.into_scalar_expr()));
                    } else {
                        work_stack.push(ScalarFrame::Leaf(llkv_expr::expr::ScalarExpr::column(
                            ident.value.clone(),
                        )));
                    }
                }
                SqlExpr::CompoundIdentifier(idents) => {
                    if idents.is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "invalid compound identifier".into(),
                        ));
                    }

                    if let (Some(resolver), Some(ctx)) = (resolver, context) {
                        let parts: Vec<String> =
                            idents.iter().map(|ident| ident.value.clone()).collect();
                        let resolution = resolver.resolve(&parts, (*ctx).clone())?;
                        work_stack.push(ScalarFrame::Leaf(resolution.into_scalar_expr()));
                    } else {
                        let column_name = idents[0].value.clone();
                        let mut result = llkv_expr::expr::ScalarExpr::column(column_name);

                        for part in &idents[1..] {
                            let field_name = part.value.clone();
                            result = llkv_expr::expr::ScalarExpr::get_field(result, field_name);
                        }

                        work_stack.push(ScalarFrame::Leaf(result));
                    }
                }
                SqlExpr::Value(value) => {
                    let result = literal_from_value(value)?;
                    work_stack.push(ScalarFrame::Leaf(result));
                }
                SqlExpr::BinaryOp { left, op, right } => {
                    work_stack.push(ScalarFrame::Exit(ScalarExitContext::BinaryOp {
                        op: op.clone(),
                    }));
                    work_stack.push(ScalarFrame::Enter(right));
                    work_stack.push(ScalarFrame::Enter(left));
                }
                SqlExpr::UnaryOp {
                    op: UnaryOperator::Minus,
                    expr: inner,
                } => {
                    work_stack.push(ScalarFrame::Exit(ScalarExitContext::UnaryMinus));
                    work_stack.push(ScalarFrame::Enter(inner));
                }
                SqlExpr::UnaryOp {
                    op: UnaryOperator::Plus,
                    expr: inner,
                } => {
                    work_stack.push(ScalarFrame::Exit(ScalarExitContext::UnaryPlus));
                    work_stack.push(ScalarFrame::Enter(inner));
                }
                SqlExpr::Nested(inner) => {
                    work_stack.push(ScalarFrame::Exit(ScalarExitContext::Nested));
                    work_stack.push(ScalarFrame::Enter(inner));
                }
                SqlExpr::Cast { expr: inner, .. } => {
                    // TODO: implement typed CAST semantics once executor supports runtime coercions.
                    // For now, treat CAST as a passthrough so the inner expression is evaluated normally.
                    work_stack.push(ScalarFrame::Enter(inner));
                }
                SqlExpr::Function(func) => {
                    if let Some(agg_call) = try_parse_aggregate_function(func)? {
                        work_stack.push(ScalarFrame::Leaf(llkv_expr::expr::ScalarExpr::aggregate(
                            agg_call,
                        )));
                    } else {
                        return Err(Error::InvalidArgumentError(format!(
                            "unsupported function in scalar expression: {:?}",
                            func.name
                        )));
                    }
                }
                SqlExpr::Dictionary(fields) => {
                    // Process dictionary fields iteratively to avoid recursion
                    let mut struct_fields = Vec::new();
                    for entry in fields {
                        let key = entry.key.value.clone();
                        // Reuse scalar translation for nested values while honoring identifier context.
                        // Dictionaries rarely nest deeply, so recursion here is acceptable.
                        let value_expr =
                            translate_scalar_internal(&entry.value, resolver, context)?;
                        match value_expr {
                            llkv_expr::expr::ScalarExpr::Literal(lit) => {
                                struct_fields.push((key, Box::new(lit)));
                            }
                            _ => {
                                return Err(Error::InvalidArgumentError(
                                    "Dictionary values must be literals".to_string(),
                                ));
                            }
                        }
                    }
                    work_stack.push(ScalarFrame::Leaf(llkv_expr::expr::ScalarExpr::literal(
                        Literal::Struct(struct_fields),
                    )));
                }
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported scalar expression: {other:?}"
                    )));
                }
            },
            ScalarFrame::Leaf(translated) => {
                result_stack.push(translated);
            }
            ScalarFrame::Exit(exit_context) => match exit_context {
                ScalarExitContext::BinaryOp { op } => {
                    let right_expr = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_scalar: result stack underflow for BinaryOp right".into(),
                        )
                    })?;
                    let left_expr = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_scalar: result stack underflow for BinaryOp left".into(),
                        )
                    })?;
                    let binary_op = match op {
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
                    result_stack.push(llkv_expr::expr::ScalarExpr::binary(
                        left_expr, binary_op, right_expr,
                    ));
                }
                ScalarExitContext::UnaryMinus => {
                    let inner = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_scalar: result stack underflow for UnaryMinus".into(),
                        )
                    })?;
                    match inner {
                        llkv_expr::expr::ScalarExpr::Literal(lit) => match lit {
                            Literal::Integer(v) => {
                                result_stack.push(llkv_expr::expr::ScalarExpr::literal(
                                    Literal::Integer(-v),
                                ));
                            }
                            Literal::Float(v) => {
                                result_stack
                                    .push(llkv_expr::expr::ScalarExpr::literal(Literal::Float(-v)));
                            }
                            Literal::Boolean(_) => {
                                return Err(Error::InvalidArgumentError(
                                    "cannot negate boolean literal".into(),
                                ));
                            }
                            Literal::String(_) => {
                                return Err(Error::InvalidArgumentError(
                                    "cannot negate string literal".into(),
                                ));
                            }
                            Literal::Struct(_) => {
                                return Err(Error::InvalidArgumentError(
                                    "cannot negate struct literal".into(),
                                ));
                            }
                            Literal::Null => {
                                result_stack
                                    .push(llkv_expr::expr::ScalarExpr::literal(Literal::Null));
                            }
                        },
                        other => {
                            let zero = llkv_expr::expr::ScalarExpr::literal(Literal::Integer(0));
                            result_stack.push(llkv_expr::expr::ScalarExpr::binary(
                                zero,
                                llkv_expr::expr::BinaryOp::Subtract,
                                other,
                            ));
                        }
                    }
                }
                ScalarExitContext::UnaryPlus => {
                    // Unary plus is a no-op - just pass through
                }
                ScalarExitContext::Nested => {
                    // Nested is a no-op - just pass through
                }
            },
        }
    }

    result_stack
        .pop()
        .ok_or_else(|| Error::Internal("translate_scalar: empty result stack".into()))
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
        Value::Boolean(value) => Ok(llkv_expr::expr::ScalarExpr::literal(Literal::Boolean(
            *value,
        ))),
        Value::Null => Ok(llkv_expr::expr::ScalarExpr::literal(Literal::Null)),
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
        SqlDataType::Boolean => Ok(DataType::Boolean),
        SqlDataType::Custom(name, args) => {
            if name.0.len() == 1
                && let ObjectNamePart::Identifier(ident) = &name.0[0]
                && ident.value.eq_ignore_ascii_case("row")
            {
                return row_type_to_arrow(data_type, args);
            }
            Err(Error::InvalidArgumentError(format!(
                "unsupported SQL data type: {data_type:?}"
            )))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported SQL data type: {other:?}"
        ))),
    }
}

fn row_type_to_arrow(
    data_type: &SqlDataType,
    tokens: &[String],
) -> SqlResult<arrow::datatypes::DataType> {
    use arrow::datatypes::{DataType, Field, FieldRef, Fields};

    let row_str = data_type.to_string();
    if tokens.is_empty() {
        return Err(Error::InvalidArgumentError(
            "ROW type must define at least one field".into(),
        ));
    }

    let dialect = GenericDialect {};
    let field_definitions = resolve_row_field_types(tokens, &dialect).map_err(|err| {
        Error::InvalidArgumentError(format!("unable to parse ROW type '{row_str}': {err}"))
    })?;

    let mut fields: Vec<FieldRef> = Vec::with_capacity(field_definitions.len());
    for (field_name, field_type) in field_definitions {
        let arrow_field_type = arrow_type_from_sql(&field_type)?;
        fields.push(Arc::new(Field::new(field_name, arrow_field_type, true)));
    }

    let struct_fields: Fields = fields.into();
    Ok(DataType::Struct(struct_fields))
}

fn resolve_row_field_types(
    tokens: &[String],
    dialect: &GenericDialect,
) -> SqlResult<Vec<(String, SqlDataType)>> {
    if tokens.is_empty() {
        return Err(Error::InvalidArgumentError(
            "ROW type must define at least one field".into(),
        ));
    }

    let mut start = 0;
    let mut end = tokens.len();
    if tokens[start] == "(" {
        if end == 0 || tokens[end - 1] != ")" {
            return Err(Error::InvalidArgumentError(
                "ROW type is missing closing ')'".into(),
            ));
        }
        start += 1;
        end -= 1;
    } else if tokens[end - 1] == ")" {
        return Err(Error::InvalidArgumentError(
            "ROW type contains unmatched ')'".into(),
        ));
    }

    let slice = &tokens[start..end];
    if slice.is_empty() {
        return Err(Error::InvalidArgumentError(
            "ROW type did not provide any field definitions".into(),
        ));
    }

    let mut fields = Vec::new();
    let mut index = 0;

    while index < slice.len() {
        if slice[index] == "," {
            index += 1;
            continue;
        }

        let field_name = normalize_row_field_name(&slice[index])?;
        index += 1;

        if index >= slice.len() {
            return Err(Error::InvalidArgumentError(format!(
                "ROW field '{field_name}' is missing a type specification"
            )));
        }

        let mut last_success: Option<(usize, SqlDataType)> = None;
        let mut type_end = index;

        while type_end <= slice.len() {
            let candidate = slice[index..type_end].join(" ");
            if candidate.trim().is_empty() {
                type_end += 1;
                continue;
            }

            if let Ok(parsed_type) = parse_sql_data_type(&candidate, dialect) {
                last_success = Some((type_end, parsed_type));
            }

            if type_end == slice.len() {
                break;
            }

            if slice[type_end] == "," && last_success.is_some() {
                break;
            }

            type_end += 1;
        }

        let Some((next_index, data_type)) = last_success else {
            return Err(Error::InvalidArgumentError(format!(
                "failed to parse ROW field type for '{field_name}'"
            )));
        };

        fields.push((field_name, data_type));
        index = next_index;

        if index < slice.len() && slice[index] == "," {
            index += 1;
        }
    }

    if fields.is_empty() {
        return Err(Error::InvalidArgumentError(
            "ROW type did not provide any field definitions".into(),
        ));
    }

    Ok(fields)
}

/// Parse SQL string into statements with increased recursion limit.
///
/// This helper wraps sqlparser's `Parser` with a custom recursion limit to handle
/// deeply nested queries that exceed the default limit of 50.
///
/// # Arguments
///
/// * `dialect` - SQL dialect to use for parsing
/// * `sql` - SQL string to parse
///
/// # Returns
///
/// Parsed statements or parser error
fn parse_sql_with_recursion_limit(
    dialect: &GenericDialect,
    sql: &str,
) -> Result<Vec<Statement>, sqlparser::parser::ParserError> {
    Parser::new(dialect)
        .with_recursion_limit(PARSER_RECURSION_LIMIT)
        .try_with_sql(sql)?
        .parse_statements()
}

fn normalize_row_field_name(raw: &str) -> SqlResult<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(Error::InvalidArgumentError(
            "ROW field name must not be empty".into(),
        ));
    }

    if let Some(stripped) = trimmed.strip_prefix('"') {
        let without_end = stripped.strip_suffix('"').ok_or_else(|| {
            Error::InvalidArgumentError(format!("unterminated quoted ROW field name: {trimmed}"))
        })?;
        let name = without_end.replace("\"\"", "\"");
        return Ok(name);
    }

    Ok(trimmed.to_string())
}

fn parse_sql_data_type(type_str: &str, dialect: &GenericDialect) -> SqlResult<SqlDataType> {
    let trimmed = type_str.trim();
    let sql = format!("CREATE TABLE __row(__field {trimmed});");
    let statements = parse_sql_with_recursion_limit(dialect, &sql).map_err(|err| {
        Error::InvalidArgumentError(format!("failed to parse ROW field type '{trimmed}': {err}"))
    })?;

    let stmt = statements.into_iter().next().ok_or_else(|| {
        Error::InvalidArgumentError(format!(
            "ROW field type '{trimmed}' did not produce a statement"
        ))
    })?;

    match stmt {
        Statement::CreateTable(table) => table
            .columns
            .first()
            .map(|col| col.data_type.clone())
            .ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "ROW field type '{trimmed}' missing column definition"
                ))
            }),
        other => Err(Error::InvalidArgumentError(format!(
            "unexpected statement while parsing ROW field type: {other:?}"
        ))),
    }
}

/// Extract VALUES data from a derived table in FROM clause.
/// Returns (rows, column_names) if the pattern matches: SELECT ... FROM (VALUES ...) alias(col1, col2, ...)
type ExtractValuesResult = Option<(Vec<Vec<PlanValue>>, Vec<String>)>;

#[allow(clippy::type_complexity)]
fn extract_values_from_derived_table(from: &[TableWithJoins]) -> SqlResult<ExtractValuesResult> {
    if from.len() != 1 {
        return Ok(None);
    }

    let table_with_joins = &from[0];
    if !table_with_joins.joins.is_empty() {
        return Ok(None);
    }

    match &table_with_joins.relation {
        TableFactor::Derived {
            subquery, alias, ..
        } => {
            // Check if the subquery is a VALUES expression
            let values = match subquery.body.as_ref() {
                SetExpr::Values(v) => v,
                _ => return Ok(None),
            };

            // Extract column names from alias
            let column_names = if let Some(alias) = alias {
                alias
                    .columns
                    .iter()
                    .map(|col_def| col_def.name.value.clone())
                    .collect::<Vec<_>>()
            } else {
                // Generate default column names if no alias provided
                if values.rows.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "VALUES expression must have at least one row".into(),
                    ));
                }
                let first_row = &values.rows[0];
                (0..first_row.len())
                    .map(|i| format!("column{}", i))
                    .collect()
            };

            // Extract rows
            if values.rows.is_empty() {
                return Err(Error::InvalidArgumentError(
                    "VALUES expression must have at least one row".into(),
                ));
            }

            let mut rows = Vec::with_capacity(values.rows.len());
            for row in &values.rows {
                if row.len() != column_names.len() {
                    return Err(Error::InvalidArgumentError(format!(
                        "VALUES row has {} columns but table alias specifies {} columns",
                        row.len(),
                        column_names.len()
                    )));
                }

                let mut converted_row = Vec::with_capacity(row.len());
                for expr in row {
                    let value = SqlValue::try_from_expr(expr)?;
                    converted_row.push(PlanValue::from(value));
                }
                rows.push(converted_row);
            }

            Ok(Some((rows, column_names)))
        }
        _ => Ok(None),
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
        TableFactor::Derived { alias, .. } => {
            // Derived table (subquery) - use the alias as the table name if provided
            // For CTAS, this allows: CREATE TABLE t AS SELECT * FROM (VALUES ...) v(id)
            let table_name = alias
                .as_ref()
                .map(|a| a.name.value.clone())
                .unwrap_or_else(|| "derived".to_string());
            let canonical = table_name.to_ascii_lowercase();
            Ok((table_name, canonical))
        }
        _ => Err(Error::InvalidArgumentError(
            "queries require a plain table name or derived table".into(),
        )),
    }
}

/// Extract table references from a FROM clause, flattening supported JOINs.
fn extract_tables(from: &[TableWithJoins]) -> SqlResult<Vec<llkv_plan::TableRef>> {
    let mut tables = Vec::new();

    for item in from {
        push_table_factor(&item.relation, &mut tables)?;

        for join in &item.joins {
            match &join.join_operator {
                JoinOperator::CrossJoin(JoinConstraint::None)
                | JoinOperator::Inner(JoinConstraint::None) => {
                    push_table_factor(&join.relation, &mut tables)?;
                }
                JoinOperator::CrossJoin(_) => {
                    return Err(Error::InvalidArgumentError(
                        "CROSS JOIN with constraints is not supported".into(),
                    ));
                }
                _ => {
                    return Err(Error::InvalidArgumentError(
                        "only CROSS JOIN without constraints is supported".into(),
                    ));
                }
            }
        }
    }

    Ok(tables)
}

fn push_table_factor(factor: &TableFactor, tables: &mut Vec<llkv_plan::TableRef>) -> SqlResult<()> {
    match factor {
        TableFactor::Table { name, alias, .. } => {
            let (schema_opt, table) = parse_schema_qualified_name(name)?;
            let schema = schema_opt.unwrap_or_default();
            let alias_name = alias.as_ref().map(|a| a.name.value.clone());
            tables.push(llkv_plan::TableRef::with_alias(schema, table, alias_name));
            Ok(())
        }
        TableFactor::Derived { .. } => Err(Error::InvalidArgumentError(
            "JOIN clauses require base tables; derived tables are not supported".into(),
        )),
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
        assert!(matches!(
            result[0],
            RuntimeStatementResult::CreateTable { .. }
        ));

        let result = engine
            .execute("INSERT INTO people (id, name) VALUES (1, 'alice'), (2, 'bob')")
            .expect("insert rows");
        assert!(matches!(
            result[0],
            RuntimeStatementResult::Insert {
                rows_inserted: 2,
                ..
            }
        ));

        let mut result = engine
            .execute("SELECT name FROM people WHERE id = 2")
            .expect("select rows");
        let select_result = result.remove(0);
        let batches = match select_result {
            RuntimeStatementResult::Select { execution, .. } => {
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
            RuntimeStatementResult::Insert {
                rows_inserted: 1,
                ..
            }
        ));

        let result = engine
            .execute("INSERT INTO integers SELECT CAST(NULL AS VARCHAR)")
            .expect("insert null literal");
        assert!(matches!(
            result[0],
            RuntimeStatementResult::Insert {
                rows_inserted: 1,
                ..
            }
        ));

        let mut result = engine
            .execute("SELECT * FROM integers")
            .expect("select rows");
        let select_result = result.remove(0);
        let batches = match select_result {
            RuntimeStatementResult::Select { execution, .. } => {
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
    fn not_null_comparison_filters_all_rows() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE single(col INTEGER)")
            .expect("create table");
        engine
            .execute("INSERT INTO single VALUES (1)")
            .expect("insert row");

        let batches = engine
            .sql("SELECT * FROM single WHERE NOT ( NULL ) >= NULL")
            .expect("run constant null comparison");

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(total_rows, 0, "expected filter to remove all rows");
    }

    #[test]
    fn cross_join_not_null_comparison_filters_all_rows() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE left_side(col INTEGER)")
            .expect("create left table");
        engine
            .execute("CREATE TABLE right_side(col INTEGER)")
            .expect("create right table");
        engine
            .execute("INSERT INTO left_side VALUES (1)")
            .expect("insert left row");
        engine
            .execute("INSERT INTO right_side VALUES (2)")
            .expect("insert right row");

        let batches = engine
            .sql("SELECT * FROM left_side CROSS JOIN right_side WHERE NOT ( NULL ) >= NULL")
            .expect("run cross join null comparison");

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(
            total_rows, 0,
            "expected cross join filter to remove all rows"
        );
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
            RuntimeStatementResult::Update {
                rows_updated: 1,
                ..
            }
        ));

        let mut result = engine
            .execute("SELECT * FROM strings ORDER BY cast(a AS INTEGER)")
            .expect("select rows");
        let select_result = result.remove(0);
        let batches = match select_result {
            RuntimeStatementResult::Select { execution, .. } => {
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
            RuntimeStatementResult::Select { execution, .. } => {
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
            RuntimeStatementResult::Select { execution, .. } => {
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

    #[test]
    fn arrow_type_from_row_returns_struct_fields() {
        let dialect = GenericDialect {};
        let statements = parse_sql_with_recursion_limit(
            &dialect,
            "CREATE TABLE row_types(payload ROW(a INTEGER, b VARCHAR));",
        )
        .expect("parse ROW type definition");

        let data_type = match &statements[0] {
            Statement::CreateTable(stmt) => stmt.columns[0].data_type.clone(),
            other => panic!("unexpected statement: {other:?}"),
        };

        let arrow_type = arrow_type_from_sql(&data_type).expect("convert ROW type");
        match arrow_type {
            arrow::datatypes::DataType::Struct(fields) => {
                assert_eq!(fields.len(), 2, "unexpected field count");
                assert_eq!(fields[0].name(), "a");
                assert_eq!(fields[1].name(), "b");
                assert_eq!(fields[0].data_type(), &arrow::datatypes::DataType::Int64);
                assert_eq!(fields[1].data_type(), &arrow::datatypes::DataType::Utf8);
            }
            other => panic!("expected struct type, got {other:?}"),
        }
    }
}
