use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::convert::TryFrom;
use std::ops::Bound;
use std::sync::{
    Arc, OnceLock, RwLock,
    atomic::{AtomicBool, Ordering as AtomicOrdering},
};

use crate::{SqlResult, SqlValue, interval::parse_interval_literal};
use arrow::array::{Array, ArrayRef, BooleanArray, Int32Array, StringArray, UInt32Array};
use arrow::compute::{concat_batches, take};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, SortField};

use llkv_executor::{SelectExecution, push_query_label};
use llkv_expr::decimal::DecimalValue;
use llkv_expr::literal::Literal;
use llkv_plan::validation::{
    ensure_known_columns_case_insensitive, ensure_non_empty, ensure_unique_case_insensitive,
};
use llkv_plan::{SubqueryCorrelatedColumnTracker, SubqueryCorrelatedTracker, TransformFrame};
use llkv_result::Error;
use llkv_runtime::TEMPORARY_NAMESPACE_ID;
use llkv_runtime::{
    AggregateExpr, AssignmentValue, ColumnAssignment, ColumnStoreWriteHints, CreateIndexPlan,
    CreateTablePlan, CreateTableSource, CreateViewPlan, DeletePlan,
    ForeignKeyAction as PlanForeignKeyAction, ForeignKeySpec, IndexColumnPlan,
    InsertConflictAction, InsertPlan, InsertSource, MultiColumnUniqueSpec, OrderByPlan,
    OrderSortType, OrderTarget, PlanColumnSpec, PlanStatement, PlanValue, ReindexPlan,
    RenameTablePlan, RuntimeContext, RuntimeEngine, RuntimeSession, RuntimeStatementResult,
    SelectPlan, SelectProjection, TruncatePlan, UpdatePlan, extract_rows_from_range,
};
use llkv_storage::pager::{BoxedPager, Pager};
use llkv_table::catalog::{ColumnResolution, IdentifierContext, IdentifierResolver};
use llkv_table::{CatalogDdl, ConstraintEnforcementMode, TriggerEventMeta, TriggerTimingMeta};
use regex::Regex;
use simd_r_drive_entry_handle::EntryHandle;
use sqlparser::ast::{
    AlterColumnOperation, AlterTableOperation, Assignment, AssignmentTarget, BeginTransactionKind,
    BinaryOperator, ColumnOption, ColumnOptionDef, ConstraintCharacteristics, CreateTrigger,
    DataType as SqlDataType, Delete, Distinct, DropTrigger, ExceptionWhen, Expr as SqlExpr,
    FromTable, FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Ident, JoinConstraint,
    JoinOperator, LimitClause, NullsDistinctOption, ObjectName, ObjectNamePart, ObjectType,
    OrderBy, OrderByKind, Query, ReferentialAction, SchemaName, Select, SelectItem,
    SelectItemQualifiedWildcardKind, Set, SetExpr, SetOperator, SetQuantifier, SqlOption,
    Statement, TableConstraint, TableFactor, TableObject, TableWithJoins, TransactionMode,
    TransactionModifier, TriggerEvent, TriggerObject, TriggerPeriod, UnaryOperator,
    UpdateTableFromKind, VacuumStatement, Value, ValueWithSpan,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::Span;

type SqlPager = BoxedPager;
type SqlRuntimePager = SqlPager;
type SqlStatementResult = RuntimeStatementResult<SqlPager>;
type SqlContext = RuntimeContext<SqlPager>;
type SqlSession = RuntimeSession;
type P = SqlRuntimePager;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatementExpectation {
    Ok,
    Error,
    Count(u64),
}

thread_local! {
    static PENDING_STATEMENT_EXPECTATIONS: RefCell<VecDeque<StatementExpectation>> = const {
        RefCell::new(VecDeque::new())
    };
}

pub(crate) const PARAM_SENTINEL_PREFIX: &str = "__llkv_param__";
pub(crate) const PARAM_SENTINEL_SUFFIX: &str = "__";

thread_local! {
    static ACTIVE_PARAMETER_STATE: RefCell<Option<ParameterState>> = const {
        RefCell::new(None)
    };
}

#[derive(Default)]
struct ParameterState {
    assigned: FxHashMap<String, usize>,
    next_auto: usize,
    max_index: usize,
}

impl ParameterState {
    fn register(&mut self, raw: &str) -> SqlResult<usize> {
        if raw == "?" {
            self.next_auto += 1;
            let idx = self.next_auto;
            self.max_index = self.max_index.max(idx);
            return Ok(idx);
        }

        if let Some(&idx) = self.assigned.get(raw) {
            return Ok(idx);
        }

        let idx = if let Some(rest) = raw.strip_prefix('?') {
            parse_numeric_placeholder(rest, raw)?
        } else if let Some(rest) = raw.strip_prefix('$') {
            parse_numeric_placeholder(rest, raw)?
        } else if let Some(rest) = raw.strip_prefix(':') {
            if rest.is_empty() {
                return Err(Error::InvalidArgumentError(
                    "named parameters must include an identifier".into(),
                ));
            }
            self.max_index + 1
        } else {
            return Err(Error::InvalidArgumentError(format!(
                "unsupported SQL parameter placeholder: {raw}",
            )));
        };

        self.assigned.insert(raw.to_string(), idx);
        self.max_index = self.max_index.max(idx);
        self.next_auto = self.next_auto.max(idx);
        Ok(idx)
    }

    fn max_index(&self) -> usize {
        self.max_index
    }
}

fn parse_numeric_placeholder(text: &str, raw: &str) -> SqlResult<usize> {
    if text.is_empty() {
        return Err(Error::InvalidArgumentError(format!(
            "parameter placeholder '{raw}' is missing an index",
        )));
    }
    text.parse::<usize>().map_err(|_| {
        Error::InvalidArgumentError(format!(
            "parameter placeholder '{raw}' must end with digits"
        ))
    })
}

struct ParameterScope {
    finished: bool,
}

impl ParameterScope {
    fn new() -> Self {
        ACTIVE_PARAMETER_STATE.with(|cell| {
            debug_assert!(
                cell.borrow().is_none(),
                "nested parameter scopes not supported"
            );
            *cell.borrow_mut() = Some(ParameterState::default());
        });
        Self { finished: false }
    }

    fn finish(mut self) -> ParameterState {
        let state = ACTIVE_PARAMETER_STATE
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        self.finished = true;
        state
    }
}

impl Drop for ParameterScope {
    fn drop(&mut self) {
        if !self.finished {
            ACTIVE_PARAMETER_STATE.with(|cell| {
                cell.borrow_mut().take();
            });
        }
    }
}

pub(crate) fn register_placeholder(raw: &str) -> SqlResult<usize> {
    ACTIVE_PARAMETER_STATE.with(|cell| {
        let mut guard = cell.borrow_mut();
        let state = guard.as_mut().ok_or_else(|| {
            Error::InvalidArgumentError(
                "SQL parameters can only be used with prepared statements".into(),
            )
        })?;
        state.register(raw)
    })
}

pub(crate) fn placeholder_marker(index: usize) -> String {
    format!("{PARAM_SENTINEL_PREFIX}{index}{PARAM_SENTINEL_SUFFIX}")
}

pub(crate) fn literal_placeholder(index: usize) -> Literal {
    Literal::String(placeholder_marker(index))
}

fn parse_placeholder_marker(text: &str) -> Option<usize> {
    let stripped = text.strip_prefix(PARAM_SENTINEL_PREFIX)?;
    let numeric = stripped.strip_suffix(PARAM_SENTINEL_SUFFIX)?;
    numeric.parse().ok()
}

#[derive(Clone, Debug)]
pub enum SqlParamValue {
    Null,
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Date32(i32),
}

impl SqlParamValue {
    fn as_literal(&self) -> Literal {
        match self {
            SqlParamValue::Null => Literal::Null,
            SqlParamValue::Integer(v) => Literal::Integer(i128::from(*v)),
            SqlParamValue::Float(v) => Literal::Float(*v),
            SqlParamValue::Boolean(v) => Literal::Boolean(*v),
            SqlParamValue::String(s) => Literal::String(s.clone()),
            SqlParamValue::Date32(days) => Literal::Date32(*days),
        }
    }

    fn as_plan_value(&self) -> PlanValue {
        match self {
            SqlParamValue::Null => PlanValue::Null,
            SqlParamValue::Integer(v) => PlanValue::Integer(*v),
            SqlParamValue::Float(v) => PlanValue::Float(*v),
            SqlParamValue::Boolean(v) => PlanValue::Integer(if *v { 1 } else { 0 }),
            SqlParamValue::String(s) => PlanValue::String(s.clone()),
            SqlParamValue::Date32(days) => PlanValue::Date32(*days),
        }
    }
}

impl From<i64> for SqlParamValue {
    fn from(value: i64) -> Self {
        SqlParamValue::Integer(value)
    }
}

impl From<f64> for SqlParamValue {
    fn from(value: f64) -> Self {
        SqlParamValue::Float(value)
    }
}

impl From<bool> for SqlParamValue {
    fn from(value: bool) -> Self {
        SqlParamValue::Boolean(value)
    }
}

impl From<String> for SqlParamValue {
    fn from(value: String) -> Self {
        SqlParamValue::String(value)
    }
}

impl From<i32> for SqlParamValue {
    fn from(value: i32) -> Self {
        SqlParamValue::Date32(value)
    }
}

impl From<&str> for SqlParamValue {
    fn from(value: &str) -> Self {
        SqlParamValue::String(value.to_string())
    }
}

#[derive(Clone)]
struct PreparedPlan {
    plan: PlanStatement,
    param_count: usize,
}

#[derive(Clone)]
pub struct PreparedStatement {
    inner: Arc<PreparedPlan>,
}

impl PreparedStatement {
    fn new(inner: Arc<PreparedPlan>) -> Self {
        Self { inner }
    }

    pub fn parameter_count(&self) -> usize {
        self.inner.param_count
    }
}

pub fn register_statement_expectation(expectation: StatementExpectation) {
    PENDING_STATEMENT_EXPECTATIONS.with(|queue| {
        queue.borrow_mut().push_back(expectation);
    });
}

pub fn clear_pending_statement_expectations() {
    PENDING_STATEMENT_EXPECTATIONS.with(|queue| {
        queue.borrow_mut().clear();
    });
}

fn next_statement_expectation() -> StatementExpectation {
    PENDING_STATEMENT_EXPECTATIONS
        .with(|queue| queue.borrow_mut().pop_front())
        .unwrap_or(StatementExpectation::Ok)
}

// TODO: Extract to constants.rs
// TODO: Rename to SQL_PARSER_RECURSION_LIMIT
/// Maximum recursion depth for SQL parser.
///
/// The default in sqlparser is 50, which can be exceeded by deeply nested queries
/// (e.g., SQLite test suite). This value allows for more complex expressions while
/// still preventing stack overflows.
const PARSER_RECURSION_LIMIT: usize = 200;

trait ScalarSubqueryResolver {
    fn handle_scalar_subquery(
        &mut self,
        subquery: &Query,
        resolver: &IdentifierResolver<'_>,
        context: &IdentifierContext,
        outer_scopes: &[IdentifierContext],
    ) -> SqlResult<llkv_expr::expr::ScalarExpr<String>>;
}

/// Helper trait for requesting placeholders directly from catalog resolutions.
trait SubqueryCorrelatedTrackerExt {
    fn placeholder_for_resolution(
        &mut self,
        resolution: &llkv_table::catalog::ColumnResolution,
    ) -> Option<String>;
}

impl SubqueryCorrelatedTrackerExt for SubqueryCorrelatedTracker<'_> {
    fn placeholder_for_resolution(
        &mut self,
        resolution: &llkv_table::catalog::ColumnResolution,
    ) -> Option<String> {
        self.placeholder_for_column_path(resolution.column(), resolution.field_path())
    }
}

/// Convenience extension so optional tracker references can be reborrowed without
/// repeating `as_mut` callers across translation helpers.
trait SubqueryCorrelatedTrackerOptionExt {
    fn reborrow(&mut self) -> Option<&mut SubqueryCorrelatedColumnTracker>;
}

impl SubqueryCorrelatedTrackerOptionExt for Option<&mut SubqueryCorrelatedColumnTracker> {
    fn reborrow(&mut self) -> Option<&mut SubqueryCorrelatedColumnTracker> {
        self.as_mut().map(|tracker| &mut **tracker)
    }
}

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
/// Maximum number of literal `VALUES` rows to accumulate before forcing a flush.
///
/// This keeps memory usage predictable when ingesting massive SQL scripts while still
/// providing large batched inserts for throughput.
const MAX_BUFFERED_INSERT_ROWS: usize = 8192;

/// Accumulates literal `INSERT` payloads so multiple statements can be flushed together.
///
/// Each buffered statement tracks its individual row count while sharing a single literal
/// payload vector. When the buffer flushes we can emit one `RuntimeStatementResult::Insert`
/// per original statement without re-planning intermediate work.
struct InsertBuffer {
    table_name: String,
    columns: Vec<String>,
    /// Conflict resolution action
    on_conflict: InsertConflictAction,
    /// Total literal rows gathered so far (sums `statement_row_counts`).
    total_rows: usize,
    /// Row counts for each original INSERT statement so we can emit per-statement results.
    statement_row_counts: Vec<usize>,
    /// Literal row payloads in execution order.
    rows: Vec<Vec<PlanValue>>,
}

impl InsertBuffer {
    fn new(
        table_name: String,
        columns: Vec<String>,
        rows: Vec<Vec<PlanValue>>,
        on_conflict: InsertConflictAction,
    ) -> Self {
        let row_count = rows.len();
        Self {
            table_name,
            columns,
            on_conflict,
            total_rows: row_count,
            statement_row_counts: vec![row_count],
            rows,
        }
    }

    fn can_accept(
        &self,
        table_name: &str,
        columns: &[String],
        on_conflict: InsertConflictAction,
    ) -> bool {
        self.table_name == table_name && self.columns == columns && self.on_conflict == on_conflict
    }

    fn push_statement(&mut self, rows: Vec<Vec<PlanValue>>) {
        let row_count = rows.len();
        self.total_rows += row_count;
        self.statement_row_counts.push(row_count);
        self.rows.extend(rows);
    }

    fn should_flush(&self) -> bool {
        self.total_rows >= MAX_BUFFERED_INSERT_ROWS
    }
}

/// Describes how a parsed `INSERT` should flow through the execution pipeline after we
/// canonicalize the AST.
///
/// Literal `VALUES` payloads (including constant folds such as `SELECT 1`) are rewritten into
/// [`PlanValue`] rows so they can be stitched together with other buffered statements before we
/// hit the planner. Non-literal sources stay as full [`InsertPlan`]s and execute immediately.
enum PreparedInsert {
    Values {
        table_name: String,
        columns: Vec<String>,
        rows: Vec<Vec<PlanValue>>,
        on_conflict: InsertConflictAction,
    },
    Immediate(InsertPlan),
}

/// Return value from [`SqlEngine::buffer_insert`], exposing any buffered flush results along with
/// the row-count placeholder for the currently processed statement.
struct BufferedInsertResult {
    flushed: Vec<SqlStatementResult>,
    current: Option<SqlStatementResult>,
}

pub struct SqlEngine {
    engine: RuntimeEngine,
    default_nulls_first: AtomicBool,
    /// Buffer for batching INSERTs across execute() calls for massive performance gains.
    insert_buffer: RefCell<Option<InsertBuffer>>,
    /// Tracks whether cross-statement INSERT buffering is enabled for this engine instance.
    ///
    /// Batch mode is disabled by default so unit tests and non-bulk ingest callers observe the
    /// runtime's native per-statement semantics. Long-running workloads (for example, the SLT
    /// harness) can opt in via [`SqlEngine::set_insert_buffering`] to trade immediate visibility
    /// for much lower planning overhead.
    insert_buffering_enabled: AtomicBool,
    information_schema_ready: AtomicBool,
    statement_cache: RwLock<FxHashMap<String, Arc<PreparedPlan>>>,
}

const DROPPED_TABLE_TRANSACTION_ERR: &str = "another transaction has dropped this table";

impl Drop for SqlEngine {
    fn drop(&mut self) {
        // Flush remaining INSERTs when engine is dropped
        if let Err(e) = self.flush_buffer_results() {
            tracing::warn!("Failed to flush INSERT buffer on drop: {:?}", e);
        }
    }
}

impl Clone for SqlEngine {
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
            insert_buffer: RefCell::new(None),
            insert_buffering_enabled: AtomicBool::new(
                self.insert_buffering_enabled.load(AtomicOrdering::Relaxed),
            ),
            information_schema_ready: AtomicBool::new(
                self.information_schema_ready.load(AtomicOrdering::Relaxed),
            ),
            statement_cache: RwLock::new(FxHashMap::default()),
        }
    }
}

#[allow(dead_code)]
impl SqlEngine {
    /// Instantiate a new SQL engine from an existing context.
    ///
    /// The `default_nulls_first` parameter controls the default sort order for `NULL` values.
    pub fn with_context(context: Arc<SqlContext>, default_nulls_first: bool) -> Self {
        Self::from_runtime_engine(
            RuntimeEngine::from_context(context),
            default_nulls_first,
            false,
        )
    }

    /// Expose the underlying runtime context for advanced callers (bulk loaders, tooling).
    ///
    /// This should only be used when the higher-level SQL interface lacks the necessary
    /// hook—for example, enabling specialized constraint caches or inspecting catalog state.
    pub fn runtime_context(&self) -> Arc<SqlContext> {
        self.engine.context()
    }

    /// Fetch write-sizing hints from the underlying column store.
    pub fn column_store_write_hints(&self) -> ColumnStoreWriteHints {
        self.runtime_context().column_store_write_hints()
    }

    fn ensure_information_schema_ready(&self) -> SqlResult<()> {
        if !self
            .information_schema_ready
            .load(AtomicOrdering::Acquire)
        {
            self.engine.refresh_information_schema()?;
            self.information_schema_ready
                .store(true, AtomicOrdering::Release);
        }
        Ok(())
    }

    fn from_runtime_engine(
        engine: RuntimeEngine,
        default_nulls_first: bool,
        insert_buffering_enabled: bool,
    ) -> Self {
        Self {
            engine,
            default_nulls_first: AtomicBool::new(default_nulls_first),
            insert_buffer: RefCell::new(None),
            insert_buffering_enabled: AtomicBool::new(insert_buffering_enabled),
            information_schema_ready: AtomicBool::new(false),
            statement_cache: RwLock::new(FxHashMap::default()),
        }
    }

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

    fn execute_plan_statement(&self, statement: PlanStatement) -> SqlResult<SqlStatementResult> {
        // Don't apply table error mapping to CREATE VIEW or DROP VIEW statements
        // because the "table" name is the view being created/dropped, not a referenced table.
        // Any "unknown table" errors from CREATE VIEW are about tables referenced in the SELECT.
        let should_map_error = !matches!(
            &statement,
            PlanStatement::CreateView(_) | PlanStatement::DropView(_)
        );

        let table = if should_map_error {
            llkv_runtime::statement_table_name(&statement).map(str::to_string)
        } else {
            None
        };

        self.engine.execute_statement(statement).map_err(|err| {
            if let Some(table_name) = table {
                Self::map_table_error(&table_name, err)
            } else {
                err
            }
        })
    }

    /// Construct a new engine backed by the provided pager with insert buffering disabled.
    ///
    /// Callers that intend to stream large amounts of literal `INSERT ... VALUES` input can
    /// enable batching later using [`SqlEngine::set_insert_buffering`].
    pub fn new<Pg>(pager: Arc<Pg>) -> Self
    where
        Pg: Pager<Blob = EntryHandle> + Send + Sync + 'static,
    {
        let engine = RuntimeEngine::new(pager);
        Self::from_runtime_engine(engine, false, false)
    }

    /// Strip TPC-H style `CONNECT TO database;` statements.
    ///
    /// The TPCH tooling emits `CONNECT TO <name>;` directives inside the referential
    /// integrity script even though LLKV always operates within a single database.
    /// Treat these commands as no-ops so the scripts can run unmodified.
    fn preprocess_tpch_connect_syntax(sql: &str) -> String {
        crate::tpch::strip_tpch_connect_statements(sql)
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

    /// Preprocess SQL to handle empty IN lists.
    ///
    /// SQLite permits `expr IN ()` and `expr NOT IN ()` as degenerate forms of IN expressions.
    /// The sqlparser library rejects these, so we convert them to constant boolean expressions:
    /// `expr IN ()` becomes `expr = NULL AND 0 = 1` (always false), and `expr NOT IN ()`
    /// becomes `expr = NULL OR 1 = 1` (always true). The `expr = NULL` component ensures the
    /// original expression is still evaluated (in case it has side effects), while the constant
    /// comparison determines the final result.
    fn preprocess_empty_in_lists(sql: &str) -> String {
        static EMPTY_IN_REGEX: OnceLock<Regex> = OnceLock::new();

        // Match: expression NOT IN () or expression IN ()
        // Matches: parenthesized expressions, quoted strings, hex literals, identifiers, or numbers
        let re = EMPTY_IN_REGEX.get_or_init(|| {
            Regex::new(r"(?i)(\([^)]*\)|x'[0-9a-fA-F]*'|'(?:[^']|'')*'|[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*|\d+(?:\.\d+)?)\s+(NOT\s+)?IN\s*\(\s*\)")
                .expect("valid empty IN regex")
        });

        re.replace_all(sql, |caps: &regex::Captures| {
            let expr = &caps[1];
            if caps.get(2).is_some() {
                // expr NOT IN () → always true (but still evaluate expr)
                format!("({} = NULL OR 1 = 1)", expr)
            } else {
                // expr IN () → always false (but still evaluate expr)
                format!("({} = NULL AND 0 = 1)", expr)
            }
        })
        .to_string()
    }

    /// Strip SQLite index hints (INDEXED BY, NOT INDEXED) from table references.
    ///
    /// SQLite supports index hints like `FROM table INDEXED BY index_name` or `FROM table NOT INDEXED`.
    /// These are query optimization hints that guide the query planner's index selection.
    /// Since sqlparser doesn't support this syntax and our planner makes its own index decisions,
    /// we strip these hints during preprocessing for compatibility with SQLite SQL Logic Tests.
    fn preprocess_index_hints(sql: &str) -> String {
        static INDEX_HINT_REGEX: OnceLock<Regex> = OnceLock::new();

        // Match: INDEXED BY index_name or NOT INDEXED
        // Pattern captures the table reference and removes the index hint
        let re = INDEX_HINT_REGEX.get_or_init(|| {
            Regex::new(r"(?i)\s+(INDEXED\s+BY\s+[a-zA-Z_][a-zA-Z0-9_]*|NOT\s+INDEXED)\b")
                .expect("valid index hint regex")
        });

        re.replace_all(sql, "").to_string()
    }

    /// Convert SQLite standalone REINDEX to VACUUM REINDEX for sqlparser.
    ///
    /// SQLite supports `REINDEX index_name` as a standalone statement, but sqlparser
    /// only recognizes REINDEX as part of the VACUUM statement syntax. This preprocessor
    /// converts the SQLite form to the VACUUM REINDEX form that sqlparser can parse.
    fn preprocess_reindex_syntax(sql: &str) -> String {
        static REINDEX_REGEX: OnceLock<Regex> = OnceLock::new();

        // Match: REINDEX followed by an identifier (index name)
        // Captures the full statement to replace it with VACUUM REINDEX
        let re = REINDEX_REGEX.get_or_init(|| {
            Regex::new(r"(?i)\bREINDEX\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\b")
                .expect("valid reindex regex")
        });

        re.replace_all(sql, "VACUUM REINDEX $1").to_string()
    }

    /// Normalize SQLite trigger shorthand so sqlparser accepts the syntax.
    ///
    /// SQLite allows omitting the trigger timing (defaulting to AFTER) and the
    /// `FOR EACH ROW` clause (defaulting to row-level triggers). sqlparser
    /// requires both pieces to be explicit, so we inject them before parsing.
    ///
    /// # TODO
    ///
    /// This is a temporary workaround. The proper fix is to extend sqlparser's
    /// `SQLiteDialect::parse_statement` to handle CREATE TRIGGER with optional
    /// timing/FOR EACH ROW clauses, matching SQLite's actual grammar. That would
    /// eliminate this fragile regex preprocessing entirely.
    fn preprocess_sqlite_trigger_shorthand(sql: &str) -> String {
        static IDENT_PATTERN: OnceLock<String> = OnceLock::new();
        static COLUMN_IDENT_PATTERN: OnceLock<String> = OnceLock::new();
        static TIMING_REGEX: OnceLock<Regex> = OnceLock::new();
        static FOR_EACH_BEGIN_REGEX: OnceLock<Regex> = OnceLock::new();
        static FOR_EACH_WHEN_REGEX: OnceLock<Regex> = OnceLock::new();

        IDENT_PATTERN.get_or_init(|| {
            // Matches optional dotted identifiers with standard or quoted segments.
            r#"(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[a-zA-Z_][a-zA-Z0-9_]*)(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*"#
                .to_string()
        });
        COLUMN_IDENT_PATTERN
            .get_or_init(|| r#"(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[a-zA-Z_][a-zA-Z0-9_]*)"#.to_string());

        let timing_re = TIMING_REGEX.get_or_init(|| {
            let event = format!(
                "UPDATE(?:\\s+OF\\s+{col}(?:\\s*,\\s*{col})*)?|DELETE|INSERT",
                col = COLUMN_IDENT_PATTERN
                    .get()
                    .expect("column ident pattern initialized")
            );
            let pattern = format!(
                r"(?ix)(?P<head>CREATE\s+TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?{ident})\s+(?P<event>{event})\s+ON",
                ident = IDENT_PATTERN
                    .get()
                    .expect("ident pattern initialized"),
                event = event
            );
            Regex::new(&pattern).expect("valid trigger timing regex")
        });

        let with_timing = timing_re
            .replace_all(sql, |caps: &regex::Captures| {
                let head = caps.name("head").unwrap().as_str();
                let event = caps.name("event").unwrap().as_str().trim();
                format!("{head} AFTER {event} ON")
            })
            .to_string();

        let for_each_begin_re = FOR_EACH_BEGIN_REGEX.get_or_init(|| {
            let pattern = format!(
                r"(?ix)(?P<prefix>ON\s+{ident})\s+(?P<keyword>BEGIN\b)",
                ident = IDENT_PATTERN.get().expect("ident pattern initialized"),
            );
            Regex::new(&pattern).expect("valid trigger FOR EACH BEGIN regex")
        });

        let with_for_each_begin = for_each_begin_re
            .replace_all(&with_timing, |caps: &regex::Captures| {
                let prefix = caps.name("prefix").unwrap().as_str();
                let keyword = caps.name("keyword").unwrap().as_str();
                format!("{prefix} FOR EACH ROW {keyword}")
            })
            .to_string();

        let for_each_when_re = FOR_EACH_WHEN_REGEX.get_or_init(|| {
            let pattern = format!(
                r"(?ix)(?P<prefix>ON\s+{ident})\s+(?P<keyword>WHEN\b)",
                ident = IDENT_PATTERN.get().expect("ident pattern initialized"),
            );
            Regex::new(&pattern).expect("valid trigger FOR EACH WHEN regex")
        });

        for_each_when_re
            .replace_all(&with_for_each_begin, |caps: &regex::Captures| {
                let prefix = caps.name("prefix").unwrap().as_str();
                let keyword = caps.name("keyword").unwrap().as_str();
                format!("{prefix} FOR EACH ROW {keyword}")
            })
            .to_string()
    }

    /// Preprocess SQL to convert bare table names in IN clauses to subqueries.
    ///
    /// SQLite allows `expr IN tablename` as shorthand for `expr IN (SELECT * FROM tablename)`.
    /// The sqlparser library requires parentheses, so we convert the shorthand form.
    fn preprocess_bare_table_in_clauses(sql: &str) -> String {
        static BARE_TABLE_IN_REGEX: OnceLock<Regex> = OnceLock::new();

        // Match: [NOT] IN identifier followed by whitespace/newline/end or specific punctuation
        // Avoid matching "IN (" which is already a valid subquery
        let re = BARE_TABLE_IN_REGEX.get_or_init(|| {
            Regex::new(r"(?i)\b(NOT\s+)?IN\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)(\s|$|;|,|\))")
                .expect("valid bare table IN regex")
        });

        re.replace_all(sql, |caps: &regex::Captures| {
            let table_name = &caps[2];
            let trailing = &caps[3];
            if let Some(not_keyword) = caps.get(1) {
                format!(
                    "{}IN (SELECT * FROM {}){}",
                    not_keyword.as_str(),
                    table_name,
                    trailing
                )
            } else {
                format!("IN (SELECT * FROM {}){}", table_name, trailing)
            }
        })
        .to_string()
    }

    /// Toggle literal `INSERT` buffering for the engine.
    ///
    /// When enabled, consecutive `INSERT ... VALUES` statements that target the same table and
    /// column list are accumulated and flushed together, dramatically lowering planning and
    /// execution overhead for workloads that stream tens of thousands of literal inserts.
    /// Disabling buffering reverts to SQLite-style immediate execution and is appropriate for
    /// unit tests or workloads that rely on per-statement side effects (errors, triggers,
    /// constraint violations) happening synchronously.
    ///
    /// Calling this method with `false` forces any pending batched rows to flush before
    /// returning, guaranteeing that subsequent reads observe the latest state.
    pub fn set_insert_buffering(&self, enabled: bool) -> SqlResult<()> {
        if !enabled {
            let _ = self.flush_buffer_results()?;
        }
        self.insert_buffering_enabled
            .store(enabled, AtomicOrdering::Relaxed);
        Ok(())
    }

    #[cfg(test)]
    fn default_nulls_first_for_tests(&self) -> bool {
        self.default_nulls_first.load(AtomicOrdering::Relaxed)
    }

    fn has_active_transaction(&self) -> bool {
        self.engine.session().has_active_transaction()
    }

    /// Get a reference to the underlying session (for advanced use like error handling in test harnesses).
    pub fn session(&self) -> &SqlSession {
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
    pub fn execute(&self, sql: &str) -> SqlResult<Vec<SqlStatementResult>> {
        tracing::trace!("DEBUG SQL execute: {}", sql);

        // Preprocess SQL
        let processed_sql = Self::preprocess_sql_input(sql);

        let dialect = GenericDialect {};
        let statements = match parse_sql_with_recursion_limit(&dialect, &processed_sql) {
            Ok(stmts) => stmts,
            Err(parse_err) => {
                // SQLite allows omitting BEFORE/AFTER and FOR EACH ROW in CREATE TRIGGER.
                // If parsing fails and the SQL contains CREATE TRIGGER, attempt to expand
                // the shorthand form and retry. This is a workaround until sqlparser's
                // SQLite dialect properly supports the abbreviated syntax.
                if processed_sql.to_uppercase().contains("CREATE TRIGGER") {
                    let expanded = Self::preprocess_sqlite_trigger_shorthand(&processed_sql);
                    parse_sql_with_recursion_limit(&dialect, &expanded).map_err(|_| {
                        Error::InvalidArgumentError(format!("failed to parse SQL: {parse_err}"))
                    })?
                } else {
                    return Err(Error::InvalidArgumentError(format!(
                        "failed to parse SQL: {parse_err}"
                    )));
                }
            }
        };
        let mut results = Vec::with_capacity(statements.len());
        for statement in statements.iter() {
            let statement_expectation = next_statement_expectation();
            match statement {
                Statement::Insert(insert) => {
                    let mut outcome = self.buffer_insert(insert.clone(), statement_expectation)?;
                    if let Some(current) = outcome.current.take() {
                        results.push(current);
                    }
                    results.append(&mut outcome.flushed);
                }
                Statement::StartTransaction { .. }
                | Statement::Commit { .. }
                | Statement::Rollback { .. } => {
                    // Flush before transaction boundaries
                    let mut flushed = self.flush_buffer_results()?;
                    let current = self.execute_statement(statement.clone())?;
                    results.push(current);
                    results.append(&mut flushed);
                }
                _ => {
                    // Flush before any non-INSERT
                    let mut flushed = self.flush_buffer_results()?;
                    let current = self.execute_statement(statement.clone())?;
                    results.push(current);
                    results.append(&mut flushed);
                }
            }
        }

        Ok(results)
    }

    fn preprocess_sql_input(sql: &str) -> String {
        let processed_sql = Self::preprocess_tpch_connect_syntax(sql);
        let processed_sql = Self::preprocess_create_type_syntax(&processed_sql);
        let processed_sql = Self::preprocess_exclude_syntax(&processed_sql);
        let processed_sql = Self::preprocess_trailing_commas_in_values(&processed_sql);
        let processed_sql = Self::preprocess_bare_table_in_clauses(&processed_sql);
        let processed_sql = Self::preprocess_empty_in_lists(&processed_sql);
        let processed_sql = Self::preprocess_index_hints(&processed_sql);
        Self::preprocess_reindex_syntax(&processed_sql)
    }

    /// Flush any buffered literal `INSERT` statements and return their per-statement results.
    ///
    /// Workloads that stream many INSERT statements without interleaving reads can invoke this
    /// to force persistence without waiting for the next non-INSERT statement or the engine
    /// drop hook.
    pub fn flush_pending_inserts(&self) -> SqlResult<Vec<SqlStatementResult>> {
        self.flush_buffer_results()
    }

    /// Prepare a single SQL statement for repeated execution.
    ///
    /// Prepared statements currently support `UPDATE` queries with positional or named
    /// parameters. Callers must provide parameter bindings when executing the returned handle.
    pub fn prepare(&self, sql: &str) -> SqlResult<PreparedStatement> {
        let processed_sql = Self::preprocess_sql_input(sql);

        if let Some(existing) = self
            .statement_cache
            .read()
            .expect("statement cache poisoned")
            .get(&processed_sql)
        {
            return Ok(PreparedStatement::new(Arc::clone(existing)));
        }

        let dialect = GenericDialect {};
        let statements = parse_sql_with_recursion_limit(&dialect, &processed_sql)
            .map_err(|err| Error::InvalidArgumentError(format!("failed to parse SQL: {err}")))?;

        if statements.len() != 1 {
            return Err(Error::InvalidArgumentError(
                "prepared statements must contain exactly one SQL statement".into(),
            ));
        }

        let statement = statements
            .into_iter()
            .next()
            .expect("statement count checked");

        let scope = ParameterScope::new();

        let plan = match statement {
            Statement::Update {
                table,
                assignments,
                from,
                selection,
                returning,
                ..
            } => {
                let update_plan =
                    self.build_update_plan(table, assignments, from, selection, returning)?;
                PlanStatement::Update(update_plan)
            }
            other => {
                return Err(Error::InvalidArgumentError(format!(
                    "prepared statements do not yet support {other:?}"
                )));
            }
        };

        let parameter_meta = scope.finish();
        let prepared = Arc::new(PreparedPlan {
            plan,
            param_count: parameter_meta.max_index(),
        });

        self.statement_cache
            .write()
            .expect("statement cache poisoned")
            .insert(processed_sql, Arc::clone(&prepared));

        Ok(PreparedStatement::new(prepared))
    }

    /// Execute a previously prepared statement with the supplied parameters.
    pub fn execute_prepared(
        &self,
        statement: &PreparedStatement,
        params: &[SqlParamValue],
    ) -> SqlResult<Vec<SqlStatementResult>> {
        if params.len() != statement.parameter_count() {
            return Err(Error::InvalidArgumentError(format!(
                "prepared statement expected {} parameters, received {}",
                statement.parameter_count(),
                params.len()
            )));
        }

        let mut flushed = self.flush_buffer_results()?;
        let mut plan = statement.inner.plan.clone();
        bind_plan_parameters(&mut plan, params, statement.parameter_count())?;
        let current = self.execute_plan_statement(plan)?;
        flushed.insert(0, current);
        Ok(flushed)
    }

    /// Buffer an `INSERT` statement when batching is enabled, or execute it immediately
    /// otherwise.
    ///
    /// The return value includes any flushed results (when a batch boundary is crossed) as well
    /// as the per-statement placeholder that preserves the original `RuntimeStatementResult`
    /// ordering expected by callers like the SLT harness.
    fn buffer_insert(
        &self,
        insert: sqlparser::ast::Insert,
        expectation: StatementExpectation,
    ) -> SqlResult<BufferedInsertResult> {
        // Expectations serve two purposes: (a) ensure we surface synchronous errors when the
        // SLT harness anticipates them, and (b) force a flush when the harness is validating the
        // rows-affected count. In both situations we bypass the buffer entirely so the runtime
        // executes the statement immediately.
        let execute_immediately = matches!(
            expectation,
            StatementExpectation::Error | StatementExpectation::Count(_)
        );
        if execute_immediately {
            let flushed = self.flush_buffer_results()?;
            let current = self.handle_insert(insert)?;
            return Ok(BufferedInsertResult {
                flushed,
                current: Some(current),
            });
        }

        // When buffering is disabled for this engine (the default for unit tests and most
        // production callers), short-circuit to immediate execution so callers see the results
        // they expect without having to register additional expectations.
        if !self.insert_buffering_enabled.load(AtomicOrdering::Relaxed) {
            let flushed = self.flush_buffer_results()?;
            let current = self.handle_insert(insert)?;
            return Ok(BufferedInsertResult {
                flushed,
                current: Some(current),
            });
        }

        let prepared = self.prepare_insert(insert)?;
        match prepared {
            PreparedInsert::Values {
                table_name,
                columns,
                rows,
                on_conflict,
            } => {
                let mut flushed = Vec::new();
                let statement_rows = rows.len();
                let mut buf = self.insert_buffer.borrow_mut();
                match buf.as_mut() {
                    Some(buffer) if buffer.can_accept(&table_name, &columns, on_conflict) => {
                        buffer.push_statement(rows);
                        if buffer.should_flush() {
                            drop(buf);
                            flushed = self.flush_buffer_results()?;
                            return Ok(BufferedInsertResult {
                                flushed,
                                current: None,
                            });
                        }
                        Ok(BufferedInsertResult {
                            flushed,
                            current: Some(RuntimeStatementResult::Insert {
                                table_name,
                                rows_inserted: statement_rows,
                            }),
                        })
                    }
                    Some(_) => {
                        drop(buf);
                        flushed = self.flush_buffer_results()?;
                        let mut buf = self.insert_buffer.borrow_mut();
                        *buf = Some(InsertBuffer::new(
                            table_name.clone(),
                            columns,
                            rows,
                            on_conflict,
                        ));
                        Ok(BufferedInsertResult {
                            flushed,
                            current: Some(RuntimeStatementResult::Insert {
                                table_name,
                                rows_inserted: statement_rows,
                            }),
                        })
                    }
                    None => {
                        *buf = Some(InsertBuffer::new(
                            table_name.clone(),
                            columns,
                            rows,
                            on_conflict,
                        ));
                        Ok(BufferedInsertResult {
                            flushed,
                            current: Some(RuntimeStatementResult::Insert {
                                table_name,
                                rows_inserted: statement_rows,
                            }),
                        })
                    }
                }
            }
            PreparedInsert::Immediate(plan) => {
                let flushed = self.flush_buffer_results()?;
                let executed = self.execute_plan_statement(PlanStatement::Insert(plan))?;
                Ok(BufferedInsertResult {
                    flushed,
                    current: Some(executed),
                })
            }
        }
    }

    /// Flush buffered INSERTs, returning one result per original statement.
    fn flush_buffer_results(&self) -> SqlResult<Vec<SqlStatementResult>> {
        let mut buf = self.insert_buffer.borrow_mut();
        let buffer = match buf.take() {
            Some(b) => b,
            None => return Ok(Vec::new()),
        };
        drop(buf);

        let InsertBuffer {
            table_name,
            columns,
            on_conflict,
            total_rows,
            statement_row_counts,
            rows,
        } = buffer;

        if total_rows == 0 {
            return Ok(Vec::new());
        }

        let plan = InsertPlan {
            table: table_name.clone(),
            columns,
            source: InsertSource::Rows(rows),
            on_conflict,
        };

        let executed = self.execute_plan_statement(PlanStatement::Insert(plan))?;
        let inserted = match executed {
            RuntimeStatementResult::Insert { rows_inserted, .. } => {
                if rows_inserted != total_rows {
                    tracing::warn!(
                        "Buffered INSERT row count mismatch: expected {}, runtime inserted {}",
                        total_rows,
                        rows_inserted
                    );
                }
                rows_inserted
            }
            other => {
                return Err(Error::Internal(format!(
                    "expected Insert result when flushing buffer, got {other:?}"
                )));
            }
        };

        let mut per_statement = Vec::with_capacity(statement_row_counts.len());
        let mut assigned = 0usize;
        for rows in statement_row_counts {
            assigned += rows;
            per_statement.push(RuntimeStatementResult::Insert {
                table_name: table_name.clone(),
                rows_inserted: rows,
            });
        }

        if inserted != assigned {
            tracing::warn!(
                "Buffered INSERT per-statement totals ({}) do not match runtime ({}).",
                assigned,
                inserted
            );
        }

        Ok(per_statement)
    }

    /// Canonicalizes an `INSERT` statement so buffered workloads can share literal payloads while
    /// complex sources still execute eagerly.
    ///
    /// The translation enforces dialect constraints up front, rewrites `VALUES` clauses (and any
    /// constant `SELECT` forms) into `PlanValue` rows, and returns them under
    /// [`PreparedInsert::Values`] so [`Self::buffer_insert`] can append them to the rolling
    /// batch. Statements whose payload must be evaluated at runtime fall back to a fully planned
    /// [`InsertPlan`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgumentError`] whenever the incoming AST uses syntactic forms we
    /// do not currently support or when the literal payload is empty.
    fn prepare_insert(&self, stmt: sqlparser::ast::Insert) -> SqlResult<PreparedInsert> {
        let table_name_debug =
            Self::table_name_from_insert(&stmt).unwrap_or_else(|_| "unknown".to_string());
        tracing::trace!(
            "DEBUG SQL prepare_insert called for table={}",
            table_name_debug
        );

        if !self.engine.session().has_active_transaction()
            && self.is_table_marked_dropped(&table_name_debug)?
        {
            return Err(Error::TransactionContextError(
                DROPPED_TABLE_TRANSACTION_ERR.into(),
            ));
        }

        // Extract conflict resolution action
        use sqlparser::ast::SqliteOnConflict;
        let on_conflict = if stmt.replace_into {
            InsertConflictAction::Replace
        } else if stmt.ignore {
            InsertConflictAction::Ignore
        } else if let Some(or_clause) = stmt.or {
            match or_clause {
                SqliteOnConflict::Replace => InsertConflictAction::Replace,
                SqliteOnConflict::Ignore => InsertConflictAction::Ignore,
                SqliteOnConflict::Abort => InsertConflictAction::Abort,
                SqliteOnConflict::Fail => InsertConflictAction::Fail,
                SqliteOnConflict::Rollback => InsertConflictAction::Rollback,
            }
        } else {
            InsertConflictAction::None
        };

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

        match source_expr.body.as_ref() {
            SetExpr::Values(values) => {
                if values.rows.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "INSERT VALUES list must contain at least one row".into(),
                    ));
                }
                let mut rows: Vec<Vec<PlanValue>> = Vec::with_capacity(values.rows.len());
                for row in &values.rows {
                    let mut converted = Vec::with_capacity(row.len());
                    for expr in row {
                        converted.push(PlanValue::from(SqlValue::try_from_expr(expr)?));
                    }
                    rows.push(converted);
                }
                Ok(PreparedInsert::Values {
                    table_name: display_name,
                    columns,
                    rows,
                    on_conflict,
                })
            }
            SetExpr::Select(select) => {
                if let Some(rows) = extract_constant_select_rows(select.as_ref())? {
                    return Ok(PreparedInsert::Values {
                        table_name: display_name,
                        columns,
                        rows,
                        on_conflict,
                    });
                }
                if let Some(range_rows) = extract_rows_from_range(select.as_ref())? {
                    return Ok(PreparedInsert::Values {
                        table_name: display_name,
                        columns,
                        rows: range_rows.into_rows(),
                        on_conflict,
                    });
                }

                let select_plan = self.build_select_plan((**source_expr).clone())?;
                Ok(PreparedInsert::Immediate(InsertPlan {
                    table: display_name,
                    columns,
                    source: InsertSource::Select {
                        plan: Box::new(select_plan),
                    },
                    on_conflict,
                }))
            }
            _ => Err(Error::InvalidArgumentError(
                "unsupported INSERT source".into(),
            )),
        }
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
        if results.is_empty() {
            return Err(Error::InvalidArgumentError(
                "SqlEngine::sql expects a SELECT statement".into(),
            ));
        }

        let primary = results.remove(0);

        match primary {
            RuntimeStatementResult::Select { execution, .. } => execution.collect(),
            other => Err(Error::InvalidArgumentError(format!(
                "SqlEngine::sql requires a SELECT statement, got {other:?}",
            ))),
        }
    }

    fn execute_statement(&self, statement: Statement) -> SqlResult<SqlStatementResult> {
        let statement_sql = statement.to_string();
        let _query_label_guard = push_query_label(statement_sql.clone());
        tracing::debug!("SQL execute_statement: {}", statement_sql.trim());
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
    ) -> SqlResult<SqlStatementResult> {
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
            Statement::CreateTrigger(create_trigger) => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: CreateTrigger");
                self.handle_create_trigger(create_trigger)
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
            Statement::DropTrigger(drop_trigger) => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: DropTrigger");
                self.handle_drop_trigger(drop_trigger)
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
            Statement::Vacuum(vacuum) => {
                tracing::trace!("DEBUG SQL execute_statement_non_transactional: Vacuum");
                self.handle_vacuum(vacuum)
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
    ) -> SqlResult<FxHashSet<String>> {
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
            Ok(specs) if !specs.is_empty() => Ok(specs
                .into_iter()
                .map(|spec| spec.name.to_ascii_lowercase())
                .collect()),
            Ok(_) => {
                if let Some(table_id) = context.catalog().table_id(&canonical_name)
                    && let Some(resolver) = context.catalog().field_resolver(table_id)
                {
                    let fallback: FxHashSet<String> = resolver
                        .field_names()
                        .into_iter()
                        .map(|name| name.to_ascii_lowercase())
                        .collect();
                    tracing::debug!(
                        "collect_known_columns: using resolver fallback for '{}': {:?}",
                        display_name,
                        fallback
                    );
                    return Ok(fallback);
                }
                Ok(FxHashSet::default())
            }
            Err(err) => {
                if Self::is_table_missing_error(&err) {
                    Ok(FxHashSet::default())
                } else {
                    Err(Self::map_table_error(display_name, err))
                }
            }
        }
    }

    fn parse_view_query(sql: &str) -> SqlResult<Query> {
        use sqlparser::ast::Statement;

        let dialect = GenericDialect {};
        let mut statements = Parser::parse_sql(&dialect, sql).map_err(|err| {
            Error::InvalidArgumentError(format!("failed to parse view definition: {}", err))
        })?;

        if statements.len() != 1 {
            return Err(Error::InvalidArgumentError(
                "view definition must contain a single SELECT statement".into(),
            ));
        }

        match statements.pop().unwrap() {
            Statement::Query(query) => Ok(*query),
            _ => Err(Error::InvalidArgumentError(
                "view definition must be expressed as a SELECT query".into(),
            )),
        }
    }

    fn is_table_marked_dropped(&self, table_name: &str) -> SqlResult<bool> {
        let canonical = table_name.to_ascii_lowercase();
        Ok(self.engine.context().is_table_marked_dropped(&canonical))
    }

    fn handle_create_table(
        &self,
        mut stmt: sqlparser::ast::CreateTable,
    ) -> SqlResult<SqlStatementResult> {
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
        let column_names_lower: FxHashSet<String> = column_names
            .iter()
            .map(|name| name.to_ascii_lowercase())
            .collect();

        let mut columns: Vec<PlanColumnSpec> = Vec::with_capacity(column_defs_ast.len());
        let mut primary_key_columns: FxHashSet<String> = FxHashSet::default();
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
            let mut column_lookup: FxHashMap<String, usize> =
                FxHashMap::with_capacity_and_hasher(columns.len(), Default::default());
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
        let mut seen_column_names: FxHashSet<String> = FxHashSet::default();
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
    ) -> SqlResult<PlanForeignKeyAction> {
        match action {
            None | Some(ReferentialAction::NoAction) => Ok(PlanForeignKeyAction::NoAction),
            Some(ReferentialAction::Restrict) => Ok(PlanForeignKeyAction::Restrict),
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
        known_columns_lower: &FxHashSet<String>,
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
        columns: Vec<sqlparser::ast::ViewColumnDef>,
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

        // Capture and normalize optional column list definitions.
        let column_names: Vec<String> = columns
            .into_iter()
            .map(|column| column.name.value)
            .collect();

        if !column_names.is_empty() {
            ensure_unique_case_insensitive(column_names.iter().map(|name| name.as_str()), |dup| {
                format!("duplicate column name '{}' in CREATE VIEW column list", dup)
            })?;
        }

        let mut query_ast = *query;

        if !column_names.is_empty() {
            let select = match query_ast.body.as_mut() {
                sqlparser::ast::SetExpr::Select(select) => select,
                _ => {
                    return Err(Error::InvalidArgumentError(
                        "CREATE VIEW column list requires a SELECT query".into(),
                    ));
                }
            };

            for item in &select.projection {
                if matches!(
                    item,
                    SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(_, _)
                ) {
                    return Err(Error::InvalidArgumentError(
                        "CREATE VIEW column lists with wildcard projections are not supported yet"
                            .into(),
                    ));
                }
            }

            if select.projection.len() != column_names.len() {
                return Err(Error::InvalidArgumentError(format!(
                    "CREATE VIEW column list specifies {} column(s) but SELECT projection yields {}",
                    column_names.len(),
                    select.projection.len()
                )));
            }

            for (item, column_name) in select.projection.iter_mut().zip(column_names.iter()) {
                match item {
                    SelectItem::ExprWithAlias { alias, .. } => {
                        alias.value = column_name.clone();
                    }
                    SelectItem::UnnamedExpr(expr) => {
                        *item = SelectItem::ExprWithAlias {
                            expr: expr.clone(),
                            alias: Ident::new(column_name.clone()),
                        };
                    }
                    _ => {
                        return Err(Error::InvalidArgumentError(
                            "CREATE VIEW column list requires simple SELECT projections".into(),
                        ));
                    }
                }
            }
        }

        // Validate the view SELECT plan after applying optional column aliases and capture it.
        let select_plan = self.build_select_plan(query_ast.clone())?;

        // Convert query to SQL string for storage (after applying column aliases when present)
        let view_definition = query_ast.to_string();

        // Build CreateViewPlan with namespace routing (same pattern as CREATE TABLE)
        let namespace = if temporary {
            Some(TEMPORARY_NAMESPACE_ID.to_string())
        } else {
            None
        };

        let plan = CreateViewPlan {
            name: display_name.clone(),
            if_not_exists,
            view_definition,
            select_plan: Box::new(select_plan),
            namespace,
        };

        self.execute_plan_statement(PlanStatement::CreateView(plan))?;

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

    fn handle_create_trigger(
        &self,
        create_trigger: CreateTrigger,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        let CreateTrigger {
            or_alter,
            or_replace,
            is_constraint,
            name,
            period,
            period_before_table: _,
            events,
            table_name,
            referenced_table_name,
            referencing,
            trigger_object,
            include_each: _,
            condition,
            exec_body,
            statements_as,
            statements,
            characteristics,
        } = create_trigger;

        if or_alter {
            return Err(Error::InvalidArgumentError(
                "CREATE OR ALTER TRIGGER is not supported".into(),
            ));
        }

        if or_replace {
            return Err(Error::InvalidArgumentError(
                "CREATE OR REPLACE TRIGGER is not supported".into(),
            ));
        }

        if is_constraint {
            return Err(Error::InvalidArgumentError(
                "CREATE TRIGGER ... CONSTRAINT is not supported".into(),
            ));
        }

        if referenced_table_name.is_some() {
            return Err(Error::InvalidArgumentError(
                "CREATE TRIGGER referencing another table is not supported".into(),
            ));
        }

        if !referencing.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TRIGGER REFERENCING clauses are not supported".into(),
            ));
        }

        if characteristics.is_some() {
            return Err(Error::InvalidArgumentError(
                "CREATE TRIGGER constraint characteristics are not supported".into(),
            ));
        }

        if events.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TRIGGER requires at least one event".into(),
            ));
        }

        if events.len() != 1 {
            return Err(Error::InvalidArgumentError(
                "CREATE TRIGGER currently supports exactly one trigger event".into(),
            ));
        }

        let timing = match period {
            TriggerPeriod::Before => TriggerTimingMeta::Before,
            TriggerPeriod::After | TriggerPeriod::For => TriggerTimingMeta::After,
            TriggerPeriod::InsteadOf => TriggerTimingMeta::InsteadOf,
        };

        let event_meta = match events.into_iter().next().expect("checked length") {
            TriggerEvent::Insert => TriggerEventMeta::Insert,
            TriggerEvent::Delete => TriggerEventMeta::Delete,
            TriggerEvent::Update(columns) => TriggerEventMeta::Update {
                columns: columns
                    .into_iter()
                    .map(|ident| ident.value.to_ascii_lowercase())
                    .collect(),
            },
            TriggerEvent::Truncate => {
                return Err(Error::InvalidArgumentError(
                    "CREATE TRIGGER for TRUNCATE events is not supported".into(),
                ));
            }
        };

        let (trigger_display_name, canonical_trigger_name) = canonical_object_name(&name)?;
        let (table_display_name, canonical_table_name) = canonical_object_name(&table_name)?;

        let condition_sql = condition.map(|expr| expr.to_string());

        let body_sql = if let Some(exec_body) = exec_body {
            format!("EXECUTE {exec_body}")
        } else if let Some(statements) = statements {
            let rendered = statements.to_string();
            if statements_as {
                format!("AS {rendered}")
            } else {
                rendered
            }
        } else {
            return Err(Error::InvalidArgumentError(
                "CREATE TRIGGER requires a trigger body".into(),
            ));
        };

        let for_each_row = matches!(trigger_object, TriggerObject::Row);

        self.engine.context().create_trigger(
            &trigger_display_name,
            &canonical_trigger_name,
            &table_display_name,
            &canonical_table_name,
            timing,
            event_meta,
            for_each_row,
            condition_sql,
            body_sql,
            false,
        )?;

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

    fn handle_drop_trigger(
        &self,
        drop_trigger: DropTrigger,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        let DropTrigger {
            if_exists,
            trigger_name,
            table_name,
            option,
        } = drop_trigger;

        if option.is_some() {
            return Err(Error::InvalidArgumentError(
                "DROP TRIGGER CASCADE/RESTRICT options are not supported".into(),
            ));
        }

        let (trigger_display_name, canonical_trigger_name) = canonical_object_name(&trigger_name)?;

        let (table_display, table_canonical) = if let Some(table_name) = table_name {
            let (display, canonical) = canonical_object_name(&table_name)?;
            (Some(display), Some(canonical))
        } else {
            (None, None)
        };

        let table_display_hint = table_display.as_deref();
        let table_canonical_hint = table_canonical.as_deref();

        self.engine.context().drop_trigger(
            &trigger_display_name,
            &canonical_trigger_name,
            table_display_hint,
            table_canonical_hint,
            if_exists,
        )?;

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
                on_conflict: InsertConflictAction::None,
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

        let alias = match &table_with_joins.relation {
            TableFactor::Table { alias, .. } => alias.as_ref().map(|a| a.name.value.clone()),
            _ => None,
        };

        let mut view = InlineView::new(vec![
            InlineColumn::int32("cid", false),
            InlineColumn::utf8("name", false),
            InlineColumn::utf8("type", false),
            InlineColumn::bool("notnull", false),
            InlineColumn::utf8("dflt_value", true),
            InlineColumn::bool("pk", false),
        ]);

        for (idx, col) in columns.iter().enumerate() {
            view.add_row(vec![
                InlineValue::Int32(Some(idx as i32)),
                InlineValue::String(Some(col.name.clone())),
                InlineValue::String(Some(format!("{:?}", col.data_type))),
                InlineValue::Bool(Some(!col.nullable)),
                InlineValue::String(None),
                InlineValue::Bool(Some(col.primary_key)),
            ])?;
        }

        self.execute_inline_view(select, query, table_name, alias.as_deref(), view)
    }

    fn execute_inline_view(
        &self,
        select: &Select,
        query: &Query,
        table_label: String,
        table_alias: Option<&str>,
        mut view: InlineView,
    ) -> SqlResult<Option<RuntimeStatementResult<P>>> {
        ensure_inline_select_supported(select, query)?;

        if let Some(selection) = &select.selection {
            view.apply_selection(selection, table_alias)?;
        }

        let (schema, batch) = view.into_record_batch()?;
        self.finalize_inline_select(select, query, table_label, schema, batch)
    }

    fn finalize_inline_select(
        &self,
        select: &Select,
        query: &Query,
        table_name: String,
        schema: Arc<Schema>,
        batch: RecordBatch,
    ) -> SqlResult<Option<RuntimeStatementResult<P>>> {
        use arrow::array::ArrayRef;

        let mut working_schema = schema;
        let mut working_batch = batch;

        let projection_indices: Vec<usize> = select
            .projection
            .iter()
            .filter_map(|item| match item {
                SelectItem::UnnamedExpr(SqlExpr::Identifier(ident)) => {
                    working_schema.index_of(&ident.value).ok()
                }
                SelectItem::ExprWithAlias { expr, .. } => {
                    if let SqlExpr::Identifier(ident) = expr {
                        working_schema.index_of(&ident.value).ok()
                    } else {
                        None
                    }
                }
                SelectItem::Wildcard(_) => None,
                _ => None,
            })
            .collect();

        if !projection_indices.is_empty() {
            let projected_fields: Vec<Field> = projection_indices
                .iter()
                .map(|&idx| working_schema.field(idx).clone())
                .collect();
            let projected_schema = Arc::new(Schema::new(projected_fields));
            let projected_columns: Vec<ArrayRef> = projection_indices
                .iter()
                .map(|&idx| Arc::clone(working_batch.column(idx)))
                .collect();
            working_batch = RecordBatch::try_new(Arc::clone(&projected_schema), projected_columns)
                .map_err(|e| Error::Internal(format!("failed to project columns: {}", e)))?;
            working_schema = projected_schema;
        }

        if let Some(order_by) = &query.order_by {
            use arrow::compute::{SortColumn, lexsort_to_indices};
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
                    && let Ok(col_idx) = working_schema.index_of(&ident.value)
                {
                    let options = arrow::compute::SortOptions {
                        descending: !order_expr.options.asc.unwrap_or(true),
                        nulls_first: order_expr.options.nulls_first.unwrap_or(false),
                    };
                    sort_columns.push(SortColumn {
                        values: Arc::clone(working_batch.column(col_idx)),
                        options: Some(options),
                    });
                }
            }

            if !sort_columns.is_empty() {
                let indices = lexsort_to_indices(&sort_columns, None)
                    .map_err(|e| Error::Internal(format!("failed to sort: {}", e)))?;
                let sorted_columns: Result<Vec<ArrayRef>, _> = working_batch
                    .columns()
                    .iter()
                    .map(|col| take(col.as_ref(), &indices, None))
                    .collect();
                working_batch = RecordBatch::try_new(
                    Arc::clone(&working_schema),
                    sorted_columns
                        .map_err(|e| Error::Internal(format!("failed to apply sort: {}", e)))?,
                )
                .map_err(|e| Error::Internal(format!("failed to create sorted batch: {}", e)))?;
            }
        }

        if let Some(limit) = extract_limit_count(query)?
            && limit < working_batch.num_rows()
        {
            let indices: Vec<u32> = (0..limit as u32).collect();
            let take_indices = UInt32Array::from(indices);
            let limited_columns: Result<Vec<ArrayRef>, _> = working_batch
                .columns()
                .iter()
                .map(|col| take(col.as_ref(), &take_indices, None))
                .collect();
            working_batch = RecordBatch::try_new(
                Arc::clone(&working_schema),
                limited_columns
                    .map_err(|e| Error::Internal(format!("failed to apply limit: {}", e)))?,
            )
            .map_err(|e| Error::Internal(format!("failed to create limited batch: {}", e)))?;
        }

        let execution = SelectExecution::new_single_batch(
            table_name.clone(),
            Arc::clone(&working_schema),
            working_batch,
        );

        Ok(Some(RuntimeStatementResult::Select {
            table_name,
            schema: working_schema,
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
        use arrow::array::{ArrayRef, Date32Builder, Float64Builder, Int64Builder, StringBuilder};
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
                PlanValue::Date32(_) => (DataType::Date32, false),
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
                DataType::Date32 => {
                    let mut builder = Date32Builder::with_capacity(rows.len());
                    for row in &rows {
                        match &row[col_idx] {
                            PlanValue::Date32(v) => builder.append_value(*v),
                            PlanValue::Null => builder.append_null(),
                            other => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "type mismatch in VALUES: expected DATE, got {:?}",
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
        match self.prepare_insert(stmt)? {
            PreparedInsert::Values {
                table_name,
                columns,
                rows,
                on_conflict,
            } => {
                tracing::trace!(
                    "DEBUG SQL handle_insert executing buffered-values insert for table={}",
                    table_name
                );
                let plan = InsertPlan {
                    table: table_name,
                    columns,
                    source: InsertSource::Rows(rows),
                    on_conflict,
                };
                self.execute_plan_statement(PlanStatement::Insert(plan))
            }
            PreparedInsert::Immediate(plan) => {
                let table_name = plan.table.clone();
                tracing::trace!(
                    "DEBUG SQL handle_insert executing immediate insert for table={}",
                    table_name
                );
                self.execute_plan_statement(PlanStatement::Insert(plan))
            }
        }
    }

    fn build_update_plan(
        &self,
        table: TableWithJoins,
        assignments: Vec<Assignment>,
        from: Option<UpdateTableFromKind>,
        selection: Option<SqlExpr>,
        returning: Option<Vec<SelectItem>>,
    ) -> SqlResult<UpdatePlan> {
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

        // Use a HashMap to track column assignments. If a column appears multiple times,
        // the last assignment wins (SQLite-compatible behavior).
        let mut assignments_map: FxHashMap<String, (String, sqlparser::ast::Expr)> =
            FxHashMap::with_capacity_and_hasher(assignments.len(), FxBuildHasher);
        for assignment in assignments {
            let column_name = resolve_assignment_column_name(&assignment.target)?;
            let normalized = column_name.to_ascii_lowercase();
            // Store in map - last assignment wins
            assignments_map.insert(normalized, (column_name, assignment.value.clone()));
        }

        let mut column_assignments = Vec::with_capacity(assignments_map.len());
        for (_normalized, (column_name, expr)) in assignments_map {
            let value = match SqlValue::try_from_expr(&expr) {
                Ok(literal) => AssignmentValue::Literal(PlanValue::from(literal)),
                Err(Error::InvalidArgumentError(msg))
                    if msg.contains("unsupported literal expression") =>
                {
                    let normalized_expr = self.materialize_in_subquery(expr.clone())?;
                    let translated = translate_scalar_with_context(
                        &resolver,
                        IdentifierContext::new(table_id),
                        &normalized_expr,
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
                let mut subqueries = Vec::new();
                let predicate = translate_condition_with_context(
                    self,
                    &resolver,
                    IdentifierContext::new(table_id),
                    &materialized_expr,
                    &[],
                    &mut subqueries,
                    None,
                )?;
                if subqueries.is_empty() {
                    Some(predicate)
                } else {
                    return Err(Error::InvalidArgumentError(
                        "EXISTS subqueries are not supported in UPDATE WHERE clauses".into(),
                    ));
                }
            }
            None => None,
        };

        let plan = UpdatePlan {
            table: display_name.clone(),
            assignments: column_assignments,
            filter,
        };
        Ok(plan)
    }

    fn handle_update(
        &self,
        table: TableWithJoins,
        assignments: Vec<Assignment>,
        from: Option<UpdateTableFromKind>,
        selection: Option<SqlExpr>,
        returning: Option<Vec<SelectItem>>,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        let plan = self.build_update_plan(table, assignments, from, selection, returning)?;
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

        let filter = if let Some(expr) = selection {
            let materialized_expr = self.materialize_in_subquery(expr)?;
            let mut subqueries = Vec::new();
            let predicate = translate_condition_with_context(
                self,
                &resolver,
                IdentifierContext::new(table_id),
                &materialized_expr,
                &[],
                &mut subqueries,
                None,
            )?;
            if !subqueries.is_empty() {
                return Err(Error::InvalidArgumentError(
                    "EXISTS subqueries are not supported in DELETE WHERE clauses".into(),
                ));
            }
            Some(predicate)
        } else {
            None
        };

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
            ObjectType::View => {
                if cascade || restrict {
                    return Err(Error::InvalidArgumentError(
                        "DROP VIEW CASCADE/RESTRICT is not supported".into(),
                    ));
                }

                for name in names {
                    let view_name = Self::object_name_to_string(&name)?;
                    let plan = llkv_plan::DropViewPlan::new(view_name).if_exists(if_exists);
                    self.execute_plan_statement(llkv_plan::PlanStatement::DropView(plan))?;
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
    fn try_materialize_avg_subquery(&self, query: &Query) -> SqlResult<Option<SqlExpr>> {
        use sqlparser::ast::{
            DuplicateTreatment, FunctionArg, FunctionArgExpr, FunctionArguments, ObjectName,
            ObjectNamePart, SelectItem, SetExpr,
        };

        let select = match query.body.as_ref() {
            SetExpr::Select(select) => select.as_ref(),
            _ => return Ok(None),
        };

        if select.projection.len() != 1
            || select.distinct.is_some()
            || select.top.is_some()
            || select.value_table_mode.is_some()
            || select.having.is_some()
            || !group_by_is_empty(&select.group_by)
            || select.into.is_some()
            || !select.lateral_views.is_empty()
        {
            return Ok(None);
        }

        let func = match &select.projection[0] {
            SelectItem::UnnamedExpr(SqlExpr::Function(func)) => func,
            _ => return Ok(None),
        };

        if func.uses_odbc_syntax
            || func.filter.is_some()
            || func.null_treatment.is_some()
            || func.over.is_some()
            || !func.within_group.is_empty()
        {
            return Ok(None);
        }

        let func_name = func.name.to_string().to_ascii_lowercase();
        if func_name != "avg" {
            return Ok(None);
        }

        let args = match &func.args {
            FunctionArguments::List(list) => {
                if matches!(list.duplicate_treatment, Some(DuplicateTreatment::Distinct))
                    || !list.clauses.is_empty()
                {
                    return Ok(None);
                }
                &list.args
            }
            _ => return Ok(None),
        };

        if args.len() != 1 {
            return Ok(None);
        }

        match &args[0] {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(_)) => {}
            _ => return Ok(None),
        };

        let mut sum_query = query.clone();
        let mut count_query = query.clone();

        let build_replacement = |target_query: &mut Query, name: &str| -> SqlResult<()> {
            let select = match target_query.body.as_mut() {
                SetExpr::Select(select) => select,
                _ => {
                    return Err(Error::Internal(
                        "expected SELECT query in AVG materialization".into(),
                    ));
                }
            };

            let mut replacement_func = func.clone();
            replacement_func.name = ObjectName(vec![ObjectNamePart::Identifier(Ident {
                value: name.to_string(),
                quote_style: None,
                span: Span::empty(),
            })]);
            select.projection = vec![SelectItem::UnnamedExpr(SqlExpr::Function(replacement_func))];
            Ok(())
        };

        build_replacement(&mut sum_query, "sum")?;
        build_replacement(&mut count_query, "count")?;

        let sum_value = self.execute_scalar_int64(sum_query)?;
        let count_value = self.execute_scalar_int64(count_query)?;

        let Some(count_value) = count_value else {
            return Ok(Some(SqlExpr::Value(ValueWithSpan {
                value: Value::Null,
                span: Span::empty(),
            })));
        };

        if count_value == 0 {
            return Ok(Some(SqlExpr::Value(ValueWithSpan {
                value: Value::Null,
                span: Span::empty(),
            })));
        }

        let sum_value = match sum_value {
            Some(value) => value,
            None => {
                return Ok(Some(SqlExpr::Value(ValueWithSpan {
                    value: Value::Null,
                    span: Span::empty(),
                })));
            }
        };

        let avg = (sum_value as f64) / (count_value as f64);
        let value = ValueWithSpan {
            value: Value::Number(avg.to_string(), false),
            span: Span::empty(),
        };
        Ok(Some(SqlExpr::Value(value)))
    }

    fn execute_scalar_int64(&self, query: Query) -> SqlResult<Option<i64>> {
        let result = self.handle_query(query)?;
        let execution = match result {
            RuntimeStatementResult::Select { execution, .. } => execution,
            _ => {
                return Err(Error::InvalidArgumentError(
                    "scalar aggregate must be a SELECT statement".into(),
                ));
            }
        };

        let batches = execution.collect()?;
        let mut captured: Option<Option<i64>> = None;

        for batch in batches {
            if batch.num_columns() == 0 {
                continue;
            }
            if batch.num_columns() != 1 {
                return Err(Error::InvalidArgumentError(
                    "scalar aggregate must return exactly one column".into(),
                ));
            }

            let array = batch.column(0);
            let values = array
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "scalar aggregate result must be an INT64 value".into(),
                    )
                })?;

            for idx in 0..values.len() {
                if captured.is_some() {
                    return Err(Error::InvalidArgumentError(
                        "scalar aggregate returned more than one row".into(),
                    ));
                }
                if values.is_null(idx) {
                    captured = Some(None);
                } else {
                    captured = Some(Some(values.value(idx)));
                }
            }
        }

        Ok(captured.unwrap_or(None))
    }

    fn materialize_scalar_subquery(&self, subquery: Query) -> SqlResult<SqlExpr> {
        if let Some(avg_literal) = self.try_materialize_avg_subquery(&subquery)? {
            return Ok(avg_literal);
        }

        let result = self.handle_query(subquery)?;
        let execution = match result {
            RuntimeStatementResult::Select { execution, .. } => execution,
            _ => {
                return Err(Error::InvalidArgumentError(
                    "scalar subquery must be a SELECT statement".into(),
                ));
            }
        };

        let batches = execution.collect()?;
        let mut captured_value: Option<ValueWithSpan> = None;

        for batch in batches {
            if batch.num_columns() == 0 {
                continue;
            }
            if batch.num_columns() != 1 {
                return Err(Error::InvalidArgumentError(
                    "scalar subquery must return exactly one column".into(),
                ));
            }

            let column = batch.column(0);
            for row_idx in 0..batch.num_rows() {
                if captured_value.is_some() {
                    return Err(Error::InvalidArgumentError(
                        "scalar subquery returned more than one row".into(),
                    ));
                }

                let value = if column.is_null(row_idx) {
                    Value::Null
                } else {
                    use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray};
                    match column.data_type() {
                        arrow::datatypes::DataType::Int64 => {
                            let array =
                                column
                                    .as_any()
                                    .downcast_ref::<Int64Array>()
                                    .ok_or_else(|| {
                                        Error::Internal(
                                            "expected Int64 array for scalar subquery".into(),
                                        )
                                    })?;
                            Value::Number(array.value(row_idx).to_string(), false)
                        }
                        arrow::datatypes::DataType::Float64 => {
                            let array = column.as_any().downcast_ref::<Float64Array>().ok_or_else(
                                || {
                                    Error::Internal(
                                        "expected Float64 array for scalar subquery".into(),
                                    )
                                },
                            )?;
                            Value::Number(array.value(row_idx).to_string(), false)
                        }
                        arrow::datatypes::DataType::Utf8 => {
                            let array =
                                column
                                    .as_any()
                                    .downcast_ref::<StringArray>()
                                    .ok_or_else(|| {
                                        Error::Internal(
                                            "expected String array for scalar subquery".into(),
                                        )
                                    })?;
                            Value::SingleQuotedString(array.value(row_idx).to_string())
                        }
                        arrow::datatypes::DataType::Boolean => {
                            let array = column.as_any().downcast_ref::<BooleanArray>().ok_or_else(
                                || {
                                    Error::Internal(
                                        "expected Boolean array for scalar subquery".into(),
                                    )
                                },
                            )?;
                            Value::Boolean(array.value(row_idx))
                        }
                        other => {
                            return Err(Error::InvalidArgumentError(format!(
                                "unsupported data type in scalar subquery result: {other:?}"
                            )));
                        }
                    }
                };

                captured_value = Some(ValueWithSpan {
                    value,
                    span: Span::empty(),
                });
            }
        }

        let final_value = captured_value.unwrap_or(ValueWithSpan {
            value: Value::Null,
            span: Span::empty(),
        });
        Ok(SqlExpr::Value(final_value))
    }

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
                                        if batch.num_columns() > 1 {
                                            return Err(Error::InvalidArgumentError(format!(
                                                "IN subquery must return exactly one column, got {}",
                                                batch.num_columns()
                                            )));
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
                        SqlExpr::Subquery(subquery) => {
                            let scalar_expr = self.materialize_scalar_subquery(*subquery)?;
                            result_stack.push(scalar_expr);
                        }
                        SqlExpr::Case {
                            case_token,
                            end_token,
                            operand,
                            conditions,
                            else_result,
                        } => {
                            let new_operand = match operand {
                                Some(expr) => Some(Box::new(self.materialize_in_subquery(*expr)?)),
                                None => None,
                            };
                            let mut new_conditions = Vec::with_capacity(conditions.len());
                            for branch in conditions {
                                let condition = self.materialize_in_subquery(branch.condition)?;
                                let result = self.materialize_in_subquery(branch.result)?;
                                new_conditions.push(sqlparser::ast::CaseWhen { condition, result });
                            }
                            let new_else = match else_result {
                                Some(expr) => Some(Box::new(self.materialize_in_subquery(*expr)?)),
                                None => None,
                            };
                            result_stack.push(SqlExpr::Case {
                                case_token,
                                end_token,
                                operand: new_operand,
                                conditions: new_conditions,
                                else_result: new_else,
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
        let mut visited_views = FxHashSet::default();
        self.execute_query_with_view_support(query, &mut visited_views)
    }

    fn execute_query_with_view_support(
        &self,
        query: Query,
        visited_views: &mut FxHashSet<String>,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        self.ensure_information_schema_ready()?;

        if let Some(result) = self.try_execute_simple_view_select(&query, visited_views)? {
            return Ok(result);
        }

        if let Some(result) = self.try_execute_view_set_operation(&query, visited_views)? {
            return Ok(result);
        }

        if let Some(result) = self.try_execute_simple_derived_select(&query, visited_views)? {
            return Ok(result);
        }

        // Check for pragma_table_info() table function first
        if let Some(result) = self.try_handle_pragma_table_info(&query)? {
            return Ok(result);
        }

        let select_plan = self.build_select_plan(query)?;
        self.execute_plan_statement(PlanStatement::Select(Box::new(select_plan)))
    }

    fn try_execute_simple_view_select(
        &self,
        query: &Query,
        visited_views: &mut FxHashSet<String>,
    ) -> SqlResult<Option<RuntimeStatementResult<P>>> {
        use sqlparser::ast::SetExpr;

        // Reject complex query forms upfront.
        if query.with.is_some() || query.order_by.is_some() || query.limit_clause.is_some() {
            return Ok(None);
        }

        let select = match query.body.as_ref() {
            SetExpr::Select(select) => select,
            _ => return Ok(None),
        };

        if select.distinct.is_some()
            || select.selection.is_some()
            || !group_by_is_empty(&select.group_by)
            || select.having.is_some()
            || !select.cluster_by.is_empty()
            || !select.distribute_by.is_empty()
            || !select.sort_by.is_empty()
            || select.top.is_some()
            || select.value_table_mode.is_some()
            || !select.named_window.is_empty()
            || select.qualify.is_some()
            || select.connect_by.is_some()
        {
            return Ok(None);
        }

        if select.from.len() != 1 {
            return Ok(None);
        }

        let table_with_joins = &select.from[0];
        if table_with_joins_has_join(table_with_joins) {
            return Ok(None);
        }

        let (view_display_name, view_canonical_name, table_alias) = match &table_with_joins.relation
        {
            TableFactor::Table { name, alias, .. } => {
                let (display, canonical) = canonical_object_name(name)?;
                let catalog = self.engine.context().table_catalog();
                let Some(table_id) = catalog.table_id(&canonical) else {
                    return Ok(None);
                };
                if !self.engine.context().is_view(table_id)? {
                    return Ok(None);
                }
                let alias_name = alias.as_ref().map(|a| a.name.value.clone());
                (display, canonical, alias_name)
            }
            _ => return Ok(None),
        };

        // Gather projection mapping for simple column selection.
        enum ProjectionKind {
            All,
            Columns(Vec<(String, String)>), // (source column, output name)
        }

        let projection = if select
            .projection
            .iter()
            .any(|item| matches!(item, SelectItem::Wildcard(_)))
        {
            if select.projection.len() != 1 {
                return Ok(None);
            }
            ProjectionKind::All
        } else {
            let mut columns = Vec::with_capacity(select.projection.len());
            for item in &select.projection {
                match item {
                    SelectItem::UnnamedExpr(SqlExpr::Identifier(ident)) => {
                        let name = ident.value.clone();
                        columns.push((name.clone(), name));
                    }
                    SelectItem::ExprWithAlias { expr, alias } => {
                        let source = match expr {
                            SqlExpr::Identifier(ident) => ident.value.clone(),
                            SqlExpr::CompoundIdentifier(idents) => {
                                if idents.is_empty() {
                                    return Ok(None);
                                }
                                idents.last().unwrap().value.clone()
                            }
                            _ => return Ok(None),
                        };
                        columns.push((source, alias.value.clone()));
                    }
                    SelectItem::UnnamedExpr(SqlExpr::CompoundIdentifier(parts)) => {
                        if parts.is_empty() {
                            return Ok(None);
                        }
                        // Allow optional table qualifier matching alias or view name.
                        if parts.len() == 2 {
                            let qualifier = parts[0].value.to_ascii_lowercase();
                            if let Some(ref alias_name) = table_alias {
                                if qualifier != alias_name.to_ascii_lowercase() {
                                    return Ok(None);
                                }
                            } else if qualifier != view_display_name.to_ascii_lowercase() {
                                return Ok(None);
                            }
                        } else if parts.len() != 1 {
                            return Ok(None);
                        }
                        let column_name = parts.last().unwrap().value.clone();
                        columns.push((column_name.clone(), column_name));
                    }
                    _ => return Ok(None),
                }
            }
            ProjectionKind::Columns(columns)
        };

        let context = self.engine.context();
        let definition = context
            .view_definition(&view_canonical_name)?
            .ok_or_else(|| {
                Error::CatalogError(format!(
                    "Binder Error: view '{}' does not have a stored definition",
                    view_display_name
                ))
            })?;

        if !visited_views.insert(view_canonical_name.clone()) {
            return Err(Error::CatalogError(format!(
                "Binder Error: cyclic view reference involving '{}'",
                view_display_name
            )));
        }

        let view_query = Self::parse_view_query(&definition)?;
        let view_result = self.execute_query_with_view_support(view_query, visited_views);
        visited_views.remove(&view_canonical_name);
        let view_result = view_result?;

        let RuntimeStatementResult::Select {
            execution: view_execution,
            schema: view_schema,
            ..
        } = view_result
        else {
            return Err(Error::InvalidArgumentError(format!(
                "view '{}' definition did not produce a SELECT result",
                view_display_name
            )));
        };

        match projection {
            ProjectionKind::All => {
                // Reuse the original execution and schema.
                let select_result = RuntimeStatementResult::Select {
                    execution: view_execution,
                    table_name: view_display_name,
                    schema: view_schema,
                };
                Ok(Some(select_result))
            }
            ProjectionKind::Columns(columns) => {
                let exec = *view_execution;
                let batches = exec.collect()?;
                let view_fields = view_schema.fields();
                let mut name_to_index =
                    FxHashMap::with_capacity_and_hasher(view_fields.len(), Default::default());
                for (idx, field) in view_fields.iter().enumerate() {
                    name_to_index.insert(field.name().to_ascii_lowercase(), idx);
                }

                let mut column_indices = Vec::with_capacity(columns.len());
                let mut projected_fields = Vec::with_capacity(columns.len());

                for (source, output) in columns {
                    let lookup = source.to_ascii_lowercase();
                    let Some(&idx) = name_to_index.get(&lookup) else {
                        return Err(Error::InvalidArgumentError(format!(
                            "Binder Error: view '{}' does not have a column named '{}'",
                            view_display_name, source
                        )));
                    };
                    column_indices.push(idx);
                    let origin_field = view_fields[idx].clone();
                    let projected_field = Field::new(
                        &output,
                        origin_field.data_type().clone(),
                        origin_field.is_nullable(),
                    )
                    .with_metadata(origin_field.metadata().clone());
                    projected_fields.push(projected_field);
                }

                let projected_schema = Arc::new(Schema::new(projected_fields));

                let mut projected_batches = Vec::with_capacity(batches.len());
                for batch in batches {
                    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_indices.len());
                    for idx in &column_indices {
                        arrays.push(Arc::clone(batch.column(*idx)));
                    }
                    let projected = RecordBatch::try_new(Arc::clone(&projected_schema), arrays)?;
                    projected_batches.push(projected);
                }

                let combined_batch = if projected_batches.is_empty() {
                    RecordBatch::new_empty(Arc::clone(&projected_schema))
                } else if projected_batches.len() == 1 {
                    projected_batches.remove(0)
                } else {
                    concat_batches(&projected_schema, projected_batches.iter())?
                };

                let select_execution = SelectExecution::from_batch(
                    view_display_name.clone(),
                    Arc::clone(&projected_schema),
                    combined_batch,
                );

                Ok(Some(RuntimeStatementResult::Select {
                    execution: Box::new(select_execution),
                    table_name: view_display_name,
                    schema: projected_schema,
                }))
            }
        }
    }

    fn try_execute_simple_derived_select(
        &self,
        query: &Query,
        visited_views: &mut FxHashSet<String>,
    ) -> SqlResult<Option<RuntimeStatementResult<P>>> {
        use sqlparser::ast::{Expr as SqlExpr, SelectItem, SetExpr};

        // Support only plain SELECT queries without WITH, ORDER BY, LIMIT, or FETCH clauses.
        if query.with.is_some() || query.order_by.is_some() || query.limit_clause.is_some() {
            return Ok(None);
        }
        if query.fetch.is_some() {
            return Ok(None);
        }

        let select = match query.body.as_ref() {
            SetExpr::Select(select) => select,
            _ => return Ok(None),
        };

        if select.from.len() != 1 {
            return Ok(None);
        }

        let table_with_joins = &select.from[0];
        let (subquery, alias, lateral) = match &table_with_joins.relation {
            TableFactor::Derived {
                subquery,
                alias,
                lateral,
                ..
            } => (subquery, alias.as_ref(), *lateral),
            _ => return Ok(None),
        };

        if table_with_joins_has_join(table_with_joins) {
            return Err(Error::InvalidArgumentError(
                "Binder Error: derived table queries with JOINs are not supported yet".into(),
            ));
        }

        if lateral {
            return Err(Error::InvalidArgumentError(
                "Binder Error: LATERAL derived tables are not supported yet".into(),
            ));
        }
        if select.distinct.is_some()
            || select.selection.is_some()
            || !group_by_is_empty(&select.group_by)
            || select.having.is_some()
            || !select.cluster_by.is_empty()
            || !select.distribute_by.is_empty()
            || !select.sort_by.is_empty()
            || select.top.is_some()
            || select.value_table_mode.is_some()
            || !select.named_window.is_empty()
            || select.qualify.is_some()
            || select.connect_by.is_some()
        {
            return Err(Error::InvalidArgumentError(
                "Binder Error: advanced clauses are not supported for derived table queries yet"
                    .into(),
            ));
        }

        let inner_query = *subquery.clone();
        let inner_result = self.execute_query_with_view_support(inner_query, visited_views)?;
        let (inner_exec, inner_schema, inner_table_name) =
            self.extract_select_result(inner_result)?;

        let alias_name = alias.map(|a| a.name.value.clone());
        let alias_columns = alias.and_then(|a| {
            if a.columns.is_empty() {
                None
            } else {
                Some(
                    a.columns
                        .iter()
                        .map(|col| col.name.value.clone())
                        .collect::<Vec<_>>(),
                )
            }
        });

        if let Some(ref columns) = alias_columns
            && columns.len() != inner_schema.fields().len()
        {
            return Err(Error::InvalidArgumentError(
                "Binder Error: derived table column alias count must match projection".into(),
            ));
        }

        let alias_lower = alias_name.as_ref().map(|name| name.to_ascii_lowercase());
        let inner_lower = inner_table_name.to_ascii_lowercase();

        enum DerivedProjection {
            All,
            Columns(Vec<(String, String)>),
        }

        let resolve_compound_identifier = |parts: &[Ident]| -> SqlResult<String> {
            if parts.is_empty() {
                return Err(Error::InvalidArgumentError(
                    "Binder Error: empty identifier in derived table projection".into(),
                ));
            }
            if parts.len() == 1 {
                return Ok(parts[0].value.clone());
            }
            if parts.len() == 2 {
                let qualifier_lower = parts[0].value.to_ascii_lowercase();
                let qualifier_display = parts[0].value.clone();
                if let Some(ref alias_lower) = alias_lower {
                    if qualifier_lower != *alias_lower {
                        return Err(Error::InvalidArgumentError(format!(
                            "Binder Error: derived table qualifier '{}' does not match alias '{}'",
                            qualifier_display,
                            alias_name.as_deref().unwrap_or(""),
                        )));
                    }
                } else if qualifier_lower != inner_lower {
                    return Err(Error::InvalidArgumentError(format!(
                        "Binder Error: derived table qualifier '{}' does not match subquery name '{}'",
                        qualifier_display, inner_table_name
                    )));
                }
                return Ok(parts[1].value.clone());
            }
            Err(Error::InvalidArgumentError(
                "Binder Error: multi-part qualified identifiers are not supported for derived tables yet"
                    .into(),
            ))
        };

        let build_projection_columns = |items: &[SelectItem]| -> SqlResult<Vec<(String, String)>> {
            let mut columns = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    SelectItem::UnnamedExpr(SqlExpr::Identifier(ident)) => {
                        let name = ident.value.clone();
                        columns.push((name.clone(), name));
                    }
                    SelectItem::ExprWithAlias { expr, alias } => {
                        let source = match expr {
                            SqlExpr::Identifier(ident) => ident.value.clone(),
                            SqlExpr::CompoundIdentifier(parts) => {
                                resolve_compound_identifier(parts)?
                            }
                            _ => {
                                return Err(Error::InvalidArgumentError(
                                    "Binder Error: complex expressions in derived table projections are not supported yet"
                                        .into(),
                                ));
                            }
                        };
                        columns.push((source, alias.value.clone()))
                    }
                    SelectItem::UnnamedExpr(SqlExpr::CompoundIdentifier(parts)) => {
                        let column = resolve_compound_identifier(parts)?;
                        columns.push((column.clone(), column));
                    }
                    other => {
                        return Err(Error::InvalidArgumentError(format!(
                            "Binder Error: unsupported derived table projection {:?}",
                            other
                        )));
                    }
                }
            }
            Ok(columns)
        };

        let projection = if select.projection.len() == 1 {
            match &select.projection[0] {
                SelectItem::Wildcard(_) => DerivedProjection::All,
                SelectItem::QualifiedWildcard(kind, _) => match kind {
                    SelectItemQualifiedWildcardKind::ObjectName(name) => {
                        let qualifier = Self::object_name_to_string(name)?;
                        let qualifier_lower = qualifier.to_ascii_lowercase();
                        if let Some(ref alias_lower) = alias_lower {
                            if qualifier_lower != *alias_lower {
                                return Err(Error::InvalidArgumentError(format!(
                                    "Binder Error: derived table qualifier '{}' does not match alias '{}'",
                                    qualifier,
                                    alias_name.as_deref().unwrap_or(""),
                                )));
                            }
                        } else if qualifier_lower != inner_lower {
                            return Err(Error::InvalidArgumentError(format!(
                                "Binder Error: derived table qualifier '{}' does not match subquery name '{}'",
                                qualifier, inner_table_name
                            )));
                        }
                        DerivedProjection::All
                    }
                    SelectItemQualifiedWildcardKind::Expr(_) => {
                        return Err(Error::InvalidArgumentError(
                                "Binder Error: expression-qualified wildcards are not supported for derived tables yet"
                                    .into(),
                            ));
                    }
                },
                _ => DerivedProjection::Columns(build_projection_columns(&select.projection)?),
            }
        } else {
            if select.projection.iter().any(|item| {
                matches!(
                    item,
                    SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(_, _)
                )
            }) {
                return Err(Error::InvalidArgumentError(
                    "Binder Error: derived table projections cannot mix wildcards with explicit columns"
                        .into(),
                ));
            }
            DerivedProjection::Columns(build_projection_columns(&select.projection)?)
        };

        let mut batches = inner_exec.collect()?;
        let output_table_name = alias_name.clone().unwrap_or(inner_table_name.clone());

        let mut name_to_index = FxHashMap::default();
        for (idx, field) in inner_schema.fields().iter().enumerate() {
            name_to_index.insert(field.name().to_ascii_lowercase(), idx);
        }
        if let Some(ref columns) = alias_columns {
            for (idx, alias_col) in columns.iter().enumerate() {
                name_to_index.insert(alias_col.to_ascii_lowercase(), idx);
            }
        }

        let mut build_projected_result = |column_mappings: Vec<(String, String)>| -> SqlResult<_> {
            let mut column_indices = Vec::with_capacity(column_mappings.len());
            let mut projected_fields = Vec::with_capacity(column_mappings.len());

            for (source, output) in column_mappings {
                let key = source.to_ascii_lowercase();
                let Some(&idx) = name_to_index.get(&key) else {
                    return Err(Error::InvalidArgumentError(format!(
                        "Binder Error: derived table does not provide a column named '{}'",
                        source
                    )));
                };
                column_indices.push(idx);
                let origin_field = inner_schema.field(idx).clone();
                let projected_field = Field::new(
                    &output,
                    origin_field.data_type().clone(),
                    origin_field.is_nullable(),
                )
                .with_metadata(origin_field.metadata().clone());
                projected_fields.push(projected_field);
            }

            let projected_schema = Arc::new(Schema::new(projected_fields));
            let mut projected_batches = Vec::with_capacity(batches.len());
            for batch in batches.drain(..) {
                let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_indices.len());
                for idx in &column_indices {
                    arrays.push(Arc::clone(batch.column(*idx)));
                }
                let projected = RecordBatch::try_new(Arc::clone(&projected_schema), arrays)
                    .map_err(|err| {
                        Error::Internal(format!(
                            "failed to construct derived table projection batch: {err}"
                        ))
                    })?;
                projected_batches.push(projected);
            }

            let combined_batch = if projected_batches.is_empty() {
                RecordBatch::new_empty(Arc::clone(&projected_schema))
            } else if projected_batches.len() == 1 {
                projected_batches.remove(0)
            } else {
                concat_batches(&projected_schema, projected_batches.iter()).map_err(|err| {
                    Error::Internal(format!(
                        "failed to concatenate derived table batches: {err}"
                    ))
                })?
            };

            Ok((projected_schema, combined_batch))
        };

        let (final_schema, combined_batch) = match projection {
            DerivedProjection::All => {
                if let Some(columns) = alias_columns {
                    let mappings = columns
                        .iter()
                        .map(|name| (name.clone(), name.clone()))
                        .collect::<Vec<_>>();
                    build_projected_result(mappings)?
                } else {
                    let schema = Arc::clone(&inner_schema);
                    let combined = if batches.is_empty() {
                        RecordBatch::new_empty(Arc::clone(&schema))
                    } else if batches.len() == 1 {
                        batches.remove(0)
                    } else {
                        concat_batches(&schema, batches.iter()).map_err(|err| {
                            Error::Internal(format!(
                                "failed to concatenate derived table batches: {err}"
                            ))
                        })?
                    };
                    (schema, combined)
                }
            }
            DerivedProjection::Columns(mappings) => build_projected_result(mappings)?,
        };

        let execution = SelectExecution::from_batch(
            output_table_name.clone(),
            Arc::clone(&final_schema),
            combined_batch,
        );

        Ok(Some(RuntimeStatementResult::Select {
            execution: Box::new(execution),
            table_name: output_table_name,
            schema: final_schema,
        }))
    }

    fn try_execute_view_set_operation(
        &self,
        query: &Query,
        visited_views: &mut FxHashSet<String>,
    ) -> SqlResult<Option<RuntimeStatementResult<P>>> {
        if !matches!(query.body.as_ref(), SetExpr::SetOperation { .. }) {
            return Ok(None);
        }

        if query.with.is_some()
            || query.order_by.is_some()
            || query.limit_clause.is_some()
            || query.fetch.is_some()
        {
            return Ok(None);
        }

        if !self.set_expr_contains_view(query.body.as_ref())? {
            return Ok(None);
        }

        let result = self.evaluate_set_expr(query.body.as_ref(), visited_views)?;
        Ok(Some(result))
    }

    fn evaluate_set_expr(
        &self,
        expr: &SetExpr,
        visited_views: &mut FxHashSet<String>,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        match expr {
            SetExpr::SetOperation {
                left,
                right,
                op,
                set_quantifier,
            } => {
                let left_result = self.evaluate_set_expr(left.as_ref(), visited_views)?;
                let right_result = self.evaluate_set_expr(right.as_ref(), visited_views)?;
                self.combine_set_results(left_result, right_result, *op, *set_quantifier)
            }
            SetExpr::Query(subquery) => {
                self.execute_query_with_view_support(*subquery.clone(), visited_views)
            }
            _ => self.execute_setexpr_query(expr, visited_views),
        }
    }

    fn execute_setexpr_query(
        &self,
        expr: &SetExpr,
        visited_views: &mut FxHashSet<String>,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        let sql = expr.to_string();
        let dialect = GenericDialect {};
        let statements = parse_sql_with_recursion_limit(&dialect, &sql).map_err(|err| {
            Error::InvalidArgumentError(format!(
                "failed to parse expanded view query '{sql}': {err}"
            ))
        })?;

        let mut iter = statements.into_iter();
        let statement = iter.next().ok_or_else(|| {
            Error::InvalidArgumentError("expanded view query did not produce a statement".into())
        })?;
        if iter.next().is_some() {
            return Err(Error::InvalidArgumentError(
                "expanded view query produced multiple statements".into(),
            ));
        }

        let query = match statement {
            Statement::Query(q) => *q,
            other => {
                return Err(Error::InvalidArgumentError(format!(
                    "expanded view query did not produce a SELECT statement: {other:?}"
                )));
            }
        };

        self.execute_query_with_view_support(query, visited_views)
    }

    fn combine_set_results(
        &self,
        left: RuntimeStatementResult<P>,
        right: RuntimeStatementResult<P>,
        op: SetOperator,
        quantifier: SetQuantifier,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        match op {
            SetOperator::Union => self.union_select_results(left, right, quantifier),
            other => Err(Error::InvalidArgumentError(format!(
                "Binder Error: unsupported set operator {other:?} in view query"
            ))),
        }
    }

    fn union_select_results(
        &self,
        left: RuntimeStatementResult<P>,
        right: RuntimeStatementResult<P>,
        quantifier: SetQuantifier,
    ) -> SqlResult<RuntimeStatementResult<P>> {
        let (left_exec, left_schema, left_name) = self.extract_select_result(left)?;
        let (right_exec, right_schema, _) = self.extract_select_result(right)?;

        self.ensure_schemas_compatible(&left_schema, &right_schema)?;

        let mut batches = Vec::new();
        batches.extend(left_exec.collect()?);
        batches.extend(right_exec.collect()?);

        let mut combined_batch = if batches.is_empty() {
            RecordBatch::new_empty(Arc::clone(&left_schema))
        } else if batches.len() == 1 {
            batches.pop().expect("length checked above")
        } else {
            concat_batches(&left_schema, batches.iter()).map_err(|err| {
                Error::Internal(format!("failed to concatenate UNION batches: {err}"))
            })?
        };

        if matches!(quantifier, SetQuantifier::Distinct) {
            combined_batch = self.distinct_batch(&left_schema, combined_batch)?;
        }

        let execution = SelectExecution::from_batch(
            left_name.clone(),
            Arc::clone(&left_schema),
            combined_batch,
        );

        Ok(RuntimeStatementResult::Select {
            execution: Box::new(execution),
            table_name: left_name,
            schema: left_schema,
        })
    }

    fn extract_select_result(
        &self,
        result: RuntimeStatementResult<P>,
    ) -> SqlResult<(SelectExecution<P>, Arc<Schema>, String)> {
        match result {
            RuntimeStatementResult::Select {
                execution,
                schema,
                table_name,
            } => Ok((*execution, schema, table_name)),
            _ => Err(Error::InvalidArgumentError(
                "expected SELECT result while evaluating set operation".into(),
            )),
        }
    }

    fn ensure_schemas_compatible(&self, left: &Arc<Schema>, right: &Arc<Schema>) -> SqlResult<()> {
        if left.fields().len() != right.fields().len() {
            return Err(Error::InvalidArgumentError(
                "Binder Error: UNION inputs project different column counts".into(),
            ));
        }

        for (idx, (l_field, r_field)) in left.fields().iter().zip(right.fields().iter()).enumerate()
        {
            if l_field.data_type() != r_field.data_type() {
                return Err(Error::InvalidArgumentError(format!(
                    "Binder Error: UNION column {} type mismatch ({:?} vs {:?})",
                    idx + 1,
                    l_field.data_type(),
                    r_field.data_type()
                )));
            }
        }

        Ok(())
    }

    fn distinct_batch(&self, schema: &Arc<Schema>, batch: RecordBatch) -> SqlResult<RecordBatch> {
        if batch.num_rows() <= 1 {
            return Ok(batch);
        }

        let sort_fields: Vec<SortField> = schema
            .fields()
            .iter()
            .map(|field| SortField::new(field.data_type().clone()))
            .collect();

        let converter = RowConverter::new(sort_fields)
            .map_err(|err| Error::Internal(format!("failed to initialize row converter: {err}")))?;
        let rows = converter
            .convert_columns(batch.columns())
            .map_err(|err| Error::Internal(format!("failed to row-encode union result: {err}")))?;

        let mut seen = FxHashSet::default();
        let mut indices = Vec::new();
        let mut has_duplicates = false;
        for (idx, row) in rows.iter().enumerate() {
            if seen.insert(row) {
                indices.push(idx as u32);
            } else {
                has_duplicates = true;
            }
        }

        if !has_duplicates {
            return Ok(batch);
        }

        let index_array = UInt32Array::from(indices);
        let mut columns = Vec::with_capacity(batch.num_columns());
        for column in batch.columns() {
            let taken = take(column.as_ref(), &index_array, None).map_err(|err| {
                Error::Internal(format!("failed to materialize DISTINCT rows: {err}"))
            })?;
            columns.push(taken);
        }

        RecordBatch::try_new(Arc::clone(schema), columns)
            .map_err(|err| Error::Internal(format!("failed to build DISTINCT RecordBatch: {err}")))
    }

    fn set_expr_contains_view(&self, expr: &SetExpr) -> SqlResult<bool> {
        match expr {
            SetExpr::Select(select) => self.select_contains_view(select.as_ref()),
            SetExpr::Query(query) => self.set_expr_contains_view(&query.body),
            SetExpr::SetOperation { left, right, .. } => Ok(self
                .set_expr_contains_view(left.as_ref())?
                || self.set_expr_contains_view(right.as_ref())?),
            _ => Ok(false),
        }
    }

    fn select_contains_view(&self, select: &Select) -> SqlResult<bool> {
        for from_item in &select.from {
            if self.table_with_joins_contains_view(from_item)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn table_with_joins_contains_view(&self, table: &TableWithJoins) -> SqlResult<bool> {
        if self.table_factor_contains_view(&table.relation)? {
            return Ok(true);
        }

        for join in &table.joins {
            if self.table_factor_contains_view(&join.relation)? {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn table_factor_contains_view(&self, factor: &TableFactor) -> SqlResult<bool> {
        match factor {
            TableFactor::Table { name, .. } => {
                let (_, canonical) = canonical_object_name(name)?;
                let catalog = self.engine.context().table_catalog();
                let Some(table_id) = catalog.table_id(&canonical) else {
                    return Ok(false);
                };
                self.engine.context().is_view(table_id)
            }
            TableFactor::Derived { subquery, .. } => self.set_expr_contains_view(&subquery.body),
            TableFactor::NestedJoin {
                table_with_joins, ..
            } => self.table_with_joins_contains_view(table_with_joins),
            _ => Ok(false),
        }
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

        let (mut select_plan, select_context) =
            self.translate_query_body(query.body.as_ref(), &resolver)?;
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

    /// Internal version of build_select_plan that supports correlated subqueries.
    ///
    /// # Parameters
    /// - `query`: The SQL query to translate
    /// - `resolver`: Identifier resolver for column lookups
    /// - `outer_scopes`: Stack of outer query contexts for correlated column resolution
    /// - `subqueries`: Accumulator for EXISTS subqueries encountered during translation
    /// - `correlated_tracker`: Optional tracker for recording correlated column references
    fn build_select_plan_internal(
        &self,
        query: Query,
        resolver: &IdentifierResolver<'_>,
        outer_scopes: &[IdentifierContext],
        subqueries: &mut Vec<llkv_plan::FilterSubquery>,
        correlated_tracker: Option<&mut SubqueryCorrelatedColumnTracker>,
    ) -> SqlResult<SelectPlan> {
        if self.engine.session().has_active_transaction() && self.engine.session().is_aborted() {
            return Err(Error::TransactionContextError(
                "TransactionContext Error: transaction is aborted".into(),
            ));
        }

        validate_simple_query(&query)?;

        let (mut select_plan, select_context) = self.translate_query_body_internal(
            query.body.as_ref(),
            resolver,
            outer_scopes,
            subqueries,
            correlated_tracker,
        )?;
        if let Some(order_by) = &query.order_by {
            if !select_plan.aggregates.is_empty() {
                return Err(Error::InvalidArgumentError(
                    "ORDER BY is not supported for aggregate queries".into(),
                ));
            }
            let order_plan = self.translate_order_by(resolver, select_context, order_by)?;
            select_plan = select_plan.with_order_by(order_plan);
        }
        Ok(select_plan)
    }

    fn translate_select(
        &self,
        select: &Select,
        resolver: &IdentifierResolver<'_>,
    ) -> SqlResult<(SelectPlan, IdentifierContext)> {
        let mut subqueries = Vec::new();
        let result =
            self.translate_select_internal(select, resolver, &[], &mut subqueries, None)?;
        if !subqueries.is_empty() {
            return Err(Error::Internal(
                "translate_select: unexpected subqueries from non-correlated translation".into(),
            ));
        }
        Ok(result)
    }

    fn translate_query_body(
        &self,
        body: &SetExpr,
        resolver: &IdentifierResolver<'_>,
    ) -> SqlResult<(SelectPlan, IdentifierContext)> {
        let mut subqueries = Vec::new();
        let result =
            self.translate_query_body_internal(body, resolver, &[], &mut subqueries, None)?;
        if !subqueries.is_empty() {
            return Err(Error::Internal(
                "translate_query_body: unexpected subqueries from non-correlated translation"
                    .into(),
            ));
        }
        Ok(result)
    }

    fn translate_query_body_internal(
        &self,
        body: &SetExpr,
        resolver: &IdentifierResolver<'_>,
        outer_scopes: &[IdentifierContext],
        subqueries: &mut Vec<llkv_plan::FilterSubquery>,
        mut correlated_tracker: Option<&mut SubqueryCorrelatedColumnTracker>,
    ) -> SqlResult<(SelectPlan, IdentifierContext)> {
        match body {
            SetExpr::Select(select) => self.translate_select_internal(
                select.as_ref(),
                resolver,
                outer_scopes,
                subqueries,
                correlated_tracker,
            ),
            SetExpr::Query(query) => self.translate_query_body_internal(
                &query.body,
                resolver,
                outer_scopes,
                subqueries,
                correlated_tracker,
            ),
            SetExpr::SetOperation {
                left,
                right,
                op,
                set_quantifier,
            } => {
                let left_tracker = correlated_tracker.reborrow();
                let (left_plan, left_context) = self.translate_query_body_internal(
                    left.as_ref(),
                    resolver,
                    outer_scopes,
                    subqueries,
                    left_tracker,
                )?;

                let right_tracker = correlated_tracker.reborrow();
                let (right_plan, _) = self.translate_query_body_internal(
                    right.as_ref(),
                    resolver,
                    outer_scopes,
                    subqueries,
                    right_tracker,
                )?;

                let operator = match op {
                    sqlparser::ast::SetOperator::Union => llkv_plan::CompoundOperator::Union,
                    sqlparser::ast::SetOperator::Intersect => {
                        llkv_plan::CompoundOperator::Intersect
                    }
                    sqlparser::ast::SetOperator::Except | sqlparser::ast::SetOperator::Minus => {
                        llkv_plan::CompoundOperator::Except
                    }
                };

                let quantifier = match set_quantifier {
                    SetQuantifier::All => llkv_plan::CompoundQuantifier::All,
                    _ => llkv_plan::CompoundQuantifier::Distinct,
                };

                let mut compound = if let Some(existing) = left_plan.compound {
                    existing
                } else {
                    llkv_plan::CompoundSelectPlan::new(left_plan)
                };
                compound.push_operation(operator, quantifier, right_plan);

                let result_plan = SelectPlan::new("").with_compound(compound);

                Ok((result_plan, left_context))
            }
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported query expression: {other:?}"
            ))),
        }
    }

    fn translate_select_internal(
        &self,
        select: &Select,
        resolver: &IdentifierResolver<'_>,
        outer_scopes: &[IdentifierContext],
        subqueries: &mut Vec<llkv_plan::FilterSubquery>,
        mut correlated_tracker: Option<&mut SubqueryCorrelatedColumnTracker>,
    ) -> SqlResult<(SelectPlan, IdentifierContext)> {
        let mut distinct = match &select.distinct {
            None => false,
            Some(Distinct::Distinct) => true,
            Some(Distinct::On(_)) => {
                return Err(Error::InvalidArgumentError(
                    "SELECT DISTINCT ON is not supported".into(),
                ));
            }
        };
        if matches!(
            select.value_table_mode,
            Some(
                sqlparser::ast::ValueTableMode::DistinctAsStruct
                    | sqlparser::ast::ValueTableMode::DistinctAsValue
            )
        ) {
            distinct = true;
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
        if !select.cluster_by.is_empty()
            || !select.distribute_by.is_empty()
            || !select.sort_by.is_empty()
        {
            return Err(Error::InvalidArgumentError(
                "CLUSTER/DISTRIBUTE/SORT BY clauses are not supported".into(),
            ));
        }
        if !select.named_window.is_empty()
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

        let has_joins = select.from.iter().any(table_with_joins_has_join);
        let mut join_conditions: Vec<Option<SqlExpr>> = Vec::new();
        let mut scalar_subqueries: Vec<llkv_plan::ScalarSubquery> = Vec::new();
        // Handle different FROM clause scenarios
        let catalog = self.engine.context().table_catalog();
        let has_group_by = !group_by_is_empty(&select.group_by);
        let (mut plan, id_context) = if select.from.is_empty() {
            if has_group_by {
                return Err(Error::InvalidArgumentError(
                    "GROUP BY requires a FROM clause".into(),
                ));
            }
            // No FROM clause - use empty string for table context (e.g., SELECT 42, SELECT {'a': 1} AS x)
            let mut p = SelectPlan::new("");
            let projections = self.build_projection_list(
                resolver,
                IdentifierContext::new(None),
                &select.projection,
                outer_scopes,
                &mut scalar_subqueries,
                correlated_tracker.reborrow(),
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
            if let Some(alias) = table_alias.as_ref() {
                validate_projection_alias_qualifiers(&select.projection, alias)?;
            }
            if !has_group_by
                && let Some(aggregates) = self.detect_simple_aggregates(&select.projection)?
            {
                p = p.with_aggregates(aggregates);
            } else {
                let projections = self.build_projection_list(
                    resolver,
                    single_table_context.clone(),
                    &select.projection,
                    outer_scopes,
                    &mut scalar_subqueries,
                    correlated_tracker.reborrow(),
                )?;
                p = p.with_projections(projections);
            }
            (p, single_table_context)
        } else {
            // Multiple tables or explicit joins - treat as cross product for now
            let (tables, join_metadata, extracted_filters) = extract_tables(&select.from)?;
            join_conditions = extracted_filters;
            let mut p = SelectPlan::with_tables(tables).with_joins(join_metadata);
            // For multi-table queries, we'll build projections differently
            // For now, just handle simple column references
            let projections = self.build_projection_list(
                resolver,
                IdentifierContext::new(None),
                &select.projection,
                outer_scopes,
                &mut scalar_subqueries,
                correlated_tracker.reborrow(),
            )?;
            p = p.with_projections(projections);
            (p, IdentifierContext::new(None))
        };

        let mut filter_components: Vec<llkv_expr::expr::Expr<'static, String>> = Vec::new();
        let mut all_subqueries = Vec::new();

        if let Some(expr) = &select.selection {
            let materialized_expr = self.materialize_in_subquery(expr.clone())?;
            filter_components.push(translate_condition_with_context(
                self,
                resolver,
                id_context.clone(),
                &materialized_expr,
                outer_scopes,
                &mut all_subqueries,
                correlated_tracker.reborrow(),
            )?);
        }

        // Translate JOIN ON predicates and attach them to the join metadata. INNER joins can
        // safely push these predicates into the WHERE clause, while LEFT joins must retain
        // them on the join so that unmatched left rows are preserved.
        for (idx, join_expr_opt) in join_conditions.iter().enumerate() {
            let Some(join_expr) = join_expr_opt else {
                continue;
            };

            let materialized_expr = self.materialize_in_subquery(join_expr.clone())?;
            let translated = translate_condition_with_context(
                self,
                resolver,
                id_context.clone(),
                &materialized_expr,
                outer_scopes,
                &mut all_subqueries,
                correlated_tracker.reborrow(),
            )?;

            let is_left_join = plan
                .joins
                .get(idx)
                .map(|j| j.join_type == llkv_plan::JoinPlan::Left)
                .unwrap_or(false);

            if let Some(join_meta) = plan.joins.get_mut(idx) {
                join_meta.on_condition = Some(translated.clone());
            }

            if !is_left_join {
                filter_components.push(translated);
            }
        }

        let having_expr = if let Some(having) = &select.having {
            let materialized_expr = self.materialize_in_subquery(having.clone())?;
            let translated = translate_condition_with_context(
                self,
                resolver,
                id_context.clone(),
                &materialized_expr,
                outer_scopes,
                &mut all_subqueries,
                correlated_tracker.reborrow(),
            )?;
            Some(translated)
        } else {
            None
        };

        subqueries.append(&mut all_subqueries);

        let filter = match filter_components.len() {
            0 => None,
            1 if subqueries.is_empty() => Some(llkv_plan::SelectFilter {
                predicate: filter_components.into_iter().next().unwrap(),
                subqueries: Vec::new(),
            }),
            1 => Some(llkv_plan::SelectFilter {
                predicate: filter_components.into_iter().next().unwrap(),
                subqueries: std::mem::take(subqueries),
            }),
            _ => Some(llkv_plan::SelectFilter {
                predicate: llkv_expr::expr::Expr::And(filter_components),
                subqueries: std::mem::take(subqueries),
            }),
        };
        plan = plan.with_filter(filter);
        plan = plan.with_having(having_expr);
        plan = plan.with_scalar_subqueries(std::mem::take(&mut scalar_subqueries));
        plan = plan.with_distinct(distinct);

        let group_by_columns = if has_group_by {
            self.translate_group_by_columns(resolver, id_context.clone(), &select.group_by)?
        } else {
            Vec::new()
        };
        plan = plan.with_group_by(group_by_columns);

        let value_mode = select.value_table_mode.map(convert_value_table_mode);
        plan = plan.with_value_table_mode(value_mode);
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
                    OrderTarget::Column(self.resolve_simple_column_expr(
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
                    OrderTarget::Column(self.resolve_simple_column_expr(
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

    fn translate_group_by_columns(
        &self,
        resolver: &IdentifierResolver<'_>,
        id_context: IdentifierContext,
        group_by: &GroupByExpr,
    ) -> SqlResult<Vec<String>> {
        use sqlparser::ast::Expr as SqlExpr;

        match group_by {
            GroupByExpr::All(_) => Err(Error::InvalidArgumentError(
                "GROUP BY ALL is not supported".into(),
            )),
            GroupByExpr::Expressions(exprs, modifiers) => {
                if !modifiers.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "GROUP BY modifiers are not supported".into(),
                    ));
                }
                let mut columns = Vec::with_capacity(exprs.len());
                for expr in exprs {
                    let parts: Vec<String> = match expr {
                        SqlExpr::Identifier(ident) => vec![ident.value.clone()],
                        SqlExpr::CompoundIdentifier(idents) => {
                            idents.iter().map(|id| id.value.clone()).collect()
                        }
                        _ => {
                            return Err(Error::InvalidArgumentError(
                                "GROUP BY expressions must be simple column references".into(),
                            ));
                        }
                    };
                    let resolution = resolver.resolve(&parts, id_context.clone())?;
                    if !resolution.is_simple() {
                        return Err(Error::InvalidArgumentError(
                            "GROUP BY nested field references are not supported".into(),
                        ));
                    }
                    columns.push(resolution.column().to_string());
                }
                Ok(columns)
            }
        }
    }

    fn resolve_simple_column_expr(
        &self,
        resolver: &IdentifierResolver<'_>,
        context: IdentifierContext,
        expr: &SqlExpr,
    ) -> SqlResult<String> {
        let normalized_expr = self.materialize_in_subquery(expr.clone())?;
        let scalar = translate_scalar_with_context(resolver, context, &normalized_expr)?;
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
                                    "DISTINCT aggregates must be applied to columns not *, e.g. table columns like: 1,0,2,2".into(),
                                ));
                            }
                            AggregateExpr::count_star(alias, false)
                        }
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(arg_expr)) => {
                            if !is_simple_aggregate_column(arg_expr) {
                                return Ok(None);
                            }
                            let column = resolve_column_name(arg_expr)?;
                            AggregateExpr::count_column(column, alias, is_distinct)
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

                    if is_distinct {
                        return Ok(None);
                    }

                    if func_name == "sum" {
                        if let Some(column) = parse_count_nulls_case(arg_expr)? {
                            AggregateExpr::count_nulls(column, alias)
                        } else {
                            if !is_simple_aggregate_column(arg_expr) {
                                return Ok(None);
                            }
                            let column = resolve_column_name(arg_expr)?;
                            AggregateExpr::sum_int64(column, alias)
                        }
                    } else {
                        if !is_simple_aggregate_column(arg_expr) {
                            return Ok(None);
                        }
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
        outer_scopes: &[IdentifierContext],
        scalar_subqueries: &mut Vec<llkv_plan::ScalarSubquery>,
        mut correlated_tracker: Option<&mut SubqueryCorrelatedColumnTracker>,
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
                        // Use the original SQL expression string as the alias for complex expressions.
                        // This preserves operators like unary plus (e.g., "+ col2" rather than "col2").
                        let alias = expr.to_string();
                        let normalized_expr = if matches!(expr, SqlExpr::Subquery(_)) {
                            expr.clone()
                        } else {
                            self.materialize_in_subquery(expr.clone())?
                        };
                        let scalar = {
                            let tracker_view = correlated_tracker.reborrow();
                            let mut builder = ScalarSubqueryPlanner {
                                engine: self,
                                scalar_subqueries,
                            };
                            let mut tracker_wrapper =
                                SubqueryCorrelatedTracker::from_option(tracker_view);
                            translate_scalar_internal(
                                &normalized_expr,
                                Some(resolver),
                                Some(&id_context),
                                outer_scopes,
                                &mut tracker_wrapper,
                                Some(&mut builder),
                            )?
                        };
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
                        let normalized_expr = if matches!(expr, SqlExpr::Subquery(_)) {
                            expr.clone()
                        } else {
                            self.materialize_in_subquery(expr.clone())?
                        };
                        let scalar = {
                            let tracker_view = correlated_tracker.reborrow();
                            let mut builder = ScalarSubqueryPlanner {
                                engine: self,
                                scalar_subqueries,
                            };
                            let mut tracker_wrapper =
                                SubqueryCorrelatedTracker::from_option(tracker_view);
                            translate_scalar_internal(
                                &normalized_expr,
                                Some(resolver),
                                Some(&id_context),
                                outer_scopes,
                                &mut tracker_wrapper,
                                Some(&mut builder),
                            )?
                        };
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
                    "constraint_enforcement_mode" => {
                        if values.len() != 1 {
                            return Err(Error::InvalidArgumentError(
                                "SET constraint_enforcement_mode expects exactly one value".into(),
                            ));
                        }

                        let normalized = match &values[0] {
                            SqlExpr::Value(value_with_span) => value_with_span
                                .value
                                .clone()
                                .into_string()
                                .map(|s| s.to_ascii_lowercase()),
                            SqlExpr::Identifier(ident) => Some(ident.value.to_ascii_lowercase()),
                            _ => None,
                        };

                        let mode = match normalized.as_deref() {
                            Some("immediate") => ConstraintEnforcementMode::Immediate,
                            Some("deferred") => ConstraintEnforcementMode::Deferred,
                            _ => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "unsupported value for SET constraint_enforcement_mode: {}",
                                    values[0]
                                )));
                            }
                        };

                        self.engine.session().set_constraint_enforcement_mode(mode);

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

    fn handle_vacuum(&self, vacuum: VacuumStatement) -> SqlResult<RuntimeStatementResult<P>> {
        // Only support REINDEX with a table name (which is treated as index name in LLKV)
        if vacuum.reindex {
            let index_name = vacuum.table_name.ok_or_else(|| {
                Error::InvalidArgumentError("REINDEX requires an index name".to_string())
            })?;

            let (display_name, canonical_name) = canonical_object_name(&index_name)?;

            let plan = ReindexPlan::new(display_name.clone()).with_canonical(canonical_name);

            let statement = PlanStatement::Reindex(plan);
            self.engine.execute_statement(statement).map_err(|err| {
                tracing::error!("REINDEX failed for '{}': {}", display_name, err);
                err
            })
        } else {
            // Other VACUUM variants are not supported
            Err(Error::InvalidArgumentError(
                "Only REINDEX is supported; general VACUUM is not implemented".to_string(),
            ))
        }
    }
}

fn bind_plan_parameters(
    plan: &mut PlanStatement,
    params: &[SqlParamValue],
    expected_count: usize,
) -> SqlResult<()> {
    if expected_count == 0 {
        return Ok(());
    }

    match plan {
        PlanStatement::Update(update) => bind_update_plan_parameters(update, params),
        other => Err(Error::InvalidArgumentError(format!(
            "prepared execution is not yet supported for {:?}",
            other
        ))),
    }
}

fn bind_update_plan_parameters(plan: &mut UpdatePlan, params: &[SqlParamValue]) -> SqlResult<()> {
    for assignment in &mut plan.assignments {
        match &mut assignment.value {
            AssignmentValue::Literal(value) => bind_plan_value(value, params)?,
            AssignmentValue::Expression(expr) => bind_scalar_expr(expr, params)?,
        }
    }

    if let Some(filter) = &mut plan.filter {
        bind_predicate_expr(filter, params)?;
    }

    Ok(())
}

fn bind_plan_value(value: &mut PlanValue, params: &[SqlParamValue]) -> SqlResult<()> {
    match value {
        PlanValue::String(text) => {
            if let Some(index) = parse_placeholder_marker(text) {
                let param = params
                    .get(index.saturating_sub(1))
                    .ok_or_else(|| missing_parameter_error(index, params.len()))?;
                *value = param.as_plan_value();
            }
        }
        PlanValue::Struct(fields) => {
            for field in fields.values_mut() {
                bind_plan_value(field, params)?;
            }
        }
        PlanValue::Null
        | PlanValue::Integer(_)
        | PlanValue::Float(_)
        | PlanValue::Decimal(_)
        | PlanValue::Date32(_)
        | PlanValue::Interval(_) => {}
    }
    Ok(())
}

fn bind_scalar_expr(
    expr: &mut llkv_expr::expr::ScalarExpr<String>,
    params: &[SqlParamValue],
) -> SqlResult<()> {
    use llkv_expr::expr::ScalarExpr;

    match expr {
        ScalarExpr::Column(_) => {}
        ScalarExpr::Literal(lit) => bind_literal(lit, params)?,
        ScalarExpr::Binary { left, right, .. } => {
            bind_scalar_expr(left, params)?;
            bind_scalar_expr(right, params)?;
        }
        ScalarExpr::Not(inner) => bind_scalar_expr(inner, params)?,
        ScalarExpr::IsNull { expr: inner, .. } => bind_scalar_expr(inner, params)?,
        ScalarExpr::Aggregate(_) => {
            return Err(Error::InvalidArgumentError(
                "parameters inside aggregate expressions are not supported".into(),
            ));
        }
        ScalarExpr::GetField { base, .. } => bind_scalar_expr(base, params)?,
        ScalarExpr::Cast { expr: inner, .. } => bind_scalar_expr(inner, params)?,
        ScalarExpr::Compare { left, right, .. } => {
            bind_scalar_expr(left, params)?;
            bind_scalar_expr(right, params)?;
        }
        ScalarExpr::Coalesce(list) => {
            for item in list {
                bind_scalar_expr(item, params)?;
            }
        }
        ScalarExpr::ScalarSubquery(_) => {
            return Err(Error::InvalidArgumentError(
                "parameters inside scalar subqueries are not supported yet".into(),
            ));
        }
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            if let Some(op) = operand {
                bind_scalar_expr(op, params)?;
            }
            for (when, then) in branches {
                bind_scalar_expr(when, params)?;
                bind_scalar_expr(then, params)?;
            }
            if let Some(else_expr) = else_expr {
                bind_scalar_expr(else_expr, params)?;
            }
        }
        ScalarExpr::Random => {}
    }

    Ok(())
}

fn bind_predicate_expr(
    expr: &mut llkv_expr::expr::Expr<'static, String>,
    params: &[SqlParamValue],
) -> SqlResult<()> {
    use llkv_expr::expr::Expr;

    match expr {
        Expr::And(list) | Expr::Or(list) => {
            for sub in list {
                bind_predicate_expr(sub, params)?;
            }
        }
        Expr::Not(inner) => bind_predicate_expr(inner, params)?,
        Expr::Pred(filter) => bind_filter_operator(&mut filter.op, params)?,
        Expr::Compare { left, right, .. } => {
            bind_scalar_expr(left, params)?;
            bind_scalar_expr(right, params)?;
        }
        Expr::InList {
            expr: inner, list, ..
        } => {
            bind_scalar_expr(inner, params)?;
            for item in list {
                bind_scalar_expr(item, params)?;
            }
        }
        Expr::IsNull { expr: inner, .. } => bind_scalar_expr(inner, params)?,
        Expr::Literal(_) => {}
        Expr::Exists(_) => {
            return Err(Error::InvalidArgumentError(
                "parameters inside EXISTS subqueries are not supported yet".into(),
            ));
        }
    }
    Ok(())
}

fn bind_filter_operator(
    op: &mut llkv_expr::expr::Operator<'static>,
    params: &[SqlParamValue],
) -> SqlResult<()> {
    use llkv_expr::expr::Operator;

    match op {
        Operator::Equals(lit)
        | Operator::GreaterThan(lit)
        | Operator::GreaterThanOrEquals(lit)
        | Operator::LessThan(lit)
        | Operator::LessThanOrEquals(lit) => bind_literal(lit, params),
        Operator::Range { lower, upper } => {
            bind_bound_literal(lower, params)?;
            bind_bound_literal(upper, params)
        }
        Operator::In(list) => {
            for lit in *list {
                if let Literal::String(text) = lit
                    && parse_placeholder_marker(text).is_some()
                {
                    return Err(Error::InvalidArgumentError(
                        "IN predicates do not yet support bound parameters".into(),
                    ));
                }
            }
            Ok(())
        }
        Operator::StartsWith { pattern, .. }
        | Operator::EndsWith { pattern, .. }
        | Operator::Contains { pattern, .. } => {
            if pattern.contains(PARAM_SENTINEL_PREFIX) {
                return Err(Error::InvalidArgumentError(
                    "LIKE-style predicates do not yet support bound parameters".into(),
                ));
            }
            Ok(())
        }
        Operator::IsNull | Operator::IsNotNull => Ok(()),
    }
}

fn bind_bound_literal(bound: &mut Bound<Literal>, params: &[SqlParamValue]) -> SqlResult<()> {
    match bound {
        Bound::Included(lit) | Bound::Excluded(lit) => bind_literal(lit, params),
        Bound::Unbounded => Ok(()),
    }
}

fn bind_literal(literal: &mut Literal, params: &[SqlParamValue]) -> SqlResult<()> {
    match literal {
        Literal::String(text) => {
            if let Some(index) = parse_placeholder_marker(text) {
                let param = params
                    .get(index.saturating_sub(1))
                    .ok_or_else(|| missing_parameter_error(index, params.len()))?;
                *literal = param.as_literal();
            }
            Ok(())
        }
        Literal::Struct(fields) => {
            for (_, value) in fields.iter_mut() {
                bind_literal(value, params)?;
            }
            Ok(())
        }
        Literal::Integer(_)
        | Literal::Float(_)
        | Literal::Decimal(_)
        | Literal::Boolean(_)
        | Literal::Null
        | Literal::Date32(_)
        | Literal::Interval(_) => Ok(()),
    }
}

fn missing_parameter_error(index: usize, provided: usize) -> Error {
    Error::InvalidArgumentError(format!(
        "missing parameter value for placeholder {} ({} provided)",
        index, provided
    ))
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

    let column_names_lower: FxHashSet<String> = column_names
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

#[derive(Clone)]
struct InlineColumn {
    name: &'static str,
    data_type: DataType,
    nullable: bool,
}

impl InlineColumn {
    fn utf8(name: &'static str, nullable: bool) -> Self {
        Self {
            name,
            data_type: DataType::Utf8,
            nullable,
        }
    }

    fn bool(name: &'static str, nullable: bool) -> Self {
        Self {
            name,
            data_type: DataType::Boolean,
            nullable,
        }
    }

    fn int32(name: &'static str, nullable: bool) -> Self {
        Self {
            name,
            data_type: DataType::Int32,
            nullable,
        }
    }
}

#[derive(Clone)]
struct InlineRow {
    values: Vec<InlineValue>,
}

#[derive(Clone, Debug)]
enum InlineValue {
    String(Option<String>),
    Int32(Option<i32>),
    Bool(Option<bool>),
    Null,
}

struct InlineView {
    columns: Vec<InlineColumn>,
    rows: Vec<InlineRow>,
}

impl InlineView {
    fn new(columns: Vec<InlineColumn>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
        }
    }

    fn add_row(&mut self, values: Vec<InlineValue>) -> SqlResult<()> {
        if values.len() != self.columns.len() {
            return Err(Error::Internal(
                "inline row does not match column layout".into(),
            ));
        }
        for (value, column) in values.iter().zip(self.columns.iter()) {
            if !value.matches_type(&column.data_type) {
                return Err(Error::Internal(format!(
                    "inline value type mismatch for column {}",
                    column.name
                )));
            }
        }
        self.rows.push(InlineRow { values });
        Ok(())
    }

    fn apply_selection(&mut self, expr: &SqlExpr, alias: Option<&str>) -> SqlResult<()> {
        let mut filtered = Vec::with_capacity(self.rows.len());
        for row in self.rows.drain(..) {
            if evaluate_predicate(expr, &row, &self.columns, alias)? {
                filtered.push(row);
            }
        }
        self.rows = filtered;
        Ok(())
    }

    fn into_record_batch(self) -> SqlResult<(Arc<Schema>, RecordBatch)> {
        let mut fields = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            fields.push(Field::new(
                column.name,
                column.data_type.clone(),
                column.nullable,
            ));
        }
        let schema = Arc::new(Schema::new(fields));

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.columns.len());
        for (idx, column) in self.columns.iter().enumerate() {
            match &column.data_type {
                DataType::Utf8 => {
                    let mut values = Vec::with_capacity(self.rows.len());
                    for row in &self.rows {
                        match &row.values[idx] {
                            InlineValue::String(val) => values.push(val.clone()),
                            InlineValue::Null => values.push(None),
                            other => {
                                return Err(Error::Internal(format!(
                                    "unexpected string value for column {}: {:?}",
                                    column.name, other
                                )));
                            }
                        }
                    }
                    arrays.push(Arc::new(StringArray::from(values)) as ArrayRef);
                }
                DataType::Boolean => {
                    let mut values = Vec::with_capacity(self.rows.len());
                    for row in &self.rows {
                        match &row.values[idx] {
                            InlineValue::Bool(val) => values.push(*val),
                            InlineValue::Null => values.push(None),
                            other => {
                                return Err(Error::Internal(format!(
                                    "unexpected boolean value for column {}: {:?}",
                                    column.name, other
                                )));
                            }
                        }
                    }
                    arrays.push(Arc::new(BooleanArray::from(values)) as ArrayRef);
                }
                DataType::Int32 => {
                    let mut values = Vec::with_capacity(self.rows.len());
                    for row in &self.rows {
                        match &row.values[idx] {
                            InlineValue::Int32(val) => values.push(*val),
                            InlineValue::Null => values.push(None),
                            other => {
                                return Err(Error::Internal(format!(
                                    "unexpected integer value for column {}: {:?}",
                                    column.name, other
                                )));
                            }
                        }
                    }
                    arrays.push(Arc::new(Int32Array::from(values)) as ArrayRef);
                }
                other => {
                    return Err(Error::Internal(format!(
                        "unsupported inline column type: {:?}",
                        other
                    )));
                }
            }
        }

        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)
            .map_err(|e| Error::Internal(format!("failed to build inline batch: {}", e)))?;
        Ok((schema, batch))
    }
}

impl InlineValue {
    fn is_null(&self) -> bool {
        matches!(
            self,
            InlineValue::Null
                | InlineValue::String(None)
                | InlineValue::Int32(None)
                | InlineValue::Bool(None)
        )
    }

    fn matches_type(&self, data_type: &DataType) -> bool {
        if matches!(self, InlineValue::Null) {
            return true;
        }
        match data_type {
            DataType::Utf8 => matches!(self, InlineValue::String(_) | InlineValue::Null),
            DataType::Boolean => matches!(self, InlineValue::Bool(_) | InlineValue::Null),
            DataType::Int32 => matches!(self, InlineValue::Int32(_) | InlineValue::Null),
            _ => false,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            InlineValue::Bool(value) => *value,
            _ => None,
        }
    }
}

fn ensure_inline_select_supported(select: &Select, query: &Query) -> SqlResult<()> {
    if select.distinct.is_some() {
        return Err(Error::InvalidArgumentError(
            "DISTINCT is not supported for information_schema or pragma queries".into(),
        ));
    }
    if select.top.is_some() {
        return Err(Error::InvalidArgumentError(
            "TOP clauses are not supported for information_schema or pragma queries".into(),
        ));
    }
    if select.exclude.is_some()
        || select.into.is_some()
        || !select.lateral_views.is_empty()
        || select.prewhere.is_some()
    {
        return Err(Error::InvalidArgumentError(
            "the requested SELECT features are not supported for information_schema or pragma queries"
                .into(),
        ));
    }
    if !group_by_is_empty(&select.group_by) {
        return Err(Error::InvalidArgumentError(
            "GROUP BY is not supported for information_schema or pragma queries".into(),
        ));
    }
    if select.having.is_some() {
        return Err(Error::InvalidArgumentError(
            "HAVING clauses are not supported for information_schema or pragma queries".into(),
        ));
    }
    if !select.cluster_by.is_empty()
        || !select.distribute_by.is_empty()
        || !select.sort_by.is_empty()
        || select.value_table_mode.is_some()
        || select.connect_by.is_some()
        || !select.named_window.is_empty()
        || select.qualify.is_some()
    {
        return Err(Error::InvalidArgumentError(
            "the requested SELECT modifiers are not supported for information_schema or pragma queries"
                .into(),
        ));
    }
    if query.with.is_some() {
        return Err(Error::InvalidArgumentError(
            "WITH clauses are not supported for information_schema or pragma queries".into(),
        ));
    }
    if query.fetch.is_some() {
        return Err(Error::InvalidArgumentError(
            "FETCH clauses are not supported for information_schema or pragma queries".into(),
        ));
    }
    Ok(())
}

fn extract_limit_count(query: &Query) -> SqlResult<Option<usize>> {
    use sqlparser::ast::{Expr, LimitClause};

    let Some(limit_clause) = &query.limit_clause else {
        return Ok(None);
    };

    match limit_clause {
        LimitClause::LimitOffset {
            limit,
            offset,
            limit_by,
        } => {
            if offset.is_some() || !limit_by.is_empty() {
                return Err(Error::InvalidArgumentError(
                    "OFFSET/LIMIT BY are not supported for information_schema or pragma queries"
                        .into(),
                ));
            }
            match limit {
                None => Ok(None),
                Some(Expr::Value(value)) => match &value.value {
                    sqlparser::ast::Value::Number(text, _) => {
                        let parsed = text.parse::<i64>().map_err(|_| {
                            Error::InvalidArgumentError("LIMIT must be a positive integer".into())
                        })?;
                        if parsed < 0 {
                            return Err(Error::InvalidArgumentError(
                                "LIMIT must be non-negative".into(),
                            ));
                        }
                        Ok(Some(parsed as usize))
                    }
                    sqlparser::ast::Value::SingleQuotedString(text) => {
                        let parsed = text.parse::<i64>().map_err(|_| {
                            Error::InvalidArgumentError("LIMIT must be a positive integer".into())
                        })?;
                        if parsed < 0 {
                            return Err(Error::InvalidArgumentError(
                                "LIMIT must be non-negative".into(),
                            ));
                        }
                        Ok(Some(parsed as usize))
                    }
                    _ => Err(Error::InvalidArgumentError(
                        "LIMIT must be a numeric literal for information_schema or pragma queries"
                            .into(),
                    )),
                },
                Some(_) => Err(Error::InvalidArgumentError(
                    "LIMIT must be a literal for information_schema or pragma queries".into(),
                )),
            }
        }
        LimitClause::OffsetCommaLimit { .. } => Err(Error::InvalidArgumentError(
            "LIMIT with comma offset is not supported for information_schema or pragma queries"
                .into(),
        )),
    }
}

fn evaluate_predicate(
    expr: &SqlExpr,
    row: &InlineRow,
    columns: &[InlineColumn],
    alias: Option<&str>,
) -> SqlResult<bool> {
    match expr {
        SqlExpr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => Ok(evaluate_predicate(left, row, columns, alias)?
                && evaluate_predicate(right, row, columns, alias)?),
            BinaryOperator::Or => Ok(evaluate_predicate(left, row, columns, alias)?
                || evaluate_predicate(right, row, columns, alias)?),
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Gt
            | BinaryOperator::Lt
            | BinaryOperator::GtEq
            | BinaryOperator::LtEq => {
                let left_value = evaluate_scalar(left, row, columns, alias)?;
                let right_value = evaluate_scalar(right, row, columns, alias)?;
                compare_values(&left_value, &right_value, op)
            }
            _ => Err(Error::InvalidArgumentError(format!(
                "unsupported operator '{}' in inline predicate",
                op
            ))),
        },
        SqlExpr::IsNull(inner) => Ok(evaluate_scalar(inner, row, columns, alias)?.is_null()),
        SqlExpr::IsNotNull(inner) => Ok(!evaluate_scalar(inner, row, columns, alias)?.is_null()),
        SqlExpr::Nested(inner) => evaluate_predicate(inner, row, columns, alias),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Not,
            expr,
        } => Ok(!evaluate_predicate(expr, row, columns, alias)?),
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) | SqlExpr::Value(_) => {
            let scalar = evaluate_scalar(expr, row, columns, alias)?;
            scalar
                .as_bool()
                .ok_or_else(|| Error::InvalidArgumentError("expression is not boolean".into()))
        }
        _ => Err(Error::InvalidArgumentError(
            "unsupported predicate in inline query".into(),
        )),
    }
}

fn evaluate_scalar(
    expr: &SqlExpr,
    row: &InlineRow,
    columns: &[InlineColumn],
    alias: Option<&str>,
) -> SqlResult<InlineValue> {
    match expr {
        SqlExpr::Identifier(ident) => {
            let idx =
                column_index(columns, &ident.value).ok_or_else(|| invalid_column(&ident.value))?;
            Ok(row.values[idx].clone())
        }
        SqlExpr::CompoundIdentifier(parts) => {
            let column = resolve_identifier_from_parts(parts, alias).ok_or_else(|| {
                invalid_column(
                    &parts
                        .last()
                        .map(|ident| ident.value.clone())
                        .unwrap_or_else(|| "<unknown>".into()),
                )
            })?;
            let idx = column_index(columns, &column).ok_or_else(|| invalid_column(&column))?;
            Ok(row.values[idx].clone())
        }
        SqlExpr::Value(value) => literal_to_inline_value(value),
        SqlExpr::Nested(inner) => evaluate_scalar(inner, row, columns, alias),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => evaluate_scalar(expr, row, columns, alias),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => {
            let value = evaluate_scalar(expr, row, columns, alias)?;
            match value {
                InlineValue::Int32(Some(v)) => Ok(InlineValue::Int32(Some(-v))),
                InlineValue::Int32(None) => Ok(InlineValue::Int32(None)),
                InlineValue::Null => Ok(InlineValue::Null),
                _ => Err(Error::InvalidArgumentError(
                    "UNARY - is only supported for numeric expressions in inline queries".into(),
                )),
            }
        }
        _ => Err(Error::InvalidArgumentError(format!(
            "unsupported expression '{}' in inline query",
            expr
        ))),
    }
}

fn resolve_identifier_from_parts(parts: &[Ident], alias: Option<&str>) -> Option<String> {
    if parts.is_empty() {
        return None;
    }
    if parts.len() == 1 {
        return Some(parts[0].value.clone());
    }
    if parts.len() == 2
        && let Some(alias_name) = alias
        && alias_name.eq_ignore_ascii_case(&parts[0].value)
    {
        return Some(parts[1].value.clone());
    }
    parts.last().map(|ident| ident.value.clone())
}

fn column_index(columns: &[InlineColumn], name: &str) -> Option<usize> {
    columns
        .iter()
        .position(|column| column.name.eq_ignore_ascii_case(name))
}

fn compare_values(left: &InlineValue, right: &InlineValue, op: &BinaryOperator) -> SqlResult<bool> {
    if left.is_null() || right.is_null() {
        return Ok(false);
    }
    match op {
        BinaryOperator::Eq => match (left, right) {
            (InlineValue::String(Some(a)), InlineValue::String(Some(b))) => Ok(a == b),
            (InlineValue::Int32(Some(a)), InlineValue::Int32(Some(b))) => Ok(a == b),
            (InlineValue::Bool(Some(a)), InlineValue::Bool(Some(b))) => Ok(a == b),
            _ => Err(Error::InvalidArgumentError(
                "type mismatch in comparison".into(),
            )),
        },
        BinaryOperator::NotEq => {
            compare_values(left, right, &BinaryOperator::Eq).map(|result| !result)
        }
        BinaryOperator::Gt | BinaryOperator::Lt | BinaryOperator::GtEq | BinaryOperator::LtEq => {
            match (left, right) {
                (InlineValue::String(Some(a)), InlineValue::String(Some(b))) => match op {
                    BinaryOperator::Gt => Ok(a > b),
                    BinaryOperator::Lt => Ok(a < b),
                    BinaryOperator::GtEq => Ok(a >= b),
                    BinaryOperator::LtEq => Ok(a <= b),
                    _ => unreachable!(),
                },
                (InlineValue::Int32(Some(a)), InlineValue::Int32(Some(b))) => match op {
                    BinaryOperator::Gt => Ok(a > b),
                    BinaryOperator::Lt => Ok(a < b),
                    BinaryOperator::GtEq => Ok(a >= b),
                    BinaryOperator::LtEq => Ok(a <= b),
                    _ => unreachable!(),
                },
                _ => Err(Error::InvalidArgumentError(
                    "type mismatch in comparison".into(),
                )),
            }
        }
        _ => Err(Error::InvalidArgumentError(
            "unsupported comparison operator".into(),
        )),
    }
}

fn literal_to_inline_value(value: &ValueWithSpan) -> SqlResult<InlineValue> {
    if let Some(text) = value.clone().value.into_string() {
        return Ok(InlineValue::String(Some(text)));
    }

    match &value.value {
        sqlparser::ast::Value::Number(text, _) => {
            let parsed = text
                .parse::<i64>()
                .map_err(|_| Error::InvalidArgumentError("numeric literal is too large".into()))?;
            if parsed < i64::from(i32::MIN) || parsed > i64::from(i32::MAX) {
                return Err(Error::InvalidArgumentError(
                    "numeric literal is out of range".into(),
                ));
            }
            Ok(InlineValue::Int32(Some(parsed as i32)))
        }
        sqlparser::ast::Value::Boolean(flag) => Ok(InlineValue::Bool(Some(*flag))),
        sqlparser::ast::Value::Null => Ok(InlineValue::Null),
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported literal '{other}' in inline query"
        ))),
    }
}

fn invalid_column(name: &str) -> Error {
    Error::InvalidArgumentError(format!("column '{name}' does not exist"))
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
        SqlExpr::Nested(inner) => resolve_column_name(inner),
        // Handle unary +/- by recursively resolving the inner expression
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus | UnaryOperator::Minus,
            expr,
        } => resolve_column_name(expr),
        _ => Err(Error::InvalidArgumentError(
            "aggregate arguments must be plain column identifiers".into(),
        )),
    }
}

fn is_simple_aggregate_column(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => true,
        SqlExpr::Nested(inner) => is_simple_aggregate_column(inner),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => is_simple_aggregate_column(expr),
        _ => false,
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
        llkv_expr::expr::ScalarExpr::Compare { left, right, .. } => {
            expr_contains_aggregate(left) || expr_contains_aggregate(right)
        }
        llkv_expr::expr::ScalarExpr::Not(inner) => expr_contains_aggregate(inner),
        llkv_expr::expr::ScalarExpr::IsNull { expr, .. } => expr_contains_aggregate(expr),
        llkv_expr::expr::ScalarExpr::GetField { base, .. } => expr_contains_aggregate(base),
        llkv_expr::expr::ScalarExpr::Cast { expr, .. } => expr_contains_aggregate(expr),
        llkv_expr::expr::ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            operand
                .as_deref()
                .map(expr_contains_aggregate)
                .unwrap_or(false)
                || branches.iter().any(|(when_expr, then_expr)| {
                    expr_contains_aggregate(when_expr) || expr_contains_aggregate(then_expr)
                })
                || else_expr
                    .as_deref()
                    .map(expr_contains_aggregate)
                    .unwrap_or(false)
        }
        llkv_expr::expr::ScalarExpr::Coalesce(items) => items.iter().any(expr_contains_aggregate),
        llkv_expr::expr::ScalarExpr::Column(_)
        | llkv_expr::expr::ScalarExpr::Literal(_)
        | llkv_expr::expr::ScalarExpr::Random => false,
        llkv_expr::expr::ScalarExpr::ScalarSubquery(_) => false,
    }
}

fn try_parse_aggregate_function(
    func: &sqlparser::ast::Function,
    resolver: Option<&IdentifierResolver<'_>>,
    context: Option<&IdentifierContext>,
    outer_scopes: &[IdentifierContext],
    tracker: &mut SubqueryCorrelatedTracker<'_>,
) -> SqlResult<Option<llkv_expr::expr::AggregateCall<String>>> {
    use sqlparser::ast::{
        DuplicateTreatment, FunctionArg, FunctionArgExpr, FunctionArguments, ObjectNamePart,
    };

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

    // Check for DISTINCT modifier
    let distinct = match &func.args {
        FunctionArguments::List(list) => {
            if !list.clauses.is_empty() {
                return Ok(None);
            }
            matches!(list.duplicate_treatment, Some(DuplicateTreatment::Distinct))
        }
        _ => false,
    };

    let args_slice: &[FunctionArg] = match &func.args {
        FunctionArguments::List(list) => &list.args,
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
                    if distinct {
                        return Err(Error::InvalidArgumentError(
                            "COUNT(DISTINCT *) is not supported".into(),
                        ));
                    }
                    llkv_expr::expr::AggregateCall::CountStar
                }
                FunctionArg::Unnamed(FunctionArgExpr::Expr(arg_expr)) => {
                    let expr = translate_scalar_internal(
                        arg_expr,
                        resolver,
                        context,
                        outer_scopes,
                        tracker,
                        None,
                    )?;
                    llkv_expr::expr::AggregateCall::Count {
                        expr: Box::new(expr),
                        distinct,
                    }
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
                if distinct {
                    return Err(Error::InvalidArgumentError(
                        "DISTINCT not supported for COUNT(CASE ...) pattern".into(),
                    ));
                }
                llkv_expr::expr::AggregateCall::CountNulls(Box::new(
                    llkv_expr::expr::ScalarExpr::column(column),
                ))
            } else {
                let expr = translate_scalar_internal(
                    arg_expr,
                    resolver,
                    context,
                    outer_scopes,
                    tracker,
                    None,
                )?;
                llkv_expr::expr::AggregateCall::Sum {
                    expr: Box::new(expr),
                    distinct,
                }
            }
        }
        "total" => {
            if args_slice.len() != 1 {
                return Err(Error::InvalidArgumentError(
                    "TOTAL accepts exactly one argument".into(),
                ));
            }
            let arg_expr = match &args_slice[0] {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
                _ => {
                    return Err(Error::InvalidArgumentError(
                        "TOTAL requires a column argument".into(),
                    ));
                }
            };

            let expr = translate_scalar_internal(
                arg_expr,
                resolver,
                context,
                outer_scopes,
                tracker,
                None,
            )?;
            llkv_expr::expr::AggregateCall::Total {
                expr: Box::new(expr),
                distinct,
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
            let expr = translate_scalar_internal(
                arg_expr,
                resolver,
                context,
                outer_scopes,
                tracker,
                None,
            )?;
            llkv_expr::expr::AggregateCall::Min(Box::new(expr))
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
            let expr = translate_scalar_internal(
                arg_expr,
                resolver,
                context,
                outer_scopes,
                tracker,
                None,
            )?;
            llkv_expr::expr::AggregateCall::Max(Box::new(expr))
        }
        "avg" => {
            if args_slice.len() != 1 {
                return Err(Error::InvalidArgumentError(
                    "AVG accepts exactly one argument".into(),
                ));
            }
            let arg_expr = match &args_slice[0] {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
                _ => {
                    return Err(Error::InvalidArgumentError(
                        "AVG requires a column argument".into(),
                    ));
                }
            };
            let expr = translate_scalar_internal(
                arg_expr,
                resolver,
                context,
                outer_scopes,
                tracker,
                None,
            )?;
            llkv_expr::expr::AggregateCall::Avg {
                expr: Box::new(expr),
                distinct,
            }
        }
        "group_concat" => {
            if args_slice.is_empty() || args_slice.len() > 2 {
                return Err(Error::InvalidArgumentError(
                    "GROUP_CONCAT accepts one or two arguments".into(),
                ));
            }

            // First argument is the column/expression
            let arg_expr = match &args_slice[0] {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
                _ => {
                    return Err(Error::InvalidArgumentError(
                        "GROUP_CONCAT requires a column argument".into(),
                    ));
                }
            };

            let expr = translate_scalar_internal(
                arg_expr,
                resolver,
                context,
                outer_scopes,
                tracker,
                None,
            )?;

            // Second argument (optional) is the separator
            let separator = if args_slice.len() == 2 {
                match &args_slice[1] {
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Value(
                        ValueWithSpan {
                            value: sqlparser::ast::Value::SingleQuotedString(s),
                            ..
                        },
                    ))) => Some(s.clone()),
                    _ => {
                        return Err(Error::InvalidArgumentError(
                            "GROUP_CONCAT separator must be a string literal".into(),
                        ));
                    }
                }
            } else {
                None
            };

            // SQLite doesn't support DISTINCT with a custom separator
            if distinct && separator.is_some() {
                return Err(Error::InvalidArgumentError(
                    "GROUP_CONCAT does not support DISTINCT with a custom separator".into(),
                ));
            }

            llkv_expr::expr::AggregateCall::GroupConcat {
                expr: Box::new(expr),
                distinct,
                separator,
            }
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

fn strip_sql_expr_nesting(expr: &SqlExpr) -> &SqlExpr {
    match expr {
        SqlExpr::Nested(inner) => strip_sql_expr_nesting(inner),
        other => other,
    }
}

struct BetweenBounds<'a> {
    lower: &'a SqlExpr,
    upper: &'a SqlExpr,
}

fn translate_between_expr(
    resolver: &IdentifierResolver<'_>,
    context: IdentifierContext,
    between_expr: &SqlExpr,
    bounds: BetweenBounds<'_>,
    negated: bool,
    outer_scopes: &[IdentifierContext],
    mut correlated_tracker: Option<&mut SubqueryCorrelatedColumnTracker>,
) -> SqlResult<llkv_expr::expr::Expr<'static, String>> {
    let lower_op = if negated {
        BinaryOperator::Lt
    } else {
        BinaryOperator::GtEq
    };
    let upper_op = if negated {
        BinaryOperator::Gt
    } else {
        BinaryOperator::LtEq
    };

    let lower_bound = translate_comparison_with_context(
        resolver,
        context.clone(),
        between_expr,
        lower_op,
        bounds.lower,
        outer_scopes,
        correlated_tracker.reborrow(),
    )?;
    let upper_bound = translate_comparison_with_context(
        resolver,
        context,
        between_expr,
        upper_op,
        bounds.upper,
        outer_scopes,
        correlated_tracker,
    )?;

    if negated {
        Ok(llkv_expr::expr::Expr::Or(vec![lower_bound, upper_bound]))
    } else {
        Ok(llkv_expr::expr::Expr::And(vec![lower_bound, upper_bound]))
    }
}

fn correlated_scalar_from_resolution(
    placeholder: String,
    resolution: &ColumnResolution,
) -> llkv_expr::expr::ScalarExpr<String> {
    let mut expr = llkv_expr::expr::ScalarExpr::column(placeholder);
    for field in resolution.field_path() {
        expr = llkv_expr::expr::ScalarExpr::get_field(expr, field.clone());
    }
    expr
}

fn resolve_correlated_identifier(
    resolver: &IdentifierResolver<'_>,
    parts: &[String],
    outer_scopes: &[IdentifierContext],
    mut tracker: SubqueryCorrelatedTracker<'_>,
) -> SqlResult<Option<llkv_expr::expr::ScalarExpr<String>>> {
    if !tracker.is_active() {
        return Ok(None);
    }

    for scope in outer_scopes.iter().rev() {
        match resolver.resolve(parts, scope.clone()) {
            Ok(resolution) => {
                if let Some(placeholder) = tracker.placeholder_for_resolution(&resolution) {
                    let expr = correlated_scalar_from_resolution(placeholder, &resolution);
                    return Ok(Some(expr));
                }
            }
            Err(_) => continue,
        }
    }

    Ok(None)
}

fn resolve_identifier_expr(
    resolver: &IdentifierResolver<'_>,
    context: &IdentifierContext,
    parts: Vec<String>,
    outer_scopes: &[IdentifierContext],
    tracker: SubqueryCorrelatedTracker<'_>,
) -> SqlResult<llkv_expr::expr::ScalarExpr<String>> {
    match resolver.resolve(&parts, context.clone()) {
        Ok(resolution) => Ok(resolution.into_scalar_expr()),
        Err(err) => {
            if let Some(expr) =
                resolve_correlated_identifier(resolver, &parts, outer_scopes, tracker)?
            {
                Ok(expr)
            } else {
                Err(err)
            }
        }
    }
}

fn translate_condition_with_context(
    engine: &SqlEngine,
    resolver: &IdentifierResolver<'_>,
    context: IdentifierContext,
    expr: &SqlExpr,
    outer_scopes: &[IdentifierContext],
    subqueries: &mut Vec<llkv_plan::FilterSubquery>,
    mut correlated_tracker: Option<&mut SubqueryCorrelatedColumnTracker>,
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
                            outer_scopes,
                            correlated_tracker.reborrow(),
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
                    let inner_stripped = strip_sql_expr_nesting(inner);
                    if let SqlExpr::Between {
                        expr: between_expr,
                        negated: inner_negated,
                        low,
                        high,
                    } = inner_stripped
                    {
                        let negated_mode = !*inner_negated;
                        let between_expr_result = translate_between_expr(
                            resolver,
                            context.clone(),
                            between_expr,
                            BetweenBounds {
                                lower: low,
                                upper: high,
                            },
                            negated_mode,
                            outer_scopes,
                            correlated_tracker.reborrow(),
                        )?;
                        work_stack.push(ConditionFrame::Leaf(between_expr_result));
                        continue;
                    }
                    // Note: Do not short-circuit NOT on NULL comparisons.
                    // NULL comparisons evaluate to NULL, and NOT NULL should also be NULL,
                    // not FALSE. Let the normal evaluation handle NULL propagation.
                    work_stack.push(ConditionFrame::Exit(ConditionExitContext::Not));
                    work_stack.push(ConditionFrame::Enter(inner));
                }
                SqlExpr::Nested(inner) => {
                    work_stack.push(ConditionFrame::Exit(ConditionExitContext::Nested));
                    work_stack.push(ConditionFrame::Enter(inner));
                }
                SqlExpr::IsNull(inner) => {
                    let scalar = translate_scalar_with_context_scoped(
                        resolver,
                        context.clone(),
                        inner,
                        outer_scopes,
                        correlated_tracker.reborrow(),
                    )?;
                    match scalar {
                        llkv_expr::expr::ScalarExpr::Column(column) => {
                            // Optimize simple column checks to use Filter
                            work_stack.push(ConditionFrame::Leaf(llkv_expr::expr::Expr::Pred(
                                llkv_expr::expr::Filter {
                                    field_id: column,
                                    op: llkv_expr::expr::Operator::IsNull,
                                },
                            )));
                        }
                        // NOTE: Do NOT constant-fold IsNull(Literal(Null)) to Literal(true).
                        // While technically correct (NULL IS NULL = TRUE), it breaks NULL
                        // propagation in boolean expressions like NOT (NOT NULL = NULL).
                        // The executor's evaluate_having_expr handles these correctly.
                        other => {
                            // For all expressions including literals, use the IsNull variant
                            work_stack.push(ConditionFrame::Leaf(llkv_expr::expr::Expr::IsNull {
                                expr: other,
                                negated: false,
                            }));
                        }
                    }
                }
                SqlExpr::IsNotNull(inner) => {
                    let scalar = translate_scalar_with_context_scoped(
                        resolver,
                        context.clone(),
                        inner,
                        outer_scopes,
                        correlated_tracker.reborrow(),
                    )?;
                    match scalar {
                        llkv_expr::expr::ScalarExpr::Column(column) => {
                            // Optimize simple column checks to use Filter
                            work_stack.push(ConditionFrame::Leaf(llkv_expr::expr::Expr::Pred(
                                llkv_expr::expr::Filter {
                                    field_id: column,
                                    op: llkv_expr::expr::Operator::IsNotNull,
                                },
                            )));
                        }
                        // NOTE: Do NOT constant-fold IsNotNull(Literal(Null)) to Literal(false).
                        // While technically correct (NULL IS NOT NULL = FALSE), it breaks NULL
                        // propagation in boolean expressions like NOT (NOT NULL = NULL).
                        // The executor's evaluate_having_expr handles these correctly.
                        other => {
                            // For all expressions including literals, use the IsNull variant with negation
                            work_stack.push(ConditionFrame::Leaf(llkv_expr::expr::Expr::IsNull {
                                expr: other,
                                negated: true,
                            }));
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
                        let target = translate_scalar_with_context_scoped(
                            resolver,
                            context.clone(),
                            in_expr,
                            outer_scopes,
                            correlated_tracker.reborrow(),
                        )?;
                        let mut values = Vec::with_capacity(list.len());
                        for value_expr in list {
                            let scalar = translate_scalar_with_context_scoped(
                                resolver,
                                context.clone(),
                                value_expr,
                                outer_scopes,
                                correlated_tracker.reborrow(),
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
                    let between_expr_result = translate_between_expr(
                        resolver,
                        context.clone(),
                        between_expr,
                        BetweenBounds {
                            lower: low,
                            upper: high,
                        },
                        *negated,
                        outer_scopes,
                        correlated_tracker.reborrow(),
                    )?;
                    work_stack.push(ConditionFrame::Leaf(between_expr_result));
                }
                SqlExpr::Exists { subquery, negated } => {
                    // Build nested select plan for the subquery
                    let mut nested_scopes = outer_scopes.to_vec();
                    nested_scopes.push(context.clone());

                    let mut tracker = SubqueryCorrelatedColumnTracker::new();
                    let mut nested_subqueries = Vec::new();

                    // Translate the subquery in an extended scope
                    let subquery_plan = engine.build_select_plan_internal(
                        (**subquery).clone(),
                        resolver,
                        &nested_scopes,
                        &mut nested_subqueries,
                        Some(&mut tracker),
                    )?;

                    let subquery_id = llkv_expr::SubqueryId(subqueries.len() as u32);
                    let filter_subquery = llkv_plan::FilterSubquery {
                        id: subquery_id,
                        plan: Box::new(subquery_plan),
                        correlated_columns: tracker.into_columns(),
                    };
                    subqueries.push(filter_subquery);

                    work_stack.push(ConditionFrame::Leaf(llkv_expr::expr::Expr::Exists(
                        llkv_expr::SubqueryExpr {
                            id: subquery_id,
                            negated: *negated,
                        },
                    )));
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
                    // Optimize: NOT (expr IS NULL) -> expr IS NOT NULL by flipping negation
                    match inner {
                        llkv_expr::expr::Expr::IsNull { expr, negated } => {
                            result_stack.push(llkv_expr::expr::Expr::IsNull {
                                expr,
                                negated: !negated,
                            });
                        }
                        other => {
                            result_stack.push(llkv_expr::expr::Expr::not(other));
                        }
                    }
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

fn peel_unparenthesized_not_chain(expr: &SqlExpr) -> (usize, &SqlExpr) {
    let mut count: usize = 0;
    let mut current = expr;
    while let SqlExpr::UnaryOp {
        op: UnaryOperator::Not,
        expr: inner,
    } = current
    {
        if matches!(inner.as_ref(), SqlExpr::Nested(_)) {
            break;
        }
        count += 1;
        current = inner.as_ref();
    }
    (count, current)
}

fn translate_comparison_with_context(
    resolver: &IdentifierResolver<'_>,
    context: IdentifierContext,
    left: &SqlExpr,
    op: BinaryOperator,
    right: &SqlExpr,
    outer_scopes: &[IdentifierContext],
    mut correlated_tracker: Option<&mut SubqueryCorrelatedColumnTracker>,
) -> SqlResult<llkv_expr::expr::Expr<'static, String>> {
    let (not_count, comparison_left) = peel_unparenthesized_not_chain(left);

    let left_scalar = {
        let tracker = correlated_tracker.reborrow();
        translate_scalar_with_context_scoped(
            resolver,
            context.clone(),
            comparison_left,
            outer_scopes,
            tracker,
        )?
    };
    let right_scalar = {
        let tracker = correlated_tracker.reborrow();
        translate_scalar_with_context_scoped(resolver, context, right, outer_scopes, tracker)?
    };
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

    let mut expr = llkv_expr::expr::Expr::Compare {
        left: left_scalar.clone(),
        op: compare_op,
        right: right_scalar.clone(),
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
        expr = llkv_expr::expr::Expr::Pred(llkv_expr::expr::Filter {
            field_id: column.clone(),
            op,
        });
    } else if let (
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
        expr = llkv_expr::expr::Expr::Pred(llkv_expr::expr::Filter {
            field_id: column.clone(),
            op,
        });
    }

    let mut wrapped = expr;
    for _ in 0..not_count {
        wrapped = llkv_expr::expr::Expr::Not(Box::new(wrapped));
    }

    Ok(wrapped)
}

fn compare_op_to_filter_operator(
    op: llkv_expr::expr::CompareOp,
    literal: &Literal,
) -> Option<llkv_expr::expr::Operator<'static>> {
    if matches!(literal, Literal::Null) {
        return None;
    }
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
    let mut tracker = SubqueryCorrelatedTracker::from_option(None);
    translate_scalar_internal(
        expr,
        Some(resolver),
        Some(&context),
        &[],
        &mut tracker,
        None,
    )
}

fn translate_scalar_with_context_scoped(
    resolver: &IdentifierResolver<'_>,
    context: IdentifierContext,
    expr: &SqlExpr,
    outer_scopes: &[IdentifierContext],
    correlated_tracker: Option<&mut SubqueryCorrelatedColumnTracker>,
) -> SqlResult<llkv_expr::expr::ScalarExpr<String>> {
    let mut tracker = SubqueryCorrelatedTracker::from_option(correlated_tracker);
    translate_scalar_internal(
        expr,
        Some(resolver),
        Some(&context),
        outer_scopes,
        &mut tracker,
        None,
    )
}

#[allow(dead_code)]
fn translate_scalar(expr: &SqlExpr) -> SqlResult<llkv_expr::expr::ScalarExpr<String>> {
    let mut tracker = SubqueryCorrelatedTracker::from_option(None);
    translate_scalar_internal(expr, None, None, &[], &mut tracker, None)
}

fn translate_scalar_internal(
    expr: &SqlExpr,
    resolver: Option<&IdentifierResolver<'_>>,
    context: Option<&IdentifierContext>,
    outer_scopes: &[IdentifierContext],
    tracker: &mut SubqueryCorrelatedTracker<'_>,
    mut subquery_resolver: Option<&mut dyn ScalarSubqueryResolver>,
) -> SqlResult<llkv_expr::expr::ScalarExpr<String>> {
    // Iterative postorder traversal using the TransformFrame pattern.
    // See llkv-plan::traversal module documentation for pattern details.
    //
    // This avoids stack overflow on deeply nested expressions (50k+ nodes) by using
    // explicit work_stack and result_stack instead of recursion.

    /// Context passed through Exit frames during scalar expression translation
    enum ScalarExitContext {
        BinaryOp {
            op: BinaryOperator,
        },
        Compare {
            op: llkv_expr::expr::CompareOp,
        },
        UnaryNot,
        UnaryMinus,
        UnaryPlus,
        Nested,
        Cast(DataType),
        IsNull {
            negated: bool,
        },
        Between {
            negated: bool,
        },
        InList {
            list_len: usize,
            negated: bool,
        },
        Case {
            branch_count: usize,
            has_operand: bool,
            has_else: bool,
        },
        BuiltinFunction {
            func: BuiltinScalarFunction,
            arg_count: usize,
        },
    }

    #[derive(Clone, Copy)]
    enum BuiltinScalarFunction {
        Abs,
        Coalesce,
        NullIf,
        Floor,
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
                        let tracker_view = tracker.reborrow();
                        let expr = resolve_identifier_expr(
                            resolver,
                            ctx,
                            parts,
                            outer_scopes,
                            tracker_view,
                        )?;
                        work_stack.push(ScalarFrame::Leaf(expr));
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
                        let tracker_view = tracker.reborrow();
                        let expr = resolve_identifier_expr(
                            resolver,
                            ctx,
                            parts,
                            outer_scopes,
                            tracker_view,
                        )?;
                        work_stack.push(ScalarFrame::Leaf(expr));
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
                SqlExpr::Interval(interval) => {
                    let parsed = parse_interval_literal(interval)?;
                    let literal = llkv_expr::expr::ScalarExpr::literal(Literal::Interval(parsed));
                    work_stack.push(ScalarFrame::Leaf(literal));
                }
                SqlExpr::BinaryOp { left, op, right } => match op {
                    BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
                    | BinaryOperator::Modulo
                    | BinaryOperator::And
                    | BinaryOperator::Or
                    | BinaryOperator::PGBitwiseShiftLeft
                    | BinaryOperator::PGBitwiseShiftRight => {
                        work_stack.push(ScalarFrame::Exit(ScalarExitContext::BinaryOp {
                            op: op.clone(),
                        }));
                        work_stack.push(ScalarFrame::Enter(right));
                        work_stack.push(ScalarFrame::Enter(left));
                    }
                    BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Lt
                    | BinaryOperator::LtEq
                    | BinaryOperator::Gt
                    | BinaryOperator::GtEq => {
                        let compare_op = match op {
                            BinaryOperator::Eq => llkv_expr::expr::CompareOp::Eq,
                            BinaryOperator::NotEq => llkv_expr::expr::CompareOp::NotEq,
                            BinaryOperator::Lt => llkv_expr::expr::CompareOp::Lt,
                            BinaryOperator::LtEq => llkv_expr::expr::CompareOp::LtEq,
                            BinaryOperator::Gt => llkv_expr::expr::CompareOp::Gt,
                            BinaryOperator::GtEq => llkv_expr::expr::CompareOp::GtEq,
                            _ => unreachable!(),
                        };
                        work_stack.push(ScalarFrame::Exit(ScalarExitContext::Compare {
                            op: compare_op,
                        }));
                        work_stack.push(ScalarFrame::Enter(right));
                        work_stack.push(ScalarFrame::Enter(left));
                    }
                    other => {
                        return Err(Error::InvalidArgumentError(format!(
                            "unsupported scalar binary operator: {other:?}"
                        )));
                    }
                },
                SqlExpr::UnaryOp {
                    op: UnaryOperator::Not,
                    expr: inner,
                } => {
                    work_stack.push(ScalarFrame::Exit(ScalarExitContext::UnaryNot));
                    work_stack.push(ScalarFrame::Enter(inner));
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
                SqlExpr::Cast {
                    expr: inner,
                    data_type,
                    ..
                } => {
                    let target_type = arrow_type_from_sql(data_type)?;
                    work_stack.push(ScalarFrame::Exit(ScalarExitContext::Cast(target_type)));
                    work_stack.push(ScalarFrame::Enter(inner));
                }
                SqlExpr::Case {
                    operand,
                    conditions,
                    else_result,
                    ..
                } => {
                    work_stack.push(ScalarFrame::Exit(ScalarExitContext::Case {
                        branch_count: conditions.len(),
                        has_operand: operand.is_some(),
                        has_else: else_result.is_some(),
                    }));
                    if let Some(else_expr) = else_result.as_deref() {
                        work_stack.push(ScalarFrame::Enter(else_expr));
                    }
                    for case_when in conditions.iter().rev() {
                        work_stack.push(ScalarFrame::Enter(&case_when.result));
                        work_stack.push(ScalarFrame::Enter(&case_when.condition));
                    }
                    if let Some(opnd) = operand.as_deref() {
                        work_stack.push(ScalarFrame::Enter(opnd));
                    }
                }
                SqlExpr::InList {
                    expr: in_expr,
                    list,
                    negated,
                } => {
                    if list.is_empty() {
                        let literal_value = if *negated {
                            llkv_expr::expr::ScalarExpr::literal(Literal::Integer(1))
                        } else {
                            llkv_expr::expr::ScalarExpr::literal(Literal::Integer(0))
                        };
                        work_stack.push(ScalarFrame::Leaf(literal_value));
                    } else {
                        work_stack.push(ScalarFrame::Exit(ScalarExitContext::InList {
                            list_len: list.len(),
                            negated: *negated,
                        }));
                        for value_expr in list.iter().rev() {
                            work_stack.push(ScalarFrame::Enter(value_expr));
                        }
                        work_stack.push(ScalarFrame::Enter(in_expr));
                    }
                }
                SqlExpr::IsNull(inner) => {
                    work_stack.push(ScalarFrame::Exit(ScalarExitContext::IsNull {
                        negated: false,
                    }));
                    work_stack.push(ScalarFrame::Enter(inner));
                }
                SqlExpr::IsNotNull(inner) => {
                    work_stack.push(ScalarFrame::Exit(ScalarExitContext::IsNull {
                        negated: true,
                    }));
                    work_stack.push(ScalarFrame::Enter(inner));
                }
                SqlExpr::Between {
                    expr: between_expr,
                    negated,
                    low,
                    high,
                } => {
                    work_stack.push(ScalarFrame::Exit(ScalarExitContext::Between {
                        negated: *negated,
                    }));
                    work_stack.push(ScalarFrame::Enter(high));
                    work_stack.push(ScalarFrame::Enter(low));
                    work_stack.push(ScalarFrame::Enter(between_expr));
                }
                SqlExpr::Function(func) => {
                    if let Some(agg_call) = try_parse_aggregate_function(
                        func,
                        resolver,
                        context,
                        outer_scopes,
                        tracker,
                    )? {
                        work_stack.push(ScalarFrame::Leaf(llkv_expr::expr::ScalarExpr::aggregate(
                            agg_call,
                        )));
                    } else {
                        use sqlparser::ast::{
                            FunctionArg, FunctionArgExpr, FunctionArguments, ObjectNamePart,
                        };

                        if func.uses_odbc_syntax
                            || !matches!(func.parameters, FunctionArguments::None)
                            || func.filter.is_some()
                            || func.null_treatment.is_some()
                            || func.over.is_some()
                            || !func.within_group.is_empty()
                        {
                            return Err(Error::InvalidArgumentError(format!(
                                "unsupported function in scalar expression: {:?}",
                                func.name
                            )));
                        }

                        let func_name = if func.name.0.len() == 1 {
                            match &func.name.0[0] {
                                ObjectNamePart::Identifier(ident) => {
                                    ident.value.to_ascii_lowercase()
                                }
                                _ => {
                                    return Err(Error::InvalidArgumentError(format!(
                                        "unsupported function in scalar expression: {:?}",
                                        func.name
                                    )));
                                }
                            }
                        } else {
                            return Err(Error::InvalidArgumentError(format!(
                                "unsupported function in scalar expression: {:?}",
                                func.name
                            )));
                        };

                        match func_name.as_str() {
                            "abs" => {
                                let args_slice: &[FunctionArg] = match &func.args {
                                    FunctionArguments::List(list) => {
                                        if list.duplicate_treatment.is_some()
                                            || !list.clauses.is_empty()
                                        {
                                            return Err(Error::InvalidArgumentError(
                                                "ABS does not support qualifiers".into(),
                                            ));
                                        }
                                        &list.args
                                    }
                                    _ => {
                                        return Err(Error::InvalidArgumentError(
                                            "ABS requires exactly one argument".into(),
                                        ));
                                    }
                                };

                                if args_slice.len() != 1 {
                                    return Err(Error::InvalidArgumentError(
                                        "ABS requires exactly one argument".into(),
                                    ));
                                }

                                let arg_expr = match &args_slice[0] {
                                    FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
                                    _ => {
                                        return Err(Error::InvalidArgumentError(
                                            "ABS argument must be an expression".into(),
                                        ));
                                    }
                                };

                                work_stack.push(ScalarFrame::Exit(
                                    ScalarExitContext::BuiltinFunction {
                                        func: BuiltinScalarFunction::Abs,
                                        arg_count: 1,
                                    },
                                ));
                                work_stack.push(ScalarFrame::Enter(arg_expr));
                                continue;
                            }
                            "floor" => {
                                let args_slice: &[FunctionArg] = match &func.args {
                                    FunctionArguments::List(list) => {
                                        if list.duplicate_treatment.is_some()
                                            || !list.clauses.is_empty()
                                        {
                                            return Err(Error::InvalidArgumentError(
                                                "FLOOR does not support qualifiers".into(),
                                            ));
                                        }
                                        &list.args
                                    }
                                    _ => {
                                        return Err(Error::InvalidArgumentError(
                                            "FLOOR requires exactly one argument".into(),
                                        ));
                                    }
                                };

                                if args_slice.len() != 1 {
                                    return Err(Error::InvalidArgumentError(
                                        "FLOOR requires exactly one argument".into(),
                                    ));
                                }

                                let arg_expr = match &args_slice[0] {
                                    FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
                                    _ => {
                                        return Err(Error::InvalidArgumentError(
                                            "FLOOR argument must be an expression".into(),
                                        ));
                                    }
                                };

                                work_stack.push(ScalarFrame::Exit(
                                    ScalarExitContext::BuiltinFunction {
                                        func: BuiltinScalarFunction::Floor,
                                        arg_count: 1,
                                    },
                                ));
                                work_stack.push(ScalarFrame::Enter(arg_expr));
                                continue;
                            }
                            "coalesce" => {
                                let args_slice: &[FunctionArg] = match &func.args {
                                    FunctionArguments::List(list) => {
                                        if list.duplicate_treatment.is_some()
                                            || !list.clauses.is_empty()
                                        {
                                            return Err(Error::InvalidArgumentError(
                                                "COALESCE does not support qualifiers".into(),
                                            ));
                                        }
                                        &list.args
                                    }
                                    _ => {
                                        return Err(Error::InvalidArgumentError(
                                            "COALESCE requires at least one argument".into(),
                                        ));
                                    }
                                };

                                if args_slice.is_empty() {
                                    return Err(Error::InvalidArgumentError(
                                        "COALESCE requires at least one argument".into(),
                                    ));
                                }

                                work_stack.push(ScalarFrame::Exit(
                                    ScalarExitContext::BuiltinFunction {
                                        func: BuiltinScalarFunction::Coalesce,
                                        arg_count: args_slice.len(),
                                    },
                                ));

                                for arg in args_slice.iter().rev() {
                                    let arg_expr = match arg {
                                        FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
                                        _ => {
                                            return Err(Error::InvalidArgumentError(
                                                "COALESCE arguments must be expressions".into(),
                                            ));
                                        }
                                    };
                                    work_stack.push(ScalarFrame::Enter(arg_expr));
                                }
                                continue;
                            }
                            "nullif" => {
                                let args_slice: &[FunctionArg] = match &func.args {
                                    FunctionArguments::List(list) => {
                                        if list.duplicate_treatment.is_some()
                                            || !list.clauses.is_empty()
                                        {
                                            return Err(Error::InvalidArgumentError(
                                                "NULLIF does not support qualifiers".into(),
                                            ));
                                        }
                                        &list.args
                                    }
                                    _ => {
                                        return Err(Error::InvalidArgumentError(
                                            "NULLIF requires exactly two arguments".into(),
                                        ));
                                    }
                                };

                                if args_slice.len() != 2 {
                                    return Err(Error::InvalidArgumentError(
                                        "NULLIF requires exactly two arguments".into(),
                                    ));
                                }

                                work_stack.push(ScalarFrame::Exit(
                                    ScalarExitContext::BuiltinFunction {
                                        func: BuiltinScalarFunction::NullIf,
                                        arg_count: 2,
                                    },
                                ));

                                for arg in args_slice.iter().rev() {
                                    let arg_expr = match arg {
                                        FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
                                        _ => {
                                            return Err(Error::InvalidArgumentError(
                                                "NULLIF arguments must be expressions".into(),
                                            ));
                                        }
                                    };
                                    work_stack.push(ScalarFrame::Enter(arg_expr));
                                }
                                continue;
                            }
                            "random" | "rand" => {
                                let args_slice: &[FunctionArg] = match &func.args {
                                    FunctionArguments::List(list) => {
                                        if list.duplicate_treatment.is_some()
                                            || !list.clauses.is_empty()
                                        {
                                            return Err(Error::InvalidArgumentError(
                                                "RANDOM does not support qualifiers".into(),
                                            ));
                                        }
                                        &list.args
                                    }
                                    FunctionArguments::None => &[],
                                    _ => {
                                        return Err(Error::InvalidArgumentError(
                                            "RANDOM does not accept arguments".into(),
                                        ));
                                    }
                                };

                                if !args_slice.is_empty() {
                                    return Err(Error::InvalidArgumentError(
                                        "RANDOM does not accept arguments".into(),
                                    ));
                                }

                                work_stack
                                    .push(ScalarFrame::Leaf(llkv_expr::expr::ScalarExpr::random()));
                                continue;
                            }
                            _ => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "unsupported function in scalar expression: {:?}",
                                    func.name
                                )));
                            }
                        }
                    }
                }
                SqlExpr::Dictionary(fields) => {
                    // Process dictionary fields iteratively to avoid recursion
                    let mut struct_fields = Vec::new();
                    for entry in fields {
                        let key = entry.key.value.clone();
                        // Reuse scalar translation for nested values while honoring identifier context.
                        // Dictionaries rarely nest deeply, so recursion here is acceptable.
                        let mut tracker_view = tracker.reborrow();
                        let value_expr = translate_scalar_internal(
                            &entry.value,
                            resolver,
                            context,
                            outer_scopes,
                            &mut tracker_view,
                            None,
                        )?;
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
                SqlExpr::Subquery(subquery) => {
                    let handler = subquery_resolver.as_mut().ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "Correlated scalar subqueries not yet fully implemented - requires plan-level support".
                                to_string(),
                        )
                    })?;
                    let resolver_ref = resolver.ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "scalar subquery translation requires identifier resolver".into(),
                        )
                    })?;
                    let context_ref = context.ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "scalar subquery translation requires identifier context".into(),
                        )
                    })?;
                    let translated = handler.handle_scalar_subquery(
                        subquery.as_ref(),
                        resolver_ref,
                        context_ref,
                        outer_scopes,
                    )?;
                    work_stack.push(ScalarFrame::Leaf(translated));
                }
                SqlExpr::Floor { expr, field } => {
                    // sqlparser treats FLOOR as datetime extraction when field is specified
                    // We only support simple FLOOR (NoDateTime field)
                    use sqlparser::ast::{CeilFloorKind, DateTimeField};
                    if !matches!(
                        field,
                        CeilFloorKind::DateTimeField(DateTimeField::NoDateTime)
                    ) {
                        return Err(Error::InvalidArgumentError(format!(
                            "FLOOR with datetime field or scale not supported: {field:?}"
                        )));
                    }

                    // Treat FLOOR as a unary function: CAST(expr AS INT64)
                    // Note: CAST truncates towards zero, not true floor (towards -infinity)
                    work_stack.push(ScalarFrame::Exit(ScalarExitContext::Cast(DataType::Int64)));
                    work_stack.push(ScalarFrame::Enter(expr.as_ref()));
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
                    match op {
                        BinaryOperator::Plus => {
                            let expr = llkv_expr::expr::ScalarExpr::binary(
                                left_expr,
                                llkv_expr::expr::BinaryOp::Add,
                                right_expr,
                            );
                            result_stack.push(expr);
                        }
                        BinaryOperator::Minus => {
                            let expr = llkv_expr::expr::ScalarExpr::binary(
                                left_expr,
                                llkv_expr::expr::BinaryOp::Subtract,
                                right_expr,
                            );
                            result_stack.push(expr);
                        }
                        BinaryOperator::Multiply => {
                            let expr = llkv_expr::expr::ScalarExpr::binary(
                                left_expr,
                                llkv_expr::expr::BinaryOp::Multiply,
                                right_expr,
                            );
                            result_stack.push(expr);
                        }
                        BinaryOperator::Divide => {
                            let expr = llkv_expr::expr::ScalarExpr::binary(
                                left_expr,
                                llkv_expr::expr::BinaryOp::Divide,
                                right_expr,
                            );
                            result_stack.push(expr);
                        }
                        BinaryOperator::Modulo => {
                            let expr = llkv_expr::expr::ScalarExpr::binary(
                                left_expr,
                                llkv_expr::expr::BinaryOp::Modulo,
                                right_expr,
                            );
                            result_stack.push(expr);
                        }
                        BinaryOperator::And => {
                            let expr = llkv_expr::expr::ScalarExpr::binary(
                                left_expr,
                                llkv_expr::expr::BinaryOp::And,
                                right_expr,
                            );
                            result_stack.push(expr);
                        }
                        BinaryOperator::Or => {
                            let expr = llkv_expr::expr::ScalarExpr::binary(
                                left_expr,
                                llkv_expr::expr::BinaryOp::Or,
                                right_expr,
                            );
                            result_stack.push(expr);
                        }
                        BinaryOperator::PGBitwiseShiftLeft => {
                            let expr = llkv_expr::expr::ScalarExpr::binary(
                                left_expr,
                                llkv_expr::expr::BinaryOp::BitwiseShiftLeft,
                                right_expr,
                            );
                            result_stack.push(expr);
                        }
                        BinaryOperator::PGBitwiseShiftRight => {
                            let expr = llkv_expr::expr::ScalarExpr::binary(
                                left_expr,
                                llkv_expr::expr::BinaryOp::BitwiseShiftRight,
                                right_expr,
                            );
                            result_stack.push(expr);
                        }
                        other => {
                            return Err(Error::InvalidArgumentError(format!(
                                "unsupported scalar binary operator: {other:?}"
                            )));
                        }
                    }
                }
                ScalarExitContext::Compare { op } => {
                    let right_expr = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_scalar: result stack underflow for Compare right".into(),
                        )
                    })?;
                    let left_expr = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_scalar: result stack underflow for Compare left".into(),
                        )
                    })?;
                    result_stack.push(llkv_expr::expr::ScalarExpr::compare(
                        left_expr, op, right_expr,
                    ));
                }
                ScalarExitContext::BuiltinFunction { func, arg_count } => {
                    if result_stack.len() < arg_count {
                        return Err(Error::Internal(
                            "translate_scalar: result stack underflow for builtin function".into(),
                        ));
                    }

                    let mut args: Vec<llkv_expr::expr::ScalarExpr<String>> =
                        Vec::with_capacity(arg_count);
                    for _ in 0..arg_count {
                        if let Some(expr) = result_stack.pop() {
                            args.push(expr);
                        }
                    }
                    args.reverse();

                    let result_expr = match func {
                        BuiltinScalarFunction::Abs => {
                            debug_assert_eq!(args.len(), 1);
                            build_abs_case_expr(args.pop().expect("ABS expects one argument"))
                        }
                        BuiltinScalarFunction::Coalesce => {
                            llkv_expr::expr::ScalarExpr::coalesce(args)
                        }
                        BuiltinScalarFunction::NullIf => {
                            debug_assert_eq!(args.len(), 2);
                            let left = args.remove(0);
                            let right = args.remove(0);
                            let condition = llkv_expr::expr::ScalarExpr::compare(
                                left.clone(),
                                llkv_expr::expr::CompareOp::Eq,
                                right,
                            );
                            llkv_expr::expr::ScalarExpr::Case {
                                operand: None,
                                branches: vec![(
                                    condition,
                                    llkv_expr::expr::ScalarExpr::literal(Literal::Null),
                                )],
                                else_expr: Some(Box::new(left)),
                            }
                        }
                        BuiltinScalarFunction::Floor => {
                            debug_assert_eq!(args.len(), 1);
                            let arg = args.pop().expect("FLOOR expects one argument");
                            // Implement FLOOR as CAST to INT64 which truncates towards zero
                            // Note: This is not mathematically correct for negative numbers
                            // (should round towards negative infinity), but works for positive values
                            llkv_expr::expr::ScalarExpr::cast(arg, DataType::Int64)
                        }
                    };

                    result_stack.push(result_expr);
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
                            Literal::Date32(_) => {
                                return Err(Error::InvalidArgumentError(
                                    "cannot negate date literal".into(),
                                ));
                            }
                            Literal::Interval(interval) => {
                                let negated = interval.checked_neg().ok_or_else(|| {
                                    Error::InvalidArgumentError("interval overflow".into())
                                })?;
                                result_stack.push(llkv_expr::expr::ScalarExpr::literal(
                                    Literal::Interval(negated),
                                ));
                            }
                            Literal::Null => {
                                result_stack
                                    .push(llkv_expr::expr::ScalarExpr::literal(Literal::Null));
                            }
                            Literal::Decimal(value) => {
                                let negated_raw =
                                    value.raw_value().checked_neg().ok_or_else(|| {
                                        Error::InvalidArgumentError(
                                            "decimal overflow when applying unary minus".into(),
                                        )
                                    })?;
                                let negated = DecimalValue::new(negated_raw, value.scale())
                                    .map_err(|err| {
                                        Error::InvalidArgumentError(format!(
                                            "failed to negate decimal literal: {err}"
                                        ))
                                    })?;
                                result_stack.push(llkv_expr::expr::ScalarExpr::literal(
                                    Literal::Decimal(negated),
                                ));
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
                ScalarExitContext::UnaryNot => {
                    let inner = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_scalar: result stack underflow for UnaryNot".into(),
                        )
                    })?;
                    result_stack.push(llkv_expr::expr::ScalarExpr::logical_not(inner));
                }
                ScalarExitContext::UnaryPlus => {
                    // Unary plus is an identity operation in SQL - it returns the value unchanged.
                    // Unlike unary minus, it does NOT force numeric conversion; it's purely syntactic.
                    // SQLite treats `+col` identically to `col` in all contexts.
                    let inner = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_scalar: result stack underflow for UnaryPlus".into(),
                        )
                    })?;
                    result_stack.push(inner);
                }
                ScalarExitContext::Nested => {
                    // Nested is a no-op - just pass through
                }
                ScalarExitContext::Cast(target_type) => {
                    let inner = result_stack.pop().ok_or_else(|| {
                        Error::Internal("translate_scalar: result stack underflow for CAST".into())
                    })?;
                    result_stack.push(llkv_expr::expr::ScalarExpr::cast(inner, target_type));
                }
                ScalarExitContext::InList { list_len, negated } => {
                    let mut list_exprs = Vec::with_capacity(list_len);
                    for _ in 0..list_len {
                        let value_expr = result_stack.pop().ok_or_else(|| {
                            Error::Internal(
                                "translate_scalar: result stack underflow for IN list value".into(),
                            )
                        })?;
                        list_exprs.push(value_expr);
                    }
                    list_exprs.reverse();

                    let target_expr = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_scalar: result stack underflow for IN list target".into(),
                        )
                    })?;

                    let mut comparisons: Vec<llkv_expr::expr::ScalarExpr<String>> =
                        Vec::with_capacity(list_len);
                    for value in &list_exprs {
                        comparisons.push(llkv_expr::expr::ScalarExpr::compare(
                            target_expr.clone(),
                            llkv_expr::expr::CompareOp::Eq,
                            value.clone(),
                        ));
                    }

                    let mut branches: Vec<(
                        llkv_expr::expr::ScalarExpr<String>,
                        llkv_expr::expr::ScalarExpr<String>,
                    )> = Vec::with_capacity(list_len.saturating_mul(2));

                    for comparison in &comparisons {
                        branches.push((
                            comparison.clone(),
                            llkv_expr::expr::ScalarExpr::literal(Literal::Integer(1)),
                        ));
                    }

                    for comparison in comparisons {
                        let comparison_is_null =
                            llkv_expr::expr::ScalarExpr::is_null(comparison, false);
                        branches.push((
                            comparison_is_null,
                            llkv_expr::expr::ScalarExpr::literal(Literal::Null),
                        ));
                    }

                    let else_expr = Some(llkv_expr::expr::ScalarExpr::literal(Literal::Integer(0)));
                    let in_result = llkv_expr::expr::ScalarExpr::case(None, branches, else_expr);
                    let final_expr = if negated {
                        llkv_expr::expr::ScalarExpr::logical_not(in_result)
                    } else {
                        in_result
                    };

                    result_stack.push(final_expr);
                }
                ScalarExitContext::Case {
                    branch_count,
                    has_operand,
                    has_else,
                } => {
                    let else_expr = if has_else {
                        Some(result_stack.pop().ok_or_else(|| {
                            Error::Internal(
                                "translate_scalar: result stack underflow for CASE ELSE".into(),
                            )
                        })?)
                    } else {
                        None
                    };

                    let mut branches_rev = Vec::with_capacity(branch_count);
                    for _ in 0..branch_count {
                        let then_expr = result_stack.pop().ok_or_else(|| {
                            Error::Internal(
                                "translate_scalar: result stack underflow for CASE THEN".into(),
                            )
                        })?;
                        let when_expr = result_stack.pop().ok_or_else(|| {
                            Error::Internal(
                                "translate_scalar: result stack underflow for CASE WHEN".into(),
                            )
                        })?;
                        branches_rev.push((when_expr, then_expr));
                    }
                    branches_rev.reverse();

                    let operand_expr = if has_operand {
                        Some(result_stack.pop().ok_or_else(|| {
                            Error::Internal(
                                "translate_scalar: result stack underflow for CASE operand".into(),
                            )
                        })?)
                    } else {
                        None
                    };

                    let case_expr =
                        llkv_expr::expr::ScalarExpr::case(operand_expr, branches_rev, else_expr);
                    result_stack.push(case_expr);
                }
                ScalarExitContext::IsNull { negated } => {
                    let inner = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_scalar: result stack underflow for IS NULL operand".into(),
                        )
                    })?;
                    result_stack.push(llkv_expr::expr::ScalarExpr::is_null(inner, negated));
                }
                ScalarExitContext::Between { negated } => {
                    let high = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_scalar: result stack underflow for BETWEEN upper".into(),
                        )
                    })?;
                    let low = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_scalar: result stack underflow for BETWEEN lower".into(),
                        )
                    })?;
                    let expr_value = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_scalar: result stack underflow for BETWEEN operand".into(),
                        )
                    })?;

                    let between_expr = if negated {
                        let less_than = llkv_expr::expr::ScalarExpr::compare(
                            expr_value.clone(),
                            llkv_expr::expr::CompareOp::Lt,
                            low.clone(),
                        );
                        let greater_than = llkv_expr::expr::ScalarExpr::compare(
                            expr_value,
                            llkv_expr::expr::CompareOp::Gt,
                            high,
                        );
                        llkv_expr::expr::ScalarExpr::binary(
                            less_than,
                            llkv_expr::expr::BinaryOp::Or,
                            greater_than,
                        )
                    } else {
                        let greater_or_equal = llkv_expr::expr::ScalarExpr::compare(
                            expr_value.clone(),
                            llkv_expr::expr::CompareOp::GtEq,
                            low,
                        );
                        let less_or_equal = llkv_expr::expr::ScalarExpr::compare(
                            expr_value,
                            llkv_expr::expr::CompareOp::LtEq,
                            high,
                        );
                        llkv_expr::expr::ScalarExpr::binary(
                            greater_or_equal,
                            llkv_expr::expr::BinaryOp::And,
                            less_or_equal,
                        )
                    };
                    result_stack.push(between_expr);
                }
            },
        }
    }

    result_stack
        .pop()
        .ok_or_else(|| Error::Internal("translate_scalar: empty result stack".into()))
}

struct ScalarSubqueryPlanner<'engine, 'vec> {
    engine: &'engine SqlEngine,
    scalar_subqueries: &'vec mut Vec<llkv_plan::ScalarSubquery>,
}

impl<'engine, 'vec> ScalarSubqueryResolver for ScalarSubqueryPlanner<'engine, 'vec> {
    fn handle_scalar_subquery(
        &mut self,
        subquery: &Query,
        resolver: &IdentifierResolver<'_>,
        context: &IdentifierContext,
        outer_scopes: &[IdentifierContext],
    ) -> SqlResult<llkv_expr::expr::ScalarExpr<String>> {
        let mut nested_scopes = outer_scopes.to_vec();
        nested_scopes.push(context.clone());

        let mut tracker = SubqueryCorrelatedColumnTracker::new();
        let mut nested_filter_subqueries = Vec::new();

        let plan = self.engine.build_select_plan_internal(
            subquery.clone(),
            resolver,
            &nested_scopes,
            &mut nested_filter_subqueries,
            Some(&mut tracker),
        )?;

        debug_assert!(nested_filter_subqueries.is_empty());

        let id = u32::try_from(self.scalar_subqueries.len()).map_err(|_| {
            Error::InvalidArgumentError(
                "scalar subquery limit exceeded for current query".to_string(),
            )
        })?;
        let subquery_id = llkv_expr::SubqueryId(id);
        self.scalar_subqueries.push(llkv_plan::ScalarSubquery {
            id: subquery_id,
            plan: Box::new(plan),
            correlated_columns: tracker.into_columns(),
        });

        Ok(llkv_expr::expr::ScalarExpr::scalar_subquery(subquery_id))
    }
}

fn build_abs_case_expr(
    arg: llkv_expr::expr::ScalarExpr<String>,
) -> llkv_expr::expr::ScalarExpr<String> {
    use llkv_expr::expr::{BinaryOp, CompareOp, ScalarExpr};

    let zero = ScalarExpr::literal(Literal::Integer(0));
    let condition = ScalarExpr::compare(arg.clone(), CompareOp::Lt, zero.clone());
    let negated = ScalarExpr::binary(zero.clone(), BinaryOp::Subtract, arg.clone());

    ScalarExpr::case(None, vec![(condition, negated)], Some(arg))
}

fn literal_from_value(value: &ValueWithSpan) -> SqlResult<llkv_expr::expr::ScalarExpr<String>> {
    match &value.value {
        Value::Placeholder(name) => {
            let index = register_placeholder(name)?;
            Ok(llkv_expr::expr::ScalarExpr::literal(literal_placeholder(
                index,
            )))
        }
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
        SqlDataType::Decimal(exact_number_info) | SqlDataType::Numeric(exact_number_info) => {
            // Parse DECIMAL(precision, scale) or use defaults
            match exact_number_info {
                sqlparser::ast::ExactNumberInfo::PrecisionAndScale(p, s) => {
                    Ok(DataType::Decimal128(*p as u8, *s as i8))
                }
                sqlparser::ast::ExactNumberInfo::Precision(p) => {
                    // DECIMAL(p) means scale = 0
                    Ok(DataType::Decimal128(*p as u8, 0))
                }
                sqlparser::ast::ExactNumberInfo::None => {
                    // DECIMAL without precision defaults to DECIMAL(38, 0) per SQL standard
                    Ok(DataType::Decimal128(38, 0))
                }
            }
        }
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

    if table_with_joins_has_join(item) {
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
        TableFactor::NestedJoin { .. } => Err(Error::InvalidArgumentError(
            "JOIN clauses are not supported yet".into(),
        )),
        _ => Err(Error::InvalidArgumentError(
            "queries require a plain table name or derived table".into(),
        )),
    }
}

// TODO: Rename for clarity?  i.e. "...nested_joins" or something?
fn table_with_joins_has_join(item: &TableWithJoins) -> bool {
    if !item.joins.is_empty() {
        return true;
    }
    match &item.relation {
        TableFactor::NestedJoin {
            table_with_joins, ..
        } => table_with_joins_has_join(table_with_joins.as_ref()),
        _ => false,
    }
}

/// Extract table references from a FROM clause, flattening supported JOINs and
/// collecting any join predicates that must be applied as filters.
///
type ExtractedJoinData = (
    Vec<llkv_plan::TableRef>,
    Vec<llkv_plan::JoinMetadata>,
    Vec<Option<SqlExpr>>,
);

/// Returns [`ExtractedJoinData`] (tables, join metadata, join filters).
/// - `tables`: list of all table references in order
/// - `join_metadata`: [`llkv_plan::JoinMetadata`] entries pairing consecutive tables
/// - `join_filters`: ON conditions to be merged into WHERE clause
fn extract_tables(from: &[TableWithJoins]) -> SqlResult<ExtractedJoinData> {
    let mut tables = Vec::new();
    let mut join_metadata = Vec::new();
    let mut join_filters = Vec::new();

    for item in from {
        flatten_table_with_joins(item, &mut tables, &mut join_metadata, &mut join_filters)?;
    }

    Ok((tables, join_metadata, join_filters))
}

fn push_table_factor(
    factor: &TableFactor,
    tables: &mut Vec<llkv_plan::TableRef>,
    join_metadata: &mut Vec<llkv_plan::JoinMetadata>,
    join_filters: &mut Vec<Option<SqlExpr>>,
) -> SqlResult<()> {
    match factor {
        TableFactor::Table { name, alias, .. } => {
            // Note: Index hints (INDEXED BY, NOT INDEXED) are SQLite-specific query hints
            // that are ignored by the `..` pattern. We accept them for compatibility.
            let (schema_opt, table) = parse_schema_qualified_name(name)?;
            let schema = schema_opt.unwrap_or_default();
            let alias_name = alias.as_ref().map(|a| a.name.value.clone());
            tables.push(llkv_plan::TableRef::with_alias(schema, table, alias_name));
            Ok(())
        }
        TableFactor::NestedJoin {
            table_with_joins,
            alias,
        } => {
            if alias.is_some() {
                return Err(Error::InvalidArgumentError(
                    "parenthesized JOINs with aliases are not supported yet".into(),
                ));
            }
            flatten_table_with_joins(
                table_with_joins.as_ref(),
                tables,
                join_metadata,
                join_filters,
            )
        }
        TableFactor::Derived { .. } => Err(Error::InvalidArgumentError(
            "JOIN clauses require base tables; derived tables are not supported".into(),
        )),
        _ => Err(Error::InvalidArgumentError(
            "queries require a plain table name".into(),
        )),
    }
}

fn flatten_table_with_joins(
    item: &TableWithJoins,
    tables: &mut Vec<llkv_plan::TableRef>,
    join_metadata: &mut Vec<llkv_plan::JoinMetadata>,
    join_filters: &mut Vec<Option<SqlExpr>>,
) -> SqlResult<()> {
    push_table_factor(&item.relation, tables, join_metadata, join_filters)?;

    for join in &item.joins {
        let left_table_index = tables.len() - 1;

        match &join.join_operator {
            JoinOperator::CrossJoin(JoinConstraint::None)
            | JoinOperator::Join(JoinConstraint::None)
            | JoinOperator::Inner(JoinConstraint::None) => {
                push_table_factor(&join.relation, tables, join_metadata, join_filters)?;
                join_metadata.push(llkv_plan::JoinMetadata {
                    left_table_index,
                    join_type: llkv_plan::JoinPlan::Inner,
                    on_condition: None,
                });
                join_filters.push(None);
            }
            JoinOperator::Join(JoinConstraint::On(condition))
            | JoinOperator::Inner(JoinConstraint::On(condition)) => {
                push_table_factor(&join.relation, tables, join_metadata, join_filters)?;
                join_filters.push(Some(condition.clone()));
                join_metadata.push(llkv_plan::JoinMetadata {
                    left_table_index,
                    join_type: llkv_plan::JoinPlan::Inner,
                    on_condition: None,
                });
            }
            JoinOperator::Left(JoinConstraint::On(condition))
            | JoinOperator::LeftOuter(JoinConstraint::On(condition)) => {
                push_table_factor(&join.relation, tables, join_metadata, join_filters)?;
                join_filters.push(Some(condition.clone()));
                join_metadata.push(llkv_plan::JoinMetadata {
                    left_table_index,
                    join_type: llkv_plan::JoinPlan::Left,
                    on_condition: None,
                });
            }
            JoinOperator::Left(JoinConstraint::None)
            | JoinOperator::LeftOuter(JoinConstraint::None) => {
                push_table_factor(&join.relation, tables, join_metadata, join_filters)?;
                join_metadata.push(llkv_plan::JoinMetadata {
                    left_table_index,
                    join_type: llkv_plan::JoinPlan::Left,
                    on_condition: None,
                });
                join_filters.push(None);
            }
            JoinOperator::CrossJoin(_) => {
                return Err(Error::InvalidArgumentError(
                    "CROSS JOIN with constraints is not supported".into(),
                ));
            }
            JoinOperator::Join(JoinConstraint::Using(_))
            | JoinOperator::Inner(JoinConstraint::Using(_))
            | JoinOperator::Left(JoinConstraint::Using(_))
            | JoinOperator::LeftOuter(JoinConstraint::Using(_)) => {
                return Err(Error::InvalidArgumentError(
                    "JOIN ... USING (...) is not supported yet".into(),
                ));
            }
            JoinOperator::Join(JoinConstraint::Natural)
            | JoinOperator::Inner(JoinConstraint::Natural)
            | JoinOperator::Left(JoinConstraint::Natural)
            | JoinOperator::LeftOuter(JoinConstraint::Natural)
            | JoinOperator::Right(_)
            | JoinOperator::RightOuter(_)
            | JoinOperator::FullOuter(_)
            | JoinOperator::Semi(_)
            | JoinOperator::LeftSemi(_)
            | JoinOperator::LeftAnti(_)
            | JoinOperator::RightSemi(_)
            | JoinOperator::RightAnti(_)
            | JoinOperator::CrossApply
            | JoinOperator::OuterApply
            | JoinOperator::Anti(_)
            | JoinOperator::StraightJoin(_) => {
                return Err(Error::InvalidArgumentError(
                    "only INNER JOIN and LEFT JOIN with optional ON constraints are supported"
                        .into(),
                ));
            }
            other => {
                return Err(Error::InvalidArgumentError(format!(
                    "unsupported JOIN clause: {other:?}"
                )));
            }
        }
    }

    Ok(())
}

fn group_by_is_empty(expr: &GroupByExpr) -> bool {
    matches!(
        expr,
        GroupByExpr::Expressions(exprs, modifiers)
            if exprs.is_empty() && modifiers.is_empty()
    )
}

fn convert_value_table_mode(mode: sqlparser::ast::ValueTableMode) -> llkv_plan::ValueTableMode {
    use llkv_plan::ValueTableMode as PlanMode;
    match mode {
        sqlparser::ast::ValueTableMode::AsStruct => PlanMode::AsStruct,
        sqlparser::ast::ValueTableMode::AsValue => PlanMode::AsValue,
        sqlparser::ast::ValueTableMode::DistinctAsStruct => PlanMode::DistinctAsStruct,
        sqlparser::ast::ValueTableMode::DistinctAsValue => PlanMode::DistinctAsValue,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, Int32Array, Int64Array, StringArray};
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
    fn set_constraint_mode_updates_session() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        assert_eq!(
            engine.session().constraint_enforcement_mode(),
            ConstraintEnforcementMode::Immediate
        );

        engine
            .execute("SET constraint_enforcement_mode = deferred")
            .expect("set deferred mode");

        assert_eq!(
            engine.session().constraint_enforcement_mode(),
            ConstraintEnforcementMode::Deferred
        );

        engine
            .execute("SET constraint_enforcement_mode = IMMEDIATE")
            .expect("set immediate mode");

        assert_eq!(
            engine.session().constraint_enforcement_mode(),
            ConstraintEnforcementMode::Immediate
        );
    }

    #[test]
    fn set_constraint_mode_is_session_scoped() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(Arc::clone(&pager));
        let shared_context = engine.runtime_context();
        let peer = SqlEngine::with_context(
            Arc::clone(&shared_context),
            engine.default_nulls_first_for_tests(),
        );

        engine
            .execute("SET constraint_enforcement_mode = deferred")
            .expect("set deferred mode");

        assert_eq!(
            engine.session().constraint_enforcement_mode(),
            ConstraintEnforcementMode::Deferred
        );
        assert_eq!(
            peer.session().constraint_enforcement_mode(),
            ConstraintEnforcementMode::Immediate
        );
    }

    #[test]
    fn test_interval_expr_structure() {
        use sqlparser::ast::{BinaryOperator, Expr as SqlExprAst, Query, SetExpr, Statement};
        use sqlparser::dialect::GenericDialect;

        let dialect = GenericDialect {};
        let sql = "SELECT CAST('1998-12-01' AS DATE) - INTERVAL '90' DAY";
        let statements = Parser::parse_sql(&dialect, sql).unwrap();

        assert_eq!(statements.len(), 1, "expected single statement");

        let Statement::Query(query) = &statements[0] else {
            panic!("expected Query statement");
        };

        let Query { body, .. } = query.as_ref();
        let SetExpr::Select(select) = body.as_ref() else {
            panic!("expected Select body");
        };

        assert_eq!(select.projection.len(), 1, "expected single projection");

        // Verify the projection is a BinaryOp with Minus operator and Interval on the right
        match &select.projection[0] {
            sqlparser::ast::SelectItem::UnnamedExpr(SqlExprAst::BinaryOp { left, op, right }) => {
                // Left side should be a CAST expression
                assert!(
                    matches!(left.as_ref(), SqlExprAst::Cast { .. }),
                    "expected CAST on left"
                );

                // Operator should be Minus
                assert_eq!(*op, BinaryOperator::Minus, "expected Minus operator");

                // Right side should be an Interval
                assert!(
                    matches!(right.as_ref(), SqlExprAst::Interval(_)),
                    "expected Interval on right"
                );

                if let SqlExprAst::Interval(interval) = right.as_ref() {
                    assert_eq!(
                        interval.leading_field,
                        Some(sqlparser::ast::DateTimeField::Day)
                    );
                }
            }
            other => panic!("unexpected projection structure: {other:?}"),
        }
    }

    #[test]
    fn test_insert_batching_across_calls() {
        let engine = SqlEngine::new(Arc::new(MemPager::default()));

        // Create table
        engine.execute("CREATE TABLE test (id INTEGER)").unwrap();

        // Insert two rows in SEPARATE execute() calls (simulating SLT)
        engine.execute("INSERT INTO test VALUES (1)").unwrap();
        engine.execute("INSERT INTO test VALUES (2)").unwrap();

        // SELECT will flush the buffer - result will have [INSERT, SELECT]
        let result = engine.execute("SELECT * FROM test ORDER BY id").unwrap();
        let select_result = result
            .into_iter()
            .find_map(|res| match res {
                RuntimeStatementResult::Select { execution, .. } => {
                    Some(execution.collect().unwrap())
                }
                _ => None,
            })
            .expect("expected SELECT result in response");
        let batches = select_result;
        assert_eq!(
            batches[0].num_rows(),
            2,
            "Should have 2 rows after cross-call batching"
        );
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
    fn not_null_in_list_filters_all_rows() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE tab0(col0 INTEGER, col1 INTEGER, col2 INTEGER)")
            .expect("create table");
        engine
            .execute("INSERT INTO tab0 VALUES (1, 2, 3)")
            .expect("insert row");

        let batches = engine
            .sql("SELECT * FROM tab0 WHERE NOT ( NULL ) IN ( - col2 * + col2 )")
            .expect("run IN list null comparison");

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(total_rows, 0, "expected IN list filter to remove all rows");
    }

    #[test]
    fn empty_in_list_filters_all_rows() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE test_table(col INTEGER)")
            .expect("create table");
        engine
            .execute("INSERT INTO test_table VALUES (1), (2), (3)")
            .expect("insert rows");

        let batches = engine
            .sql("SELECT * FROM test_table WHERE col IN ()")
            .expect("run empty IN list");

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(total_rows, 0, "expected empty IN list to filter all rows");
    }

    #[test]
    fn empty_not_in_list_preserves_all_rows() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE test_table(col INTEGER)")
            .expect("create table");
        engine
            .execute("INSERT INTO test_table VALUES (1), (2), (3)")
            .expect("insert rows");

        let batches = engine
            .sql("SELECT * FROM test_table WHERE col NOT IN () ORDER BY col")
            .expect("run empty NOT IN list");

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(
            total_rows, 3,
            "expected empty NOT IN list to preserve all rows"
        );

        let mut values: Vec<i64> = Vec::new();
        for batch in &batches {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int column");
            for idx in 0..column.len() {
                if !column.is_null(idx) {
                    values.push(column.value(idx));
                }
            }
        }

        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn empty_in_list_with_constant_expression() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        let batches = engine
            .sql("SELECT 1 IN ()")
            .expect("run constant empty IN list");

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(total_rows, 1, "expected one result row");

        let value = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int column")
            .value(0);

        assert_eq!(value, 0, "expected 1 IN () to evaluate to 0 (false)");
    }

    #[test]
    fn empty_not_in_list_with_constant_expression() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        let batches = engine
            .sql("SELECT 1 NOT IN ()")
            .expect("run constant empty NOT IN list");

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(total_rows, 1, "expected one result row");

        let value = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int column")
            .value(0);

        assert_eq!(value, 1, "expected 1 NOT IN () to evaluate to 1 (true)");
    }

    #[test]
    fn not_in_with_cast_preserves_rows_for_self_comparison() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE tab2(col1 INTEGER, col2 INTEGER)")
            .expect("create tab2");
        engine
            .execute("INSERT INTO tab2 VALUES (51, 51), (67, 67), (77, 77)")
            .expect("seed tab2");

        let batches = engine
            .sql(
                "SELECT col1 FROM tab2 WHERE NOT col2 NOT IN ( + CAST ( + + col2 AS REAL ) ) ORDER BY col1",
            )
            .expect("run NOT IN self comparison query");

        let mut values: Vec<i64> = Vec::new();
        for batch in &batches {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int column");
            for idx in 0..column.len() {
                if !column.is_null(idx) {
                    values.push(column.value(idx));
                }
            }
        }

        assert_eq!(values, vec![51, 67, 77]);
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
    fn not_between_null_bounds_matches_sqlite_behavior() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE tab2(col1 INTEGER, col2 INTEGER)")
            .expect("create tab2");
        engine
            .execute("INSERT INTO tab2 VALUES (1, 2), (-5, 7), (NULL, 11)")
            .expect("seed rows");

        let batches = engine
            .sql(
                "SELECT DISTINCT - col2 AS col1 FROM tab2 WHERE NOT ( col1 ) BETWEEN ( NULL ) AND ( + col1 - col2 )",
            )
            .expect("run NOT BETWEEN query with NULL bounds");

        let mut values: Vec<i64> = Vec::new();
        for batch in &batches {
            let column = batch.column(0);
            match column.data_type() {
                arrow::datatypes::DataType::Int64 => {
                    let array = column
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .expect("int64 column");
                    for idx in 0..array.len() {
                        if !array.is_null(idx) {
                            values.push(array.value(idx));
                        }
                    }
                }
                arrow::datatypes::DataType::Int32 => {
                    let array = column
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .expect("int32 column");
                    for idx in 0..array.len() {
                        if !array.is_null(idx) {
                            values.push(array.value(idx) as i64);
                        }
                    }
                }
                other => panic!("unexpected data type: {other:?}"),
            }
        }

        values.sort_unstable();
        assert_eq!(values, vec![-7, -2]);
    }

    #[test]
    fn not_chain_precedence_matches_sqlite_behavior() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE tab1(col0 INTEGER)")
            .expect("create tab1");
        engine
            .execute("INSERT INTO tab1 VALUES (1), (2)")
            .expect("seed tab1");

        use sqlparser::ast::Statement;
        use sqlparser::dialect::SQLiteDialect;
        use sqlparser::parser::Parser;

        let dialect = SQLiteDialect {};
        let mut statements = Parser::parse_sql(
            &dialect,
            "SELECT DISTINCT 85 AS value FROM tab1 WHERE NOT + 84 < - + 69 GROUP BY col0, col0",
        )
        .expect("parse sql");
        let statement = statements.pop().expect("expected single statement");
        let Statement::Query(query_ast) = statement else {
            panic!("expected SELECT query");
        };
        let plan = engine
            .build_select_plan(*query_ast)
            .expect("build select plan");
        let filter_expr = plan.filter.expect("expected filter predicate").predicate;
        if let llkv_expr::expr::Expr::Not(inner) = &filter_expr {
            if !matches!(inner.as_ref(), llkv_expr::expr::Expr::Compare { .. }) {
                panic!("expected NOT to wrap comparison, got: {inner:?}");
            }
        } else {
            panic!("expected filter to be NOT-wrapped comparison: {filter_expr:?}");
        }

        let batches = engine
            .sql(
                "SELECT DISTINCT 85 AS value FROM tab1 WHERE NOT + 84 < - + 69 GROUP BY col0, col0",
            )
            .expect("run NOT precedence query");

        let mut values: Vec<i64> = Vec::new();
        for batch in &batches {
            let column = batch.column(0);
            match column.data_type() {
                arrow::datatypes::DataType::Int64 => {
                    let array = column
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .expect("int64 column");
                    for idx in 0..array.len() {
                        if !array.is_null(idx) {
                            values.push(array.value(idx));
                        }
                    }
                }
                arrow::datatypes::DataType::Int32 => {
                    let array = column
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .expect("int32 column");
                    for idx in 0..array.len() {
                        if !array.is_null(idx) {
                            values.push(array.value(idx) as i64);
                        }
                    }
                }
                arrow::datatypes::DataType::Float64 => {
                    let array = column
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .expect("float64 column");
                    for idx in 0..array.len() {
                        if !array.is_null(idx) {
                            values.push(array.value(idx) as i64);
                        }
                    }
                }
                other => panic!("unexpected data type: {other:?}"),
            }
        }

        values.sort_unstable();
        assert_eq!(values, vec![85]);
    }

    #[test]
    fn not_between_null_bounds_matches_harness_fixture() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE tab2(col0 INTEGER, col1 INTEGER, col2 INTEGER)")
            .expect("create tab2");
        engine
            .execute("INSERT INTO tab2 VALUES (7, 31, 27), (79, 17, 38), (78, 59, 26)")
            .expect("seed rows");

        let batches = engine
            .sql(
                "SELECT DISTINCT - col2 AS col1 FROM tab2 WHERE NOT ( col1 ) BETWEEN ( NULL ) AND ( + col1 - col2 )",
            )
            .expect("run harness-matched NOT BETWEEN query");

        let mut values: Vec<i64> = Vec::new();
        for batch in &batches {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("integer column");
            for idx in 0..column.len() {
                if !column.is_null(idx) {
                    values.push(column.value(idx));
                }
            }
        }

        values.sort_unstable();
        assert_eq!(values, vec![-38, -27, -26]);
    }

    #[test]
    fn not_between_null_bounds_parser_negated_flag() {
        use sqlparser::ast::{Expr as SqlExprAst, Statement};
        use sqlparser::dialect::SQLiteDialect;
        use sqlparser::parser::Parser;

        let dialect = SQLiteDialect {};
        let sql = "SELECT DISTINCT - col2 AS col1 FROM tab2 WHERE NOT ( col1 ) BETWEEN ( NULL ) AND ( + col1 - col2 )";

        let mut statements = Parser::parse_sql(&dialect, sql).expect("parse sql");
        let statement = statements.pop().expect("expected single statement");
        let Statement::Query(query) = statement else {
            panic!("expected SELECT query");
        };
        let select = query.body.as_select().expect("expected SELECT body");
        let where_expr = select.selection.as_ref().expect("expected WHERE clause");

        match where_expr {
            SqlExprAst::UnaryOp {
                op: sqlparser::ast::UnaryOperator::Not,
                expr,
            } => match expr.as_ref() {
                SqlExprAst::Between { negated, .. } => {
                    assert!(
                        !negated,
                        "expected BETWEEN parser to treat leading NOT as part of expression"
                    );
                }
                other => panic!("unexpected inner expression: {other:?}"),
            },
            other => panic!("unexpected where expression: {other:?}"),
        }
    }

    #[test]
    fn double_negated_between_null_bounds_filters_all_rows() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE tab2(col0 INTEGER, col1 INTEGER, col2 INTEGER)")
            .expect("create tab2");
        engine
            .execute("INSERT INTO tab2 VALUES (1, 2, 3), (-2, -13, 19), (NULL, 5, 7)")
            .expect("seed rows");

        let batches = engine
            .sql(
                "SELECT - col1 * + col2 FROM tab2 WHERE NOT ( col1 ) NOT BETWEEN ( NULL ) AND ( col0 )",
            )
            .expect("run double NOT BETWEEN query with NULL bounds");

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(
            total_rows, 0,
            "expected double NOT BETWEEN to filter all rows"
        );
    }

    #[test]
    fn not_scalar_less_than_null_filters_all_rows() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute("CREATE TABLE tab(col0 INTEGER, col2 INTEGER)")
            .expect("create tab");
        engine
            .execute("INSERT INTO tab VALUES (1, 2), (5, 10), (-3, 7)")
            .expect("seed rows");

        let batches = engine
            .sql("SELECT col0 FROM tab WHERE NOT ( - col0 / - col2 + - col0 ) < NULL")
            .expect("run NOT < NULL query");

        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(total_rows, 0, "expected NOT < NULL to filter all rows");
    }

    #[test]
    fn left_join_not_is_not_null_on_literal_flips_to_is_null() {
        use sqlparser::ast::Statement;
        use sqlparser::dialect::SQLiteDialect;
        use sqlparser::parser::Parser;

        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute(
                "CREATE TABLE tab0(col0 INTEGER, col1 INTEGER, col2 INTEGER);\
                 CREATE TABLE tab1(col0 INTEGER, col1 INTEGER, col2 INTEGER);",
            )
            .expect("create tables");

        let sql = "SELECT DISTINCT * FROM tab1 AS cor0 LEFT JOIN tab1 AS cor1 ON NOT 86 IS NOT NULL, tab0 AS cor2";
        let dialect = SQLiteDialect {};
        let mut statements = Parser::parse_sql(&dialect, sql).expect("parse sql");
        let statement = statements.pop().expect("expected statement");
        let Statement::Query(query) = statement else {
            panic!("expected SELECT query");
        };

        let plan = engine.build_select_plan(*query).expect("build select plan");

        assert_eq!(plan.joins.len(), 1, "expected single explicit join entry");

        let left_join = &plan.joins[0];
        let on_condition = left_join
            .on_condition
            .as_ref()
            .expect("left join should preserve ON predicate");

        match on_condition {
            llkv_expr::expr::Expr::IsNull { expr, negated } => {
                assert!(!negated, "expected NOT to flip into IS NULL");
                assert!(matches!(
                    expr,
                    llkv_expr::expr::ScalarExpr::Literal(llkv_expr::literal::Literal::Integer(86))
                ));
            }
            other => panic!("unexpected ON predicate: {other:?}"),
        }
    }

    #[test]
    fn left_join_constant_false_preserves_left_rows_with_null_right() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        engine
            .execute(
                "CREATE TABLE tab0(col0 INTEGER, col1 INTEGER, col2 INTEGER);\
                 CREATE TABLE tab1(col0 INTEGER, col1 INTEGER, col2 INTEGER);",
            )
            .expect("create tables");

        engine
            .execute(
                "INSERT INTO tab0 VALUES (1, 2, 3), (4, 5, 6);\
                 INSERT INTO tab1 VALUES (10, 11, 12), (13, 14, 15);",
            )
            .expect("seed rows");

        let batches = engine
            .sql(
                "SELECT * FROM tab1 AS cor0 LEFT JOIN tab1 AS cor1 ON NOT 86 IS NOT NULL, tab0 AS cor2 ORDER BY cor0.col0, cor2.col0",
            )
            .expect("execute join query");

        let mut total_rows = 0;
        for batch in &batches {
            total_rows += batch.num_rows();

            // Columns 0-2 belong to cor0, 3-5 to cor1, 6-8 to cor2.
            for row_idx in 0..batch.num_rows() {
                for col_idx in 3..6 {
                    assert!(
                        batch.column(col_idx).is_null(row_idx),
                        "expected right table column {} to be NULL in row {}",
                        col_idx,
                        row_idx
                    );
                }
            }
        }

        // Two left rows cross two tab0 rows -> four total results.
        assert_eq!(total_rows, 4, "expected Cartesian product with tab0 only");
    }

    #[test]
    fn cross_join_duplicate_table_name_resolves_columns() {
        let pager = Arc::new(MemPager::default());
        let engine = SqlEngine::new(pager);

        use sqlparser::ast::{SetExpr, Statement};
        use sqlparser::dialect::SQLiteDialect;
        use sqlparser::parser::Parser;

        engine
            .execute("CREATE TABLE tab1(col0 INTEGER, col1 INTEGER, col2 INTEGER)")
            .expect("create tab1");
        engine
            .execute("INSERT INTO tab1 VALUES (7, 8, 9)")
            .expect("insert tab1 row");

        let dialect = SQLiteDialect {};
        let ast = Parser::parse_sql(
            &dialect,
            "SELECT tab1.col2 FROM tab1 AS cor0 CROSS JOIN tab1",
        )
        .expect("parse cross join query");
        let Statement::Query(query) = &ast[0] else {
            panic!("expected SELECT query");
        };
        let select = match query.body.as_ref() {
            SetExpr::Select(select) => select.as_ref(),
            other => panic!("unexpected query body: {other:?}"),
        };
        assert_eq!(select.from.len(), 1);
        assert!(!select.from[0].joins.is_empty());

        let batches = engine
            .sql("SELECT tab1.col2 FROM tab1 AS cor0 CROSS JOIN tab1")
            .expect("run cross join with alias and base table");

        let mut values = Vec::new();
        for batch in &batches {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int64 column");
            for idx in 0..column.len() {
                if !column.is_null(idx) {
                    values.push(column.value(idx));
                }
            }
        }
        assert_eq!(values, vec![9]);

        engine
            .execute("CREATE TABLE strings(a TEXT)")
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
