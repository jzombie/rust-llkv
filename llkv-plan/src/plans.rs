//! Logical query plan structures for LLKV.
//!
//! This module defines the plan structures that represent logical query operations
//! before they are executed. Plans are created by SQL parsers or fluent builders and
//! consumed by execution engines.

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use llkv_expr::expr::SubqueryId;
use llkv_result::Error;
use rustc_hash::FxHashMap;

/// Result type for plan operations.
pub type PlanResult<T> = llkv_result::Result<T>;

// ============================================================================
// Filter Metadata
// ============================================================================

/// Boolean predicate plus correlated subquery metadata attached to a [`SelectPlan`].
#[derive(Clone, Debug)]
pub struct SelectFilter {
    /// Predicate applied to rows before projections execute.
    pub predicate: llkv_expr::expr::Expr<'static, String>,
    /// Correlated subqueries required to evaluate the predicate.
    pub subqueries: Vec<FilterSubquery>,
}

/// Correlated subquery invoked from within a filter predicate.
#[derive(Clone, Debug)]
pub struct FilterSubquery {
    /// Identifier referenced by [`llkv_expr::expr::Expr::Exists`].
    pub id: SubqueryId,
    /// Logical plan for the subquery.
    pub plan: Box<SelectPlan>,
    /// Mappings for correlated column placeholders to real outer columns.
    pub correlated_columns: Vec<CorrelatedColumn>,
}

/// Correlated subquery invoked from within a scalar projection expression.
#[derive(Clone, Debug)]
pub struct ScalarSubquery {
    /// Identifier referenced by [`llkv_expr::expr::ScalarExpr::ScalarSubquery`].
    pub id: SubqueryId,
    /// Logical plan for the subquery.
    pub plan: Box<SelectPlan>,
    /// Mappings for correlated column placeholders to real outer columns.
    pub correlated_columns: Vec<CorrelatedColumn>,
}

/// Description of a correlated column captured by an EXISTS predicate.
#[derive(Clone, Debug)]
pub struct CorrelatedColumn {
    /// Placeholder column name injected into the subquery expression tree.
    pub placeholder: String,
    /// Canonical outer column name.
    pub column: String,
    /// Optional nested field path for struct lookups.
    pub field_path: Vec<String>,
}

// ============================================================================
// PlanValue Types
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
pub enum PlanValue {
    Null,
    Integer(i64),
    Float(f64),
    String(String),
    Struct(FxHashMap<String, PlanValue>),
}

impl From<&str> for PlanValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<String> for PlanValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<i64> for PlanValue {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for PlanValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<bool> for PlanValue {
    fn from(value: bool) -> Self {
        // Store booleans as integers for compatibility
        if value {
            Self::Integer(1)
        } else {
            Self::Integer(0)
        }
    }
}

impl From<i32> for PlanValue {
    fn from(value: i32) -> Self {
        Self::Integer(value as i64)
    }
}

/// Convert a `Literal` from llkv-expr into a `PlanValue`.
///
/// This is useful for evaluating predicates that contain literal values,
/// such as in HAVING clauses or filter expressions.
pub fn plan_value_from_literal(literal: &llkv_expr::Literal) -> PlanResult<PlanValue> {
    use llkv_expr::Literal;

    match literal {
        Literal::Null => Ok(PlanValue::Null),
        Literal::Integer(i) => {
            // Convert i128 to i64, checking for overflow
            if *i > i64::MAX as i128 || *i < i64::MIN as i128 {
                Err(Error::InvalidArgumentError(format!(
                    "Integer literal {} out of range for i64",
                    i
                )))
            } else {
                Ok(PlanValue::Integer(*i as i64))
            }
        }
        Literal::Float(f) => Ok(PlanValue::Float(*f)),
        Literal::String(s) => Ok(PlanValue::String(s.clone())),
        Literal::Boolean(b) => Ok(PlanValue::from(*b)),
        Literal::Struct(fields) => {
            let mut map = FxHashMap::with_capacity_and_hasher(fields.len(), Default::default());
            for (name, value) in fields {
                let plan_value = plan_value_from_literal(value)?;
                map.insert(name.clone(), plan_value);
            }
            Ok(PlanValue::Struct(map))
        }
    }
}

// ============================================================================
// CREATE TABLE Plan
// ============================================================================

/// Multi-column unique constraint specification.
#[derive(Clone, Debug)]
pub struct MultiColumnUniqueSpec {
    /// Optional name for the unique constraint
    pub name: Option<String>,
    /// Column names participating in this UNIQUE constraint
    pub columns: Vec<String>,
}

/// Plan for creating a table.
#[derive(Clone, Debug)]
pub struct CreateTablePlan {
    pub name: String,
    pub if_not_exists: bool,
    pub or_replace: bool,
    pub columns: Vec<PlanColumnSpec>,
    pub source: Option<CreateTableSource>,
    /// Optional storage namespace for the table.
    pub namespace: Option<String>,
    pub foreign_keys: Vec<ForeignKeySpec>,
    pub multi_column_uniques: Vec<MultiColumnUniqueSpec>,
}

impl CreateTablePlan {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            if_not_exists: false,
            or_replace: false,
            columns: Vec::new(),
            source: None,
            namespace: None,
            foreign_keys: Vec::new(),
            multi_column_uniques: Vec::new(),
        }
    }
}

// ============================================================================
// DROP TABLE Plan
// ============================================================================

/// Plan for dropping a table.
#[derive(Clone, Debug)]
pub struct DropTablePlan {
    pub name: String,
    pub if_exists: bool,
}

impl DropTablePlan {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            if_exists: false,
        }
    }

    pub fn if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }
}

// ============================================================================
// CREATE VIEW Plan
// ============================================================================

/// Plan for creating a view.
#[derive(Clone, Debug)]
pub struct CreateViewPlan {
    pub name: String,
    pub if_not_exists: bool,
    pub view_definition: String,
    pub select_plan: Box<SelectPlan>,
    /// Optional storage namespace for the view (e.g., "temp" for temporary views).
    pub namespace: Option<String>,
}

impl CreateViewPlan {
    pub fn new(name: impl Into<String>, view_definition: String, select_plan: SelectPlan) -> Self {
        Self {
            name: name.into(),
            if_not_exists: false,
            view_definition,
            select_plan: Box::new(select_plan),
            namespace: None,
        }
    }
}

// ============================================================================
// DROP VIEW Plan
// ============================================================================

/// Plan for dropping a view.
#[derive(Clone, Debug)]
pub struct DropViewPlan {
    pub name: String,
    pub if_exists: bool,
}

impl DropViewPlan {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            if_exists: false,
        }
    }

    pub fn if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }
}

// ============================================================================
// RENAME TABLE Plan
// ============================================================================

/// Plan for renaming a table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenameTablePlan {
    pub current_name: String,
    pub new_name: String,
    pub if_exists: bool,
}

impl RenameTablePlan {
    pub fn new(current_name: impl Into<String>, new_name: impl Into<String>) -> Self {
        Self {
            current_name: current_name.into(),
            new_name: new_name.into(),
            if_exists: false,
        }
    }

    pub fn if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }
}

/// Plan for dropping an index.
#[derive(Clone, Debug, PartialEq)]
pub struct DropIndexPlan {
    pub name: String,
    pub canonical_name: String,
    pub if_exists: bool,
}

impl DropIndexPlan {
    pub fn new(name: impl Into<String>) -> Self {
        let display = name.into();
        Self {
            canonical_name: display.to_ascii_lowercase(),
            name: display,
            if_exists: false,
        }
    }

    pub fn with_canonical(mut self, canonical: impl Into<String>) -> Self {
        self.canonical_name = canonical.into();
        self
    }

    pub fn if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }
}

/// Plan for rebuilding an index.
#[derive(Clone, Debug, PartialEq)]
pub struct ReindexPlan {
    pub name: String,
    pub canonical_name: String,
}

impl ReindexPlan {
    pub fn new(name: impl Into<String>) -> Self {
        let display = name.into();
        Self {
            canonical_name: display.to_ascii_lowercase(),
            name: display,
        }
    }

    pub fn with_canonical(mut self, canonical: impl Into<String>) -> Self {
        self.canonical_name = canonical.into();
        self
    }
}

// ============================================================================
// ALTER TABLE Plan Structures
// ============================================================================

/// Plan for ALTER TABLE operations.
#[derive(Clone, Debug, PartialEq)]
pub struct AlterTablePlan {
    pub table_name: String,
    pub if_exists: bool,
    pub operation: AlterTableOperation,
}

/// Specific ALTER TABLE operation to perform.
#[derive(Clone, Debug, PartialEq)]
pub enum AlterTableOperation {
    /// RENAME COLUMN old_name TO new_name
    RenameColumn {
        old_column_name: String,
        new_column_name: String,
    },
    /// ALTER COLUMN column_name SET DATA TYPE new_type
    SetColumnDataType {
        column_name: String,
        new_data_type: String, // SQL type string like "INTEGER", "VARCHAR", etc.
    },
    /// DROP COLUMN column_name
    DropColumn {
        column_name: String,
        if_exists: bool,
        cascade: bool,
    },
}

impl AlterTablePlan {
    pub fn new(table_name: impl Into<String>, operation: AlterTableOperation) -> Self {
        Self {
            table_name: table_name.into(),
            if_exists: false,
            operation,
        }
    }

    pub fn if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }
}

// ============================================================================
// FOREIGN KEY Plan Structures
// ============================================================================

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ForeignKeyAction {
    #[default]
    NoAction,
    Restrict,
}

#[derive(Clone, Debug)]
pub struct ForeignKeySpec {
    pub name: Option<String>,
    pub columns: Vec<String>,
    pub referenced_table: String,
    pub referenced_columns: Vec<String>,
    pub on_delete: ForeignKeyAction,
    pub on_update: ForeignKeyAction,
}

// ============================================================================
// CREATE INDEX Plan
// ============================================================================

/// Column specification for CREATE INDEX statements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexColumnPlan {
    pub name: String,
    pub ascending: bool,
    pub nulls_first: bool,
}

impl IndexColumnPlan {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ascending: true,
            nulls_first: false,
        }
    }

    pub fn with_sort(mut self, ascending: bool, nulls_first: bool) -> Self {
        self.ascending = ascending;
        self.nulls_first = nulls_first;
        self
    }
}

/// Plan for creating an index on a table.
#[derive(Clone, Debug)]
pub struct CreateIndexPlan {
    pub name: Option<String>,
    pub table: String,
    pub unique: bool,
    pub if_not_exists: bool,
    pub columns: Vec<IndexColumnPlan>,
}

impl CreateIndexPlan {
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            name: None,
            table: table.into(),
            unique: false,
            if_not_exists: false,
            columns: Vec::new(),
        }
    }

    pub fn with_name(mut self, name: Option<String>) -> Self {
        self.name = name;
        self
    }

    pub fn with_unique(mut self, unique: bool) -> Self {
        self.unique = unique;
        self
    }

    pub fn with_if_not_exists(mut self, if_not_exists: bool) -> Self {
        self.if_not_exists = if_not_exists;
        self
    }

    pub fn with_columns(mut self, columns: Vec<IndexColumnPlan>) -> Self {
        self.columns = columns;
        self
    }
}

/// Column specification produced by the logical planner.
///
/// This struct flows from the planner into the runtime/executor so callers can
/// reason about column metadata without duplicating field definitions.
#[derive(Clone, Debug)]
pub struct PlanColumnSpec {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
    pub unique: bool,
    /// Optional CHECK constraint expression (SQL string).
    /// Example: "t.t=42" for CHECK(t.t=42)
    pub check_expr: Option<String>,
}

impl PlanColumnSpec {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
            primary_key: false,
            unique: false,
            check_expr: None,
        }
    }

    pub fn with_primary_key(mut self, primary_key: bool) -> Self {
        self.primary_key = primary_key;
        if primary_key {
            self.unique = true;
        }
        self
    }

    pub fn with_unique(mut self, unique: bool) -> Self {
        if unique {
            self.unique = true;
        }
        self
    }

    pub fn with_check(mut self, check_expr: Option<String>) -> Self {
        self.check_expr = check_expr;
        self
    }
}

/// Trait for types that can be converted into a [`PlanColumnSpec`].
pub trait IntoPlanColumnSpec {
    fn into_plan_column_spec(self) -> PlanColumnSpec;
}

/// Column nullability specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnNullability {
    Nullable,
    NotNull,
}

impl ColumnNullability {
    pub fn is_nullable(self) -> bool {
        matches!(self, ColumnNullability::Nullable)
    }
}

/// Convenience constant for nullable columns.
#[allow(non_upper_case_globals)]
pub const Nullable: ColumnNullability = ColumnNullability::Nullable;

/// Convenience constant for non-null columns.
#[allow(non_upper_case_globals)]
pub const NotNull: ColumnNullability = ColumnNullability::NotNull;

impl IntoPlanColumnSpec for PlanColumnSpec {
    fn into_plan_column_spec(self) -> PlanColumnSpec {
        self
    }
}

impl<T> IntoPlanColumnSpec for &T
where
    T: Clone + IntoPlanColumnSpec,
{
    fn into_plan_column_spec(self) -> PlanColumnSpec {
        self.clone().into_plan_column_spec()
    }
}

impl IntoPlanColumnSpec for (&str, DataType) {
    fn into_plan_column_spec(self) -> PlanColumnSpec {
        PlanColumnSpec::new(self.0, self.1, true)
    }
}

impl IntoPlanColumnSpec for (&str, DataType, bool) {
    fn into_plan_column_spec(self) -> PlanColumnSpec {
        PlanColumnSpec::new(self.0, self.1, self.2)
    }
}

impl IntoPlanColumnSpec for (&str, DataType, ColumnNullability) {
    fn into_plan_column_spec(self) -> PlanColumnSpec {
        PlanColumnSpec::new(self.0, self.1, self.2.is_nullable())
    }
}

/// Source data for CREATE TABLE AS SELECT.
#[derive(Clone, Debug)]
pub enum CreateTableSource {
    Batches {
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    },
    Select {
        plan: Box<SelectPlan>,
    },
}

// ============================================================================
// INSERT Plan
// ============================================================================

/// SQLite conflict resolution action for INSERT statements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertConflictAction {
    /// Standard INSERT behavior - fail on constraint violation
    None,
    /// INSERT OR REPLACE - update existing row on conflict
    Replace,
    /// INSERT OR IGNORE - skip row on conflict
    Ignore,
    /// INSERT OR ABORT - abort transaction on conflict
    Abort,
    /// INSERT OR FAIL - fail statement on conflict (but don't rollback)
    Fail,
    /// INSERT OR ROLLBACK - rollback transaction on conflict
    Rollback,
}

/// Plan for inserting data into a table.
#[derive(Clone, Debug)]
pub struct InsertPlan {
    pub table: String,
    pub columns: Vec<String>,
    pub source: InsertSource,
    pub on_conflict: InsertConflictAction,
}

/// Source data for INSERT operations.
#[derive(Clone, Debug)]
pub enum InsertSource {
    Rows(Vec<Vec<PlanValue>>),
    Batches(Vec<RecordBatch>),
    Select { plan: Box<SelectPlan> },
}

// ============================================================================
// UPDATE Plan
// ============================================================================

/// Plan for updating rows in a table.
#[derive(Clone, Debug)]
pub struct UpdatePlan {
    pub table: String,
    pub assignments: Vec<ColumnAssignment>,
    pub filter: Option<llkv_expr::expr::Expr<'static, String>>,
}

/// Value to assign in an UPDATE.
#[derive(Clone, Debug)]
pub enum AssignmentValue {
    Literal(PlanValue),
    Expression(llkv_expr::expr::ScalarExpr<String>),
}

/// Column assignment for UPDATE.
#[derive(Clone, Debug)]
pub struct ColumnAssignment {
    pub column: String,
    pub value: AssignmentValue,
}

// ============================================================================
// DELETE Plan
// ============================================================================

/// Plan for deleting rows from a table.
#[derive(Clone, Debug)]
pub struct DeletePlan {
    pub table: String,
    pub filter: Option<llkv_expr::expr::Expr<'static, String>>,
}

// ============================================================================
// TRUNCATE Plan
// ============================================================================

/// Plan for TRUNCATE TABLE operation (removes all rows).
#[derive(Clone, Debug)]
pub struct TruncatePlan {
    pub table: String,
}

// ============================================================================
// SELECT Plan
// ============================================================================

/// Table reference in FROM clause.
#[derive(Clone, Debug)]
pub struct TableRef {
    pub schema: String,
    pub table: String,
    pub alias: Option<String>,
}

impl TableRef {
    pub fn new(schema: impl Into<String>, table: impl Into<String>) -> Self {
        Self {
            schema: schema.into(),
            table: table.into(),
            alias: None,
        }
    }

    pub fn with_alias(
        schema: impl Into<String>,
        table: impl Into<String>,
        alias: Option<String>,
    ) -> Self {
        Self {
            schema: schema.into(),
            table: table.into(),
            alias,
        }
    }

    /// Preferred display name for the table (alias if present).
    pub fn display_name(&self) -> String {
        self.alias
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.qualified_name())
    }

    pub fn qualified_name(&self) -> String {
        if self.schema.is_empty() {
            self.table.clone()
        } else {
            format!("{}.{}", self.schema, self.table)
        }
    }
}

// ============================================================================
// Join Metadata
// ============================================================================

/// Type of join operation for query planning.
///
/// This is a plan-layer type that mirrors `llkv_join::JoinType` but exists
/// separately to avoid circular dependencies (llkv-join depends on llkv-table
/// which depends on llkv-plan). The executor converts `JoinPlan` to `llkv_join::JoinType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinPlan {
    /// Emit only matching row pairs.
    Inner,
    /// Emit all left rows; unmatched left rows have NULL right columns.
    Left,
    /// Emit all right rows; unmatched right rows have NULL left columns.
    Right,
    /// Emit all rows from both sides; unmatched rows have NULLs.
    Full,
}

/// Metadata describing a join between consecutive tables in the FROM clause.
///
/// Tracks the join type and optional ON condition filter for each join.
/// The join connects table at index `left_table_index` with `left_table_index + 1`.
/// Replaces the older `join_types`/`join_filters` vectors so executors can
/// inspect a single compact structure when coordinating join evaluation.
#[derive(Clone, Debug)]
pub struct JoinMetadata {
    /// Index of the left table in the `SelectPlan.tables` vector.
    pub left_table_index: usize,
    /// Type of join (INNER, LEFT, RIGHT, etc.).
    pub join_type: JoinPlan,
    /// Optional ON condition filter expression. Translators also thread this
    /// predicate through [`SelectPlan::filter`] so the optimizer can merge it
    /// with other WHERE clauses, but keeping it here enables join-specific
    /// rewrites (e.g., push-down or hash join pruning).
    pub on_condition: Option<llkv_expr::expr::Expr<'static, String>>,
}

/// Logical query plan for SELECT operations.
///
/// The `tables` collection preserves the FROM clause order while [`Self::joins`]
/// captures how adjacent tables are connected via [`JoinMetadata`]. This keeps
/// join semantics alongside table references instead of parallel vectors and
/// mirrors what the executor expects when materialising join pipelines.
#[derive(Clone, Debug)]
pub struct SelectPlan {
    /// Tables to query. Empty vec means no FROM clause (e.g., SELECT 42).
    /// Single element for simple queries, multiple for joins/cross products.
    pub tables: Vec<TableRef>,
    /// Join metadata describing how tables are joined.
    /// If empty, all tables are implicitly cross-joined (Cartesian product).
    /// Each entry describes a join between `tables[i]` and `tables[i + 1]`.
    pub joins: Vec<JoinMetadata>,
    pub projections: Vec<SelectProjection>,
    /// Optional WHERE predicate plus dependent correlated subqueries.
    pub filter: Option<SelectFilter>,
    /// Optional HAVING predicate applied after grouping.
    pub having: Option<llkv_expr::expr::Expr<'static, String>>,
    /// Scalar subqueries referenced by projections, keyed by `SubqueryId`.
    pub scalar_subqueries: Vec<ScalarSubquery>,
    pub aggregates: Vec<AggregateExpr>,
    pub order_by: Vec<OrderByPlan>,
    pub distinct: bool,
    /// Optional compound (set-operation) plan.
    pub compound: Option<CompoundSelectPlan>,
    /// Columns used in GROUP BY clauses (canonical names).
    pub group_by: Vec<String>,
    /// Optional value table output mode (BigQuery style).
    pub value_table_mode: Option<ValueTableMode>,
}

impl SelectPlan {
    /// Create a SelectPlan for a single table.
    pub fn new(table: impl Into<String>) -> Self {
        let table_name = table.into();
        let tables = if table_name.is_empty() {
            Vec::new()
        } else {
            // Parse "schema.table" or just "table"
            let parts: Vec<&str> = table_name.split('.').collect();
            if parts.len() >= 2 {
                let table_part = parts[1..].join(".");
                vec![TableRef::new(parts[0], table_part)]
            } else {
                vec![TableRef::new("", table_name)]
            }
        };

        Self {
            tables,
            joins: Vec::new(),
            projections: Vec::new(),
            filter: None,
            having: None,
            scalar_subqueries: Vec::new(),
            aggregates: Vec::new(),
            order_by: Vec::new(),
            distinct: false,
            compound: None,
            group_by: Vec::new(),
            value_table_mode: None,
        }
    }

    /// Create a SelectPlan with multiple tables for cross product/joins.
    ///
    /// The returned plan leaves [`Self::joins`] empty, which means any
    /// evaluation engine should treat the tables as a Cartesian product until
    /// [`Self::with_joins`] populates concrete join relationships.
    pub fn with_tables(tables: Vec<TableRef>) -> Self {
        Self {
            tables,
            joins: Vec::new(),
            projections: Vec::new(),
            filter: None,
            having: None,
            scalar_subqueries: Vec::new(),
            aggregates: Vec::new(),
            order_by: Vec::new(),
            distinct: false,
            compound: None,
            group_by: Vec::new(),
            value_table_mode: None,
        }
    }

    pub fn with_projections(mut self, projections: Vec<SelectProjection>) -> Self {
        self.projections = projections;
        self
    }

    pub fn with_filter(mut self, filter: Option<SelectFilter>) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_having(mut self, having: Option<llkv_expr::expr::Expr<'static, String>>) -> Self {
        self.having = having;
        self
    }

    /// Attach scalar subqueries discovered during SELECT translation.
    pub fn with_scalar_subqueries(mut self, scalar_subqueries: Vec<ScalarSubquery>) -> Self {
        self.scalar_subqueries = scalar_subqueries;
        self
    }

    pub fn with_aggregates(mut self, aggregates: Vec<AggregateExpr>) -> Self {
        self.aggregates = aggregates;
        self
    }

    pub fn with_order_by(mut self, order_by: Vec<OrderByPlan>) -> Self {
        self.order_by = order_by;
        self
    }

    pub fn with_distinct(mut self, distinct: bool) -> Self {
        self.distinct = distinct;
        self
    }

    /// Attach join metadata describing how tables are connected.
    ///
    /// Each [`JoinMetadata`] entry pairs `tables[i]` with `tables[i + 1]`. The
    /// builder should supply exactly `tables.len().saturating_sub(1)` entries
    /// when explicit joins are required; otherwise consumers fall back to a
    /// Cartesian product.
    pub fn with_joins(mut self, joins: Vec<JoinMetadata>) -> Self {
        self.joins = joins;
        self
    }

    /// Attach a compound (set operation) plan.
    pub fn with_compound(mut self, compound: CompoundSelectPlan) -> Self {
        self.compound = Some(compound);
        self
    }

    pub fn with_group_by(mut self, group_by: Vec<String>) -> Self {
        self.group_by = group_by;
        self
    }

    pub fn with_value_table_mode(mut self, mode: Option<ValueTableMode>) -> Self {
        self.value_table_mode = mode;
        self
    }
}

/// Set operation applied between SELECT statements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompoundOperator {
    Union,
    Intersect,
    Except,
}

/// Quantifier associated with set operations (e.g., UNION vs UNION ALL).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompoundQuantifier {
    Distinct,
    All,
}

/// Component of a compound SELECT (set operation).
#[derive(Clone, Debug)]
pub struct CompoundSelectComponent {
    pub operator: CompoundOperator,
    pub quantifier: CompoundQuantifier,
    pub plan: SelectPlan,
}

/// Compound SELECT plan representing a tree of set operations.
#[derive(Clone, Debug)]
pub struct CompoundSelectPlan {
    pub initial: Box<SelectPlan>,
    pub operations: Vec<CompoundSelectComponent>,
}

impl CompoundSelectPlan {
    pub fn new(initial: SelectPlan) -> Self {
        Self {
            initial: Box::new(initial),
            operations: Vec::new(),
        }
    }

    pub fn push_operation(
        &mut self,
        operator: CompoundOperator,
        quantifier: CompoundQuantifier,
        plan: SelectPlan,
    ) {
        self.operations.push(CompoundSelectComponent {
            operator,
            quantifier,
            plan,
        });
    }
}

/// Projection specification for SELECT.
#[derive(Clone, Debug)]
pub enum SelectProjection {
    AllColumns,
    AllColumnsExcept {
        exclude: Vec<String>,
    },
    Column {
        name: String,
        alias: Option<String>,
    },
    Computed {
        expr: llkv_expr::expr::ScalarExpr<String>,
        alias: String,
    },
}

/// Value table output modes (BigQuery-style).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueTableMode {
    AsStruct,
    AsValue,
    DistinctAsStruct,
    DistinctAsValue,
}

// ============================================================================
// Aggregate Plans
// ============================================================================

/// Aggregate expression in SELECT.
#[derive(Clone, Debug)]
pub enum AggregateExpr {
    CountStar {
        alias: String,
        distinct: bool,
    },
    Column {
        column: String,
        alias: String,
        function: AggregateFunction,
        distinct: bool,
    },
}

/// Supported aggregate functions.
#[derive(Clone, Debug)]
pub enum AggregateFunction {
    Count,
    SumInt64,
    TotalInt64,
    MinInt64,
    MaxInt64,
    CountNulls,
    GroupConcat,
}

impl AggregateExpr {
    pub fn count_star(alias: impl Into<String>, distinct: bool) -> Self {
        Self::CountStar {
            alias: alias.into(),
            distinct,
        }
    }

    pub fn count_column(
        column: impl Into<String>,
        alias: impl Into<String>,
        distinct: bool,
    ) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::Count,
            distinct,
        }
    }

    pub fn sum_int64(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::SumInt64,
            distinct: false,
        }
    }

    pub fn total_int64(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::TotalInt64,
            distinct: false,
        }
    }

    pub fn min_int64(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::MinInt64,
            distinct: false,
        }
    }

    pub fn max_int64(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::MaxInt64,
            distinct: false,
        }
    }

    pub fn count_nulls(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::CountNulls,
            distinct: false,
        }
    }
}

/// Helper to convert an Arrow array cell into a plan-level Value.
pub fn plan_value_from_array(array: &ArrayRef, index: usize) -> PlanResult<PlanValue> {
    if array.is_null(index) {
        return Ok(PlanValue::Null);
    }
    match array.data_type() {
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Boolean array in INSERT SELECT".into())
                })?;
            Ok(PlanValue::Integer(if values.value(index) { 1 } else { 0 }))
        }
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                Error::InvalidArgumentError("expected Int64 array in INSERT SELECT".into())
            })?;
            Ok(PlanValue::Integer(values.value(index)))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Float64 array in INSERT SELECT".into())
                })?;
            Ok(PlanValue::Float(values.value(index)))
        }
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Utf8 array in INSERT SELECT".into())
                })?;
            Ok(PlanValue::String(values.value(index).to_string()))
        }
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Date32 array in INSERT SELECT".into())
                })?;
            Ok(PlanValue::Integer(values.value(index) as i64))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported data type in INSERT SELECT: {other:?}"
        ))),
    }
}

// ============================================================================
// ORDER BY Plan
// ============================================================================

/// ORDER BY specification.
#[derive(Clone, Debug)]
pub struct OrderByPlan {
    pub target: OrderTarget,
    pub sort_type: OrderSortType,
    pub ascending: bool,
    pub nulls_first: bool,
}

/// Sort type for ORDER BY.
#[derive(Clone, Debug)]
pub enum OrderSortType {
    Native,
    CastTextToInteger,
}

/// Target column/expression for ORDER BY.
#[derive(Clone, Debug)]
pub enum OrderTarget {
    Column(String),
    Index(usize),
    All,
}

// ============================================================================
// Operation Enum for Transaction Replay
// ============================================================================

/// Recordable plan operation for transaction replay.
#[derive(Clone, Debug)]
pub enum PlanOperation {
    CreateTable(CreateTablePlan),
    DropTable(DropTablePlan),
    Insert(InsertPlan),
    Update(UpdatePlan),
    Delete(DeletePlan),
    Truncate(TruncatePlan),
    Select(Box<SelectPlan>),
}

/// Top-level plan statements that can be executed against a `Session`.
#[derive(Clone, Debug)]
pub enum PlanStatement {
    BeginTransaction,
    CommitTransaction,
    RollbackTransaction,
    CreateTable(CreateTablePlan),
    DropTable(DropTablePlan),
    CreateView(CreateViewPlan),
    DropView(DropViewPlan),
    DropIndex(DropIndexPlan),
    AlterTable(AlterTablePlan),
    CreateIndex(CreateIndexPlan),
    Reindex(ReindexPlan),
    Insert(InsertPlan),
    Update(UpdatePlan),
    Delete(DeletePlan),
    Truncate(TruncatePlan),
    Select(Box<SelectPlan>),
}
