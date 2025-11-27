//! Conversion utilities from SQL AST to Plan types.
//!
//! This module provides functions for converting sqlparser AST nodes into
//! llkv-plan data structures, particularly for literal value conversion
//! and range SELECT parsing.

use llkv_expr::literal::Literal;
use llkv_result::{Error, Result};
use llkv_types::decimal::DecimalValue;
use rustc_hash::FxHashMap;
use sqlparser::ast::{
    DataType, Expr as SqlExpr, FunctionArg, FunctionArgExpr, GroupByExpr, ObjectName,
    ObjectNamePart, Select, SelectItem, SelectItemQualifiedWildcardKind, TableAlias, TableFactor,
    TypedString, UnaryOperator, Value, ValueWithSpan,
};

use crate::PlanValue;
use llkv_compute::date::parse_date32_literal;
use llkv_compute::interval::parse_interval_literal;

impl PlanValue {
    pub fn from_operator_literal(op_value: &llkv_expr::literal::Literal) -> Option<PlanValue> {
        match op_value {
            Literal::Int128(v) => i64::try_from(*v).ok().map(PlanValue::Integer),
            Literal::Float64(v) => Some(PlanValue::Float(*v)),
            Literal::Boolean(v) => Some(PlanValue::Integer(if *v { 1 } else { 0 })),
            Literal::String(v) => Some(PlanValue::String(v.clone())),
            _ => None,
        }
    }

    pub fn from_literal_for_join(literal: &Literal) -> Option<PlanValue> {
        match literal {
            Literal::Int128(v) => i64::try_from(*v).ok().map(PlanValue::Integer),
            Literal::Float64(v) => Some(PlanValue::Float(*v)),
            Literal::Boolean(v) => Some(PlanValue::Integer(if *v { 1 } else { 0 })),
            Literal::String(v) => Some(PlanValue::String(v.clone())),
            _ => None,
        }
    }

    /// Convert a SQL expression to a PlanValue literal.
    pub fn from_sql_expr(expr: &SqlExpr) -> Result<Self> {
        match expr {
            SqlExpr::Value(value) => Self::from_sql_value(value),
            SqlExpr::TypedString(typed) => Self::from_typed_string(typed),
            SqlExpr::Interval(interval) => {
                let parsed = parse_interval_literal(interval)?;
                Ok(PlanValue::Interval(parsed))
            }
            SqlExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => match Self::from_sql_expr(expr)? {
                PlanValue::Integer(v) => Ok(PlanValue::Integer(-v)),
                PlanValue::Float(v) => Ok(PlanValue::Float(-v)),
                PlanValue::Decimal(value) => {
                    let negated = value.raw_value().checked_neg().ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "decimal literal overflow when applying unary minus".into(),
                        )
                    })?;
                    let decimal = DecimalValue::new(negated, value.scale()).map_err(|err| {
                        Error::InvalidArgumentError(format!(
                            "failed to negate decimal literal: {err}"
                        ))
                    })?;
                    Ok(PlanValue::Decimal(decimal))
                }
                PlanValue::Null
                | PlanValue::String(_)
                | PlanValue::Struct(_)
                | PlanValue::Date32(_)
                | PlanValue::Interval(_) => Err(Error::InvalidArgumentError(
                    "cannot negate non-numeric literal".into(),
                )),
            },
            SqlExpr::UnaryOp {
                op: UnaryOperator::Plus,
                expr,
            } => Self::from_sql_expr(expr),
            SqlExpr::Nested(inner) => Self::from_sql_expr(inner),
            SqlExpr::Dictionary(fields) => {
                let mut map = FxHashMap::with_capacity_and_hasher(fields.len(), Default::default());
                for field in fields {
                    let key = field.key.value.clone();
                    let value = Self::from_sql_expr(&field.value)?;
                    map.insert(key, value);
                }
                Ok(PlanValue::Struct(map))
            }
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported literal expression: {other:?}"
            ))),
        }
    }

    /// Convert a SQL value literal to a PlanValue.
    pub fn from_sql_value(value: &ValueWithSpan) -> Result<Self> {
        match &value.value {
            Value::Null => Ok(PlanValue::Null),
            Value::Number(text, _) => {
                if text.contains(['.', 'e', 'E']) {
                    let parsed = text.parse::<f64>().map_err(|err| {
                        Error::InvalidArgumentError(format!("invalid float literal: {err}"))
                    })?;
                    Ok(PlanValue::Float(parsed))
                } else {
                    let parsed = text.parse::<i64>().map_err(|err| {
                        Error::InvalidArgumentError(format!("invalid integer literal: {err}"))
                    })?;
                    Ok(PlanValue::Integer(parsed))
                }
            }
            Value::Boolean(_) => Err(Error::InvalidArgumentError(
                "BOOLEAN literals are not supported yet".into(),
            )),
            other => {
                if let Some(text) = other.clone().into_string() {
                    Ok(PlanValue::String(text))
                } else {
                    Err(Error::InvalidArgumentError(format!(
                        "unsupported literal: {other:?}"
                    )))
                }
            }
        }
    }

    /// Convert a typed string literal to a PlanValue.
    fn from_typed_string(typed: &TypedString) -> Result<Self> {
        let text = typed.value.value.clone().into_string().ok_or_else(|| {
            Error::InvalidArgumentError("typed string literal must be a quoted string".into())
        })?;

        match typed.data_type {
            DataType::Date => {
                let days = parse_date32_literal(&text)?;
                Ok(PlanValue::Date32(days))
            }
            _ => Ok(PlanValue::String(text)),
        }
    }
}

// ============================================================================
// Range SELECT Support
// ============================================================================

/// Result of parsing a range() SELECT statement.
#[derive(Clone)]
pub struct RangeSelectRows {
    rows: Vec<Vec<PlanValue>>,
}

impl RangeSelectRows {
    /// Convert into the underlying rows vector.
    pub fn into_rows(self) -> Vec<Vec<PlanValue>> {
        self.rows
    }
}

#[derive(Clone)]
enum RangeProjection {
    Column,
    Literal(PlanValue),
}

#[derive(Clone)]
struct RangeSpec {
    start: i64,
    #[allow(dead_code)] // Used for validation, computed into row_count
    end: i64,
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

/// Extract rows from a range() SELECT statement.
///
/// Parses SELECT statements of the form:
/// - `SELECT * FROM range(10)`
/// - `SELECT * FROM range(5, 15)`  
/// - `SELECT range, range * 2 FROM range(100)`
///
/// Returns `None` if the SELECT is not a range() query.
///
/// # Errors
///
/// Returns an error if:
/// - The range() function has invalid arguments
/// - Unsupported SELECT clauses are used (WHERE, JOIN, etc.)
/// - Column references don't match the range source
pub fn extract_rows_from_range(select: &Select) -> Result<Option<RangeSelectRows>> {
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

    // If projection is empty, treat it as SELECT * (implicit wildcard)
    if select.projection.is_empty() {
        projections.push(RangeProjection::Column);
    } else {
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
    }

    let mut rows: Vec<Vec<PlanValue>> = Vec::with_capacity(spec.row_count);
    for idx in 0..spec.row_count {
        let mut row: Vec<PlanValue> = Vec::with_capacity(projections.len());
        let value = spec.start + (idx as i64);
        for projection in &projections {
            match projection {
                RangeProjection::Column => row.push(PlanValue::Integer(value)),
                RangeProjection::Literal(value) => row.push(value.clone()),
            }
        }
        rows.push(row);
    }

    Ok(Some(RangeSelectRows { rows }))
}

fn build_range_projection_expr(expr: &SqlExpr, spec: &RangeSpec) -> Result<RangeProjection> {
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
        other => Ok(RangeProjection::Literal(PlanValue::from_sql_expr(other)?)),
    }
}

fn parse_range_spec(select: &Select) -> Result<Option<RangeSpec>> {
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
) -> Result<Option<RangeSpec>> {
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

    if args.is_empty() || args.len() > 2 {
        return Err(Error::InvalidArgumentError(
            "range() requires one or two arguments".into(),
        ));
    }

    // Helper to extract integer from argument
    let extract_int = |arg: &FunctionArg| -> Result<i64> {
        let arg_expr = match arg {
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

        let value = PlanValue::from_sql_expr(arg_expr)?;
        match value {
            PlanValue::Integer(v) => Ok(v),
            _ => Err(Error::InvalidArgumentError(
                "range() argument must be an integer literal".into(),
            )),
        }
    };

    let (start, end, row_count) = if args.len() == 1 {
        // range(count) - generate [0, count)
        let count = extract_int(&args[0])?;
        if count < 0 {
            return Err(Error::InvalidArgumentError(
                "range() argument must be non-negative".into(),
            ));
        }
        (0, count, count as usize)
    } else {
        // range(start, end) - generate [start, end)
        let start = extract_int(&args[0])?;
        let end = extract_int(&args[1])?;
        if end < start {
            return Err(Error::InvalidArgumentError(
                "range() end must be >= start".into(),
            ));
        }
        let row_count = (end - start) as usize;
        (start, end, row_count)
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
        start,
        end,
        row_count,
        column_name_lower,
        table_alias_lower,
    }))
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
    use llkv_types::IntervalValue;
    use sqlparser::ast::{Expr as SqlExpr, SelectItem, SetExpr, Statement, Value, ValueWithSpan};
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    fn value_with_span(v: Value) -> ValueWithSpan {
        ValueWithSpan {
            value: v,
            span: sqlparser::tokenizer::Span::empty(),
        }
    }

    #[test]
    fn test_null_value() {
        let value = value_with_span(Value::Null);
        assert_eq!(PlanValue::from_sql_value(&value).unwrap(), PlanValue::Null);
    }

    #[test]
    fn test_integer_value() {
        let value = value_with_span(Value::Number("42".to_string(), false));
        assert_eq!(
            PlanValue::from_sql_value(&value).unwrap(),
            PlanValue::Integer(42)
        );
    }

    #[test]
    fn test_negative_integer() {
        let value = value_with_span(Value::Number("42".to_string(), false));
        let expr = SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr: Box::new(SqlExpr::Value(value)),
        };
        assert_eq!(
            PlanValue::from_sql_expr(&expr).unwrap(),
            PlanValue::Integer(-42)
        );
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_float_value() {
        let value = value_with_span(Value::Number("3.14".to_string(), false));
        assert_eq!(
            PlanValue::from_sql_value(&value).unwrap(),
            PlanValue::Float(3.14)
        );
    }

    #[test]
    fn test_string_value() {
        let value = value_with_span(Value::SingleQuotedString("hello".to_string()));
        assert_eq!(
            PlanValue::from_sql_value(&value).unwrap(),
            PlanValue::String("hello".to_string())
        );
    }

    #[test]
    fn test_nested_expression() {
        let value = value_with_span(Value::Number("100".to_string(), false));
        let expr = SqlExpr::Nested(Box::new(SqlExpr::Value(value)));
        assert_eq!(
            PlanValue::from_sql_expr(&expr).unwrap(),
            PlanValue::Integer(100)
        );
    }

    #[test]
    fn test_plus_operator() {
        let value = value_with_span(Value::Number("50".to_string(), false));
        let expr = SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr: Box::new(SqlExpr::Value(value)),
        };
        assert_eq!(
            PlanValue::from_sql_expr(&expr).unwrap(),
            PlanValue::Integer(50)
        );
    }

    #[test]
    fn test_interval_literal_expression() {
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(&dialect, "SELECT INTERVAL '7' DAY").unwrap();
        let statement = statements.first().expect("statement");
        let expr = match statement {
            Statement::Query(query) => match query.body.as_ref() {
                SetExpr::Select(select) => match select.projection.first().expect("projection") {
                    SelectItem::UnnamedExpr(expr) => expr,
                    other => panic!("unexpected projection {other:?}"),
                },
                other => panic!("unexpected set expr {other:?}"),
            },
            other => panic!("unexpected statement {other:?}"),
        };

        assert_eq!(
            PlanValue::from_sql_expr(expr).unwrap(),
            PlanValue::Interval(IntervalValue::new(0, 7, 0))
        );
    }

    #[test]
    fn test_cannot_negate_string() {
        let value = value_with_span(Value::SingleQuotedString("test".to_string()));
        let expr = SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr: Box::new(SqlExpr::Value(value)),
        };
        assert!(PlanValue::from_sql_expr(&expr).is_err());
    }

    #[test]
    fn test_boolean_not_supported() {
        let value = value_with_span(Value::Boolean(true));
        assert!(PlanValue::from_sql_value(&value).is_err());
    }
}
