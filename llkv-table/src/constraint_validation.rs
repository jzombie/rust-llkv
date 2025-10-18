use arrow::datatypes::DataType;
use llkv_plan::PlanValue;
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use sqlparser::ast::{self, Expr as SqlExpr};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::types::FieldId;

/// Lightweight column descriptor used for constraint validation.
#[derive(Clone, Debug)]
pub struct ConstraintColumnInfo {
    pub name: String,
    pub field_id: FieldId,
    pub data_type: DataType,
    pub nullable: bool,
    pub check_expr: Option<String>,
}

/// Canonical representation of values participating in UNIQUE or PRIMARY KEY checks.
#[derive(Hash, Eq, PartialEq, Debug, Clone)]
pub enum UniqueKey {
    Int(i64),
    Float(u64),
    Str(String),
    Composite(Vec<UniqueKey>),
}

/// Validate all CHECK constraints for the provided rows.
pub fn validate_check_constraints(
    columns: &[ConstraintColumnInfo],
    rows: &[Vec<PlanValue>],
    column_order: &[usize],
) -> LlkvResult<()> {
    if rows.is_empty() {
        return Ok(());
    }

    let dialect = GenericDialect {};

    let mut parsed_checks: Vec<(usize, String, String, SqlExpr)> = Vec::new();

    for (idx, column) in columns.iter().enumerate() {
        if let Some(expr_str) = &column.check_expr {
            let expr = parse_check_expression(&dialect, expr_str)?;
            parsed_checks.push((idx, column.name.clone(), expr_str.clone(), expr));
        }
    }

    if parsed_checks.is_empty() {
        return Ok(());
    }

    let mut name_lookup: FxHashMap<String, usize> = FxHashMap::default();
    for (idx, column) in columns.iter().enumerate() {
        name_lookup.insert(column.name.to_ascii_lowercase(), idx);
    }

    for row in rows {
        for (_schema_idx, column_name, expr_str, expr) in &parsed_checks {
            let result = evaluate_check_expression(expr, row, column_order, columns, &name_lookup)?;

            if !result {
                return Err(Error::ConstraintError(format!(
                    "CHECK constraint failed for column '{}': {}",
                    column_name, expr_str
                )));
            }
        }
    }

    Ok(())
}

/// Ensure that the provided multi-column values maintain uniqueness.
pub fn ensure_multi_column_unique(
    existing_rows: &[Vec<PlanValue>],
    new_rows: &[Vec<PlanValue>],
    column_names: &[String],
) -> LlkvResult<()> {
    let mut existing_keys: FxHashSet<UniqueKey> = FxHashSet::default();
    for values in existing_rows {
        if let Some(key) = build_composite_unique_key(values, column_names)? {
            existing_keys.insert(key);
        }
    }

    let mut new_keys: FxHashSet<UniqueKey> = FxHashSet::default();
    for values in new_rows {
        if let Some(key) = build_composite_unique_key(values, column_names)? {
            if existing_keys.contains(&key) || !new_keys.insert(key) {
                return Err(Error::ConstraintError(format!(
                    "constraint violation on columns '{}'",
                    column_names.join(", ")
                )));
            }
        }
    }

    Ok(())
}

/// Build a unique key component for a single value.
pub fn unique_key_component(value: &PlanValue, column_name: &str) -> LlkvResult<Option<UniqueKey>> {
    match value {
        PlanValue::Null => Ok(None),
        PlanValue::Integer(v) => Ok(Some(UniqueKey::Int(*v))),
        PlanValue::Float(v) => Ok(Some(UniqueKey::Float(v.to_bits()))),
        PlanValue::String(s) => Ok(Some(UniqueKey::Str(s.clone()))),
        PlanValue::Struct(_) => Err(Error::InvalidArgumentError(format!(
            "UNIQUE index is not supported on struct column '{}'",
            column_name
        ))),
    }
}

/// Build a composite unique key from column values.
pub fn build_composite_unique_key(
    values: &[PlanValue],
    column_names: &[String],
) -> LlkvResult<Option<UniqueKey>> {
    if values.is_empty() {
        return Ok(None);
    }

    let mut components = Vec::with_capacity(values.len());
    for (value, column_name) in values.iter().zip(column_names) {
        match unique_key_component(value, column_name)? {
            Some(component) => components.push(component),
            None => return Ok(None),
        }
    }

    Ok(Some(UniqueKey::Composite(components)))
}

fn parse_check_expression(dialect: &GenericDialect, check_expr_str: &str) -> LlkvResult<SqlExpr> {
    let sql = format!("SELECT {}", check_expr_str);
    let mut ast = Parser::parse_sql(dialect, &sql).map_err(|e| {
        Error::InvalidArgumentError(format!(
            "Failed to parse CHECK expression '{}': {}",
            check_expr_str, e
        ))
    })?;

    let stmt = ast.pop().ok_or_else(|| {
        Error::InvalidArgumentError(format!(
            "CHECK expression '{}' resulted in empty AST",
            check_expr_str
        ))
    })?;

    let query = match stmt {
        ast::Statement::Query(q) => q,
        _ => {
            return Err(Error::InvalidArgumentError(format!(
                "CHECK expression '{}' did not parse as SELECT",
                check_expr_str
            )));
        }
    };

    let body = match *query.body {
        ast::SetExpr::Select(s) => s,
        _ => {
            return Err(Error::InvalidArgumentError(format!(
                "CHECK expression '{}' is not a simple SELECT",
                check_expr_str
            )));
        }
    };

    if body.projection.len() != 1 {
        return Err(Error::InvalidArgumentError(format!(
            "CHECK expression '{}' must have exactly one projection",
            check_expr_str
        )));
    }

    match &body.projection[0] {
        ast::SelectItem::UnnamedExpr(expr) | ast::SelectItem::ExprWithAlias { expr, .. } => {
            Ok(expr.clone())
        }
        _ => Err(Error::InvalidArgumentError(format!(
            "CHECK expression '{}' projection is not a simple expression",
            check_expr_str
        ))),
    }
}

fn evaluate_check_expression(
    expr: &SqlExpr,
    row: &[PlanValue],
    column_order: &[usize],
    columns: &[ConstraintColumnInfo],
    name_lookup: &FxHashMap<String, usize>,
) -> LlkvResult<bool> {
    use sqlparser::ast::BinaryOperator;

    match expr {
        SqlExpr::BinaryOp { left, op, right } => {
            let left_val =
                evaluate_check_expr_value(left, row, column_order, columns, name_lookup)?;
            let right_val =
                evaluate_check_expr_value(right, row, column_order, columns, name_lookup)?;

            match op {
                BinaryOperator::Eq => {
                    if matches!(left_val, PlanValue::Null) || matches!(right_val, PlanValue::Null) {
                        Ok(true)
                    } else {
                        Ok(left_val == right_val)
                    }
                }
                BinaryOperator::NotEq => {
                    if matches!(left_val, PlanValue::Null) || matches!(right_val, PlanValue::Null) {
                        Ok(true)
                    } else {
                        Ok(left_val != right_val)
                    }
                }
                BinaryOperator::Lt => compare_numeric(&left_val, &right_val, |l, r| l < r),
                BinaryOperator::LtEq => compare_numeric(&left_val, &right_val, |l, r| l <= r),
                BinaryOperator::Gt => compare_numeric(&left_val, &right_val, |l, r| l > r),
                BinaryOperator::GtEq => compare_numeric(&left_val, &right_val, |l, r| l >= r),
                _ => Err(Error::InvalidArgumentError(format!(
                    "Unsupported operator in CHECK constraint: {:?}",
                    op
                ))),
            }
        }
        SqlExpr::IsNull(inner) => {
            let value = evaluate_check_expr_value(inner, row, column_order, columns, name_lookup)?;
            Ok(matches!(value, PlanValue::Null))
        }
        SqlExpr::IsNotNull(inner) => {
            let value = evaluate_check_expr_value(inner, row, column_order, columns, name_lookup)?;
            Ok(!matches!(value, PlanValue::Null))
        }
        SqlExpr::Nested(inner) => {
            evaluate_check_expression(inner, row, column_order, columns, name_lookup)
        }
        _ => Err(Error::InvalidArgumentError(format!(
            "Unsupported expression in CHECK constraint: {:?}",
            expr
        ))),
    }
}

fn evaluate_check_expr_value(
    expr: &SqlExpr,
    row: &[PlanValue],
    column_order: &[usize],
    columns: &[ConstraintColumnInfo],
    name_lookup: &FxHashMap<String, usize>,
) -> LlkvResult<PlanValue> {
    use sqlparser::ast::{BinaryOperator, Expr as SqlExpr};

    match expr {
        SqlExpr::BinaryOp { left, op, right } => {
            let left_val =
                evaluate_check_expr_value(left, row, column_order, columns, name_lookup)?;
            let right_val =
                evaluate_check_expr_value(right, row, column_order, columns, name_lookup)?;

            match op {
                BinaryOperator::Plus => apply_numeric_op(left_val, right_val, |l, r| l + r),
                BinaryOperator::Minus => apply_numeric_op(left_val, right_val, |l, r| l - r),
                BinaryOperator::Multiply => apply_numeric_op(left_val, right_val, |l, r| l * r),
                BinaryOperator::Divide => divide_numeric(left_val, right_val),
                _ => Err(Error::InvalidArgumentError(format!(
                    "Unsupported binary operator in CHECK constraint value expression: {:?}",
                    op
                ))),
            }
        }
        SqlExpr::Identifier(ident) => {
            let column_idx = lookup_column_index(name_lookup, &ident.value)?;
            extract_row_value(row, column_order, column_idx, &ident.value)
        }
        SqlExpr::CompoundIdentifier(idents) => {
            if idents.len() == 2 {
                let column_name = &idents[0].value;
                let field_name = &idents[1].value;
                let column_idx = lookup_column_index(name_lookup, column_name)?;
                let value = extract_row_value(row, column_order, column_idx, column_name)?;
                extract_struct_field(value, column_name, field_name)
            } else if idents.len() == 3 {
                let column_name = &idents[1].value;
                let field_name = &idents[2].value;
                let column_idx = lookup_column_index(name_lookup, column_name)?;
                let value = extract_row_value(row, column_order, column_idx, column_name)?;
                extract_struct_field(value, column_name, field_name)
            } else {
                Err(Error::InvalidArgumentError(format!(
                    "Unsupported compound identifier in CHECK constraint: {} parts",
                    idents.len()
                )))
            }
        }
        SqlExpr::Value(val_with_span) => match &val_with_span.value {
            ast::Value::Number(n, _) => {
                if let Ok(i) = n.parse::<i64>() {
                    Ok(PlanValue::Integer(i))
                } else if let Ok(f) = n.parse::<f64>() {
                    Ok(PlanValue::Float(f))
                } else {
                    Err(Error::InvalidArgumentError(format!(
                        "Invalid number in CHECK constraint: {}",
                        n
                    )))
                }
            }
            ast::Value::SingleQuotedString(s) | ast::Value::DoubleQuotedString(s) => {
                Ok(PlanValue::String(s.clone()))
            }
            ast::Value::Null => Ok(PlanValue::Null),
            _ => Err(Error::InvalidArgumentError(format!(
                "Unsupported value type in CHECK constraint: {:?}",
                val_with_span.value
            ))),
        },
        SqlExpr::Nested(inner) => {
            evaluate_check_expr_value(inner, row, column_order, columns, name_lookup)
        }
        _ => Err(Error::InvalidArgumentError(format!(
            "Unsupported expression type in CHECK constraint: {:?}",
            expr
        ))),
    }
}

fn lookup_column_index(
    name_lookup: &FxHashMap<String, usize>,
    column_name: &str,
) -> LlkvResult<usize> {
    name_lookup
        .get(&column_name.to_ascii_lowercase())
        .copied()
        .ok_or_else(|| {
            Error::InvalidArgumentError(format!(
                "Unknown column '{}' in CHECK constraint",
                column_name
            ))
        })
}

fn extract_row_value(
    row: &[PlanValue],
    column_order: &[usize],
    schema_idx: usize,
    column_name: &str,
) -> LlkvResult<PlanValue> {
    let insert_pos = column_order
        .iter()
        .position(|&dest_idx| dest_idx == schema_idx)
        .ok_or_else(|| {
            Error::InvalidArgumentError(format!("Column '{}' not provided in INSERT", column_name))
        })?;

    Ok(row[insert_pos].clone())
}

fn extract_struct_field(
    value: PlanValue,
    column_name: &str,
    field_name: &str,
) -> LlkvResult<PlanValue> {
    match value {
        PlanValue::Struct(fields) => fields
            .into_iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(field_name))
            .map(|(_, val)| val)
            .ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "Struct field '{}' not found in column '{}'",
                    field_name, column_name
                ))
            }),
        _ => Err(Error::InvalidArgumentError(format!(
            "Column '{}' is not a struct, cannot access field '{}'",
            column_name, field_name
        ))),
    }
}

fn compare_numeric<F>(left: &PlanValue, right: &PlanValue, compare: F) -> LlkvResult<bool>
where
    F: Fn(f64, f64) -> bool,
{
    if matches!(left, PlanValue::Null) || matches!(right, PlanValue::Null) {
        return Ok(true);
    }

    match (left, right) {
        (PlanValue::Integer(l), PlanValue::Integer(r)) => Ok(compare(*l as f64, *r as f64)),
        (PlanValue::Float(l), PlanValue::Float(r)) => Ok(compare(*l, *r)),
        (PlanValue::Integer(l), PlanValue::Float(r)) => Ok(compare(*l as f64, *r)),
        (PlanValue::Float(l), PlanValue::Integer(r)) => Ok(compare(*l, *r as f64)),
        _ => Err(Error::InvalidArgumentError(
            "CHECK constraint comparison requires numeric values".into(),
        )),
    }
}

fn apply_numeric_op(
    left: PlanValue,
    right: PlanValue,
    op: fn(f64, f64) -> f64,
) -> LlkvResult<PlanValue> {
    if matches!(left, PlanValue::Null) || matches!(right, PlanValue::Null) {
        return Ok(PlanValue::Null);
    }

    match (left, right) {
        (PlanValue::Integer(l), PlanValue::Integer(r)) => {
            let result = op(l as f64, r as f64);
            if result.fract() == 0.0 {
                Ok(PlanValue::Integer(result as i64))
            } else {
                Ok(PlanValue::Float(result))
            }
        }
        (PlanValue::Float(l), PlanValue::Float(r)) => Ok(PlanValue::Float(op(l, r))),
        (PlanValue::Integer(l), PlanValue::Float(r)) => Ok(PlanValue::Float(op(l as f64, r))),
        (PlanValue::Float(l), PlanValue::Integer(r)) => Ok(PlanValue::Float(op(l, r as f64))),
        _ => Err(Error::InvalidArgumentError(
            "CHECK constraint arithmetic requires numeric values".into(),
        )),
    }
}

fn divide_numeric(left: PlanValue, right: PlanValue) -> LlkvResult<PlanValue> {
    if matches!(left, PlanValue::Null) || matches!(right, PlanValue::Null) {
        return Ok(PlanValue::Null);
    }

    match (left, right) {
        (PlanValue::Integer(l), PlanValue::Integer(r)) => {
            if r == 0 {
                Err(Error::InvalidArgumentError(
                    "Division by zero in CHECK constraint".into(),
                ))
            } else {
                Ok(PlanValue::Integer(l / r))
            }
        }
        (PlanValue::Float(l), PlanValue::Float(r)) => {
            if r == 0.0 {
                Err(Error::InvalidArgumentError(
                    "Division by zero in CHECK constraint".into(),
                ))
            } else {
                Ok(PlanValue::Float(l / r))
            }
        }
        (PlanValue::Integer(l), PlanValue::Float(r)) => {
            if r == 0.0 {
                Err(Error::InvalidArgumentError(
                    "Division by zero in CHECK constraint".into(),
                ))
            } else {
                Ok(PlanValue::Float(l as f64 / r))
            }
        }
        (PlanValue::Float(l), PlanValue::Integer(r)) => {
            if r == 0 {
                Err(Error::InvalidArgumentError(
                    "Division by zero in CHECK constraint".into(),
                ))
            } else {
                Ok(PlanValue::Float(l / r as f64))
            }
        }
        _ => Err(Error::InvalidArgumentError(
            "CHECK constraint / operator requires numeric values".into(),
        )),
    }
}
