use arrow::datatypes::DataType;
use llkv_plan::{ForeignKeyAction as PlanForeignKeyAction, ForeignKeySpec, PlanValue};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::{FxHashMap, FxHashSet};
use sqlparser::ast::{self, Expr as SqlExpr};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use super::types::ForeignKeyAction;
use crate::sys_catalog::MultiColumnUniqueEntryMeta;
use crate::types::{FieldId, TableId};

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
        if let Some(key) = build_composite_unique_key(values, column_names)?
            && !existing_keys.insert(key.clone())
        {
            return Err(Error::ConstraintError(format!(
                "constraint violation on columns '{}'",
                column_names.join(", ")
            )));
        }
    }

    let mut new_keys: FxHashSet<UniqueKey> = FxHashSet::default();
    for values in new_rows {
        if let Some(key) = build_composite_unique_key(values, column_names)?
            && (existing_keys.contains(&key) || !new_keys.insert(key))
        {
            return Err(Error::ConstraintError(format!(
                "constraint violation on columns '{}'",
                column_names.join(", ")
            )));
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

#[allow(clippy::only_used_in_recursion)]
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
        // In SQL, any comparison with NULL yields UNKNOWN.
        // For CHECK constraints, UNKNOWN is treated as TRUE (constraint passes).
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

// ============================================================================
// Foreign key validation
// ============================================================================

/// Column metadata used when validating foreign key definitions.
#[derive(Clone, Debug)]
pub struct ForeignKeyColumn {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
    pub unique: bool,
    pub field_id: FieldId,
}

/// Table metadata used when validating foreign key definitions.
#[derive(Clone, Debug)]
pub struct ForeignKeyTableInfo {
    pub display_name: String,
    pub canonical_name: String,
    pub table_id: TableId,
    pub columns: Vec<ForeignKeyColumn>,
    pub multi_column_uniques: Vec<MultiColumnUniqueEntryMeta>,
}

/// Result of validating a foreign key specification.
#[derive(Clone, Debug)]
pub struct ValidatedForeignKey {
    pub name: Option<String>,
    pub referencing_indices: Vec<usize>,
    pub referencing_field_ids: Vec<FieldId>,
    pub referencing_column_names: Vec<String>,
    pub referenced_table_id: TableId,
    pub referenced_table_display: String,
    pub referenced_table_canonical: String,
    pub referenced_field_ids: Vec<FieldId>,
    pub referenced_column_names: Vec<String>,
    pub on_delete: ForeignKeyAction,
    pub on_update: ForeignKeyAction,
}

/// Validate a set of foreign key specifications against the provided table schemas.
pub fn validate_foreign_keys<F>(
    referencing_table: &ForeignKeyTableInfo,
    specs: &[ForeignKeySpec],
    mut lookup_table: F,
) -> LlkvResult<Vec<ValidatedForeignKey>>
where
    F: FnMut(&str) -> LlkvResult<ForeignKeyTableInfo>,
{
    if specs.is_empty() {
        return Ok(Vec::new());
    }

    let mut referencing_lookup: FxHashMap<String, (usize, &ForeignKeyColumn)> =
        FxHashMap::default();
    for (idx, column) in referencing_table.columns.iter().enumerate() {
        referencing_lookup.insert(column.name.to_ascii_lowercase(), (idx, column));
    }

    let mut results = Vec::with_capacity(specs.len());

    for spec in specs {
        if spec.columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "FOREIGN KEY requires at least one referencing column".into(),
            ));
        }

        let mut seen_referencing = FxHashSet::default();
        let mut referencing_indices = Vec::with_capacity(spec.columns.len());
        let mut referencing_field_ids = Vec::with_capacity(spec.columns.len());
        let mut referencing_column_defs = Vec::with_capacity(spec.columns.len());
        let mut referencing_column_names = Vec::with_capacity(spec.columns.len());

        for column_name in &spec.columns {
            let normalized = column_name.to_ascii_lowercase();
            if !seen_referencing.insert(normalized.clone()) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in FOREIGN KEY constraint",
                    column_name
                )));
            }

            let (idx, column) = referencing_lookup.get(&normalized).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column '{}' in FOREIGN KEY constraint",
                    column_name
                ))
            })?;

            referencing_indices.push(*idx);
            referencing_field_ids.push(column.field_id);
            referencing_column_defs.push((*column).clone());
            referencing_column_names.push(column.name.clone());
        }

        let referenced_table_info = lookup_table(&spec.referenced_table)?;

        let referenced_columns = if spec.referenced_columns.is_empty() {
            referenced_table_info
                .columns
                .iter()
                .filter(|col| col.primary_key)
                .map(|col| col.name.clone())
                .collect::<Vec<_>>()
        } else {
            spec.referenced_columns.clone()
        };

        if referenced_columns.is_empty() {
            return Err(Error::InvalidArgumentError(format!(
                "there is no primary key for referenced table '{}'",
                spec.referenced_table
            )));
        }

        if spec.columns.len() != referenced_columns.len() {
            return Err(Error::InvalidArgumentError(format!(
                "number of referencing columns ({}) does not match number of referenced columns ({})",
                spec.columns.len(),
                referenced_columns.len()
            )));
        }

        let mut seen_referenced = FxHashSet::default();
        let mut referenced_lookup: FxHashMap<String, &ForeignKeyColumn> = FxHashMap::default();
        for column in &referenced_table_info.columns {
            referenced_lookup.insert(column.name.to_ascii_lowercase(), column);
        }

        let mut referenced_field_ids = Vec::with_capacity(referenced_columns.len());
        let mut referenced_column_defs = Vec::with_capacity(referenced_columns.len());
        let mut referenced_column_names = Vec::with_capacity(referenced_columns.len());

        for column_name in referenced_columns.iter() {
            let normalized = column_name.to_ascii_lowercase();
            if !seen_referenced.insert(normalized.clone()) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate referenced column '{}' in FOREIGN KEY constraint",
                    column_name
                )));
            }

            let column = referenced_lookup.get(&normalized).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown referenced column '{}' in table '{}'",
                    column_name, referenced_table_info.display_name
                ))
            })?;

            referenced_field_ids.push(column.field_id);
            referenced_column_defs.push((*column).clone());
            referenced_column_names.push(column.name.clone());
        }

        // Validate that the referenced columns form a UNIQUE or PRIMARY KEY constraint
        if referenced_columns.len() == 1 {
            // Single column: check if it has UNIQUE or PRIMARY KEY constraint
            let column = &referenced_column_defs[0];
            if !column.primary_key && !column.unique {
                return Err(Error::InvalidArgumentError(format!(
                    "FOREIGN KEY references column '{}' in table '{}' that is not UNIQUE or PRIMARY KEY",
                    column.name, referenced_table_info.display_name
                )));
            }
        } else {
            // Multiple columns: check if they form a multi-column PRIMARY KEY or UNIQUE constraint

            // First check if all columns have primary_key = true (multi-column PRIMARY KEY)
            let all_primary_key = referenced_column_defs.iter().all(|col| col.primary_key);

            // Also check if they form a multi-column UNIQUE constraint
            let has_multi_column_unique =
                referenced_table_info
                    .multi_column_uniques
                    .iter()
                    .any(|unique_entry| {
                        // Check if this unique constraint matches our referenced columns
                        if unique_entry.column_ids.len() != referenced_field_ids.len() {
                            return false;
                        }
                        // Check if all field IDs match (order-independent)
                        let unique_set: FxHashSet<_> =
                            unique_entry.column_ids.iter().copied().collect();
                        let referenced_set: FxHashSet<_> =
                            referenced_field_ids.iter().copied().collect();
                        unique_set == referenced_set
                    });

            if !all_primary_key && !has_multi_column_unique {
                return Err(Error::InvalidArgumentError(format!(
                    "FOREIGN KEY references columns ({}) in table '{}' that do not form a UNIQUE or PRIMARY KEY constraint",
                    referenced_column_names.join(", "),
                    referenced_table_info.display_name
                )));
            }
        }

        for (child_col, parent_col) in referencing_column_defs
            .iter()
            .zip(referenced_column_defs.iter())
        {
            if child_col.data_type != parent_col.data_type {
                return Err(Error::InvalidArgumentError(format!(
                    "FOREIGN KEY column '{}' type {:?} does not match referenced column '{}' type {:?}",
                    child_col.name, child_col.data_type, parent_col.name, parent_col.data_type
                )));
            }

            // Nullable child referencing non-null parent is allowed; no additional action required.
        }

        results.push(ValidatedForeignKey {
            name: spec.name.clone(),
            referencing_indices,
            referencing_field_ids,
            referencing_column_names,
            referenced_table_id: referenced_table_info.table_id,
            referenced_table_display: referenced_table_info.display_name.clone(),
            referenced_table_canonical: referenced_table_info.canonical_name.clone(),
            referenced_field_ids,
            referenced_column_names,
            on_delete: map_plan_action(spec.on_delete.clone()),
            on_update: map_plan_action(spec.on_update.clone()),
        });
    }

    Ok(results)
}

fn map_plan_action(action: PlanForeignKeyAction) -> ForeignKeyAction {
    match action {
        PlanForeignKeyAction::NoAction => ForeignKeyAction::NoAction,
        PlanForeignKeyAction::Restrict => ForeignKeyAction::Restrict,
    }
}

// ============================================================================
// Runtime constraint helpers
// ============================================================================

/// Ensure existing + incoming values remain unique for a single column.
pub fn ensure_single_column_unique(
    existing_values: &[PlanValue],
    new_values: &[PlanValue],
    column_name: &str,
) -> LlkvResult<()> {
    let mut seen: FxHashSet<UniqueKey> = FxHashSet::default();

    for value in existing_values {
        if let Some(key) = unique_key_component(value, column_name)?
            && !seen.insert(key.clone())
        {
            return Err(Error::ConstraintError(format!(
                "constraint violation on column '{}'",
                column_name
            )));
        }
    }

    for value in new_values {
        if let Some(key) = unique_key_component(value, column_name)?
            && !seen.insert(key.clone())
        {
            return Err(Error::ConstraintError(format!(
                "constraint violation on column '{}'",
                column_name
            )));
        }
    }

    Ok(())
}

/// Ensure primary key values remain unique and non-null.
pub fn ensure_primary_key(
    existing_rows: &[Vec<PlanValue>],
    new_rows: &[Vec<PlanValue>],
    column_names: &[String],
) -> LlkvResult<()> {
    let pk_label = if column_names.len() == 1 {
        "column"
    } else {
        "columns"
    };
    let pk_display = if column_names.len() == 1 {
        column_names[0].clone()
    } else {
        column_names.join(", ")
    };

    let mut seen: FxHashSet<UniqueKey> = FxHashSet::default();

    for row_values in existing_rows {
        if row_values.len() != column_names.len() {
            continue;
        }

        let key = build_composite_unique_key(row_values, column_names)?;
        let key = key.ok_or_else(|| {
            Error::ConstraintError(format!(
                "constraint failed: NOT NULL constraint failed for PRIMARY KEY {pk_label} '{pk_display}'"
            ))
        })?;

        if !seen.insert(key.clone()) {
            return Err(Error::ConstraintError(format!(
                "Duplicate key violates primary key constraint on {pk_label} '{pk_display}' (PRIMARY KEY or UNIQUE constraint violation)"
            )));
        }
    }

    for row_values in new_rows {
        if row_values.len() != column_names.len() {
            continue;
        }

        let key = build_composite_unique_key(row_values, column_names)?;
        let key = key.ok_or_else(|| {
            Error::ConstraintError(format!(
                "constraint failed: NOT NULL constraint failed for PRIMARY KEY {pk_label} '{pk_display}'"
            ))
        })?;

        if !seen.insert(key.clone()) {
            return Err(Error::ConstraintError(format!(
                "Duplicate key violates primary key constraint on {pk_label} '{pk_display}' (PRIMARY KEY or UNIQUE constraint violation)"
            )));
        }
    }

    Ok(())
}

/// Ensure that referencing rows satisfy the foreign key constraint by matching existing parent keys.
pub fn validate_foreign_key_rows(
    constraint_name: Option<&str>,
    referencing_table: &str,
    referenced_table: &str,
    referenced_column_names: &[String],
    parent_keys: &[Vec<PlanValue>],
    candidate_keys: &[Vec<PlanValue>],
) -> LlkvResult<()> {
    if parent_keys.is_empty() {
        // If there are no parent keys, every non-null candidate will fail.
        for key in candidate_keys {
            if key.iter().all(|value| !matches!(value, PlanValue::Null)) {
                let constraint_label = constraint_name.unwrap_or("FOREIGN KEY");
                let referenced_columns = if referenced_column_names.is_empty() {
                    String::from("<unknown>")
                } else {
                    referenced_column_names.join(", ")
                };
                return Err(Error::ConstraintError(format!(
                    "Violates foreign key constraint '{}' on table '{}' referencing '{}' (columns: {}) - does not exist in the referenced table",
                    constraint_label, referencing_table, referenced_table, referenced_columns,
                )));
            }
        }
        return Ok(());
    }

    for key in candidate_keys {
        if key.iter().any(|value| matches!(value, PlanValue::Null)) {
            continue;
        }

        if parent_keys.iter().any(|existing| existing == key) {
            continue;
        }

        let constraint_label = constraint_name.unwrap_or("FOREIGN KEY");
        let referenced_columns = if referenced_column_names.is_empty() {
            String::from("<unknown>")
        } else {
            referenced_column_names.join(", ")
        };

        return Err(Error::ConstraintError(format!(
            "Violates foreign key constraint '{}' on table '{}' referencing '{}' (columns: {}) - does not exist in the referenced table",
            constraint_label, referencing_table, referenced_table, referenced_columns,
        )));
    }

    Ok(())
}
