use crate::ExecutorResult;
use crate::types::{ExecutorSchema, ExecutorTable};
use arrow::datatypes::{DataType, Field, Schema};
use llkv_expr::expr::ScalarExpr;
use llkv_expr::literal::Literal;
use llkv_result::Error;
use llkv_storage::pager::Pager;
use llkv_table::table::ScanProjection;
use llkv_table::types::FieldId;
use simd_r_drive_entry_handle::EntryHandle;
use std::collections::HashMap;
use std::sync::Arc;

pub fn schema_for_projections<P>(
    table: &ExecutorTable<P>,
    projections: &[ScanProjection],
) -> ExecutorResult<Arc<Schema>>
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
                let dtype = infer_computed_data_type(table.schema.as_ref(), expr)?;
                let field = Field::new(alias, dtype, true);
                fields.push(field);
            }
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn infer_computed_data_type(
    schema: &ExecutorSchema,
    expr: &ScalarExpr<FieldId>,
) -> ExecutorResult<DataType> {
    match expr {
        ScalarExpr::Literal(Literal::Integer(_)) => Ok(DataType::Int64),
        ScalarExpr::Literal(Literal::Float(_)) => Ok(DataType::Float64),
        ScalarExpr::Literal(Literal::Boolean(_)) => Ok(DataType::Boolean),
        ScalarExpr::Literal(Literal::String(_)) => Ok(DataType::Utf8),
        ScalarExpr::Literal(Literal::Null) => Ok(DataType::Null),
        ScalarExpr::Literal(Literal::Struct(_)) => Ok(DataType::Utf8),
        ScalarExpr::Column(field_id) => {
            let column = schema.column_by_field_id(*field_id).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column with field id {} in computed projection",
                    field_id
                ))
            })?;
            Ok(normalized_numeric_type(&column.data_type))
        }
        ScalarExpr::Binary { .. } => {
            if expression_uses_float(schema, expr)? {
                Ok(DataType::Float64)
            } else {
                Ok(DataType::Int64)
            }
        }
        ScalarExpr::Compare { .. } => Ok(DataType::Int64),
        ScalarExpr::Aggregate(_) => Ok(DataType::Int64),
        ScalarExpr::GetField { base, field_name } => {
            let field_type = resolve_struct_field_type(schema, base, field_name)?;
            Ok(field_type)
        }
        ScalarExpr::Cast { data_type, .. } => Ok(data_type.clone()),
        ScalarExpr::Case { .. } => {
            if expression_uses_float(schema, expr)? {
                Ok(DataType::Float64)
            } else {
                Ok(DataType::Int64)
            }
        }
        ScalarExpr::ScalarSubquery(_) => {
            // TODO: Infer type from subquery result
            Ok(DataType::Utf8)
        }
    }
}

fn normalized_numeric_type(dtype: &DataType) -> DataType {
    match dtype {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Boolean => DataType::Int64,
        DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64 => DataType::Float64,
        other => other.clone(),
    }
}

fn expression_uses_float(
    schema: &ExecutorSchema,
    expr: &ScalarExpr<FieldId>,
) -> ExecutorResult<bool> {
    match expr {
        ScalarExpr::Literal(Literal::Float(_)) => Ok(true),
        ScalarExpr::Literal(Literal::Integer(_))
        | ScalarExpr::Literal(Literal::Boolean(_))
        | ScalarExpr::Literal(Literal::Null)
        | ScalarExpr::Literal(Literal::String(_))
        | ScalarExpr::Literal(Literal::Struct(_)) => Ok(false),
        ScalarExpr::Column(field_id) => {
            let column = schema.column_by_field_id(*field_id).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column with field id {} in computed projection",
                    field_id
                ))
            })?;
            Ok(matches!(
                normalized_numeric_type(&column.data_type),
                DataType::Float64
            ))
        }
        ScalarExpr::Binary { left, right, .. } => {
            let left_float = expression_uses_float(schema, left)?;
            if left_float {
                return Ok(true);
            }
            expression_uses_float(schema, right)
        }
        ScalarExpr::Compare { left, right, .. } => {
            let left_float = expression_uses_float(schema, left)?;
            if left_float {
                return Ok(true);
            }
            expression_uses_float(schema, right)
        }
        ScalarExpr::Aggregate(_) => Ok(false),
        ScalarExpr::GetField { base, field_name } => {
            let field_type = resolve_struct_field_type(schema, base, field_name)?;
            Ok(matches!(
                normalized_numeric_type(&field_type),
                DataType::Float64
            ))
        }
        ScalarExpr::Cast { expr, data_type } => {
            if matches!(normalized_numeric_type(data_type), DataType::Float64) {
                return Ok(true);
            }
            expression_uses_float(schema, expr)
        }
        ScalarExpr::Case {
            branches,
            else_expr,
            ..
        } => {
            for (_, then_expr) in branches {
                if expression_uses_float(schema, then_expr)? {
                    return Ok(true);
                }
            }
            if let Some(inner) = else_expr.as_deref() {
                if expression_uses_float(schema, inner)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        ScalarExpr::ScalarSubquery(_) => Ok(false),
    }
}

fn resolve_struct_field_type(
    schema: &ExecutorSchema,
    base: &ScalarExpr<FieldId>,
    field_name: &str,
) -> ExecutorResult<DataType> {
    let base_type = infer_computed_data_type(schema, base)?;
    if let DataType::Struct(fields) = base_type {
        fields
            .iter()
            .find(|f| f.name() == field_name)
            .map(|f| f.data_type().clone())
            .ok_or_else(|| {
                Error::InvalidArgumentError(format!("Field '{}' not found in struct", field_name))
            })
    } else {
        Err(Error::InvalidArgumentError(
            "GetField can only be applied to struct types".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ExecutorColumn;
    use arrow::datatypes::DataType;
    use llkv_expr::expr::BinaryOp;
    use rustc_hash::FxHashMap;

    #[test]
    fn inferred_type_for_integer_expression_remains_int64() {
        const COL0: FieldId = 1;
        const COL1: FieldId = 2;

        let columns = vec![
            ExecutorColumn {
                name: "tab1.col0".into(),
                data_type: DataType::Int64,
                nullable: false,
                primary_key: false,
                unique: false,
                field_id: COL0,
                check_expr: None,
            },
            ExecutorColumn {
                name: "tab1.col1".into(),
                data_type: DataType::Int64,
                nullable: false,
                primary_key: false,
                unique: false,
                field_id: COL1,
                check_expr: None,
            },
        ];
        let schema = ExecutorSchema {
            columns,
            lookup: FxHashMap::default(),
        };

        let col0 = ScalarExpr::Column(COL0);
        let col1 = ScalarExpr::Column(COL1);
        let divisor = ScalarExpr::Literal(Literal::Null);
        let division = ScalarExpr::Binary {
            left: Box::new(col1.clone()),
            op: BinaryOp::Divide,
            right: Box::new(divisor),
        };
        let neg_col1 = ScalarExpr::Binary {
            left: Box::new(ScalarExpr::Literal(Literal::Integer(0))),
            op: BinaryOp::Subtract,
            right: Box::new(col1.clone()),
        };
        let sum_left = ScalarExpr::Binary {
            left: Box::new(col0),
            op: BinaryOp::Add,
            right: Box::new(division),
        };
        let expr = ScalarExpr::Binary {
            left: Box::new(sum_left),
            op: BinaryOp::Add,
            right: Box::new(neg_col1),
        };

        let dtype = infer_computed_data_type(&schema, &expr).expect("type inference succeeds");
        assert_eq!(dtype, DataType::Int64);
    }
}
