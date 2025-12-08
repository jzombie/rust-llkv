use crate::plans::PlanResult;
use crate::schema::PlanSchema;
use arrow::datatypes::{DataType, Field, Schema};
use llkv_compute::projection::infer_literal_datatype;
use llkv_expr::expr::{BinaryOp, ScalarExpr};
use llkv_result::Error;
use llkv_scan::ScanProjection;
use llkv_types::FieldId;
use llkv_types::decimal::DecimalValue;
use std::collections::HashMap;
use std::sync::Arc;

pub fn schema_for_projections(
    schema: &PlanSchema,
    projections: &[ScanProjection],
) -> PlanResult<Arc<Schema>> {
    let mut fields: Vec<Field> = Vec::with_capacity(projections.len());
    for projection in projections {
        match projection {
            ScanProjection::Column(proj) => {
                let field_id = proj.logical_field_id.field_id();
                let column = schema.column_by_field_id(field_id).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unknown column with field id {} in projection",
                        field_id
                    ))
                })?;
                let name = proj.alias.clone().unwrap_or_else(|| column.name.clone());
                let mut metadata = HashMap::new();
                metadata.insert(
                    "field_id".to_string(), // Was llkv_table::constants::FIELD_ID_META_KEY
                    column.field_id.to_string(),
                );
                let field = Field::new(&name, column.data_type.clone(), column.is_nullable)
                    .with_metadata(metadata);
                fields.push(field);
            }
            ScanProjection::Computed { alias, expr } => {
                let dtype = infer_computed_data_type(schema, expr)?;
                let field = Field::new(alias, dtype, true);
                fields.push(field);
            }
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}

pub fn infer_computed_data_type(
    schema: &PlanSchema,
    expr: &ScalarExpr<FieldId>,
) -> PlanResult<DataType> {
    match expr {
        ScalarExpr::Literal(lit) => {
            let res = infer_literal_datatype(lit);
            res
        },
        ScalarExpr::Column(field_id) => {
            let column = schema.column_by_field_id(*field_id).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column with field id {} in computed projection",
                    field_id
                ))
            })?;
            Ok(normalized_numeric_type(&column.data_type))
        }
        ScalarExpr::Binary { left, op, right } => {
            let left_type = infer_computed_data_type(schema, left)?;
            let right_type = infer_computed_data_type(schema, right)?;

            match (left_type, right_type) {
                (DataType::Decimal128(_, s1), DataType::Decimal128(_, s2)) => {
                    let scale = match op {
                        BinaryOp::Add | BinaryOp::Subtract => s1.max(s2),
                        BinaryOp::Multiply => (s1 + s2).min(38),
                        BinaryOp::Divide => (s1.max(s2) + 4).min(38),
                        BinaryOp::Modulo => s1.max(s2),
                        _ => s1.max(s2),
                    };
                    Ok(DataType::Decimal128(38, scale))
                }
                (DataType::Decimal128(_, s), _) | (_, DataType::Decimal128(_, s)) => {
                    let scale = match op {
                        BinaryOp::Multiply => s,
                        BinaryOp::Divide => (s + 4).min(38),
                        _ => s,
                    };
                    Ok(DataType::Decimal128(38, scale))
                }
                (l, r) => Ok(llkv_compute::kernels::common_type_for_op(&l, &r, *op)),
            }
        }
        ScalarExpr::Not(_) => Ok(DataType::Int64),
        ScalarExpr::IsNull { .. } => Ok(DataType::Int64),
        ScalarExpr::Compare { .. } => Ok(DataType::Int64),
        ScalarExpr::Aggregate(agg) => match agg {
            llkv_expr::AggregateCall::Count { .. }
            | llkv_expr::AggregateCall::CountStar
            | llkv_expr::AggregateCall::CountNulls(_) => Ok(DataType::Int64),
            llkv_expr::AggregateCall::Sum { expr, .. }
            | llkv_expr::AggregateCall::Min(expr)
            | llkv_expr::AggregateCall::Max(expr) => infer_computed_data_type(schema, expr),
            llkv_expr::AggregateCall::Avg { expr, .. } => {
                let input_type = infer_computed_data_type(schema, expr)?;
                match input_type {
                    DataType::Decimal128(_, s) => Ok(DataType::Decimal128(38, s)),
                    _ => Ok(DataType::Float64),
                }
            }
            llkv_expr::AggregateCall::Total { expr, .. } => infer_computed_data_type(schema, expr),
            llkv_expr::AggregateCall::GroupConcat { .. } => Ok(DataType::Utf8),
        },
        ScalarExpr::GetField { base, field_name } => {
            let field_type = resolve_struct_field_type(schema, base, field_name)?;
            Ok(field_type)
        }
        ScalarExpr::Cast { data_type, .. } => Ok(data_type.clone()),
        ScalarExpr::Case {
            branches,
            else_expr,
            ..
        } => {
            let mut is_float = false;
            let mut is_decimal = false;
            let mut max_scale = 0;

            for (_, then_expr) in branches {
                let dtype = infer_computed_data_type(schema, then_expr)?;
                match dtype {
                    DataType::Float64 => is_float = true,
                    DataType::Decimal128(_, s) => {
                        is_decimal = true;
                        max_scale = max_scale.max(s);
                    }
                    _ => {}
                }
            }
            if let Some(inner) = else_expr.as_deref() {
                let dtype = infer_computed_data_type(schema, inner)?;
                match dtype {
                    DataType::Float64 => is_float = true,
                    DataType::Decimal128(_, s) => {
                        is_decimal = true;
                        max_scale = max_scale.max(s);
                    }
                    _ => {}
                }
            }

            if is_float {
                Ok(DataType::Float64)
            } else if is_decimal {
                Ok(DataType::Decimal128(38, max_scale))
            } else {
                Ok(DataType::Int64)
            }
        }
        ScalarExpr::Coalesce(items) => {
            let mut is_float = false;
            let mut is_decimal = false;
            let mut max_scale = 0;

            for item in items {
                let dtype = infer_computed_data_type(schema, item)?;
                match dtype {
                    DataType::Float64 => is_float = true,
                    DataType::Decimal128(_, s) => {
                        is_decimal = true;
                        max_scale = max_scale.max(s);
                    }
                    _ => {}
                }
            }

            if is_float {
                Ok(DataType::Float64)
            } else if is_decimal {
                Ok(DataType::Decimal128(38, max_scale))
            } else {
                Ok(DataType::Int64)
            }
        }
        ScalarExpr::Random => Ok(DataType::Float64),
        ScalarExpr::ScalarSubquery(sub) => Ok(sub.data_type.clone()),
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
        DataType::Decimal128(precision, scale) => DataType::Decimal128(*precision, *scale),
        other => other.clone(),
    }
}

fn resolve_struct_field_type(
    schema: &PlanSchema,
    base: &ScalarExpr<FieldId>,
    field_name: &str,
) -> PlanResult<DataType> {
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

#[allow(dead_code)]
fn decimal_literal_behaves_like_integer(value: &DecimalValue) -> bool {
    value.scale() == 0
        && value.raw_value() >= i128::from(i64::MIN)
        && value.raw_value() <= i128::from(i64::MAX)
}
