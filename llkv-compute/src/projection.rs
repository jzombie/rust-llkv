use std::hash::Hash;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float64Array, Int64Array,
    IntervalMonthDayNanoArray, StringArray, StructArray, new_null_array,
};
use arrow::compute;
use arrow::datatypes::{DataType, Field, IntervalUnit, Schema};
use arrow::record_batch::RecordBatch;
use llkv_expr::ScalarExpr;
use llkv_expr::literal::Literal;
use llkv_result::{Error, Result as LlkvResult};

use crate::scalar::interval::interval_value_to_arrow;

#[derive(Clone)]
pub struct ComputedLiteralInfo<F> {
    pub expr: ScalarExpr<F>,
    pub alias: String,
}

#[derive(Clone)]
pub enum ProjectionLiteral<F> {
    Column {
        data_type: DataType,
    },
    Computed {
        info: ComputedLiteralInfo<F>,
        data_type: DataType,
    },
}

pub fn emit_synthetic_null_batch<F: Hash + Eq + Copy>(
    projections: &[ProjectionLiteral<F>],
    out_schema: &Arc<Schema>,
    row_count: usize,
) -> LlkvResult<Option<RecordBatch>>
where
    F: 'static,
{
    if row_count == 0 {
        return Ok(None);
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(projections.len());
    for (idx, projection) in projections.iter().enumerate() {
        let array = match projection {
            ProjectionLiteral::Column { data_type } => new_null_array(data_type, row_count),
            ProjectionLiteral::Computed { info, data_type } => {
                synthesize_computed_literal_array(info, data_type, row_count)?
            }
        };
        // Ensure alignment with the provided schema, which is already built in the caller.
        debug_assert_eq!(
            data_type_of_projection(projection),
            out_schema.field(idx).data_type()
        );
        columns.push(array);
    }

    let batch = RecordBatch::try_new(Arc::clone(out_schema), columns)?;
    Ok(Some(batch))
}

pub fn synthesize_computed_literal_array<F: Hash + Eq + Copy>(
    info: &ComputedLiteralInfo<F>,
    data_type: &DataType,
    row_count: usize,
) -> LlkvResult<ArrayRef> {
    if row_count == 0 {
        return Ok(new_null_array(data_type, 0));
    }

    match &info.expr {
        ScalarExpr::Literal(Literal::Integer(value)) => {
            let v = i64::try_from(*value).map_err(|_| {
                Error::InvalidArgumentError(
                    "integer literal exceeds supported range for INT64 column".into(),
                )
            })?;
            Ok(Arc::new(Int64Array::from(vec![v; row_count])) as ArrayRef)
        }
        ScalarExpr::Literal(Literal::Float(value)) => {
            Ok(Arc::new(Float64Array::from(vec![*value; row_count])) as ArrayRef)
        }
        ScalarExpr::Literal(Literal::Decimal(value)) => {
            let iter = std::iter::repeat_n(value.raw_value(), row_count);
            let precision = std::cmp::max(value.precision(), value.scale() as u8);
            let array = Decimal128Array::from_iter_values(iter)
                .with_precision_and_scale(precision, value.scale())
                .map_err(|err| {
                    Error::InvalidArgumentError(format!(
                        "failed to build Decimal128 literal array: {err}"
                    ))
                })?;
            Ok(Arc::new(array) as ArrayRef)
        }
        ScalarExpr::Literal(Literal::Boolean(value)) => {
            Ok(Arc::new(BooleanArray::from(vec![*value; row_count])) as ArrayRef)
        }
        ScalarExpr::Literal(Literal::String(value)) => {
            Ok(Arc::new(StringArray::from(vec![value.clone(); row_count])) as ArrayRef)
        }
        ScalarExpr::Literal(Literal::Date32(value)) => {
            Ok(Arc::new(Date32Array::from(vec![*value; row_count])) as ArrayRef)
        }
        ScalarExpr::Literal(Literal::Interval(value)) => {
            let native = interval_value_to_arrow(*value);
            Ok(Arc::new(IntervalMonthDayNanoArray::from(vec![native; row_count])) as ArrayRef)
        }
        ScalarExpr::Literal(Literal::Null) => Ok(new_null_array(data_type, row_count)),
        ScalarExpr::Literal(Literal::Struct(fields)) => {
            // Build a struct array from the literal fields
            let mut field_arrays = Vec::new();
            let mut arrow_fields = Vec::new();

            for (field_name, field_literal) in fields {
                // Infer the proper data type from the literal
                let field_dtype = infer_literal_datatype(field_literal.as_ref())?;
                arrow_fields.push(Field::new(field_name.clone(), field_dtype.clone(), true));

                // Create the array for this field
                let field_array = match field_literal.as_ref() {
                    Literal::Integer(v) => {
                        let int_val = i64::try_from(*v).unwrap_or(0);
                        Arc::new(Int64Array::from(vec![int_val; row_count])) as ArrayRef
                    }
                    Literal::Float(v) => {
                        Arc::new(Float64Array::from(vec![*v; row_count])) as ArrayRef
                    }
                    Literal::Decimal(v) => {
                        let iter = std::iter::repeat_n(v.raw_value(), row_count);
                        let precision = std::cmp::max(v.precision(), v.scale() as u8);
                        let array = Decimal128Array::from_iter_values(iter)
                            .with_precision_and_scale(precision, v.scale())
                            .map_err(|err| {
                                Error::InvalidArgumentError(format!(
                                    "failed to build Decimal128 literal array: {err}"
                                ))
                            })?;
                        Arc::new(array) as ArrayRef
                    }
                    Literal::Boolean(v) => {
                        Arc::new(BooleanArray::from(vec![*v; row_count])) as ArrayRef
                    }
                    Literal::String(v) => {
                        Arc::new(StringArray::from(vec![v.clone(); row_count])) as ArrayRef
                    }
                    Literal::Date32(v) => {
                        Arc::new(Date32Array::from(vec![*v; row_count])) as ArrayRef
                    }
                    Literal::Interval(v) => Arc::new(IntervalMonthDayNanoArray::from(vec![
                        interval_value_to_arrow(*v);
                        row_count
                    ])) as ArrayRef,
                    Literal::Null => new_null_array(&field_dtype, row_count),
                    Literal::Struct(nested_fields) => {
                        // Recursively build nested struct
                        let nested_info = ComputedLiteralInfo {
                            expr: ScalarExpr::<F>::Literal(Literal::Struct(nested_fields.clone())),
                            alias: field_name.clone(),
                        };
                        synthesize_computed_literal_array(&nested_info, &field_dtype, row_count)?
                    }
                };

                field_arrays.push(field_array);
            }

            let struct_array = StructArray::try_new(
                arrow_fields.into(),
                field_arrays,
                None, // No null buffer
            )
            .map_err(|e| Error::Internal(format!("failed to create struct array: {}", e)))?;

            Ok(Arc::new(struct_array) as ArrayRef)
        }
        ScalarExpr::Cast {
            expr,
            data_type: target_type,
        } => {
            let inner_dtype = match expr.as_ref() {
                ScalarExpr::Literal(lit) => infer_literal_datatype(lit)?,
                ScalarExpr::Cast { data_type, .. } => data_type.clone(),
                _ => return Ok(new_null_array(data_type, row_count)),
            };

            let inner_info = ComputedLiteralInfo {
                expr: expr.as_ref().clone(),
                alias: info.alias.clone(),
            };
            let inner = synthesize_computed_literal_array(&inner_info, &inner_dtype, row_count)?;
            compute::cast(&inner, target_type)
                .map_err(|e| Error::InvalidArgumentError(format!("failed to cast literal: {e}")))
        }
        ScalarExpr::Column(_)
        | ScalarExpr::Binary { .. }
        | ScalarExpr::Compare { .. }
        | ScalarExpr::Aggregate(_)
        | ScalarExpr::GetField { .. }
        | ScalarExpr::Not(_)
        | ScalarExpr::IsNull { .. }
        | ScalarExpr::Case { .. }
        | ScalarExpr::Coalesce(_)
        | ScalarExpr::Random => Ok(new_null_array(data_type, row_count)),
        ScalarExpr::ScalarSubquery(_) => Ok(new_null_array(data_type, row_count)),
    }
}

pub fn infer_literal_datatype(literal: &Literal) -> LlkvResult<DataType> {
    match literal {
        Literal::Integer(_) => Ok(DataType::Int64),
        Literal::Float(_) => Ok(DataType::Float64),
        Literal::Decimal(value) => Ok(DataType::Decimal128(value.precision(), value.scale())),
        Literal::Boolean(_) => Ok(DataType::Boolean),
        Literal::String(_) => Ok(DataType::Utf8),
        Literal::Date32(_) => Ok(DataType::Date32),
        Literal::Interval(_) => Ok(DataType::Interval(IntervalUnit::MonthDayNano)),
        Literal::Null => Ok(DataType::Null),
        Literal::Struct(fields) => {
            let inferred_fields = fields
                .iter()
                .map(|(name, nested)| {
                    let dtype = infer_literal_datatype(nested.as_ref())?;
                    Ok(Field::new(name.clone(), dtype, true))
                })
                .collect::<LlkvResult<Vec<_>>>()?;
            Ok(DataType::Struct(inferred_fields.into()))
        }
    }
}

fn data_type_of_projection<F>(projection: &ProjectionLiteral<F>) -> &DataType {
    match projection {
        ProjectionLiteral::Column { data_type } => data_type,
        ProjectionLiteral::Computed { data_type, .. } => data_type,
    }
}
