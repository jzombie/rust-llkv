//! Helper functions for value coercion and data preparation used during inserts.

use crate::utils::date::parse_date32_literal;
use crate::utils::interval::interval_value_to_arrow;
use crate::utils::{
    align_decimal_to_scale,
    decimal_from_f64,
    decimal_from_i64,
    decimal_truthy,
    truncate_decimal_to_i64,
};
use arrow::array::{
    ArrayRef, BooleanBuilder, Date32Builder, Decimal128Array, Float64Builder, Int64Builder,
    IntervalMonthDayNanoArray, StringBuilder,
};
use arrow::datatypes::{DataType, FieldRef, IntervalUnit};
use llkv_plan::PlanValue;
use llkv_result::{Error, Result};
use std::sync::Arc;

use crate::{ExecutorColumn, ExecutorSchema};

/// Resolve the user-specified column list for an INSERT statement into indexes
/// of the executor schema. If no columns were provided, return the identity order.
pub fn resolve_insert_columns(columns: &[String], schema: &ExecutorSchema) -> Result<Vec<usize>> {
    if columns.is_empty() {
        return Ok((0..schema.columns.len()).collect());
    }

    let mut resolved = Vec::with_capacity(columns.len());
    for column in columns {
        let normalized = column.to_ascii_lowercase();
        let index = schema.lookup.get(&normalized).ok_or_else(|| {
            Error::InvalidArgumentError(format!(
                "Binder Error: does not have a column named '{}'",
                column
            ))
        })?;
        resolved.push(*index);
    }
    Ok(resolved)
}

/// Coerce a `PlanValue` into the Arrow data type required by the executor column.
pub fn normalize_insert_value_for_column(
    column: &ExecutorColumn,
    value: PlanValue,
) -> Result<PlanValue> {
    match (&column.data_type, value) {
        (_, PlanValue::Null) => Ok(PlanValue::Null),
        (DataType::Int64, PlanValue::Integer(v)) => Ok(PlanValue::Integer(v)),
        (DataType::Int64, PlanValue::Float(v)) => Ok(PlanValue::Integer(v as i64)),
        (DataType::Int64, PlanValue::Decimal(decimal)) => {
            let coerced = truncate_decimal_to_i64(decimal).map_err(|err| {
                Error::InvalidArgumentError(format!(
                    "cannot insert decimal literal {} into INT column '{}': {err}",
                    decimal, column.name
                ))
            })?;
            Ok(PlanValue::Integer(coerced))
        }
        (DataType::Int64, other) => Err(Error::InvalidArgumentError(format!(
            "cannot insert {other:?} into INT column '{}'",
            column.name
        ))),
        (DataType::Boolean, PlanValue::Integer(v)) => {
            Ok(PlanValue::Integer(if v != 0 { 1 } else { 0 }))
        }
        (DataType::Boolean, PlanValue::Float(v)) => {
            Ok(PlanValue::Integer(if v != 0.0 { 1 } else { 0 }))
        }
        (DataType::Boolean, PlanValue::Decimal(decimal)) => {
            Ok(PlanValue::Integer(if decimal_truthy(decimal) { 1 } else { 0 }))
        }
        (DataType::Boolean, PlanValue::String(s)) => {
            let normalized = s.trim().to_ascii_lowercase();
            let value = match normalized.as_str() {
                "true" | "t" | "1" => 1,
                "false" | "f" | "0" => 0,
                _ => {
                    return Err(Error::InvalidArgumentError(format!(
                        "cannot insert string '{}' into BOOLEAN column '{}'",
                        s, column.name
                    )));
                }
            };
            Ok(PlanValue::Integer(value))
        }
        (DataType::Boolean, PlanValue::Struct(_)) => Err(Error::InvalidArgumentError(format!(
            "cannot insert struct into BOOLEAN column '{}'",
            column.name
        ))),
        (DataType::Float64, PlanValue::Integer(v)) => Ok(PlanValue::Float(v as f64)),
        (DataType::Float64, PlanValue::Float(v)) => Ok(PlanValue::Float(v)),
        (DataType::Float64, PlanValue::Decimal(decimal)) => {
            Ok(PlanValue::Float(decimal.to_f64()))
        }
        (DataType::Float64, other) => Err(Error::InvalidArgumentError(format!(
            "cannot insert {other:?} into DOUBLE column '{}'",
            column.name
        ))),
        (DataType::Utf8, PlanValue::Integer(v)) => Ok(PlanValue::String(v.to_string())),
        (DataType::Utf8, PlanValue::Float(v)) => Ok(PlanValue::String(v.to_string())),
        (DataType::Utf8, PlanValue::Decimal(decimal)) => {
            Ok(PlanValue::String(decimal.to_string()))
        }
        (DataType::Utf8, PlanValue::String(s)) => Ok(PlanValue::String(s)),
        (DataType::Utf8, PlanValue::Struct(_)) => Err(Error::InvalidArgumentError(format!(
            "cannot insert struct into STRING column '{}'",
            column.name
        ))),
        (DataType::Date32, PlanValue::Date32(days)) => Ok(PlanValue::Date32(days)),
        (DataType::Date32, PlanValue::Integer(days)) => {
            let casted = i32::try_from(days).map_err(|_| {
                Error::InvalidArgumentError(format!(
                    "integer literal out of range for DATE column '{}'",
                    column.name
                ))
            })?;
            Ok(PlanValue::Date32(casted))
        }
        (DataType::Date32, PlanValue::String(text)) => {
            let days = parse_date32_literal(&text)?;
            Ok(PlanValue::Date32(days))
        }
        (DataType::Date32, PlanValue::Decimal(_)) => Err(Error::InvalidArgumentError(format!(
            "cannot insert decimal literal into DATE column '{}'",
            column.name
        ))),
        (DataType::Date32, other) => Err(Error::InvalidArgumentError(format!(
            "cannot insert {other:?} into DATE column '{}'",
            column.name
        ))),
        (DataType::Struct(_), PlanValue::Struct(map)) => Ok(PlanValue::Struct(map)),
        (DataType::Struct(_), other) => Err(Error::InvalidArgumentError(format!(
            "expected struct value for struct column '{}', got {other:?}",
            column.name
        ))),
        (DataType::Interval(IntervalUnit::MonthDayNano), PlanValue::Interval(interval)) => {
            Ok(PlanValue::Interval(interval))
        }
        (DataType::Interval(IntervalUnit::MonthDayNano), other) => {
            Err(Error::InvalidArgumentError(format!(
                "cannot insert {other:?} into INTERVAL column '{}'",
                column.name
            )))
        }
        (DataType::Decimal128(precision, scale), PlanValue::Decimal(decimal)) => {
            let aligned = align_decimal_to_scale(decimal, *precision, *scale).map_err(|err| {
                Error::InvalidArgumentError(format!(
                    "decimal literal {} incompatible with DECIMAL({}, {}) column '{}': {err}",
                    decimal, precision, scale, column.name
                ))
            })?;
            Ok(PlanValue::Decimal(aligned))
        }
        (DataType::Decimal128(precision, scale), PlanValue::Integer(value)) => {
            let decimal = decimal_from_i64(value, *precision, *scale).map_err(|err| {
                Error::InvalidArgumentError(format!(
                    "integer literal {value} incompatible with DECIMAL({}, {}) column '{}': {err}",
                    precision, scale, column.name
                ))
            })?;
            Ok(PlanValue::Decimal(decimal))
        }
        (DataType::Decimal128(precision, scale), PlanValue::Float(value)) => {
            let decimal = decimal_from_f64(value, *precision, *scale).map_err(|err| {
                Error::InvalidArgumentError(format!(
                    "float literal {value} incompatible with DECIMAL({}, {}) column '{}': {err}",
                    precision, scale, column.name
                ))
            })?;
            Ok(PlanValue::Decimal(decimal))
        }
        (DataType::Decimal128(_, _), other) => Err(Error::InvalidArgumentError(format!(
            "cannot insert {other:?} into DECIMAL column '{}'",
            column.name
        ))),
        (other_type, other_value) => Err(Error::InvalidArgumentError(format!(
            "unsupported Arrow data type {:?} for INSERT value {:?} in column '{}'",
            other_type, other_value, column.name
        ))),
    }
}

/// Build an Arrow array that matches the executor column's data type from the provided values.
pub fn build_array_for_column(dtype: &DataType, values: &[PlanValue]) -> Result<ArrayRef> {
    match dtype {
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(*v),
                    PlanValue::Float(v) => builder.append_value(*v as i64),
                    PlanValue::Decimal(decimal) => {
                        let coerced = truncate_decimal_to_i64(*decimal).map_err(|err| {
                            Error::InvalidArgumentError(format!(
                                "cannot insert decimal literal {} into INT column: {err}",
                                decimal
                            ))
                        })?;
                        builder.append_value(coerced);
                    }
                    PlanValue::Date32(days) => builder.append_value(i64::from(*days)),
                    PlanValue::String(_) | PlanValue::Struct(_) | PlanValue::Interval(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert non-integer into INT column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Boolean => {
            let mut builder = BooleanBuilder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(*v != 0),
                    PlanValue::Float(v) => builder.append_value(*v != 0.0),
                    PlanValue::Decimal(decimal) => {
                        builder.append_value(decimal_truthy(*decimal));
                    }
                    PlanValue::Date32(days) => builder.append_value(*days != 0),
                    PlanValue::String(s) => {
                        let normalized = s.trim().to_ascii_lowercase();
                        match normalized.as_str() {
                            "true" | "t" | "1" => builder.append_value(true),
                            "false" | "f" | "0" => builder.append_value(false),
                            _ => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "cannot insert string '{}' into BOOLEAN column",
                                    s
                                )));
                            }
                        }
                    }
                    PlanValue::Struct(_) | PlanValue::Interval(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert struct into BOOLEAN column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(*v as f64),
                    PlanValue::Float(v) => builder.append_value(*v),
                    PlanValue::Decimal(decimal) => {
                        builder.append_value(decimal.to_f64());
                    }
                    PlanValue::Date32(days) => builder.append_value(f64::from(*days)),
                    PlanValue::String(_) | PlanValue::Struct(_) | PlanValue::Interval(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert non-numeric into DOUBLE column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Utf8 => {
            let mut builder = StringBuilder::with_capacity(values.len(), values.len() * 8);
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(v.to_string()),
                    PlanValue::Float(v) => builder.append_value(v.to_string()),
                    PlanValue::Decimal(decimal) => {
                        builder.append_value(decimal.to_string());
                    }
                    PlanValue::Date32(days) => builder.append_value(days.to_string()),
                    PlanValue::String(s) => builder.append_value(s),
                    PlanValue::Struct(_) | PlanValue::Interval(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert struct into STRING column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Date32 => {
            let mut builder = Date32Builder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(days) => {
                        let casted = i32::try_from(*days).map_err(|_| {
                            Error::InvalidArgumentError(
                                "integer literal out of range for DATE column".into(),
                            )
                        })?;
                        builder.append_value(casted);
                    }
                    PlanValue::Date32(days) => builder.append_value(*days),
                    PlanValue::Decimal(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert decimal literal into DATE column".into(),
                        ));
                    }
                    PlanValue::Float(_) | PlanValue::Struct(_) | PlanValue::Interval(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert non-date value into DATE column".into(),
                        ));
                    }
                    PlanValue::String(text) => {
                        let days = parse_date32_literal(text)?;
                        builder.append_value(days);
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Struct(fields) => {
            use arrow::array::StructArray;
            let mut field_arrays: Vec<(FieldRef, ArrayRef)> = Vec::with_capacity(fields.len());

            for field in fields.iter() {
                let field_name = field.name();
                let field_type = field.data_type();
                let mut field_values = Vec::with_capacity(values.len());

                for value in values {
                    match value {
                        PlanValue::Null => field_values.push(PlanValue::Null),
                        PlanValue::Struct(map) => {
                            let field_value =
                                map.get(field_name).cloned().unwrap_or(PlanValue::Null);
                            field_values.push(field_value);
                        }
                        _ => {
                            return Err(Error::InvalidArgumentError(format!(
                                "expected struct value for struct column, got {:?}",
                                value
                            )));
                        }
                    }
                }

                let field_array = build_array_for_column(field_type, &field_values)?;
                field_arrays.push((Arc::clone(field), field_array));
            }

            Ok(Arc::new(StructArray::from(field_arrays)))
        }
        DataType::Decimal128(precision, scale) => {
            let mut raw_values: Vec<Option<i128>> = Vec::with_capacity(values.len());
            for value in values {
                let entry = match value {
                    PlanValue::Null => None,
                    PlanValue::Decimal(decimal) => {
                        let aligned = align_decimal_to_scale(*decimal, *precision, *scale)
                            .map_err(|err| {
                                Error::InvalidArgumentError(format!(
                                    "decimal literal {} incompatible with DECIMAL({}, {}): {err}",
                                    decimal, precision, scale
                                ))
                            })?;
                        Some(aligned.raw_value())
                    }
                    PlanValue::Integer(value) => {
                        let decimal = decimal_from_i64(*value, *precision, *scale).map_err(|err| {
                            Error::InvalidArgumentError(format!(
                                "integer literal {value} incompatible with DECIMAL({}, {}): {err}",
                                precision, scale
                            ))
                        })?;
                        Some(decimal.raw_value())
                    }
                    PlanValue::Float(value) => {
                        let decimal = decimal_from_f64(*value, *precision, *scale).map_err(|err| {
                            Error::InvalidArgumentError(format!(
                                "float literal {value} incompatible with DECIMAL({}, {}): {err}",
                                precision, scale
                            ))
                        })?;
                        Some(decimal.raw_value())
                    }
                    _ => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert non-decimal value into DECIMAL column".into(),
                        ));
                    }
                };
                raw_values.push(entry);
            }

            let array = Decimal128Array::from_iter(raw_values.into_iter())
                .with_precision_and_scale(*precision, *scale)
                .map_err(|err| Error::InvalidArgumentError(format!(
                    "failed to build Decimal128 array: {err}"
                )))?;
            Ok(Arc::new(array) as ArrayRef)
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            let mut converted: Vec<Option<_>> = Vec::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => converted.push(None),
                    PlanValue::Interval(interval) => {
                        converted.push(Some(interval_value_to_arrow(*interval)))
                    }
                    other => {
                        return Err(Error::InvalidArgumentError(format!(
                            "cannot insert {other:?} into INTERVAL column"
                        )));
                    }
                }
            }
            Ok(Arc::new(IntervalMonthDayNanoArray::from(converted)) as ArrayRef)
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported Arrow data type for INSERT: {other:?}"
        ))),
    }
}
