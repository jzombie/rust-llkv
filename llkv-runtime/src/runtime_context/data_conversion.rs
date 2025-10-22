//! Data conversion helpers for insert operations and type coercion

use arrow::array::{
    ArrayRef, BooleanBuilder, Date32Builder, Float64Builder, Int64Builder, StringBuilder,
};
use arrow::datatypes::{DataType, FieldRef};
use llkv_executor::{ExecutorColumn, ExecutorSchema};
use llkv_plan::PlanValue;
use llkv_result::{Error, Result};
use std::sync::Arc;
use time::{Date, Month};

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

pub(crate) fn normalize_insert_value_for_column(
    column: &ExecutorColumn,
    value: PlanValue,
) -> Result<PlanValue> {
    match (&column.data_type, value) {
        (_, PlanValue::Null) => Ok(PlanValue::Null),
        (DataType::Int64, PlanValue::Integer(v)) => Ok(PlanValue::Integer(v)),
        (DataType::Int64, PlanValue::Float(v)) => Ok(PlanValue::Integer(v as i64)),
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
        (DataType::Float64, other) => Err(Error::InvalidArgumentError(format!(
            "cannot insert {other:?} into DOUBLE column '{}'",
            column.name
        ))),
        (DataType::Utf8, PlanValue::Integer(v)) => Ok(PlanValue::String(v.to_string())),
        (DataType::Utf8, PlanValue::Float(v)) => Ok(PlanValue::String(v.to_string())),
        (DataType::Utf8, PlanValue::String(s)) => Ok(PlanValue::String(s)),
        (DataType::Utf8, PlanValue::Struct(_)) => Err(Error::InvalidArgumentError(format!(
            "cannot insert struct into STRING column '{}'",
            column.name
        ))),
        (DataType::Date32, PlanValue::Integer(days)) => {
            let casted = i32::try_from(days).map_err(|_| {
                Error::InvalidArgumentError(format!(
                    "integer literal out of range for DATE column '{}'",
                    column.name
                ))
            })?;
            Ok(PlanValue::Integer(casted as i64))
        }
        (DataType::Date32, PlanValue::String(text)) => {
            let days = parse_date32_literal(&text)?;
            Ok(PlanValue::Integer(days as i64))
        }
        (DataType::Date32, other) => Err(Error::InvalidArgumentError(format!(
            "cannot insert {other:?} into DATE column '{}'",
            column.name
        ))),
        (DataType::Struct(_), PlanValue::Struct(map)) => Ok(PlanValue::Struct(map)),
        (DataType::Struct(_), other) => Err(Error::InvalidArgumentError(format!(
            "expected struct value for struct column '{}', got {other:?}",
            column.name
        ))),
        (other_type, other_value) => Err(Error::InvalidArgumentError(format!(
            "unsupported Arrow data type {:?} for INSERT value {:?} in column '{}'",
            other_type, other_value, column.name
        ))),
    }
}

pub fn build_array_for_column(dtype: &DataType, values: &[PlanValue]) -> Result<ArrayRef> {
    match dtype {
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(*v),
                    PlanValue::Float(v) => builder.append_value(*v as i64),
                    PlanValue::String(_) | PlanValue::Struct(_) => {
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
                    PlanValue::Struct(_) => {
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
                    PlanValue::String(_) | PlanValue::Struct(_) => {
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
                    PlanValue::String(s) => builder.append_value(s),
                    PlanValue::Struct(_) => {
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
                    PlanValue::Float(_) | PlanValue::Struct(_) => {
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
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported Arrow data type for INSERT: {other:?}"
        ))),
    }
}

pub(crate) fn parse_date32_literal(text: &str) -> Result<i32> {
    let mut parts = text.split('-');
    let year_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    let month_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    let day_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError(format!(
            "invalid DATE literal '{text}'"
        )));
    }

    let year = year_str.parse::<i32>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid year in DATE literal '{text}'"))
    })?;
    let month_num = month_str.parse::<u8>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid month in DATE literal '{text}'"))
    })?;
    let day = day_str.parse::<u8>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid day in DATE literal '{text}'"))
    })?;

    let month = Month::try_from(month_num).map_err(|_| {
        Error::InvalidArgumentError(format!("invalid month in DATE literal '{text}'"))
    })?;

    let date = Date::from_calendar_date(year, month, day).map_err(|err| {
        Error::InvalidArgumentError(format!("invalid DATE literal '{text}': {err}"))
    })?;
    let days = date.to_julian_day() - epoch_julian_day();
    Ok(days)
}

fn epoch_julian_day() -> i32 {
    Date::from_calendar_date(1970, Month::January, 1)
        .expect("1970-01-01 is a valid date")
        .to_julian_day()
}
