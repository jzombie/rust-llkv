//! Canonical scalar representations shared across planner consumers.
//!
//! These structures provide stable hashing and equality semantics for values
//! produced by the planner or execution layers. They normalize tricky cases
//! (like `-0.0` versus `0.0` and multiple NaN payloads) so higher layers can
//! deduplicate rows without reimplementing Arrow-specific comparisons.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, IntervalMonthDayNanoArray,
    StringArray,
};
use arrow::datatypes::{DataType, Field, IntervalUnit};
use arrow::record_batch::RecordBatch;
use llkv_expr::literal::IntervalValue;
use llkv_result::{Error, Result as LlkvResult};

use crate::plans::PlanValue;

/// Canonical scalar value with stable hashing semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CanonicalScalar {
    Null,
    Int64(i64),
    Float(u64),
    FloatNaN,
    Utf8(Arc<str>),
    Boolean(bool),
    Date32(i32),
    Interval(IntervalValue),
}

impl CanonicalScalar {
    /// Build a canonical scalar from a planner value.
    pub fn from_plan_value(value: &PlanValue) -> LlkvResult<Self> {
        match value {
            PlanValue::Null => Ok(CanonicalScalar::Null),
            PlanValue::Integer(v) => Ok(CanonicalScalar::Int64(*v)),
            PlanValue::Float(v) => Ok(Self::from_f64(*v)),
            PlanValue::String(v) => Ok(CanonicalScalar::Utf8(Arc::<str>::from(v.as_str()))),
            PlanValue::Date32(days) => Ok(CanonicalScalar::Date32(*days)),
            PlanValue::Interval(interval) => Ok(CanonicalScalar::Interval(*interval)),
            PlanValue::Struct(_) => Err(Error::InvalidArgumentError(
                "struct values are not supported in canonical scalar conversion".into(),
            )),
        }
    }

    /// Build a canonical scalar from an Arrow array value.
    pub fn from_arrow(field: &Field, array: &ArrayRef, row_idx: usize) -> LlkvResult<Self> {
        if array.is_null(row_idx) {
            return Ok(CanonicalScalar::Null);
        }

        match field.data_type() {
            DataType::Int64 => {
                let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "expected INT64 array when building canonical scalar".into(),
                    )
                })?;
                Ok(CanonicalScalar::Int64(values.value(row_idx)))
            }
            DataType::Float64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expected FLOAT64 array when building canonical scalar".into(),
                        )
                    })?;
                Ok(Self::from_f64(values.value(row_idx)))
            }
            DataType::Utf8 => {
                let values = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expected UTF8 array when building canonical scalar".into(),
                        )
                    })?;
                Ok(CanonicalScalar::Utf8(Arc::<str>::from(
                    values.value(row_idx),
                )))
            }
            DataType::Boolean => {
                let values = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expected BOOLEAN array when building canonical scalar".into(),
                        )
                    })?;
                Ok(CanonicalScalar::Boolean(values.value(row_idx)))
            }
            DataType::Date32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expected DATE32 array when building canonical scalar".into(),
                        )
                    })?;
                Ok(CanonicalScalar::Date32(values.value(row_idx)))
            }
            DataType::Null => Ok(CanonicalScalar::Null),
            DataType::Interval(unit) => match unit {
                IntervalUnit::MonthDayNano => {
                    let values = array
                        .as_any()
                        .downcast_ref::<IntervalMonthDayNanoArray>()
                        .ok_or_else(|| {
                            Error::InvalidArgumentError(
                                "expected INTERVAL MonthDayNano array when building canonical scalar"
                                    .into(),
                            )
                        })?;
                    let raw = values.value(row_idx);
                    Ok(CanonicalScalar::Interval(IntervalValue::new(
                        raw.months,
                        raw.days,
                        raw.nanoseconds,
                    )))
                }
                other => Err(Error::InvalidArgumentError(format!(
                    "building canonical scalar is not supported for interval unit {other:?}"
                ))),
            },
            other => Err(Error::InvalidArgumentError(format!(
                "building canonical scalar is not supported for column type {other:?}"
            ))),
        }
    }

    fn from_f64(raw: f64) -> CanonicalScalar {
        if raw.is_nan() {
            CanonicalScalar::FloatNaN
        } else {
            let normalized = if raw == 0.0 { 0.0 } else { raw };
            CanonicalScalar::Float(normalized.to_bits())
        }
    }
}

/// Canonical row representation used for hashing and equality checks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CanonicalRow(pub Vec<CanonicalScalar>);

impl CanonicalRow {
    /// Build a canonical row from an Arrow record batch.
    pub fn from_batch(batch: &RecordBatch, row_idx: usize) -> LlkvResult<Self> {
        let schema = batch.schema();
        let mut values = Vec::with_capacity(batch.num_columns());

        for col_idx in 0..batch.num_columns() {
            let field = schema.field(col_idx);
            let array = batch.column(col_idx);
            values.push(CanonicalScalar::from_arrow(field, array, row_idx)?);
        }

        Ok(CanonicalRow(values))
    }

    /// Return an immutable view of the canonical values.
    pub fn values(&self) -> &[CanonicalScalar] {
        &self.0
    }

    /// Consume the row and return its underlying values.
    pub fn into_inner(self) -> Vec<CanonicalScalar> {
        self.0
    }
}

impl From<Vec<CanonicalScalar>> for CanonicalRow {
    fn from(values: Vec<CanonicalScalar>) -> Self {
        CanonicalRow(values)
    }
}
