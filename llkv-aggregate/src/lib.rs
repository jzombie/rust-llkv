// TODO: Can portions of this be offloaded to llkv-compute as vectorized ops?

//! Runtime aggregation utilities used by the planner and executor.
//!
//! The crate evaluates logical aggregates described by [`llkv_plan::AggregateExpr`] against Arrow
//! batches. It supports streaming accumulation with overflow checks and COUNT DISTINCT tracking for
//! a subset of scalar types.
use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Float64Array, Float64Builder, Int64Array,
    Int64Builder, RecordBatch, StringArray,
};
use arrow::compute;
use arrow::datatypes::{DataType, Field};
use llkv_result::Error;
use llkv_types::FieldId;
use rustc_hash::FxHashSet;
use std::sync::Arc;
use std::{cmp::Ordering, convert::TryFrom};

pub use llkv_plan::{AggregateExpr, AggregateFunction};

pub mod stream;
pub use stream::{AggregateStream, GroupedAggregateStream};

/// Result type alias for aggregation routines.
pub type AggregateResult<T> = Result<T, Error>;

/// Specification for an aggregate operation.
#[derive(Clone)]
pub struct AggregateSpec {
    pub alias: String,
    pub kind: AggregateKind,
}

/// Type of aggregate operation.
#[derive(Clone)]
pub enum AggregateKind {
    Count {
        field_id: Option<FieldId>,
        distinct: bool,
    },
    Sum {
        field_id: FieldId,
        data_type: DataType,
        distinct: bool,
    },
    Total {
        field_id: FieldId,
        data_type: DataType,
        distinct: bool,
    },
    Avg {
        field_id: FieldId,
        data_type: DataType,
        distinct: bool,
    },
    Min {
        field_id: FieldId,
        data_type: DataType,
    },
    Max {
        field_id: FieldId,
        data_type: DataType,
    },
    CountNulls {
        field_id: FieldId,
    },
    GroupConcat {
        field_id: FieldId,
        distinct: bool,
        separator: String,
    },
}

impl AggregateKind {
    /// Returns the field ID referenced by this aggregate, if any.
    pub fn field_id(&self) -> Option<FieldId> {
        match self {
            AggregateKind::Count { field_id, .. } => *field_id,
            AggregateKind::Sum { field_id, .. }
            | AggregateKind::Total { field_id, .. }
            | AggregateKind::Avg { field_id, .. }
            | AggregateKind::Min { field_id, .. }
            | AggregateKind::Max { field_id, .. }
            | AggregateKind::CountNulls { field_id }
            | AggregateKind::GroupConcat { field_id, .. } => Some(*field_id),
        }
    }
}

/// Runtime state for an aggregate computation.
#[derive(Clone)]
pub struct AggregateState {
    pub alias: String,
    pub accumulator: AggregateAccumulator,
    pub override_value: Option<i64>,
}

/// Accumulator for incremental aggregate computation.
#[derive(Clone)]
pub enum AggregateAccumulator {
    CountStar {
        value: i64,
    },
    CountColumn {
        column_index: usize,
        value: i64,
    },
    CountDistinctColumn {
        column_index: usize,
        seen: FxHashSet<DistinctKey>,
    },
    SumInt64 {
        column_index: usize,
        value: Option<i64>, // None = overflow, Some = current sum or initial state
        has_values: bool,   // Track whether we've seen any values
    },
    SumDistinctInt64 {
        column_index: usize,
        sum: Option<i64>, // None = overflow
        seen: FxHashSet<DistinctKey>,
    },
    SumFloat64 {
        column_index: usize,
        value: f64,
        saw_value: bool,
    },
    SumDistinctFloat64 {
        column_index: usize,
        sum: f64,
        seen: FxHashSet<DistinctKey>,
    },
    TotalInt64 {
        column_index: usize,
        value: f64, // TOTAL always returns float to avoid overflow
    },
    TotalDistinctInt64 {
        column_index: usize,
        sum: f64, // TOTAL always returns float to avoid overflow
        seen: FxHashSet<DistinctKey>,
    },
    TotalFloat64 {
        column_index: usize,
        value: f64,
    },
    TotalDistinctFloat64 {
        column_index: usize,
        sum: f64,
        seen: FxHashSet<DistinctKey>,
    },
    AvgInt64 {
        column_index: usize,
        sum: i64,
        count: i64,
    },
    AvgDistinctInt64 {
        column_index: usize,
        sum: i64,
        seen: FxHashSet<DistinctKey>,
    },
    AvgFloat64 {
        column_index: usize,
        sum: f64,
        count: i64,
    },
    AvgDistinctFloat64 {
        column_index: usize,
        sum: f64,
        seen: FxHashSet<DistinctKey>,
    },
    SumDecimal128 {
        column_index: usize,
        sum: i128,
        precision: u8,
        scale: i8,
    },
    SumDistinctDecimal128 {
        column_index: usize,
        sum: i128,
        seen: FxHashSet<DistinctKey>,
        precision: u8,
        scale: i8,
    },
    TotalDecimal128 {
        column_index: usize,
        sum: i128,
        precision: u8,
        scale: i8,
    },
    TotalDistinctDecimal128 {
        column_index: usize,
        sum: i128,
        seen: FxHashSet<DistinctKey>,
        precision: u8,
        scale: i8,
    },
    AvgDecimal128 {
        column_index: usize,
        sum: i128,
        count: i64,
        precision: u8,
        scale: i8,
    },
    AvgDistinctDecimal128 {
        column_index: usize,
        sum: i128,
        seen: FxHashSet<DistinctKey>,
        precision: u8,
        scale: i8,
    },
    MinInt64 {
        column_index: usize,
        value: Option<i64>,
    },
    MinFloat64 {
        column_index: usize,
        value: Option<f64>,
    },
    MinDecimal128 {
        column_index: usize,
        value: Option<i128>,
        precision: u8,
        scale: i8,
    },
    MaxInt64 {
        column_index: usize,
        value: Option<i64>,
    },
    MaxFloat64 {
        column_index: usize,
        value: Option<f64>,
    },
    MaxDecimal128 {
        column_index: usize,
        value: Option<i128>,
        precision: u8,
        scale: i8,
    },
    CountNulls {
        column_index: usize,
        non_null_rows: i64,
        total_rows_seen: i64,
    },
    GroupConcat {
        column_index: usize,
        values: Vec<String>,
        separator: String,
    },
    GroupConcatDistinct {
        column_index: usize,
        seen: FxHashSet<String>,
        values: Vec<String>,
        separator: String,
    },
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub enum DistinctKey {
    Int(i64),
    Float(u64),
    Str(String),
    Bool(bool),
    Date(i32),
    Decimal(i128), // Store raw decimal value for exact comparison
}

impl DistinctKey {
    fn from_array(array: &ArrayRef, index: usize) -> AggregateResult<Self> {
        match array.data_type() {
            DataType::Int64 => {
                let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "COUNT(DISTINCT) expected an INT64 column in execution".into(),
                    )
                })?;
                Ok(DistinctKey::Int(values.value(index)))
            }
            DataType::Float64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "COUNT(DISTINCT) expected a FLOAT64 column in execution".into(),
                        )
                    })?;
                Ok(DistinctKey::Float(values.value(index).to_bits()))
            }
            DataType::Utf8 => {
                let values = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "COUNT(DISTINCT) expected a UTF8 column in execution".into(),
                        )
                    })?;
                Ok(DistinctKey::Str(values.value(index).to_owned()))
            }
            DataType::Boolean => {
                let values = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "COUNT(DISTINCT) expected a BOOLEAN column in execution".into(),
                        )
                    })?;
                Ok(DistinctKey::Bool(values.value(index)))
            }
            DataType::Date32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "COUNT(DISTINCT) expected a DATE32 column in execution".into(),
                        )
                    })?;
                Ok(DistinctKey::Date(values.value(index)))
            }
            DataType::Decimal128(_, _) => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::Decimal128Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "COUNT(DISTINCT) expected a DECIMAL128 column in execution".into(),
                        )
                    })?;
                Ok(DistinctKey::Decimal(values.value(index)))
            }
            // Null type can occur when all values are NULL - treat as Int for accumulator purposes
            // The actual value will always be NULL and won't be counted as distinct
            DataType::Null => Ok(DistinctKey::Int(0)),
            other => Err(Error::InvalidArgumentError(format!(
                "COUNT(DISTINCT) is not supported for column type {other:?}"
            ))),
        }
    }
}

/// Helper function to convert an array value to a string representation.
///
/// # Arguments
///
/// - `array`: The array to extract the value from
/// - `index`: The row index
///
/// # Errors
///
/// Returns an error if the type is unsupported or conversion fails.
fn array_value_to_string(array: &ArrayRef, index: usize) -> AggregateResult<String> {
    match array.data_type() {
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::InvalidArgumentError("Expected String array".into()))?;
            Ok(arr.value(index).to_string())
        }
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::InvalidArgumentError("Expected Int64 array".into()))?;
            Ok(arr.value(index).to_string())
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| Error::InvalidArgumentError("Expected Float64 array".into()))?;
            Ok(arr.value(index).to_string())
        }
        DataType::Boolean => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| Error::InvalidArgumentError("Expected Boolean array".into()))?;
            Ok(if arr.value(index) {
                "1".to_string()
            } else {
                "0".to_string()
            })
        }
        other => Err(Error::InvalidArgumentError(format!(
            "group_concat does not support column type {other:?}"
        ))),
    }
}

/// Helper function to extract a numeric f64 value from an array with SQLite-style type coercion.
///
/// SQLite behavior: String and BLOB values that do not look like numbers are interpreted as 0.
/// This function implements that coercion for SUM and AVG operations on string columns.
///
/// # Arguments
///
/// - `array`: The array to extract the value from
/// - `index`: The row index
///
/// # Returns
///
/// Returns the numeric value as f64. Non-numeric strings return 0.0.
fn array_value_to_numeric(array: &ArrayRef, index: usize) -> AggregateResult<f64> {
    match array.data_type() {
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::InvalidArgumentError("Expected Int64 array".into()))?;
            Ok(arr.value(index) as f64)
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| Error::InvalidArgumentError("Expected Float64 array".into()))?;
            Ok(arr.value(index))
        }
        DataType::Decimal128(_, scale) => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow::array::Decimal128Array>()
                .ok_or_else(|| Error::InvalidArgumentError("Expected Decimal128 array".into()))?;
            let value_i128 = arr.value(index);
            // Convert decimal to f64: value / 10^scale
            let scale_factor = 10_f64.powi(*scale as i32);
            Ok(value_i128 as f64 / scale_factor)
        }
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::InvalidArgumentError("Expected String array".into()))?;
            let s = arr.value(index);
            // SQLite behavior: try to parse as number, if it fails use 0.0
            Ok(s.trim().parse::<f64>().unwrap_or(0.0))
        }
        DataType::Boolean => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| Error::InvalidArgumentError("Expected Boolean array".into()))?;
            Ok(if arr.value(index) { 1.0 } else { 0.0 })
        }
        // Null type can occur when all values are NULL - return 0.0 as default
        // (though these values won't actually be accumulated since they're NULL)
        DataType::Null => Ok(0.0),
        other => Err(Error::InvalidArgumentError(format!(
            "Numeric coercion not supported for column type {other:?}"
        ))),
    }
}

impl AggregateAccumulator {
    /// Creates an accumulator using the projection index from a batch.
    ///
    /// # Arguments
    ///
    /// - `spec`: Aggregate specification defining the operation type
    /// - `projection_idx`: Column position in the batch; `None` for `CountStar`
    /// - `_total_rows_hint`: Unused optimization hint for row count
    ///
    /// # Errors
    ///
    /// Returns an error if the aggregate kind requires a projection index but none is provided.
    pub fn new_with_projection_index(
        spec: &AggregateSpec,
        projection_idx: Option<usize>,
        _total_rows_hint: Option<i64>,
    ) -> AggregateResult<Self> {
        match &spec.kind {
            AggregateKind::Count { field_id, distinct } => {
                if field_id.is_none() {
                    return Ok(AggregateAccumulator::CountStar { value: 0 });
                }
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("Count aggregate requires projection index".into())
                })?;
                if *distinct {
                    Ok(AggregateAccumulator::CountDistinctColumn {
                        column_index: idx,
                        seen: FxHashSet::default(),
                    })
                } else {
                    Ok(AggregateAccumulator::CountColumn {
                        column_index: idx,
                        value: 0,
                    })
                }
            }
            AggregateKind::Sum {
                data_type,
                distinct,
                ..
            } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("Sum aggregate requires projection index".into())
                })?;
                match (data_type, *distinct) {
                    (&DataType::Int64, true) => Ok(AggregateAccumulator::SumDistinctInt64 {
                        column_index: idx,
                        sum: Some(0),
                        seen: FxHashSet::default(),
                    }),
                    (&DataType::Int64, false) => Ok(AggregateAccumulator::SumInt64 {
                        column_index: idx,
                        value: Some(0),
                        has_values: false,
                    }),
                    (&DataType::Decimal128(precision, scale), true) => {
                        Ok(AggregateAccumulator::SumDistinctDecimal128 {
                            column_index: idx,
                            sum: 0,
                            seen: FxHashSet::default(),
                            precision,
                            scale,
                        })
                    }
                    (&DataType::Decimal128(precision, scale), false) => {
                        Ok(AggregateAccumulator::SumDecimal128 {
                            column_index: idx,
                            sum: 0,
                            precision,
                            scale,
                        })
                    }
                    // For Float64 and Utf8, use Float64 accumulator with numeric coercion
                    (&DataType::Float64, true) | (&DataType::Utf8, true) => {
                        Ok(AggregateAccumulator::SumDistinctFloat64 {
                            column_index: idx,
                            sum: 0.0,
                            seen: FxHashSet::default(),
                        })
                    }
                    (&DataType::Float64, false) | (&DataType::Utf8, false) => {
                        Ok(AggregateAccumulator::SumFloat64 {
                            column_index: idx,
                            value: 0.0,
                            saw_value: false,
                        })
                    }
                    other => Err(Error::InvalidArgumentError(format!(
                        "SUM aggregate not supported for column type {:?}",
                        other.0
                    ))),
                }
            }
            AggregateKind::Total {
                data_type,
                distinct,
                ..
            } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("Total aggregate requires projection index".into())
                })?;
                match (data_type, *distinct) {
                    (&DataType::Int64, true) => Ok(AggregateAccumulator::TotalDistinctInt64 {
                        column_index: idx,
                        sum: 0.0,
                        seen: FxHashSet::default(),
                    }),
                    (&DataType::Int64, false) => Ok(AggregateAccumulator::TotalInt64 {
                        column_index: idx,
                        value: 0.0,
                    }),
                    (&DataType::Decimal128(precision, scale), true) => {
                        Ok(AggregateAccumulator::TotalDistinctDecimal128 {
                            column_index: idx,
                            sum: 0,
                            seen: FxHashSet::default(),
                            precision,
                            scale,
                        })
                    }
                    (&DataType::Decimal128(precision, scale), false) => {
                        Ok(AggregateAccumulator::TotalDecimal128 {
                            column_index: idx,
                            sum: 0,
                            precision,
                            scale,
                        })
                    }
                    // For Float64 and Utf8, use Float64 accumulator with numeric coercion
                    (&DataType::Float64, true) | (&DataType::Utf8, true) => {
                        Ok(AggregateAccumulator::TotalDistinctFloat64 {
                            column_index: idx,
                            sum: 0.0,
                            seen: FxHashSet::default(),
                        })
                    }
                    (&DataType::Float64, false) | (&DataType::Utf8, false) => {
                        Ok(AggregateAccumulator::TotalFloat64 {
                            column_index: idx,
                            value: 0.0,
                        })
                    }
                    other => Err(Error::InvalidArgumentError(format!(
                        "TOTAL aggregate not supported for column type {:?}",
                        other.0
                    ))),
                }
            }
            AggregateKind::Avg {
                data_type,
                distinct,
                ..
            } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("Avg aggregate requires projection index".into())
                })?;
                match (data_type, *distinct) {
                    (&DataType::Int64, true) => Ok(AggregateAccumulator::AvgDistinctInt64 {
                        column_index: idx,
                        sum: 0,
                        seen: FxHashSet::default(),
                    }),
                    (&DataType::Int64, false) => Ok(AggregateAccumulator::AvgInt64 {
                        column_index: idx,
                        sum: 0,
                        count: 0,
                    }),
                    (&DataType::Decimal128(precision, scale), true) => {
                        Ok(AggregateAccumulator::AvgDistinctDecimal128 {
                            column_index: idx,
                            sum: 0,
                            seen: FxHashSet::default(),
                            precision,
                            scale,
                        })
                    }
                    (&DataType::Decimal128(precision, scale), false) => {
                        Ok(AggregateAccumulator::AvgDecimal128 {
                            column_index: idx,
                            sum: 0,
                            count: 0,
                            precision,
                            scale,
                        })
                    }
                    // For Float64 and Utf8, use Float64 accumulator with numeric coercion
                    (&DataType::Float64, true) | (&DataType::Utf8, true) => {
                        Ok(AggregateAccumulator::AvgDistinctFloat64 {
                            column_index: idx,
                            sum: 0.0,
                            seen: FxHashSet::default(),
                        })
                    }
                    (&DataType::Float64, false) | (&DataType::Utf8, false) => {
                        Ok(AggregateAccumulator::AvgFloat64 {
                            column_index: idx,
                            sum: 0.0,
                            count: 0,
                        })
                    }
                    other => Err(Error::InvalidArgumentError(format!(
                        "AVG aggregate not supported for column type {:?}",
                        other.0
                    ))),
                }
            }
            AggregateKind::Min { data_type, .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("Min aggregate requires projection index".into())
                })?;
                match data_type {
                    &DataType::Int64 => Ok(AggregateAccumulator::MinInt64 {
                        column_index: idx,
                        value: None,
                    }),
                    &DataType::Decimal128(precision, scale) => {
                        Ok(AggregateAccumulator::MinDecimal128 {
                            column_index: idx,
                            value: None,
                            precision,
                            scale,
                        })
                    }
                    // For Float64 and Utf8, use Float64 accumulator with numeric coercion
                    &DataType::Float64 | &DataType::Utf8 => Ok(AggregateAccumulator::MinFloat64 {
                        column_index: idx,
                        value: None,
                    }),
                    other => Err(Error::InvalidArgumentError(format!(
                        "MIN aggregate not supported for column type {:?}",
                        other
                    ))),
                }
            }
            AggregateKind::Max { data_type, .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("Max aggregate requires projection index".into())
                })?;
                match data_type {
                    &DataType::Int64 => Ok(AggregateAccumulator::MaxInt64 {
                        column_index: idx,
                        value: None,
                    }),
                    &DataType::Decimal128(precision, scale) => {
                        Ok(AggregateAccumulator::MaxDecimal128 {
                            column_index: idx,
                            value: None,
                            precision,
                            scale,
                        })
                    }
                    // For Float64 and Utf8, use Float64 accumulator with numeric coercion
                    &DataType::Float64 | &DataType::Utf8 => Ok(AggregateAccumulator::MaxFloat64 {
                        column_index: idx,
                        value: None,
                    }),
                    other => Err(Error::InvalidArgumentError(format!(
                        "MAX aggregate not supported for column type {:?}",
                        other
                    ))),
                }
            }
            AggregateKind::CountNulls { .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("CountNulls aggregate requires projection index".into())
                })?;
                Ok(AggregateAccumulator::CountNulls {
                    column_index: idx,
                    non_null_rows: 0,
                    total_rows_seen: 0,
                })
            }
            AggregateKind::GroupConcat {
                distinct,
                separator,
                ..
            } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("GroupConcat aggregate requires projection index".into())
                })?;
                if *distinct {
                    Ok(AggregateAccumulator::GroupConcatDistinct {
                        column_index: idx,
                        seen: FxHashSet::default(),
                        values: Vec::new(),
                        separator: separator.clone(),
                    })
                } else {
                    Ok(AggregateAccumulator::GroupConcat {
                        column_index: idx,
                        values: Vec::new(),
                        separator: separator.clone(),
                    })
                }
            }
        }
    }

    /// Updates the accumulator with values from a new batch.
    ///
    /// # Arguments
    ///
    /// - `batch`: RecordBatch containing the column data to aggregate
    ///
    /// # Errors
    ///
    /// Returns an error if column types mismatch, overflow occurs, or downcast fails.
    pub fn update(&mut self, batch: &RecordBatch) -> AggregateResult<()> {
        match self {
            AggregateAccumulator::CountStar { value } => {
                let rows = i64::try_from(batch.num_rows()).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *value = value.checked_add(rows).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
            }
            AggregateAccumulator::CountColumn {
                column_index,
                value,
            } => {
                let array = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                // COUNT(NULL column) should be 0, not the row count
                if matches!(array.data_type(), DataType::Null) {
                    return Ok(());
                }
                let non_null = (array.len() - array.null_count()) as i64;
                *value = value.checked_add(non_null).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
            }
            AggregateAccumulator::CountDistinctColumn { column_index, seen } => {
                let array = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(array.data_type(), DataType::Null) {
                    return Ok(());
                }
                for i in 0..array.len() {
                    if !array.is_valid(i) {
                        continue;
                    }
                    let value = DistinctKey::from_array(array, i)?;
                    seen.insert(value);
                }
            }
            AggregateAccumulator::SumInt64 {
                column_index,
                value,
                has_values,
            } => {
                let array = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(array.data_type(), DataType::Null) {
                    return Ok(());
                }
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "SUM aggregate expected an INT column in execution".into(),
                    )
                })?;
                
                if let Some(s) = compute::sum(array) {
                    *has_values = true;
                    *value = match *value {
                        Some(current) => Some(current.checked_add(s).ok_or_else(|| {
                            Error::InvalidArgumentError("integer overflow".into())
                        })?),
                        None => Some(s),
                    };
                }
            }
            AggregateAccumulator::SumDistinctInt64 {
                column_index,
                sum,
                seen,
            } => {
                let array = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(array.data_type(), DataType::Null) {
                    return Ok(());
                }
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "SUM(DISTINCT) aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let col_array = batch.column(*column_index);
                        let key = DistinctKey::from_array(col_array, i)?;
                        if seen.insert(key.clone()) {
                            // Only add to sum if we haven't seen this value before
                            if let DistinctKey::Int(v) = key {
                                *sum = match *sum {
                                    Some(current) => {
                                        Some(current.checked_add(v).ok_or_else(|| {
                                            Error::InvalidArgumentError("integer overflow".into())
                                        })?)
                                    }
                                    None => {
                                        return Err(Error::InvalidArgumentError(
                                            "integer overflow".into(),
                                        ));
                                    }
                                };
                            }
                        }
                    }
                }
            }
            AggregateAccumulator::SumFloat64 {
                column_index,
                value,
                saw_value,
            } => {
                let column = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(column.data_type(), DataType::Null) {
                    return Ok(());
                }
                
                if let Some(array) = column.as_any().downcast_ref::<Float64Array>() {
                    if let Some(s) = compute::sum(array) {
                        *value += s;
                        *saw_value = true;
                    }
                } else {
                    // Use generic numeric coercion to support Utf8 and other types
                    for i in 0..column.len() {
                        if column.is_valid(i) {
                            let v = array_value_to_numeric(column, i)?;
                            *value += v;
                            *saw_value = true;
                        }
                    }
                }
            }
            AggregateAccumulator::SumDistinctFloat64 {
                column_index,
                sum,
                seen,
            } => {
                let column = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(column.data_type(), DataType::Null) {
                    return Ok(());
                }
                for i in 0..column.len() {
                    if !column.is_valid(i) {
                        continue;
                    }
                    // Track distinctness based on original value (string, number, etc.)
                    let key = DistinctKey::from_array(column, i)?;
                    if seen.insert(key.clone()) {
                        // Convert to numeric using SQLite-style coercion
                        let v = match key {
                            DistinctKey::Float(bits) => f64::from_bits(bits),
                            DistinctKey::Int(int_val) => int_val as f64,
                            DistinctKey::Str(_) => array_value_to_numeric(column, i)?,
                            DistinctKey::Bool(b) => {
                                if b {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            DistinctKey::Date(d) => d as f64,
                            DistinctKey::Decimal(_) => array_value_to_numeric(column, i)?,
                        };
                        *sum += v;
                    }
                }
            }
            AggregateAccumulator::SumDecimal128 {
                column_index, sum, ..
            } => {
                let column = batch.column(*column_index);
                let arr = column
                    .as_any()
                    .downcast_ref::<arrow::array::Decimal128Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError("Expected Decimal128 array".into())
                    })?;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        *sum = sum.checked_add(arr.value(i)).ok_or_else(|| {
                            Error::InvalidArgumentError("Decimal128 sum overflow".into())
                        })?;
                    }
                }
            }
            AggregateAccumulator::SumDistinctDecimal128 {
                column_index,
                sum,
                seen,
                ..
            } => {
                let column = batch.column(*column_index);
                let arr = column
                    .as_any()
                    .downcast_ref::<arrow::array::Decimal128Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError("Expected Decimal128 array".into())
                    })?;
                for i in 0..arr.len() {
                    if !arr.is_valid(i) {
                        continue;
                    }
                    let key = DistinctKey::from_array(column, i)?;
                    if seen.insert(key) {
                        *sum = sum.checked_add(arr.value(i)).ok_or_else(|| {
                            Error::InvalidArgumentError("Decimal128 sum overflow".into())
                        })?;
                    }
                }
            }
            AggregateAccumulator::TotalInt64 {
                column_index,
                value,
            } => {
                let array = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(array.data_type(), DataType::Null) {
                    return Ok(());
                }
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "TOTAL aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let v = array.value(i);
                        // TOTAL never overflows - accumulate as float
                        *value += v as f64;
                    }
                }
            }
            AggregateAccumulator::TotalDistinctInt64 {
                column_index,
                sum,
                seen,
            } => {
                let array = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(array.data_type(), DataType::Null) {
                    return Ok(());
                }
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "TOTAL(DISTINCT) aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let col_array = batch.column(*column_index);
                        let key = DistinctKey::from_array(col_array, i)?;
                        if seen.insert(key.clone())
                            && let DistinctKey::Int(v) = key
                        {
                            // TOTAL never overflows - accumulate as float
                            *sum += v as f64;
                        }
                    }
                }
            }
            AggregateAccumulator::TotalFloat64 {
                column_index,
                value,
            } => {
                let column = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(column.data_type(), DataType::Null) {
                    return Ok(());
                }
                // Use generic numeric coercion to support Utf8 and other types
                for i in 0..column.len() {
                    if column.is_valid(i) {
                        let v = array_value_to_numeric(column, i)?;
                        *value += v;
                    }
                }
            }
            AggregateAccumulator::TotalDistinctFloat64 {
                column_index,
                sum,
                seen,
            } => {
                let column = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(column.data_type(), DataType::Null) {
                    return Ok(());
                }
                for i in 0..column.len() {
                    if !column.is_valid(i) {
                        continue;
                    }
                    // Track distinctness based on original value (string, number, etc.)
                    let key = DistinctKey::from_array(column, i)?;
                    if seen.insert(key.clone()) {
                        // Convert to numeric using SQLite-style coercion
                        let v = match key {
                            DistinctKey::Float(bits) => f64::from_bits(bits),
                            DistinctKey::Int(int_val) => int_val as f64,
                            DistinctKey::Str(_) => array_value_to_numeric(column, i)?,
                            DistinctKey::Bool(b) => {
                                if b {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            DistinctKey::Date(d) => d as f64,
                            DistinctKey::Decimal(_) => array_value_to_numeric(column, i)?,
                        };
                        *sum += v;
                    }
                }
            }
            AggregateAccumulator::TotalDecimal128 {
                column_index, sum, ..
            } => {
                let column = batch.column(*column_index);
                let arr = column
                    .as_any()
                    .downcast_ref::<arrow::array::Decimal128Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError("Expected Decimal128 array".into())
                    })?;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        *sum = sum.checked_add(arr.value(i)).ok_or_else(|| {
                            Error::InvalidArgumentError("Decimal128 total overflow".into())
                        })?;
                    }
                }
            }
            AggregateAccumulator::TotalDistinctDecimal128 {
                column_index,
                sum,
                seen,
                ..
            } => {
                let column = batch.column(*column_index);
                let arr = column
                    .as_any()
                    .downcast_ref::<arrow::array::Decimal128Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError("Expected Decimal128 array".into())
                    })?;
                for i in 0..arr.len() {
                    if !arr.is_valid(i) {
                        continue;
                    }
                    let key = DistinctKey::from_array(column, i)?;
                    if seen.insert(key) {
                        *sum = sum.checked_add(arr.value(i)).ok_or_else(|| {
                            Error::InvalidArgumentError("Decimal128 total overflow".into())
                        })?;
                    }
                }
            }
            AggregateAccumulator::AvgInt64 {
                column_index,
                sum,
                count,
            } => {
                let array = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(array.data_type(), DataType::Null) {
                    return Ok(());
                }
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "AVG aggregate expected an INT column in execution".into(),
                    )
                })?;
                
                if let Some(s) = compute::sum(array) {
                    *sum = sum.checked_add(s).ok_or_else(|| {
                        Error::InvalidArgumentError("AVG aggregate sum exceeds i64 range".into())
                    })?;
                    let c = (array.len() - array.null_count()) as i64;
                    *count = count.checked_add(c).ok_or_else(|| {
                        Error::InvalidArgumentError("AVG aggregate count exceeds i64 range".into())
                    })?;
                }
            }
            AggregateAccumulator::AvgDistinctInt64 {
                column_index,
                sum,
                seen,
            } => {
                let array = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(array.data_type(), DataType::Null) {
                    return Ok(());
                }
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "AVG(DISTINCT) aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let col_array = batch.column(*column_index);
                        let key = DistinctKey::from_array(col_array, i)?;
                        if seen.insert(key.clone()) {
                            // Only add to sum if we haven't seen this value before
                            if let DistinctKey::Int(v) = key {
                                *sum = sum.checked_add(v).ok_or_else(|| {
                                    Error::InvalidArgumentError(
                                        "AVG(DISTINCT) aggregate sum exceeds i64 range".into(),
                                    )
                                })?;
                            }
                        }
                    }
                }
            }
            AggregateAccumulator::AvgFloat64 {
                column_index,
                sum,
                count,
            } => {
                let column = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(column.data_type(), DataType::Null) {
                    return Ok(());
                }
                
                if let Some(array) = column.as_any().downcast_ref::<Float64Array>() {
                    if let Some(s) = compute::sum(array) {
                        *sum += s;
                        let c = (array.len() - array.null_count()) as i64;
                        *count = count.checked_add(c).ok_or_else(|| {
                            Error::InvalidArgumentError("AVG aggregate count exceeds i64 range".into())
                        })?;
                    }
                } else {
                    // Use generic numeric coercion to support Utf8 and other types
                    for i in 0..column.len() {
                        if column.is_valid(i) {
                            let v = array_value_to_numeric(column, i)?;
                            *sum += v;
                            *count = count.checked_add(1).ok_or_else(|| {
                                Error::InvalidArgumentError(
                                    "AVG aggregate count exceeds i64 range".into(),
                                )
                            })?;
                        }
                    }
                }
            }
            AggregateAccumulator::AvgDistinctFloat64 {
                column_index,
                sum,
                seen,
            } => {
                let column = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(column.data_type(), DataType::Null) {
                    return Ok(());
                }
                for i in 0..column.len() {
                    if !column.is_valid(i) {
                        continue;
                    }
                    // Track distinctness based on original value (string, number, etc.)
                    let key = DistinctKey::from_array(column, i)?;
                    if seen.insert(key.clone()) {
                        // Convert to numeric using SQLite-style coercion
                        let v = match key {
                            DistinctKey::Float(bits) => f64::from_bits(bits),
                            DistinctKey::Int(int_val) => int_val as f64,
                            DistinctKey::Str(_) => array_value_to_numeric(column, i)?,
                            DistinctKey::Bool(b) => {
                                if b {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            DistinctKey::Date(d) => d as f64,
                            DistinctKey::Decimal(_) => array_value_to_numeric(column, i)?,
                        };
                        *sum += v;
                    }
                }
            }
            AggregateAccumulator::AvgDecimal128 {
                column_index,
                sum,
                count,
                ..
            } => {
                let column = batch.column(*column_index);
                let arr = column
                    .as_any()
                    .downcast_ref::<arrow::array::Decimal128Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError("Expected Decimal128 array".into())
                    })?;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        *sum = sum.checked_add(arr.value(i)).ok_or_else(|| {
                            Error::InvalidArgumentError("Decimal128 sum overflow".into())
                        })?;
                        *count = count.checked_add(1).ok_or_else(|| {
                            Error::InvalidArgumentError("AVG count overflow".into())
                        })?;
                    }
                }
            }
            AggregateAccumulator::AvgDistinctDecimal128 {
                column_index,
                sum,
                seen,
                ..
            } => {
                let column = batch.column(*column_index);
                let arr = column
                    .as_any()
                    .downcast_ref::<arrow::array::Decimal128Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError("Expected Decimal128 array".into())
                    })?;
                for i in 0..arr.len() {
                    if !arr.is_valid(i) {
                        continue;
                    }
                    let key = DistinctKey::from_array(column, i)?;
                    if seen.insert(key) {
                        *sum = sum.checked_add(arr.value(i)).ok_or_else(|| {
                            Error::InvalidArgumentError("Decimal128 sum overflow".into())
                        })?;
                    }
                }
            }
            AggregateAccumulator::MinInt64 {
                column_index,
                value,
            } => {
                let array = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(array.data_type(), DataType::Null) {
                    return Ok(());
                }
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "MIN aggregate expected an INT column in execution".into(),
                    )
                })?;
                
                if let Some(m) = compute::min(array) {
                    *value = Some(match *value {
                        Some(current) => current.min(m),
                        None => m,
                    });
                }
            }
            AggregateAccumulator::MinFloat64 {
                column_index,
                value,
            } => {
                let column = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(column.data_type(), DataType::Null) {
                    return Ok(());
                }
                
                if let Some(array) = column.as_any().downcast_ref::<Float64Array>() {
                    if let Some(m) = compute::min(array) {
                        *value = Some(match *value {
                            Some(current) => match m.partial_cmp(&current) {
                                Some(Ordering::Less) => m,
                                _ => current,
                            },
                            None => m,
                        });
                    }
                } else {
                    // Use generic numeric coercion to support Utf8 and other types
                    for i in 0..column.len() {
                        if column.is_valid(i) {
                            let v = array_value_to_numeric(column, i)?;
                            *value = Some(match *value {
                                Some(current) => match v.partial_cmp(&current) {
                                    Some(Ordering::Less) => v,
                                    _ => current,
                                },
                                None => v,
                            });
                        }
                    }
                }
            }
            AggregateAccumulator::MinDecimal128 {
                column_index,
                value,
                ..
            } => {
                let column = batch.column(*column_index);
                let arr = column
                    .as_any()
                    .downcast_ref::<arrow::array::Decimal128Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError("Expected Decimal128 array".into())
                    })?;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let v = arr.value(i);
                        *value = Some(match *value {
                            Some(current) => current.min(v),
                            None => v,
                        });
                    }
                }
            }
            AggregateAccumulator::MaxInt64 {
                column_index,
                value,
            } => {
                let array = batch.column(*column_index);
                if matches!(array.data_type(), DataType::Null) {
                    return Ok(());
                }
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "MAX aggregate expected an INT column in execution".into(),
                    )
                })?;
                
                if let Some(m) = compute::max(array) {
                    *value = Some(match *value {
                        Some(current) => current.max(m),
                        None => m,
                    });
                }
            }
            AggregateAccumulator::MaxFloat64 {
                column_index,
                value,
            } => {
                let column = batch.column(*column_index);
                // Skip accumulation for Null-type columns - all values are implicitly NULL
                if matches!(column.data_type(), DataType::Null) {
                    return Ok(());
                }
                
                if let Some(array) = column.as_any().downcast_ref::<Float64Array>() {
                    if let Some(m) = compute::max(array) {
                        *value = Some(match *value {
                            Some(current) => match m.partial_cmp(&current) {
                                Some(Ordering::Greater) => m,
                                _ => current,
                            },
                            None => m,
                        });
                    }
                } else {
                    // Use generic numeric coercion to support Utf8 and other types
                    for i in 0..column.len() {
                        if column.is_valid(i) {
                            let v = array_value_to_numeric(column, i)?;
                            *value = Some(match *value {
                                Some(current) => match v.partial_cmp(&current) {
                                    Some(Ordering::Greater) => v,
                                    _ => current,
                                },
                                None => v,
                            });
                        }
                    }
                }
            }
            AggregateAccumulator::MaxDecimal128 {
                column_index,
                value,
                ..
            } => {
                let column = batch.column(*column_index);
                let arr = column
                    .as_any()
                    .downcast_ref::<arrow::array::Decimal128Array>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError("Expected Decimal128 array".into())
                    })?;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let v = arr.value(i);
                        *value = Some(match *value {
                            Some(current) => current.max(v),
                            None => v,
                        });
                    }
                }
            }
            AggregateAccumulator::CountNulls {
                column_index,
                non_null_rows,
                total_rows_seen,
            } => {
                let rows = i64::try_from(batch.num_rows()).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *total_rows_seen = total_rows_seen.checked_add(rows).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;

                let array = batch.column(*column_index);
                // If column is Null type, non_null_rows doesn't increase
                if !matches!(array.data_type(), DataType::Null) {
                    let non_null = (0..array.len()).filter(|idx| array.is_valid(*idx)).count();
                    let non_null = i64::try_from(non_null).map_err(|_| {
                        Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                    })?;
                    *non_null_rows = non_null_rows.checked_add(non_null).ok_or_else(|| {
                        Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                    })?;
                }
            }
            AggregateAccumulator::GroupConcat {
                column_index,
                values,
                separator: _,
            } => {
                let array = batch.column(*column_index);
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let str_val = array_value_to_string(array, i)?;
                        values.push(str_val);
                    }
                }
            }
            AggregateAccumulator::GroupConcatDistinct {
                column_index,
                seen,
                values,
                separator: _,
            } => {
                let array = batch.column(*column_index);
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let str_val = array_value_to_string(array, i)?;
                        if seen.insert(str_val.clone()) {
                            values.push(str_val);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Returns the output field definition for this accumulator.
    pub fn output_field(&self) -> Field {
        match self {
            AggregateAccumulator::CountStar { .. } => Field::new("count", DataType::Int64, false),
            AggregateAccumulator::CountColumn { .. } => Field::new("count", DataType::Int64, false),
            AggregateAccumulator::CountDistinctColumn { .. } => Field::new("count_distinct", DataType::Int64, false),
            AggregateAccumulator::SumInt64 { .. } => Field::new("sum", DataType::Int64, true),
            AggregateAccumulator::SumDistinctInt64 { .. } => Field::new("sum_distinct", DataType::Int64, true),
            AggregateAccumulator::SumFloat64 { .. } => Field::new("sum", DataType::Float64, true),
            AggregateAccumulator::SumDistinctFloat64 { .. } => Field::new("sum_distinct", DataType::Float64, true),
            AggregateAccumulator::SumDecimal128 { precision, scale, .. } => {
                Field::new("sum", DataType::Decimal128(*precision, *scale), true)
            }
            AggregateAccumulator::SumDistinctDecimal128 { precision, scale, .. } => {
                Field::new("sum_distinct", DataType::Decimal128(*precision, *scale), true)
            }
            AggregateAccumulator::TotalInt64 { .. } => Field::new("total", DataType::Float64, false),
            AggregateAccumulator::TotalDistinctInt64 { .. } => Field::new("total_distinct", DataType::Float64, false),
            AggregateAccumulator::TotalFloat64 { .. } => Field::new("total", DataType::Float64, false),
            AggregateAccumulator::TotalDistinctFloat64 { .. } => Field::new("total_distinct", DataType::Float64, false),
            AggregateAccumulator::TotalDecimal128 { precision, scale, .. } => {
                Field::new("total", DataType::Decimal128(*precision, *scale), false)
            }
            AggregateAccumulator::TotalDistinctDecimal128 { precision, scale, .. } => {
                Field::new("total_distinct", DataType::Decimal128(*precision, *scale), false)
            }
            AggregateAccumulator::AvgInt64 { .. } => Field::new("avg", DataType::Float64, true),
            AggregateAccumulator::AvgDistinctInt64 { .. } => Field::new("avg_distinct", DataType::Float64, true),
            AggregateAccumulator::AvgFloat64 { .. } => Field::new("avg", DataType::Float64, true),
            AggregateAccumulator::AvgDistinctFloat64 { .. } => Field::new("avg_distinct", DataType::Float64, true),
            AggregateAccumulator::AvgDecimal128 { precision, scale, .. } => {
                Field::new("avg", DataType::Decimal128(*precision, *scale), true)
            }
            AggregateAccumulator::AvgDistinctDecimal128 { precision, scale, .. } => {
                Field::new("avg_distinct", DataType::Decimal128(*precision, *scale), true)
            }
            AggregateAccumulator::MinInt64 { .. } => Field::new("min", DataType::Int64, true),
            AggregateAccumulator::MinFloat64 { .. } => Field::new("min", DataType::Float64, true),
            AggregateAccumulator::MinDecimal128 { precision, scale, .. } => {
                Field::new("min", DataType::Decimal128(*precision, *scale), true)
            }
            AggregateAccumulator::MaxInt64 { .. } => Field::new("max", DataType::Int64, true),
            AggregateAccumulator::MaxFloat64 { .. } => Field::new("max", DataType::Float64, true),
            AggregateAccumulator::MaxDecimal128 { precision, scale, .. } => {
                Field::new("max", DataType::Decimal128(*precision, *scale), true)
            }
            AggregateAccumulator::CountNulls { .. } => Field::new("count_nulls", DataType::Int64, false),
            AggregateAccumulator::GroupConcat { .. } => Field::new("group_concat", DataType::Utf8, true),
            AggregateAccumulator::GroupConcatDistinct { .. } => Field::new("group_concat_distinct", DataType::Utf8, true),
        }
    }

    /// Finalizes the accumulator and produces the resulting field and array.
    ///
    /// # Returns
    ///
    /// A tuple containing the output field schema and the computed aggregate value array.
    ///
    /// # Errors
    ///
    /// Returns an error if result conversion to i64 fails or arithmetic underflow occurs.
    pub fn finalize(self) -> AggregateResult<(Field, ArrayRef)> {
        match self {
            AggregateAccumulator::CountStar { value } => {
                let mut builder = Int64Builder::with_capacity(1);
                builder.append_value(value);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("count", DataType::Int64, false), array))
            }
            AggregateAccumulator::CountColumn { value, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                builder.append_value(value);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("count", DataType::Int64, false), array))
            }
            AggregateAccumulator::CountDistinctColumn { seen, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                let count = i64::try_from(seen.len()).map_err(|_| {
                    Error::InvalidArgumentError("COUNT(DISTINCT) result exceeds i64 range".into())
                })?;
                builder.append_value(count);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("count_distinct", DataType::Int64, false), array))
            }
            AggregateAccumulator::SumInt64 {
                value, has_values, ..
            } => {
                // If overflow occurred (value is None after seeing values), return error
                // to match SQLite behavior where integer overflow in SUM throws exception
                if has_values && value.is_none() {
                    return Err(Error::InvalidArgumentError("integer overflow".into()));
                }

                let mut builder = Int64Builder::with_capacity(1);
                if !has_values {
                    builder.append_null(); // No values seen
                } else {
                    match value {
                        Some(v) => builder.append_value(v),
                        None => unreachable!(), // Already handled above
                    }
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("sum", DataType::Int64, true), array))
            }
            AggregateAccumulator::SumDistinctInt64 { sum, seen, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                if seen.is_empty() {
                    builder.append_null();
                } else {
                    match sum {
                        Some(v) => builder.append_value(v),
                        None => builder.append_null(), // Overflow occurred
                    }
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("sum_distinct", DataType::Int64, true), array))
            }
            AggregateAccumulator::SumFloat64 {
                value, saw_value, ..
            } => {
                let mut builder = Float64Builder::with_capacity(1);
                if saw_value {
                    builder.append_value(value);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("sum", DataType::Float64, true), array))
            }
            AggregateAccumulator::SumDistinctFloat64 { sum, seen, .. } => {
                let mut builder = Float64Builder::with_capacity(1);
                if !seen.is_empty() {
                    builder.append_value(sum);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("sum_distinct", DataType::Float64, true), array))
            }
            AggregateAccumulator::SumDecimal128 {
                sum,
                precision,
                scale,
                ..
            } => {
                let data_type = DataType::Decimal128(precision, scale);
                let array = Arc::new(
                    arrow::array::Decimal128Array::from(vec![sum])
                        .with_precision_and_scale(precision, scale)
                        .map_err(|e| {
                            Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                        })?,
                ) as ArrayRef;
                Ok((Field::new("sum", data_type, true), array))
            }
            AggregateAccumulator::SumDistinctDecimal128 {
                sum,
                seen,
                precision,
                scale,
                ..
            } => {
                let data_type = DataType::Decimal128(precision, scale);
                let array = if seen.is_empty() {
                    Arc::new(
                        arrow::array::Decimal128Array::from(vec![Option::<i128>::None])
                            .with_precision_and_scale(precision, scale)
                            .map_err(|e| {
                                Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                            })?,
                    ) as ArrayRef
                } else {
                    Arc::new(
                        arrow::array::Decimal128Array::from(vec![sum])
                            .with_precision_and_scale(precision, scale)
                            .map_err(|e| {
                                Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                            })?,
                    ) as ArrayRef
                };
                Ok((Field::new("sum_distinct", data_type, true), array))
            }
            AggregateAccumulator::TotalInt64 { value, .. } => {
                let mut builder = Float64Builder::with_capacity(1);
                builder.append_value(value);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("total", DataType::Float64, false), array))
            }
            AggregateAccumulator::TotalDistinctInt64 { sum, .. } => {
                let mut builder = Float64Builder::with_capacity(1);
                builder.append_value(sum);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((
                    Field::new("total_distinct", DataType::Float64, false),
                    array,
                ))
            }
            AggregateAccumulator::TotalFloat64 { value, .. } => {
                let mut builder = Float64Builder::with_capacity(1);
                builder.append_value(value);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("total", DataType::Float64, false), array))
            }
            AggregateAccumulator::TotalDistinctFloat64 { sum, .. } => {
                let mut builder = Float64Builder::with_capacity(1);
                builder.append_value(sum);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((
                    Field::new("total_distinct", DataType::Float64, false),
                    array,
                ))
            }
            AggregateAccumulator::TotalDecimal128 {
                sum,
                precision,
                scale,
                ..
            } => {
                let data_type = DataType::Decimal128(precision, scale);
                let array = Arc::new(
                    arrow::array::Decimal128Array::from(vec![sum])
                        .with_precision_and_scale(precision, scale)
                        .map_err(|e| {
                            Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                        })?,
                ) as ArrayRef;
                Ok((Field::new("total", data_type, false), array))
            }
            AggregateAccumulator::TotalDistinctDecimal128 {
                sum,
                precision,
                scale,
                ..
            } => {
                let data_type = DataType::Decimal128(precision, scale);
                let array = Arc::new(
                    arrow::array::Decimal128Array::from(vec![sum])
                        .with_precision_and_scale(precision, scale)
                        .map_err(|e| {
                            Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                        })?,
                ) as ArrayRef;
                Ok((Field::new("total_distinct", data_type, false), array))
            }
            AggregateAccumulator::AvgInt64 { sum, count, .. } => {
                let mut builder = Float64Builder::with_capacity(1);
                if count > 0 {
                    // Compute average as floating-point for SQL standard compatibility
                    let avg = (sum as f64) / (count as f64);
                    builder.append_value(avg);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("avg", DataType::Float64, true), array))
            }
            AggregateAccumulator::AvgDistinctInt64 { sum, seen, .. } => {
                let mut builder = Float64Builder::with_capacity(1);
                let count = seen.len();
                if count > 0 {
                    // Compute average as floating-point for SQL standard compatibility
                    let avg = (sum as f64) / (count as f64);
                    builder.append_value(avg);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("avg_distinct", DataType::Float64, true), array))
            }
            AggregateAccumulator::AvgFloat64 { sum, count, .. } => {
                let mut builder = Float64Builder::with_capacity(1);
                if count > 0 {
                    let avg = sum / (count as f64);
                    builder.append_value(avg);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("avg", DataType::Float64, true), array))
            }
            AggregateAccumulator::AvgDistinctFloat64 { sum, seen, .. } => {
                let mut builder = Float64Builder::with_capacity(1);
                let count = seen.len();
                if count > 0 {
                    let avg = sum / (count as f64);
                    builder.append_value(avg);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("avg_distinct", DataType::Float64, true), array))
            }
            AggregateAccumulator::AvgDecimal128 {
                sum,
                count,
                precision,
                scale,
                ..
            } => {
                let data_type = DataType::Decimal128(precision, scale);
                let array = if count > 0 {
                    // Compute average in decimal space: sum / count
                    // Use rounding division instead of truncating division
                    let count_i128 = count as i128;
                    let mut avg = sum / count_i128;
                    let rem = sum % count_i128;

                    // Round half away from zero
                    if rem.abs() * 2 >= count_i128.abs() {
                        if sum.signum() == count_i128.signum() {
                            avg += 1;
                        } else {
                            avg -= 1;
                        }
                    }

                    Arc::new(
                        arrow::array::Decimal128Array::from(vec![avg])
                            .with_precision_and_scale(precision, scale)
                            .map_err(|e| {
                                Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                            })?,
                    ) as ArrayRef
                } else {
                    Arc::new(
                        arrow::array::Decimal128Array::from(vec![Option::<i128>::None])
                            .with_precision_and_scale(precision, scale)
                            .map_err(|e| {
                                Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                            })?,
                    ) as ArrayRef
                };
                Ok((Field::new("avg", data_type, true), array))
            }
            AggregateAccumulator::AvgDistinctDecimal128 {
                sum,
                seen,
                precision,
                scale,
                ..
            } => {
                let data_type = DataType::Decimal128(precision, scale);
                let count = seen.len();
                let array = if count > 0 {
                    // Compute average in decimal space: sum / count
                    // Use rounding division instead of truncating division
                    let count_i128 = count as i128;
                    let mut avg = sum / count_i128;
                    let rem = sum % count_i128;

                    // Round half away from zero
                    if rem.abs() * 2 >= count_i128.abs() {
                        if sum.signum() == count_i128.signum() {
                            avg += 1;
                        } else {
                            avg -= 1;
                        }
                    }

                    Arc::new(
                        arrow::array::Decimal128Array::from(vec![avg])
                            .with_precision_and_scale(precision, scale)
                            .map_err(|e| {
                                Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                            })?,
                    ) as ArrayRef
                } else {
                    Arc::new(
                        arrow::array::Decimal128Array::from(vec![Option::<i128>::None])
                            .with_precision_and_scale(precision, scale)
                            .map_err(|e| {
                                Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                            })?,
                    ) as ArrayRef
                };
                Ok((Field::new("avg_distinct", data_type, true), array))
            }
            AggregateAccumulator::MinInt64 { value, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                if let Some(v) = value {
                    builder.append_value(v);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("min", DataType::Int64, true), array))
            }
            AggregateAccumulator::MinFloat64 { value, .. } => {
                let mut builder = Float64Builder::with_capacity(1);
                if let Some(v) = value {
                    builder.append_value(v);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("min", DataType::Float64, true), array))
            }
            AggregateAccumulator::MinDecimal128 {
                value,
                precision,
                scale,
                ..
            } => {
                let data_type = DataType::Decimal128(precision, scale);
                let array = match value {
                    Some(v) => Arc::new(
                        arrow::array::Decimal128Array::from(vec![v])
                            .with_precision_and_scale(precision, scale)
                            .map_err(|e| {
                                Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                            })?,
                    ) as ArrayRef,
                    None => Arc::new(
                        arrow::array::Decimal128Array::from(vec![Option::<i128>::None])
                            .with_precision_and_scale(precision, scale)
                            .map_err(|e| {
                                Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                            })?,
                    ) as ArrayRef,
                };
                Ok((Field::new("min", data_type, true), array))
            }
            AggregateAccumulator::MaxInt64 { value, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                if let Some(v) = value {
                    builder.append_value(v);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("max", DataType::Int64, true), array))
            }
            AggregateAccumulator::MaxFloat64 { value, .. } => {
                let mut builder = Float64Builder::with_capacity(1);
                if let Some(v) = value {
                    builder.append_value(v);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("max", DataType::Float64, true), array))
            }
            AggregateAccumulator::MaxDecimal128 {
                value,
                precision,
                scale,
                ..
            } => {
                let data_type = DataType::Decimal128(precision, scale);
                let array = match value {
                    Some(v) => Arc::new(
                        arrow::array::Decimal128Array::from(vec![v])
                            .with_precision_and_scale(precision, scale)
                            .map_err(|e| {
                                Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                            })?,
                    ) as ArrayRef,
                    None => Arc::new(
                        arrow::array::Decimal128Array::from(vec![Option::<i128>::None])
                            .with_precision_and_scale(precision, scale)
                            .map_err(|e| {
                                Error::InvalidArgumentError(format!("Invalid decimal: {}", e))
                            })?,
                    ) as ArrayRef,
                };
                Ok((Field::new("max", data_type, true), array))
            }
            AggregateAccumulator::CountNulls {
                non_null_rows,
                total_rows_seen,
                ..
            } => {
                let nulls = total_rows_seen.checked_sub(non_null_rows).ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "NULL-count aggregate observed more non-null rows than total rows".into(),
                    )
                })?;
                let mut builder = Int64Builder::with_capacity(1);
                builder.append_value(nulls);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("count_nulls", DataType::Int64, false), array))
            }
            AggregateAccumulator::GroupConcat {
                values, separator, ..
            } => {
                use arrow::array::StringBuilder;
                let mut builder = StringBuilder::with_capacity(1, 256);
                if values.is_empty() {
                    builder.append_null();
                } else {
                    let result = values.join(&separator);
                    builder.append_value(&result);
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("group_concat", DataType::Utf8, true), array))
            }
            AggregateAccumulator::GroupConcatDistinct {
                values, separator, ..
            } => {
                use arrow::array::StringBuilder;
                let mut builder = StringBuilder::with_capacity(1, 256);
                if values.is_empty() {
                    builder.append_null();
                } else {
                    let result = values.join(&separator);
                    builder.append_value(&result);
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("group_concat", DataType::Utf8, true), array))
            }
        }
    }
}

impl AggregateState {
    /// Creates a new aggregate state with the given components.
    ///
    /// # Arguments
    ///
    /// - `alias`: Output column name for the aggregate result
    /// - `accumulator`: The aggregator instance performing the computation
    /// - `override_value`: Optional fixed value to replace the computed result
    pub fn new(
        alias: String,
        accumulator: AggregateAccumulator,
        override_value: Option<i64>,
    ) -> Self {
        Self {
            alias,
            accumulator,
            override_value,
        }
    }

    /// Updates the state with values from a new batch.
    ///
    /// # Arguments
    ///
    /// - `batch`: RecordBatch containing the column data to aggregate
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying accumulator update fails.
    pub fn update(&mut self, batch: &RecordBatch) -> AggregateResult<()> {
        self.accumulator.update(batch)
    }

    /// Returns the output field definition for this state.
    pub fn output_field(&self) -> Field {
        self.accumulator.output_field().with_name(&self.alias)
    }

    /// Finalizes the state and produces the resulting field and array.
    ///
    /// Applies the alias and override value if present.
    ///
    /// # Returns
    ///
    /// A tuple containing the output field schema and the computed aggregate value array.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying accumulator finalization fails.
    pub fn finalize(self) -> AggregateResult<(Field, ArrayRef)> {
        let (mut field, array) = self.accumulator.finalize()?;
        field = field.with_name(self.alias);
        if let Some(value) = self.override_value {
            let mut builder = Int64Builder::with_capacity(1);
            builder.append_value(value);
            let array = Arc::new(builder.finish()) as ArrayRef;
            return Ok((field, array));
        }
        Ok((field, array))
    }
}
