//! Runtime aggregation utilities used by the planner and executor.
//!
//! The crate evaluates logical aggregates described by [`llkv_plan::AggregateExpr`] against Arrow
//! batches. It supports streaming accumulation with overflow checks and COUNT DISTINCT tracking for
//! a subset of scalar types.
use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, Int64Builder,
    RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field};
use llkv_column_map::types::FieldId;
use llkv_result::Error;
use rustc_hash::FxHashSet;
use std::sync::Arc;

pub use llkv_plan::{AggregateExpr, AggregateFunction};

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
}

impl AggregateKind {
    /// Returns the field ID referenced by this aggregate, if any.
    pub fn field_id(&self) -> Option<FieldId> {
        match self {
            AggregateKind::Count { field_id, .. } => *field_id,
            AggregateKind::Sum { field_id, .. }
            | AggregateKind::Avg { field_id, .. }
            | AggregateKind::Min { field_id, .. }
            | AggregateKind::Max { field_id, .. }
            | AggregateKind::CountNulls { field_id } => Some(*field_id),
        }
    }
}

/// Runtime state for an aggregate computation.
pub struct AggregateState {
    pub alias: String,
    pub accumulator: AggregateAccumulator,
    pub override_value: Option<i64>,
}

/// Accumulator for incremental aggregate computation.
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
        value: i64,
        saw_value: bool,
    },
    SumDistinctInt64 {
        column_index: usize,
        sum: i64,
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
    MinInt64 {
        column_index: usize,
        value: Option<i64>,
    },
    MaxInt64 {
        column_index: usize,
        value: Option<i64>,
    },
    CountNulls {
        column_index: usize,
        non_null_rows: i64,
        total_rows_seen: i64,
    },
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub enum DistinctKey {
    Int(i64),
    Float(u64),
    Str(String),
    Bool(bool),
    Date(i32),
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
            other => Err(Error::InvalidArgumentError(format!(
                "COUNT(DISTINCT) is not supported for column type {other:?}"
            ))),
        }
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
            AggregateKind::Sum { distinct, .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("Sum aggregate requires projection index".into())
                })?;
                if *distinct {
                    Ok(AggregateAccumulator::SumDistinctInt64 {
                        column_index: idx,
                        sum: 0,
                        seen: FxHashSet::default(),
                    })
                } else {
                    Ok(AggregateAccumulator::SumInt64 {
                        column_index: idx,
                        value: 0,
                        saw_value: false,
                    })
                }
            }
            AggregateKind::Avg { distinct, .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("Avg aggregate requires projection index".into())
                })?;
                if *distinct {
                    Ok(AggregateAccumulator::AvgDistinctInt64 {
                        column_index: idx,
                        sum: 0,
                        seen: FxHashSet::default(),
                    })
                } else {
                    Ok(AggregateAccumulator::AvgInt64 {
                        column_index: idx,
                        sum: 0,
                        count: 0,
                    })
                }
            }
            AggregateKind::Min { .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("Min aggregate requires projection index".into())
                })?;
                Ok(AggregateAccumulator::MinInt64 {
                    column_index: idx,
                    value: None,
                })
            }
            AggregateKind::Max { .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("Max aggregate requires projection index".into())
                })?;
                Ok(AggregateAccumulator::MaxInt64 {
                    column_index: idx,
                    value: None,
                })
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
                let non_null = (0..array.len()).filter(|idx| array.is_valid(*idx)).count();
                let non_null = i64::try_from(non_null).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *value = value.checked_add(non_null).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
            }
            AggregateAccumulator::CountDistinctColumn { column_index, seen } => {
                let array = batch.column(*column_index);
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
                saw_value,
            } => {
                let array = batch.column(*column_index);
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "SUM aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let v = array.value(i);
                        *value = value.checked_add(v).ok_or_else(|| {
                            Error::InvalidArgumentError(
                                "SUM aggregate result exceeds i64 range".into(),
                            )
                        })?;
                        *saw_value = true;
                    }
                }
            }
            AggregateAccumulator::SumDistinctInt64 {
                column_index,
                sum,
                seen,
            } => {
                let array = batch.column(*column_index);
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
                                *sum = sum.checked_add(v).ok_or_else(|| {
                                    Error::InvalidArgumentError(
                                        "SUM(DISTINCT) aggregate result exceeds i64 range".into(),
                                    )
                                })?;
                            }
                        }
                    }
                }
            }
            AggregateAccumulator::AvgInt64 {
                column_index,
                sum,
                count,
            } => {
                let array = batch.column(*column_index);
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "AVG aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let v = array.value(i);
                        *sum = sum.checked_add(v).ok_or_else(|| {
                            Error::InvalidArgumentError(
                                "AVG aggregate sum exceeds i64 range".into(),
                            )
                        })?;
                        *count = count.checked_add(1).ok_or_else(|| {
                            Error::InvalidArgumentError(
                                "AVG aggregate count exceeds i64 range".into(),
                            )
                        })?;
                    }
                }
            }
            AggregateAccumulator::AvgDistinctInt64 {
                column_index,
                sum,
                seen,
            } => {
                let array = batch.column(*column_index);
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
            AggregateAccumulator::MinInt64 {
                column_index,
                value,
            } => {
                let array = batch.column(*column_index);
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "MIN aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let v = array.value(i);
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
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "MAX aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let v = array.value(i);
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
                let array = batch.column(*column_index);
                let batch_size = i64::try_from(array.len()).map_err(|_| {
                    Error::InvalidArgumentError("Batch size exceeds i64 range".into())
                })?;
                let non_null = (0..array.len()).filter(|idx| array.is_valid(*idx)).count();
                let non_null = i64::try_from(non_null).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *total_rows_seen = total_rows_seen.checked_add(batch_size).ok_or_else(|| {
                    Error::InvalidArgumentError("Total rows exceeds i64 range".into())
                })?;
                *non_null_rows = non_null_rows.checked_add(non_null).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
            }
        }
        Ok(())
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
                value, saw_value, ..
            } => {
                let mut builder = Int64Builder::with_capacity(1);
                if saw_value {
                    builder.append_value(value);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("sum", DataType::Int64, true), array))
            }
            AggregateAccumulator::SumDistinctInt64 { sum, seen, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                if !seen.is_empty() {
                    builder.append_value(sum);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("sum_distinct", DataType::Int64, true), array))
            }
            AggregateAccumulator::AvgInt64 { sum, count, .. } => {
                use arrow::array::Float64Builder;
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
                use arrow::array::Float64Builder;
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
