use arrow::array::{Array, ArrayRef, Int64Array, Int64Builder, RecordBatch};
use arrow::datatypes::{DataType, Field};
use llkv_result::Error;
use llkv_table::types::FieldId;
use std::sync::Arc;

pub type AggregateResult<T> = Result<T, Error>;

/// Specification for an aggregate operation
#[derive(Clone)]
pub struct AggregateSpec {
    pub alias: String,
    pub kind: AggregateKind,
}

/// Type of aggregate operation
#[derive(Clone)]
pub enum AggregateKind {
    CountStar,
    CountField { field_id: FieldId },
    SumInt64 { field_id: FieldId },
    MinInt64 { field_id: FieldId },
    MaxInt64 { field_id: FieldId },
    CountNulls { field_id: FieldId },
}

impl AggregateKind {
    /// Get the field_id associated with this aggregate, if any
    pub fn field_id(&self) -> Option<FieldId> {
        match self {
            AggregateKind::CountStar => None,
            AggregateKind::CountField { field_id }
            | AggregateKind::SumInt64 { field_id }
            | AggregateKind::MinInt64 { field_id }
            | AggregateKind::MaxInt64 { field_id }
            | AggregateKind::CountNulls { field_id } => Some(*field_id),
        }
    }
}

/// Runtime state for an aggregate computation
pub struct AggregateState {
    pub alias: String,
    pub accumulator: AggregateAccumulator,
    pub override_value: Option<i64>,
}

/// Accumulator for incremental aggregate computation
pub enum AggregateAccumulator {
    CountStar {
        value: i64,
    },
    CountColumn {
        column_index: usize,
        value: i64,
    },
    SumInt64 {
        column_index: usize,
        value: i64,
        saw_value: bool,
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
        total_rows: i64,
    },
}

impl AggregateAccumulator {
    /// Create accumulator using projection index (position in the batch columns)
    /// instead of table schema position. projection_idx is None for CountStar.
    pub fn new_with_projection_index(
        spec: &AggregateSpec,
        projection_idx: Option<usize>,
        total_rows_hint: Option<i64>,
    ) -> AggregateResult<Self> {
        match spec.kind {
            AggregateKind::CountStar => Ok(AggregateAccumulator::CountStar { value: 0 }),
            AggregateKind::CountField { .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("CountField aggregate requires projection index".into())
                })?;
                Ok(AggregateAccumulator::CountColumn {
                    column_index: idx,
                    value: 0,
                })
            }
            AggregateKind::SumInt64 { .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("SumInt64 aggregate requires projection index".into())
                })?;
                Ok(AggregateAccumulator::SumInt64 {
                    column_index: idx,
                    value: 0,
                    saw_value: false,
                })
            }
            AggregateKind::MinInt64 { .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("MinInt64 aggregate requires projection index".into())
                })?;
                Ok(AggregateAccumulator::MinInt64 {
                    column_index: idx,
                    value: None,
                })
            }
            AggregateKind::MaxInt64 { .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("MaxInt64 aggregate requires projection index".into())
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
                let total_rows = total_rows_hint.ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "SUM(CASE WHEN ... IS NULL ...) with WHERE clauses is not supported yet"
                            .into(),
                    )
                })?;
                Ok(AggregateAccumulator::CountNulls {
                    column_index: idx,
                    non_null_rows: 0,
                    total_rows,
                })
            }
        }
    }

    /// Update the accumulator with a new batch of data
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
                total_rows: _,
            } => {
                let array = batch.column(*column_index);
                let non_null = (0..array.len()).filter(|idx| array.is_valid(*idx)).count();
                let non_null = i64::try_from(non_null).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *non_null_rows = non_null_rows.checked_add(non_null).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
            }
        }
        Ok(())
    }

    /// Finalize the accumulator and produce the result
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
                total_rows,
                ..
            } => {
                let nulls = total_rows.checked_sub(non_null_rows).ok_or_else(|| {
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
    /// Create a new aggregate state
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

    /// Update the state with a new batch of data
    pub fn update(&mut self, batch: &RecordBatch) -> AggregateResult<()> {
        self.accumulator.update(batch)
    }

    /// Finalize the state and produce the result
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
