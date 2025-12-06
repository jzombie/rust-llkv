use arrow::array::{ArrayRef, RecordBatch, UInt32Array};
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::row::{OwnedRow, RowConverter, SortField};
use arrow::compute::{concat, take};
use llkv_plan::plans::SelectPlan;
use llkv_plan::schema::PlanSchema;
use llkv_result::Error;
use llkv_types::FieldId;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::{AggregateAccumulator, AggregateKind, AggregateSpec, AggregateState};

pub struct AggregateStream<I> {
    states: Vec<AggregateState>,
    input: I,
    done: bool,
    schema: SchemaRef,
}

impl<I> AggregateStream<I>
where
    I: Iterator<Item = Result<RecordBatch, Error>>,
{
    pub fn new(
        input: I,
        plan: &SelectPlan,
        logical_schema: &PlanSchema,
        physical_schema: &SchemaRef,
    ) -> Result<Self, Error> {
        let states = build_aggregate_states(plan, logical_schema, physical_schema)?;
        
        let fields: Vec<arrow::datatypes::Field> = states.iter().map(|s| s.output_field()).collect();
        let schema = Arc::new(Schema::new(fields));

        Ok(Self {
            states,
            input,
            done: false,
            schema,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl<I> Iterator for AggregateStream<I>
where
    I: Iterator<Item = Result<RecordBatch, Error>>,
{
    type Item = Result<RecordBatch, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        for batch in self.input.by_ref() {
            let batch = match batch {
                Ok(b) => b,
                Err(e) => return Some(Err(e)),
            };

            for state in self.states.iter_mut() {
                if let Err(e) = state.update(&batch) {
                    return Some(Err(e));
                }
            }
        }

        self.done = true;

        let mut fields = Vec::with_capacity(self.states.len());
        let mut arrays = Vec::with_capacity(self.states.len());
        for state in self.states.drain(..) {
            let (field, array) = match state.finalize() {
                Ok(res) => res,
                Err(e) => return Some(Err(e)),
            };
            fields.push(field);
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        match RecordBatch::try_new(schema, arrays) {
            Ok(batch) => {
                Some(Ok(batch))
            },
            Err(e) => Some(Err(Error::Arrow(e))),
        }
    }
}

fn concat_arrays(chunks: &[ArrayRef]) -> Result<ArrayRef, Error> {
    if chunks.is_empty() {
        return Err(Error::Internal(
            "cannot concatenate empty array list".into(),
        ));
    }

    if chunks.len() == 1 {
        return Ok(chunks[0].clone());
    }

    let slices: Vec<&dyn arrow::array::Array> =
        chunks.iter().map(|a| a.as_ref()).collect();

    concat(&slices).map_err(|e| Error::Internal(e.to_string()))
}

pub struct GroupedAggregateStream<I> {
    input: I,
    key_indices: Vec<usize>,
    template_states: Vec<AggregateState>,
    schema: SchemaRef,
    converter: RowConverter,
    finished: bool,
}

impl<I> GroupedAggregateStream<I>
where
    I: Iterator<Item = Result<RecordBatch, Error>>,
{
    pub fn new(
        input: I,
        key_indices: Vec<usize>,
        plan: &SelectPlan,
        logical_schema: &PlanSchema,
        physical_schema: &SchemaRef,
    ) -> Result<Self, Error> {
        let template_states = build_aggregate_states(plan, logical_schema, physical_schema)?;

        let key_fields: Vec<Field> = key_indices
            .iter()
            .map(|&idx| physical_schema.field(idx).clone())
            .collect();

        let sort_fields: Vec<SortField> = key_fields
            .iter()
            .map(|f| SortField::new(f.data_type().clone()))
            .collect();
        let converter = RowConverter::new(sort_fields).map_err(|e| Error::Internal(e.to_string()))?;

        let mut fields: Vec<Field> = key_fields.clone();
        let agg_fields: Vec<Field> = template_states.iter().map(|s| s.output_field()).collect();
        fields.extend(agg_fields);
        let schema = Arc::new(Schema::new(fields));

        Ok(Self {
            input,
            key_indices,
            template_states,
            schema,
            converter,
            finished: false,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl<I> Iterator for GroupedAggregateStream<I>
where
    I: Iterator<Item = Result<RecordBatch, Error>>,
{
    type Item = Result<RecordBatch, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let mut key_chunks: Vec<Vec<ArrayRef>> = vec![Vec::new(); self.key_indices.len()];
        let mut group_states: Vec<Vec<AggregateState>> = Vec::new();
        let mut key_map: FxHashMap<OwnedRow, usize> = FxHashMap::default();

        for batch_res in self.input.by_ref() {
            let batch = match batch_res {
                Ok(b) => b,
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            };

            if batch.num_rows() == 0 {
                continue;
            }

            let key_columns: Vec<ArrayRef> = self
                .key_indices
                .iter()
                .map(|&idx| batch.column(idx).clone())
                .collect();

            let rows = match self.converter.convert_columns(&key_columns) {
                Ok(r) => r,
                Err(e) => {
                    self.finished = true;
                    return Some(Err(Error::Internal(e.to_string())));
                }
            };

            let mut rows_for_group: FxHashMap<usize, Vec<u32>> = FxHashMap::default();

            for row_idx in 0..batch.num_rows() {
                let owned = rows.row(row_idx).owned();
                let group_idx = if let Some(idx) = key_map.get(&owned) {
                    *idx
                } else {
                    let new_idx = group_states.len();
                    key_map.insert(owned, new_idx);
                    group_states.push(self.template_states.clone());

                    for (col_pos, &key_col_idx) in self.key_indices.iter().enumerate() {
                        let single_index = UInt32Array::from(vec![row_idx as u32]);
                        let value = match take(batch.column(key_col_idx), &single_index, None) {
                            Ok(v) => v,
                            Err(e) => {
                                self.finished = true;
                                return Some(Err(Error::Internal(e.to_string())));
                            }
                        };
                        key_chunks[col_pos].push(value);
                    }

                    new_idx
                };

                rows_for_group
                    .entry(group_idx)
                    .or_default()
                    .push(row_idx as u32);
            }

            for (gidx, indices) in rows_for_group {
                let idx_array = UInt32Array::from(indices);
                let mut columns = Vec::with_capacity(batch.num_columns());

                for col in batch.columns() {
                    let taken = match take(col, &idx_array, None) {
                        Ok(v) => v,
                        Err(e) => {
                            self.finished = true;
                            return Some(Err(Error::Internal(e.to_string())));
                        }
                    };
                    columns.push(taken);
                }

                let grouped_batch = match RecordBatch::try_new(batch.schema(), columns) {
                    Ok(b) => b,
                    Err(e) => {
                        self.finished = true;
                        return Some(Err(Error::Internal(e.to_string())));
                    }
                };

                if let Some(states) = group_states.get_mut(gidx) {
                    for state in states.iter_mut() {
                        if let Err(e) = state.update(&grouped_batch) {
                            self.finished = true;
                            return Some(Err(e));
                        }
                    }
                }
            }
        }

        self.finished = true;

        if group_states.is_empty() {
            let empty_cols: Vec<ArrayRef> = self
                .schema
                .fields()
                .iter()
                .map(|f| arrow::array::new_empty_array(f.data_type()))
                .collect();

            let batch = RecordBatch::try_new(self.schema.clone(), empty_cols)
                .map_err(|e| Error::Internal(e.to_string()));
            return Some(batch);
        }

        let mut agg_arrays_by_col: Vec<Vec<ArrayRef>> = vec![Vec::new(); self.template_states.len()];
        let mut agg_fields: Vec<Field> = Vec::new();

        for states in group_states {
            for (idx, state) in states.into_iter().enumerate() {
                let (field, array) = match state.finalize() {
                    Ok(res) => res,
                    Err(e) => return Some(Err(e)),
                };

                if agg_fields.len() <= idx {
                    agg_fields.push(field);
                }

                agg_arrays_by_col[idx].push(array);
            }
        }

        let mut agg_columns = Vec::new();
        for arrays in agg_arrays_by_col {
            match concat_arrays(&arrays) {
                Ok(arr) => agg_columns.push(arr),
                Err(e) => return Some(Err(e)),
            }
        }

        let mut key_columns = Vec::new();
        for chunks in key_chunks {
            match concat_arrays(&chunks) {
                Ok(arr) => key_columns.push(arr),
                Err(e) => return Some(Err(e)),
            }
        }

        let mut columns = Vec::new();
        columns.extend(key_columns);
        columns.extend(agg_columns);

        let batch = RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| Error::Internal(e.to_string()));

        Some(batch)
    }
}

fn build_aggregate_states(
    plan: &SelectPlan,
    logical_schema: &PlanSchema,
    physical_schema: &SchemaRef,
) -> Result<Vec<AggregateState>, Error> {
    let mut field_index_by_id: FxHashMap<FieldId, usize> = FxHashMap::default();
    for (idx, field) in physical_schema.fields().iter().enumerate() {
        if let Some(fid_str) = field.metadata().get("field_id") {
            if let Ok(fid) = fid_str.parse::<FieldId>() {
                field_index_by_id.insert(fid, idx);
            }
        }
    }

    let mut specs = Vec::with_capacity(plan.aggregates.len());
    for agg in &plan.aggregates {
        match agg {
            llkv_plan::AggregateExpr::CountStar { alias, distinct } => {
                specs.push(AggregateSpec {
                    alias: alias.clone(),
                    kind: AggregateKind::Count {
                        field_id: None,
                        distinct: *distinct,
                    },
                });
            }
            llkv_plan::AggregateExpr::Column {
                column,
                alias,
                function,
                distinct,
            } => {
                let col = logical_schema.column_by_name(column).ok_or_else(|| {
                    Error::InvalidArgumentError(format!("unknown column '{}' in aggregate", column))
                })?;

                let kind = match function {
                    llkv_plan::AggregateFunction::Count => AggregateKind::Count {
                        field_id: Some(col.field_id),
                        distinct: *distinct,
                    },
                    llkv_plan::AggregateFunction::SumInt64 => AggregateKind::Sum {
                        field_id: col.field_id,
                        data_type: col.data_type.clone(),
                        distinct: *distinct,
                    },
                    llkv_plan::AggregateFunction::TotalInt64 => AggregateKind::Total {
                        field_id: col.field_id,
                        data_type: col.data_type.clone(),
                        distinct: *distinct,
                    },
                    llkv_plan::AggregateFunction::MinInt64 => AggregateKind::Min {
                        field_id: col.field_id,
                        data_type: col.data_type.clone(),
                    },
                    llkv_plan::AggregateFunction::MaxInt64 => AggregateKind::Max {
                        field_id: col.field_id,
                        data_type: col.data_type.clone(),
                    },
                    llkv_plan::AggregateFunction::CountNulls => AggregateKind::CountNulls {
                        field_id: col.field_id,
                    },
                    llkv_plan::AggregateFunction::GroupConcat => {
                        return Err(Error::InvalidArgumentError(
                            "GROUP_CONCAT aggregate is not supported yet".into(),
                        ));
                    }
                };

                specs.push(AggregateSpec {
                    alias: alias.clone(),
                    kind,
                });
            }
        }
    }

    let mut states = Vec::with_capacity(specs.len());
    for spec in specs {
        let projection_idx = spec
            .kind
            .field_id()
            .and_then(|fid| field_index_by_id.get(&fid).copied());

        if spec.kind.field_id().is_some() && projection_idx.is_none() {
            return Err(Error::InvalidArgumentError(
                "aggregate input column missing from scan output".into(),
            ));
        }

        let accumulator =
            AggregateAccumulator::new_with_projection_index(&spec, projection_idx, None)?;
        states.push(AggregateState::new(spec.alias.clone(), accumulator, None));
    }

    Ok(states)
}
