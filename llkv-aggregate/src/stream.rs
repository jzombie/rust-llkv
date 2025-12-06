use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};
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
