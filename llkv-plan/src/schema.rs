use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use llkv_types::FieldId;
use crate::plans::PlanValue;
use rustc_hash::FxHashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct PlanColumn {
    pub name: String,
    pub data_type: DataType,
    pub field_id: FieldId,
    pub is_nullable: bool,
    pub is_primary_key: bool,
    pub is_unique: bool,
    pub default_value: Option<PlanValue>,
    pub check_expr: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PlanSchema {
    pub columns: Vec<PlanColumn>,
    pub name_to_index: FxHashMap<String, usize>,
}

impl PlanSchema {
    pub fn new(columns: Vec<PlanColumn>) -> Self {
        let mut name_to_index = FxHashMap::default();
        for (i, col) in columns.iter().enumerate() {
            name_to_index.insert(col.name.to_ascii_lowercase(), i);
        }
        Self {
            columns,
            name_to_index,
        }
    }

    pub fn column_by_name(&self, name: &str) -> Option<&PlanColumn> {
        self.name_to_index.get(&name.to_ascii_lowercase()).map(|&i| &self.columns[i])
    }

    pub fn column_by_field_id(&self, field_id: FieldId) -> Option<&PlanColumn> {
        self.columns.iter().find(|c| c.field_id == field_id)
    }

    pub fn first_field_id(&self) -> Option<FieldId> {
        self.columns.first().map(|c| c.field_id)
    }

    pub fn to_arrow_schema(&self) -> SchemaRef {
        let fields: Vec<Field> = self.columns.iter().map(|c| {
            Field::new(&c.name, c.data_type.clone(), c.is_nullable)
        }).collect();
        Arc::new(Schema::new(fields))
    }
}
