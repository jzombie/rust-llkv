use std::sync::Arc;
use llkv_plan::SelectPlan;
use llkv_plan::physical::PhysicalPlan;
use crate::physical_plan::ScanExec;
use crate::types::ExecutorTable;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use rustc_hash::FxHashMap;
use llkv_expr::{Expr, Filter, Operator};
use llkv_types::literal::Literal;
use arrow::datatypes::DataType;
use llkv_column_map::store::scan::ranges::{IntRanges, RangeKey};
use std::ops::Bound;
use llkv_types::LogicalFieldId;
use crate::translation::translate_predicate;
use llkv_result::Error;

pub struct PhysicalPlanner<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    tables: FxHashMap<String, Arc<ExecutorTable<P>>>,
}

impl<P> PhysicalPlanner<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(tables: FxHashMap<String, Arc<ExecutorTable<P>>>) -> Self {
        Self { tables }
    }

    pub fn create_physical_plan(&self, plan: &SelectPlan) -> Result<Arc<dyn PhysicalPlan>, String> {
        if let Some(table_ref) = plan.tables.first() {
             let table_name = &table_ref.table;
             let table = self.tables.get(table_name).ok_or_else(|| format!("Table not found: {}", table_name))?;
             
             // Use the table's schema
             let fields: Vec<arrow::datatypes::Field> = table.schema.columns.iter().map(|col| {
                 arrow::datatypes::Field::new(&col.name, col.data_type.clone(), col.nullable)
             }).collect();
             let schema = Arc::new(arrow::datatypes::Schema::new(fields));
             
             // Try to extract ranges from filter
             let mut ranges = None;
             let mut driving_column = None;
             let mut physical_filter = None;

             if let Some(filter) = &plan.filter {
                 // Translate the filter to physical expression
                 // We need to clone the predicate because translate_predicate takes ownership
                 let predicate = filter.predicate.clone();
                 match translate_predicate(predicate, &table.schema, |_| Error::Internal("Unknown column".to_string())) {
                     Ok(expr) => physical_filter = Some(expr),
                     Err(e) => return Err(format!("Failed to translate filter: {}", e)),
                 }

                 // Iterate over columns to find a driving column
                 for col in &table.schema.columns {
                     if let Some(r) = extract_ranges(&filter.predicate, &col.name, &col.data_type) {
                         ranges = Some(r);
                         let logical_field_id = LogicalFieldId::for_user(table.table_id(), col.field_id);
                         driving_column = Some(logical_field_id);
                          // TODO: Support multiple driving columns? It's currently first-match wins.
                         break;
                     }
                 }
             }

             Ok(Arc::new(ScanExec::new(table_name.clone(), schema, table.clone(), ranges, driving_column, physical_filter)))
        } else {
            Err("No tables in select plan".to_string())
        }
    }
}

fn extract_ranges(expr: &Expr<'static, String>, target_col_name: &str, target_type: &DataType) -> Option<IntRanges> {
    match expr {
        Expr::Pred(Filter { field_id, op }) if field_id == target_col_name => {
            let mut ranges = IntRanges::default();
            match target_type {
                DataType::Int64 => {
                    let (lb, ub) = op_to_bounds_i64(op)?;
                    i64::store(&mut ranges, lb, ub);
                    Some(ranges)
                }
                DataType::UInt64 => {
                    let (lb, ub) = op_to_bounds_u64(op)?;
                    u64::store(&mut ranges, lb, ub);
                    Some(ranges)
                }
                // TODO: Add support for other types
                _ => None,
            }
        }
        Expr::And(exprs) => {
             let mut combined_ranges: Option<IntRanges> = None;
             for e in exprs {
                 let r = extract_ranges(e, target_col_name, target_type);
                 match (combined_ranges, r) {
                     (Some(cr), Some(_nr)) => {
                         // TODO: Implement intersection
                         combined_ranges = Some(cr);
                     }
                     (None, Some(nr)) => combined_ranges = Some(nr),
                     (Some(cr), None) => combined_ranges = Some(cr),
                     (None, None) => {}
                 }
             }
             combined_ranges
        }
        _ => None,
    }
}

fn op_to_bounds_i64(op: &Operator) -> Option<(Bound<i64>, Bound<i64>)> {
    match op {
        Operator::Equals(lit) => {
            let val = literal_to_i64(lit)?;
            Some((Bound::Included(val), Bound::Included(val)))
        }
        Operator::Range { lower, upper } => {
            let l = bound_literal_to_i64(lower)?;
            let u = bound_literal_to_i64(upper)?;
            Some((l, u))
        }
        Operator::GreaterThan(lit) => {
            let val = literal_to_i64(lit)?;
            Some((Bound::Excluded(val), Bound::Unbounded))
        }
        Operator::GreaterThanOrEquals(lit) => {
            let val = literal_to_i64(lit)?;
            Some((Bound::Included(val), Bound::Unbounded))
        }
        Operator::LessThan(lit) => {
            let val = literal_to_i64(lit)?;
            Some((Bound::Unbounded, Bound::Excluded(val)))
        }
        Operator::LessThanOrEquals(lit) => {
            let val = literal_to_i64(lit)?;
            Some((Bound::Unbounded, Bound::Included(val)))
        }
        _ => None,
    }
}

fn op_to_bounds_u64(op: &Operator) -> Option<(Bound<u64>, Bound<u64>)> {
    match op {
        Operator::Equals(lit) => {
            let val = literal_to_u64(lit)?;
            Some((Bound::Included(val), Bound::Included(val)))
        }
        Operator::Range { lower, upper } => {
            let l = bound_literal_to_u64(lower)?;
            let u = bound_literal_to_u64(upper)?;
            Some((l, u))
        }
        Operator::GreaterThan(lit) => {
            let val = literal_to_u64(lit)?;
            Some((Bound::Excluded(val), Bound::Unbounded))
        }
        Operator::GreaterThanOrEquals(lit) => {
            let val = literal_to_u64(lit)?;
            Some((Bound::Included(val), Bound::Unbounded))
        }
        Operator::LessThan(lit) => {
            let val = literal_to_u64(lit)?;
            Some((Bound::Unbounded, Bound::Excluded(val)))
        }
        Operator::LessThanOrEquals(lit) => {
            let val = literal_to_u64(lit)?;
            Some((Bound::Unbounded, Bound::Included(val)))
        }
        _ => None,
    }
}

fn literal_to_i64(lit: &Literal) -> Option<i64> {
    match lit {
        Literal::Int128(v) => (*v).try_into().ok(),
        Literal::Float64(v) => Some(*v as i64), // Warning: lossy
        _ => None,
    }
}

fn literal_to_u64(lit: &Literal) -> Option<u64> {
    match lit {
        Literal::Int128(v) => (*v).try_into().ok(),
        Literal::Float64(v) => Some(*v as u64), // Warning: lossy
        _ => None,
    }
}

fn bound_literal_to_i64(bound: &Bound<Literal>) -> Option<Bound<i64>> {
    match bound {
        Bound::Included(lit) => literal_to_i64(lit).map(Bound::Included),
        Bound::Excluded(lit) => literal_to_i64(lit).map(Bound::Excluded),
        Bound::Unbounded => Some(Bound::Unbounded),
    }
}

fn bound_literal_to_u64(bound: &Bound<Literal>) -> Option<Bound<u64>> {
    match bound {
        Bound::Included(lit) => literal_to_u64(lit).map(Bound::Included),
        Bound::Excluded(lit) => literal_to_u64(lit).map(Bound::Excluded),
        Bound::Unbounded => Some(Bound::Unbounded),
    }
}
