use arrow::datatypes::DataType;
use llkv_column_map::store::RangeKey;
use llkv_column_map::store::scan::ranges::IntRanges;
use llkv_expr::expr::{Expr, Filter, Operator};
use std::ops::Bound;

pub fn extract_ranges(
    expr: &Expr<'static, String>,
    target_col_name: &str,
    target_type: &DataType,
) -> Option<IntRanges> {
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

fn map_bound<T, U, F>(bound: &Bound<T>, f: F) -> Option<Bound<U>>
where
    F: Fn(&T) -> Option<U>,
{
    match bound {
        Bound::Included(v) => f(v).map(Bound::Included),
        Bound::Excluded(v) => f(v).map(Bound::Excluded),
        Bound::Unbounded => Some(Bound::Unbounded),
    }
}

fn op_to_bounds_i64(op: &Operator<'_>) -> Option<(Bound<i64>, Bound<i64>)> {
    match op {
        Operator::Range { lower, upper } => {
            let l = map_bound(lower, |lit| lit.as_i64())?;
            let u = map_bound(upper, |lit| lit.as_i64())?;
            Some((l, u))
        }
        Operator::Equals(lit) => {
            let v = lit.as_i64()?;
            Some((Bound::Included(v), Bound::Included(v)))
        }
        // TODO: Handle other operators
        _ => None,
    }
}

fn op_to_bounds_u64(op: &Operator<'_>) -> Option<(Bound<u64>, Bound<u64>)> {
    match op {
        Operator::Range { lower, upper } => {
            let l = map_bound(lower, |lit| lit.as_u64())?;
            let u = map_bound(upper, |lit| lit.as_u64())?;
            Some((l, u))
        }
        Operator::Equals(lit) => {
            let v = lit.as_u64()?;
            Some((Bound::Included(v), Bound::Included(v)))
        }
        // TODO: Handle other operators
        _ => None,
    }
}
