//! Canonical scalar helpers for planner consumers.
//!
//! Concrete canonical types live in `llkv-types`. This module wires them up to
//! planner-specific values (`PlanValue`).

use std::sync::Arc;

use llkv_result::{Error, Result as LlkvResult};
use llkv_types::CanonicalScalar;

use crate::plans::PlanValue;

/// Build a canonical scalar from a planner value.
pub fn canonical_scalar_from_plan_value(value: &PlanValue) -> LlkvResult<CanonicalScalar> {
    match value {
        PlanValue::Null => Ok(CanonicalScalar::Null),
        PlanValue::Integer(v) => Ok(CanonicalScalar::Int64(*v)),
        PlanValue::Float(v) => Ok(CanonicalScalar::from_f64(*v)),
        PlanValue::Decimal(v) => Ok(CanonicalScalar::Decimal(*v)),
        PlanValue::String(v) => Ok(CanonicalScalar::Utf8(Arc::<str>::from(v.as_str()))),
        PlanValue::Date32(days) => Ok(CanonicalScalar::Date32(*days)),
        PlanValue::Interval(interval) => Ok(CanonicalScalar::Interval(*interval)),
        PlanValue::Struct(_) => Err(Error::InvalidArgumentError(
            "struct values are not supported in canonical scalar conversion".into(),
        )),
    }
}
