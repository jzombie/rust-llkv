//! Interval conversion helpers shared across executor modules.

use std::cmp::Ordering;

use arrow_buffer::IntervalMonthDayNano;
use llkv_types::interval::IntervalValue;

#[inline]
pub fn interval_value_to_arrow(value: IntervalValue) -> IntervalMonthDayNano {
    IntervalMonthDayNano::new(value.months, value.days, value.nanos)
}

#[inline]
pub fn interval_value_from_arrow(value: IntervalMonthDayNano) -> IntervalValue {
    IntervalValue::new(value.months, value.days, value.nanoseconds)
}

#[inline]
pub fn compare_interval_values(lhs: IntervalValue, rhs: IntervalValue) -> Ordering {
    (lhs.months, lhs.days, lhs.nanos).cmp(&(rhs.months, rhs.days, rhs.nanos))
}
