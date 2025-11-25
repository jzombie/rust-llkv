//! Interval value stored as a combination of calendar months, whole days, and nanoseconds.

/// Interval value stored as a combination of calendar months, whole days, and nanoseconds.
///
/// Months capture both month and year components (12 months == 1 year). Days represent
/// whole 24-hour periods and nanoseconds account for sub-day precision. This mirrors the
/// semantics of Arrow's `IntervalMonthDayNano` while keeping arithmetic manageable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IntervalValue {
    pub months: i32,
    pub days: i32,
    pub nanos: i64,
}

impl IntervalValue {
    pub const fn new(months: i32, days: i32, nanos: i64) -> Self {
        Self {
            months,
            days,
            nanos,
        }
    }

    pub const fn zero() -> Self {
        Self::new(0, 0, 0)
    }

    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            months: self.months.checked_add(other.months)?,
            days: self.days.checked_add(other.days)?,
            nanos: self.nanos.checked_add(other.nanos)?,
        })
    }

    pub fn checked_sub(self, other: Self) -> Option<Self> {
        Some(Self {
            months: self.months.checked_sub(other.months)?,
            days: self.days.checked_sub(other.days)?,
            nanos: self.nanos.checked_sub(other.nanos)?,
        })
    }

    pub fn checked_neg(self) -> Option<Self> {
        Some(Self {
            months: self.months.checked_neg()?,
            days: self.days.checked_neg()?,
            nanos: self.nanos.checked_neg()?,
        })
    }

    pub fn checked_scale(self, factor: i64) -> Option<Self> {
        let months = i64::from(self.months).checked_mul(factor)?;
        let days = i64::from(self.days).checked_mul(factor)?;
        let nanos = self.nanos.checked_mul(factor)?;
        Some(Self {
            months: months.try_into().ok()?,
            days: days.try_into().ok()?,
            nanos,
        })
    }

    pub const fn is_zero(self) -> bool {
        self.months == 0 && self.days == 0 && self.nanos == 0
    }
}
