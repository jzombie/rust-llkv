//! Date literal and arithmetic utilities shared across planner consumers.
//!
//! These helpers provide consistent conversions between SQL literal syntax,
//! Arrow `Date32` values, and calendar arithmetic involving `INTERVAL`
//! literals. Having a single implementation avoids drift between crates and
//! keeps overflow/validation handling centralized.

use llkv_types::IntervalValue;
use llkv_result::{Error, Result};
use time::{Date, Duration, Month};

const NANOS_PER_DAY: i64 = 86_400_000_000_000;

/// Parse a SQL `DATE 'YYYY-MM-DD'` literal into the Arrow `Date32` encoding
/// (days since the Unix epoch).
pub fn parse_date32_literal(text: &str) -> Result<i32> {
    let mut parts = text.split('-');
    let year_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    let month_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    let day_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError(format!(
            "invalid DATE literal '{text}'"
        )));
    }

    let year = year_str.parse::<i32>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid year in DATE literal '{text}'"))
    })?;
    let month_num = month_str.parse::<u8>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid month in DATE literal '{text}'"))
    })?;
    let day = day_str.parse::<u8>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid day in DATE literal '{text}'"))
    })?;

    let month = month_from_number(month_num)?;
    let date = Date::from_calendar_date(year, month, day).map_err(|err| {
        Error::InvalidArgumentError(format!("invalid DATE literal '{text}': {err}"))
    })?;
    Ok(date_to_days(date))
}

/// Add an interval to a `Date32` value, returning the adjusted `Date32` days.
pub fn add_interval_to_date32(days: i32, interval: IntervalValue) -> Result<i32> {
    let date = days_to_date(days)?;
    let result = apply_interval(date, interval)?;
    Ok(date_to_days(result))
}

/// Subtract an interval from a `Date32` value, returning the adjusted days.
pub fn subtract_interval_from_date32(days: i32, interval: IntervalValue) -> Result<i32> {
    let negated = interval.checked_neg().ok_or_else(|| {
        Error::InvalidArgumentError("interval overflow while negating for DATE arithmetic".into())
    })?;
    add_interval_to_date32(days, negated)
}

fn apply_interval(date: Date, interval: IntervalValue) -> Result<Date> {
    let mut result = date;

    if interval.months != 0 {
        result = add_months(result, interval.months)?;
    }

    let extra_days = total_days_from_components(interval.days, interval.nanos)?;
    if extra_days != 0 {
        let duration = Duration::days(extra_days);
        result = result.checked_add(duration).ok_or_else(|| {
            Error::InvalidArgumentError("date overflow while applying day component".into())
        })?;
    }

    Ok(result)
}

fn add_months(date: Date, months_delta: i32) -> Result<Date> {
    if months_delta == 0 {
        return Ok(date);
    }

    let current_year = i64::from(date.year());
    let current_month = i64::from(date.month() as u8);
    let base_index = current_year
        .checked_mul(12)
        .and_then(|value| value.checked_add(current_month - 1))
        .ok_or_else(|| {
            Error::InvalidArgumentError("date overflow while computing month offset".into())
        })?;

    let target_index = base_index
        .checked_add(i64::from(months_delta))
        .ok_or_else(|| {
            Error::InvalidArgumentError("date overflow while applying month component".into())
        })?;

    let new_year = target_index.div_euclid(12);
    if new_year < i64::from(i32::MIN) || new_year > i64::from(i32::MAX) {
        return Err(Error::InvalidArgumentError(
            "resulting date year out of range after month arithmetic".into(),
        ));
    }
    let new_year_i32 = new_year as i32;

    let month_index = target_index.rem_euclid(12) as u8; // 0-based month
    let new_month = month_from_number(month_index + 1)?;

    let current_day = date.day();
    let max_day = new_month.length(new_year_i32);
    let new_day = current_day.min(max_day);

    Date::from_calendar_date(new_year_i32, new_month, new_day).map_err(|err| {
        Error::InvalidArgumentError(format!(
            "resulting date is invalid after month arithmetic: {err}"
        ))
    })
}

fn total_days_from_components(days: i32, nanos: i64) -> Result<i64> {
    if nanos % NANOS_PER_DAY != 0 {
        return Err(Error::InvalidArgumentError(
            "cannot apply sub-day interval components to a DATE value".into(),
        ));
    }

    let nanos_days = nanos / NANOS_PER_DAY;
    i64::from(days).checked_add(nanos_days).ok_or_else(|| {
        Error::InvalidArgumentError("date overflow while combining day components".into())
    })
}

fn days_to_date(days: i32) -> Result<Date> {
    let julian = i64::from(days) + i64::from(epoch_julian_day());
    if julian < i64::from(i32::MIN) || julian > i64::from(i32::MAX) {
        return Err(Error::InvalidArgumentError(
            "DATE literal out of range for Julian conversion".into(),
        ));
    }
    Date::from_julian_day(julian as i32)
        .map_err(|err| Error::InvalidArgumentError(format!("DATE literal out of range: {err}")))
}

fn date_to_days(date: Date) -> i32 {
    date.to_julian_day() - epoch_julian_day()
}

fn epoch_julian_day() -> i32 {
    Date::from_calendar_date(1970, Month::January, 1)
        .expect("1970-01-01 is a valid date")
        .to_julian_day()
}

fn month_from_number(raw: u8) -> Result<Month> {
    if raw == 0 || raw > 12 {
        return Err(Error::InvalidArgumentError(format!(
            "invalid month '{raw}' for DATE literal"
        )));
    }
    Ok(Month::January.nth_next(raw - 1))
}

/// Format an Arrow `Date32` day count into `YYYY-MM-DD` text.
pub fn format_date32_literal(days: i32) -> Result<String> {
    let julian = epoch_julian_day()
        .checked_add(days)
        .ok_or_else(|| Error::InvalidArgumentError("date literal out of range".into()))?;

    let date = Date::from_julian_day(julian)
        .map_err(|err| Error::InvalidArgumentError(format!("invalid DATE value: {err}")))?;
    let (year, month, day) = date.to_calendar_date();
    let month_number = month as u8;
    Ok(format!("{:04}-{:02}-{:02}", year, month_number, day))
}
