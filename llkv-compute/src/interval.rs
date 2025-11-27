//! Parse SQL `INTERVAL` literals into [`IntervalValue`] instances.
//!
//! Centralized here to keep interval parsing alongside other date helpers.

use std::convert::TryFrom;
use std::sync::OnceLock;

use llkv_result::{Error, Result};
use llkv_types::IntervalValue;
use regex::Regex;
use sqlparser::ast::{DateTimeField, Expr as SqlExpr, Interval as SqlInterval, UnaryOperator};

const NANOS_PER_SECOND: i64 = 1_000_000_000;
const NANOS_PER_MINUTE: i64 = 60 * NANOS_PER_SECOND;
const NANOS_PER_HOUR: i64 = 60 * NANOS_PER_MINUTE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntervalUnit {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
}

impl IntervalUnit {
    fn rank(self) -> u8 {
        match self {
            IntervalUnit::Year => 0,
            IntervalUnit::Month => 1,
            IntervalUnit::Day => 2,
            IntervalUnit::Hour => 3,
            IntervalUnit::Minute => 4,
            IntervalUnit::Second => 5,
        }
    }
}

pub fn parse_interval_literal(interval: &SqlInterval) -> Result<IntervalValue> {
    let value_text = interval_value_text(&interval.value)?;
    let leading = interval.leading_field.as_ref().map(map_field).transpose()?;
    let last = interval.last_field.as_ref().map(map_field).transpose()?;

    if let (Some(leading_unit), Some(last_unit)) = (leading, last)
        && leading_unit.rank() > last_unit.rank()
    {
        return Err(Error::InvalidArgumentError(format!(
            "invalid interval field order: {leading_unit:?} TO {last_unit:?}"
        )));
    }

    if leading.is_none() && interval.leading_precision.is_some() {
        return Err(Error::InvalidArgumentError(
            "interval leading precision requires a leading field".into(),
        ));
    }

    if let Some(unit) = leading
        && unit != IntervalUnit::Second
        && interval.fractional_seconds_precision.is_some()
    {
        return Err(Error::InvalidArgumentError(
            "fractional seconds precision only applies to SECOND intervals".into(),
        ));
    }

    match leading {
        Some(IntervalUnit::Year) => {
            parse_year_interval(&value_text, last, interval.leading_precision)
        }
        Some(IntervalUnit::Month) => parse_month_interval(&value_text, interval.leading_precision),
        Some(IntervalUnit::Day) => parse_day_interval(
            &value_text,
            last,
            interval.leading_precision,
            interval.fractional_seconds_precision,
        ),
        Some(IntervalUnit::Hour) => parse_hour_interval(
            &value_text,
            last,
            interval.leading_precision,
            interval.fractional_seconds_precision,
        ),
        Some(IntervalUnit::Minute) => parse_minute_interval(
            &value_text,
            last,
            interval.leading_precision,
            interval.fractional_seconds_precision,
        ),
        Some(IntervalUnit::Second) => parse_second_interval(
            &value_text,
            interval.leading_precision,
            interval.fractional_seconds_precision,
        ),
        None => parse_unqualified_interval(&value_text),
    }
}

fn parse_year_interval(
    value: &str,
    last: Option<IntervalUnit>,
    leading_precision: Option<u64>,
) -> Result<IntervalValue> {
    match last {
        None => {
            let years = parse_signed_integer(value, "year")?;
            enforce_precision(years.digits_len, leading_precision, "year")?;
            let months = years
                .value
                .checked_mul(12)
                .ok_or_else(|| Error::InvalidArgumentError("interval overflow".into()))?;
            IntervalValue::from_components(months, 0, 0)
        }
        Some(IntervalUnit::Month) => parse_year_to_month(value, leading_precision),
        Some(other) => Err(Error::InvalidArgumentError(format!(
            "unsupported interval field combination: YEAR TO {:?}",
            other
        ))),
    }
}

fn parse_year_to_month(value: &str, leading_precision: Option<u64>) -> Result<IntervalValue> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Error::InvalidArgumentError(
            "YEAR TO MONTH interval requires a value".into(),
        ));
    }

    let (sign, rest) = split_sign(trimmed);
    let rest = rest.trim();
    let (years_part, months_part) = match rest.find('-') {
        Some(idx) => {
            let months_slice = &rest[idx + 1..];
            (&rest[..idx], months_slice)
        }
        None => (rest, "0"),
    };

    let years_part_trim = years_part.trim();
    let years = parse_unsigned_digits(years_part_trim, "year")?;
    enforce_precision(years_part_trim.len(), leading_precision, "year")?;
    let months_part_trim = months_part.trim();
    let months = if months_part_trim.is_empty() {
        0
    } else {
        parse_unsigned_digits(months_part_trim, "month")?
    };
    if months >= 12 {
        return Err(Error::InvalidArgumentError(
            "month component in YEAR TO MONTH interval must be between 0 and 11".into(),
        ));
    }

    let years_i64 = i64::try_from(years)
        .map_err(|_| Error::InvalidArgumentError("year component out of range".into()))?;
    let months_i64 = i64::try_from(months)
        .map_err(|_| Error::InvalidArgumentError("month component out of range".into()))?;
    let total_months = years_i64
        .checked_mul(12)
        .and_then(|base| base.checked_add(months_i64))
        .ok_or_else(|| Error::InvalidArgumentError("interval overflow".into()))?;
    let total_months = total_months * i64::from(sign);
    IntervalValue::from_components(total_months, 0, 0)
}

fn parse_month_interval(value: &str, leading_precision: Option<u64>) -> Result<IntervalValue> {
    let months = parse_signed_integer(value, "month")?;
    enforce_precision(months.digits_len, leading_precision, "month")?;
    IntervalValue::from_components(months.value, 0, 0)
}

fn parse_day_interval(
    value: &str,
    last: Option<IntervalUnit>,
    leading_precision: Option<u64>,
    fractional_precision: Option<u64>,
) -> Result<IntervalValue> {
    let (days, remainder) = split_primary_component(value, "day")?;
    enforce_precision(days.digits_len, leading_precision, "day")?;

    let mut nanos: i64 = 0;
    match last {
        None => {
            if remainder.is_some() {
                return Err(Error::InvalidArgumentError(
                    "unexpected time component in DAY interval".into(),
                ));
            }
        }
        Some(IntervalUnit::Hour) => {
            let hours_str = remainder.ok_or_else(|| {
                Error::InvalidArgumentError("DAY TO HOUR interval requires hours".into())
            })?;
            let hours = parse_unsigned_digits(&hours_str, "hour")?;
            if hours >= 24 {
                return Err(Error::InvalidArgumentError(
                    "hour component in DAY TO HOUR interval must be between 0 and 23".into(),
                ));
            }
            let signed_hours = signed_component(days.sign, hours, "hour")?;
            nanos = checked_mul(signed_hours, NANOS_PER_HOUR, "hour")?;
        }
        Some(IntervalUnit::Minute) => {
            let time_str = remainder.ok_or_else(|| {
                Error::InvalidArgumentError("DAY TO MINUTE interval requires time".into())
            })?;
            let (hours, minutes) = parse_hh_mm(&time_str)?;
            if hours >= 24 || minutes >= 60 {
                return Err(Error::InvalidArgumentError(
                    "time component out of range for DAY TO MINUTE interval".into(),
                ));
            }
            let signed_hours = signed_component(days.sign, hours, "hour")?;
            let signed_minutes = signed_component(days.sign, minutes, "minute")?;
            nanos = compose_time_nanos(signed_hours, signed_minutes, 0, 0)?;
        }
        Some(IntervalUnit::Second) => {
            let time_str = remainder.ok_or_else(|| {
                Error::InvalidArgumentError("DAY TO SECOND interval requires time".into())
            })?;
            let (hours, minutes, seconds, fraction) =
                parse_hh_mm_ss(&time_str, fractional_precision)?;
            if hours >= 24 || minutes >= 60 || seconds >= 60 {
                return Err(Error::InvalidArgumentError(
                    "time component out of range for DAY TO SECOND interval".into(),
                ));
            }
            let signed_hours = signed_component(days.sign, hours, "hour")?;
            let signed_minutes = signed_component(days.sign, minutes, "minute")?;
            let signed_seconds = signed_component(days.sign, seconds, "second")?;
            let signed_fraction = apply_sign_i64(days.sign, fraction);
            nanos = compose_time_nanos(
                signed_hours,
                signed_minutes,
                signed_seconds,
                signed_fraction,
            )?;
        }
        Some(other) => {
            return Err(Error::InvalidArgumentError(format!(
                "unsupported interval field combination: DAY TO {:?}",
                other
            )));
        }
    }

    IntervalValue::from_components(0, days.value, nanos)
}

fn parse_hour_interval(
    value: &str,
    last: Option<IntervalUnit>,
    leading_precision: Option<u64>,
    fractional_precision: Option<u64>,
) -> Result<IntervalValue> {
    match last {
        None => {
            let hours = parse_signed_integer(value, "hour")?;
            enforce_precision(hours.digits_len, leading_precision, "hour")?;
            let nanos = checked_mul(hours.value, NANOS_PER_HOUR, "hour")?;
            IntervalValue::from_components(0, 0, nanos)
        }
        Some(IntervalUnit::Minute) => {
            let (hours, minutes) = parse_signed_hh_mm(value)?;
            enforce_precision(hours.digits_len, leading_precision, "hour")?;
            if minutes >= 60 {
                return Err(Error::InvalidArgumentError(
                    "minute component in HOUR TO MINUTE interval must be between 0 and 59".into(),
                ));
            }
            let signed_minutes = signed_component(hours.sign, minutes, "minute")?;
            let nanos = compose_time_nanos(hours.value, signed_minutes, 0, 0)?;
            IntervalValue::from_components(0, 0, nanos)
        }
        Some(IntervalUnit::Second) => {
            let (hours, minutes, seconds, fraction) =
                parse_signed_hh_mm_ss(value, fractional_precision)?;
            enforce_precision(hours.digits_len, leading_precision, "hour")?;
            if minutes >= 60 || seconds >= 60 {
                return Err(Error::InvalidArgumentError(
                    "time component out of range for HOUR TO SECOND interval".into(),
                ));
            }
            let signed_minutes = signed_component(hours.sign, minutes, "minute")?;
            let signed_seconds = signed_component(hours.sign, seconds, "second")?;
            let signed_fraction = apply_sign_i64(hours.sign, fraction);
            let nanos =
                compose_time_nanos(hours.value, signed_minutes, signed_seconds, signed_fraction)?;
            IntervalValue::from_components(0, 0, nanos)
        }
        Some(other) => Err(Error::InvalidArgumentError(format!(
            "unsupported interval field combination: HOUR TO {:?}",
            other
        ))),
    }
}

fn parse_minute_interval(
    value: &str,
    last: Option<IntervalUnit>,
    leading_precision: Option<u64>,
    fractional_precision: Option<u64>,
) -> Result<IntervalValue> {
    match last {
        None => {
            let minutes = parse_signed_integer(value, "minute")?;
            enforce_precision(minutes.digits_len, leading_precision, "minute")?;
            let nanos = checked_mul(minutes.value, NANOS_PER_MINUTE, "minute")?;
            IntervalValue::from_components(0, 0, nanos)
        }
        Some(IntervalUnit::Second) => {
            let (minutes, seconds, fraction) = parse_signed_mm_ss(value, fractional_precision)?;
            enforce_precision(minutes.digits_len, leading_precision, "minute")?;
            if seconds >= 60 {
                return Err(Error::InvalidArgumentError(
                    "second component in MINUTE TO SECOND interval must be between 0 and 59"
                        .into(),
                ));
            }
            let signed_seconds = signed_component(minutes.sign, seconds, "second")?;
            let signed_fraction = apply_sign_i64(minutes.sign, fraction);
            let nanos = compose_time_nanos(0, minutes.value, signed_seconds, signed_fraction)?;
            IntervalValue::from_components(0, 0, nanos)
        }
        Some(other) => Err(Error::InvalidArgumentError(format!(
            "unsupported interval field combination: MINUTE TO {:?}",
            other
        ))),
    }
}

fn parse_second_interval(
    value: &str,
    leading_precision: Option<u64>,
    fractional_precision: Option<u64>,
) -> Result<IntervalValue> {
    let seconds = parse_signed_decimal_seconds(value, fractional_precision)?;
    enforce_precision(seconds.int_digits, leading_precision, "second")?;
    let total_nanos = checked_mul(seconds.seconds, NANOS_PER_SECOND, "second")?
        .checked_add(seconds.nanos)
        .ok_or_else(|| Error::InvalidArgumentError("interval overflow".into()))?;
    IntervalValue::from_components(0, 0, total_nanos)
}

fn parse_unqualified_interval(value: &str) -> Result<IntervalValue> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Error::InvalidArgumentError(
            "interval literal cannot be empty".into(),
        ));
    }

    static TOKEN_RE: OnceLock<Regex> = OnceLock::new();
    let regex = TOKEN_RE.get_or_init(|| {
        Regex::new(r"(?ix)([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)").expect("valid interval token regex")
    });

    let mut months: i64 = 0;
    let mut days: i64 = 0;
    let mut nanos: i128 = 0;
    let mut index = 0;
    for caps in regex.captures_iter(trimmed) {
        let whole = caps.get(0).expect("capture 0");
        if trimmed[index..whole.start()].trim().is_empty() {
            index = whole.end();
        } else {
            return Err(Error::InvalidArgumentError(format!(
                "invalid token '{}' in interval literal",
                &trimmed[index..whole.start()]
            )));
        }

        let value_str = caps.get(1).unwrap().as_str();
        let unit_str = caps.get(2).unwrap().as_str().to_ascii_lowercase();

        match unit_str.as_str() {
            "year" | "years" => {
                ensure_no_fraction(value_str, "year")?;
                let value = value_str.parse::<i64>().map_err(|_| {
                    Error::InvalidArgumentError("year component out of range".into())
                })?;
                months =
                    months
                        .checked_add(value.checked_mul(12).ok_or_else(|| {
                            Error::InvalidArgumentError("interval overflow".into())
                        })?)
                        .ok_or_else(|| Error::InvalidArgumentError("interval overflow".into()))?;
            }
            "month" | "months" => {
                ensure_no_fraction(value_str, "month")?;
                let value = value_str.parse::<i64>().map_err(|_| {
                    Error::InvalidArgumentError("month component out of range".into())
                })?;
                months = months
                    .checked_add(value)
                    .ok_or_else(|| Error::InvalidArgumentError("interval overflow".into()))?;
            }
            "day" | "days" => {
                ensure_no_fraction(value_str, "day")?;
                let value = value_str.parse::<i64>().map_err(|_| {
                    Error::InvalidArgumentError("day component out of range".into())
                })?;
                days = days
                    .checked_add(value)
                    .ok_or_else(|| Error::InvalidArgumentError("interval overflow".into()))?;
            }
            "hour" | "hours" => {
                ensure_no_fraction(value_str, "hour")?;
                let value = value_str.parse::<i64>().map_err(|_| {
                    Error::InvalidArgumentError("hour component out of range".into())
                })?;
                nanos = nanos
                    .checked_add(
                        i128::from(value)
                            .checked_mul(i128::from(NANOS_PER_HOUR))
                            .ok_or_else(|| {
                                Error::InvalidArgumentError("interval overflow".into())
                            })?,
                    )
                    .ok_or_else(|| Error::InvalidArgumentError("interval overflow".into()))?;
            }
            "minute" | "minutes" => {
                ensure_no_fraction(value_str, "minute")?;
                let value = value_str.parse::<i64>().map_err(|_| {
                    Error::InvalidArgumentError("minute component out of range".into())
                })?;
                nanos = nanos
                    .checked_add(
                        i128::from(value)
                            .checked_mul(i128::from(NANOS_PER_MINUTE))
                            .ok_or_else(|| {
                                Error::InvalidArgumentError("interval overflow".into())
                            })?,
                    )
                    .ok_or_else(|| Error::InvalidArgumentError("interval overflow".into()))?;
            }
            "second" | "seconds" => {
                let decimal = parse_signed_decimal_seconds(value_str, None)?;
                nanos = nanos
                    .checked_add(
                        i128::from(decimal.seconds)
                            .checked_mul(i128::from(NANOS_PER_SECOND))
                            .ok_or_else(|| {
                                Error::InvalidArgumentError("interval overflow".into())
                            })?,
                    )
                    .ok_or_else(|| Error::InvalidArgumentError("interval overflow".into()))?;
                nanos = nanos
                    .checked_add(i128::from(decimal.nanos))
                    .ok_or_else(|| Error::InvalidArgumentError("interval overflow".into()))?;
            }
            other => {
                return Err(Error::InvalidArgumentError(format!(
                    "unsupported interval unit '{other}'"
                )));
            }
        }
    }

    if trimmed[index..].trim().is_empty() {
        let nanos = nanos
            .try_into()
            .map_err(|_| Error::InvalidArgumentError("interval overflow".into()))?;
        IntervalValue::from_components(months, days, nanos)
    } else {
        Err(Error::InvalidArgumentError(format!(
            "invalid trailing content '{}' in interval literal",
            &trimmed[index..]
        )))
    }
}

fn parse_hh_mm(value: &str) -> Result<(u64, u64)> {
    let mut parts = value.split(':');
    let hour = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError("hour component missing".into()))?;
    let minute = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError("minute component missing".into()))?;
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError("expected HH:MM format".into()));
    }
    let hours = parse_unsigned_digits(hour, "hour")?;
    let minutes = parse_unsigned_digits(minute, "minute")?;
    Ok((hours, minutes))
}

fn parse_hh_mm_ss(
    value: &str,
    fractional_precision: Option<u64>,
) -> Result<(u64, u64, u64, i64)> {
    let mut parts = value.split(':');
    let hour = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError("hour component missing".into()))?;
    let minute = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError("minute component missing".into()))?;
    let second = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError("second component missing".into()))?;
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError(
            "expected HH:MM:SS format".into(),
        ));
    }
    let hours = parse_unsigned_digits(hour, "hour")?;
    let minutes = parse_unsigned_digits(minute, "minute")?;
    let (seconds, fraction) = parse_unsigned_seconds(second, fractional_precision)?;
    Ok((hours, minutes, seconds, fraction))
}

fn parse_signed_hh_mm(value: &str) -> Result<(SignedInteger, u64)> {
    let mut parts = value.split(':');
    let hour = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError("hour component missing".into()))?;
    let minute = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError("minute component missing".into()))?;
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError("expected HH:MM format".into()));
    }
    let hours = parse_signed_integer(hour, "hour")?;
    let minutes = parse_unsigned_digits(minute, "minute")?;
    Ok((hours, minutes))
}

fn parse_signed_hh_mm_ss(
    value: &str,
    fractional_precision: Option<u64>,
) -> Result<(SignedInteger, u64, u64, i64)> {
    let mut parts = value.split(':');
    let hour = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError("hour component missing".into()))?;
    let minute = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError("minute component missing".into()))?;
    let second = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError("second component missing".into()))?;
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError(
            "expected HH:MM:SS format".into(),
        ));
    }
    let hours = parse_signed_integer(hour, "hour")?;
    let minutes = parse_unsigned_digits(minute, "minute")?;
    let (seconds, fraction) = parse_unsigned_seconds(second, fractional_precision)?;
    Ok((hours, minutes, seconds, fraction))
}

fn parse_signed_mm_ss(
    value: &str,
    fractional_precision: Option<u64>,
) -> Result<(SignedInteger, u64, i64)> {
    let mut parts = value.split(':');
    let minute = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError("minute component missing".into()))?;
    let second = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError("second component missing".into()))?;
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError("expected MM:SS format".into()));
    }
    let minutes = parse_signed_integer(minute, "minute")?;
    let (seconds, fraction) = parse_unsigned_seconds(second, fractional_precision)?;
    Ok((minutes, seconds, fraction))
}

fn parse_unsigned_seconds(value: &str, fractional_precision: Option<u64>) -> Result<(u64, i64)> {
    let trimmed = value.trim();
    if trimmed.starts_with(['+', '-']) {
        return Err(Error::InvalidArgumentError(
            "time component must not include a sign".into(),
        ));
    }
    let mut parts = trimmed.split('.');
    let whole = parts.next().unwrap_or("0");
    let fraction = parts.next();
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError(format!(
            "invalid second component '{value}'"
        )));
    }
    let seconds = parse_unsigned_digits(whole, "second")?;
    let mut nanos = 0i64;
    if let Some(frac) = fraction {
        if let Some(precision) = fractional_precision
            && frac.len() as u64 > precision
        {
            return Err(Error::InvalidArgumentError(format!(
                "fractional second precision exceeds declared precision ({precision})"
            )));
        }
        if frac.len() > 9 {
            return Err(Error::InvalidArgumentError(
                "fractional second precision beyond nanoseconds not supported".into(),
            ));
        }
        if !frac.chars().all(|c| c.is_ascii_digit()) {
            return Err(Error::InvalidArgumentError(format!(
                "invalid fractional seconds '{value}'"
            )));
        }
        let mut fraction_value = frac
            .parse::<i64>()
            .map_err(|_| Error::InvalidArgumentError("fractional second out of range".into()))?;
        for _ in 0..(9 - frac.len()) {
            fraction_value *= 10;
        }
        nanos = fraction_value;
    }
    Ok((seconds, nanos))
}

fn parse_signed_decimal_seconds(
    value: &str,
    fractional_precision: Option<u64>,
) -> Result<SignedDecimal> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Error::InvalidArgumentError(
            "second component cannot be empty".into(),
        ));
    }
    let (sign, digits) = split_sign(trimmed);
    let mut parts = digits.split('.');
    let whole = parts.next().unwrap_or("0");
    let fraction = parts.next();
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError(format!(
            "invalid second component '{value}'"
        )));
    }
    if !whole.chars().all(|c| c.is_ascii_digit()) {
        return Err(Error::InvalidArgumentError(format!(
            "invalid second component '{value}'"
        )));
    }
    let seconds = whole
        .parse::<i64>()
        .map_err(|_| Error::InvalidArgumentError("second component out of range".into()))?;
    let mut nanos = 0i64;
    if let Some(frac) = fraction {
        if let Some(precision) = fractional_precision
            && frac.len() as u64 > precision
        {
            return Err(Error::InvalidArgumentError(format!(
                "fractional second precision exceeds declared precision ({precision})"
            )));
        }
        if frac.len() > 9 {
            return Err(Error::InvalidArgumentError(
                "fractional second precision beyond nanoseconds not supported".into(),
            ));
        }
        if !frac.chars().all(|c| c.is_ascii_digit()) {
            return Err(Error::InvalidArgumentError(format!(
                "invalid fractional seconds '{value}'"
            )));
        }
        let mut fraction_value = frac
            .parse::<i64>()
            .map_err(|_| Error::InvalidArgumentError("fractional second out of range".into()))?;
        for _ in 0..(9 - frac.len()) {
            fraction_value *= 10;
        }
        nanos = fraction_value;
    }
    Ok(SignedDecimal {
        seconds: seconds * i64::from(sign),
        nanos: nanos * i64::from(sign),
        int_digits: whole.len(),
    })
}

fn split_primary_component(value: &str, label: &str) -> Result<(SignedInteger, Option<String>)> {
    let mut parts = value.split_whitespace();
    let primary = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("{label} component missing")))?;
    let primary_parsed = parse_signed_integer(primary, label)?;
    let remainder = parts
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError(
            "too many components in interval literal".into(),
        ));
    }
    Ok((primary_parsed, remainder))
}

fn signed_component(sign: i32, magnitude: u64, label: &str) -> Result<i64> {
    let magnitude_i64 = i64::try_from(magnitude)
        .map_err(|_| Error::InvalidArgumentError(format!("{label} component out of range")))?;
    Ok(i64::from(sign) * magnitude_i64)
}

fn apply_sign_i64(sign: i32, magnitude: i64) -> i64 {
    i64::from(sign) * magnitude
}

fn compose_time_nanos(hours: i64, minutes: i64, seconds: i64, nanos: i64) -> Result<i64> {
    let hour_part = checked_mul(hours, NANOS_PER_HOUR, "hour")?;
    let minute_part = checked_mul(minutes, NANOS_PER_MINUTE, "minute")?;
    let second_part = checked_mul(seconds, NANOS_PER_SECOND, "second")?;
    hour_part
        .checked_add(minute_part)
        .and_then(|t| t.checked_add(second_part))
        .and_then(|t| t.checked_add(nanos))
        .ok_or_else(|| Error::InvalidArgumentError("interval overflow".into()))
}

fn checked_mul(value: i64, factor: i64, label: &str) -> Result<i64> {
    value.checked_mul(factor).ok_or_else(|| {
        Error::InvalidArgumentError(format!("{label} component overflowed interval range"))
    })
}

fn parse_unsigned_digits(value: &str, label: &str) -> Result<u64> {
    let trimmed = value.trim();
    if trimmed.is_empty() || !trimmed.chars().all(|c| c.is_ascii_digit()) {
        return Err(Error::InvalidArgumentError(format!(
            "invalid {label} component '{value}'"
        )));
    }
    trimmed
        .parse::<u64>()
        .map_err(|_| Error::InvalidArgumentError(format!("{label} component out of range")))
}

fn parse_signed_integer(value: &str, label: &str) -> Result<SignedInteger> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Error::InvalidArgumentError(format!(
            "{label} component cannot be empty"
        )));
    }
    let (sign, digits) = split_sign(trimmed);
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return Err(Error::InvalidArgumentError(format!(
            "invalid {label} component '{value}'"
        )));
    }
    let magnitude = digits
        .parse::<i64>()
        .map_err(|_| Error::InvalidArgumentError(format!("{label} component out of range")))?;
    Ok(SignedInteger {
        value: magnitude * i64::from(sign),
        sign,
        digits_len: digits.len(),
    })
}

fn enforce_precision(length: usize, precision: Option<u64>, label: &str) -> Result<()> {
    if let Some(max_digits) = precision
        && length as u64 > max_digits
    {
        return Err(Error::InvalidArgumentError(format!(
            "{label} exceeds leading precision {max_digits}"
        )));
    }
    Ok(())
}

fn ensure_no_fraction(value: &str, label: &str) -> Result<()> {
    if value.contains('.') {
        return Err(Error::InvalidArgumentError(format!(
            "fractional value not allowed for {label}"
        )));
    }
    Ok(())
}

fn split_sign(text: &str) -> (i32, &str) {
    match text.chars().next() {
        Some('-') => (-1, &text[1..]),
        Some('+') => (1, &text[1..]),
        _ => (1, text),
    }
}

fn interval_value_text(expr: &SqlExpr) -> Result<String> {
    match expr {
        SqlExpr::Value(value) => match &value.value {
            sqlparser::ast::Value::Number(text, _) => Ok(text.clone()),
            other => other.clone().into_string().ok_or_else(|| {
                Error::InvalidArgumentError("interval literal must be numeric or string".into())
            }),
        },
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => {
            let inner = interval_value_text(expr)?;
            if inner.starts_with('-') {
                Ok(inner)
            } else {
                Ok(format!("-{}", inner.trim()))
            }
        }
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => interval_value_text(expr),
        _ => Err(Error::InvalidArgumentError(
            "interval expression must be a literal value".into(),
        )),
    }
}

fn map_field(field: &DateTimeField) -> Result<IntervalUnit> {
    match field {
        DateTimeField::Year | DateTimeField::Years => Ok(IntervalUnit::Year),
        DateTimeField::Month | DateTimeField::Months => Ok(IntervalUnit::Month),
        DateTimeField::Day | DateTimeField::Days | DateTimeField::Date => Ok(IntervalUnit::Day),
        DateTimeField::Hour | DateTimeField::Hours => Ok(IntervalUnit::Hour),
        DateTimeField::Minute | DateTimeField::Minutes => Ok(IntervalUnit::Minute),
        DateTimeField::Second | DateTimeField::Seconds => Ok(IntervalUnit::Second),
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported interval field: {other:?}"
        ))),
    }
}

struct SignedInteger {
    value: i64,
    sign: i32,
    digits_len: usize,
}

struct SignedDecimal {
    seconds: i64,
    nanos: i64,
    int_digits: usize,
}
