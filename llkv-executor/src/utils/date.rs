use llkv_result::{Error, Result as LlkvResult};
use time::{Date, Month};

/// Parse a string literal formatted as `YYYY-MM-DD` into the Arrow `Date32` day count.
pub fn parse_date32_literal(text: &str) -> LlkvResult<i32> {
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

    let month = Month::try_from(month_num).map_err(|_| {
        Error::InvalidArgumentError(format!("invalid month in DATE literal '{text}'"))
    })?;

    let date = Date::from_calendar_date(year, month, day).map_err(|err| {
        Error::InvalidArgumentError(format!("invalid DATE literal '{text}': {err}"))
    })?;
    let days = date.to_julian_day() - epoch_julian_day();
    Ok(days)
}

pub fn epoch_julian_day() -> i32 {
    Date::from_calendar_date(1970, Month::January, 1)
        .expect("1970-01-01 is a valid date")
        .to_julian_day()
}

/// Format an Arrow `Date32` day count into `YYYY-MM-DD` text.
pub fn format_date32_literal(days: i32) -> LlkvResult<String> {
    let julian = epoch_julian_day()
        .checked_add(days)
        .ok_or_else(|| Error::InvalidArgumentError("date literal out of range".into()))?;

    let date = Date::from_julian_day(julian)
        .map_err(|err| Error::InvalidArgumentError(format!("invalid DATE value: {err}")))?;
    let (year, month, day) = date.to_calendar_date();
    let month_number = month as u8;
    Ok(format!("{:04}-{:02}-{:02}", year, month_number, day))
}
