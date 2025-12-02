//! Query qualification helpers derived from the official TPC-H tooling.
//!
//! The TPC-H specification requires each implementation to validate query answers
//! against the published "qualification" answer sets before any performance
//! metrics are meaningful. This module loads those answer sets, executes the
//! canonical queries with the prescribed substitution parameters, and compares the
//! results with type-aware tolerances so engineers can spot regressions quickly.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Instant,
};

use arrow::{
    array::{
        Array, ArrayRef, Date32Array, Decimal128Array, Float32Array, Float64Array, Int32Array,
        Int64Array, LargeStringArray, StringArray, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, TimeUnit},
    record_batch::RecordBatch,
};
use llkv_sql::SqlEngine;
use rust_decimal::{
    Decimal,
    prelude::{FromPrimitive, ToPrimitive},
};
use time::{
    Date, Duration, Month, OffsetDateTime, format_description::FormatItem,
    macros::format_description,
};

use crate::queries::{QueryOptions, RenderedQuery, StatementKind, render_tpch_query};
use crate::{Result, SchemaPaths, TpchError};

const QUALIFICATION_QUERY_COUNT: u8 = 22;
const FLOAT_TOLERANCE: f64 = 1e-9;
const DATE_ONLY_FORMAT: &[FormatItem<'static>] = format_description!("[year]-[month]-[day]");
const DATETIME_FORMAT: &[FormatItem<'static>] =
    format_description!("[year]-[month]-[day] [hour]:[minute]:[second]");

static COLUMN_KINDS: OnceLock<Vec<Vec<ValueKind>>> = OnceLock::new();

/// Configuration for executing a TPC-H qualification run.
#[derive(Debug, Clone)]
pub struct QualificationOptions {
    dataset_dir: PathBuf,
    answers_dir: PathBuf,
    stream_number: u32,
    queries: Vec<u8>,
}

impl QualificationOptions {
    /// Create a qualification configuration rooted at the provided answer-set directory.
    pub fn new(dataset_dir: impl Into<PathBuf>) -> Self {
        let dataset_dir = dataset_dir.into();
        Self {
            answers_dir: dataset_dir.clone(),
            dataset_dir,
            stream_number: 0,
            queries: (1..=QUALIFICATION_QUERY_COUNT).collect(),
        }
    }

    /// Build options targeting the bundled `ref_data/<scale>` folder.
    pub fn from_scale(paths: &SchemaPaths, scale: impl AsRef<Path>) -> Self {
        Self::new(paths.ref_data_dir(scale)).with_answers(paths.answers_dir())
    }

    /// Override the stream number used for substitution parameters (defaults to 1).
    pub fn with_stream_number(mut self, stream: u32) -> Self {
        self.stream_number = stream;
        self
    }

    /// Restrict validation to the provided 1-based query numbers.
    pub fn with_queries(mut self, queries: impl Into<Vec<u8>>) -> Self {
        let mut entries = queries.into();
        entries.sort_unstable();
        entries.dedup();
        self.queries = entries;
        self
    }

    /// Override the dataset directory while preserving other parameters.
    pub fn with_dataset(mut self, dataset_dir: impl Into<PathBuf>) -> Self {
        self.dataset_dir = dataset_dir.into();
        self
    }

    /// Override the directory containing the expected answer sets.
    pub fn with_answers(mut self, answers_dir: impl Into<PathBuf>) -> Self {
        self.answers_dir = answers_dir.into();
        self
    }

    pub fn dataset_dir(&self) -> &Path {
        &self.dataset_dir
    }

    pub fn answers_dir(&self) -> &Path {
        &self.answers_dir
    }

    pub fn queries(&self) -> &[u8] {
        &self.queries
    }

    pub fn stream_number(&self) -> u32 {
        self.stream_number
    }
}

/// Summary of a single qualification comparison.
#[derive(Debug, Clone)]
pub struct QualificationReport {
    pub query: u8,
    pub status: QualificationStatus,
    pub expected_row_count: usize,
    pub actual_row_count: usize,
    pub missing_rows: Vec<Vec<String>>,
    pub extra_rows: Vec<Vec<String>>,
}

impl QualificationReport {
    /// Return `true` when the answer set matches the expected canonical output.
    pub fn passed(&self) -> bool {
        self.status == QualificationStatus::Pass
    }
}

/// Indicates whether a qualification comparison succeeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationStatus {
    Pass,
    Fail,
}

/// Execute a qualification run for the provided SQL engine.
///
/// The caller must supply a directory containing canonical answer sets named
/// `q1.out` through `q22.out` and the corresponding `subparam_<stream>` file.
/// Each answer set must follow the format expected by the official TPC-H
/// `check_answers` scripts: a single header line followed by pipe-delimited data
/// rows without leading or trailing pipe characters.
#[allow(clippy::print_stdout)]
pub fn run_qualification(
    engine: &SqlEngine,
    paths: &SchemaPaths,
    schema: &str,
    options: &QualificationOptions,
) -> Result<Vec<QualificationReport>> {
    validate_query_range(options.queries())?;

    let column_kinds = column_kinds(paths)?;
    let stream_parameters = if options.stream_number() == 0 {
        HashMap::new()
    } else {
        load_stream_parameters(options.dataset_dir(), options.stream_number())?
    };

    let mut reports = Vec::with_capacity(options.queries().len());

    for &query in options.queries() {
        print!("  Q{:02}: executing... ", query);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let kinds = column_kinds
            .get((query - 1) as usize)
            .ok_or_else(|| TpchError::Parse(format!("no column metadata for query {query}")))?;

        let answer_path = options.answers_dir().join(format!("q{query}.out"));
        let expected_rows = load_answer_set(&answer_path, kinds)?;
        let rendered = render_query(
            paths,
            query,
            schema,
            options.stream_number(),
            &stream_parameters,
        )?;
        let exec_start = Instant::now();
        let actual_rows = execute_query(engine, &rendered, kinds)?;
        let exec_elapsed = exec_start.elapsed();

        print!("({:.2}s) comparing... ", exec_elapsed.as_secs_f64());
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let compare_start = Instant::now();
        let diff = diff_rows(&expected_rows, &actual_rows, kinds);
        let compare_elapsed = compare_start.elapsed();

        let status = if diff.missing.is_empty() && diff.extra.is_empty() {
            println!(
                "PASS ({} rows) [exec {:.2}s | compare {:.2}s]",
                actual_rows.len(),
                exec_elapsed.as_secs_f64(),
                compare_elapsed.as_secs_f64()
            );
            QualificationStatus::Pass
        } else {
            println!(
                "FAIL (expected {}, actual {}) [exec {:.2}s | compare {:.2}s]",
                expected_rows.len(),
                actual_rows.len(),
                exec_elapsed.as_secs_f64(),
                compare_elapsed.as_secs_f64()
            );
            QualificationStatus::Fail
        };

        reports.push(QualificationReport {
            query,
            status,
            expected_row_count: expected_rows.len(),
            actual_row_count: actual_rows.len(),
            missing_rows: diff.missing.into_iter().map(|row| row.display).collect(),
            extra_rows: diff.extra.into_iter().map(|row| row.display).collect(),
        });
    }

    Ok(reports)
}

/// Ensure all required qualification artifacts exist before running comparisons.
pub fn verify_qualification_assets(options: &QualificationOptions) -> Result<()> {
    let dir = options.dataset_dir();
    if !dir.is_dir() {
        return Err(TpchError::Parse(format!(
            "qualification dataset directory '{}' does not exist",
            dir.display()
        )));
    }
    if options.stream_number() != 0 {
        let subparam_path = dir.join(format!("subparam_{}", options.stream_number()));
        if !subparam_path.is_file() {
            return Err(TpchError::Parse(format!(
                "qualification dataset missing stream parameters file '{}'",
                subparam_path.display()
            )));
        }
    }
    for &query in options.queries() {
        let answer_path = options.answers_dir().join(format!("q{query}.out"));
        if !answer_path.is_file() {
            return Err(TpchError::Parse(format!(
                "qualification dataset missing answer set '{}'",
                answer_path.display()
            )));
        }
    }
    Ok(())
}

fn render_query(
    paths: &SchemaPaths,
    number: u8,
    schema: &str,
    stream_number: u32,
    stream_parameters: &HashMap<u8, Vec<String>>,
) -> Result<RenderedQuery> {
    let mut options = QueryOptions {
        stream_number,
        parameter_overrides: BTreeMap::new(),
    };

    if let Some(values) = stream_parameters.get(&number) {
        for (idx, value) in values.iter().enumerate() {
            options.parameter_overrides.insert(idx + 1, value.clone());
        }
    }

    render_tpch_query(paths, number, schema, &options)
}

#[allow(clippy::print_stdout)]
fn execute_query(
    engine: &SqlEngine,
    rendered: &RenderedQuery,
    kinds: &[ValueKind],
) -> Result<Vec<QualificationRow>> {
    let mut last_set = None;

    for (idx, statement) in rendered.statements.iter().enumerate() {
        match statement.kind {
            StatementKind::Command => {
                print!("[cmd {}]... ", idx + 1);
                std::io::Write::flush(&mut std::io::stdout()).ok();
                engine.execute(&statement.sql).map(|_| ())?;
            }
            StatementKind::Query => {
                print!("[query {}]... ", idx + 1);
                std::io::Write::flush(&mut std::io::stdout()).ok();
                let batches = engine.sql(&statement.sql)?;
                print!("[collect]... ");
                std::io::Write::flush(&mut std::io::stdout()).ok();
                let rows = collect_rows(&batches, kinds)?;
                last_set = Some(rows);
            }
        }
    }

    last_set.ok_or_else(|| {
        TpchError::Parse(format!(
            "query {} did not produce a result set",
            rendered.number
        ))
    })
}

fn collect_rows(batches: &[RecordBatch], kinds: &[ValueKind]) -> Result<Vec<QualificationRow>> {
    let mut rows = Vec::new();

    for batch in batches {
        if batch.num_columns() != kinds.len() {
            return Err(TpchError::Parse(format!(
                "result produced {} columns but qualification metadata expects {}",
                batch.num_columns(),
                kinds.len()
            )));
        }

        for row_idx in 0..batch.num_rows() {
            let mut values = Vec::with_capacity(kinds.len());
            let mut display = Vec::with_capacity(kinds.len());

            for (col_idx, kind) in kinds.iter().enumerate() {
                let array = batch.column(col_idx);
                let value = extract_value(array, row_idx, *kind)?;
                display.push(format_value(&value));
                values.push(value);
            }

            rows.push(QualificationRow { values, display });
        }
    }

    Ok(rows)
}

fn extract_value(array: &ArrayRef, row: usize, kind: ValueKind) -> Result<QualificationValue> {
    if array.is_null(row) {
        return Ok(QualificationValue::Null);
    }

    match kind {
        ValueKind::String => Ok(QualificationValue::String(extract_string(array, row)?)),
        ValueKind::Integer => Ok(QualificationValue::Int(extract_integer(array, row)?)),
        ValueKind::Decimal => Ok(QualificationValue::Decimal(extract_decimal(array, row)?)),
        ValueKind::Float => Ok(QualificationValue::Float(extract_float(array, row)?)),
    }
}

fn extract_string(array: &ArrayRef, row: usize) -> Result<String> {
    match array.data_type() {
        DataType::Utf8 => Ok(array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| TpchError::Parse("expected Utf8 array".into()))?
            .value(row)
            .to_string()),
        DataType::LargeUtf8 => Ok(array
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .ok_or_else(|| TpchError::Parse("expected LargeUtf8 array".into()))?
            .value(row)
            .to_string()),
        DataType::Date32 => {
            let days = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| TpchError::Parse("expected Date32 array".into()))?
                .value(row);
            Ok(date32_to_string(days))
        }
        DataType::Date64 => {
            // Date64 is milliseconds since epoch.
            let millis = array
                .as_any()
                .downcast_ref::<arrow::array::Date64Array>()
                .ok_or_else(|| TpchError::Parse("expected Date64 array".into()))?
                .value(row);
            let dt = OffsetDateTime::from_unix_timestamp_nanos(millis as i128 * 1_000_000)
                .map_err(|_| TpchError::Parse("invalid Date64 value".into()))?;
            Ok(format_date(dt.date()))
        }
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            let micros = array
                .as_any()
                .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
                .ok_or_else(|| TpchError::Parse("expected TimestampMicrosecond array".into()))?
                .value(row);
            let dt = OffsetDateTime::from_unix_timestamp_nanos(micros as i128 * 1_000)
                .map_err(|_| TpchError::Parse("invalid timestamp".into()))?;
            Ok(format_datetime(dt))
        }
        // Fallback to string formatting for numeric arrays when they are projected into
        // character columns (should not happen in canonical queries, but keeps the
        // conversion robust if the engine widens types internally).
        DataType::Int32 => Ok(array
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| TpchError::Parse("expected Int32 array".into()))?
            .value(row)
            .to_string()),
        DataType::Int64 => Ok(array
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| TpchError::Parse("expected Int64 array".into()))?
            .value(row)
            .to_string()),
        DataType::UInt32 => Ok(array
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| TpchError::Parse("expected UInt32 array".into()))?
            .value(row)
            .to_string()),
        DataType::UInt64 => Ok(array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| TpchError::Parse("expected UInt64 array".into()))?
            .value(row)
            .to_string()),
        DataType::Decimal128(_, scale) => {
            let value = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| TpchError::Parse("expected Decimal128 array".into()))?
                .value(row);
            let decimal = Decimal::from_i128_with_scale(value, *scale as u32);
            Ok(decimal.normalize().to_string())
        }
        DataType::Float64 => Ok(array
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| TpchError::Parse("expected Float64 array".into()))?
            .value(row)
            .to_string()),
        DataType::Float32 => Ok(array
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| TpchError::Parse("expected Float32 array".into()))?
            .value(row)
            .to_string()),
        other => Err(TpchError::Parse(format!(
            "unsupported Arrow type {:?} for string projection",
            other
        ))),
    }
}

fn extract_integer(array: &ArrayRef, row: usize) -> Result<i128> {
    match array.data_type() {
        DataType::Int32 => Ok(array
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| TpchError::Parse("expected Int32 array".into()))?
            .value(row) as i128),
        DataType::Int64 => Ok(array
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| TpchError::Parse("expected Int64 array".into()))?
            .value(row) as i128),
        DataType::UInt32 => Ok(array
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| TpchError::Parse("expected UInt32 array".into()))?
            .value(row) as i128),
        DataType::UInt64 => Ok(array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| TpchError::Parse("expected UInt64 array".into()))?
            .value(row) as i128),
        DataType::Decimal128(_, scale) => {
            if *scale != 0 {
                return Err(TpchError::Parse(
                    "expected integral decimal with zero scale".into(),
                ));
            }
            let value = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| TpchError::Parse("expected Decimal128 array".into()))?
                .value(row);
            Ok(value)
        }
        _ => extract_string(array, row)?
            .parse::<i128>()
            .map_err(|err| TpchError::Parse(format!("unable to parse integer: {err}"))),
    }
}

fn extract_decimal(array: &ArrayRef, row: usize) -> Result<Decimal> {
    match array.data_type() {
        DataType::Decimal128(_, scale) => {
            let value = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| TpchError::Parse("expected Decimal128 array".into()))?
                .value(row);
            Ok(Decimal::from_i128_with_scale(value, *scale as u32).normalize())
        }
        DataType::Int32 => Ok(Decimal::from(
            array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| TpchError::Parse("expected Int32 array".into()))?
                .value(row),
        )),
        DataType::Int64 => Ok(Decimal::from(
            array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| TpchError::Parse("expected Int64 array".into()))?
                .value(row),
        )),
        DataType::UInt32 => Ok(Decimal::from(
            array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| TpchError::Parse("expected UInt32 array".into()))?
                .value(row),
        )),
        DataType::UInt64 => Ok(Decimal::from(
            array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| TpchError::Parse("expected UInt64 array".into()))?
                .value(row),
        )),
        DataType::Float64 => {
            let value = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| TpchError::Parse("expected Float64 array".into()))?
                .value(row);
            Decimal::from_f64(value)
                .ok_or_else(|| TpchError::Parse("unable to convert float to decimal".into()))
        }
        DataType::Float32 => {
            let value = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| TpchError::Parse("expected Float32 array".into()))?
                .value(row);
            Decimal::from_f32(value)
                .ok_or_else(|| TpchError::Parse("unable to convert float to decimal".into()))
        }
        _ => {
            let text = extract_string(array, row)?;
            Decimal::from_str_exact(&text).map_err(|err| {
                TpchError::Parse(format!("unable to parse decimal literal '{text}': {err}"))
            })
        }
    }
}

fn extract_float(array: &ArrayRef, row: usize) -> Result<f64> {
    match array.data_type() {
        DataType::Float64 => Ok(array
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| TpchError::Parse("expected Float64 array".into()))?
            .value(row)),
        DataType::Float32 => Ok(array
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| TpchError::Parse("expected Float32 array".into()))?
            .value(row) as f64),
        DataType::Decimal128(_, scale) => {
            let value = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| TpchError::Parse("expected Decimal128 array".into()))?
                .value(row);
            let decimal = Decimal::from_i128_with_scale(value, *scale as u32);
            decimal.to_f64().ok_or_else(|| {
                TpchError::Parse("unable to convert decimal to floating-point".into())
            })
        }
        DataType::Int32 => Ok(array
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| TpchError::Parse("expected Int32 array".into()))?
            .value(row) as f64),
        DataType::Int64 => Ok(array
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| TpchError::Parse("expected Int64 array".into()))?
            .value(row) as f64),
        _ => extract_string(array, row)?
            .parse::<f64>()
            .map_err(|err| TpchError::Parse(format!("unable to parse float: {err}"))),
    }
}

fn date32_to_string(days: i32) -> String {
    let base = unix_epoch_date();
    let delta = Duration::days(days as i64);
    let date = base.checked_add(delta).unwrap_or(base);
    format_date(date)
}

fn unix_epoch_date() -> Date {
    Date::from_calendar_date(1970, Month::January, 1).expect("valid epoch")
}

fn format_date(date: Date) -> String {
    date.format(DATE_ONLY_FORMAT)
        .unwrap_or_else(|_| "1970-01-01".to_string())
}

fn format_datetime(dt: OffsetDateTime) -> String {
    dt.format(DATETIME_FORMAT)
        .unwrap_or_else(|_| "1970-01-01 00:00:00".to_string())
}

fn format_value(value: &QualificationValue) -> String {
    match value {
        QualificationValue::Null => "NULL".to_string(),
        QualificationValue::String(text) => text.clone(),
        QualificationValue::Int(v) => v.to_string(),
        QualificationValue::Decimal(v) => format_decimal(v),
        QualificationValue::Float(v) => format_float(*v),
    }
}

fn format_decimal(value: &Decimal) -> String {
    let mut text = value.normalize().to_string();
    if text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.push('0');
        }
    }
    text
}

fn format_float(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }

    let precision = 12;
    let mut text = format!("{value:.precision$}");
    if text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.push('0');
        }
    }
    text
}

#[derive(Debug, Clone)]
struct QualificationRow {
    values: Vec<QualificationValue>,
    display: Vec<String>,
}

#[derive(Debug, Clone)]
struct RowDiff {
    missing: Vec<QualificationRow>,
    extra: Vec<QualificationRow>,
}

fn diff_rows(
    expected: &[QualificationRow],
    actual: &[QualificationRow],
    kinds: &[ValueKind],
) -> RowDiff {
    let mut missing_flags = vec![true; expected.len()];
    let mut extra = Vec::new();

    for actual_row in actual {
        if let Some((idx, _)) = expected.iter().enumerate().find(|(idx, expected_row)| {
            missing_flags[*idx] && rows_equal(expected_row, actual_row, kinds)
        }) {
            missing_flags[idx] = false;
        } else {
            extra.push(actual_row.clone());
        }
    }

    let missing = expected
        .iter()
        .zip(missing_flags.iter())
        .filter_map(|(row, missing)| if *missing { Some(row.clone()) } else { None })
        .collect();

    RowDiff { missing, extra }
}

fn rows_equal(a: &QualificationRow, b: &QualificationRow, kinds: &[ValueKind]) -> bool {
    for ((expected, actual), kind) in a.values.iter().zip(&b.values).zip(kinds) {
        if !values_equal(expected, actual, *kind) {
            return false;
        }
    }
    true
}

fn values_equal(
    expected: &QualificationValue,
    actual: &QualificationValue,
    kind: ValueKind,
) -> bool {
    match (expected, actual) {
        (QualificationValue::Null, QualificationValue::Null) => true,
        (QualificationValue::Null, _) | (_, QualificationValue::Null) => false,
        (QualificationValue::String(a), QualificationValue::String(b)) => a == b,
        (QualificationValue::Int(a), QualificationValue::Int(b)) => a == b,
        (QualificationValue::Decimal(a), QualificationValue::Decimal(b)) => a == b,
        (QualificationValue::Float(a), QualificationValue::Float(b)) => {
            (a - b).abs() <= FLOAT_TOLERANCE
        }
        // Allow small tolerances when the engine emits decimals for float-typed results or vice versa.
        (QualificationValue::Decimal(a), QualificationValue::Float(b)) => {
            if let Some(decimal_b) = Decimal::from_f64(*b) {
                (*a - decimal_b).abs() <= Decimal::from_f64(FLOAT_TOLERANCE).unwrap_or_default()
            } else {
                false
            }
        }
        (QualificationValue::Float(a), QualificationValue::Decimal(b)) => {
            if let Some(decimal_a) = Decimal::from_f64(*a) {
                (decimal_a - *b).abs()
                    <= Decimal::from_f64(FLOAT_TOLERANCE).unwrap_or_else(|| Decimal::from(0))
            } else {
                false
            }
        }
        _ => match kind {
            ValueKind::String => false,
            ValueKind::Integer => false,
            ValueKind::Decimal => false,
            ValueKind::Float => false,
        },
    }
}

fn load_stream_parameters(
    dataset_dir: &Path,
    stream_number: u32,
) -> Result<HashMap<u8, Vec<String>>> {
    let path = dataset_dir.join(format!("subparam_{stream_number}"));
    let contents = fs::read_to_string(&path).map_err(|err| TpchError::Io {
        path: path.clone(),
        source: err,
    })?;

    let mut map = HashMap::new();

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let fields: Vec<&str> = trimmed.split('\t').collect();
        if fields.is_empty() {
            continue;
        }
        let query = fields[0].parse::<u8>().map_err(|err| {
            TpchError::Parse(format!(
                "unable to parse query number '{:?}' in {}: {err}",
                fields.first(),
                path.display()
            ))
        })?;
        let values = fields[1..]
            .iter()
            .map(|value| value.trim().to_string())
            .collect();
        map.insert(query, values);
    }

    Ok(map)
}

fn load_answer_set(path: &Path, kinds: &[ValueKind]) -> Result<Vec<QualificationRow>> {
    let contents = fs::read_to_string(path).map_err(|err| TpchError::Io {
        path: path.to_path_buf(),
        source: err,
    })?;

    let mut rows = Vec::new();
    let mut header_skipped = false;

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !header_skipped {
            header_skipped = true;
            continue;
        }
        rows.push(parse_answer_row(trimmed, kinds)?);
    }

    Ok(rows)
}

fn parse_answer_row(line: &str, kinds: &[ValueKind]) -> Result<QualificationRow> {
    let parts: Vec<&str> = line.split('|').map(|field| field.trim()).collect();
    if parts.len() != kinds.len() {
        return Err(TpchError::Parse(format!(
            "answer row has {} fields but query expects {}",
            parts.len(),
            kinds.len()
        )));
    }

    let mut values = Vec::with_capacity(parts.len());
    let mut display = Vec::with_capacity(parts.len());

    for (field, kind) in parts.iter().zip(kinds) {
        let value = parse_expected_value(field, *kind)?;
        display.push((*field).to_string());
        values.push(value);
    }

    Ok(QualificationRow { values, display })
}

fn parse_expected_value(text: &str, kind: ValueKind) -> Result<QualificationValue> {
    if text.eq_ignore_ascii_case("NULL") {
        return Ok(QualificationValue::Null);
    }

    match kind {
        ValueKind::String => Ok(QualificationValue::String(text.to_string())),
        ValueKind::Integer => text
            .parse::<i128>()
            .map(QualificationValue::Int)
            .map_err(|err| TpchError::Parse(format!("unable to parse integer '{text}': {err}"))),
        ValueKind::Decimal => Decimal::from_str_exact(text)
            .map(|decimal| QualificationValue::Decimal(decimal.normalize()))
            .map_err(|err| TpchError::Parse(format!("unable to parse decimal '{text}': {err}"))),
        ValueKind::Float => text
            .parse::<f64>()
            .map(QualificationValue::Float)
            .map_err(|err| TpchError::Parse(format!("unable to parse float '{text}': {err}"))),
    }
}

fn column_kinds(paths: &SchemaPaths) -> Result<&'static Vec<Vec<ValueKind>>> {
    if let Some(kinds) = COLUMN_KINDS.get() {
        return Ok(kinds);
    }

    let parsed = parse_column_kinds(paths)?;
    let _ = COLUMN_KINDS.set(parsed);
    COLUMN_KINDS
        .get()
        .ok_or_else(|| TpchError::Parse("failed to cache column metadata".into()))
}

fn parse_column_kinds(paths: &SchemaPaths) -> Result<Vec<Vec<ValueKind>>> {
    let path = paths.check_answers_dir().join("colprecision.txt");
    let contents = fs::read_to_string(&path).map_err(|err| TpchError::Io { path, source: err })?;

    let mut kinds = Vec::new();

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let tokens = trimmed.split_whitespace();
        let mut row_kinds = Vec::new();
        for token in tokens {
            row_kinds.push(ValueKind::from_token(token)?);
        }
        kinds.push(row_kinds);
    }

    if kinds.len() != QUALIFICATION_QUERY_COUNT as usize {
        return Err(TpchError::Parse(format!(
            "colprecision.txt lists {} queries, expected {}",
            kinds.len(),
            QUALIFICATION_QUERY_COUNT
        )));
    }

    Ok(kinds)
}

fn validate_query_range(queries: &[u8]) -> Result<()> {
    if queries.is_empty() {
        return Err(TpchError::Parse(
            "qualification set may not be empty".into(),
        ));
    }

    for &query in queries {
        if query == 0 || query > QUALIFICATION_QUERY_COUNT {
            return Err(TpchError::Parse(format!(
                "query {query} is outside the canonical TPC-H range"
            )));
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum ValueKind {
    String,
    Integer,
    Decimal,
    Float,
}

impl ValueKind {
    fn from_token(token: &str) -> Result<Self> {
        match token {
            "str" => Ok(ValueKind::String),
            "int" | "cnt" => Ok(ValueKind::Integer),
            "sum" | "num" => Ok(ValueKind::Decimal),
            "avg" | "rat" => Ok(ValueKind::Float),
            other => Err(TpchError::Parse(format!(
                "unrecognized colprecision token '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
enum QualificationValue {
    Null,
    String(String),
    Int(i128),
    Decimal(Decimal),
    Float(f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_expected_value_variants() {
        assert!(matches!(
            parse_expected_value("NULL", ValueKind::Integer).unwrap(),
            QualificationValue::Null
        ));
        assert!(matches!(
            parse_expected_value("hello", ValueKind::String).unwrap(),
            QualificationValue::String(ref value) if value == "hello"
        ));
        assert!(matches!(
            parse_expected_value("42", ValueKind::Integer).unwrap(),
            QualificationValue::Int(42)
        ));
        assert!(matches!(
            parse_expected_value("12.50", ValueKind::Decimal).unwrap(),
            QualificationValue::Decimal(ref value) if value == &Decimal::from_str_exact("12.50").unwrap()
        ));
        assert!(matches!(
            parse_expected_value("0.125", ValueKind::Float).unwrap(),
            QualificationValue::Float(value) if (value - 0.125).abs() < 1e-9
        ));
    }

    #[test]
    fn test_values_equal_decimal_vs_float() {
        let decimal = QualificationValue::Decimal(Decimal::from_str_exact("1.2345").unwrap());
        let float = QualificationValue::Float(1.2345);
        assert!(values_equal(&decimal, &float, ValueKind::Float));
        assert!(values_equal(&float, &decimal, ValueKind::Float));
    }

    #[test]
    fn test_diff_rows_detects_missing_and_extra() {
        let kinds = vec![ValueKind::Integer];
        let expected = vec![QualificationRow {
            values: vec![QualificationValue::Int(1)],
            display: vec!["1".to_string()],
        }];
        let actual = vec![QualificationRow {
            values: vec![QualificationValue::Int(2)],
            display: vec!["2".to_string()],
        }];
        let diff = diff_rows(&expected, &actual, &kinds);
        assert_eq!(diff.missing.len(), 1);
        assert_eq!(diff.extra.len(), 1);
    }
}
