//! CSV schema inference helpers shared by the reader and ingestion pipeline.
//!
//! The inference pass samples the input file, coerces numeric-looking strings, and returns both
//! the normalized schema and per-column type overrides so the reader can request the desired Arrow
//! types.

use std::fs::File;
use std::io::{BufRead, Cursor};
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use llkv_column_map::store::ROW_ID_COLUMN_NAME;

use crate::{CsvReadOptions, CsvResult};

#[derive(Default, Clone)]
struct NumericStats {
    numeric: usize,
    non_numeric: usize,
    has_decimal: bool,
}

pub(crate) struct InferenceOutcome {
    pub target_schema: SchemaRef,
    pub raw_schema: SchemaRef,
    pub type_overrides: Vec<Option<DataType>>,
}

pub(crate) fn infer(path: &Path, options: &CsvReadOptions) -> CsvResult<InferenceOutcome> {
    let mut file = File::open(path)?;
    let format = options.to_format();

    let mut reader = std::io::BufReader::new(&mut file);
    let mut buf = String::new();
    reader.read_line(&mut buf)?; // header (with newline)
    let mut out = buf.clone();

    let delim = options.delimiter as char;
    let header_line = buf.trim_end_matches(&['\r', '\n'][..]);
    let header_fields = split_csv_line(header_line, delim);
    let mut stats = vec![NumericStats::default(); header_fields.len()];
    buf.clear();

    let mut count = 0usize;
    let max = options.max_read_records.unwrap_or(1000);
    while count < max && reader.read_line(&mut buf)? > 0 {
        let line = buf.trim_end_matches(&['\r', '\n'][..]);
        let fields = split_csv_line(line, delim);
        let mut rebuilt = String::new();

        for (idx, f) in fields.iter().enumerate() {
            if idx >= stats.len() {
                rebuilt.push_str(f);
                if idx + 1 < fields.len() {
                    rebuilt.push(delim);
                }
                continue;
            }

            let field_trimmed = f.trim();
            let inner = if field_trimmed.starts_with('"')
                && field_trimmed.ends_with('"')
                && field_trimmed.len() >= 2
            {
                &field_trimmed[1..field_trimmed.len() - 1]
            } else {
                field_trimmed
            };

            if let Some(token) = &options.null_token {
                if inner.eq_ignore_ascii_case(token.as_str()) {
                    // empty field
                } else if let Some((numeric, has_decimal)) = normalize_numeric_like(inner) {
                    stats[idx].numeric += 1;
                    stats[idx].has_decimal |= has_decimal;
                    rebuilt.push_str(&numeric);
                } else {
                    if !inner.is_empty() {
                        stats[idx].non_numeric += 1;
                    }
                    rebuilt.push_str(f);
                }
            } else if let Some((numeric, has_decimal)) = normalize_numeric_like(inner) {
                stats[idx].numeric += 1;
                stats[idx].has_decimal |= has_decimal;
                rebuilt.push_str(&numeric);
            } else {
                if !inner.is_empty() {
                    stats[idx].non_numeric += 1;
                }
                rebuilt.push_str(f);
            }

            if idx + 1 < fields.len() {
                rebuilt.push(delim);
            }
        }

        rebuilt.push('\n');
        out.push_str(&rebuilt);
        buf.clear();
        count += 1;
    }

    let mut cursor = Cursor::new(out.into_bytes());
    let (target_schema_raw, _) = format.infer_schema(&mut cursor, options.max_read_records)?;

    // Also infer the "raw" schema without normalization to determine the
    // types Arrow would parse directly from the file.
    let mut raw_file = File::open(path)?;
    let format_raw = options.to_format();
    let (raw_schema, _) = format_raw.infer_schema(&mut raw_file, options.max_read_records)?;

    let mut adjusted_fields: Vec<Field> = Vec::with_capacity(target_schema_raw.fields().len());
    let mut type_overrides = Vec::with_capacity(target_schema_raw.fields().len());

    for (idx, field) in target_schema_raw.fields().iter().enumerate() {
        if field.name() == ROW_ID_COLUMN_NAME {
            adjusted_fields.push(field.as_ref().clone());
            type_overrides.push(None);
            continue;
        }

        let mut target_field = field.as_ref().clone();
        let mut override_type: Option<DataType> = None;

        if idx < stats.len() {
            let stat = &stats[idx];
            if stat.numeric > 0 && stat.non_numeric == 0 {
                let data_type = if stat.has_decimal {
                    DataType::Float64
                } else {
                    DataType::Int64
                };
                target_field = Field::new(field.name(), data_type.clone(), true)
                    .with_metadata(field.metadata().clone());
                override_type = Some(data_type);
            }
        }

        adjusted_fields.push(target_field);

        let raw_dtype = raw_schema.field(idx).data_type();
        if let Some(ref desired) = override_type {
            if raw_dtype != desired {
                type_overrides.push(Some(desired.clone()))
            } else {
                type_overrides.push(None);
            }
        } else {
            type_overrides.push(None);
        }
    }

    let target_schema = Arc::new(Schema::new_with_metadata(
        adjusted_fields,
        target_schema_raw.metadata().clone(),
    ));

    Ok(InferenceOutcome {
        target_schema,
        raw_schema: Arc::new(raw_schema),
        type_overrides,
    })
}

pub(crate) fn split_csv_line(line: &str, delim: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut in_quotes = false;
    let mut start = 0usize;
    let mut i = 0usize;
    let bytes = line.as_bytes();
    while i < bytes.len() {
        let b = bytes[i] as char;
        if b == '"' {
            if in_quotes && i + 1 < bytes.len() && bytes[i + 1] as char == '"' {
                i += 2;
                continue;
            }
            in_quotes = !in_quotes;
        } else if b == delim && !in_quotes {
            parts.push(&line[start..i]);
            start = i + 1;
        }
        i += 1;
    }
    if start <= line.len() {
        parts.push(&line[start..]);
    }
    parts
}

pub(crate) fn normalize_numeric_like(s: &str) -> Option<(String, bool)> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (inner, neg) = if s.starts_with('(') && s.ends_with(')') && s.len() > 2 {
        (&s[1..s.len() - 1], true)
    } else {
        (s, false)
    };

    let mut cleaned = String::with_capacity(inner.len() + 1);
    for c in inner.chars() {
        match c {
            '$' | ',' | '%' | '\u{FEFF}' | '\u{00A0}' => {}
            _ => cleaned.push(c),
        }
    }

    if cleaned.is_empty() {
        return None;
    }

    if neg {
        cleaned.insert(0, '-');
    }

    let mut has_decimal = false;
    for (idx, ch) in cleaned.chars().enumerate() {
        match ch {
            '-' if idx == 0 => {}
            '.' => {
                if has_decimal {
                    return None;
                }
                has_decimal = true;
            }
            '0'..='9' => {}
            _ => return None,
        }
    }

    Some((cleaned, has_decimal))
}
