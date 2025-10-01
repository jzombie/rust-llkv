use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::csv::reader::{Format, Reader, ReaderBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use llkv_column_map::store::ROW_ID_COLUMN_NAME;

pub type CsvResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

pub mod csv_ingest;
pub use csv_ingest::{append_csv_into_table, append_csv_into_table_with_mapping};

pub mod csv_export;
pub use csv_export::{
    CsvExportColumn, CsvWriteOptions, export_csv_from_table, export_csv_from_table_with_filter,
    export_csv_from_table_with_projections, export_csv_to_writer_with_filter,
    export_csv_to_writer_with_projections,
};

#[derive(Debug, Clone)]
pub struct CsvReadOptions {
    pub has_header: bool,
    pub delimiter: u8,
    pub max_read_records: Option<usize>,
    pub batch_size: Option<usize>,
    pub null_token: Option<String>,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        Self {
            has_header: true,
            delimiter: b',',
            max_read_records: None,
            batch_size: None,
            null_token: None,
        }
    }
}

impl CsvReadOptions {
    fn to_format(&self) -> Format {
        let mut format = Format::default().with_header(self.has_header);
        if self.delimiter != b',' {
            format = format.with_delimiter(self.delimiter);
        }
        format
    }
}

#[derive(Default, Clone)]
struct NumericStats {
    numeric: usize,
    non_numeric: usize,
    has_decimal: bool,
}

pub(crate) fn split_csv_line<'a>(line: &'a str, delim: char) -> Vec<&'a str> {
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

fn infer_schema_internal(
    path: &Path,
    options: &CsvReadOptions,
) -> CsvResult<(SchemaRef, SchemaRef, Vec<Option<DataType>>)> {
    use std::io::{BufRead, Cursor};

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

    Ok((target_schema, Arc::new(raw_schema), type_overrides))
}

pub fn infer_schema(path: &Path, options: &CsvReadOptions) -> CsvResult<SchemaRef> {
    let (schema, _, _) = infer_schema_internal(path, options)?;
    Ok(schema)
}

pub fn open_csv_reader(
    path: &Path,
    options: &CsvReadOptions,
) -> CsvResult<(SchemaRef, Reader<File>, Vec<Option<DataType>>)> {
    let (target_schema, raw_schema, overrides) = infer_schema_internal(path, options)?;
    let file = File::open(path)?;
    let format = options.to_format();
    let mut builder = ReaderBuilder::new(Arc::clone(&raw_schema)).with_format(format);
    if let Some(batch_size) = options.batch_size {
        builder = builder.with_batch_size(batch_size);
    }

    let reader = builder.build(file)?;
    Ok((target_schema, reader, overrides))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::DataType;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_sample_csv() -> NamedTempFile {
        let mut tmp = NamedTempFile::new().expect("create tmp");
        writeln!(tmp, "id,price,flag,timestamp,text").unwrap();
        writeln!(tmp, "1,3.14,true,2024-01-01T12:34:56Z,hello").unwrap();
        writeln!(tmp, "2,2.71,false,2024-01-02T01:02:03Z,world").unwrap();
        tmp
    }

    #[test]
    fn infer_schema_detects_types() {
        let tmp = write_sample_csv();
        let options = CsvReadOptions::default();
        let schema = infer_schema(tmp.path(), &options).expect("infer");

        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert_eq!(schema.field(1).data_type(), &DataType::Float64);
        assert_eq!(schema.field(2).data_type(), &DataType::Boolean);
        assert!(matches!(
            schema.field(3).data_type(),
            DataType::Timestamp(_, _)
        ));
        assert_eq!(schema.field(4).data_type(), &DataType::Utf8);
    }

    #[test]
    fn reader_streams_batches() {
        let tmp = write_sample_csv();
        let options = CsvReadOptions {
            batch_size: Some(1),
            ..Default::default()
        };
    let (schema, mut reader, _) = open_csv_reader(tmp.path(), &options).expect("open reader");
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);

        let first = reader.next().expect("first batch").expect("batch ok");
        assert_eq!(first.num_rows(), 1);
        assert_eq!(first.column(0).data_type(), &DataType::Int64);
    }
}
