use std::error::Error;
use std::fs::File;
use std::io::Seek;
use std::path::Path;
use std::sync::Arc;

use arrow::csv::reader::{Format, Reader, ReaderBuilder};
use arrow::datatypes::SchemaRef;

pub type CsvResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

pub mod csv_ingest;
pub use csv_ingest::append_csv_into_table;

#[derive(Debug, Clone)]
pub struct CsvReadOptions {
    pub has_header: bool,
    pub delimiter: u8,
    pub max_read_records: Option<usize>,
    pub batch_size: Option<usize>,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        Self {
            has_header: true,
            delimiter: b',',
            max_read_records: None,
            batch_size: None,
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

pub fn infer_schema(path: &Path, options: &CsvReadOptions) -> CsvResult<SchemaRef> {
    let mut file = File::open(path)?;
    let format = options.to_format();
    let (schema, _) = format.infer_schema(&mut file, options.max_read_records)?;
    Ok(Arc::new(schema))
}

pub fn open_csv_reader(
    path: &Path,
    options: &CsvReadOptions,
) -> CsvResult<(SchemaRef, Reader<File>)> {
    let mut file = File::open(path)?;
    let format = options.to_format();
    let (schema, _) = format.infer_schema(&mut file, options.max_read_records)?;
    file.rewind()?;

    let schema_ref = Arc::new(schema);
    let mut builder = ReaderBuilder::new(Arc::clone(&schema_ref)).with_format(format);
    if let Some(batch_size) = options.batch_size {
        builder = builder.with_batch_size(batch_size);
    }

    let reader = builder.build(file)?;
    Ok((schema_ref, reader))
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
        let (schema, mut reader) = open_csv_reader(tmp.path(), &options).expect("open reader");
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);

        let first = reader.next().expect("first batch").expect("batch ok");
        assert_eq!(first.num_rows(), 1);
        assert_eq!(first.column(0).data_type(), &DataType::Int64);
    }
}
