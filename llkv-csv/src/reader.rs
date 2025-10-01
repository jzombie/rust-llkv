use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::csv::reader::{Format, Reader, ReaderBuilder};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;

use crate::CsvResult;
use crate::inference;

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
    pub(crate) fn to_format(&self) -> Format {
        let mut format = Format::default().with_header(self.has_header);
        if self.delimiter != b',' {
            format = format.with_delimiter(self.delimiter);
        }
        format
    }
}

#[derive(Debug, Clone, Default)]
pub struct CsvReader {
    options: CsvReadOptions,
}

impl CsvReader {
    pub fn new(options: CsvReadOptions) -> Self {
        Self { options }
    }

    pub fn with_options(options: CsvReadOptions) -> Self {
        Self::new(options)
    }

    pub fn options(&self) -> &CsvReadOptions {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut CsvReadOptions {
        &mut self.options
    }

    pub fn into_options(self) -> CsvReadOptions {
        self.options
    }

    pub fn infer_schema(&self, path: &Path) -> CsvResult<SchemaRef> {
        let outcome = inference::infer(path, &self.options)?;
        Ok(outcome.target_schema)
    }

    pub fn open(&self, path: &Path) -> CsvResult<CsvReadSession> {
        let outcome = inference::infer(path, &self.options)?;
        let file = File::open(path)?;

        let mut builder = ReaderBuilder::new(Arc::clone(&outcome.raw_schema))
            .with_format(self.options.to_format());
        if let Some(batch_size) = self.options.batch_size {
            builder = builder.with_batch_size(batch_size);
        }

        let reader = builder.build(file)?;
        Ok(CsvReadSession {
            schema: outcome.target_schema,
            type_overrides: outcome.type_overrides,
            reader,
        })
    }
}

pub struct CsvReadSession {
    schema: SchemaRef,
    type_overrides: Vec<Option<DataType>>,
    reader: Reader<File>,
}

impl CsvReadSession {
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    pub fn type_overrides(&self) -> &[Option<DataType>] {
        &self.type_overrides
    }

    pub fn into_parts(self) -> (SchemaRef, Reader<File>, Vec<Option<DataType>>) {
        (self.schema, self.reader, self.type_overrides)
    }

    pub fn reader(&mut self) -> &mut Reader<File> {
        &mut self.reader
    }
}

impl Iterator for CsvReadSession {
    type Item = arrow::error::Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.next()
    }
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
        let reader = CsvReader::default();
        let schema = reader.infer_schema(tmp.path()).expect("infer");

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
        let reader = CsvReader::new(options);

        let mut session = reader.open(tmp.path()).expect("open reader");
        assert_eq!(session.schema().field(0).data_type(), &DataType::Int64);

        let first = session.next().expect("first batch").expect("batch ok");
        assert_eq!(first.num_rows(), 1);
        assert_eq!(first.column(0).data_type(), &DataType::Int64);
    }
}
