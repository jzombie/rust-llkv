use std::io::Write;
use tempfile::NamedTempFile;

use arrow::datatypes::DataType;
use llkv_csv::{CsvReadOptions, infer_schema, open_csv_reader};

#[test]
fn integration_infer_schema_with_numeric_types() {
    let mut tmp = NamedTempFile::new().expect("create tmp");
    writeln!(tmp, "id,price,flag").unwrap();
    writeln!(tmp, "1,3.14,true").unwrap();
    writeln!(tmp, "2,2.71,false").unwrap();

    let options = CsvReadOptions::default();
    let schema = infer_schema(tmp.path(), &options).expect("infer");

    assert_eq!(schema.field(0).data_type(), &DataType::Int64);
    assert_eq!(schema.field(1).data_type(), &DataType::Float64);
    assert_eq!(schema.field(2).data_type(), &DataType::Boolean);

    let (_, mut reader) = open_csv_reader(tmp.path(), &options).expect("open reader");
    let batch = reader.next().expect("first batch").expect("batch ok");
    assert_eq!(batch.column(0).data_type(), &DataType::Int64);
    assert_eq!(batch.column(1).data_type(), &DataType::Float64);
    assert_eq!(batch.column(2).data_type(), &DataType::Boolean);
}
