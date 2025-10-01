use std::io::Write;
use tempfile::NamedTempFile;

use arrow::datatypes::DataType;
use llkv_csv::CsvReader;

#[test]
fn integration_infer_schema_with_boolean_and_date_types() {
    let mut tmp = NamedTempFile::new().expect("create tmp");
    writeln!(tmp, "id,price,flag,created_on").unwrap();
    writeln!(tmp, "1,3.14,true,2024-01-01").unwrap();
    writeln!(tmp, "2,2.71,false,2024-01-02").unwrap();

    let reader = CsvReader::default();
    let schema = reader.infer_schema(tmp.path()).expect("infer");

    assert_eq!(schema.field(0).data_type(), &DataType::Int64);
    assert_eq!(schema.field(1).data_type(), &DataType::Float64);
    assert_eq!(schema.field(2).data_type(), &DataType::Boolean);
    assert_eq!(schema.field(3).data_type(), &DataType::Date32);

    let mut session = reader.open(tmp.path()).expect("open reader");
    let batch = session.next().expect("first batch").expect("batch ok");
    assert_eq!(batch.column(0).data_type(), &DataType::Int64);
    assert_eq!(batch.column(1).data_type(), &DataType::Float64);
    assert_eq!(batch.column(2).data_type(), &DataType::Boolean);
    assert_eq!(batch.column(3).data_type(), &DataType::Date32);
}
