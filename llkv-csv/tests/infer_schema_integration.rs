use std::io::Write;
use tempfile::NamedTempFile;

use llkv_csv::infer_schema_from_header;
use arrow::datatypes::DataType;

#[test]
fn integration_infer_schema_from_header() {
    let mut tmp = NamedTempFile::new().expect("create tmp");
    writeln!(tmp, "col1,col2,col3").unwrap();
    writeln!(tmp, "a,b,c").unwrap();

    let schema = infer_schema_from_header(tmp.path()).expect("infer");
    assert_eq!(schema.fields().len(), 3);
    assert_eq!(schema.field(0).name(), "col1");
    assert_eq!(schema.field(1).name(), "col2");
    assert_eq!(schema.field(2).name(), "col3");
    assert!(matches!(schema.field(0).data_type(), DataType::Utf8));
}
