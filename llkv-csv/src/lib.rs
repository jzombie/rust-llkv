use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::{Field, Schema, DataType};

/// Infer a simple Arrow schema from the first line (header) of a CSV file.
///
/// This builds a Schema where every column in the header is assigned DataType::Utf8
/// and is nullable. This is small, deterministic, and useful for quick validation tests.
pub fn infer_schema_from_header(path: &Path) -> Result<Arc<Schema>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut header_line = String::new();
    let _ = reader.read_line(&mut header_line)?;
    let header_line = header_line.trim_end_matches(['\n', '\r'].as_slice());
    let names: Vec<&str> = header_line.split(',').collect();

    let fields = names
        .into_iter()
        .map(|name| Field::new(name, DataType::Utf8, true))
        .collect::<Vec<Field>>();

    Ok(Arc::new(Schema::new(fields)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn infer_schema_simple_header() {
        let mut tmp = NamedTempFile::new().expect("create tmp");
        writeln!(tmp, "a,b,c").unwrap();
        writeln!(tmp, "1,2,3").unwrap();

        let schema = infer_schema_from_header(tmp.path()).expect("infer");
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "a");
        assert_eq!(schema.field(1).name(), "b");
        assert_eq!(schema.field(2).name(), "c");
        assert!(matches!(schema.field(0).data_type(), DataType::Utf8));
    }
}
