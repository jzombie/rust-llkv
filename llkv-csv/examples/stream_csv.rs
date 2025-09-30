use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::env;

use std::sync::Arc;

use arrow::csv as arrow_csv;
use arrow::datatypes::{Field, Schema, DataType};
use arrow::util::pretty::print_batches;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("Usage: stream_csv <path-to-csv>");
            std::process::exit(2);
        }
    };

    let file = File::open(&path)?;
    let mut reader = BufReader::new(file);

    // Read the first line to extract header names and build a simple Utf8 schema.
    let mut header_line = String::new();
    let _ = reader.read_line(&mut header_line)?;
    let header_line = header_line.trim_end_matches(['\n', '\r'].as_slice());
    let names: Vec<&str> = header_line.split(',').collect();

    let fields = names
        .into_iter()
        .map(|name| Field::new(name, DataType::Utf8, true))
        .collect::<Vec<Field>>();

    let schema = Arc::new(Schema::new(fields));

    // Build the arrow-csv reader with the schema we created. The ReaderBuilder expects a SchemaRef.
    let csv_reader = arrow_csv::reader::ReaderBuilder::new(schema)
        .with_header(true)
        .build(reader)
        .map_err(|e| Box::<dyn Error>::from(e))?;

    // `csv::Reader` implements Iterator<Item = ArrowResult<RecordBatch>>
    for maybe_batch in csv_reader {
        let batch = maybe_batch.map_err(|e| Box::<dyn Error>::from(e))?;
        // Print the record batch to stdout in a human friendly table format.
        print_batches(&[batch]).map_err(|e| Box::<dyn Error>::from(e))?;
    }

    Ok(())
}
