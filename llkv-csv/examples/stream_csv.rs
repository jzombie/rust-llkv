use std::env;
use std::path::PathBuf;

use arrow::util::pretty::print_batches;
use llkv_csv::{CsvReadOptions, CsvResult, open_csv_reader};

fn main() -> CsvResult<()> {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("Usage: stream_csv <path-to-csv>");
            std::process::exit(2);
        }
    };

    let options = CsvReadOptions::default();
    let (schema, reader) = open_csv_reader(path.as_path(), &options)?;

    println!("Inferred schema:\n{schema:#?}");

    for maybe_batch in reader {
        let batch = maybe_batch?;
        print_batches(&[batch])?;
    }

    Ok(())
}
