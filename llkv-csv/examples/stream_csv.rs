use std::env;
use std::path::PathBuf;

use arrow::util::pretty::print_batches;
use llkv_csv::{CsvReadOptions, CsvReader, CsvResult};

#[allow(clippy::print_stdout, clippy::print_stderr)]
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
    let reader = CsvReader::with_options(options.clone());
    let session = reader.open(path.as_path())?;
    let schema = session.schema();

    println!("Inferred schema:\n{schema:#?}");

    for maybe_batch in session {
        let batch = maybe_batch?;
        print_batches(&[batch])?;
    }

    Ok(())
}
