use std::io::{self, Read, Write, IsTerminal};
use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;
use llkv_runtime::RuntimeStatementResult;

fn print_banner() {
    // Use Cargo package metadata baked into the binary at compile time
    const NAME: &str = env!("CARGO_PKG_NAME");
    const VER: &str = env!("CARGO_PKG_VERSION");
    println!("{} version {}", NAME, VER);
    println!("Enter \".help\" for usage hints.");
    println!("Connected to a transient in-memory database.");
    println!("Use \".open FILENAME\" to reopen on a persistent database.");
}

fn print_help() {
    println!(".help           Show this message");
    println!(".open FILE      Open persistent database file");
    println!(".exit/.quit     Exit the REPL");
    println!("Any other line is echoed as SQL (no execution in this stub)");
}

fn repl() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut line = String::new();

    loop {
        line.clear();
        print!("llkv> ");
        stdout.flush()?;
        if stdin.read_line(&mut line)? == 0 {
            // EOF
            println!("");
            break;
        }
        let input = line.trim_end().trim();
        if input.is_empty() {
            continue;
        }
        if input.starts_with('.') {
            let mut parts = input.split_whitespace();
            let cmd = parts.next().unwrap_or("");
            match cmd {
                ".help" => print_help(),
                ".open" => {
                    if let Some(fname) = parts.next() {
                        println!("Reopening on persistent database: {}", fname);
                    } else {
                        println!(".open requires a filename");
                    }
                }
                ".exit" | ".quit" => break,
                _ => println!("Unknown command: {}", cmd),
            }
        } else {
            // Echo SQL - placeholder for future execution
            println!("SQL> {}", input);
        }
    }

    Ok(())
}

fn process_stream<R: std::io::Read>(reader: R) -> io::Result<()> {
    let mut buf = String::new();
    let mut rdr = std::io::BufReader::new(reader);
    rdr.read_to_string(&mut buf)?;
    // Create an in-memory SQL engine for this session (mirrors examples/simple_query.rs)
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    let sql = buf.trim();
    if sql.is_empty() {
        return Ok(());
    }

    match engine.execute(sql) {
        Ok(results) => {
            for res in results {
                match res {
                    RuntimeStatementResult::Select { execution, .. } => {
                        match execution.collect() {
                            Ok(batches) => {
                                if batches.is_empty() {
                                    println!("No batches returned");
                                } else {
                                    match pretty_format_batches(&batches) {
                                        Ok(s) => println!("{}", s),
                                        Err(e) => eprintln!("Query executed but failed to format batches: {:?}", e),
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to collect SELECT results: {:?}", e);
                            }
                        }
                    }
                    other => {
                        println!("OK: {:?}", other);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Execution failed: {:?}", e);
        }
    }
    Ok(())
}

fn main() {
    print_banner();

    // Use std's IsTerminal to detect whether stdin is coming from a TTY.
    // If it's not a terminal, treat stdin as a piped script and process it
    // non-interactively. This mirrors `sqlite3 DATABASE < file.sql` behavior.
    if !std::io::stdin().is_terminal() {
        if let Err(e) = process_stream(std::io::stdin()) {
            eprintln!("Error processing stdin: {}", e);
            std::process::exit(1);
        }
        return;
    }

    if let Err(e) = repl() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
