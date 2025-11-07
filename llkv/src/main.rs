use std::io::{self, IsTerminal, Read, Write};
use std::path::Path;
use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use llkv::{Error as LlkvError, RuntimeStatementResult, SqlEngine, storage::MemPager};
use llkv_slt_tester::{LlkvSltRunner, RuntimeKind};

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
    println!();
    println!("Command-line options:");
    println!("  --slt PATH            Run a single SLT file or directory");
    println!("  --slt-runtime MODE    Override runtime (current|multi). Defaults to current");
    println!("  --help                Show this usage information");
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
            println!();
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
                    RuntimeStatementResult::Select { execution, .. } => match execution.collect() {
                        Ok(batches) => {
                            if batches.is_empty() {
                                println!("No batches returned");
                            } else {
                                match pretty_format_batches(&batches) {
                                    Ok(s) => println!("{}", s),
                                    Err(e) => eprintln!(
                                        "Query executed but failed to format batches: {:?}",
                                        e
                                    ),
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to collect SELECT results: {:?}", e);
                        }
                    },
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
    let mut args = std::env::args().skip(1).peekable();
    let mut slt_target: Option<String> = None;
    let mut runtime_kind = RuntimeKind::CurrentThread;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--slt" => {
                slt_target = args.next();
                if slt_target.is_none() {
                    eprintln!("--slt requires a path to a file or directory");
                    std::process::exit(1);
                }
            }
            "--slt-runtime" => {
                if let Some(mode) = args.next() {
                    runtime_kind = match mode.as_str() {
                        "current" => RuntimeKind::CurrentThread,
                        "multi" | "multi-thread" => RuntimeKind::MultiThread,
                        other => {
                            eprintln!("Unknown runtime mode: {}", other);
                            std::process::exit(1);
                        }
                    };
                } else {
                    eprintln!("--slt-runtime requires a mode (current|multi)");
                    std::process::exit(1);
                }
            }
            "--help" | "-h" => {
                print_banner();
                print_help();
                return;
            }
            other => {
                // Preserve unhandled arguments for future database selection.
                // Currently unused but kept to avoid silently swallowing input.
                eprintln!("Unrecognized argument: {}", other);
                print_help();
                std::process::exit(1);
            }
        }
    }

    if let Some(target) = slt_target {
        if let Err(e) = run_slt_command(&target, runtime_kind) {
            eprintln!("SLT execution failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

    print_banner();

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

fn run_slt_command(path: &str, runtime_kind: RuntimeKind) -> Result<(), LlkvError> {
    let runner = LlkvSltRunner::in_memory().with_runtime_kind(runtime_kind);
    let metadata = std::fs::metadata(path)
        .map_err(|e| LlkvError::Internal(format!("failed to access {}: {}", path, e)))?;
    if metadata.is_dir() {
        runner.run_directory(path)
    } else if metadata.is_file() {
        runner.run_file(Path::new(path))
    } else {
        Err(LlkvError::Internal(format!(
            "path is neither file nor directory: {}",
            path
        )))
    }
}
