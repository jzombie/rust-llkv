use std::io::{self, Write};

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

fn main() {
    print_banner();
    if let Err(e) = repl() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
