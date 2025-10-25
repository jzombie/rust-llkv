use std::path::{Path, PathBuf};

use crate::RuntimeKind;
use crate::engine::HarnessFactory;
use crate::parser::{
    expand_loops_with_mapping, filter_conditional_blocks, map_temp_error_message,
    normalize_inline_connections,
};
use libtest_mimic::{Arguments, Conclusion, Failed, Trial};
use llkv_result::Error;
use sqllogictest::{AsyncDB, DefaultColumnType, Runner};

/// Path where the last failed SLT test content is persisted for debugging.
///
/// When a test fails, the normalized/processed SLT content is saved to this location
/// so developers can inspect the actual test content that caused the failure.
const LAST_FAILED_SLT_PATH: &str = "target/last_failed_slt.tmp";

/// Database engine identifiers for conditional test filtering.
///
/// LLKV is treated as compatible with these engines for the purposes of conditional
/// directives in SLT test files. This means:
/// - Tests marked `onlyif sqlite` or `onlyif duckdb` will be included
/// - Tests marked `skipif sqlite` or `skipif duckdb` will be excluded
/// - Tests marked `onlyif mysql/postgresql/etc` will be excluded
/// - Tests marked `skipif mysql/postgresql/etc` will be included
const SLT_ENGINE_COMPAT: &[&str] = &["sqlite", "duckdb"];

/// Generate a clickable file link for VS Code terminal.
///
/// Creates a vscode://file URI with proper URL encoding for special characters.
fn make_vscode_file_link(path: &str, line: Option<usize>) -> String {
    // URL encode the path - properly handle all special characters
    let encoded = path
        .chars()
        .map(|c| match c {
            ' ' => "%20".to_string(),
            '!' => "%21".to_string(),
            '"' => "%22".to_string(),
            '#' => "%23".to_string(),
            '$' => "%24".to_string(),
            '%' => "%25".to_string(),
            '&' => "%26".to_string(),
            '\'' => "%27".to_string(),
            '(' => "%28".to_string(),
            ')' => "%29".to_string(),
            '*' => "%2A".to_string(),
            '+' => "%2B".to_string(),
            ',' => "%2C".to_string(),
            ';' => "%3B".to_string(),
            '=' => "%3D".to_string(),
            '?' => "%3F".to_string(),
            '@' => "%40".to_string(),
            '[' => "%5B".to_string(),
            ']' => "%5D".to_string(),
            c => c.to_string(),
        })
        .collect::<String>();

    if let Some(line_num) = line {
        format!("vscode://file/{}:{}", encoded, line_num)
    } else {
        format!("vscode://file/{}", encoded)
    }
}

/// Generate a shell-escaped path suitable for command-line use.
///
/// This escapes special characters so the path can be used directly in shell commands.
fn make_shell_escaped_path(path: &str) -> String {
    // For POSIX shells (bash, zsh, etc.), escape special characters
    path.chars()
        .map(|c| match c {
            ' ' | '\t' | '\n' | '|' | '&' | ';' | '(' | ')' | '<' | '>' | '"' | '\'' | '\\'
            | '*' | '?' | '[' | ']' | '{' | '}' | '$' | '`' => {
                format!("\\{}", c)
            }
            c => c.to_string(),
        })
        .collect()
}

/// Run SLT content that originated from the provided path using the supplied factory.
pub async fn run_slt_text_with_factory<F, Fut, D, E>(
    text: &str,
    origin: &Path,
    factory: F,
) -> Result<(), Error>
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<D, E>> + Send,
    D: AsyncDB<Error = Error, ColumnType = DefaultColumnType> + Send + 'static,
    E: std::fmt::Debug,
{
    let raw_lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    let (expanded_lines, mapping) = expand_loops_with_mapping(&raw_lines, 0)?;

    // Filter out conditional blocks based on database engine compatibility.
    // Apply filtering with all our compatible engines at once.
    let (expanded_lines, mapping) =
        filter_conditional_blocks(expanded_lines, mapping, SLT_ENGINE_COMPAT);

    let (expanded_lines, mapping) = {
        let mut filtered_lines = Vec::with_capacity(expanded_lines.len());
        let mut filtered_mapping = Vec::with_capacity(mapping.len());
        for (line, orig_line) in expanded_lines.into_iter().zip(mapping.into_iter()) {
            if line.trim_start().starts_with("load ") {
                tracing::warn!(
                    "Ignoring unsupported SLT directive `load`: {}:{} -> {}",
                    origin.display(),
                    orig_line,
                    line.trim()
                );
                continue;
            }
            filtered_lines.push(line);
            filtered_mapping.push(orig_line);
        }
        (filtered_lines, filtered_mapping)
    };
    let (normalized_lines, mapping) = normalize_inline_connections(expanded_lines, mapping);

    let expanded_text = normalized_lines.join("\n");
    let mut named = tempfile::NamedTempFile::new()
        .map_err(|e| Error::Internal(format!("failed to create temp slt file: {}", e)))?;
    use std::io::Write as _;
    named
        .write_all(expanded_text.as_bytes())
        .map_err(|e| Error::Internal(format!("failed to write temp slt file: {}", e)))?;
    if std::env::var("LLKV_DUMP_SLT").is_ok() {
        let dump_path = std::path::Path::new("target/normalized.slt");
        if let Some(parent) = dump_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::write(dump_path, &expanded_text) {
            tracing::warn!("failed to dump normalized slt file: {}", e);
        }
    }
    let tmp = named.path().to_path_buf();

    let mut runner = Runner::new(|| async {
        factory()
            .await
            .map_err(|e| Error::Internal(format!("factory error: {:?}", e)))
    });

    runner.with_hash_threshold(256);

    if let Err(e) = runner.run_file_async(&tmp).await {
        let (mapped, opt_orig_line) =
            map_temp_error_message(&format!("{}", e), &tmp, &normalized_lines, &mapping, origin);

        // Persist the temp file for debugging when there's an error
        let persist_path = std::path::Path::new(LAST_FAILED_SLT_PATH);
        if let Some(parent) = persist_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let persisted = if let Err(_) = std::fs::copy(&tmp, persist_path) {
            None
        } else {
            // Convert to absolute path for easier debugging
            std::fs::canonicalize(persist_path)
                .ok()
                .map(|p| p.display().to_string())
        };
        drop(named);

        if let Some(line_num) = opt_orig_line {
            // Print the line from the normalized/processed content with context
            if let Some(line) = normalized_lines.get(line_num.saturating_sub(1)) {
                eprintln!(
                    "[llkv-slt] Error at line {} in normalized content: {}",
                    line_num,
                    line.trim()
                );
            }

            // Also try to show the original line if available
            if let Some(line) = text.lines().nth(line_num.saturating_sub(1)) {
                eprintln!(
                    "[llkv-slt] Original source line {}: {}",
                    line_num,
                    line.trim()
                );
            }
        }

        if let Some(path) = &persisted {
            eprintln!("[llkv-slt] Normalized SLT saved to: {}", path);
            if let Some(line_num) = opt_orig_line {
                eprintln!("[llkv-slt] View context: head -n {} \"{}\" | tail -20", line_num.saturating_add(10), path);
            }
        }

        // Build enhanced error message showing both remote URL and local debug file
        let enhanced_msg = if let Some(debug_path) = persisted {
            if let Some(line_num) = opt_orig_line {
                let vscode_link = make_vscode_file_link(&debug_path, Some(line_num));
                let shell_path = make_shell_escaped_path(&debug_path);
                format!(
                    "slt runner failed: {}\n  at: {}:{}\n  debug: {}:{}\n  vscode: {}",
                    mapped,
                    origin.display(),
                    line_num,
                    shell_path,
                    line_num,
                    vscode_link
                )
            } else {
                let vscode_link = make_vscode_file_link(&debug_path, None);
                let shell_path = make_shell_escaped_path(&debug_path);
                format!(
                    "slt runner failed: {}\n  at: {}\n  debug: {}\n  vscode: {}",
                    mapped,
                    origin.display(),
                    shell_path,
                    vscode_link
                )
            }
        } else {
            format!("slt runner failed: {}", mapped)
        };

        return Err(Error::Internal(enhanced_msg));
    }

    drop(named);
    Ok(())
}

/// Run a single slt file using the provided AsyncDB factory. The factory is
/// a closure that returns a future resolving to a new DB instance for the
/// runner. This mirrors sqllogictest's Runner::new signature and behavior.
pub async fn run_slt_file_with_factory<F, Fut, D, E>(path: &Path, factory: F) -> Result<(), Error>
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<D, E>> + Send,
    D: AsyncDB<Error = Error, ColumnType = DefaultColumnType> + Send + 'static,
    E: std::fmt::Debug,
{
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Internal(format!("failed to read slt file: {}", e)))?;
    run_slt_text_with_factory(&text, path, factory).await
}

/// Helper to construct a Tokio runtime based on the requested mode.
fn build_runtime(kind: RuntimeKind) -> Result<tokio::runtime::Runtime, Error> {
    let mut builder = match kind {
        RuntimeKind::CurrentThread => tokio::runtime::Builder::new_current_thread(),
        RuntimeKind::MultiThread => tokio::runtime::Builder::new_multi_thread(),
    };
    builder
        .enable_all()
        .build()
        .map_err(|e| Error::Internal(format!("failed to build tokio runtime: {e}")))
}

/// Run a single file by internally creating a Tokio runtime.
pub fn run_slt_file_blocking_with_runtime(
    path: &Path,
    factory: HarnessFactory,
    kind: RuntimeKind,
) -> Result<(), Error> {
    let rt = build_runtime(kind)?;
    let result = rt.block_on(async move { run_slt_file_with_factory(path, factory).await });
    drop(rt);
    result
}

/// Run SLT text by creating a Tokio runtime matching the provided kind.
pub fn run_slt_text_blocking_with_runtime(
    origin: &Path,
    script: &str,
    factory: HarnessFactory,
    kind: RuntimeKind,
) -> Result<(), Error> {
    let origin_buf: PathBuf = origin.to_path_buf();
    let script_owned = script.to_owned();
    let rt = build_runtime(kind)?;
    let result = rt.block_on(async move {
        run_slt_text_with_factory(&script_owned, origin_buf.as_path(), factory).await
    });
    drop(rt);
    result
}

/// Convenience wrapper that defaults to a current-thread runtime for text inputs.
pub fn run_slt_text_blocking(
    origin: &Path,
    script: &str,
    factory: HarnessFactory,
) -> Result<(), Error> {
    run_slt_text_blocking_with_runtime(origin, script, factory, RuntimeKind::CurrentThread)
}

/// Convenience wrapper that defaults to a current-thread runtime.
pub fn run_slt_file_blocking(path: &Path, factory: HarnessFactory) -> Result<(), Error> {
    run_slt_file_blocking_with_runtime(path, factory, RuntimeKind::CurrentThread)
}

/// Discover `.slt` files and execute them as libtest trials, returning a conclusion.
pub fn run_slt_harness<FF, F, Fut, D, E>(slt_dir: &str, factory_factory: FF)
where
    FF: Fn() -> F + Send + Sync + 'static + Clone,
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<D, E>> + Send + 'static,
    D: AsyncDB<Error = Error, ColumnType = DefaultColumnType> + Send + 'static,
    E: std::fmt::Debug + Send + 'static,
{
    let args = Arguments::from_args();
    let conclusion = run_slt_harness_with_args(slt_dir, factory_factory, args);
    if conclusion.has_failed() {
        panic!(
            "SLT harness reported {} failed test(s)",
            conclusion.num_failed
        );
    }
}

pub fn run_slt_harness_with_args<FF, F, Fut, D, E>(
    slt_dir: &str,
    factory_factory: FF,
    args: Arguments,
) -> Conclusion
where
    FF: Fn() -> F + Send + Sync + 'static + Clone,
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<D, E>> + Send + 'static,
    D: AsyncDB<Error = Error, ColumnType = DefaultColumnType> + Send + 'static,
    E: std::fmt::Debug + Send + 'static,
{
    let base = std::path::Path::new(slt_dir);
    let files = {
        let mut out = Vec::new();
        if base.exists() {
            let mut stack = vec![base.to_path_buf()];
            while let Some(p) = stack.pop() {
                if p.is_dir() {
                    if let Ok(read) = std::fs::read_dir(&p) {
                        for entry in read.flatten() {
                            stack.push(entry.path());
                        }
                    }
                } else if let Some(ext) = p.extension()
                    && (ext == "slt" || ext == "slturl")
                {
                    out.push(p);
                }
            }
        }
        out.sort();
        out
    };

    let base_parent = base.parent();
    let mut trials: Vec<Trial> = Vec::new();
    for f in files {
        let name_path = base_parent
            .and_then(|parent| f.strip_prefix(parent).ok())
            .or_else(|| f.strip_prefix(base).ok())
            .unwrap_or(&f);
        let mut name = name_path.to_string_lossy().to_string();
        if std::path::MAIN_SEPARATOR != '/' {
            name = name.replace(std::path::MAIN_SEPARATOR, "/");
        }
        let name = name.trim_start_matches(&['/', '\\'][..]).to_string();
        let path_clone = f.clone();
        let factory_factory_clone = factory_factory.clone();

        // Check if this is a .slturl pointer file
        let is_url_pointer = f.extension().is_some_and(|ext| ext == "slturl");

        trials.push(Trial::test(name, move || {
            let p = path_clone.clone();
            let fac = factory_factory_clone();

            // Spawn thread with larger stack size (16MB) to handle deeply nested SQL expressions
            // Default thread stack is ~2MB which is insufficient for complex SLT test queries
            std::thread::Builder::new()
                // TODO: Make stack size configurable via env var or argument
                .stack_size(16 * 1024 * 1024)
                .spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| Failed::from(format!("failed to build tokio runtime: {e}")))?;

                    let res: Result<(), Error> = if is_url_pointer {
                        // Read the URL from the pointer file and fetch remotely
                        let url = std::fs::read_to_string(&p)
                            .map_err(|e| {
                                Error::Internal(format!("failed to read .slturl file: {e}"))
                            })?
                            .trim()
                            .to_string();

                        // Fetch and run the remote SLT file
                        let response = reqwest::blocking::get(&url).map_err(|e| {
                            Error::Internal(format!("failed to fetch SLT URL {url}: {e}"))
                        })?;
                        let script = response.text().map_err(|e| {
                            Error::Internal(format!(
                                "failed to read SLT response body for {url}: {e}"
                            ))
                        })?;

                        let origin = std::path::PathBuf::from(format!("url:{}", url));
                        rt.block_on(async move {
                            run_slt_text_with_factory(&script, origin.as_path(), fac).await
                        })
                    } else {
                        // Run local .slt file as before
                        rt.block_on(async move { run_slt_file_with_factory(&p, fac).await })
                    };

                    res.map_err(|e| Failed::from(format!("slt runner error: {e}")))
                })
                .map_err(|e| Failed::from(format!("failed to spawn test thread: {e}")))?
                .join()
                .map_err(|e| Failed::from(format!("test thread panicked: {e:?}")))?
        }));
    }

    libtest_mimic::run(&args, trials)
}

/// Run every `.slt` file located under `dir`, using the supplied factory factory and runtime kind.
pub fn run_slt_dir_blocking<F>(
    dir: &str,
    factory_factory: F,
    kind: RuntimeKind,
) -> Result<(), Error>
where
    F: Fn() -> HarnessFactory + Send + Sync + 'static,
{
    let base = Path::new(dir);
    if !base.exists() {
        return Err(Error::Internal(format!(
            "slt directory does not exist: {}",
            dir
        )));
    }

    for entry in walkdir::WalkDir::new(base)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "slt"))
    {
        let factory = factory_factory();
        run_slt_file_blocking_with_runtime(entry.path(), factory, kind)?;
    }

    Ok(())
}
