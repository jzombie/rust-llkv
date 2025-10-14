use libtest_mimic::{Arguments, Conclusion, Failed, Trial};
use llkv_result::Error;
use sqllogictest::{AsyncDB, DefaultColumnType, Runner};
use std::path::Path;

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
    let raw_lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    let (expanded_lines, mapping) = expand_loops_with_mapping(&raw_lines, 0)?;
    let (expanded_lines, mapping) = {
        let mut filtered_lines = Vec::with_capacity(expanded_lines.len());
        let mut filtered_mapping = Vec::with_capacity(mapping.len());
        for (line, orig_line) in expanded_lines.into_iter().zip(mapping.into_iter()) {
            if line.trim_start().starts_with("load ") {
                tracing::warn!(
                    "Ignoring unsupported SLT directive `load`: {}:{} -> {}",
                    path.display(),
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
    if let Err(e) = runner.run_file_async(&tmp).await {
        let (mapped, opt_orig_line) =
            map_temp_error_message(&format!("{}", e), &tmp, &normalized_lines, &mapping, path);
        if let Some(orig_line) = opt_orig_line
            && let Ok(text) = std::fs::read_to_string(path)
            && let Some(line) = text.lines().nth(orig_line - 1)
        {
            eprintln!(
                "[llkv-slt] original {}:{}: {}",
                path.display(),
                orig_line,
                line.trim()
            );
        }
        drop(named);
        return Err(Error::Internal(format!("slt runner failed: {}", mapped)));
    }

    drop(named);
    Ok(())
}

/// Discover `.slt` files under the given directory and run them as
/// libtest_mimic trials using the provided AsyncDB factory constructor.
///
/// The `factory_factory` closure is called once per test file and should return
/// a factory closure that creates DB instances. This allows each test file to
/// have isolated state while enabling multiple connections within a test to
/// share state. This keeps the harness engine-agnostic so different crates
/// can provide their own engine adapters.
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

/// Same as [`run_slt_harness`], but accepts pre-parsed [`Arguments`] so callers
/// can control CLI parsing (e.g. custom binaries).
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
    // Discover files
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
                    && ext == "slt"
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
        trials.push(Trial::test(name, move || {
            let p = path_clone.clone();
            // Call the factory_factory to get a fresh factory for this test file
            let fac = factory_factory_clone();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| Failed::from(format!("failed to build tokio runtime: {e}")))?;
            let res: Result<(), Error> =
                rt.block_on(async move { run_slt_file_with_factory(&p, fac).await });
            res.map_err(|e| Failed::from(format!("slt runner error: {e}")))
        }));
    }

    libtest_mimic::run(&args, trials)
}

/// Expand `loop var start count` directives, returning the expanded lines and
/// a mapping from expanded line index to the original 1-based source line.
pub fn expand_loops_with_mapping(
    lines: &[String],
    base_index: usize,
) -> Result<(Vec<String>, Vec<usize>), Error> {
    let mut out_lines: Vec<String> = Vec::new();
    let mut out_map: Vec<usize> = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i].trim_start().to_string();
        if line.starts_with("loop ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(Error::Internal(format!(
                    "malformed loop directive: {}",
                    line
                )));
            }
            let var = parts[1];
            let start: i64 = parts[2]
                .parse()
                .map_err(|e| Error::Internal(format!("invalid loop start: {}", e)))?;
            let count: i64 = parts[3]
                .parse()
                .map_err(|e| Error::Internal(format!("invalid loop count: {}", e)))?;

            let mut j = i + 1;
            while j < lines.len() && lines[j].trim_start() != "endloop" {
                j += 1;
            }
            if j >= lines.len() {
                return Err(Error::Internal("unterminated loop in slt".to_string()));
            }

            let inner = &lines[i + 1..j];
            let (expanded_inner, inner_map) = expand_loops_with_mapping(inner, base_index + i + 1)?;

            for k in 0..count {
                let val = (start + k).to_string();
                for (s, &orig_line) in expanded_inner.iter().zip(inner_map.iter()) {
                    let substituted = s.replace(&format!("${}", var), &val);
                    out_lines.push(substituted);
                    out_map.push(orig_line);
                }
            }

            i = j + 1;
        } else {
            out_lines.push(lines[i].clone());
            out_map.push(base_index + i + 1);
            i += 1;
        }
    }
    Ok((out_lines, out_map))
}

/// Convert sqllogictest inline connection syntax (e.g. `statement ok con1`)
/// into explicit `connection` records so the upstream parser can understand them.
/// Also ensures proper termination of statement error blocks by adding a blank line
/// after ---- when there's no expected error pattern.
#[allow(clippy::type_complexity)] // TODO: Refactor type complexity
fn normalize_inline_connections(
    lines: Vec<String>,
    mapping: Vec<usize>,
) -> (Vec<String>, Vec<usize>) {
    fn collect_statement_error_block(
        lines: &[String],
        mapping: &[usize],
        start: usize,
    ) -> (
        Vec<(String, usize)>,
        Option<String>,
        Vec<(String, usize)>,
        bool,
        usize,
    ) {
        let mut sql_lines = Vec::new();
        let mut message_lines = Vec::new();
        let mut regex_pattern = None;
        let mut idx = start;
        let mut saw_separator = false;

        while idx < lines.len() {
            let line = &lines[idx];
            let trimmed = line.trim_start();
            if trimmed == "----" {
                saw_separator = true;
                idx += 1;
                break;
            }
            sql_lines.push((line.clone(), mapping[idx]));
            idx += 1;
        }

        if saw_separator {
            while idx < lines.len() {
                let line = &lines[idx];
                let trimmed_full = line.trim();
                if trimmed_full.is_empty() {
                    idx += 1;
                    break;
                }
                if let Some(pattern) = trimmed_full.strip_prefix("<REGEX>:") {
                    regex_pattern = Some(pattern.to_string());
                    idx += 1;
                    while idx < lines.len() && lines[idx].trim().is_empty() {
                        idx += 1;
                    }
                    message_lines.clear();
                    break;
                }
                message_lines.push((line.clone(), mapping[idx]));
                idx += 1;
            }
        }

        (sql_lines, regex_pattern, message_lines, saw_separator, idx)
    }

    fn is_connection_token(token: &str) -> bool {
        token
            .strip_prefix("con")
            .map(|suffix| !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()))
            .unwrap_or(false)
    }

    let mut out_lines = Vec::with_capacity(lines.len());
    let mut out_map = Vec::with_capacity(mapping.len());

    let mut i = 0usize;
    while i < lines.len() {
        let line = &lines[i];
        let orig = mapping[i];
        let trimmed = line.trim_start();

        // Handle connection syntax normalization
        if trimmed.starts_with("statement ") || trimmed.starts_with("query ") {
            let mut tokens: Vec<&str> = trimmed.split_whitespace().collect();
            if tokens.len() >= 3 && tokens.last().is_some_and(|last| is_connection_token(last)) {
                let conn = tokens.pop().unwrap();
                let indent_len = line.len() - trimmed.len();
                let indent = &line[..indent_len];

                out_lines.push(format!("{indent}connection {conn}"));
                out_map.push(orig);

                let normalized = format!("{indent}{}", tokens.join(" "));
                let normalized_trimmed = normalized.trim_start();
                if normalized_trimmed.starts_with("statement error") {
                    let (sql_lines, regex_pattern, message_lines, saw_separator, new_idx) =
                        collect_statement_error_block(&lines, &mapping, i + 1);
                    i = new_idx;

                    let has_regex = regex_pattern.is_some();
                    if let Some(pattern) = regex_pattern {
                        out_lines.push(format!("{indent}connection {conn}"));
                        out_map.push(orig);
                        out_lines.push(format!("{indent}statement error {}", pattern));
                        out_map.push(orig);
                    } else {
                        out_lines.push(normalized.clone());
                        out_map.push(orig);
                    }
                    for (sql_line, sql_map) in sql_lines {
                        out_lines.push(sql_line);
                        out_map.push(sql_map);
                    }
                    // Only output ---- when there are actual error message lines
                    if saw_separator && !has_regex && !message_lines.is_empty() {
                        out_lines.push(format!("{indent}----"));
                        out_map.push(orig);
                        for (msg_line, msg_map) in message_lines {
                            out_lines.push(msg_line);
                            out_map.push(msg_map);
                        }
                        // Add extra blank line after plain text error messages to prevent multiline interpretation
                        out_lines.push(String::new());
                        out_map.push(orig);
                    }
                    out_lines.push(String::new());
                    out_map.push(orig);
                    continue;
                } else {
                    // Not a statement error, just output the normalized line
                    out_lines.push(normalized);
                    out_map.push(orig);
                    i += 1;
                    continue;
                }
            }
        }

        // Check if this is a statement error (without inline connection) followed by ----
        if trimmed.starts_with("statement error") {
            let indent = &line[..line.len() - trimmed.len()];
            let (sql_lines, regex_pattern, message_lines, saw_separator, new_idx) =
                collect_statement_error_block(&lines, &mapping, i + 1);
            i = new_idx;

            let has_regex = regex_pattern.is_some();
            if let Some(pattern) = regex_pattern {
                out_lines.push(format!("{indent}statement error {}", pattern));
                out_map.push(orig);
            } else {
                out_lines.push(line.clone());
                out_map.push(orig);
            }
            for (sql_line, sql_map) in sql_lines {
                out_lines.push(sql_line);
                out_map.push(sql_map);
            }
            // Only output ---- when there are actual error message lines
            if saw_separator && !has_regex && !message_lines.is_empty() {
                out_lines.push(format!("{indent}----"));
                out_map.push(orig);
                for (msg_line, msg_map) in message_lines {
                    out_lines.push(msg_line);
                    out_map.push(msg_map);
                }
                // Add extra blank line after plain text error messages to prevent multiline interpretation
                out_lines.push(String::new());
                out_map.push(orig);
            }
            out_lines.push(String::new());
            out_map.push(orig);
            continue;
        }

        out_lines.push(line.clone());
        out_map.push(orig);
        i += 1;
    }

    (out_lines, out_map)
}

/// Map a temporary expanded-file error message back to the original file path
/// and line; returns (mapped_message, optional original line number).
pub fn map_temp_error_message(
    err_msg: &str,
    tmp_path: &Path,
    expanded_lines: &[String],
    mapping: &[usize],
    orig_path: &Path,
) -> (String, Option<usize>) {
    let tmp_str = tmp_path.to_string_lossy().to_string();
    let mut out = err_msg.to_string();
    if let Some(pos) = out.find(&tmp_str) {
        let after = &out[pos + tmp_str.len()..];
        if let Some(stripped) = after.strip_prefix(':') {
            let mut digits = String::new();
            for ch in stripped.chars() {
                if ch.is_ascii_digit() {
                    digits.push(ch);
                } else {
                    break;
                }
            }
            if let Ok(expanded_line) = digits.parse::<usize>() {
                let candidates: [isize; 3] = [1, 0, -1];
                for &off in &candidates {
                    let idx = (expanded_line as isize - 1) + off;
                    if idx >= 0 && (idx as usize) < mapping.len() {
                        let idx_us = idx as usize;
                        let expanded_text =
                            expanded_lines.get(idx_us).map(|s| s.trim()).unwrap_or("");
                        if expanded_text.is_empty() {
                            continue;
                        }
                        let orig_line = mapping[idx_us];
                        let replacement = format!("{}:{}", orig_path.display(), orig_line);
                        out = out.replacen(
                            &format!("{}:{}", tmp_str, expanded_line),
                            &replacement,
                            1,
                        );
                        return (out, Some(orig_line));
                    }
                }
            }
        }
    }
    (out, None)
}
