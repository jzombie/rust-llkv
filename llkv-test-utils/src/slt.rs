use sqllogictest::{AsyncDB, DefaultColumnType, Runner};
use std::path::Path;

/// Run a single slt file using the provided AsyncDB factory. The factory is
/// a closure that returns a future resolving to a new DB instance for the
/// runner. This mirrors sqllogictest's Runner::new signature and behavior.
pub async fn run_slt_file_with_factory<F, Fut, D, E>(
    path: &Path,
    factory: F,
) -> Result<(), llkv_result::Error>
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<D, E>> + Send,
    D: AsyncDB<Error = llkv_result::Error, ColumnType = DefaultColumnType> + Send + 'static,
    E: std::fmt::Debug,
{
    let text = std::fs::read_to_string(path)
        .map_err(|e| llkv_result::Error::Internal(format!("failed to read slt file: {}", e)))?;
    let raw_lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    let (expanded_lines, mapping) = expand_loops_with_mapping(&raw_lines, 0)?;
    let (normalized_lines, mapping) = normalize_inline_connections(expanded_lines, mapping);

    let expanded_text = normalized_lines.join("\n");
    let mut named = tempfile::NamedTempFile::new().map_err(|e| {
        llkv_result::Error::Internal(format!("failed to create temp slt file: {}", e))
    })?;
    use std::io::Write as _;
    named.write_all(expanded_text.as_bytes()).map_err(|e| {
        llkv_result::Error::Internal(format!("failed to write temp slt file: {}", e))
    })?;
    let tmp = named.path().to_path_buf();

    let mut runner = Runner::new(|| async {
        factory()
            .await
            .map_err(|e| llkv_result::Error::Internal(format!("factory error: {:?}", e)))
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
        return Err(llkv_result::Error::Internal(format!(
            "slt runner failed: {}",
            mapped
        )));
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
    D: AsyncDB<Error = llkv_result::Error, ColumnType = DefaultColumnType> + Send + 'static,
    E: std::fmt::Debug + Send + 'static,
{
    use sqllogictest::harness::{Arguments, Trial, run};

    // Discover files
    let files = {
        let mut out = Vec::new();
        let base = std::path::Path::new(slt_dir);
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

    let mut trials: Vec<Trial> = Vec::new();
    for f in files {
        let name = f
            .strip_prefix("tests")
            .unwrap_or(&f)
            .to_string_lossy()
            .trim_start_matches('/')
            .to_string();
        let path_clone = f.clone();
        let factory_factory_clone = factory_factory.clone();
        trials.push(Trial::test(name, move || {
            let p = path_clone.clone();
            // Call the factory_factory to get a fresh factory for this test file
            let fac = factory_factory_clone();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio rt");
            let res: Result<(), llkv_result::Error> =
                rt.block_on(async move { run_slt_file_with_factory(&p, fac).await });
            match res {
                Ok(()) => Ok(()),
                Err(e) => panic!("slt runner error: {}", e),
            }
        }));
    }

    let args = Arguments::from_args();
    let _ = run(&args, trials);
}

/// Expand `loop var start count` directives, returning the expanded lines and
/// a mapping from expanded line index to the original 1-based source line.
pub fn expand_loops_with_mapping(
    lines: &[String],
    base_index: usize,
) -> Result<(Vec<String>, Vec<usize>), llkv_result::Error> {
    let mut out_lines: Vec<String> = Vec::new();
    let mut out_map: Vec<usize> = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i].trim_start().to_string();
        if line.starts_with("loop ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(llkv_result::Error::Internal(format!(
                    "malformed loop directive: {}",
                    line
                )));
            }
            let var = parts[1];
            let start: i64 = parts[2]
                .parse()
                .map_err(|e| llkv_result::Error::Internal(format!("invalid loop start: {}", e)))?;
            let count: i64 = parts[3]
                .parse()
                .map_err(|e| llkv_result::Error::Internal(format!("invalid loop count: {}", e)))?;

            let mut j = i + 1;
            while j < lines.len() && lines[j].trim_start() != "endloop" {
                j += 1;
            }
            if j >= lines.len() {
                return Err(llkv_result::Error::Internal(
                    "unterminated loop in slt".to_string(),
                ));
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

/// Convert legacy sqllogictest inline connection syntax (e.g. `statement ok con1`)
/// into explicit `connection` records so the upstream parser can understand them.
/// Also ensures proper termination of statement error blocks by adding a blank line
/// after ---- when there's no expected error pattern.
fn normalize_inline_connections(
    lines: Vec<String>,
    mapping: Vec<usize>,
) -> (Vec<String>, Vec<usize>) {
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

                // Check if this is a statement error that needs special handling
                if normalized_trimmed.starts_with("statement error") {
                    out_lines.push(normalized.clone());
                    out_map.push(orig);
                    i += 1;

                    // Collect the SQL statement lines until we hit ----
                    let mut sql_lines = Vec::new();
                    while i < lines.len() {
                        let next_line = &lines[i];
                        let next_trimmed = next_line.trim_start();

                        if next_trimmed == "----" {
                            // Found the delimiter - now check what follows
                            i += 1;

                            let has_expected_pattern = if i < lines.len() {
                                let following = lines[i].trim();
                                !following.is_empty() && following.starts_with("<REGEX>:")
                            } else {
                                false
                            };

                            if has_expected_pattern {
                                // OMIT the ---- and output the expected pattern line directly
                                for sql_line in sql_lines {
                                    out_lines.push(sql_line);
                                    out_map.push(orig);
                                }
                                // Output the expected pattern line WITHOUT the ---- delimiter
                                out_lines.push(lines[i].clone());
                                out_map.push(mapping[i]);
                                i += 1; // Move past the expected pattern line
                            } else {
                                // No expected pattern - omit the ---- delimiter entirely
                                for sql_line in sql_lines {
                                    out_lines.push(sql_line);
                                    out_map.push(orig);
                                }
                                // Add a blank line to ensure proper termination
                                out_lines.push(String::new());
                                out_map.push(mapping[i - 1]);
                            }
                            break;
                        } else {
                            sql_lines.push(next_line.clone());
                            i += 1;
                        }
                    }
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
            out_lines.push(line.clone());
            out_map.push(orig);
            i += 1;

            // Collect the SQL statement lines until we hit ----
            let mut sql_lines = Vec::new();
            while i < lines.len() {
                let next_line = &lines[i];
                let next_trimmed = next_line.trim_start();

                if next_trimmed == "----" {
                    // Found the delimiter - now check what follows
                    i += 1;

                    let has_expected_pattern = if i < lines.len() {
                        let following = lines[i].trim();
                        !following.is_empty() && following.starts_with("<REGEX>:")
                    } else {
                        false
                    };

                    if has_expected_pattern {
                        // OMIT the ---- and output the expected pattern line directly
                        for sql_line in sql_lines {
                            out_lines.push(sql_line);
                            out_map.push(orig); // Use same orig for SQL lines
                        }
                        // Output the expected pattern line WITHOUT the ---- delimiter
                        out_lines.push(lines[i].clone());
                        out_map.push(mapping[i]);
                        i += 1; // Move past the expected pattern line
                    // The blank line terminating the error pattern should already be in the input
                    } else {
                        // No expected pattern - omit the ---- delimiter entirely
                        for sql_line in sql_lines {
                            out_lines.push(sql_line);
                            out_map.push(orig); // Use same orig for SQL lines
                        }
                        // Add a blank line to ensure proper termination
                        out_lines.push(String::new());
                        out_map.push(mapping[i - 1]);
                    }
                    break;
                } else {
                    // Part of the SQL statement
                    sql_lines.push(next_line.clone());
                    i += 1;
                }
            }
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
