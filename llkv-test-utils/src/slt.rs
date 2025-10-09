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

    let expanded_text = expanded_lines.join("\n");
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
            map_temp_error_message(&format!("{}", e), &tmp, &expanded_lines, &mapping, path);
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
/// The `factory` closure should return a future that constructs a new DB
/// instance for each trial. This keeps the harness engine-agnostic so
/// different crates can provide their own engine adapters.
#[cfg(feature = "harness")]
pub fn run_slt_harness<F, Fut, D, E>(slt_dir: &str, factory: F)
where
    F: Fn() -> Fut + Send + Sync + 'static + Clone,
    Fut: std::future::Future<Output = Result<D, E>> + Send + 'static,
    D: AsyncDB<Error = llkv_result::Error, ColumnType = DefaultColumnType>
        + Send
        + 'static,
    E: std::fmt::Debug + Send + 'static,
{
    use libtest_mimic::{Arguments, Trial};

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
                } else if let Some(ext) = p.extension() && ext == "slt" {
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
        let factory_clone = factory.clone();
        trials.push(Trial::test(name, move || {
            let p = path_clone.clone();
            let fac = factory_clone.clone();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio rt");
            let res: Result<(), llkv_result::Error> = rt.block_on(async move {
                run_slt_file_with_factory(&p, fac).await
            });
            match res {
                Ok(()) => Ok(()),
                Err(e) => panic!("slt runner error: {}", e),
            }
        }));
    }

    let args = Arguments::from_args();
    let _ = libtest_mimic::run(&args, trials);
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
