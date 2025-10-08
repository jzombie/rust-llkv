// TODO: Move to `llkv-test-utils`

use libtest_mimic::{Arguments, Trial};
use std::path::PathBuf;

// Discover all .slt files under tests/slt
fn find_slt_files(dir: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let base = std::path::Path::new(dir);
    if !base.exists() {
        return out;
    }
    let mut stack = vec![base.to_path_buf()];
    while let Some(p) = stack.pop() {
        if p.is_dir() {
            if let Ok(read) = std::fs::read_dir(&p) {
                for entry in read.flatten() {
                    stack.push(entry.path());
                }
            }
        } else if let Some(ext) = p.extension() {
            if ext == "slt" {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

// Expand `loop var start count` directives and run the sqllogictest Runner on the expanded script.
// This expansion also tracks the mapping from each expanded line back to the
// original source line number (1-based) so failures in the temporary file
// can be mapped back to the original .slt file and line.
fn expand_loops_with_mapping(
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
            // Recursively expand inner region, passing the correct base index so
            // returned mapping contains global original line numbers.
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
            // original line number (1-based) is base_index + i + 1
            out_map.push(base_index + i + 1);
            i += 1;
        }
    }
    Ok((out_lines, out_map))
}

// Helper to replace occurrences of the temporary file path + line in an
// error message with the original file path and mapped line number. Also
// append the original source line text for convenience when available.
// Returns (mapped_message, optional original source line number). To
// tolerate off-by-one differences between the runner's reported line and
// our mapping, we check nearby expanded indices and validate against the
// expanded_lines content.
fn map_temp_error_message(
    err_msg: &str,
    tmp_path: &std::path::Path,
    expanded_lines: &[String],
    mapping: &[usize],
    orig_path: &std::path::Path,
) -> (String, Option<usize>) {
    let tmp_str = tmp_path.to_string_lossy().to_string();
    let mut out = err_msg.to_string();
    if let Some(pos) = out.find(&tmp_str) {
        // find colon after path
        let after = &out[pos + tmp_str.len()..];
        if after.starts_with(':') {
            // parse digits
            let mut digits = String::new();
            for ch in after[1..].chars() {
                if ch.is_ascii_digit() {
                    digits.push(ch);
                } else {
                    break;
                }
            }
            if let Ok(expanded_line) = digits.parse::<usize>() {
                // Try a few candidate indices to tolerate off-by-one. Prefer
                // a +1 offset first so a runner that reports a line one
                // less than our mapping will be corrected.
                let candidates: [isize; 3] = [1, 0, -1];
                for &off in &candidates {
                    let idx = (expanded_line as isize - 1) + off;
                    if idx >= 0 && (idx as usize) < mapping.len() {
                        let idx_us = idx as usize;
                        // Validate that the expanded line text exists and is
                        // non-empty (helps avoid matching on blank lines).
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

async fn run_single_slt(path: &std::path::Path) -> Result<(), llkv_result::Error> {
    use arrow::array::Array as ArrowArray;
    use llkv_sql::{SqlEngine, StatementResult};
    use llkv_storage::pager::MemPager;
    use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType, Runner};
    use std::sync::Arc;

    struct EngineHarness {
        engine: SqlEngine<MemPager>,
    }
    impl EngineHarness {
        pub fn new() -> Self {
            let pager = Arc::new(MemPager::default());
            Self {
                engine: SqlEngine::new(pager),
            }
        }
    }

    #[async_trait::async_trait]
    impl AsyncDB for EngineHarness {
        type Error = llkv_result::Error;
        type ColumnType = DefaultColumnType;

        async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
            match self.engine.execute(sql) {
                Ok(mut results) => {
                    if results.is_empty() {
                        return Ok(DBOutput::StatementComplete(0));
                    }
                    let result = results.remove(0);
                    match result {
                        StatementResult::Select { execution, .. } => {
                            let batches = execution.collect()?;
                            let mut rows: Vec<Vec<String>> = Vec::new();
                            for batch in &batches {
                                for row_idx in 0..batch.num_rows() {
                                    let mut row: Vec<String> = Vec::new();
                                    for col in 0..batch.num_columns() {
                                        let array = batch.column(col);
                                        let val = match array.data_type() {
                                            arrow::datatypes::DataType::Int64 => {
                                                let a = array
                                                    .as_any()
                                                    .downcast_ref::<arrow::array::Int64Array>()
                                                    .unwrap();
                                                if a.is_null(row_idx) {
                                                    "NULL".to_string()
                                                } else {
                                                    a.value(row_idx).to_string()
                                                }
                                            }
                                            arrow::datatypes::DataType::UInt64 => {
                                                let a = array
                                                    .as_any()
                                                    .downcast_ref::<arrow::array::UInt64Array>()
                                                    .unwrap();
                                                if a.is_null(row_idx) {
                                                    "NULL".to_string()
                                                } else {
                                                    a.value(row_idx).to_string()
                                                }
                                            }
                                            arrow::datatypes::DataType::Float64 => {
                                                let a = array
                                                    .as_any()
                                                    .downcast_ref::<arrow::array::Float64Array>()
                                                    .unwrap();
                                                if a.is_null(row_idx) {
                                                    "NULL".to_string()
                                                } else {
                                                    a.value(row_idx).to_string()
                                                }
                                            }
                                            arrow::datatypes::DataType::Utf8 => {
                                                let a = array
                                                    .as_any()
                                                    .downcast_ref::<arrow::array::StringArray>()
                                                    .unwrap();
                                                if a.is_null(row_idx) {
                                                    "NULL".to_string()
                                                } else {
                                                    a.value(row_idx).to_string()
                                                }
                                            }
                                            _ => "".to_string(),
                                        };
                                        row.push(val);
                                    }
                                    rows.push(row);
                                }
                            }

                            let types = if let Some(first) = batches.first() {
                                (0..first.num_columns())
                                    .map(|col| match first.column(col).data_type() {
                                        arrow::datatypes::DataType::Int64
                                        | arrow::datatypes::DataType::UInt64 => {
                                            DefaultColumnType::Integer
                                        }
                                        arrow::datatypes::DataType::Float64 => {
                                            DefaultColumnType::FloatingPoint
                                        }
                                        arrow::datatypes::DataType::Utf8 => DefaultColumnType::Text,
                                        _ => DefaultColumnType::Any,
                                    })
                                    .collect()
                            } else {
                                vec![]
                            };

                            Ok(DBOutput::Rows { types, rows })
                        }
                        StatementResult::Insert { rows_inserted, .. } => {
                            Ok(DBOutput::StatementComplete(rows_inserted as u64))
                        }
                        StatementResult::Update { rows_updated, .. } => {
                            Ok(DBOutput::StatementComplete(rows_updated as u64))
                        }
                        StatementResult::CreateTable { .. } => Ok(DBOutput::StatementComplete(0)),
                        StatementResult::Transaction { .. } => Ok(DBOutput::StatementComplete(0)),
                    }
                }
                Err(e) => Err(e),
            }
        }

        async fn shutdown(&mut self) {}
    }

    let text = std::fs::read_to_string(path)
        .map_err(|e| llkv_result::Error::Internal(format!("failed to read slt file: {}", e)))?;
    let raw_lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    let (expanded_lines, expanded_map) = expand_loops_with_mapping(&raw_lines, 0)?;

    let expanded_text = expanded_lines.join("\n");
    // Use a NamedTempFile to avoid collisions between concurrently running
    // tests and to ensure the file exists for the duration of the runner.
    let mut named = tempfile::NamedTempFile::new().map_err(|e| {
        llkv_result::Error::Internal(format!("failed to create temp slt file: {}", e))
    })?;
    use std::io::Write as _;
    named.write_all(expanded_text.as_bytes()).map_err(|e| {
        llkv_result::Error::Internal(format!("failed to write temp slt file: {}", e))
    })?;
    let tmp = named.path().to_path_buf();

    let mut runner = Runner::new(|| async { Ok(EngineHarness::new()) });
    if let Err(e) = runner.run_file_async(&tmp).await {
        // Map any reference to the temporary file back to the original
        // .slt path and line number for clearer diagnostics. If we can find
        // the original source line, print it to stderr as an informational
        // message (separate from the runner's output) so we don't alter
        // sqllogictest diffs.
        let (mapped, opt_orig_line) = map_temp_error_message(
            &format!("{}", e),
            &tmp,
            &expanded_lines,
            &expanded_map,
            path,
        );
        if let Some(orig_line) = opt_orig_line {
            if let Ok(text) = std::fs::read_to_string(path) {
                if let Some(line) = text.lines().nth(orig_line - 1) {
                    eprintln!(
                        "[llkv-slt] original {}:{}: {}",
                        path.display(),
                        orig_line,
                        line.trim()
                    );
                }
            }
        }
        // NamedTempFile will remove the file on drop; drop explicitly.
        drop(named);
        return Err(llkv_result::Error::Internal(format!(
            "slt runner failed: {}",
            mapped
        )));
    }

    // keep named in scope until here so file exists while runner reads it
    drop(named);
    Ok(())
}

#[test]
fn slt_harness() {
    let files = find_slt_files("tests/slt");
    let mut trials: Vec<Trial> = Vec::new();
    for f in files {
        let name = f
            .strip_prefix("tests")
            .unwrap_or(&f)
            .to_string_lossy()
            .trim_start_matches('/')
            .to_string();
        let path_clone = f.clone();
        trials.push(Trial::test(name, move || {
            let p = path_clone.clone();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio rt");
            let res: Result<(), llkv_result::Error> =
                rt.block_on(async move { run_single_slt(&p).await });
            // If the slt runner returns an error, panic with the error message so
            // libtest_mimic prints the failing line and context. Returning a
            // libtest_mimic::Failed doesn't accept a message in this version of
            // the crate, so we use panic to surface the text.
            match res {
                Ok(()) => Ok(()),
                Err(e) => panic!("slt runner error: {}", e),
            }
        }));
    }

    let args = Arguments::from_args();
    let _ = libtest_mimic::run(&args, trials);
}

#[cfg(test)]
mod unit_tests {
    use super::{expand_loops_with_mapping, map_temp_error_message};
    use std::path::Path;

    #[test]
    fn test_expand_loops_with_mapping_simple() {
        let lines = vec![
            "create table t(a int);".to_string(),
            "loop i 1 3".to_string(),
            "insert into t values($i);".to_string(),
            "endloop".to_string(),
        ];
        let (expanded, mapping) = expand_loops_with_mapping(&lines, 0).expect("expand");
        assert_eq!(expanded.len(), 4);
        assert_eq!(mapping.len(), 4);
        // mapping: line1 -> original 1, expanded lines from loop map to original 3
        assert_eq!(mapping[0], 1);
        assert_eq!(mapping[1], 3);
        assert_eq!(mapping[2], 3);
        assert_eq!(mapping[3], 3);
        assert_eq!(expanded[1], "insert into t values(1);".to_string());
        assert_eq!(expanded[2], "insert into t values(2);".to_string());
        assert_eq!(expanded[3], "insert into t values(3);".to_string());
    }

    #[test]
    fn test_map_temp_error_message_replaces_tmp_path() {
        let tmp = Path::new("/tmp/llkv_slt_123.slt");
        let orig = Path::new("tests/slt/example.slt");
        let _raw_lines = vec![
            "create table t(a int);".to_string(),
            "insert into t values(1);".to_string(),
        ];
        // mapping maps expanded line 2 to original line 2
        let mapping = vec![1usize, 2usize];
        let err = format!("error at {}:2: something failed", tmp.display());
        let expanded_lines = vec![
            "create table t(a int);".to_string(),
            "insert into t values(1);".to_string(),
        ];
        let (mapped, opt_line) = map_temp_error_message(&err, tmp, &expanded_lines, &mapping, orig);
        assert!(mapped.contains("tests/slt/example.slt:2"));
        assert_eq!(opt_line, Some(2usize));
    }
}
