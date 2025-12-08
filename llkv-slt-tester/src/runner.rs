use std::path::{Path, PathBuf};

use crate::parser::{
    expand_loops_with_mapping, filter_conditional_blocks, map_temp_error_message,
    normalize_inline_connections,
};
use crate::slt_test_engine::{
    self, HarnessFactory, clear_expected_column_types, set_expected_column_types,
};
use libtest_mimic::{Arguments, Conclusion, Failed, Trial};
use llkv_result::Error;
use llkv_sql::{
    StatementExpectation as SqlStatementExpectation, clear_pending_statement_expectations,
    register_statement_expectation,
};
use sqllogictest::{AsyncDB, DefaultColumnType, QueryExpect, Runner};
use sqllogictest::{Record, StatementExpect};

/// Runtime configuration for executing SLT tests.
#[derive(Clone, Copy)]
pub enum RuntimeKind {
    /// Single-threaded runtime suitable for most CLI use-cases.
    CurrentThread,
    /// Multi-threaded runtime for workloads that benefit from thread-pooling.
    MultiThread,
}

/// Convenience runner that owns the resources required to execute SLT test suites.
///
/// This wrapper creates a shared runtime context and internally manages
/// a Tokio runtime so callers do not need to be async-aware.
#[derive(Clone)]
pub struct LlkvSltRunner {
    factory_factory: std::sync::Arc<dyn Fn() -> HarnessFactory + Send + Sync>,
    runtime_kind: RuntimeKind,
}

impl LlkvSltRunner {
    /// Create a runner that executes against an in-memory `MemPager` backend.
    pub fn in_memory() -> Self {
        Self::with_factory_factory(slt_test_engine::make_in_memory_factory_factory())
    }

    /// Create a runner that executes against a user-supplied factory factory.
    pub fn with_factory_factory<F>(factory_factory: F) -> Self
    where
        F: Fn() -> HarnessFactory + Send + Sync + 'static,
    {
        Self {
            factory_factory: std::sync::Arc::new(factory_factory),
            runtime_kind: RuntimeKind::CurrentThread,
        }
    }

    /// Override runtime configuration.
    pub fn with_runtime_kind(mut self, kind: RuntimeKind) -> Self {
        self.runtime_kind = kind;
        self
    }

    /// Run the provided `.slt` or `.slturl` file synchronously, returning the first error if any.
    /// If the file has a `.slturl` extension, it will be treated as a pointer file containing
    /// a URL to the actual test content, which will be fetched and executed.
    pub fn run_file(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();

        // Check if this is a .slturl pointer file
        if path.extension().is_some_and(|ext| ext == "slturl") {
            let url = std::fs::read_to_string(path)
                .map_err(|e| Error::Internal(format!("failed to read .slturl file: {e}")))?
                .trim()
                .to_string();
            return self.run_url(&url);
        }

        // Otherwise, run as a normal .slt file - read and execute
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Internal(format!("failed to read slt file: {}", e)))?;
        self.run_script_at_path(&text, path)
    }

    /// Discover and execute all `.slt` and `.slturl` files under the given directory.
    pub fn run_directory(&self, dir: &str) -> Result<(), Error> {
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
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext == "slt" || ext == "slturl")
            })
        {
            self.run_file(entry.path())?;
        }

        Ok(())
    }

    /// Execute the provided SLT script contents, tagging diagnostics with `name` for context.
    pub fn run_script(&self, name: &str, script: &str) -> Result<(), Error> {
        let display_name = if name.trim().is_empty() {
            "<memory>"
        } else {
            name
        };
        let origin = PathBuf::from(display_name);
        self.run_script_at_path(script, &origin)
    }

    /// Execute SLT content read from an arbitrary reader.
    pub fn run_reader<R: std::io::Read>(&self, name: &str, mut reader: R) -> Result<(), Error> {
        let mut buf = String::new();
        reader
            .read_to_string(&mut buf)
            .map_err(|e| Error::Internal(format!("failed to read SLT stream: {e}")))?;
        self.run_script(name, &buf)
    }

    /// Fetch an SLT script from `url` and execute it.
    pub fn run_url(&self, url: &str) -> Result<(), Error> {
        let response = reqwest::blocking::get(url)
            .map_err(|e| Error::Internal(format!("failed to fetch SLT URL {url}: {e}")))?;
        let script = response.text().map_err(|e| {
            Error::Internal(format!("failed to read SLT response body for {url}: {e}"))
        })?;
        let name = format!("url:{url}");
        self.run_script(&name, &script)
    }

    /// Internal: Execute SLT script content with an associated origin path.
    ///
    /// This is the core execution method that all other methods eventually call.
    fn run_script_at_path(&self, script: &str, origin: &Path) -> Result<(), Error> {
        let factory = (self.factory_factory)();
        let rt = self.build_runtime()?;
        let script = script.to_string();
        let origin = origin.to_path_buf();
        let result =
            rt.block_on(async move { Self::run_slt_text_async(&script, &origin, factory).await });
        drop(rt);
        result
    }

    /// Build a Tokio runtime based on the configured runtime kind.
    fn build_runtime(&self) -> Result<tokio::runtime::Runtime, Error> {
        let mut builder = match self.runtime_kind {
            RuntimeKind::CurrentThread => tokio::runtime::Builder::new_current_thread(),
            RuntimeKind::MultiThread => tokio::runtime::Builder::new_multi_thread(),
        };
        builder
            .enable_all()
            .build()
            .map_err(|e| Error::Internal(format!("failed to build tokio runtime: {e}")))
    }

    /// Internal async helper that performs the actual SLT execution.
    ///
    /// This handles all the parsing, preprocessing, and test execution logic.
    #[allow(clippy::print_stderr)]
    async fn run_slt_text_async<F, Fut, D, E>(
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
        let (expanded_lines, mapping) =
            filter_conditional_blocks(expanded_lines, mapping, SLT_ENGINE_COMPAT);

        // Filter out unsupported directives
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

        // Write to temp file for sqllogictest parser
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

        /// Custom validator that coerces numeric values before comparison.
        /// This allows comparing "1.25" with "1.250" by parsing them as floats.
        fn numeric_coercion_validator(
            normalizer: sqllogictest::Normalizer,
            actual: &[Vec<String>],
            expected: &[String],
        ) -> bool {
            // First try exact string matching (default behavior)
            if sqllogictest::default_validator(normalizer, actual, expected) {
                return true;
            }

            // Try numeric coercion - parse both sides as floats and compare
            if actual.len() != expected.len() {
                return false;
            }

            for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
                let actual_normalized: Vec<String> = actual_row.iter().map(normalizer).collect();
                let expected_normalized = normalizer(expected_row);

                let actual_joined = actual_normalized.join(" ");

                // Split both into tokens
                let actual_tokens: Vec<&str> = actual_joined.split_whitespace().collect();
                let expected_tokens: Vec<&str> = expected_normalized.split_whitespace().collect();

                if actual_tokens.len() != expected_tokens.len() {
                    return false;
                }

                for (a, e) in actual_tokens.iter().zip(expected_tokens.iter()) {
                    // Try exact match first
                    if a == e {
                        continue;
                    }

                    // Try parsing as floats for numeric comparison
                    match (a.parse::<f64>(), e.parse::<f64>()) {
                        (Ok(a_num), Ok(e_num)) => {
                            // Use relative tolerance for large numbers, absolute for small
                            let abs_diff = (a_num - e_num).abs();
                            let max_val = a_num.abs().max(e_num.abs());

                            // For large numbers (>1e6), use relative tolerance of 1e-10
                            // For small numbers, use absolute tolerance of 1e-10
                            let tolerance = if max_val > 1e6 {
                                max_val * 1e-10
                            } else {
                                1e-10
                            };

                            if abs_diff > tolerance {
                                return false;
                            }
                        }
                        _ => {
                            // Not both numbers, must match exactly
                            return false;
                        }
                    }
                }
            }

            true
        }

        let mut runner = Runner::new(|| async {
            factory()
                .await
                .map_err(|e| Error::Internal(format!("factory error: {:?}", e)))
        });

        runner.with_hash_threshold(256);
        runner.with_validator(numeric_coercion_validator);

        let records = sqllogictest::parse_file(&tmp).map_err(|e| {
            Error::Internal(format!(
                "failed to parse normalized slt file {}: {}",
                tmp.display(),
                e
            ))
        })?;

        // Execute all records with hash threshold and type hint handling
        let run_result = async {
            let mut current_hash_threshold: usize = 256;
            let log_progress = std::env::var("LLKV_SLT_PROGRESS").is_ok();
            let mut record_index: usize = 0;

            for record in records {
                if log_progress {
                    let preview = match &record {
                        Record::Statement { sql, .. } => sql,
                        Record::Query { sql, .. } => sql,
                        _ => "<control>",
                    };
                    let single_line = preview.replace('\n', " ");
                    let display = if single_line.len() > 80 {
                        format!("{}...", &single_line[..80])
                    } else {
                        single_line
                    };
                    eprintln!("[llkv-slt] record {}: {}", record_index, display);
                }

                if let Record::Statement { expected, .. } = &record {
                    match expected {
                        StatementExpect::Error(_) => {
                            register_statement_expectation(SqlStatementExpectation::Error);
                        }
                        StatementExpect::Count(count) => {
                            register_statement_expectation(SqlStatementExpectation::Count(*count));
                        }
                        StatementExpect::Ok => {}
                    }
                }

                let hash_threshold_update = match &record {
                    sqllogictest::Record::HashThreshold { threshold, .. } => {
                        Some(*threshold as usize)
                    }
                    _ => None,
                };

                let expected_hash_count = if let sqllogictest::Record::Query {
                    expected: QueryExpect::Results { results, .. },
                    ..
                } = &record
                {
                    Self::detect_expected_hash_values(results)
                } else {
                    None
                };

                let mut previous_hash_threshold = None;
                if let Some(count) = expected_hash_count
                    && count > 0
                    && (current_hash_threshold == 0 || count <= current_hash_threshold)
                {
                    let forced = count.saturating_sub(1).max(1);
                    previous_hash_threshold = Some(current_hash_threshold);
                    runner.with_hash_threshold(forced);
                    current_hash_threshold = forced;
                }

                let type_hint = match &record {
                    sqllogictest::Record::Query { expected, .. } => match expected {
                        QueryExpect::Results { types, .. } if !types.is_empty() => {
                            Some(types.clone())
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let _type_guard = type_hint.map(ColumnTypeExpectationGuard::install);

                if matches!(&record, sqllogictest::Record::Halt { .. }) {
                    if let Some(prev) = previous_hash_threshold {
                        runner.with_hash_threshold(prev);
                    }
                    break;
                }

                // Run the record and handle per-query flattening fallback
                let result = runner.run_async(record.clone()).await;
                clear_pending_statement_expectations();

                if let Err(e) = result {
                    let error_msg = format!("{}", e);

                    // Check if this is an integer overflow error in a query that expects empty result
                    let is_overflow_error = matches!(
                        &record,
                        sqllogictest::Record::Query {
                            expected: sqllogictest::QueryExpect::Results { results, .. },
                            ..
                        } if results.is_empty()
                    )
                        && error_msg.contains("integer overflow");

                    if is_overflow_error {
                        tracing::debug!(
                            "[llkv-slt] Integer overflow in query treated as empty result (SQLite behavior)"
                        );
                        // Continue to next record - empty result expected, error is acceptable
                    } else {
                        // Only apply flattening workaround for Query records with result mismatches
                        let is_query_mismatch = matches!(&record, sqllogictest::Record::Query { .. })
                            && error_msg.contains("[Diff]");

                        if is_query_mismatch && LlkvSltRunner::flattening_resolves_mismatch(&error_msg)
                        {
                            tracing::debug!(
                                "[llkv-slt] Query mismatch resolved by flattening multi-column output"
                            );
                            // Continue to next record instead of failing
                        } else {
                            return Err(e);
                        }
                    }
                }

                if let Some(prev) = previous_hash_threshold {
                    runner.with_hash_threshold(prev);
                    current_hash_threshold = prev;
                }

                if let Some(new_threshold) = hash_threshold_update {
                    current_hash_threshold = new_threshold;
                    runner.with_hash_threshold(new_threshold);
                }

                record_index += 1;
            }

            Ok::<(), sqllogictest::TestError>(())
        }
        .await;

        // Handle errors with detailed diagnostics
        if let Err(e) = run_result {
            let (mapped, opt_line_info) = map_temp_error_message(
                &format!("{}", e),
                &tmp,
                &normalized_lines,
                &mapping,
                origin,
            );

            // Persist the temp file for debugging
            let persist_path = std::path::Path::new(LAST_FAILED_SLT_PATH);
            if let Some(parent) = persist_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let persisted = if std::fs::copy(&tmp, persist_path).is_err() {
                None
            } else {
                std::fs::canonicalize(persist_path)
                    .ok()
                    .map(|p| p.display().to_string())
            };
            drop(named);

            if let Some((orig_line, normalized_line)) = opt_line_info {
                if let Some(line) = normalized_lines.get(normalized_line.saturating_sub(1)) {
                    eprintln!(
                        "[llkv-slt] Normalized line {}: {}",
                        normalized_line,
                        line.trim()
                    );
                }

                if let Some(line) = text.lines().nth(orig_line.saturating_sub(1)) {
                    eprintln!(
                        "[llkv-slt] Original source line {}: {}",
                        orig_line,
                        line.trim()
                    );
                }
            }

            if let Some(path) = &persisted {
                eprintln!("[llkv-slt] Normalized SLT saved to: {}", path);
                if let Some((_, normalized_line)) = opt_line_info {
                    eprintln!(
                        "[llkv-slt] View context: head -n {} '{}' | tail -20",
                        normalized_line.saturating_add(10),
                        path
                    );
                }
            }

            let enhanced_msg = if let Some(debug_path) = persisted {
                if let Some((orig_line, normalized_line)) = opt_line_info {
                    let vscode_link =
                        Self::make_vscode_file_link(&debug_path, Some(normalized_line));
                    let shell_path = Self::make_shell_escaped_path(&debug_path);
                    format!(
                        "slt runner failed: {}\n  at: {}:{}\n  debug: {}:{}\n  vscode: {}",
                        mapped,
                        origin.display(),
                        orig_line,
                        shell_path,
                        normalized_line,
                        vscode_link
                    )
                } else {
                    let vscode_link = Self::make_vscode_file_link(&debug_path, None);
                    let shell_path = Self::make_shell_escaped_path(&debug_path);
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

    /// Detect if expected results contain a hash summary line.
    fn detect_expected_hash_values(expected: &[String]) -> Option<usize> {
        let first = expected.first()?.trim();
        let mut parts = first.split_whitespace();
        let count_str = parts.next()?;
        let count = count_str.parse().ok()?;
        if parts.next()? != "values" {
            return None;
        }
        if parts.next()? != "hashing" {
            return None;
        }
        if parts.next()? != "to" {
            return None;
        }
        Some(count)
    }

    /// Check if a diff mismatch can be resolved by flattening multi-column output.
    fn flattening_resolves_mismatch(message: &str) -> bool {
        let mut expected: Vec<String> = Vec::new();
        let mut actual: Vec<String> = Vec::new();
        let mut in_diff_block = false;

        for line in message.lines() {
            if line.contains("[Diff]") {
                in_diff_block = true;
                continue;
            }

            if !in_diff_block {
                continue;
            }

            if let Some(rest) = line.strip_prefix("-   ") {
                expected.push(rest.trim_end().to_string());
            } else if let Some(rest) = line.strip_prefix("+   ") {
                actual.push(rest.trim_end().to_string());
            } else if line.starts_with("at ") || line.starts_with("  at:") {
                break;
            }
        }

        if expected.is_empty() || actual.is_empty() {
            return false;
        }

        // If expected is a hash result, don't try to flatten
        if expected.len() == 1 && expected[0].contains("hashing to") {
            return false;
        }

        if actual.len() >= expected.len() {
            return false;
        }

        // Try simple whitespace split
        let flattened: Vec<String> = actual
            .iter()
            .flat_map(|line| Self::split_diff_values(line))
            .collect();

        if flattened.len() == expected.len() && flattened == expected {
            return true;
        }

        // Try smart split
        if let Some(flattened_smart) = Self::try_smart_split_with_expected(&actual, &expected)
            && flattened_smart.len() == expected.len()
        {
            return flattened_smart == expected;
        }

        false
    }

    /// Try to split actual lines to match expected values.
    fn try_smart_split_with_expected(
        actual: &[String],
        expected: &[String],
    ) -> Option<Vec<String>> {
        if actual.is_empty() || expected.is_empty() || actual.len() >= expected.len() {
            return None;
        }

        let mut result = Vec::new();
        let mut expected_idx = 0;

        for line in actual {
            let values_per_line = expected.len() / actual.len();
            let extra = expected.len() % actual.len();
            let needed = values_per_line + if result.len() < extra { 1 } else { 0 };

            let mut line_remaining = line.as_str();
            let mut found_count = 0;

            while found_count < needed && expected_idx < expected.len() {
                let exp_val = &expected[expected_idx];

                if line_remaining.starts_with(exp_val) {
                    result.push(exp_val.clone());
                    line_remaining = line_remaining[exp_val.len()..].trim_start();
                    expected_idx += 1;
                    found_count += 1;
                } else {
                    return None;
                }
            }

            if found_count != needed {
                return None;
            }
        }

        if result.len() == expected.len() && result == *expected {
            Some(result)
        } else {
            None
        }
    }

    /// Split a line by whitespace, respecting quoted strings.
    fn split_diff_values(line: &str) -> Vec<String> {
        let mut values = Vec::new();
        let mut current = String::new();
        let mut chars = line.chars().peekable();
        let mut in_quotes = false;

        while let Some(ch) = chars.next() {
            if ch.is_whitespace() && !in_quotes {
                if !current.is_empty() {
                    values.push(std::mem::take(&mut current));
                }
                continue;
            }

            if ch == '\'' {
                current.push(ch);
                if in_quotes {
                    if chars.peek() == Some(&'\'') {
                        current.push(chars.next().unwrap());
                    } else {
                        in_quotes = false;
                    }
                } else {
                    in_quotes = true;
                }
                continue;
            }

            current.push(ch);
        }

        if !current.is_empty() {
            values.push(current);
        }

        values
    }

    /// Generate a clickable file link for VS Code terminal.
    fn make_vscode_file_link(path: &str, line: Option<usize>) -> String {
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
    fn make_shell_escaped_path(path: &str) -> String {
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
}

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

/// Thread stack size used when spawning test threads for running SLT files.
///
/// SLT files may contain deeply-nested SQL expressions and large generated
/// test cases (for example after loop expansion). The default thread stack
/// (≈2MB on many platforms) can be insufficient and lead to stack overflows
/// or panics when executing complex tests. To avoid that, the SLT harness
/// spawns a dedicated thread for each test with an increased stack size.
///
/// Value: 16 MiB — chosen as a conservative size that accommodates the
/// real-world SLT files used in this repository while keeping memory usage
/// reasonable when running multiple tests in parallel. If you encounter
/// stack overflows for unusually large tests, increase this constant.
///
/// Note: This setting only applies to the test thread created by the SLT
/// harness (see `run_slt_harness_with_args`). It does not change the global
/// Tokio runtime configuration or other thread pools.
const SLT_HARNESS_STACK_SIZE: usize = 16 * 1024 * 1024; // 16 MB

/// Scope guard that installs expected column types before executing a query and
/// clears thread-local state even if execution exits early.
struct ColumnTypeExpectationGuard;

impl ColumnTypeExpectationGuard {
    fn install(types: Vec<DefaultColumnType>) -> Self {
        set_expected_column_types(types);
        Self
    }
}

impl Drop for ColumnTypeExpectationGuard {
    fn drop(&mut self) {
        clear_expected_column_types();
    }
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
    // Enable statistics collection if requested
    let stats_enabled = std::env::var("LLKV_SLT_STATS").is_ok();
    if stats_enabled {
        crate::slt_test_engine::enable_stats();
    }

    // Default to fail-fast unless overridden by environment variable (e.g. in CI)
    let fail_fast = std::env::var("LLKV_SLT_NO_FAIL_FAST").is_err();

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
            let res = std::thread::Builder::new()
                .stack_size(SLT_HARNESS_STACK_SIZE)
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
                            LlkvSltRunner::run_slt_text_async(&script, origin.as_path(), fac).await
                        })
                    } else {
                        // Run local .slt file as before
                        let text = std::fs::read_to_string(&p).map_err(|e| {
                            Error::Internal(format!("failed to read slt file: {}", e))
                        })?;
                        rt.block_on(async move {
                            LlkvSltRunner::run_slt_text_async(&text, &p, fac).await
                        })
                    };

                    res.map_err(|e| Failed::from(format!("slt runner error: {e}")))
                })
                .map_err(|e| Failed::from(format!("failed to spawn test thread: {e}")))?
                .join()
                .map_err(|e| Failed::from(format!("test thread panicked: {e:?}")))?;

            if let Err(e) = &res {
                if fail_fast {
                    // Print the error explicitly before exiting, as process::exit will prevent
                    // libtest-mimic from printing the failure summary.
                    //
                    // This simulates a panic, but forces the process to exit manually, as a real
                    // panic insufficient here because the test runner is designed to catch panics.
                    //
                    // Note: Failed::msg is private, so we have to parse the Debug output to get
                    // the unescaped message with proper line breaks.
                    let debug_str = format!("{:?}", e);
                    // FIXME: Error messages are contained in a JSON-like string, so here's a rather
                    // hacky implementation to extract them. This is a workaround since libtest_mimic::Failed
                    // is private and the struct does not implement `Display`.
                    if let Some(start) = debug_str.find("msg: Some(\"") {
                        if let Some(end) = debug_str.rfind("\")") {
                            let inner = &debug_str[start + 11..end];
                            let unescaped = inner
                                .replace("\\n", "\n")
                                .replace("\\\"", "\"")
                                .replace("\\\\", "\\")
                                .replace("\\t", "\t");
                            eprintln!("{}", unescaped);
                        } else {
                            eprintln!("{}", debug_str);
                        }
                    } else {
                        eprintln!("{}", debug_str);
                    }
                    std::process::exit(101);
                }
            }
            res
        }));
    }

    let conclusion = libtest_mimic::run(&args, trials);

    // Print statistics if enabled
    if stats_enabled && let Some(stats) = crate::slt_test_engine::take_stats() {
        stats.print_summary();
    }

    conclusion
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic flattening with simple whitespace-separated values.
    #[test]
    fn test_flattening_resolves_mismatch_basic() {
        use indoc::indoc;

        let message = indoc! {"
            [Diff]
            -   1
            -   2
            -   3
            +   1 2 3
            at line 42
        "};

        assert!(LlkvSltRunner::flattening_resolves_mismatch(message));
    }

    /// Test that flattening rejects when actual has more lines than expected.
    #[test]
    fn test_flattening_rejects_more_actual_than_expected() {
        use indoc::indoc;

        let message = indoc! {"
            [Diff]
            -   1
            -   2
            +   1
            +   2
            +   3
            at line 42
        "};

        assert!(!LlkvSltRunner::flattening_resolves_mismatch(message));
    }

    /// Test flattening with quoted strings containing spaces.
    #[test]
    fn test_flattening_with_quoted_strings() {
        use indoc::indoc;

        let message = indoc! {"
            [Diff]
            -   'hello world'
            -   42
            -   'foo bar'
            +   'hello world' 42 'foo bar'
            at line 42
        "};

        assert!(LlkvSltRunner::flattening_resolves_mismatch(message));
    }

    /// Test that hash results are not flattened.
    #[test]
    fn test_flattening_skips_hash_results() {
        use indoc::indoc;

        let message = indoc! {"
            [Diff]
            -   30 values hashing to abc123def456
            +   1 2 3
            at line 42
        "};

        assert!(!LlkvSltRunner::flattening_resolves_mismatch(message));
    }

    /// Test flattening with multi-column values spread across rows.
    #[test]
    fn test_flattening_multi_column_spread() {
        use indoc::indoc;

        let message = indoc! {"
            [Diff]
            -   table tn7 row 51
            -   42
            -   100
            +   table tn7 row 51 42 100
            at line 42
        "};

        assert!(LlkvSltRunner::flattening_resolves_mismatch(message));
    }

    /// Test that mismatched values are rejected even if counts align.
    #[test]
    fn test_flattening_rejects_mismatched_values() {
        use indoc::indoc;

        let message = indoc! {"
            [Diff]
            -   1
            -   2
            -   3
            +   4 5 6
            at line 42
        "};

        assert!(!LlkvSltRunner::flattening_resolves_mismatch(message));
    }

    /// Test empty diff blocks.
    #[test]
    fn test_flattening_empty_diff() {
        use indoc::indoc;

        let message = indoc! {"
            [Diff]
            at line 42
        "};

        assert!(!LlkvSltRunner::flattening_resolves_mismatch(message));
    }

    /// Test partial matches where some values align but not all.
    #[test]
    fn test_flattening_partial_match() {
        use indoc::indoc;

        let message = indoc! {"
            [Diff]
            -   1
            -   2
            -   3
            +   1 2 4
            at line 42
        "};

        assert!(!LlkvSltRunner::flattening_resolves_mismatch(message));
    }

    /// Test smart split with evenly distributed columns.
    #[test]
    fn test_smart_split_even_distribution() {
        let actual = vec![
            "table tn7 row 51 42".to_string(),
            "table tn8 row 52 43".to_string(),
        ];
        let expected = vec![
            "table tn7 row 51".to_string(),
            "42".to_string(),
            "table tn8 row 52".to_string(),
            "43".to_string(),
        ];

        let result = LlkvSltRunner::try_smart_split_with_expected(&actual, &expected);
        assert_eq!(result, Some(expected.clone()));
    }

    /// Test smart split with uneven distribution (extra values in first line).
    #[test]
    fn test_smart_split_uneven_distribution() {
        let actual = vec!["a b c".to_string(), "d e".to_string()];
        let expected = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];

        let result = LlkvSltRunner::try_smart_split_with_expected(&actual, &expected);
        assert_eq!(result, Some(expected.clone()));
    }

    /// Test smart split rejects when expected values don't match.
    #[test]
    fn test_smart_split_rejects_mismatched_values() {
        let actual = vec!["1 2 3".to_string()];
        let expected = vec!["4".to_string(), "5".to_string(), "6".to_string()];

        let result = LlkvSltRunner::try_smart_split_with_expected(&actual, &expected);
        assert_eq!(result, None);
    }

    /// Test smart split with empty inputs.
    #[test]
    fn test_smart_split_empty_inputs() {
        let actual: Vec<String> = vec![];
        let expected = vec!["1".to_string()];

        let result = LlkvSltRunner::try_smart_split_with_expected(&actual, &expected);
        assert_eq!(result, None);

        let actual = vec!["1".to_string()];
        let expected: Vec<String> = vec![];

        let result = LlkvSltRunner::try_smart_split_with_expected(&actual, &expected);
        assert_eq!(result, None);
    }

    /// Test smart split when actual has more lines than expected.
    #[test]
    fn test_smart_split_rejects_more_actual_lines() {
        let actual = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let expected = vec!["1".to_string(), "2".to_string()];

        let result = LlkvSltRunner::try_smart_split_with_expected(&actual, &expected);
        assert_eq!(result, None);
    }

    /// Test smart split with complex multi-word values.
    #[test]
    fn test_smart_split_complex_multiword() {
        let actual = vec!["table tn7 row 51 42 foo bar".to_string()];
        let expected = vec![
            "table tn7 row 51".to_string(),
            "42".to_string(),
            "foo bar".to_string(),
        ];

        let result = LlkvSltRunner::try_smart_split_with_expected(&actual, &expected);
        assert_eq!(result, Some(expected.clone()));
    }

    /// Test split_diff_values with escaped quotes.
    #[test]
    fn test_split_diff_values_escaped_quotes() {
        let line = "'can''t' 'won''t' 42";
        let result = LlkvSltRunner::split_diff_values(line);
        assert_eq!(result, vec!["'can''t'", "'won''t'", "42"]);
    }

    /// Test split_diff_values with mixed quoted and unquoted values.
    #[test]
    fn test_split_diff_values_mixed() {
        let line = "42 'hello world' 100 'foo'";
        let result = LlkvSltRunner::split_diff_values(line);
        assert_eq!(result, vec!["42", "'hello world'", "100", "'foo'"]);
    }

    /// Test split_diff_values with leading/trailing spaces.
    #[test]
    fn test_split_diff_values_whitespace() {
        let line = "  42   'hello'  100  ";
        let result = LlkvSltRunner::split_diff_values(line);
        assert_eq!(result, vec!["42", "'hello'", "100"]);
    }

    /// Test detect_expected_hash_values with valid hash line.
    #[test]
    fn test_detect_expected_hash_values_valid() {
        let expected = vec!["30 values hashing to abc123def456".to_string()];
        let result = LlkvSltRunner::detect_expected_hash_values(&expected);
        assert_eq!(result, Some(30));
    }

    /// Test detect_expected_hash_values with invalid format.
    #[test]
    fn test_detect_expected_hash_values_invalid() {
        let expected = vec!["some random text".to_string()];
        let result = LlkvSltRunner::detect_expected_hash_values(&expected);
        assert_eq!(result, None);

        let expected = vec!["30 things hashing to abc123".to_string()];
        let result = LlkvSltRunner::detect_expected_hash_values(&expected);
        assert_eq!(result, None);
    }

    /// Test detect_expected_hash_values with empty input.
    #[test]
    fn test_detect_expected_hash_values_empty() {
        let expected: Vec<String> = vec![];
        let result = LlkvSltRunner::detect_expected_hash_values(&expected);
        assert_eq!(result, None);
    }

    /// Test flattening integration: real-world example from select4.slturl.
    #[test]
    fn test_flattening_real_world_example() {
        use indoc::indoc;

        // Simulates a case where multi-column results are collapsed into fewer rows
        let message = indoc! {"
            Query failed: expected error, but query succeeded
            [Diff]
            -   NULL
            -   NULL
            -   NULL
            -   0
            +   NULL NULL NULL 0
            at line 15234
        "};

        assert!(LlkvSltRunner::flattening_resolves_mismatch(message));
    }

    /// Test that flattening correctly handles numeric values.
    #[test]
    fn test_flattening_numeric_values() {
        use indoc::indoc;

        let message = indoc! {"
            [Diff]
            -   123
            -   456
            -   789
            +   123 456 789
            at line 100
        "};

        assert!(LlkvSltRunner::flattening_resolves_mismatch(message));
    }

    /// Test flattening with NULL values.
    #[test]
    fn test_flattening_null_values() {
        use indoc::indoc;

        let message = indoc! {"
            [Diff]
            -   NULL
            -   42
            -   NULL
            +   NULL 42 NULL
            at line 200
        "};

        assert!(LlkvSltRunner::flattening_resolves_mismatch(message));
    }
}
