use llkv_result::Error;
use regex::escape;
use std::path::Path;

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
                let token_plain = format!("${}", var);
                let token_braced = format!("${{{}}}", var);
                for (s, &orig_line) in expanded_inner.iter().zip(inner_map.iter()) {
                    let substituted = s.replace(&token_braced, &val).replace(&token_plain, &val);
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

/// Filter out conditional test blocks based on database engine directives.
///
/// Handles `onlyif <engine>` and `skipif <engine>` directives for compatible engines.
///
/// Logic:
/// - `onlyif <engine_name>`: Include the block if `engine_name` is in our_engines list, skip otherwise
/// - `skipif <engine_name>`: Skip the block if `engine_name` is in our_engines list, include otherwise
///
/// This allows a test marked `onlyif sqlite` OR `onlyif duckdb` to run if we're compatible
/// with either engine, and a test marked `skipif sqlite` OR `skipif duckdb` to be skipped
/// if we're compatible with either.
///
/// Internal helper used by the test runner.
#[allow(clippy::type_complexity)]
pub(crate) fn filter_conditional_blocks(
    lines: Vec<String>,
    mapping: Vec<usize>,
    our_engines: &[&str],
) -> (Vec<String>, Vec<usize>) {
    let mut out_lines = Vec::new();
    let mut out_map = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = &lines[i];
        let trimmed = line.trim_start();

        // Check for conditional directives
        if let Some(rest) = trimmed.strip_prefix("onlyif ") {
            let engine = rest.split_whitespace().next().unwrap_or("");
            if our_engines.contains(&engine) {
                // This test is only for an engine we're compatible with - include it but skip the directive line
                i += 1;
                continue;
            } else {
                // This test is only for an engine we're not compatible with - skip the entire test block
                i += 1;
                i = skip_test_block(&lines, i);
                continue;
            }
        } else if let Some(rest) = trimmed.strip_prefix("skipif ") {
            let engine = rest.split_whitespace().next().unwrap_or("");
            if our_engines.contains(&engine) {
                // Skip this test for any of our compatible engines - skip the directive and test block
                i += 1;
                i = skip_test_block(&lines, i);
                continue;
            } else {
                // This test is skipped for another engine, so include it for us
                // but skip the directive line
                i += 1;
                continue;
            }
        }

        // Not a conditional directive - keep the line
        out_lines.push(line.clone());
        out_map.push(mapping[i]);
        i += 1;
    }

    (out_lines, out_map)
}

/// Skip a test block (query, statement, etc.) after a conditional directive.
///
/// SLT test blocks are delimited by blank lines. This function skips all non-blank
/// lines until it hits a blank line, which marks the end of the current test block.
///
/// Returns the index of the next line after the block (which should be the blank line
/// or the start of the next block if there's no trailing blank line).
fn skip_test_block(lines: &[String], start_idx: usize) -> usize {
    let mut i = start_idx;

    // Skip all non-blank lines - test blocks are terminated by blank lines in SLT format
    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty() {
            // Found the blank line terminator - skip it and we're done
            i += 1;
            break;
        }
        i += 1;
    }

    i
}

/// Convert sqllogictest inline connection syntax (e.g. `statement ok con1`)
/// into explicit `connection` records so the upstream parser can understand them.
/// Also ensures proper termination of statement error blocks by adding a blank line
/// after ---- when there's no expected error pattern.
///
/// Internal helper used by the test runner.
#[allow(clippy::type_complexity)]
pub(crate) fn normalize_inline_connections(
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

            if regex_pattern.is_none()
                && !message_lines.is_empty()
                && let Some((first_line, _)) = message_lines.first()
            {
                let trimmed_first = first_line.trim();
                if !trimmed_first.is_empty() {
                    let escaped = escape(trimmed_first);
                    regex_pattern = Some(format!(".*{}.*", escaped));
                    message_lines.clear();
                }
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
                    if saw_separator && !has_regex && !message_lines.is_empty() {
                        out_lines.push(format!("{indent}----"));
                        out_map.push(orig);
                        for (msg_line, msg_map) in message_lines {
                            out_lines.push(msg_line);
                            out_map.push(msg_map);
                        }
                        out_lines.push(String::new());
                        out_map.push(orig);
                    }
                    out_lines.push(String::new());
                    out_map.push(orig);
                    continue;
                } else {
                    out_lines.push(normalized);
                    out_map.push(orig);
                    i += 1;
                    continue;
                }
            }
        }

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
            if saw_separator && !has_regex && !message_lines.is_empty() {
                out_lines.push(format!("{indent}----"));
                out_map.push(orig);
                for (msg_line, msg_map) in message_lines {
                    out_lines.push(msg_line);
                    out_map.push(msg_map);
                }
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

/// Map a temporary expanded-file error message back to the original and
/// normalized line numbers.
///
/// Returns the rewritten message plus an optional pair of line numbers in the
/// form `(original_source_line, normalized_line)`.
///
/// Internal helper used by the test runner.
pub(crate) fn map_temp_error_message(
    err_msg: &str,
    tmp_path: &Path,
    expanded_lines: &[String],
    mapping: &[usize],
    orig_path: &Path,
) -> (String, Option<(usize, usize)>) {
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
                        let normalized_line = idx_us + 1;
                        return (out, Some((orig_line, normalized_line)));
                    }
                }
            }
        }
    }
    (out, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    fn slt_fixture(input: &str) -> (Vec<String>, Vec<usize>) {
        let lines: Vec<String> = input.split('\n').map(|line| line.to_string()).collect();
        let mapping: Vec<usize> = (1..=lines.len()).collect();
        (lines, mapping)
    }

    #[test]
    fn test_filter_conditional_blocks_onlyif_match() {
        // Test that "onlyif llkv" includes the test for llkv
        let lines = vec![
            "query I".to_string(),
            "SELECT 1".to_string(),
            "----".to_string(),
            "1".to_string(),
            "".to_string(),
            "onlyif llkv".to_string(),
            "query I".to_string(),
            "SELECT 2".to_string(),
            "----".to_string(),
            "2".to_string(),
            "".to_string(),
        ];
        let mapping: Vec<usize> = (1..=lines.len()).collect();

        let (filtered, _) = filter_conditional_blocks(lines, mapping, &["llkv"]);

        // Should include first test and second test (without the directive line)
        assert!(filtered.iter().any(|l| l.contains("SELECT 1")));
        assert!(filtered.iter().any(|l| l.contains("SELECT 2")));
        assert!(!filtered.iter().any(|l| l.contains("onlyif")));
    }

    #[test]
    fn test_filter_conditional_blocks_onlyif_no_match() {
        // Test that "onlyif mysql" excludes the test for llkv
        let lines = vec![
            "query I".to_string(),
            "SELECT 1".to_string(),
            "----".to_string(),
            "1".to_string(),
            "".to_string(),
            "onlyif mysql".to_string(),
            "query I".to_string(),
            "SELECT 2".to_string(),
            "----".to_string(),
            "2".to_string(),
            "".to_string(),
            "query I".to_string(),
            "SELECT 3".to_string(),
            "----".to_string(),
            "3".to_string(),
            "".to_string(),
        ];
        let mapping: Vec<usize> = (1..=lines.len()).collect();

        let (filtered, _) = filter_conditional_blocks(lines, mapping, &["llkv"]);

        // Should include first and third tests, but not the mysql-only test
        assert!(filtered.iter().any(|l| l.contains("SELECT 1")));
        assert!(!filtered.iter().any(|l| l.contains("SELECT 2")));
        assert!(filtered.iter().any(|l| l.contains("SELECT 3")));
        assert!(!filtered.iter().any(|l| l.contains("onlyif")));
    }

    #[test]
    fn test_filter_conditional_blocks_onlyif_skips_single_block() {
        let (lines, mapping) = slt_fixture(indoc! {
            r#"
            query I
            SELECT 1
            ----
            1

            onlyif mysql # requires mysql
            query I rowsort
            SELECT 2
            ----
            2

            query I
            SELECT 3
            ----
            3
            "#
        });

        let (filtered, filtered_mapping) = filter_conditional_blocks(lines, mapping, &["llkv"]);

        assert!(filtered.iter().any(|l| l.contains("SELECT 1")));
        assert!(!filtered.iter().any(|l| l.contains("SELECT 2")));

        let select3_idx = filtered
            .iter()
            .position(|l| l.contains("SELECT 3"))
            .expect("expected trailing block to remain");
        assert_eq!(filtered_mapping[select3_idx], 13);
    }

    #[test]
    fn test_filter_conditional_blocks_skipif_match() {
        // Test that "skipif mysql" excludes the block for engines we consider compatible
        let lines = vec![
            "query I".to_string(),
            "SELECT 1".to_string(),
            "----".to_string(),
            "1".to_string(),
            "".to_string(),
            "skipif mysql".to_string(),
            "query I".to_string(),
            "SELECT 2".to_string(),
            "----".to_string(),
            "2".to_string(),
            "".to_string(),
        ];
        let mapping: Vec<usize> = (1..=lines.len()).collect();

        let (filtered, _) = filter_conditional_blocks(lines, mapping, &["llkv", "mysql"]);

        // Should include first test but not the second (skipped for mysql)
        assert!(filtered.iter().any(|l| l.contains("SELECT 1")));
        assert!(!filtered.iter().any(|l| l.contains("SELECT 2")));
        assert!(!filtered.iter().any(|l| l.contains("skipif")));
    }

    #[test]
    fn test_filter_conditional_blocks_skipif_skips_only_next_block() {
        let (lines, mapping) = slt_fixture(indoc! {
            r#"
            skipif mysql # not compatible
            query I rowsort label-9840
            SELECT ALL CAST ( - col1 AS INTEGER ) + cor0.col0 + - CAST ( NULL AS INTEGER ) AS col0 FROM tab0 AS cor0
            ----
            NULL
            NULL
            NULL

            query I rowsort
            SELECT DISTINCT - col0 * - col1 + col1 FROM tab1 AS cor0
            ----
            104
            1053
            650
            "#
        });

        let (filtered, filtered_mapping) =
            filter_conditional_blocks(lines, mapping, &["llkv", "mysql"]);

        assert!(!filtered.iter().any(|l| l.contains("label-9840")));
        let select2_idx = filtered
            .iter()
            .position(|l| l.contains("SELECT DISTINCT"))
            .expect("expected second block to remain");
        assert_eq!(filtered_mapping[select2_idx], 10);
    }

    #[test]
    fn test_filter_conditional_blocks_skipif_no_match() {
        // Test that "skipif mysql" includes the test for llkv
        let lines = vec![
            "query I".to_string(),
            "SELECT 1".to_string(),
            "----".to_string(),
            "1".to_string(),
            "".to_string(),
            "skipif mysql".to_string(),
            "query I".to_string(),
            "SELECT 2".to_string(),
            "----".to_string(),
            "2".to_string(),
            "".to_string(),
        ];
        let mapping: Vec<usize> = (1..=lines.len()).collect();

        let (filtered, _) = filter_conditional_blocks(lines, mapping, &["llkv"]);

        // Should include both tests (second test is skipped for mysql, not llkv)
        assert!(filtered.iter().any(|l| l.contains("SELECT 1")));
        assert!(filtered.iter().any(|l| l.contains("SELECT 2")));
        assert!(!filtered.iter().any(|l| l.contains("skipif")));
    }

    #[test]
    fn test_filter_conditional_blocks_skipif_at_eof_without_blank_line() {
        // Skip a trailing block that is not terminated by a blank line
        let lines = vec![
            "query I".to_string(),
            "SELECT 1".to_string(),
            "----".to_string(),
            "1".to_string(),
            "".to_string(),
            "skipif sqlite".to_string(),
            "query I".to_string(),
            "SELECT 2".to_string(),
            "----".to_string(),
            "2".to_string(),
        ];
        let mapping: Vec<usize> = (1..=lines.len()).collect();

        let (filtered, _) = filter_conditional_blocks(lines, mapping, &["llkv", "sqlite"]);

        assert!(filtered.iter().any(|l| l.contains("SELECT 1")));
        assert!(!filtered.iter().any(|l| l.contains("SELECT 2")));
    }

    #[test]
    fn test_filter_conditional_blocks_multiple_conditions() {
        // Test multiple conditional directives in sequence
        let lines = vec![
            "onlyif mysql".to_string(),
            "query I".to_string(),
            "SELECT 1".to_string(),
            "----".to_string(),
            "1".to_string(),
            "".to_string(),
            "skipif mysql".to_string(),
            "query I".to_string(),
            "SELECT 2".to_string(),
            "----".to_string(),
            "2".to_string(),
            "".to_string(),
            "onlyif llkv".to_string(),
            "query I".to_string(),
            "SELECT 3".to_string(),
            "----".to_string(),
            "3".to_string(),
            "".to_string(),
        ];
        let mapping: Vec<usize> = (1..=lines.len()).collect();

        let (filtered, _) = filter_conditional_blocks(lines, mapping, &["llkv"]);

        // Should skip first (mysql only), include second (not mysql), include third (llkv only)
        assert!(!filtered.iter().any(|l| l.contains("SELECT 1")));
        assert!(filtered.iter().any(|l| l.contains("SELECT 2")));
        assert!(filtered.iter().any(|l| l.contains("SELECT 3")));
    }

    #[test]
    fn test_filter_conditional_blocks_statement() {
        // Test with statement blocks, not just queries
        let lines = vec![
            "onlyif mysql".to_string(),
            "statement ok".to_string(),
            "CREATE TABLE test (id INT)".to_string(),
            "".to_string(),
            "skipif postgresql".to_string(),
            "statement ok".to_string(),
            "DROP TABLE test".to_string(),
            "".to_string(),
        ];
        let mapping: Vec<usize> = (1..=lines.len()).collect();

        let (filtered, _) = filter_conditional_blocks(lines, mapping, &["llkv"]);

        // Should skip first (mysql only), include second (not postgresql)
        assert!(!filtered.iter().any(|l| l.contains("CREATE TABLE")));
        assert!(filtered.iter().any(|l| l.contains("DROP TABLE")));
    }

    #[test]
    fn test_filter_conditional_blocks_preserves_mapping() {
        // Test that line number mapping is preserved correctly
        let lines = vec![
            "query I".to_string(),
            "SELECT 1".to_string(),
            "----".to_string(),
            "1".to_string(),
            "".to_string(),
            "onlyif mysql".to_string(), // line 6
            "query I".to_string(),      // line 7 - should be skipped
            "SELECT 2".to_string(),     // line 8 - should be skipped
            "----".to_string(),         // line 9 - should be skipped
            "2".to_string(),            // line 10 - should be skipped
            "".to_string(),             // line 11 - should be skipped
            "query I".to_string(),      // line 12
            "SELECT 3".to_string(),     // line 13
            "----".to_string(),         // line 14
            "3".to_string(),            // line 15
            "".to_string(),             // line 16
        ];
        let mapping: Vec<usize> = (1..=lines.len()).collect();

        let (filtered, filtered_mapping) = filter_conditional_blocks(lines, mapping, &["llkv"]);

        // Verify that SELECT 3 is in the output and its mapping points to line 13
        let select3_idx = filtered
            .iter()
            .position(|l| l.contains("SELECT 3"))
            .unwrap();
        assert_eq!(filtered_mapping[select3_idx], 13);
    }
}
