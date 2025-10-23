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
            let (expanded_inner, inner_map) =
                expand_loops_with_mapping(inner, base_index + i + 1)?;

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

/// Convert sqllogictest inline connection syntax (e.g. `statement ok con1`)
/// into explicit `connection` records so the upstream parser can understand them.
/// Also ensures proper termination of statement error blocks by adding a blank line
/// after ---- when there's no expected error pattern.
#[allow(clippy::type_complexity)]
pub fn normalize_inline_connections(
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
