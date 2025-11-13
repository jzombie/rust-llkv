//! Utilities for reading and rendering canonical TPC-H query templates.

use crate::{Result, SchemaPaths, TpchError, read_file};
use core::ops::ControlFlow;
use regex::Regex;
use sqlparser::ast::{
    Expr as SqlExpr, Ident, LimitClause, ObjectName, ObjectNamePart, Statement, Value,
    ValueWithSpan, visit_relations_mut, visit_statements_mut,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::Span;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const PLACEHOLDER_STREAM: &str = ":s";
static NUMERIC_PLACEHOLDER_RE: OnceLock<Regex> = OnceLock::new();
static TYPED_STRING_RE: OnceLock<Regex> = OnceLock::new();
static INTERVAL_DAY_PRECISION_RE: OnceLock<Regex> = OnceLock::new();
static C_BLOCK_COMMENT_RE: OnceLock<Regex> = OnceLock::new();
static DEFAULT_QUERY_PARAMETERS: OnceLock<Vec<Vec<String>>> = OnceLock::new();

/// Controls rendering behavior for TPC-H query templates.
#[derive(Debug, Clone, Default)]
pub struct QueryOptions {
    pub stream_number: u32,
    pub parameter_overrides: BTreeMap<usize, String>,
}

/// Classification of rendered statements for downstream consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementKind {
    Query,
    Command,
}

/// A rendered SQL statement alongside its classification.
#[derive(Debug, Clone)]
pub struct RenderedStatement {
    pub sql: String,
    pub kind: StatementKind,
}

/// Collection of statements produced from a single TPC-H query template.
#[derive(Debug, Clone)]
pub struct RenderedQuery {
    pub number: u8,
    pub title: Option<String>,
    pub path: PathBuf,
    pub statements: Vec<RenderedStatement>,
}

/// Render a canonical TPC-H query from the bundled templates.
///
/// The renderer fetches a template, resolves stream and numeric placeholders using
/// defaults from `varsub.c` alongside caller-provided overrides, qualifies table names
/// with the supplied schema, and returns each resulting statement with its kind.
pub fn render_tpch_query(
    paths: &SchemaPaths,
    number: u8,
    schema: &str,
    options: &QueryOptions,
) -> Result<RenderedQuery> {
    if !(1..=22).contains(&number) {
        return Err(TpchError::Parse(format!(
            "TPC-H query {number} is not part of the canonical workload",
        )));
    }

    let template_path = paths.query_path(number);
    let raw = read_file(&template_path)?;
    let template = parse_template(&raw, number)?;

    let placeholders = collect_numeric_placeholders(&template.body);
    let values = build_parameter_values(paths, number, &placeholders, options)?;
    let rendered_sql = substitute_numeric_placeholders(&template.body, &values, number)?;
    let rendered_sql =
        substitute_stream_placeholder(rendered_sql, template.stream_placeholder, options);

    let dialect = GenericDialect {};
    let mut statements = Parser::parse_sql(&dialect, &rendered_sql).map_err(|err| {
        TpchError::Parse(format!(
            "failed to parse rendered SQL for query {number}: {err}"
        ))
    })?;

    qualify_statements(&mut statements, schema);

    if let Some(limit) = template.row_limit
        && limit > 0
    {
        apply_limit_to_last_query(&mut statements, limit)
            .map_err(|reason| TpchError::Parse(format!("query {number}: {reason}")))?;
    }

    let statements = statements
        .into_iter()
        .map(|statement| {
            let kind = if matches!(statement, Statement::Query(_)) {
                StatementKind::Query
            } else {
                StatementKind::Command
            };
            let mut sql = statement.to_string();
            sql = normalize_typed_string_literals(sql);
            if !sql.ends_with(';') {
                sql.push(';');
            }
            RenderedStatement { sql, kind }
        })
        .collect();

    Ok(RenderedQuery {
        number,
        title: template.title,
        path: template_path,
        statements,
    })
}

/// Parsed representation of a template file prior to placeholder substitution.
struct TemplateParse {
    title: Option<String>,
    body: String,
    stream_placeholder: bool,
    row_limit: Option<usize>,
}

/// Parse a template file into its descriptive metadata and SQL body.
fn parse_template(contents: &str, number: u8) -> Result<TemplateParse> {
    let mut title = None;
    let mut body_lines = Vec::new();
    let mut row_limit = None;

    for raw_line in contents.lines() {
        let line = raw_line.trim_end_matches('\r');
        let trimmed = line.trim();
        if trimmed.is_empty() {
            body_lines.push(String::new());
            continue;
        }

        if let Some(comment) = trimmed.strip_prefix("--") {
            let comment = comment.trim();
            if title.is_none() && !comment.is_empty() && comment != "$ID$" {
                title = Some(comment.to_string());
            }
            continue;
        }

        if let Some(directive) = trimmed.strip_prefix(':') {
            if let Some(limit_str) = directive.strip_prefix('n') {
                let limit_str = limit_str.trim();
                if !limit_str.is_empty() {
                    let parsed = limit_str.parse::<isize>().map_err(|err| {
                        TpchError::Parse(format!(
                            "query {number}: invalid :n directive {limit_str}: {err}"
                        ))
                    })?;
                    if parsed > 0 {
                        row_limit = Some(parsed as usize);
                    }
                }
            }
            // Other directives (:x, :o, etc.) are metadata for dbgen and can be ignored.
            continue;
        }

        body_lines.push(line.to_string());
    }

    let body = body_lines.join("\n").trim().to_string();
    let stream_placeholder = body.contains(PLACEHOLDER_STREAM);

    Ok(TemplateParse {
        title,
        body,
        stream_placeholder,
        row_limit,
    })
}

/// Collect every `:N` placeholder used inside the template body.
fn collect_numeric_placeholders(sql: &str) -> BTreeSet<usize> {
    let regex = NUMERIC_PLACEHOLDER_RE.get_or_init(|| Regex::new(r":(\d+)").expect("valid regex"));
    let mut placeholders = BTreeSet::new();
    for captures in regex.captures_iter(sql) {
        if let Some(matched) = captures.get(1)
            && let Ok(index) = matched.as_str().parse::<usize>()
        {
            placeholders.insert(index);
        }
    }
    placeholders
}

/// Construct the final parameter binding map for a template.
///
/// Defaults from the upstream toolkit are merged with caller overrides, and the result
/// is validated against the set of placeholders referenced in the template.
fn build_parameter_values(
    paths: &SchemaPaths,
    number: u8,
    required: &BTreeSet<usize>,
    options: &QueryOptions,
) -> Result<BTreeMap<usize, String>> {
    let mut values = BTreeMap::new();
    if let Some(defaults) = default_parameter_values(paths, number)? {
        for (offset, value) in defaults.iter().enumerate() {
            values.insert(offset + 1, value.clone());
        }
    }

    for (idx, value) in &options.parameter_overrides {
        values.insert(*idx, value.clone());
    }

    let missing: Vec<_> = required
        .iter()
        .copied()
        .filter(|idx| !values.contains_key(idx))
        .collect();
    if !missing.is_empty() {
        return Err(TpchError::Parse(format!(
            "query {number}: missing values for placeholders {missing:?}"
        )));
    }

    Ok(values)
}

/// Replace all numbered placeholders in the template with concrete values.
fn substitute_numeric_placeholders(
    body: &str,
    values: &BTreeMap<usize, String>,
    number: u8,
) -> Result<String> {
    if values.is_empty() {
        return Ok(body.to_string());
    }

    let mut ordered: Vec<(usize, &String)> =
        values.iter().map(|(idx, value)| (*idx, value)).collect();
    ordered.sort_by_key(|(idx, _)| Reverse(*idx));

    let mut rendered = body.to_string();
    for (idx, value) in ordered {
        let placeholder = format!(":{idx}");
        if rendered.contains(&placeholder) {
            rendered = rendered.replace(&placeholder, value);
        }
    }

    if let Some(unresolved) = NUMERIC_PLACEHOLDER_RE
        .get_or_init(|| Regex::new(r":(\d+)").expect("valid regex"))
        .find(&rendered)
    {
        return Err(TpchError::Parse(format!(
            "query {number}: unresolved placeholder {}",
            &rendered[unresolved.start()..unresolved.end()]
        )));
    }

    Ok(rendered)
}

/// Apply the stream-number substitution (`:s`) when present in the SQL.
fn substitute_stream_placeholder(sql: String, present: bool, options: &QueryOptions) -> String {
    if !present {
        return sql;
    }
    sql.replace(PLACEHOLDER_STREAM, &options.stream_number.to_string())
}

/// Rewrite DuckDB-style typed string literals into standard `CAST` syntax we can parse.
fn normalize_typed_string_literals(mut sql: String) -> String {
    let typed_regex = TYPED_STRING_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(DATE|TIME|TIMESTAMP)\s+'([^']*)'").expect("valid regex")
    });
    sql = typed_regex
        .replace_all(&sql, |caps: &regex::Captures<'_>| {
            let type_name = caps[1].to_ascii_uppercase();
            let literal = caps[2].replace('\'', "''");
            format!("CAST('{literal}' AS {type_name})")
        })
        .into_owned();

    let interval_regex = INTERVAL_DAY_PRECISION_RE.get_or_init(|| {
        Regex::new(r"(?i)INTERVAL\s+'([^']*)'\s+DAY\s*\(\s*\d+\s*\)").expect("valid regex")
    });
    interval_regex
        .replace_all(&sql, |caps: &regex::Captures<'_>| {
            let literal = caps[1].replace('\'', "''");
            format!("INTERVAL '{literal}' DAY")
        })
        .into_owned()
}

/// Inject a `LIMIT` clause into the final `SELECT` statement when the template
/// uses the `:n` directive to cap output rows.
fn apply_limit_to_last_query(
    statements: &mut [Statement],
    limit: usize,
) -> std::result::Result<(), String> {
    let Some(Statement::Query(query)) = statements
        .iter_mut()
        .rev()
        .find(|stmt| matches!(stmt, Statement::Query(_)))
    else {
        return Err("no SELECT statement found".to_string());
    };

    match &query.limit_clause {
        Some(LimitClause::LimitOffset { limit: Some(_), .. }) => {
            return Err("query already defines a LIMIT clause".to_string());
        }
        Some(LimitClause::OffsetCommaLimit { .. }) => {
            return Err("query uses OFFSET syntax for LIMIT".to_string());
        }
        _ => {}
    }

    query.limit_clause = Some(LimitClause::LimitOffset {
        limit: Some(SqlExpr::Value(ValueWithSpan {
            value: Value::Number(limit.to_string(), false),
            span: Span::empty(),
        })),
        offset: None,
        limit_by: Vec::new(),
    });
    Ok(())
}

/// Prefix relations in the rendered statements with the requested schema name.
fn qualify_statements(statements: &mut Vec<Statement>, schema: &str) {
    if schema.is_empty() {
        return;
    }

    let schema = schema.to_string();

    let _ = visit_statements_mut(statements, |statement| {
        match statement {
            Statement::CreateView { name, .. } => {
                qualify_object_name(name, &schema);
            }
            Statement::Drop { names, .. } => {
                for name in names {
                    qualify_object_name(name, &schema);
                }
            }
            _ => {}
        }
        ControlFlow::<()>::Continue(())
    });

    let _ = visit_relations_mut(statements, |relation| {
        qualify_object_name(relation, &schema);
        ControlFlow::<()>::Continue(())
    });
}

/// Insert the schema identifier at the front of an object name when needed.
fn qualify_object_name(name: &mut ObjectName, schema: &str) {
    if name.0.is_empty() {
        name.0.push(ObjectNamePart::Identifier(Ident::new(schema)));
        return;
    }
    if name.0.len() > 1 {
        return;
    }
    if let Some(ObjectNamePart::Identifier(existing)) = name.0.first()
        && existing.value.eq_ignore_ascii_case(schema)
    {
        return;
    }
    name.0
        .insert(0, ObjectNamePart::Identifier(Ident::new(schema)));
}

/// Fetch the cached default parameters for the requested query number.
fn default_parameter_values(paths: &SchemaPaths, query: u8) -> Result<Option<&'static [String]>> {
    if query == 0 {
        return Ok(None);
    }

    let defaults = ensure_query_parameter_defaults(paths)?;
    let idx = (query - 1) as usize;
    let entry = defaults.get(idx).filter(|values| !values.is_empty());
    Ok(entry.map(|values| values.as_slice()))
}

/// Load and cache the default parameter table from the upstream toolkit.
fn ensure_query_parameter_defaults(paths: &SchemaPaths) -> Result<&'static Vec<Vec<String>>> {
    if let Some(existing) = DEFAULT_QUERY_PARAMETERS.get() {
        return Ok(existing);
    }

    let parsed = parse_default_query_parameters(&paths.varsub_source)?;
    DEFAULT_QUERY_PARAMETERS
        .set(parsed)
        .map_err(|_| TpchError::Parse("default parameter cache already initialized".into()))?;
    DEFAULT_QUERY_PARAMETERS
        .get()
        .ok_or_else(|| TpchError::Parse("failed to load default parameter cache".into()))
}

/// Parse the `defaults` array in `varsub.c` into per-query parameter vectors.
fn parse_default_query_parameters(path: &Path) -> Result<Vec<Vec<String>>> {
    let source = read_file(path)?;
    let definition_start = source
        .find("char *defaults")
        .ok_or_else(|| TpchError::Parse("unable to locate defaults array in varsub.c".into()))?;
    let after_definition = &source[definition_start..];
    let open_brace = after_definition.find('{').ok_or_else(|| {
        TpchError::Parse("malformed defaults array: missing opening brace".into())
    })?;

    let mut depth = 0i32;
    let mut closing_offset = None;
    for (idx, ch) in after_definition[open_brace..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    closing_offset = Some(idx + open_brace);
                    break;
                }
            }
            _ => {}
        }
    }

    let closing_offset = closing_offset.ok_or_else(|| {
        TpchError::Parse("malformed defaults array: missing closing brace".into())
    })?;
    let body = &after_definition[open_brace + 1..closing_offset];
    let comment_free = strip_c_block_comments(body);
    let entries = extract_defaults_entries(&comment_free);

    if entries.is_empty() {
        return Err(TpchError::Parse(
            "failed to extract any default query parameter entries".into(),
        ));
    }

    Ok(entries)
}

/// Remove C-style block comments from a snippet before parsing it as an initializer.
fn strip_c_block_comments(contents: &str) -> String {
    let regex = C_BLOCK_COMMENT_RE
        .get_or_init(|| Regex::new(r"(?s)/\*.*?\*/").expect("valid C comment regex"));
    regex.replace_all(contents, "").into_owned()
}

/// Split the defaults initializer into individual query entries.
fn extract_defaults_entries(contents: &str) -> Vec<Vec<String>> {
    let mut entries = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();

    for ch in contents.chars() {
        match ch {
            '{' => {
                if depth > 0 {
                    current.push(ch);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    entries.push(parse_default_entry(&current));
                    current.clear();
                } else if depth > 0 {
                    current.push(ch);
                }
            }
            _ => {
                if depth > 0 {
                    current.push(ch);
                }
            }
        }
    }

    entries
}

/// Convert a single array entry into a list of string parameters, skipping `NULL`s.
fn parse_default_entry(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|value| value.trim())
        .filter_map(|value| {
            if value.is_empty() || value.eq_ignore_ascii_case("NULL") {
                None
            } else {
                Some(unquote(value))
            }
        })
        .collect()
}

/// Strip surrounding quotes from a C string literal and collapse escaped quotes.
fn unquote(value: &str) -> String {
    if let Some(stripped) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
        stripped.replace("\\\"", "\"")
    } else {
        value.to_string()
    }
}

// TODO: Only enable when `tpc_tools` are installed.
// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn q1_rendering_includes_schema_prefix() {
//         let paths = SchemaPaths::discover();
//         let options = QueryOptions::default();
//         let query = render_tpch_query(&paths, 1, "TPCD", &options).expect("render Q1");
//         assert_eq!(query.statements.len(), 1);
//         let sql = &query.statements[0].sql;
//         assert!(sql.to_ascii_uppercase().contains("FROM TPCD.LINEITEM"));
//         assert!(!sql.contains(':'));
//     }

//     #[test]
//     fn q3_applies_limit_clause() {
//         let paths = SchemaPaths::discover();
//         let options = QueryOptions::default();
//         let query = render_tpch_query(&paths, 3, "TPCD", &options).expect("render Q3");
//         assert!(
//             query.statements[0]
//                 .sql
//                 .to_ascii_uppercase()
//                 .contains("LIMIT 10")
//         );
//     }
// }
