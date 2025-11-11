//! Helpers for installing the canonical TPC-H schema inside an LLKV database.
//!
//! The toolkit bundles the upstream `dbgen` sources. This module reads the
//! `dss.h` metadata and `dss.ri` constraint file from that distribution, then
//! installs the schema into a `SqlEngine`. The goal is to let the TPC-H DDL run
//! unmodified while still producing a structured manifest the caller can inspect.

use llkv::Error as LlkvError;
use llkv_sql::{
    SqlEngine, canonical_table_ident, normalize_table_constraint,
    order_create_tables_by_foreign_keys,
};
use regex::Regex;
use sqlparser::ast::{
    AlterTableOperation, CreateTable, Ident, ObjectNamePart, Statement, TableConstraint,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

const DEFAULT_SCHEMA_NAME: &str = "TPCD";
const DBGEN_RELATIVE_PATH: &str = "tpc_tools/dbgen";
const DSS_HEADER_FILE: &str = "dss.h";
const DSS_DDL_FILE: &str = "dss.ddl";
const DSS_RI_FILE: &str = "dss.ri";
const DRIVER_SOURCE_FILE: &str = "driver.c";

/// Errors that can occur while installing the TPC-H schema.
#[derive(Debug, Error)]
pub enum TpchError {
    #[error("failed to read {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {0}")]
    Parse(String),
    #[error("SQL execution failed: {0}")]
    Sql(#[from] LlkvError),
}

/// Convenient alias for results returned by schema helpers.
pub type Result<T> = std::result::Result<T, TpchError>;

/// File system locations for the bundled TPC-H metadata.
#[derive(Debug, Clone)]
pub struct SchemaPaths {
    pub dss_header: PathBuf,
    pub ddl: PathBuf,
    pub referential_integrity: PathBuf,
    pub tdefs_source: PathBuf,
}

impl SchemaPaths {
    /// Discover the default toolkit layout relative to this crate.
    pub fn discover() -> Self {
        Self::from_root(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
    }

    /// Construct a `SchemaPaths` rooted at the provided directory.
    pub fn from_root(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        let dbgen_root = root.join(DBGEN_RELATIVE_PATH);
        Self {
            dss_header: dbgen_root.join(DSS_HEADER_FILE),
            ddl: dbgen_root.join(DSS_DDL_FILE),
            referential_integrity: dbgen_root.join(DSS_RI_FILE),
            tdefs_source: dbgen_root.join(DRIVER_SOURCE_FILE),
        }
    }
}

impl Default for SchemaPaths {
    fn default() -> Self {
        Self::discover()
    }
}

/// Summary of the installed TPC-H schema.
#[derive(Debug, Clone)]
pub struct TpchSchema {
    pub schema_name: String,
    pub tables: Vec<TpchTableInfo>,
}

/// Table-level metadata derived from `dss.h`.
#[derive(Debug, Clone)]
pub struct TpchTableInfo {
    pub name: String,
    pub file_name: String,
    pub description: String,
    pub base_rows: u64,
}

#[derive(Debug, Clone)]
struct RawTableDef {
    file_name: String,
    description: String,
    base_rows: u64,
}

/// Install the bundled TPC-H schema into the provided SQL engine.
///
/// This helper uses the default toolkit paths relative to the `llkv-tpc-h`
/// crate and is the easiest way to bootstrap a database for experimentation.
pub fn install_default_schema(engine: &SqlEngine) -> Result<TpchSchema> {
    install_schema(engine, &SchemaPaths::default())
}

/// Install the TPC-H schema using explicit metadata locations.
pub fn install_schema(engine: &SqlEngine, paths: &SchemaPaths) -> Result<TpchSchema> {
    let dss_header = read_file(&paths.dss_header)?;
    let macros = parse_numeric_macros(&dss_header);
    let tdefs_source = read_file(&paths.tdefs_source)?;
    let raw_tables = parse_tdefs(&tdefs_source, &macros)?;

    let ddl_sql = read_file(&paths.ddl)?;
    let (mut create_tables, table_names) = parse_ddl_with_schema(&ddl_sql, DEFAULT_SCHEMA_NAME)?;

    let ri_sql = read_file(&paths.referential_integrity)?;
    let constraint_map = parse_referential_integrity(&ri_sql)?;
    apply_constraints_to_tables(&mut create_tables, &constraint_map);
    let ordered_tables = order_create_tables_by_foreign_keys(create_tables);

    run_sql(
        engine,
        &format!("CREATE SCHEMA IF NOT EXISTS {DEFAULT_SCHEMA_NAME};"),
    )?;
    let ddl_batch_sql = render_create_tables(&ordered_tables);
    run_sql(engine, &ddl_batch_sql)?;

    let tables = build_table_infos(table_names, &raw_tables);

    Ok(TpchSchema {
        schema_name: DEFAULT_SCHEMA_NAME.to_string(),
        tables,
    })
}

fn read_file(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|source| TpchError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn run_sql(engine: &SqlEngine, sql: &str) -> Result<()> {
    if sql.trim().is_empty() {
        return Ok(());
    }
    engine.execute(sql).map(|_| ()).map_err(TpchError::Sql)
}

fn parse_numeric_macros(contents: &str) -> HashMap<String, i64> {
    let mut macros = HashMap::new();
    for line in contents.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("#define") {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let _define = parts.next();
        let name = match parts.next() {
            Some(name) => name,
            None => continue,
        };
        let value_token = match parts.next() {
            Some(value) => value,
            None => continue,
        };
        if let Some(value) = parse_numeric_literal(value_token) {
            macros.insert(name.to_string(), value);
        }
    }
    macros
}

fn parse_numeric_literal(token: &str) -> Option<i64> {
    if let Ok(value) = token.parse::<i64>() {
        return Some(value);
    }
    if let Some(hex) = token.strip_prefix("0x") {
        if let Ok(value) = i64::from_str_radix(hex, 16) {
            return Some(value);
        }
    }
    None
}

fn parse_tdefs(
    contents: &str,
    macros: &HashMap<String, i64>,
) -> Result<HashMap<String, RawTableDef>> {
    let marker = "tdef tdefs[]";
    let start = contents.find(marker).ok_or_else(|| {
        TpchError::Parse("unable to locate tdef tdefs[] declaration in driver.c".into())
    })?;
    let after_marker = &contents[start..];
    let block_start = after_marker
        .find('{')
        .ok_or_else(|| TpchError::Parse("malformed tdef array: missing opening brace".into()))?;
    let block_body = &after_marker[block_start + 1..];

    let mut depth: i32 = 0;
    let mut block_end: Option<usize> = None;
    for (idx, ch) in block_body.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                if depth == 0 {
                    block_end = Some(idx);
                    break;
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    let block_end = block_end
        .ok_or_else(|| TpchError::Parse("malformed tdef array: missing closing brace".into()))?;
    let block = &block_body[..block_end];

    let entry_re = Regex::new(r#"\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*([^,]+),"#)
        .map_err(|err| TpchError::Parse(format!("invalid tdef regex: {err}")))?;

    let mut tables = HashMap::new();
    for caps in entry_re.captures_iter(block) {
        let file_name = caps[1].to_string();
        // Keep the first definition for a file to avoid overwriting the base table
        if tables.contains_key(&file_name) {
            continue;
        }
        let description = caps[2].trim().to_string();
        let base_expr = caps[3].trim();
        let base_rows = evaluate_base_expr(base_expr, macros).map_err(|err| {
            TpchError::Parse(format!(
                "unable to evaluate base row count for {}: {err}",
                file_name
            ))
        })?;
        tables.insert(
            file_name.clone(),
            RawTableDef {
                file_name,
                description,
                base_rows,
            },
        );
    }

    Ok(tables)
}

fn evaluate_base_expr(
    expr: &str,
    macros: &HashMap<String, i64>,
) -> std::result::Result<u64, String> {
    if let Some(value) = parse_numeric_literal(expr) {
        return Ok(value as u64);
    }
    if let Some(value) = macros.get(expr) {
        return Ok(*value as u64);
    }
    Err(format!("unrecognized literal or macro '{expr}'"))
}

fn parse_ddl_with_schema(ddl_sql: &str, schema: &str) -> Result<(Vec<CreateTable>, Vec<String>)> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, ddl_sql)
        .map_err(|err| TpchError::Parse(format!("failed to parse dss.ddl: {err}")))?;

    let mut tables = Vec::new();
    let mut names = Vec::new();

    for statement in statements {
        if let Statement::CreateTable(mut create_table) = statement {
            if create_table.name.0.is_empty() {
                return Err(TpchError::Parse(
                    "CREATE TABLE statement missing table name".into(),
                ));
            }
            if create_table.name.0.len() == 1 {
                create_table
                    .name
                    .0
                    .insert(0, ObjectNamePart::Identifier(Ident::new(schema)));
            } else {
                create_table.name.0[0] = ObjectNamePart::Identifier(Ident::new(schema));
            }

            let table_name = create_table
                .name
                .0
                .last()
                .and_then(|part| part.as_ident())
                .map(|ident| ident.value.to_ascii_uppercase())
                .ok_or_else(|| {
                    TpchError::Parse("table name does not end with identifier".into())
                })?;

            names.push(table_name);
            tables.push(create_table);
        }
    }

    Ok((tables, names))
}

fn parse_referential_integrity(ri_sql: &str) -> Result<HashMap<String, Vec<TableConstraint>>> {
    let cleaned = strip_connect_statements(ri_sql);
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, &cleaned)
        .map_err(|err| TpchError::Parse(format!("failed to parse dss.ri: {err}")))?;

    let mut constraints: HashMap<String, Vec<TableConstraint>> = HashMap::new();

    for statement in statements {
        if let Statement::AlterTable {
            name, operations, ..
        } = statement
        {
            let table_name = canonical_table_ident(&name).unwrap_or_default();
            let bucket = constraints.entry(table_name).or_default();
            for op in operations {
                if let AlterTableOperation::AddConstraint { constraint, .. } = op {
                    bucket.push(constraint);
                }
            }
        }
    }

    Ok(constraints)
}

fn apply_constraints_to_tables(
    tables: &mut [CreateTable],
    constraints: &HashMap<String, Vec<TableConstraint>>,
) {
    for table in tables {
        let table_name = canonical_table_ident(&table.name).unwrap_or_default();
        if let Some(entries) = constraints.get(&table_name) {
            table
                .constraints
                .extend(entries.iter().cloned().map(normalize_table_constraint));
        }
    }
}

fn render_create_tables(tables: &[CreateTable]) -> String {
    let mut sql = String::new();
    for table in tables {
        if !sql.is_empty() {
            sql.push('\n');
        }
        let statement = Statement::CreateTable(table.clone());
        sql.push_str(&statement.to_string());
        sql.push_str(";\n");
    }
    sql
}

fn build_table_infos(
    names: Vec<String>,
    raw_tables: &HashMap<String, RawTableDef>,
) -> Vec<TpchTableInfo> {
    names
        .into_iter()
        .map(|name| {
            let file_key = format!("{}.tbl", name.to_ascii_lowercase());
            if let Some(raw) = raw_tables.get(&file_key) {
                TpchTableInfo {
                    name: name.clone(),
                    file_name: raw.file_name.clone(),
                    description: raw.description.clone(),
                    base_rows: raw.base_rows,
                }
            } else {
                TpchTableInfo {
                    name: name.clone(),
                    file_name: file_key.clone(),
                    description: format!("{} table", name.to_ascii_lowercase()),
                    base_rows: 0,
                }
            }
        })
        .collect()
}

fn strip_connect_statements(sql: &str) -> String {
    Regex::new(r#"(?im)^\s*CONNECT\s+TO\s+(?:[A-Za-z0-9_]+|'[^']+'|"[^"]+")\s*;\s*"#)
        .expect("valid CONNECT removal regex")
        .replace_all(sql, "")
        .into_owned()
}
