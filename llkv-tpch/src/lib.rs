//! Helpers for installing the canonical TPC-H schema inside an LLKV database.
//!
//! The toolkit bundles the upstream `dbgen` sources. This module reads the
//! `dss.h` metadata and `dss.ri` constraint file from that distribution, then
//! installs the schema into a `SqlEngine`. The goal is to let the TPC-H DDL run
//! unmodified while still producing a structured manifest the caller can inspect.

use llkv::Error as LlkvError;
use llkv_expr::decimal::DecimalValue;
use llkv_plan::{InsertConflictAction, InsertPlan, InsertSource, PlanValue, parse_date32_literal};
use llkv_sql::{
    SqlEngine, SqlTypeFamily, canonical_table_ident, classify_sql_data_type,
    normalize_table_constraint, order_create_tables_by_foreign_keys,
    tpch::strip_tpch_connect_statements,
};
use llkv_table::ConstraintEnforcementMode;
use regex::Regex;
use sqlparser::ast::{
    AlterTableOperation, CreateTable, Ident, ObjectNamePart, Statement, TableConstraint,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use thiserror::Error;
use tpchgen::generators::{
    CustomerGenerator, LineItemGenerator, NationGenerator, OrderGenerator, PartGenerator,
    PartSuppGenerator, RegionGenerator, SupplierGenerator,
};

pub mod qualification;
pub mod queries;

pub const DEFAULT_SCHEMA_NAME: &str = "TPCD";
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

/// File system locations for the bundled TPC-H metadata and templates.
#[derive(Debug, Clone)]
pub struct SchemaPaths {
    pub dss_header: PathBuf,
    pub ddl: PathBuf,
    pub referential_integrity: PathBuf,
    pub tdefs_source: PathBuf,
    pub queries_dir: PathBuf,
    pub varsub_source: PathBuf,
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
            queries_dir: dbgen_root.join("queries"),
            varsub_source: dbgen_root.join("varsub.c"),
        }
    }

    /// Return the canonical path to the requested TPC-H SQL query template.
    pub fn query_path(&self, query_number: u8) -> PathBuf {
        self.queries_dir.join(format!("{query_number}.sql"))
    }

    /// Return the directory containing the bundled TPC-H tooling assets.
    pub fn tools_root(&self) -> PathBuf {
        self.dss_header
            .parent()
            .and_then(|dbgen| dbgen.parent())
            .map(|root| root.to_path_buf())
            .expect("SchemaPaths missing dbgen root")
    }

    /// Return the `ref_data/<scale>` directory that ships qualification artifacts.
    pub fn ref_data_dir(&self, scale: impl AsRef<Path>) -> PathBuf {
        self.tools_root().join("ref_data").join(scale)
    }

    /// Return the directory containing the TPC-H `check_answers` helpers.
    pub fn check_answers_dir(&self) -> PathBuf {
        self.tools_root().join("dbgen").join("check_answers")
    }

    /// Return the directory containing the bundled canonical answer sets.
    pub fn answers_dir(&self) -> PathBuf {
        self.tools_root().join("dbgen").join("answers")
    }
}

impl Default for SchemaPaths {
    fn default() -> Self {
        Self::discover()
    }
}

#[derive(Debug, Clone)]
pub struct TpchToolkit {
    schema_paths: SchemaPaths,
    schema_name: String,
    tables_by_name: HashMap<String, TableSchema>,
    creation_order: Vec<String>,
}

#[derive(Debug, Clone)]
struct TableSchema {
    name: String,
    create_table: CreateTable,
    info: TpchTableInfo,
    columns: Vec<TableColumn>,
    column_names: Vec<String>,
}

#[derive(Debug, Clone)]
struct TableColumn {
    name: String,
    value_kind: ColumnValueKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnValueKind {
    String,
    Integer,
    Decimal { scale: i8 },
    Date32,
}

impl TpchToolkit {
    const PROGRESS_REPORT_INTERVAL: usize = 100_000;
    /// Build a toolkit by parsing the bundled TPC-H metadata at the provided paths.
    pub fn from_paths(paths: SchemaPaths) -> Result<Self> {
        let dss_header = read_file(&paths.dss_header)?;
        let macros = parse_numeric_macros(&dss_header);
        let tdefs_source = read_file(&paths.tdefs_source)?;
        let raw_tables = parse_tdefs(&tdefs_source, &macros)?;

        let ddl_sql = read_file(&paths.ddl)?;
        let (mut create_tables, _) = parse_ddl_with_schema(&ddl_sql, DEFAULT_SCHEMA_NAME)?;

        let ri_sql = read_file(&paths.referential_integrity)?;
        let constraint_map = parse_referential_integrity(&ri_sql)?;
        apply_constraints_to_tables(&mut create_tables, &constraint_map);

        let ordered_tables = order_create_tables_by_foreign_keys(create_tables);

        let mut tables_by_name = HashMap::with_capacity(ordered_tables.len());
        let mut creation_order = Vec::with_capacity(ordered_tables.len());

        for table in ordered_tables {
            let table_name = canonical_table_ident(&table.name).ok_or_else(|| {
                TpchError::Parse("CREATE TABLE statement missing canonical name".into())
            })?;

            let columns = build_columns(&table_name, &table)?;
            let column_names = columns.iter().map(|column| column.name.clone()).collect();

            let info = build_table_info(&table_name, &raw_tables);

            let schema = TableSchema {
                name: table_name.clone(),
                create_table: table,
                info,
                columns,
                column_names,
            };

            if tables_by_name.insert(table_name.clone(), schema).is_some() {
                return Err(TpchError::Parse(format!(
                    "duplicate table definition for {table_name}"
                )));
            }
            creation_order.push(table_name);
        }

        Ok(Self {
            schema_paths: paths,
            schema_name: DEFAULT_SCHEMA_NAME.to_string(),
            tables_by_name,
            creation_order,
        })
    }

    /// Build a toolkit using the default metadata bundled with the crate.
    pub fn with_default_paths() -> Result<Self> {
        Self::from_paths(SchemaPaths::default())
    }

    /// Return the schema name the toolkit targets (defaults to `TPCD`).
    pub fn schema_name(&self) -> &str {
        &self.schema_name
    }

    /// Expose the resolved metadata paths for callers that need to read query templates.
    pub fn schema_paths(&self) -> &SchemaPaths {
        &self.schema_paths
    }

    /// Install the TPC-H schema into the provided engine.
    pub fn install(&self, engine: &SqlEngine) -> Result<TpchSchema> {
        let create_schema_sql = format!("CREATE SCHEMA IF NOT EXISTS {};", self.schema_name);
        run_sql(engine, &create_schema_sql)?;
        let ddl_batch_sql = self.render_create_tables();
        run_sql(engine, &ddl_batch_sql)?;

        Ok(TpchSchema {
            schema_name: self.schema_name.clone(),
            tables: self.table_infos(),
        })
    }

    /// Load all TPC-H base tables using the provided generator scale factor and batch size.
    pub fn load_data(
        &self,
        engine: &SqlEngine,
        schema_name: &str,
        scale_factor: f64,
        batch_size: usize,
    ) -> Result<LoadSummary> {
        self.load_data_with_progress(engine, schema_name, scale_factor, batch_size, |_| {})
    }

    /// Load all base tables while emitting status updates through the provided callback.
    ///
    /// The callback receives a [`TableLoadEvent`] for each table when loading starts and
    /// again as batches progress and after the inserts finish.
    pub fn load_data_with_progress<F>(
        &self,
        engine: &SqlEngine,
        schema_name: &str,
        scale_factor: f64,
        batch_size: usize,
        mut on_progress: F,
    ) -> Result<LoadSummary>
    where
        F: FnMut(TableLoadEvent),
    {
        if batch_size == 0 {
            return Err(TpchError::Parse(
                "batch size must be greater than zero".into(),
            ));
        }
        let session = engine.session();
        let previous_mode = session.constraint_enforcement_mode();
        let changed_mode = previous_mode != ConstraintEnforcementMode::Deferred;
        if changed_mode {
            session.set_constraint_enforcement_mode(ConstraintEnforcementMode::Deferred);
        }

        let result = (|| -> Result<LoadSummary> {
            let mut tables = Vec::with_capacity(8);

            macro_rules! load_table_with_progress {
                ($collection:ident, $table_name:literal, $iter:expr) => {{
                    let expected = self.table_schema($table_name)?.info.base_rows;
                    on_progress(TableLoadEvent::Begin {
                        table: $table_name,
                        estimated_rows: Some(estimate_rows(expected, scale_factor)),
                    });
                    let started = Instant::now();
                    let mut last_report = started;
                    let summary = {
                        let iter = $iter;
                        let rows = iter.map(|row| row.to_string());
                        let mut forward = |rows_loaded: usize| {
                            let now = Instant::now();
                            let elapsed = now.duration_since(started);
                            let since_last = now.duration_since(last_report);
                            last_report = now;
                            on_progress(TableLoadEvent::Progress {
                                table: $table_name,
                                rows: rows_loaded,
                                elapsed,
                                since_last,
                            });
                        };
                        self.load_table_with_rows(
                            engine,
                            schema_name,
                            $table_name,
                            rows,
                            batch_size,
                            Some(&mut forward),
                        )?
                    };
                    on_progress(TableLoadEvent::Complete {
                        table: $table_name,
                        rows: summary.rows,
                        elapsed: started.elapsed(),
                    });
                    $collection.push(summary);
                }};
            }

            load_table_with_progress!(
                tables,
                "REGION",
                RegionGenerator::new(scale_factor, 1, 1).iter()
            );
            load_table_with_progress!(
                tables,
                "NATION",
                NationGenerator::new(scale_factor, 1, 1).iter()
            );
            load_table_with_progress!(
                tables,
                "SUPPLIER",
                SupplierGenerator::new(scale_factor, 1, 1).iter()
            );
            load_table_with_progress!(
                tables,
                "CUSTOMER",
                CustomerGenerator::new(scale_factor, 1, 1).iter()
            );
            load_table_with_progress!(
                tables,
                "PART",
                PartGenerator::new(scale_factor, 1, 1).iter()
            );
            load_table_with_progress!(
                tables,
                "PARTSUPP",
                PartSuppGenerator::new(scale_factor, 1, 1).iter()
            );
            load_table_with_progress!(
                tables,
                "ORDERS",
                OrderGenerator::new(scale_factor, 1, 1).iter()
            );
            load_table_with_progress!(
                tables,
                "LINEITEM",
                LineItemGenerator::new(scale_factor, 1, 1).iter()
            );

            Ok(LoadSummary { tables })
        })();

        if changed_mode {
            session.set_constraint_enforcement_mode(previous_mode);
        }

        result
    }

    /// Execute a TPC-H qualification run using the provided answer set configuration.
    ///
    /// # Errors
    ///
    /// Propagates [`TpchError::Parse`] when the qualification assets are missing or malformed
    /// and [`TpchError::Sql`] when query execution fails.
    pub fn run_qualification(
        &self,
        engine: &SqlEngine,
        schema_name: &str,
        options: &qualification::QualificationOptions,
    ) -> Result<Vec<qualification::QualificationReport>> {
        qualification::run_qualification(engine, &self.schema_paths, schema_name, options)
    }

    /// Look up the parsed table schema by canonical TPC-H name.
    ///
    /// # Errors
    ///
    /// Returns [`TpchError::Parse`] when the toolkit does not include the requested table.
    fn table_schema(&self, table_name: &str) -> Result<&TableSchema> {
        self.tables_by_name
            .get(table_name)
            .ok_or_else(|| TpchError::Parse(format!("unknown TPC-H table '{table_name}'")))
    }

    /// Serialize the ordered `CREATE TABLE` statements into an executable batch.
    fn render_create_tables(&self) -> String {
        let mut sql = String::new();
        for table_name in &self.creation_order {
            if let Some(table) = self.tables_by_name.get(table_name) {
                if !sql.is_empty() {
                    sql.push('\n');
                }
                let statement = Statement::CreateTable(table.create_table.clone());
                sql.push_str(&statement.to_string());
                sql.push_str(";\n");
            }
        }
        sql
    }

    /// Return table metadata in creation order so callers can display deterministic summaries.
    fn table_infos(&self) -> Vec<TpchTableInfo> {
        self.creation_order
            .iter()
            .filter_map(|name| self.tables_by_name.get(name))
            .map(|table| table.info.clone())
            .collect()
    }

    /// Load a single TPC-H table by streaming generated rows through batched inserts.
    fn load_table_with_rows<I, F>(
        &self,
        engine: &SqlEngine,
        schema_name: &str,
        table_name: &'static str,
        rows: I,
        batch_size: usize,
        progress: Option<&mut F>,
    ) -> Result<LoadTableSummary>
    where
        I: Iterator<Item = String>,
        F: FnMut(usize),
    {
        let table = self.table_schema(table_name)?;
        let row_count =
            self.load_table_from_lines(engine, schema_name, table, rows, batch_size, progress)?;
        Ok(LoadTableSummary {
            table: table_name,
            rows: row_count,
        })
    }

    /// Consume delimited rows, format them into SQL literals, and flush them in batches.
    fn load_table_from_lines<I, F>(
        &self,
        engine: &SqlEngine,
        schema_name: &str,
        table: &TableSchema,
        rows: I,
        batch_size: usize,
        mut progress: Option<&mut F>,
    ) -> Result<usize>
    where
        I: Iterator<Item = String>,
        F: FnMut(usize),
    {
        let canonical_name = format!("{}.{}", schema_name, table.name).to_ascii_lowercase();
        let cache_enabled = self.enable_fk_cache_for_table(engine, &canonical_name);

        let result = (|| -> Result<usize> {
            let mut batch: Vec<Vec<PlanValue>> = Vec::with_capacity(batch_size);
            let mut row_count = 0usize;

            for line in rows {
                if line.is_empty() {
                    continue;
                }
                let row_values = self.parse_row_values(table, &line)?;
                batch.push(row_values);
                row_count += 1;
                if batch.len() == batch_size {
                    self.flush_insert(engine, schema_name, table, &batch)?;
                    batch.clear();
                }
                if row_count.is_multiple_of(Self::PROGRESS_REPORT_INTERVAL)
                    && let Some(callback) = progress.as_mut()
                {
                    callback(row_count);
                }
            }

            if !batch.is_empty() {
                self.flush_insert(engine, schema_name, table, &batch)?;
            }

            Ok(row_count)
        })();

        if cache_enabled {
            self.clear_fk_cache_for_table(engine, &canonical_name);
        }

        result
    }

    /// Convert a raw delimited line into a typed value vector aligned with the table schema.
    ///
    /// # Errors
    ///
    /// Returns [`TpchError::Parse`] when the column count does not match the table schema.
    fn parse_row_values(&self, table: &TableSchema, line: &str) -> Result<Vec<PlanValue>> {
        let raw_fields: Vec<&str> = line.trim_end_matches('|').split('|').collect();
        if raw_fields.len() != table.columns.len() {
            return Err(TpchError::Parse(format!(
                "row '{}' does not match column definition (expected {}, found {})",
                line,
                table.columns.len(),
                raw_fields.len()
            )));
        }

        raw_fields
            .iter()
            .zip(table.columns.iter())
            .map(|(raw, column)| parse_column_value(column, raw))
            .collect()
    }

    /// Execute a batched INSERT using prepared plan rows.
    fn flush_insert(
        &self,
        engine: &SqlEngine,
        schema_name: &str,
        table: &TableSchema,
        rows: &[Vec<PlanValue>],
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let plan = InsertPlan {
            table: format!("{}.{}", schema_name, table.name),
            columns: table.column_names.clone(),
            source: InsertSource::Rows(rows.to_vec()),
            on_conflict: InsertConflictAction::None,
        };
        engine
            .session()
            .execute_insert_plan(plan)
            .map(|_| ())
            .map_err(TpchError::Sql)
    }

    fn enable_fk_cache_for_table(&self, engine: &SqlEngine, canonical_name: &str) -> bool {
        let context = engine.runtime_context();
        match context.table_catalog().table_id(canonical_name) {
            Some(table_id) => {
                context.enable_foreign_key_cache(table_id);
                true
            }
            None => {
                tracing::warn!(
                    target: "tpch-loader",
                    table = canonical_name,
                    "skipping foreign key cache enable; table id not found"
                );
                false
            }
        }
    }

    fn clear_fk_cache_for_table(&self, engine: &SqlEngine, canonical_name: &str) {
        let context = engine.runtime_context();
        if let Some(table_id) = context.table_catalog().table_id(canonical_name) {
            context.clear_foreign_key_cache(table_id);
        } else {
            tracing::warn!(
                target: "tpch-loader",
                table = canonical_name,
                "foreign key cache already dropped before cleanup"
            );
        }
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
/// This helper uses the default toolkit paths relative to the `llkv-tpch`
/// crate and is the easiest way to bootstrap a database for experimentation.
pub fn install_default_schema(engine: &SqlEngine) -> Result<TpchSchema> {
    let toolkit = TpchToolkit::with_default_paths()?;
    toolkit.install(engine)
}

/// Install the TPC-H schema using explicit metadata locations.
pub fn install_schema(engine: &SqlEngine, paths: &SchemaPaths) -> Result<TpchSchema> {
    let toolkit = TpchToolkit::from_paths(paths.clone())?;
    toolkit.install(engine)
}

/// Load the TPC-H data set using the default toolkit metadata paths.
pub fn load_tpch_data(
    engine: &SqlEngine,
    schema_name: &str,
    scale_factor: f64,
    batch_size: usize,
) -> Result<LoadSummary> {
    let toolkit = TpchToolkit::with_default_paths()?;
    toolkit.load_data(engine, schema_name, scale_factor, batch_size)
}

/// Load the TPC-H data set using a pre-initialized toolkit.
pub fn load_tpch_data_with_toolkit(
    toolkit: &TpchToolkit,
    engine: &SqlEngine,
    schema_name: &str,
    scale_factor: f64,
    batch_size: usize,
) -> Result<LoadSummary> {
    toolkit.load_data(engine, schema_name, scale_factor, batch_size)
}

/// Resolve the loader batch size using column-store write hints.
///
/// When `batch_override` is `None`, the column store's recommended insert batch rows
/// are used. Explicit overrides are clamped to the store's maximum to avoid building
/// enormous literal INSERT statements that would be split immediately during ingest.
pub fn resolve_loader_batch_size(engine: &SqlEngine, batch_override: Option<usize>) -> usize {
    let hints = engine.column_store_write_hints();
    let requested = batch_override
        .unwrap_or(hints.recommended_insert_batch_rows)
        .max(1);
    let resolved = hints.clamp_insert_batch_rows(requested);
    if let Some(explicit) = batch_override
        && resolved != explicit
    {
        tracing::warn!(
            target: "tpch-loader",
            requested = explicit,
            resolved,
            max = hints.max_insert_batch_rows,
            "clamped batch size override to column-store limit"
        );
    }
    resolved
}

/// Read a text file and wrap IO errors with the target path.
pub(crate) fn read_file(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|source| TpchError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use llkv::storage::MemPager;
    use std::sync::Arc;

    #[test]
    fn resolve_batch_size_defaults_to_hints() {
        let engine = SqlEngine::new(Arc::new(MemPager::default()));
        let hints = engine.column_store_write_hints();
        assert_eq!(
            resolve_loader_batch_size(&engine, None),
            hints.recommended_insert_batch_rows
        );
    }

    #[test]
    fn resolve_batch_size_clamps_override() {
        let engine = SqlEngine::new(Arc::new(MemPager::default()));
        let hints = engine.column_store_write_hints();
        let requested = hints.max_insert_batch_rows * 5;
        assert_eq!(
            resolve_loader_batch_size(&engine, Some(requested)),
            hints.max_insert_batch_rows
        );
    }
}

/// Execute a SQL batch against the provided engine, ignoring whitespace-only fragments.
fn run_sql(engine: &SqlEngine, sql: &str) -> Result<()> {
    if sql.trim().is_empty() {
        return Ok(());
    }
    engine.execute(sql).map(|_| ()).map_err(TpchError::Sql)
}

/// Scan `dss.h` and collect numeric `#define` entries keyed by macro name.
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

/// Parse decimal or hexadecimal numeric tokens supplied by the TPC-H headers.
fn parse_numeric_literal(token: &str) -> Option<i64> {
    if let Ok(value) = token.parse::<i64>() {
        return Some(value);
    }
    if let Some(hex) = token.strip_prefix("0x")
        && let Ok(value) = i64::from_str_radix(hex, 16)
    {
        return Some(value);
    }
    None
}

/// Parse the `tdef tdefs[]` manifest from `driver.c` into table metadata.
///
/// The helper resolves base-row expressions through the provided macro map so scale
/// factors match the upstream generator.
///
/// # Errors
///
/// Returns [`TpchError::Parse`] when the manifest layout is malformed or when row count
/// expressions cannot be evaluated.
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

/// Evaluate an expression describing baseline row counts for a table definition.
///
/// Returns an error string when the literal or macro cannot be resolved.
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

/// Parse the canonical TPC-H DDL and rewrite table names with the provided schema prefix.
///
/// The returned tuple contains the normalized `CREATE TABLE` statements alongside their
/// uppercase table identifiers for downstream indexing.
///
/// # Errors
///
/// Returns [`TpchError::Parse`] when the DDL fails to parse or omits expected identifiers.
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

/// Parse the constraint file `dss.ri` and bucket constraints by canonical table name.
///
/// CONNECT-specific directives are stripped so the generic sqlparser dialect can handle the
/// statements without vendor extensions.
///
/// # Errors
///
/// Returns [`TpchError::Parse`] when the file cannot be parsed or produces unexpected
/// statements.
fn parse_referential_integrity(ri_sql: &str) -> Result<HashMap<String, Vec<TableConstraint>>> {
    let cleaned = strip_tpch_connect_statements(ri_sql);
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

/// Attach parsed table constraints to the corresponding `CREATE TABLE` statements.
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

// -----------------------------------------------------------------------------
// TPC-H data loading helpers
// -----------------------------------------------------------------------------

/// Status updates emitted during table population.
#[derive(Debug, Clone, Copy)]
pub enum TableLoadEvent {
    Begin {
        table: &'static str,
        estimated_rows: Option<usize>,
    },
    Progress {
        table: &'static str,
        rows: usize,
        elapsed: Duration,
        since_last: Duration,
    },
    Complete {
        table: &'static str,
        rows: usize,
        elapsed: Duration,
    },
}

#[derive(Debug, Clone)]
pub struct LoadTableSummary {
    pub table: &'static str,
    pub rows: usize,
}

#[derive(Debug, Clone)]
pub struct LoadSummary {
    pub tables: Vec<LoadTableSummary>,
}

impl LoadSummary {
    /// Return the total number of rows loaded across all tables.
    pub fn total_rows(&self) -> usize {
        self.tables.iter().map(|entry| entry.rows).sum()
    }
}

fn estimate_rows(base_rows: u64, scale_factor: f64) -> usize {
    if base_rows == 0 {
        return 0;
    }
    let scaled = (base_rows as f64) * scale_factor;
    if !scaled.is_finite() {
        return 0;
    }
    let rounded = scaled.round();
    if scale_factor > 0.0 && rounded < 1.0 {
        1
    } else if rounded <= 0.0 {
        0
    } else {
        rounded as usize
    }
}

/// Derive ordered column metadata from a parsed `CREATE TABLE` statement.
///
/// The returned list mirrors the definition order and records how each column
/// should be converted into a [`PlanValue`] so bulk inserts can stream typed
/// rows directly into prepared plans.
///
/// # Errors
///
/// Returns [`TpchError::Parse`] when the definition omits a column list, which
/// signals that the upstream DDL parse drifted from expectations.
fn build_columns(table_name: &str, table: &CreateTable) -> Result<Vec<TableColumn>> {
    if table.columns.is_empty() {
        return Err(TpchError::Parse(format!(
            "table '{table_name}' does not declare any columns"
        )));
    }

    let mut columns = Vec::with_capacity(table.columns.len());
    for column_def in &table.columns {
        let name = column_def.name.value.clone();
        let family = classify_sql_data_type(&column_def.data_type).map_err(|err| {
            TpchError::Parse(format!(
                "unsupported SQL type for column {}.{}: {}",
                table_name, name, err
            ))
        })?;
        let value_kind = match family {
            SqlTypeFamily::String => ColumnValueKind::String,
            SqlTypeFamily::Integer => ColumnValueKind::Integer,
            SqlTypeFamily::Decimal { scale } => ColumnValueKind::Decimal { scale },
            SqlTypeFamily::Date32 => ColumnValueKind::Date32,
            SqlTypeFamily::Binary => {
                return Err(TpchError::Parse(format!(
                    "column {}.{} uses a binary type that the TPCH loader cannot parse",
                    table_name, name
                )));
            }
        };
        columns.push(TableColumn { name, value_kind });
    }

    Ok(columns)
}

/// Populate `TpchTableInfo` from parsed `driver.c` metadata, falling back to defaults.
///
/// When the upstream manifest omits a table, the helper synthesizes a file name and
/// description so callers still receive a consistent summary payload.
fn build_table_info(name: &str, raw_tables: &HashMap<String, RawTableDef>) -> TpchTableInfo {
    let file_key = format!("{}.tbl", name.to_ascii_lowercase());
    if let Some(raw) = raw_tables.get(&file_key) {
        TpchTableInfo {
            name: name.to_string(),
            file_name: raw.file_name.clone(),
            description: raw.description.clone(),
            base_rows: raw.base_rows,
        }
    } else {
        TpchTableInfo {
            name: name.to_string(),
            file_name: file_key.clone(),
            description: format!("{} table", name.to_ascii_lowercase()),
            base_rows: 0,
        }
    }
}

fn parse_column_value(column: &TableColumn, raw: &str) -> Result<PlanValue> {
    match column.value_kind {
        ColumnValueKind::String => Ok(PlanValue::String(raw.to_string())),
        ColumnValueKind::Integer => {
            if raw.is_empty() {
                return Err(TpchError::Parse(format!(
                    "missing integer value for column {}",
                    column.name
                )));
            }
            let value = raw.parse::<i64>().map_err(|err| {
                TpchError::Parse(format!(
                    "invalid integer literal '{}' for column {}: {}",
                    raw, column.name, err
                ))
            })?;
            Ok(PlanValue::Integer(value))
        }
        ColumnValueKind::Decimal { scale } => {
            if raw.is_empty() {
                return Err(TpchError::Parse(format!(
                    "missing decimal value for column {}",
                    column.name
                )));
            }
            let decimal = parse_decimal_literal(raw, scale, &column.name)?;
            Ok(PlanValue::Decimal(decimal))
        }
        ColumnValueKind::Date32 => {
            if raw.is_empty() {
                return Err(TpchError::Parse(format!(
                    "missing date value for column {}",
                    column.name
                )));
            }
            let days = parse_date32_literal(raw).map_err(|err| {
                TpchError::Parse(format!(
                    "invalid DATE literal '{}' for column {}: {}",
                    raw, column.name, err
                ))
            })?;
            Ok(PlanValue::Date32(days))
        }
    }
}

fn parse_decimal_literal(raw: &str, target_scale: i8, column_name: &str) -> Result<DecimalValue> {
    let (value, scale) = if let Some(dot) = raw.find('.') {
        let integer_part = &raw[..dot];
        let fractional = &raw[dot + 1..];
        let combined = format!("{}{}", integer_part, fractional);
        let parsed = combined.parse::<i128>().map_err(|err| {
            TpchError::Parse(format!(
                "invalid decimal literal '{}' for column {}: {}",
                raw, column_name, err
            ))
        })?;
        (parsed, fractional.len() as i8)
    } else {
        let parsed = raw.parse::<i128>().map_err(|err| {
            TpchError::Parse(format!(
                "invalid decimal literal '{}' for column {}: {}",
                raw, column_name, err
            ))
        })?;
        (parsed, 0)
    };

    let decimal = DecimalValue::new(value, scale).map_err(|err| {
        TpchError::Parse(format!(
            "invalid decimal literal '{}' for column {}: {}",
            raw, column_name, err
        ))
    })?;

    if scale == target_scale {
        Ok(decimal)
    } else {
        decimal.rescale(target_scale).map_err(|err| {
            TpchError::Parse(format!(
                "unable to rescale decimal literal '{}' for column {}: {}",
                raw, column_name, err
            ))
        })
    }
}
