//! Minimal SQL front-end that routes statements through llkv-fusion.
//!
//! This engine parses SQL statements so it can intercept `CREATE TABLE`
//! definitions, materialize them inside LLKV's [`TableCatalog`], and register
//! the resulting [`LlkvTableProvider`]s with DataFusion. All other statements
//! are delegated to a `SessionContext` that uses [`LlkvQueryPlanner`] for scan
//! planning, keeping LLKV storage as the backing store.

use std::convert::TryFrom;
use std::ops::ControlFlow;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::error::DataFusionError;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_catalog::CatalogProvider;
use datafusion_catalog::memory::MemorySchemaProvider;
use llkv_fusion::LlkvQueryPlanner;
use llkv_result::{Error, Result};
use llkv_storage::pager::BoxedPager;
use llkv_table::catalog::TableCatalog;
use sqlparser::ast::{
    AccessExpr, ColumnDef, ColumnOption, ColumnOptionDef, CreateTable, DataType as SqlDataType,
    Expr as SqlExpr, Ident, ObjectName, ObjectNamePart, ObjectType, Query, SchemaName, Statement,
    StructField, Subscript, Value, VisitMut, VisitorMut,
};
use sqlparser::dialect::{DuckDbDialect, GenericDialect, SQLiteDialect};
use sqlparser::parser::Parser;

/// Result of executing a SQL statement.
#[derive(Debug)]
pub enum SqlStatementResult {
    /// A query that produced Arrow record batches.
    Query { batches: Vec<RecordBatch> },
    /// A statement that did not return rows (affects rows_affected if known).
    Statement { rows_affected: usize },
}

/// SQL engine that intercepts DDL before delegating to DataFusion.
pub struct SqlEngine {
    ctx: SessionContext,
    catalog: Arc<TableCatalog<BoxedPager>>,
}

impl SqlEngine {
    /// Create a new SQL engine backed by the provided pager.
    pub fn new(pager: Arc<BoxedPager>) -> Result<Self> {
        let catalog = Arc::new(TableCatalog::open(pager)?);
        let session_state = SessionStateBuilder::new()
            .with_default_features()
            .with_query_planner(Arc::new(LlkvQueryPlanner::new(Arc::clone(&catalog))))
            .build();

        let ctx = SessionContext::new_with_state(session_state);
        let engine = Self { ctx, catalog };
        engine.register_existing_tables()?;
        Ok(engine)
    }

    /// Execute raw SQL text, parsing statements to intercept DDL.
    pub async fn execute(&self, sql: &str) -> Result<Vec<SqlStatementResult>> {
        let duck = DuckDbDialect {};
        let sqlite = SQLiteDialect {};
        let generic = GenericDialect {};
        let statements = Parser::parse_sql(&duck, sql)
            .or_else(|_| Parser::parse_sql(&sqlite, sql))
            .or_else(|_| Parser::parse_sql(&generic, sql))
            .map_err(|e| Error::Internal(format!("failed to parse SQL statement(s): {e}")))?;
        if statements.is_empty() {
            // Fallback to DataFusion directly if parsing produced no statements
            let result = self.execute_via_datafusion(sql).await?;
            Ok(vec![result])
        } else {
            let mut out = Vec::with_capacity(statements.len());
            for stmt in statements {
                out.push(self.execute_statement(stmt).await?);
            }
            Ok(out)
        }
    }

    async fn execute_statement(&self, stmt: Statement) -> Result<SqlStatementResult> {
        match stmt {
            Statement::CreateSchema {
                schema_name,
                if_not_exists,
                ..
            } => {
                self.handle_create_schema(schema_name, if_not_exists)?;
                Ok(SqlStatementResult::Statement { rows_affected: 0 })
            }
            Statement::CreateTable(create) => {
                self.handle_create_table(create).await?;
                Ok(SqlStatementResult::Statement { rows_affected: 0 })
            }
            Statement::Drop {
                object_type: ObjectType::Table,
                if_exists,
                names,
                ..
            } => {
                self.handle_drop_table(names, if_exists)?;
                Ok(SqlStatementResult::Statement { rows_affected: 0 })
            }
            Statement::Drop {
                object_type: ObjectType::Schema,
                if_exists,
                cascade,
                names,
                ..
            } => {
                self.handle_drop_schema(names, cascade, if_exists)?;
                Ok(SqlStatementResult::Statement { rows_affected: 0 })
            }
            mut other => {
                self.rewrite_statement_ast(&mut other)?;
                let sql = other.to_string();
                self.execute_via_datafusion(&sql).await
            }
        }
    }

    async fn execute_via_datafusion(&self, sql: &str) -> Result<SqlStatementResult> {
        let df = self
            .ctx
            .sql(sql)
            .await
            .map_err(|e| map_datafusion_error("DataFusion planning failed", e))?;
        let schema_is_empty = df.schema().fields().is_empty();

        let batches = df
            .collect()
            .await
            .map_err(|e| map_datafusion_error("DataFusion execution failed", e))?;

        if schema_is_empty {
            Ok(SqlStatementResult::Statement { rows_affected: 0 })
        } else {
            Ok(SqlStatementResult::Query { batches })
        }
    }

    async fn handle_create_table(&self, mut create: CreateTable) -> Result<()> {
        if !create.constraints.is_empty() {
            return Err(Error::Internal(
                "table constraints are not supported yet".into(),
            ));
        }

        let table_name = normalize_name(&create.name)?;
        if create.if_not_exists {
            if self.catalog.get_table(&table_name)?.is_some() {
                return Ok(());
            }
        }

        let columns = std::mem::take(&mut create.columns);
        if let Some(query) = create.query {
            return self
                .handle_create_table_as_select(table_name, columns, *query)
                .await;
        }

        let schema = build_schema(&columns)?;
        self.catalog
            .create_table(&table_name, Arc::clone(&schema))?;
        self.register_table(&table_name)?;
        Ok(())
    }

    async fn handle_create_table_as_select(
        &self,
        table_name: String,
        columns: Vec<ColumnDef>,
        query: Query,
    ) -> Result<()> {
        let query_sql = query.to_string();
        let df = self.ctx.sql(&query_sql).await.map_err(|e| {
            map_datafusion_error(&format!("CTAS planning failed for {table_name}"), e)
        })?;
        let df_schema = df.schema().clone();
        let batches = df.collect().await.map_err(|e| {
            map_datafusion_error(&format!("CTAS execution failed for {table_name}"), e)
        })?;
        let mut arrow_schema = Schema::from(df_schema);
        if !columns.is_empty() && columns.len() != arrow_schema.fields().len() {
            return Err(Error::Internal(format!(
                "CTAS column count mismatch: specified {} column(s) but query produced {}",
                columns.len(),
                arrow_schema.fields().len()
            )));
        }
        let actual_nullable: Vec<bool> = if let Some(first) = batches.first() {
            first
                .schema()
                .fields()
                .iter()
                .map(|f| f.is_nullable())
                .collect()
        } else {
            arrow_schema
                .fields()
                .iter()
                .map(|f| f.is_nullable())
                .collect()
        };
        let renamed_fields: Vec<Field> = arrow_schema
            .fields()
            .iter()
            .enumerate()
            .map(|(idx, field)| {
                let metadata = field.metadata().clone();
                let base_nullable = actual_nullable[idx];
                if let Some(col) = columns.get(idx) {
                    Field::new(
                        &col.name.value,
                        field.data_type().clone(),
                        column_nullable(col) && base_nullable,
                    )
                    .with_metadata(metadata)
                } else {
                    Field::new(field.name(), field.data_type().clone(), base_nullable)
                        .with_metadata(metadata)
                }
            })
            .collect();
        arrow_schema = Schema::new(renamed_fields);
        let schema_ref = Arc::new(arrow_schema);

        let mut builder = self
            .catalog
            .create_table(&table_name, Arc::clone(&schema_ref))?;
        let mut row_count = 0usize;
        for batch in &batches {
            builder.append_batch(batch).map_err(|e| {
                Error::Internal(format!(
                    "failed to append CTAS batch into {table_name}: {e}"
                ))
            })?;
            row_count += batch.num_rows();
        }
        let provider = builder.finish().map_err(|e| {
            Error::Internal(format!("failed to finalize CTAS table {table_name}: {e}"))
        })?;

        if row_count > 0 {
            let new_row_ids: Vec<u64> = (1..=row_count as u64).collect();
            self.catalog.update_table_rows(&table_name, new_row_ids)?;
        }

        self.ctx
            .register_table(&table_name, Arc::new(provider))
            .map_err(|e| Error::Internal(format!("failed to register table {table_name}: {e}")))?;
        Ok(())
    }

    fn register_existing_tables(&self) -> Result<()> {
        for name in self.catalog.list_tables() {
            self.register_table(&name)?;
        }
        Ok(())
    }

    fn register_table(&self, name: &str) -> Result<()> {
        self.ensure_schema_registered_for_table(name)?;
        if let Some(provider) = self.catalog.get_table(name)? {
            self.ctx
                .register_table(name, provider)
                .map_err(|e| Error::Internal(format!("failed to register table {name}: {e}")))?;
        }
        Ok(())
    }

    fn rewrite_statement_ast(&self, stmt: &mut Statement) -> Result<()> {
        let mut rewriter = SqlAstRewriter::new(&self.ctx);
        rewriter.rewrite(stmt)
    }

    fn ensure_schema_registered_for_table(&self, table_name: &str) -> Result<()> {
        if let Some(schema) = extract_schema_from_table_name(table_name) {
            let catalog = self.default_catalog()?;
            if catalog.schema(&schema).is_none() {
                catalog
                    .register_schema(&schema, Arc::new(MemorySchemaProvider::new()))
                    .map_err(|e| {
                        Error::Internal(format!(
                            "failed to register schema '{schema}' for table {table_name}: {e}"
                        ))
                    })?;
            }
        }
        Ok(())
    }

    fn default_catalog(&self) -> Result<Arc<dyn CatalogProvider>> {
        let state = self.ctx.state();
        let catalog_name = state.config().options().catalog.default_catalog.clone();
        self.ctx.catalog(&catalog_name).ok_or_else(|| {
            Error::Internal(format!("default catalog '{catalog_name}' is not available"))
        })
    }

    fn handle_create_schema(&self, schema_name: SchemaName, if_not_exists: bool) -> Result<()> {
        let name = match schema_name {
            SchemaName::Simple(object_name) => normalize_name(&object_name)?,
            other => {
                return Err(Error::Internal(format!(
                    "CREATE SCHEMA variant '{other}' is not supported yet"
                )));
            }
        };

        let catalog = self.default_catalog()?;
        if catalog.schema(&name).is_some() {
            if if_not_exists {
                return Ok(());
            }
            return Err(Error::Internal(format!("schema '{name}' already exists")));
        }
        catalog
            .register_schema(&name, Arc::new(MemorySchemaProvider::new()))
            .map_err(|e| Error::Internal(format!("failed to register schema '{name}': {e}")))?;
        Ok(())
    }

    fn handle_drop_schema(
        &self,
        names: Vec<ObjectName>,
        cascade: bool,
        if_exists: bool,
    ) -> Result<()> {
        for name in names {
            let schema = normalize_name(&name)?;
            self.drop_schema(&schema, cascade, if_exists)?;
        }
        Ok(())
    }

    fn handle_drop_table(&self, names: Vec<ObjectName>, if_exists: bool) -> Result<()> {
        for name in names {
            let table_name = normalize_name(&name)?;
            match self.catalog.drop_table(&table_name) {
                Ok(true) => {
                    let _ = self.ctx.deregister_table(&table_name);
                }
                Ok(false) => {
                    if !if_exists {
                        return Err(Error::Internal(format!(
                            "table '{table_name}' does not exist"
                        )));
                    }
                }
                Err(err) => return Err(err),
            }
        }
        Ok(())
    }

    fn drop_schema(&self, schema: &str, cascade: bool, if_exists: bool) -> Result<()> {
        let mut tables_to_drop: Vec<String> = self
            .catalog
            .list_tables()
            .into_iter()
            .filter(|table| match extract_schema_from_table_name(table) {
                Some(existing_schema) => existing_schema == schema,
                None => false,
            })
            .collect();

        if !tables_to_drop.is_empty() && !cascade {
            return Err(Error::Internal(format!(
                "schema '{schema}' is not empty; use DROP SCHEMA ... CASCADE"
            )));
        }

        for table in tables_to_drop.drain(..) {
            if self.catalog.drop_table(&table)? {
                self.ctx.deregister_table(&table).map_err(|e| {
                    Error::Internal(format!("failed to deregister table {table}: {e}"))
                })?;
            }
        }

        let catalog = self.default_catalog()?;
        match catalog.deregister_schema(schema, cascade) {
            Ok(Some(_)) => Ok(()),
            Ok(None) => {
                if if_exists {
                    Ok(())
                } else {
                    Err(Error::Internal(format!("schema '{schema}' does not exist")))
                }
            }
            Err(e) => Err(Error::Internal(format!(
                "failed to drop schema '{schema}': {e}"
            ))),
        }
    }
}

fn normalize_name(name: &ObjectName) -> Result<String> {
    if name.0.is_empty() {
        return Err(Error::Internal("table name missing".into()));
    }
    let mut parts = Vec::with_capacity(name.0.len());
    for part in &name.0 {
        let value = match part {
            ObjectNamePart::Identifier(ident) => ident.value.clone(),
            ObjectNamePart::Function(func) => func.to_string(),
        };
        parts.push(value);
    }
    Ok(parts.join("."))
}

fn extract_schema_from_table_name(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    Some(parts[parts.len() - 2].to_string())
}

fn map_datafusion_error(prefix: &str, err: DataFusionError) -> Error {
    let msg = err.to_string();
    let mut decorated = msg.clone();
    if msg.to_lowercase().contains("not found") {
        decorated = format!("{msg}; does not exist");
    }
    Error::Internal(format!("{prefix}: {decorated}"))
}

struct SqlAstRewriter<'a> {
    ctx: &'a SessionContext,
}

impl<'a> SqlAstRewriter<'a> {
    fn new(ctx: &'a SessionContext) -> Self {
        Self { ctx }
    }

    fn rewrite(&mut self, stmt: &mut Statement) -> Result<()> {
        match stmt.visit(self) {
            ControlFlow::Continue(()) => Ok(()),
            ControlFlow::Break(err) => Err(err),
        }
    }

    fn table_exists(&self, parts: &[String]) -> Result<bool> {
        if parts.is_empty() {
            return Ok(false);
        }
        let name = parts.join(".");
        self.ctx.table_exist(&name).map_err(|e| {
            Error::Internal(format!(
                "failed to resolve table reference '{name}' while rewriting struct access: {e}"
            ))
        })
    }

    fn rewrite_compound_identifier(&self, ids: &[Ident]) -> Result<Option<SqlExpr>> {
        if ids.len() < 3 {
            return Ok(None);
        }
        let values: Vec<String> = ids.iter().map(|ident| ident.value.clone()).collect();
        if values.len() < 3 {
            return Ok(None);
        }
        let max_prefix = (values.len() - 2).min(3);
        for prefix_len in (1..=max_prefix).rev() {
            if !self.table_exists(&values[..prefix_len])? {
                continue;
            }
            let remaining = &values[prefix_len..];
            if remaining.len() < 2 {
                continue;
            }
            let nested_fields = &remaining[1..];
            if nested_fields.is_empty() {
                continue;
            }

            let root_idents: Vec<Ident> = ids[..=prefix_len].to_vec();
            let mut access_chain = Vec::with_capacity(nested_fields.len());
            for field_name in nested_fields {
                let literal = SqlExpr::Value(Value::SingleQuotedString(field_name.clone()).into());
                access_chain.push(AccessExpr::Subscript(Subscript::Index { index: literal }));
            }

            return Ok(Some(SqlExpr::CompoundFieldAccess {
                root: Box::new(SqlExpr::CompoundIdentifier(root_idents)),
                access_chain,
            }));
        }

        Ok(None)
    }
}

impl<'a> VisitorMut for SqlAstRewriter<'a> {
    type Break = Error;

    fn post_visit_expr(&mut self, expr: &mut SqlExpr) -> ControlFlow<Self::Break> {
        match expr {
            SqlExpr::CompoundIdentifier(ids) => match self.rewrite_compound_identifier(ids) {
                Ok(Some(new_expr)) => {
                    *expr = new_expr;
                }
                Ok(None) => {}
                Err(err) => return ControlFlow::Break(err),
            },
            SqlExpr::InList { list, negated, .. } if list.is_empty() => {
                let value = Value::Boolean(*negated);
                *expr = SqlExpr::Value(value.into());
            }
            _ => {}
        }
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::BooleanArray;
    use llkv_storage::pager::{BoxedPager, MemPager};
    use sqlparser::ast::{SelectItem, SetExpr};
    use sqlparser::dialect::SQLiteDialect;
    use sqlparser::parser::Parser;
    use tokio::runtime::Runtime;

    #[test]
    fn rewrite_empty_in_list_to_boolean_literal() {
        let dialect = SQLiteDialect {};
        let mut stmt = Parser::parse_sql(&dialect, "SELECT 1 IN ()")
            .expect("parse sql")
            .remove(0);
        let ctx = SessionContext::new();
        let mut rewriter = SqlAstRewriter::new(&ctx);
        rewriter.rewrite(&mut stmt).expect("rewrite");
        let Statement::Query(query) = stmt else {
            panic!("expected query");
        };
        let SqlExpr::Value(value_with_span) = (match query.body.as_ref() {
            SetExpr::Select(select) => match &select.projection[0] {
                SelectItem::UnnamedExpr(expr) => expr.clone(),
                other => panic!("unexpected projection: {other:?}"),
            },
            other => panic!("unexpected body: {other:?}"),
        }) else {
            panic!("expression not rewritten to literal");
        };
        assert_eq!(value_with_span.value, Value::Boolean(false));
    }

    #[test]
    fn execute_empty_in_list_returns_false() {
        let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
        let engine = SqlEngine::new(pager).expect("sql engine");
        let rt = Runtime::new().expect("runtime");
        let mut results = rt
            .block_on(engine.execute("SELECT 1 IN ()"))
            .expect("execute");
        assert_eq!(results.len(), 1);
        match results.pop().unwrap() {
            SqlStatementResult::Query { batches } => {
                assert_eq!(batches.len(), 1);
                let batch = &batches[0];
                assert_eq!(batch.num_columns(), 1);
                assert_eq!(batch.num_rows(), 1);
                let array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("boolean array");
                assert!(!array.value(0));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn non_empty_in_list_behaves_like_sqlite() {
        let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
        let engine = SqlEngine::new(pager).expect("sql engine");
        let rt = Runtime::new().expect("runtime");
        rt.block_on(async {
            engine
                .execute("CREATE TABLE t1(x INTEGER, y TEXT);")
                .await
                .expect("create table");
            engine
                .execute("INSERT INTO t1 VALUES (1, 'true');")
                .await
                .expect("insert row");
        });
        let mut in_results = rt
            .block_on(engine.execute("SELECT 1 FROM t1 WHERE 1 IN (2)"))
            .expect("execute in");
        assert_eq!(in_results.len(), 1);
        match in_results.pop().unwrap() {
            SqlStatementResult::Query { batches } => {
                assert!(batches.iter().all(|b| b.num_rows() == 0));
            }
            other => panic!("unexpected IN result: {other:?}"),
        }

        let mut not_in_results = rt
            .block_on(engine.execute("SELECT 1 FROM t1 WHERE 1 NOT IN (2)"))
            .expect("execute not in");
        assert_eq!(not_in_results.len(), 1);
        match not_in_results.pop().unwrap() {
            SqlStatementResult::Query { batches } => {
                assert_eq!(batches.len(), 1);
                assert_eq!(batches[0].num_rows(), 1);
            }
            other => panic!("unexpected NOT IN result: {other:?}"),
        }
    }
}

fn build_schema(columns: &[ColumnDef]) -> Result<SchemaRef> {
    if columns.is_empty() {
        return Err(Error::Internal(
            "CREATE TABLE must define at least one column".into(),
        ));
    }

    let mut fields = Vec::with_capacity(columns.len());
    for column in columns {
        let data_type = map_sql_type(&column.data_type)?;
        let nullable = column_nullable(column);
        fields.push(Field::new(&column.name.value, data_type, nullable));
    }

    Ok(Arc::new(Schema::new(fields)))
}

fn column_nullable(column: &ColumnDef) -> bool {
    !column.options.iter().any(|option| match option {
        ColumnOptionDef {
            option: ColumnOption::NotNull,
            ..
        } => true,
        ColumnOptionDef {
            option: ColumnOption::Unique {
                is_primary: true, ..
            },
            ..
        } => true,
        _ => false,
    })
}

fn map_sql_type(sql_type: &SqlDataType) -> Result<DataType> {
    if is_signed_integer_type(sql_type) {
        return Ok(DataType::Int64);
    }
    if is_unsigned_integer_type(sql_type) {
        return Ok(DataType::UInt64);
    }

    use SqlDataType::*;
    let dt = match sql_type {
        Boolean | Bool => DataType::Boolean,
        Float(_) | FloatUnsigned(_) | Float4 | Float32 | Float64 | Real | RealUnsigned => {
            DataType::Float32
        }
        Double(_) | DoubleUnsigned(_) | DoublePrecision | DoublePrecisionUnsigned | Float8 => {
            DataType::Float64
        }
        Decimal(info)
        | DecimalUnsigned(info)
        | Numeric(info)
        | BigNumeric(info)
        | BigDecimal(info)
        | Dec(info)
        | DecUnsigned(info) => {
            let (precision, scale) = map_exact_number(info);
            DataType::Decimal128(precision, scale)
        }
        Character(_)
        | Char(_)
        | CharacterVarying(_)
        | CharVarying(_)
        | Varchar(_)
        | Nvarchar(_)
        | Text
        | TinyText
        | MediumText
        | LongText
        | String(_)
        | FixedString(_)
        | Clob(_)
        | CharacterLargeObject(_)
        | CharLargeObject(_)
        | Bytes(_)
        | JSON
        | JSONB => DataType::Utf8,
        Binary(_) | Varbinary(_) | Blob(_) | TinyBlob | MediumBlob | LongBlob => DataType::Binary,
        Date | Date32 => DataType::Date32,
        Timestamp(_, _) | TimestampNtz | Datetime(_) | Datetime64(_, _) => {
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None)
        }
        Struct(fields, _) => DataType::Struct(map_struct_fields(fields)?.into()),
        Custom(name, _) => {
            let ident = name.to_string();
            if ident.eq_ignore_ascii_case("row") || ident.eq_ignore_ascii_case("struct") {
                map_custom_struct_type(sql_type)?
            } else {
                return Err(Error::Internal(format!(
                    "unsupported SQL data type: {sql_type:?}"
                )));
            }
        }
        _ => {
            return Err(Error::Internal(format!(
                "unsupported SQL data type: {sql_type:?}"
            )));
        }
    };
    Ok(dt)
}

fn is_signed_integer_type(data_type: &SqlDataType) -> bool {
    use SqlDataType::*;
    matches!(
        data_type,
        TinyInt(_)
            | Int2(_)
            | SmallInt(_)
            | MediumInt(_)
            | Int(_)
            | Int4(_)
            | Int8(_)
            | Integer(_)
            | BigInt(_)
            | Int16
            | Int32
            | Int64
            | Int128
            | Int256
            | HugeInt
            | Signed
            | SignedInteger
    )
}

fn is_unsigned_integer_type(data_type: &SqlDataType) -> bool {
    use SqlDataType::*;
    matches!(
        data_type,
        TinyIntUnsigned(_)
            | UTinyInt
            | Int2Unsigned(_)
            | SmallIntUnsigned(_)
            | USmallInt
            | MediumIntUnsigned(_)
            | IntUnsigned(_)
            | Int4Unsigned(_)
            | IntegerUnsigned(_)
            | Int8Unsigned(_)
            | BigIntUnsigned(_)
            | UBigInt
            | Unsigned
            | UnsignedInteger
            | UInt8
            | UInt16
            | UInt32
            | UInt64
            | UInt128
            | UInt256
            | UHugeInt
    )
}

fn map_exact_number(info: &sqlparser::ast::ExactNumberInfo) -> (u8, i8) {
    use sqlparser::ast::ExactNumberInfo::*;
    match info {
        None => (38, 0),
        Precision(p) => (u8::try_from(*p).unwrap_or(38), 0),
        PrecisionAndScale(p, s) => (
            u8::try_from(*p).unwrap_or(38),
            i8::try_from(*s).unwrap_or(0),
        ),
    }
}

fn map_struct_fields(fields: &[StructField]) -> Result<Vec<Field>> {
    if fields.is_empty() {
        return Err(Error::Internal(
            "STRUCT/ROW types must define at least one field".into(),
        ));
    }

    let mut out = Vec::with_capacity(fields.len());
    for (idx, field) in fields.iter().enumerate() {
        let name = field
            .field_name
            .as_ref()
            .map(|ident| ident.value.clone())
            .unwrap_or_else(|| format!("field_{:02}", idx + 1));
        let data_type = map_sql_type(&field.field_type)?;
        out.push(Field::new(&name, data_type, true));
    }
    Ok(out)
}

fn map_custom_struct_type(sql_type: &SqlDataType) -> Result<DataType> {
    let modifiers = match sql_type {
        SqlDataType::Custom(_, values) => values,
        _ => return Err(Error::Internal("ROW/STRUCT type modifiers missing".into())),
    };
    if modifiers.len() < 2 || modifiers.len() % 2 != 0 {
        return Err(Error::Internal(
            "ROW/STRUCT types must define name/type pairs".into(),
        ));
    }

    let mut fields = Vec::new();
    for chunk in modifiers.chunks(2) {
        let name = chunk[0].trim();
        let ty_sql = chunk[1].trim();
        if name.is_empty() || ty_sql.is_empty() {
            return Err(Error::Internal(
                "ROW/STRUCT type modifiers must include non-empty name/type".into(),
            ));
        }
        let parsed_type = parse_type_fragment(ty_sql.to_string())?;
        let data_type = map_sql_type(&parsed_type)?;
        fields.push(Field::new(name, data_type, true));
    }

    Ok(DataType::Struct(fields.into()))
}

fn parse_type_fragment(type_sql: String) -> Result<SqlDataType> {
    let ddl = format!("CREATE TABLE __temp(__col {type_sql})");
    let duck = DuckDbDialect {};
    let generic = GenericDialect {};
    let stmts = Parser::parse_sql(&duck, &ddl).or_else(|_| Parser::parse_sql(&generic, &ddl));
    let stmts =
        stmts.map_err(|e| Error::Internal(format!("failed to parse ROW field type: {e}")))?;
    if let Some(Statement::CreateTable(table)) = stmts.into_iter().next() {
        if let Some(column) = table.columns.first() {
            return Ok(column.data_type.clone());
        }
    }
    Err(Error::Internal(
        "failed to parse ROW/STRUCT field type".into(),
    ))
}
