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
use arrow::array::{Array, ArrayRef};
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
    AccessExpr, Assignment, ColumnDef, ColumnOption, ColumnOptionDef, CreateIndex, CreateTable,
    DataType as SqlDataType, Expr as SqlExpr, Ident, ObjectName, ObjectNamePart, ObjectType, Query,
    SchemaName, Statement, StructField, Subscript, TableWithJoins, Value, VisitMut, VisitorMut,
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

    /// Get the underlying table catalog
    pub fn catalog(&self) -> &Arc<TableCatalog<BoxedPager>> {
        &self.catalog
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
            Statement::CreateIndex(create_index) => {
                self.handle_create_index(create_index)?;
                Ok(SqlStatementResult::Statement { rows_affected: 0 })
            }
            Statement::Update {
                table,
                assignments,
                from,
                selection,
                returning,
                or,
                limit,
            } => {
                if from.is_some() {
                    return Err(Error::Internal("UPDATE with FROM clause is not supported yet".into()));
                }
                if returning.is_some() {
                    return Err(Error::Internal("UPDATE with RETURNING clause is not supported yet".into()));
                }
                if or.is_some() {
                    return Err(Error::Internal("UPDATE with OR clause is not supported yet".into()));
                }
                if limit.is_some() {
                    return Err(Error::Internal("UPDATE with LIMIT clause is not supported yet".into()));
                }
                let rows_affected = self.handle_update(table, assignments, selection).await?;
                Ok(SqlStatementResult::Statement { rows_affected })
            }
            Statement::Delete(delete) => {
                if delete.using.is_some() {
                    return Err(Error::Internal("DELETE with USING clause is not supported yet".into()));
                }
                if delete.returning.is_some() {
                    return Err(Error::Internal("DELETE with RETURNING clause is not supported yet".into()));
                }
                if !delete.order_by.is_empty() {
                    return Err(Error::Internal("DELETE with ORDER BY is not supported yet".into()));
                }
                if delete.limit.is_some() {
                    return Err(Error::Internal("DELETE with LIMIT is not supported yet".into()));
                }
                if !delete.tables.is_empty() {
                    return Err(Error::Internal("DELETE with multiple tables is not supported yet".into()));
                }
                let rows_affected = self.handle_delete(delete.from, delete.selection).await?;
                Ok(SqlStatementResult::Statement { rows_affected })
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
            let row_ids = provider.get_row_ids();
            eprintln!("[DEBUG register_table] Registering table '{}' with row_count={}, row_ids={:?}", 
                name, provider.row_count(), row_ids);
            self.ctx
                .register_table(name, provider)
                .map_err(|e| Error::Internal(format!("failed to register table {name}: {e}")))?;
            eprintln!("[DEBUG register_table] Successfully registered '{}'", name);
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

    async fn handle_update(
        &self,
        table: TableWithJoins,
        assignments: Vec<Assignment>,
        selection: Option<SqlExpr>,
    ) -> Result<usize> {
        use arrow::array::*;
        
        let table_name = match &table.relation {
            sqlparser::ast::TableFactor::Table { name, .. } => normalize_name(name)?,
            _ => {
                return Err(Error::Internal("UPDATE requires a simple table name".into()));
            }
        };

        let provider = self.catalog.get_table(&table_name)?
            .ok_or_else(|| Error::Internal(format!("table '{table_name}' does not exist")))?;

        let ingest_schema = provider.ingest_schema();
        
        eprintln!("[DEBUG] Table '{}' ingest schema:", table_name);
        for (i, field) in ingest_schema.fields().iter().enumerate() {
            eprintln!("[DEBUG]   Field {}: name='{}', metadata={:?}", 
                i, field.name(), field.metadata());
        }

        if !table.joins.is_empty() {
            return Err(Error::Internal("UPDATE with JOIN is not supported yet".into()));
        }

        if assignments.is_empty() {
            return Err(Error::Internal("UPDATE requires at least one SET assignment".into()));
        }

        let mut assignment_map: std::collections::HashMap<String, &SqlExpr> = std::collections::HashMap::new();
        for assignment in &assignments {
            let col_name = match &assignment.target {
                sqlparser::ast::AssignmentTarget::ColumnName(name) => {
                    let parts: Vec<String> = name.0.iter().map(|part| {
                        match part {
                            ObjectNamePart::Identifier(ident) => ident.value.clone(),
                            ObjectNamePart::Function(func) => func.to_string(),
                        }
                    }).collect();
                    parts.join(".").to_lowercase()
                }
                _ => return Err(Error::Internal("Complex assignment targets not supported yet".into())),
            };
            
            // Validate column exists in ingest schema (skip rowid column at index 0)
            let col_exists = ingest_schema.fields()
                .iter()
                .skip(1)
                .any(|f| f.name().to_lowercase() == col_name);
            
            if !col_exists {
                return Err(Error::Internal(format!("Column '{}' does not exist in table '{}'", col_name, table_name)));
            }
            
            assignment_map.insert(col_name, &assignment.value);
        }

        let select_sql = if let Some(ref where_expr) = selection {
            format!("SELECT * FROM {table_name} WHERE {where_expr}")
        } else {
            format!("SELECT * FROM {table_name}")
        };

        let df = self.ctx.sql(&select_sql).await.map_err(|e| {
            map_datafusion_error("UPDATE SELECT failed", e)
        })?;

        let batches = df.collect().await.map_err(|e| {
            map_datafusion_error("UPDATE SELECT collection failed", e)
        })?;

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        if total_rows == 0 {
            return Ok(0);
        }

        let all_row_ids = provider.get_row_ids();
        
        let row_ids = if selection.is_some() {
            // WHERE clause: map result rows to their actual row IDs
            // Since we don't have rowid in the result, use first N row IDs
            // This is still a limitation - proper implementation needs rowid in query
            all_row_ids.into_iter().take(total_rows).collect::<Vec<_>>()
        } else {
            // No WHERE: update all rows
            all_row_ids
        };

        let mut offset = 0;
        for (batch_idx, batch) in batches.iter().enumerate() {
            let query_schema = batch.schema();
            let batch_size = batch.num_rows();
            
            eprintln!("[DEBUG] Processing batch {}: {} rows", batch_idx, batch_size);
            
            let mut rowid_builder = UInt64Builder::with_capacity(batch_size);
            for i in 0..batch_size {
                let rid = row_ids[offset + i];
                eprintln!("[DEBUG]   Row {} in batch -> rowid {}", i, rid);
                rowid_builder.append_value(rid);
            }
            let rowid_array: ArrayRef = Arc::new(rowid_builder.finish());
            
            let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(ingest_schema.fields().len());
            new_columns.push(rowid_array);
            
            for ingest_idx in 1..ingest_schema.fields().len() {
                let ingest_field = ingest_schema.field(ingest_idx);
                let col_name_lower = ingest_field.name().to_lowercase();
                
                let query_idx = query_schema.index_of(ingest_field.name())
                    .map_err(|_| Error::Internal(format!("Column '{}' not found in query result", ingest_field.name())))?;
                
                if let Some(value_expr) = assignment_map.get(&col_name_lower) {
                    let new_column = self.evaluate_update_expression(
                        value_expr,
                        ingest_field.data_type(),
                        batch.num_rows()
                    )?;
                    eprintln!("[DEBUG] Updating column '{}' (field_id from metadata: {:?})", 
                        ingest_field.name(), 
                        ingest_field.metadata().get("field_id"));
                    new_columns.push(new_column);
                } else {
                    new_columns.push(batch.column(query_idx).clone());
                }
            }
            
            let updated_batch = RecordBatch::try_new(Arc::clone(&ingest_schema), new_columns)
                .map_err(|e| Error::Internal(format!("Failed to create updated batch: {e}")))?;
            
            eprintln!("[DEBUG] Batch schema fields:");
            for (i, field) in updated_batch.schema().fields().iter().enumerate() {
                eprintln!("[DEBUG]   Field {}: name='{}', metadata={:?}", 
                    i, field.name(), field.metadata());
            }
            eprintln!("[DEBUG] Batch has {} rows", updated_batch.num_rows());
            eprintln!("[DEBUG] Batch columns:");
            for (i, col) in updated_batch.columns().iter().enumerate() {
                eprintln!("[DEBUG]   Column {}: {:?}", i, col);
            }
            
            self.catalog.store().append(&updated_batch)?;
            
            offset += batch_size;
        }
        
        eprintln!("[DEBUG] UPDATE wrote {} rows", total_rows);

        // Re-register the table so DataFusion uses a fresh provider with updated data
        // This is necessary because each get_table() call creates a new provider instance
        // with its own row_ids snapshot
        eprintln!("[DEBUG] Deregistering table '{}'", table_name);
        let dereg_result = self.ctx.deregister_table(&table_name);
        eprintln!("[DEBUG] Deregister result: {:?}", dereg_result);
        
        eprintln!("[DEBUG] Re-registering table '{}'", table_name);
        self.register_table(&table_name)?;
        eprintln!("[DEBUG] Re-registration complete");

        Ok(total_rows)
    }

    async fn handle_delete(
        &self,
        from: sqlparser::ast::FromTable,
        selection: Option<SqlExpr>,
    ) -> Result<usize> {
        let tables = match from {
            sqlparser::ast::FromTable::WithFromKeyword(tables) => tables,
            sqlparser::ast::FromTable::WithoutKeyword(tables) => tables,
        };

        if tables.is_empty() {
            return Err(Error::Internal("DELETE requires a table".into()));
        }
        if tables.len() > 1 {
            return Err(Error::Internal("DELETE from multiple tables not supported yet".into()));
        }

        let table_name = match &tables[0].relation {
            sqlparser::ast::TableFactor::Table { name, .. } => normalize_name(name)?,
            _ => {
                return Err(Error::Internal("DELETE requires a simple table name".into()));
            }
        };

        let _table = self.catalog.get_table(&table_name)?
            .ok_or_else(|| Error::Internal(format!("table '{table_name}' does not exist")))?;

        if !tables[0].joins.is_empty() {
            return Err(Error::Internal("DELETE with JOIN is not supported yet".into()));
        }

        // Delegate to DataFusion which will handle it through the planner
        let delete_sql = if let Some(ref where_expr) = selection {
            format!("DELETE FROM {table_name} WHERE {where_expr}")
        } else {
            format!("DELETE FROM {table_name}")
        };

        let result = self.execute_via_datafusion(&delete_sql).await?;
        
        match result {
            SqlStatementResult::Statement { rows_affected } => Ok(rows_affected),
            _ => Ok(0),
        }
    }

    /// Evaluate a simple UPDATE SET expression into an Arrow array.
    ///
    /// Currently supports literals and NULL. Column references, arithmetic
    /// expressions, and subqueries require implementing proper expression
    /// evaluation against the existing row data.
    fn evaluate_update_expression(
        &self,
        expr: &SqlExpr,
        data_type: &DataType,
        num_rows: usize,
    ) -> Result<ArrayRef> {
        use arrow::array::*;
        
        match expr {
            SqlExpr::Value(value_with_span) => {
                let value = &value_with_span.value;
                match (value, data_type) {
                    (Value::Number(n, _), DataType::Int64) => {
                        let val = n.parse::<i64>()
                            .map_err(|e| Error::Internal(format!("Failed to parse integer: {e}")))?;
                        let mut builder = Int64Builder::with_capacity(num_rows);
                        for _ in 0..num_rows {
                            builder.append_value(val);
                        }
                        Ok(Arc::new(builder.finish()))
                    }
                    (Value::Number(n, _), DataType::Float64) => {
                        let val = n.parse::<f64>()
                            .map_err(|e| Error::Internal(format!("Failed to parse float: {e}")))?;
                        let mut builder = Float64Builder::with_capacity(num_rows);
                        for _ in 0..num_rows {
                            builder.append_value(val);
                        }
                        Ok(Arc::new(builder.finish()))
                    }
                    (Value::SingleQuotedString(s) | Value::DoubleQuotedString(s), DataType::Utf8) => {
                        let mut builder = StringBuilder::with_capacity(num_rows, s.len() * num_rows);
                        for _ in 0..num_rows {
                            builder.append_value(s);
                        }
                        Ok(Arc::new(builder.finish()))
                    }
                    (Value::Boolean(b), DataType::Boolean) => {
                        let mut builder = BooleanBuilder::with_capacity(num_rows);
                        for _ in 0..num_rows {
                            builder.append_value(*b);
                        }
                        Ok(Arc::new(builder.finish()))
                    }
                    (Value::Null, _) => {
                        Ok(arrow::array::new_null_array(data_type, num_rows))
                    }
                    _ => Err(Error::Internal(format!(
                        "Unsupported UPDATE value type: {:?} for column type {:?}",
                        value, data_type
                    ))),
                }
            }
            SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
                Err(Error::Internal("UPDATE with column expressions not yet supported; use literal values".into()))
            }
            _ => {
                Err(Error::Internal(format!("Unsupported UPDATE expression: {}", expr)))
            }
        }
    }

    fn handle_create_index(&self, create_index: CreateIndex) -> Result<()> {
        let table_name = normalize_name(&create_index.table_name)?;

        // Verify table exists
        if self.catalog.get_table(&table_name)?.is_none() {
            return Err(Error::Internal(format!(
                "table '{table_name}' does not exist"
            )));
        }

        // Extract index name
        let index_name = create_index
            .name
            .as_ref()
            .map(|name| normalize_name(name))
            .transpose()?;

        // Validate columns
        if create_index.columns.is_empty() {
            return Err(Error::Internal(
                "CREATE INDEX requires at least one column".into(),
            ));
        }

        // Convert columns to simple column names
        // Note: sort order and nulls_first are not currently supported by the catalog layer
        let mut column_names = Vec::with_capacity(create_index.columns.len());

        for index_column in &create_index.columns {
            // Extract column name from IndexColumn's OrderByExpr
            let col_name = match &index_column.column.expr {
                SqlExpr::Identifier(ident) => ident.value.clone(),
                SqlExpr::CompoundIdentifier(parts) => {
                    if parts.len() == 1 {
                        parts[0].value.clone()
                    } else {
                        // Take the last part as the column name
                        parts.last().unwrap().value.clone()
                    }
                }
                other => {
                    return Err(Error::Internal(format!(
                        "unsupported column expression in CREATE INDEX: {other}"
                    )));
                }
            };

            column_names.push(col_name);
        }

        // Create indexes on the table's column store
        // For now, we support single-column indexes only for actual index creation
        // Multi-column indexes are registered in catalog but not materialized
        if column_names.len() == 1 {
            let column_name = &column_names[0];
            self.catalog.create_index(
                &table_name,
                column_name,
                index_name.as_deref(),
                create_index.unique,
                create_index.if_not_exists,
            )?;
        } else {
            // Multi-column indexes: register in catalog metadata only
            // The catalog's index management will handle this
            let index_name = index_name.ok_or_else(|| {
                Error::Internal("Multi-column CREATE INDEX requires an explicit index name".into())
            })?;

            self.catalog.create_multi_column_index(
                &table_name,
                &column_names,
                &index_name,
                create_index.unique,
                create_index.if_not_exists,
            )?;
        }

        Ok(())
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

/// Format an Arrow array value at the given index as a SQL literal string.
fn format_arrow_value(array: &dyn Array, row_idx: usize) -> String {
    use arrow::array::*;
    use arrow::datatypes::DataType;

    if array.is_null(row_idx) {
        return "NULL".to_string();
    }

    match array.data_type() {
        DataType::Int8 => {
            let arr = array.as_any().downcast_ref::<Int8Array>().unwrap();
            arr.value(row_idx).to_string()
        }
        DataType::Int16 => {
            let arr = array.as_any().downcast_ref::<Int16Array>().unwrap();
            arr.value(row_idx).to_string()
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            arr.value(row_idx).to_string()
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            arr.value(row_idx).to_string()
        }
        DataType::UInt8 => {
            let arr = array.as_any().downcast_ref::<UInt8Array>().unwrap();
            arr.value(row_idx).to_string()
        }
        DataType::UInt16 => {
            let arr = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            arr.value(row_idx).to_string()
        }
        DataType::UInt32 => {
            let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            arr.value(row_idx).to_string()
        }
        DataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            arr.value(row_idx).to_string()
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            arr.value(row_idx).to_string()
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            arr.value(row_idx).to_string()
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            format!("'{}'", arr.value(row_idx).replace('\'', "''"))
        }
        DataType::LargeUtf8 => {
            let arr = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            format!("'{}'", arr.value(row_idx).replace('\'', "''"))
        }
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            if arr.value(row_idx) { "TRUE" } else { "FALSE" }.to_string()
        }
        DataType::Binary => {
            let arr = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            let bytes = arr.value(row_idx);
            // Simple hex encoding without external crate
            let hex_str: String = bytes.iter()
                .map(|b| format!("{:02X}", b))
                .collect();
            format!("X'{}'", hex_str)
        }
        DataType::LargeBinary => {
            let arr = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            let bytes = arr.value(row_idx);
            // Simple hex encoding without external crate
            let hex_str: String = bytes.iter()
                .map(|b| format!("{:02X}", b))
                .collect();
            format!("X'{}'", hex_str)
        }
        _ => {
            // For other types, use a simple string representation
            format!("'{}'", array.to_data().buffers()[0].as_slice()[row_idx])
        }
    }
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
