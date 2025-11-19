//! Minimal SQL front-end that routes statements through llkv-fusion.
//!
//! This engine parses SQL statements so it can intercept `CREATE TABLE`
//! definitions, materialize them inside LLKV's [`TableCatalog`], and register
//! the resulting [`LlkvTableProvider`]s with DataFusion. All other statements
//! are delegated to a `SessionContext` that uses [`LlkvQueryPlanner`] for scan
//! planning, keeping LLKV storage as the backing store.

use std::convert::TryFrom;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use llkv_fusion::LlkvQueryPlanner;
use llkv_result::{Error, Result};
use llkv_storage::pager::BoxedPager;
use llkv_table::catalog::TableCatalog;
use sqlparser::ast::{
    ColumnDef, ColumnOption, ColumnOptionDef, CreateTable, DataType as SqlDataType, ObjectName,
    ObjectNamePart, Statement,
};
use sqlparser::dialect::GenericDialect;
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
        let dialect = GenericDialect {};
        match Parser::parse_sql(&dialect, sql) {
            Ok(statements) if !statements.is_empty() => {
                let mut out = Vec::with_capacity(statements.len());
                for stmt in statements {
                    out.push(self.execute_statement(stmt).await?);
                }
                Ok(out)
            }
            _ => {
                // Fallback to DataFusion directly if parsing failed or produced no statements
                let result = self.execute_via_datafusion(sql).await?;
                Ok(vec![result])
            }
        }
    }

    async fn execute_statement(&self, stmt: Statement) -> Result<SqlStatementResult> {
        match stmt {
            Statement::CreateTable(create) => {
                self.handle_create_table(create)?;
                Ok(SqlStatementResult::Statement { rows_affected: 0 })
            }
            other => {
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
            .map_err(|e| Error::Internal(format!("DataFusion planning failed: {}", e)))?;

        let batches = df
            .collect()
            .await
            .map_err(|e| Error::Internal(format!("DataFusion execution failed: {}", e)))?;

        if batches.is_empty() || batches[0].num_columns() == 0 {
            Ok(SqlStatementResult::Statement { rows_affected: 0 })
        } else {
            Ok(SqlStatementResult::Query { batches })
        }
    }

    fn handle_create_table(&self, create: CreateTable) -> Result<()> {
        if create.clone().query.is_some() {
            return Err(Error::Internal(
                "CREATE TABLE AS SELECT is not supported yet".into(),
            ));
        }

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

        let schema = build_schema(&create.columns)?;
        self.catalog.create_table(&table_name, Arc::clone(&schema))?;
        self.register_table(&table_name)?;
        Ok(())
    }

    fn register_existing_tables(&self) -> Result<()> {
        for name in self.catalog.list_tables() {
            self.register_table(&name)?;
        }
        Ok(())
    }

    fn register_table(&self, name: &str) -> Result<()> {
        if let Some(provider) = self.catalog.get_table(name)? {
            self.ctx
                .register_table(name, provider)
                .map_err(|e| Error::Internal(format!("failed to register table {name}: {e}")))?;
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
        _ => {
            return Err(Error::Internal(format!(
                "unsupported SQL data type: {sql_type:?}"
            )))
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
