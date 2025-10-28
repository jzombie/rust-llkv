use crate::catalog::{TableCatalog, TableMetadataView};
use crate::types::TableId;
use llkv_expr::expr::ScalarExpr;
use llkv_result::{Error, Result};

/// Context for resolving identifiers (e.g., default table from FROM clause).
#[derive(Clone, Debug, Default)]
pub struct IdentifierContext {
    default_table_id: Option<TableId>,
    table_alias: Option<String>,
}

impl IdentifierContext {
    pub fn new(default_table_id: Option<TableId>) -> Self {
        Self {
            default_table_id,
            table_alias: None,
        }
    }

    pub fn with_table_alias(mut self, alias: Option<String>) -> Self {
        self.table_alias = alias.map(|value| value.to_ascii_lowercase());
        self
    }

    pub fn default_table_id(&self) -> Option<TableId> {
        self.default_table_id
    }

    pub fn table_alias(&self) -> Option<&str> {
        self.table_alias.as_deref()
    }
}

/// Resolution result for a column identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnResolution {
    column: String,
    field_path: Vec<String>,
}

impl ColumnResolution {
    pub fn column(&self) -> &str {
        &self.column
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn is_simple(&self) -> bool {
        self.field_path.is_empty()
    }

    pub fn into_scalar_expr(self) -> ScalarExpr<String> {
        let mut expr = ScalarExpr::column(self.column);
        for field in self.field_path {
            expr = ScalarExpr::get_field(expr, field);
        }
        expr
    }
}

/// Catalog-backed identifier resolver.
pub struct IdentifierResolver<'a> {
    catalog: &'a TableCatalog,
}

impl<'a> IdentifierResolver<'a> {
    pub fn resolve(
        &self,
        parts: &[String],
        context: IdentifierContext,
    ) -> Result<ColumnResolution> {
        if parts.is_empty() {
            return Err(Error::InvalidArgumentError(
                "invalid compound identifier".into(),
            ));
        }

        if context.default_table_id().is_none() {
            return Ok(ColumnResolution {
                column: parts.join("."),
                field_path: Vec::new(),
            });
        }

        let table_id = context.default_table_id().unwrap();
        let table_meta = self.catalog.table_metadata_view(table_id).ok_or_else(|| {
            Error::CatalogError(format!(
                "Catalog Error: table id {} is not registered",
                table_id
            ))
        })?;

        let mut effective_parts = parts;
        if let Some(alias) = context.table_alias() {
            if !effective_parts.is_empty() && effective_parts[0].eq_ignore_ascii_case(alias) {
                if effective_parts.len() == 1 {
                    return Err(Error::InvalidArgumentError(format!(
                        "Binder Error: does not have a column named '{}'",
                        effective_parts[0]
                    )));
                }
                effective_parts = &effective_parts[1..];
            } else if !effective_parts.is_empty() {
                let canonical_table = table_meta.canonical_table();
                if effective_parts[0].eq_ignore_ascii_case(canonical_table) {
                    return Err(Error::InvalidArgumentError(format!(
                        "Binder Error: table '{}' not found",
                        effective_parts[0]
                    )));
                }
                if effective_parts.len() >= 2
                    && let Some(schema) = table_meta.canonical_schema()
                    && effective_parts[0].eq_ignore_ascii_case(schema)
                    && effective_parts[1].eq_ignore_ascii_case(canonical_table)
                {
                    return Err(Error::InvalidArgumentError(format!(
                        "Binder Error: table '{}' not found",
                        effective_parts[1]
                    )));
                }
            }
        }

        let alias_active = context.table_alias().is_some();
        resolve_identifier_parts(effective_parts, &table_meta, alias_active)
    }
}

fn resolve_identifier_parts(
    parts: &[String],
    table_meta: &TableMetadataView,
    alias_active: bool,
) -> Result<ColumnResolution> {
    let canonical_table = table_meta.canonical_table().to_string();
    let canonical_schema = table_meta.canonical_schema().map(str::to_string);
    let canonical_parts: Vec<String> = parts.iter().map(|part| part.to_ascii_lowercase()).collect();

    let mut column_start_idx = 0usize;

    if !alias_active {
        if let Some(schema) = canonical_schema.as_deref() {
            let schema_then_table = canonical_parts.len() == 2
                && canonical_parts[0] == schema
                && canonical_parts[1] == canonical_table;
            let starts_with_table =
                canonical_parts.len() >= 2 && canonical_parts[0] == canonical_table;

            if canonical_parts.len() >= 3
                && canonical_parts[0] == schema
                && canonical_parts[1] == canonical_table
            {
                column_start_idx = 2;
            } else if schema_then_table || starts_with_table {
                column_start_idx = 1;
            } else if canonical_parts.len() >= 3 && canonical_parts[1] == canonical_table {
                return Err(Error::InvalidArgumentError(format!(
                    "Binder Error: table '{}' not found",
                    parts[0]
                )));
            } else if canonical_parts.len() >= 3 && canonical_parts[0] == schema {
                return Err(Error::InvalidArgumentError(format!(
                    "Binder Error: table '{}' not found",
                    parts[1]
                )));
            }
        } else if canonical_parts.len() >= 2 && canonical_parts[0] == canonical_table {
            column_start_idx = 1;
        }
    }

    if column_start_idx >= parts.len() {
        return Err(Error::InvalidArgumentError(format!(
            "invalid column reference in identifier: {}",
            parts.join(".")
        )));
    }

    let column = parts[column_start_idx].clone();
    let mut field_path = Vec::new();
    if column_start_idx + 1 < parts.len() {
        field_path.extend(parts[(column_start_idx + 1)..].iter().cloned());
    }

    Ok(ColumnResolution { column, field_path })
}

impl TableCatalog {
    pub fn identifier_resolver(&self) -> IdentifierResolver<'_> {
        IdentifierResolver { catalog: self }
    }
}
