use crate::catalog::{TableCatalog, TableMetadataView};
use crate::metadata::MetadataManager;
use crate::reserved::FIRST_USER_FIELD_ID;
use crate::types::TableId;
use llkv_column_map::types::FieldId;
use llkv_expr::expr::ScalarExpr;
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::fmt;
use std::sync::{Arc, RwLock};

// ========================================================================
// Qualified table name helpers
// ========================================================================

/// Owned representation of a schema-qualified table name (preserves original casing).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QualifiedTableName {
    schema: Option<String>,
    table: String,
}

impl QualifiedTableName {
    /// Create a new qualified table name from optional schema and table components.
    pub fn new<S, T>(schema: Option<S>, table: T) -> Self
    where
        S: Into<String>,
        T: Into<String>,
    {
        Self {
            schema: schema.map(Into::into),
            table: table.into(),
        }
    }

    /// Create a qualified table name from a pre-formatted string.
    ///
    /// Strings in the form `schema.table` will be split into schema and table components.
    /// Strings without a dot are treated as bare table names.
    pub fn from_qualified(name: impl Into<String>) -> Self {
        let raw = name.into();
        let (schema, table) = split_schema_table(&raw);
        Self {
            schema: schema.map(|s| s.to_string()),
            table: table.to_string(),
        }
    }

    /// Return the schema component, if present.
    pub fn schema(&self) -> Option<&str> {
        self.schema.as_deref()
    }

    /// Return the table component.
    pub fn table(&self) -> &str {
        &self.table
    }

    /// Format as `schema.table` (or just `table` if schema is absent).
    pub fn to_display_string(&self) -> String {
        match &self.schema {
            Some(schema) => format!("{schema}.{}", self.table),
            None => self.table.clone(),
        }
    }

    /// Convert into the canonical (lowercase) key representation.
    pub(crate) fn canonical_key(&self) -> TableNameKey {
        TableNameKey::from(self)
    }
}

impl fmt::Display for QualifiedTableName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_display_string())
    }
}

impl From<&str> for QualifiedTableName {
    fn from(value: &str) -> Self {
        Self::from_qualified(value)
    }
}

impl From<String> for QualifiedTableName {
    fn from(value: String) -> Self {
        Self::from_qualified(value)
    }
}

impl<S, T> From<(S, T)> for QualifiedTableName
where
    S: Into<String>,
    T: Into<String>,
{
    fn from(value: (S, T)) -> Self {
        Self::new(Some(value.0.into()), value.1.into())
    }
}

/// Borrowed representation of a schema-qualified table name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QualifiedTableNameRef<'a> {
    schema: Option<&'a str>,
    table: &'a str,
}

impl<'a> QualifiedTableNameRef<'a> {
    /// Create a borrowed qualified name.
    pub fn new(schema: Option<&'a str>, table: &'a str) -> Self {
        Self { schema, table }
    }

    /// Parse a raw string into a borrowed qualified name.
    pub fn parse(raw: &'a str) -> Self {
        let (schema, table) = split_schema_table(raw);
        Self { schema, table }
    }

    pub(crate) fn canonical_key(self) -> TableNameKey {
        TableNameKey::from(self)
    }
}

impl<'a> From<&'a str> for QualifiedTableNameRef<'a> {
    fn from(value: &'a str) -> Self {
        Self::parse(value)
    }
}

impl<'a> From<(&'a str, &'a str)> for QualifiedTableNameRef<'a> {
    fn from(value: (&'a str, &'a str)) -> Self {
        Self::new(Some(value.0), value.1)
    }
}

impl<'a> From<QualifiedTableNameRef<'a>> for QualifiedTableName {
    fn from(value: QualifiedTableNameRef<'a>) -> Self {
        Self::new(value.schema.map(str::to_string), value.table.to_string())
    }
}

/// Canonical (lowercase) key for table name lookups.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct TableNameKey {
    schema: Option<String>,
    table: String,
}

impl TableNameKey {
    pub fn new(schema: Option<&str>, table: &str) -> Self {
        Self {
            schema: schema.map(|s| s.to_ascii_lowercase()),
            table: table.to_ascii_lowercase(),
        }
    }

    pub fn schema(&self) -> Option<&str> {
        self.schema.as_deref()
    }

    pub fn table(&self) -> &str {
        &self.table
    }
}

impl fmt::Display for TableNameKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.schema {
            Some(schema) => write!(f, "{schema}.{}", self.table),
            None => write!(f, "{}", self.table),
        }
    }
}

impl From<&QualifiedTableName> for TableNameKey {
    fn from(value: &QualifiedTableName) -> Self {
        Self::new(value.schema(), value.table())
    }
}

impl<'a> From<QualifiedTableNameRef<'a>> for TableNameKey {
    fn from(value: QualifiedTableNameRef<'a>) -> Self {
        Self::new(value.schema, value.table)
    }
}

fn split_schema_table(name: &str) -> (Option<&str>, &str) {
    if let Some(idx) = name.find('.') {
        let (schema, rest) = name.split_at(idx);
        let table = &rest[1..];
        if table.is_empty() {
            (Some(schema), "")
        } else {
            (Some(schema), table)
        }
    } else {
        (None, name)
    }
}

/// Convert a table name into its display (original) and canonical (lowercase) forms.
pub fn canonical_table_name(name: &str) -> Result<(String, String)> {
    if name.is_empty() {
        return Err(Error::InvalidArgumentError(
            "table name must not be empty".into(),
        ));
    }
    let display = name.to_string();
    let canonical = display.to_ascii_lowercase();
    Ok((display, canonical))
}

/// Resolve a table `TableId` into display + canonical names using the catalog/metadata layers.
pub fn resolve_table_name<P>(
    catalog: &TableCatalog,
    metadata: &MetadataManager<P>,
    table_id: TableId,
) -> Result<(String, String)>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    if let Some(qt) = catalog.table_name(table_id) {
        let display = qt.to_display_string();
        let canonical = display.to_ascii_lowercase();
        return Ok((display, canonical));
    }

    if let Some(meta) = metadata.table_meta(table_id)?
        && let Some(name) = meta.name
    {
        let canonical = name.to_ascii_lowercase();
        return Ok((name, canonical));
    }

    let fallback = table_id.to_string();
    Ok((fallback.clone(), fallback))
}

/// Context for resolving identifiers (e.g., default table from FROM clause).
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentifierContext {
    default_table_id: Option<TableId>,
}

impl IdentifierContext {
    pub fn new(default_table_id: Option<TableId>) -> Self {
        Self { default_table_id }
    }

    pub fn default_table_id(&self) -> Option<TableId> {
        self.default_table_id
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

        resolve_identifier_parts(parts, &table_meta)
    }
}

fn resolve_identifier_parts(
    parts: &[String],
    table_meta: &TableMetadataView,
) -> Result<ColumnResolution> {
    let canonical_table = table_meta.canonical_table().to_string();
    let canonical_schema = table_meta.canonical_schema().map(str::to_string);
    let canonical_parts: Vec<String> = parts.iter().map(|part| part.to_ascii_lowercase()).collect();

    let mut column_start_idx = 0usize;

    if let Some(schema) = canonical_schema.as_deref() {
        let schema_then_table = canonical_parts.len() == 2
            && canonical_parts[0] == schema
            && canonical_parts[1] == canonical_table;
        let starts_with_table = canonical_parts.len() >= 2 && canonical_parts[0] == canonical_table;

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

// ========================================================================
// FieldResolver - Per-table field name resolution
// ========================================================================

/// Constraint metadata recorded for each field.
#[derive(Debug, Clone, PartialEq, Eq, Default, bitcode::Encode, bitcode::Decode)]
pub struct FieldConstraints {
    pub primary_key: bool,
    pub unique: bool,
    pub check_expr: Option<String>,
}

/// Definition for registering a field with the catalog.
#[derive(Debug, Clone)]
pub struct FieldDefinition {
    display_name: String,
    constraints: FieldConstraints,
}

impl FieldDefinition {
    pub fn new(display_name: impl Into<String>) -> Self {
        Self {
            display_name: display_name.into(),
            constraints: FieldConstraints::default(),
        }
    }

    pub fn with_primary_key(mut self, primary_key: bool) -> Self {
        self.constraints.primary_key = primary_key;
        if primary_key {
            self.constraints.unique = true;
        }
        self
    }

    pub fn with_unique(mut self, unique: bool) -> Self {
        if unique {
            self.constraints.unique = true;
        }
        self
    }

    pub fn with_check_expr(mut self, check_expr: Option<String>) -> Self {
        self.constraints.check_expr = check_expr;
        self
    }

    pub fn constraints(&self) -> &FieldConstraints {
        &self.constraints
    }
}

impl From<&str> for FieldDefinition {
    fn from(value: &str) -> Self {
        FieldDefinition::new(value)
    }
}

impl From<String> for FieldDefinition {
    fn from(value: String) -> Self {
        FieldDefinition::new(value)
    }
}

/// Rich field metadata exposed by the catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldInfo {
    pub field_id: FieldId,
    pub display_name: String,
    pub canonical_name: String,
    pub constraints: FieldConstraints,
}

#[derive(Debug, Clone)]
struct FieldMetadata {
    display_name: String,
    canonical_name: String,
    constraints: FieldConstraints,
}

/// Per-table field name resolver.
#[derive(Debug, Clone)]
pub struct FieldResolver {
    inner: Arc<RwLock<FieldResolverInner>>,
}

#[derive(Debug)]
struct FieldResolverInner {
    field_name_to_id: FxHashMap<String, FieldId>,
    field_id_to_meta: FxHashMap<FieldId, FieldMetadata>,
    next_field_id: FieldId,
}

impl FieldResolver {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(FieldResolverInner {
                field_name_to_id: FxHashMap::default(),
                field_id_to_meta: FxHashMap::default(),
                next_field_id: FIRST_USER_FIELD_ID,
            })),
        }
    }

    pub fn register_field(&self, definition: impl Into<FieldDefinition>) -> Result<FieldId> {
        let FieldDefinition {
            display_name,
            constraints,
        } = definition.into();
        let canonical_name = display_name.to_ascii_lowercase();

        let mut inner = self.inner.write().map_err(|_| {
            Error::Internal("Failed to acquire field resolver write lock".to_string())
        })?;

        if inner.field_name_to_id.contains_key(&canonical_name) {
            return Err(Error::CatalogError(format!(
                "Field '{}' already exists in table",
                display_name
            )));
        }

        let field_id = inner.next_field_id;
        inner.next_field_id = inner
            .next_field_id
            .checked_add(1)
            .ok_or_else(|| Error::Internal("FieldId overflow".to_string()))?;

        inner
            .field_name_to_id
            .insert(canonical_name.clone(), field_id);
        inner.field_id_to_meta.insert(
            field_id,
            FieldMetadata {
                display_name,
                canonical_name,
                constraints,
            },
        );

        Ok(field_id)
    }

    pub fn field_id(&self, name: &str) -> Option<FieldId> {
        let canonical = name.to_ascii_lowercase();
        let inner = self.inner.read().ok()?;
        inner.field_name_to_id.get(&canonical).copied()
    }

    pub fn field_name(&self, id: FieldId) -> Option<String> {
        let inner = self.inner.read().ok()?;
        inner
            .field_id_to_meta
            .get(&id)
            .map(|meta| meta.display_name.clone())
    }

    pub fn field_exists(&self, name: &str) -> bool {
        self.field_id(name).is_some()
    }

    pub fn field_count(&self) -> usize {
        match self.inner.read() {
            Ok(inner) => inner.field_id_to_meta.len(),
            Err(_) => 0,
        }
    }

    pub fn field_names(&self) -> Vec<String> {
        match self.inner.read() {
            Ok(inner) => inner
                .field_id_to_meta
                .values()
                .map(|meta| meta.display_name.clone())
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    pub fn field_constraints(&self, id: FieldId) -> Option<FieldConstraints> {
        let inner = self.inner.read().ok()?;
        inner
            .field_id_to_meta
            .get(&id)
            .map(|meta| meta.constraints.clone())
    }

    pub fn field_constraints_by_name(&self, name: &str) -> Option<FieldConstraints> {
        let id = self.field_id(name)?;
        self.field_constraints(id)
    }

    pub fn set_field_unique(&self, name: &str, unique: bool) -> Result<()> {
        let canonical = name.to_ascii_lowercase();
        let mut inner = self.inner.write().map_err(|_| {
            Error::Internal("Failed to acquire field resolver write lock".to_string())
        })?;

        let field_id = *inner.field_name_to_id.get(&canonical).ok_or_else(|| {
            Error::CatalogError(format!("Field '{}' does not exist in table", name))
        })?;

        let metadata = inner.field_id_to_meta.get_mut(&field_id).ok_or_else(|| {
            Error::CatalogError(format!("Field '{}' metadata is missing from catalog", name))
        })?;

        if unique {
            metadata.constraints.unique = true;
        } else if !metadata.constraints.primary_key {
            metadata.constraints.unique = false;
        }

        Ok(())
    }

    pub fn field_info(&self, id: FieldId) -> Option<FieldInfo> {
        let inner = self.inner.read().ok()?;
        inner.field_id_to_meta.get(&id).map(|meta| FieldInfo {
            field_id: id,
            display_name: meta.display_name.clone(),
            canonical_name: meta.canonical_name.clone(),
            constraints: meta.constraints.clone(),
        })
    }

    pub fn field_info_by_name(&self, name: &str) -> Option<FieldInfo> {
        let id = self.field_id(name)?;
        self.field_info(id)
    }

    pub fn export_state(&self) -> FieldResolverState {
        let inner = match self.inner.read() {
            Ok(inner) => inner,
            Err(_) => {
                return FieldResolverState {
                    fields: Vec::new(),
                    next_field_id: FIRST_USER_FIELD_ID,
                };
            }
        };

        let mut fields = Vec::new();
        for (&field_id, meta) in &inner.field_id_to_meta {
            fields.push(FieldState {
                field_id,
                display_name: meta.display_name.clone(),
                canonical_name: meta.canonical_name.clone(),
                constraints: meta.constraints.clone(),
            });
        }

        FieldResolverState {
            fields,
            next_field_id: inner.next_field_id,
        }
    }

    pub fn from_state(state: FieldResolverState) -> Result<Self> {
        let mut field_name_to_id = FxHashMap::default();
        let mut field_id_to_meta = FxHashMap::default();

        for field_state in state.fields {
            let FieldState {
                field_id,
                display_name,
                canonical_name,
                constraints,
            } = field_state;

            if field_id_to_meta.contains_key(&field_id) {
                return Err(Error::CatalogError(format!(
                    "Duplicate field_id {} in field resolver state",
                    field_id
                )));
            }

            if field_name_to_id.contains_key(&canonical_name) {
                return Err(Error::CatalogError(format!(
                    "Duplicate field name '{}' in field resolver state",
                    display_name
                )));
            }

            field_name_to_id.insert(canonical_name.clone(), field_id);
            field_id_to_meta.insert(
                field_id,
                FieldMetadata {
                    display_name,
                    canonical_name,
                    constraints,
                },
            );
        }

        Ok(Self {
            inner: Arc::new(RwLock::new(FieldResolverInner {
                field_name_to_id,
                field_id_to_meta,
                next_field_id: state.next_field_id,
            })),
        })
    }
}

impl Default for FieldResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable field resolver state.
#[derive(Debug, Clone, bitcode::Encode, bitcode::Decode)]
pub struct FieldResolverState {
    pub fields: Vec<FieldState>,
    pub next_field_id: FieldId,
}

/// Serializable field state.
#[derive(Debug, Clone, bitcode::Encode, bitcode::Decode)]
pub struct FieldState {
    pub field_id: FieldId,
    pub display_name: String,
    pub canonical_name: String,
    pub constraints: FieldConstraints,
}

impl TableCatalog {
    pub fn identifier_resolver(&self) -> IdentifierResolver<'_> {
        IdentifierResolver { catalog: self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_resolver_register_and_lookup() {
        let resolver = FieldResolver::new();
        let fid = resolver.register_field("UserName").unwrap();
        assert_eq!(resolver.field_id("username"), Some(fid));
        assert_eq!(resolver.field_name(fid), Some("UserName".to_string()));
    }

    #[test]
    fn field_resolver_set_unique() {
        let resolver = FieldResolver::new();
        resolver
            .register_field(FieldDefinition::new("id").with_primary_key(true))
            .unwrap();
        resolver
            .register_field(FieldDefinition::new("email"))
            .unwrap();
        resolver.set_field_unique("email", true).unwrap();
        assert!(resolver.field_constraints_by_name("email").unwrap().unique);
        resolver.set_field_unique("email", false).unwrap();
        assert!(!resolver.field_constraints_by_name("email").unwrap().unique);
    }

    #[test]
    fn field_resolver_export_roundtrip() {
        let resolver = FieldResolver::new();
        let fid1 = resolver.register_field("field1").unwrap();
        let fid2 = resolver.register_field("Field2").unwrap();

        let state = resolver.export_state();
        let restored = FieldResolver::from_state(state).unwrap();

        assert_eq!(restored.field_id("field1"), Some(fid1));
        assert_eq!(restored.field_id("field2"), Some(fid2));
    }
}
