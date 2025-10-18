use std::fmt;

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
