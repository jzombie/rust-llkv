use llkv_types::FieldId;

/// Minimal schema lookup surface used by expression translation.
///
/// Consumers implement this for their schema representations so the
/// translation helpers avoid hard dependencies on a specific schema type.
pub trait SchemaView {
    /// Resolve a column name to its field id, if present.
    fn field_id_by_name(&self, name: &str) -> Option<FieldId>;
}
