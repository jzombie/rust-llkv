use crate::reserved::FIRST_USER_FIELD_ID;
use llkv_column_map::types::FieldId;
use llkv_result::{Error, Result};
use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};

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
    pub(super) display_name: String,
    pub(super) constraints: FieldConstraints,
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

    pub fn rename_field(&self, old_name: &str, new_name: &str) -> Result<()> {
        let old_canonical = old_name.to_ascii_lowercase();
        let new_canonical = new_name.to_ascii_lowercase();

        let mut inner = self.inner.write().map_err(|_| {
            Error::Internal("Failed to acquire field resolver write lock".to_string())
        })?;

        let field_id = *inner.field_name_to_id.get(&old_canonical).ok_or_else(|| {
            Error::CatalogError(format!("Field '{}' does not exist in table", old_name))
        })?;

        if inner.field_name_to_id.contains_key(&new_canonical) {
            return Err(Error::CatalogError(format!(
                "Field '{}' already exists in table",
                new_name
            )));
        }

        inner.field_name_to_id.remove(&old_canonical);
        inner
            .field_name_to_id
            .insert(new_canonical.clone(), field_id);

        let metadata = inner.field_id_to_meta.get_mut(&field_id).ok_or_else(|| {
            Error::CatalogError(format!(
                "Field '{}' metadata is missing from catalog",
                old_name
            ))
        })?;

        metadata.display_name = new_name.to_string();
        metadata.canonical_name = new_canonical;

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
    fn field_resolver_rename_field() {
        let resolver = FieldResolver::new();
        let fid = resolver.register_field("name").unwrap();
        resolver.rename_field("name", "name_new").unwrap();
        assert_eq!(resolver.field_id("name"), None);
        assert_eq!(resolver.field_id("name_new"), Some(fid));
        assert_eq!(resolver.field_name(fid), Some("name_new".to_string()));
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
