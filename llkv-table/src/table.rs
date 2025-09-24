use std::sync::Arc;

use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ColumnStore;
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{LogicalFieldId, Namespace};

use crate::types::FieldId;
// ----- sys_catalog wiring -----
use crate::sys_catalog::{ColMeta, SysCatalog, TableMeta};

/// Compose a 64-bit logical field id from a 32-bit table id and a 32-bit column id.
#[inline]
fn lfid_for(table_id: u32, column_id: FieldId) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_table_id(table_id)
        .with_field_id(column_id)
        .with_namespace(Namespace::UserData) // Assume standard user data
}

/// Thin configuration for the table.
#[derive(Clone, Debug, Default)]
pub struct TableCfg {}

/// ColumnMap-backed table.
pub struct Table {
    store: ColumnStore<MemPager>,
    #[allow(dead_code)]
    cfg: TableCfg,
    /// A unique 32-bit ID for this table, used to namespace its columns.
    table_id: u32,
}

impl Table {
    /// Creates a new table instance with a given ID.
    pub fn new(table_id: u32, cfg: TableCfg) -> Self {
        let pager = Arc::new(MemPager::default());
        let store = ColumnStore::open(pager).unwrap();
        Self {
            store,
            cfg,
            table_id,
        }
    }

    /// Primary method for writing data to the table.
    pub fn insert_many(&self, batch: &RecordBatch) -> Result<(), llkv_column_map::error::Error> {
        let mut new_fields = Vec::with_capacity(batch.schema().fields().len());
        for field in batch.schema().fields() {
            if field.name() == "row_id" {
                new_fields.push(field.as_ref().clone());
                continue;
            }

            let user_field_id: FieldId = field
                .metadata()
                .get("field_id")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    llkv_column_map::error::Error::Internal(format!(
                        "Field '{}' is missing a valid 'field_id' in its metadata.",
                        field.name()
                    ))
                })?;

            let lfid = lfid_for(self.table_id, user_field_id);
            let mut new_metadata = field.metadata().clone();

            // --- FIX: Use `into()` to get the underlying u64 representation ---
            let lfid_val: u64 = lfid.into();
            new_metadata.insert("field_id".to_string(), lfid_val.to_string());

            let new_field =
                Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                    .with_metadata(new_metadata);
            new_fields.push(new_field);
        }

        let new_schema = Arc::new(Schema::new(new_fields));
        let namespaced_batch = RecordBatch::try_new(new_schema, batch.columns().to_vec())?;

        self.store.append(&namespaced_batch)
    }

    #[inline]
    pub fn catalog(&self) -> SysCatalog<'_> {
        SysCatalog::new(&self.store)
    }

    #[inline]
    pub fn put_table_meta(&self, meta: &TableMeta) {
        debug_assert_eq!(meta.table_id, self.table_id);
        self.catalog().put_table_meta(meta);
    }

    #[inline]
    pub fn get_table_meta(&self) -> Option<TableMeta> {
        self.catalog().get_table_meta(self.table_id)
    }

    #[inline]
    pub fn put_col_meta(&self, meta: &ColMeta) {
        self.catalog().put_col_meta(self.table_id, meta);
    }

    #[inline]
    pub fn get_cols_meta(&self, col_ids: &[u32]) -> Vec<Option<ColMeta>> {
        self.catalog().get_cols_meta(self.table_id, col_ids)
    }

    pub fn store(&self) -> &ColumnStore<MemPager> {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use llkv_column_map::store::scan::{
        PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
        PrimitiveWithRowIdsVisitor,
    };
    use std::collections::HashMap;

    #[test]
    fn append_record_batch_and_scan() {
        let table = Table::new(1, TableCfg::default());
        const COL_A: FieldId = 10;
        const COL_B: FieldId = 11;

        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("a", DataType::UInt64, false)
                .with_metadata(HashMap::from([("field_id".to_string(), COL_A.to_string())])),
            Field::new("b", DataType::Binary, false)
                .with_metadata(HashMap::from([("field_id".to_string(), COL_B.to_string())])),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3])),
                Arc::new(UInt64Array::from(vec![100, 200, 300])),
                // --- FIX: Cast each byte string literal to a byte slice ---
                Arc::new(BinaryArray::from(vec![
                    b"foo" as &[u8],
                    b"bar" as &[u8],
                    b"baz" as &[u8],
                ])),
            ],
        )
        .unwrap();

        table.insert_many(&batch).unwrap();

        struct TestVisitor {
            vals: Vec<u64>,
        }
        impl PrimitiveVisitor for TestVisitor {
            fn u64_chunk(&mut self, a: &UInt64Array) {
                self.vals.extend_from_slice(a.values());
            }
        }
        impl PrimitiveWithRowIdsVisitor for TestVisitor {}
        impl PrimitiveSortedVisitor for TestVisitor {}
        impl PrimitiveSortedWithRowIdsVisitor for TestVisitor {}

        let mut visitor = TestVisitor { vals: vec![] };

        let global_lfid = lfid_for(1, COL_A);
        table
            .store
            .scan(global_lfid, Default::default(), &mut visitor)
            .unwrap();

        assert_eq!(visitor.vals, vec![100, 200, 300]);
    }
}
