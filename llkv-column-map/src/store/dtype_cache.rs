use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::{ColumnDescriptor, DescriptorIterator};
use crate::types::{LogicalFieldId, ROW_ID_FIELD_ID};
use arrow::datatypes::DataType;
use llkv_result::{Error, Result};
use llkv_storage::{
    pager::{BatchGet, BatchPut, GetResult, Pager},
    serialization::deserialize_array,
    types::PhysicalKey,
};
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLock};

/// In-memory cache of Arrow DataType per field id (not persisted).
pub struct DTypeCache<P: Pager> {
    pager: Arc<P>,
    catalog: Arc<RwLock<ColumnCatalog>>,
    cache: RwLock<FxHashMap<LogicalFieldId, DataType>>,
}

impl<P> Clone for DTypeCache<P>
where
    P: Pager,
{
    fn clone(&self) -> Self {
        Self {
            pager: Arc::clone(&self.pager),
            catalog: Arc::clone(&self.catalog),
            cache: RwLock::new(FxHashMap::default()),
        }
    }
}

impl<P> DTypeCache<P>
where
    P: Pager<Blob = EntryHandle>,
{
    pub fn new(pager: Arc<P>, catalog: Arc<RwLock<ColumnCatalog>>) -> Self {
        Self {
            pager,
            catalog,
            cache: RwLock::new(FxHashMap::default()),
        }
    }

    pub fn insert(&self, field_id: LogicalFieldId, dtype: DataType) {
        self.cache.write().unwrap().insert(field_id, dtype);
    }

    #[inline]
    pub fn cached_data_type(&self, field_id: LogicalFieldId) -> Option<DataType> {
        self.cache.read().unwrap().get(&field_id).cloned()
    }

    /// Stable FNV-1a 64-bit over the Debug representation of DataType.
    #[inline]
    #[allow(dead_code)] // TODO: Keep?
    pub(crate) fn dtype_fingerprint(dt: &DataType) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let s = format!("{:?}", dt);
        let mut hash = FNV_OFFSET;
        for b in s.as_bytes() {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    #[inline]
    #[allow(dead_code)] // TODO: Keep?
    pub(crate) fn desc_dtype_fingerprint(desc: &ColumnDescriptor) -> u64 {
        ((desc._padding as u64) << 32) | (desc.data_type_code as u64)
    }

    #[inline]
    #[allow(dead_code)] // TODO: Keep?
    pub(crate) fn set_desc_dtype_fingerprint(desc: &mut ColumnDescriptor, fp: u64) {
        desc.data_type_code = (fp & 0xFFFF_FFFF) as u32;
        desc._padding = ((fp >> 32) & 0xFFFF_FFFF) as u32;
    }

    /// Returns and caches the Arrow DataType for a given field id.
    #[allow(dead_code)] // TODO: Keep
    pub fn dtype_for_field(&self, field_id: LogicalFieldId) -> Result<DataType> {
        // Fast path: cached
        if let Some(dt) = self.cache.read().unwrap().get(&field_id).cloned() {
            return Ok(dt);
        }

        if field_id.field_id() == ROW_ID_FIELD_ID {
            let dt = DataType::UInt64;
            self.cache.write().unwrap().insert(field_id, dt.clone());
            return Ok(dt);
        }

        // Lookup descriptor pk
        let descriptor_pk = {
            let catalog = self.catalog.read().unwrap();
            let result = catalog.map.get(&field_id);
            if result.is_none() && (field_id.table_id() == 1 || field_id.table_id() == 2) {
                tracing::trace!(
                    "!!! DTYPE_FOR_FIELD: NOT FOUND for field_id table_id={} field_id={}, DTypeCache at {:p}, catalog has {} entries",
                    field_id.table_id(),
                    field_id.field_id(),
                    self as *const _,
                    catalog.map.len()
                );
            }
            *result.ok_or(Error::NotFound)?
        };

        // Load descriptor to gather first non-empty meta
        let desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let mut desc =
            crate::store::descriptor::ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        // If descriptor carries a fingerprint, try to fast-path known monomorphized types.
        let fp = Self::desc_dtype_fingerprint(&desc);
        if fp != 0 {
            if fp == Self::dtype_fingerprint(&DataType::UInt64) {
                let dt = DataType::UInt64;
                self.cache.write().unwrap().insert(field_id, dt.clone());
                return Ok(dt);
            }
            if fp == Self::dtype_fingerprint(&DataType::Int32) {
                let dt = DataType::Int32;
                self.cache.write().unwrap().insert(field_id, dt.clone());
                return Ok(dt);
            }
            if fp == Self::dtype_fingerprint(&DataType::Int64) {
                let dt = DataType::Int64;
                self.cache.write().unwrap().insert(field_id, dt.clone());
                return Ok(dt);
            }
            if fp == Self::dtype_fingerprint(&DataType::Float64) {
                let dt = DataType::Float64;
                self.cache.write().unwrap().insert(field_id, dt.clone());
                return Ok(dt);
            }
            if fp == Self::dtype_fingerprint(&DataType::Utf8) {
                let dt = DataType::Utf8;
                self.cache.write().unwrap().insert(field_id, dt.clone());
                return Ok(dt);
            }
            if fp == Self::dtype_fingerprint(&DataType::Date32) {
                let dt = DataType::Date32;
                self.cache.write().unwrap().insert(field_id, dt.clone());
                return Ok(dt);
            }
            // Unknown fingerprint; fall back to peeking one chunk to get exact DataType.
        }

        let mut first_chunk_pk: Option<PhysicalKey> = None;
        for m in DescriptorIterator::new(self.pager.as_ref(), desc.head_page_pk) {
            let meta = m?;
            if meta.row_count > 0 {
                first_chunk_pk = Some(meta.chunk_pk);
                break;
            }
        }
        let chunk_pk = first_chunk_pk.ok_or(Error::NotFound)?;

        // Fetch the chunk and inspect dtype
        let blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: chunk_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let any = deserialize_array(blob)?;
        let dt = any.data_type().clone();

        // Persist fingerprint for future opens.
        let fp_new = Self::dtype_fingerprint(&dt);
        if fp_new != fp {
            Self::set_desc_dtype_fingerprint(&mut desc, fp_new);
            let updated = desc.to_le_bytes();
            let _ = self.pager.batch_put(&[BatchPut::Raw {
                key: descriptor_pk,
                bytes: updated,
            }]);
        }

        // Cache and return
        self.cache.write().unwrap().insert(field_id, dt.clone());
        Ok(dt)
    }
}
