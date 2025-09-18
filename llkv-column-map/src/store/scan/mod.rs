//! Primitive column scanners and visitor traits used by ColumnStore.
//!
//! Focus: integer primitives with fast unsorted and sorted scans.
//! - Monomorphized per-type paths (no dynamic dispatch in hot loops)
//! - Batches IO with pager, minimal object churn
//! - Coalesces adjacent items into runs for sorted scans
//! - Optional row-id variants
//!
//! This module is intentionally self-contained so higher-level methods in
//! `store::mod` can delegate without code duplication.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::{Bound, RangeBounds};

use arrow::array::*;
use arrow::compute;
use arrow::datatypes::DataType;

use rustc_hash::FxHashMap;

use super::ColumnStore;
use crate::error::{Error, Result};
use crate::serialization::deserialize_array;
use crate::storage::pager::{BatchGet, GetResult, Pager};
use crate::store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator};
use crate::types::{LogicalFieldId, PhysicalKey};
use simd_r_drive_entry_handle::EntryHandle;

pub mod builder;
pub use builder::*;

pub mod options;
pub use options::*;

pub mod unsorted;
pub use unsorted::*;

pub mod ranges;
pub use ranges::*;

pub mod sorted;
pub use sorted::*;

pub mod visitors;
pub use visitors::*;

impl<P> ColumnStore<P>
where
    P: Pager<Blob = EntryHandle>,
{
    /// Unsorted scan with typed visitor callbacks per chunk.
    fn scan_visit<V: crate::store::scan::PrimitiveVisitor>(
        &self,
        field_id: LogicalFieldId,
        visitor: &mut V,
    ) -> Result<()> {
        let catalog = self.catalog.read().unwrap();
        crate::store::scan::unsorted_visit(self.pager.as_ref(), &catalog.map, field_id, visitor)
    }

    /// Convenience: sorted scan with closures over coalesced runs.
    /// Sorted scan with typed visitor callbacks over coalesced runs.
    fn scan_sorted_visit<V: crate::store::scan::PrimitiveSortedVisitor>(
        &self,
        field_id: LogicalFieldId,
        visitor: &mut V,
    ) -> Result<()> {
        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
        drop(catalog);

        let mut metas: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
        for m in crate::store::descriptor::DescriptorIterator::new(
            self.pager.as_ref(),
            desc.head_page_pk,
        ) {
            let meta = m?;
            if meta.row_count == 0 {
                continue;
            }
            if meta.value_order_perm_pk == 0 {
                return Err(Error::NotFound);
            }
            metas.push(meta);
        }
        if metas.is_empty() {
            return Ok(());
        }
        let mut gets: Vec<BatchGet> = Vec::with_capacity(metas.len() * 2);
        for m in &metas {
            gets.push(BatchGet::Raw { key: m.chunk_pk });
            gets.push(BatchGet::Raw {
                key: m.value_order_perm_pk,
            });
        }
        let results = self.pager.batch_get(&gets)?;
        let mut blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for r in results {
            if let GetResult::Raw { key, bytes } = r {
                blobs.insert(key, bytes);
            }
        }

        let first_any = crate::serialization::deserialize_array(
            blobs
                .get(&metas[0].chunk_pk)
                .ok_or(Error::NotFound)?
                .clone(),
        )?;
        match first_any.data_type() {
            DataType::UInt64 => {
                crate::store::scan::sorted_visit_u64(self.pager.as_ref(), &metas, &blobs, visitor)
            }
            DataType::UInt32 => {
                crate::store::scan::sorted_visit_u32(self.pager.as_ref(), &metas, &blobs, visitor)
            }
            DataType::UInt16 => {
                crate::store::scan::sorted_visit_u16(self.pager.as_ref(), &metas, &blobs, visitor)
            }
            DataType::UInt8 => {
                crate::store::scan::sorted_visit_u8(self.pager.as_ref(), &metas, &blobs, visitor)
            }
            DataType::Int64 => {
                crate::store::scan::sorted_visit_i64(self.pager.as_ref(), &metas, &blobs, visitor)
            }
            DataType::Int32 => {
                crate::store::scan::sorted_visit_i32(self.pager.as_ref(), &metas, &blobs, visitor)
            }
            DataType::Int16 => {
                crate::store::scan::sorted_visit_i16(self.pager.as_ref(), &metas, &blobs, visitor)
            }
            DataType::Int8 => {
                crate::store::scan::sorted_visit_i8(self.pager.as_ref(), &metas, &blobs, visitor)
            }
            _ => Err(Error::Internal("unsupported sorted dtype".into())),
        }
    }

    /// Sorted scan in reverse (descending) with typed visitor callbacks.
    fn scan_sorted_visit_reverse<V: crate::store::scan::PrimitiveSortedVisitor>(
        &self,
        field_id: LogicalFieldId,
        visitor: &mut V,
    ) -> Result<()> {
        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
        drop(catalog);

        let mut metas: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
        for m in crate::store::descriptor::DescriptorIterator::new(
            self.pager.as_ref(),
            desc.head_page_pk,
        ) {
            let meta = m?;
            if meta.row_count == 0 {
                continue;
            }
            if meta.value_order_perm_pk == 0 {
                return Err(Error::NotFound);
            }
            metas.push(meta);
        }
        if metas.is_empty() {
            return Ok(());
        }
        let mut gets: Vec<BatchGet> = Vec::with_capacity(metas.len() * 2);
        for m in &metas {
            gets.push(BatchGet::Raw { key: m.chunk_pk });
            gets.push(BatchGet::Raw {
                key: m.value_order_perm_pk,
            });
        }
        let results = self.pager.batch_get(&gets)?;
        let mut blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for r in results {
            if let GetResult::Raw { key, bytes } = r {
                blobs.insert(key, bytes);
            }
        }

        let first_any = crate::serialization::deserialize_array(
            blobs
                .get(&metas[0].chunk_pk)
                .ok_or(Error::NotFound)?
                .clone(),
        )?;
        match first_any.data_type() {
            DataType::UInt64 => crate::store::scan::sorted_visit_u64_rev(
                self.pager.as_ref(),
                &metas,
                &blobs,
                visitor,
            ),
            DataType::UInt32 => crate::store::scan::sorted_visit_u32_rev(
                self.pager.as_ref(),
                &metas,
                &blobs,
                visitor,
            ),
            DataType::UInt16 => crate::store::scan::sorted_visit_u16_rev(
                self.pager.as_ref(),
                &metas,
                &blobs,
                visitor,
            ),
            DataType::UInt8 => crate::store::scan::sorted_visit_u8_rev(
                self.pager.as_ref(),
                &metas,
                &blobs,
                visitor,
            ),
            DataType::Int64 => crate::store::scan::sorted_visit_i64_rev(
                self.pager.as_ref(),
                &metas,
                &blobs,
                visitor,
            ),
            DataType::Int32 => crate::store::scan::sorted_visit_i32_rev(
                self.pager.as_ref(),
                &metas,
                &blobs,
                visitor,
            ),
            DataType::Int16 => crate::store::scan::sorted_visit_i16_rev(
                self.pager.as_ref(),
                &metas,
                &blobs,
                visitor,
            ),
            DataType::Int8 => crate::store::scan::sorted_visit_i8_rev(
                self.pager.as_ref(),
                &metas,
                &blobs,
                visitor,
            ),
            _ => Err(Error::Internal("unsupported sorted dtype".into())),
        }
    }

    /// Unified scan entrypoint configured by ScanOptions.
    /// Requires `V` to implement both unsorted and sorted visitor traits; methods are no-ops by default.
    pub fn scan<V>(
        &self,
        field_id: LogicalFieldId,
        opts: ScanOptions,
        visitor: &mut V,
    ) -> Result<()>
    where
        V: crate::store::scan::PrimitiveVisitor
            + crate::store::scan::PrimitiveSortedVisitor
            + crate::store::scan::PrimitiveWithRowIdsVisitor
            + crate::store::scan::PrimitiveSortedWithRowIdsVisitor,
    {
        if !opts.sorted {
            if opts.with_row_ids {
                let row_fid = opts.row_id_field.ok_or_else(|| {
                    Error::Internal("row_id field id required when with_row_ids=true".into())
                })?;
                let catalog = self.catalog.read().unwrap();
                return crate::store::scan::unsorted_with_row_ids_visit(
                    self.pager.as_ref(),
                    &catalog.map,
                    field_id,
                    row_fid,
                    visitor,
                );
            }
            return self.scan_visit(field_id, visitor);
        }

        if opts.with_row_ids {
            let row_fid = opts.row_id_field.ok_or_else(|| {
                Error::Internal("row_id field id required when with_row_ids=true".into())
            })?;
            // Prepare value metas and blobs
            let catalog = self.catalog.read().unwrap();
            let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
            let desc_blob = self
                .pager
                .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
                .pop()
                .and_then(|r| match r {
                    GetResult::Raw { bytes, .. } => Some(bytes),
                    _ => None,
                })
                .ok_or(Error::NotFound)?;
            let desc =
                crate::store::descriptor::ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

            let rid_descriptor_pk = *catalog.map.get(&row_fid).ok_or(Error::NotFound)?;
            let rid_desc_blob = self
                .pager
                .batch_get(&[BatchGet::Raw {
                    key: rid_descriptor_pk,
                }])?
                .pop()
                .and_then(|r| match r {
                    GetResult::Raw { bytes, .. } => Some(bytes),
                    _ => None,
                })
                .ok_or(Error::NotFound)?;
            let rid_desc =
                crate::store::descriptor::ColumnDescriptor::from_le_bytes(rid_desc_blob.as_ref());
            drop(catalog);

            let mut metas_val: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
            for m in crate::store::descriptor::DescriptorIterator::new(
                self.pager.as_ref(),
                desc.head_page_pk,
            ) {
                let meta = m?;
                if meta.row_count == 0 {
                    continue;
                }
                if meta.value_order_perm_pk == 0 {
                    return Err(Error::NotFound);
                }
                metas_val.push(meta);
            }
            let mut metas_rid: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
            for m in crate::store::descriptor::DescriptorIterator::new(
                self.pager.as_ref(),
                rid_desc.head_page_pk,
            ) {
                let meta = m?;
                if meta.row_count == 0 {
                    continue;
                }
                metas_rid.push(meta);
            }
            if metas_val.is_empty() {
                return Ok(());
            }
            if metas_val.len() != metas_rid.len() {
                return Err(Error::Internal(
                    "sorted_with_row_ids: chunk count mismatch".into(),
                ));
            }

            // Batch get: values (chunks + perms) and rids (chunks)
            let mut gets: Vec<BatchGet> = Vec::with_capacity(metas_val.len() * 3);
            for (mv, mr) in metas_val.iter().zip(metas_rid.iter()) {
                gets.push(BatchGet::Raw { key: mv.chunk_pk });
                gets.push(BatchGet::Raw {
                    key: mv.value_order_perm_pk,
                });
                gets.push(BatchGet::Raw { key: mr.chunk_pk });
            }
            let results = self.pager.batch_get(&gets)?;
            let mut vblobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
            let mut rblobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
            for r in results {
                if let GetResult::Raw { key, bytes } = r {
                    // Heuristic: key in rid metas => rid blob, else value blob
                    if metas_rid.iter().any(|m| m.chunk_pk == key) {
                        rblobs.insert(key, bytes);
                    } else {
                        vblobs.insert(key, bytes);
                    }
                }
            }
            let first_any = crate::serialization::deserialize_array(
                vblobs
                    .get(&metas_val[0].chunk_pk)
                    .ok_or(Error::NotFound)?
                    .clone(),
            )?;
            if opts.reverse {
                match first_any.data_type() {
                    DataType::UInt64 => sorted_visit_with_rids_u64_rev(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::UInt32 => sorted_visit_with_rids_u32_rev(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::UInt16 => sorted_visit_with_rids_u16_rev(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::UInt8 => sorted_visit_with_rids_u8_rev(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::Int64 => sorted_visit_with_rids_i64_rev(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::Int32 => sorted_visit_with_rids_i32_rev(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::Int16 => sorted_visit_with_rids_i16_rev(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::Int8 => sorted_visit_with_rids_i8_rev(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    _ => Err(Error::Internal("unsupported sorted dtype".into())),
                }
            } else {
                match first_any.data_type() {
                    DataType::UInt64 => sorted_visit_with_rids_u64(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::UInt32 => sorted_visit_with_rids_u32(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::UInt16 => sorted_visit_with_rids_u16(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::UInt8 => sorted_visit_with_rids_u8(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::Int64 => sorted_visit_with_rids_i64(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::Int32 => sorted_visit_with_rids_i32(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::Int16 => sorted_visit_with_rids_i16(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    DataType::Int8 => sorted_visit_with_rids_i8(
                        self.pager.as_ref(),
                        &metas_val,
                        &metas_rid,
                        &vblobs,
                        &rblobs,
                        visitor,
                    ),
                    _ => Err(Error::Internal("unsupported sorted dtype".into())),
                }
            }
        } else {
            if opts.reverse {
                self.scan_sorted_visit_reverse(field_id, visitor)
            } else {
                self.scan_sorted_visit(field_id, visitor)
            }
        }
    }
}
