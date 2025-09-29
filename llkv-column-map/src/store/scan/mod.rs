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

use super::ColumnStore;
use crate::store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator};
use crate::types::{LogicalFieldId, Namespace};
use llkv_result::{Error, Result};
use llkv_storage::{
    pager::{BatchGet, GetResult, Pager},
    serialization::deserialize_array,
    types::PhysicalKey,
};
use simd_r_drive_entry_handle::EntryHandle;

pub mod builder;
pub use builder::*;

pub mod kmerge;
pub use kmerge::*;

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

pub mod filter;
pub use filter::*;

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
        let buffers = sorted::load_sorted_buffers(self.pager.as_ref(), &metas)?;
        let first_any =
            llkv_storage::serialization::deserialize_array(buffers.value_handle(0).clone())?;
        with_integer_arrow_type!(
            first_any.data_type().clone(),
            |ArrowTy| {
                <ArrowTy as sorted::SortedDispatch>::visit(
                    self.pager.as_ref(),
                    &metas,
                    &buffers,
                    visitor,
                )
            },
            Err(Error::Internal("unsupported sorted dtype".into())),
        )
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
        let buffers = sorted::load_sorted_buffers(self.pager.as_ref(), &metas)?;
        let first_any =
            llkv_storage::serialization::deserialize_array(buffers.value_handle(0).clone())?;
        with_integer_arrow_type!(
            first_any.data_type().clone(),
            |ArrowTy| {
                <ArrowTy as sorted::SortedDispatch>::visit_rev(
                    self.pager.as_ref(),
                    &metas,
                    &buffers,
                    visitor,
                )
            },
            Err(Error::Internal("unsupported sorted dtype".into())),
        )
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
        let paginate = opts.offset > 0 || opts.limit.is_some();
        if !opts.sorted {
            if opts.with_row_ids {
                let row_fid = field_id.with_namespace(Namespace::RowIdShadow);
                let catalog = self.catalog.read().unwrap();
                if opts.include_nulls {
                    let anchor_fid = opts.anchor_row_id_field.unwrap_or(row_fid);
                    if paginate {
                        let mut pv = crate::store::scan::PaginateVisitor::new(
                            visitor,
                            opts.offset,
                            opts.limit,
                        );
                        return crate::store::scan::unsorted_with_row_ids_and_nulls_visit(
                            self.pager.as_ref(),
                            &catalog.map,
                            field_id,
                            row_fid,
                            anchor_fid,
                            opts.nulls_first,
                            &mut pv,
                        );
                    } else {
                        return crate::store::scan::unsorted_with_row_ids_and_nulls_visit(
                            self.pager.as_ref(),
                            &catalog.map,
                            field_id,
                            row_fid,
                            anchor_fid,
                            opts.nulls_first,
                            visitor,
                        );
                    }
                }
                if paginate {
                    let mut pv =
                        crate::store::scan::PaginateVisitor::new(visitor, opts.offset, opts.limit);
                    return crate::store::scan::unsorted_with_row_ids_visit(
                        self.pager.as_ref(),
                        &catalog.map,
                        field_id,
                        row_fid,
                        &mut pv,
                    );
                } else {
                    return crate::store::scan::unsorted_with_row_ids_visit(
                        self.pager.as_ref(),
                        &catalog.map,
                        field_id,
                        row_fid,
                        visitor,
                    );
                }
            }
            if paginate {
                let mut pv =
                    crate::store::scan::PaginateVisitor::new(visitor, opts.offset, opts.limit);
                return self.scan_visit(field_id, &mut pv);
            } else {
                return self.scan_visit(field_id, visitor);
            }
        }

        if opts.with_row_ids {
            let row_fid = field_id.with_namespace(Namespace::RowIdShadow);
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

            let buffers =
                sorted::load_sorted_buffers_with_rids(self.pager.as_ref(), &metas_val, &metas_rid)?;
            let first_any = llkv_storage::serialization::deserialize_array(
                buffers.base().value_handle(0).clone(),
            )?;
            if opts.reverse {
                if paginate {
                    let mut pv = crate::store::scan::PaginateVisitor::new_with_reverse(
                        visitor,
                        opts.offset,
                        opts.limit,
                        true,
                    );
                    let vals_res = with_integer_arrow_type!(
                        first_any.data_type().clone(),
                        |ArrowTy| {
                            <ArrowTy as sorted::SortedDispatch>::visit_with_rids_rev(
                                self.pager.as_ref(),
                                &metas_val,
                                &metas_rid,
                                &buffers,
                                &mut pv,
                            )
                        },
                        Err(Error::Internal("unsupported sorted dtype".into())),
                    );
                    if opts.include_nulls {
                        let anchor_fid = opts.anchor_row_id_field.unwrap_or(row_fid);
                        let catalog = self.catalog.read().unwrap();
                        let anchor_desc_pk =
                            *catalog.map.get(&anchor_fid).ok_or(Error::NotFound)?;
                        let anchor_desc_blob = self
                            .pager
                            .batch_get(&[BatchGet::Raw {
                                key: anchor_desc_pk,
                            }])?
                            .pop()
                            .and_then(|r| match r {
                                GetResult::Raw { bytes, .. } => Some(bytes),
                                _ => None,
                            })
                            .ok_or(Error::NotFound)?;
                        let anchor_desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(
                            anchor_desc_blob.as_ref(),
                        );
                        drop(catalog);
                        let mut anchor_rids: Vec<UInt64Array> = Vec::new();
                        let mut get_keys: Vec<BatchGet> = Vec::new();
                        let mut anchor_metas: Vec<crate::store::descriptor::ChunkMetadata> =
                            Vec::new();
                        for m in crate::store::descriptor::DescriptorIterator::new(
                            self.pager.as_ref(),
                            anchor_desc.head_page_pk,
                        ) {
                            let mm = m?;
                            if mm.row_count == 0 {
                                continue;
                            }
                            anchor_metas.push(mm);
                            get_keys.push(BatchGet::Raw { key: mm.chunk_pk });
                        }
                        if !anchor_metas.is_empty() {
                            let res = self.pager.batch_get(&get_keys)?;
                            for r in res {
                                if let GetResult::Raw { bytes, .. } = r {
                                    let any = deserialize_array(bytes)?;
                                    let arr = any
                                        .as_any()
                                        .downcast_ref::<UInt64Array>()
                                        .ok_or_else(|| {
                                            Error::Internal("anchor rid not u64".into())
                                        })?
                                        .clone();
                                    anchor_rids.push(arr);
                                }
                            }
                        }
                        let mut present_rids: Vec<UInt64Array> = Vec::new();
                        for mr in &metas_rid {
                            let any = deserialize_array(
                                buffers
                                    .rid_by_pk(mr.chunk_pk)
                                    .ok_or(Error::NotFound)?
                                    .clone(),
                            )?;
                            let arr = any
                                .as_any()
                                .downcast_ref::<UInt64Array>()
                                .ok_or_else(|| Error::Internal("rid not u64".into()))?
                                .clone();
                            present_rids.push(arr);
                        }
                        let mut ai = anchor_rids.len();
                        let mut aj: isize = -1; // reverse traversal
                        let mut pi = present_rids.len();
                        let mut pj: isize = -1;
                        let mut buf: Vec<u64> = Vec::new();
                        let mut flush = |v: &mut Vec<u64>| -> Result<()> {
                            if v.is_empty() {
                                return Ok(());
                            }
                            let arr = UInt64Array::from(std::mem::take(v));
                            let len = arr.len();
                            pv.null_run(&arr, 0, len);
                            Ok(())
                        };
                        // Emit order: nulls_first => nulls then values; else values then nulls
                        if opts.nulls_first {
                            // Emit nulls first (descending rid order)
                            if ai > 0 {
                                ai -= 1;
                                aj = anchor_rids[ai].len() as isize - 1;
                            }
                            if pi > 0 {
                                pi -= 1;
                                pj = present_rids[pi].len() as isize - 1;
                            }
                            while ai < anchor_rids.len() {
                                // ai unsigned; exit when wrap logic ends
                                let a = &anchor_rids[ai];
                                while aj >= 0 {
                                    let av = a.value(aj as usize);
                                    // advance present down to <= av
                                    while pi < present_rids.len() {
                                        let p = &present_rids[pi];
                                        if pj < 0 {
                                            if pi == 0 {
                                                break;
                                            }
                                            pi -= 1;
                                            pj = present_rids[pi].len() as isize - 1;
                                            continue;
                                        }
                                        let pv2 = p.value(pj as usize);
                                        if pv2 > av {
                                            pj -= 1;
                                        } else {
                                            break;
                                        }
                                    }
                                    let present_eq = if pi < present_rids.len() {
                                        let p = &present_rids[pi];
                                        if pj >= 0 {
                                            p.value(pj as usize) == av
                                        } else {
                                            false
                                        }
                                    } else {
                                        false
                                    };
                                    if !present_eq {
                                        buf.push(av);
                                    }
                                    if present_eq {
                                        pj -= 1;
                                    }
                                    aj -= 1;
                                    if buf.len() >= 4096 {
                                        flush(&mut buf)?;
                                    }
                                }
                                if ai == 0 {
                                    break;
                                }
                                ai -= 1;
                                aj = anchor_rids[ai].len() as isize - 1;
                            }
                            flush(&mut buf)?;
                            vals_res
                        } else {
                            vals_res.and_then(|_| {
                                if ai > 0 {
                                    ai -= 1;
                                    aj = anchor_rids[ai].len() as isize - 1;
                                }
                                if pi > 0 {
                                    pi -= 1;
                                    pj = present_rids[pi].len() as isize - 1;
                                }
                                while ai < anchor_rids.len() {
                                    let a = &anchor_rids[ai];
                                    while aj >= 0 {
                                        let av = a.value(aj as usize);
                                        while pi < present_rids.len() {
                                            let p = &present_rids[pi];
                                            if pj < 0 {
                                                if pi == 0 {
                                                    break;
                                                }
                                                pi -= 1;
                                                pj = present_rids[pi].len() as isize - 1;
                                                continue;
                                            }
                                            let pv2 = p.value(pj as usize);
                                            if pv2 > av {
                                                pj -= 1;
                                            } else {
                                                break;
                                            }
                                        }
                                        let present_eq = if pi < present_rids.len() {
                                            let p = &present_rids[pi];
                                            if pj >= 0 {
                                                p.value(pj as usize) == av
                                            } else {
                                                false
                                            }
                                        } else {
                                            false
                                        };
                                        if !present_eq {
                                            buf.push(av);
                                        }
                                        if present_eq {
                                            pj -= 1;
                                        }
                                        aj -= 1;
                                        if buf.len() >= 4096 {
                                            flush(&mut buf)?;
                                        }
                                    }
                                    if ai == 0 {
                                        break;
                                    }
                                    ai -= 1;
                                    aj = anchor_rids[ai].len() as isize - 1;
                                }
                                flush(&mut buf)?;
                                Ok(())
                            })
                        }
                    } else {
                        vals_res
                    }
                } else {
                    with_integer_arrow_type!(
                        first_any.data_type().clone(),
                        |ArrowTy| {
                            <ArrowTy as sorted::SortedDispatch>::visit_with_rids_rev(
                                self.pager.as_ref(),
                                &metas_val,
                                &metas_rid,
                                &buffers,
                                visitor,
                            )
                        },
                        Err(Error::Internal("unsupported sorted dtype".into())),
                    )
                }
            } else if paginate {
                let mut pv =
                    crate::store::scan::PaginateVisitor::new(visitor, opts.offset, opts.limit);
                let vals_res = with_integer_arrow_type!(
                    first_any.data_type().clone(),
                    |ArrowTy| {
                        <ArrowTy as sorted::SortedDispatch>::visit_with_rids(
                            self.pager.as_ref(),
                            &metas_val,
                            &metas_rid,
                            &buffers,
                            &mut pv,
                        )
                    },
                    Err(Error::Internal("unsupported sorted dtype".into())),
                );
                if opts.include_nulls {
                    // nulls_first not meaningful without with_row_ids
                    let anchor_fid = opts.anchor_row_id_field.unwrap_or(row_fid);
                    // Build anchor rid arrays (ascending)
                    let catalog = self.catalog.read().unwrap();
                    let anchor_desc_pk = *catalog.map.get(&anchor_fid).ok_or(Error::NotFound)?;
                    let anchor_desc_blob = self
                        .pager
                        .batch_get(&[BatchGet::Raw {
                            key: anchor_desc_pk,
                        }])?
                        .pop()
                        .and_then(|r| match r {
                            GetResult::Raw { bytes, .. } => Some(bytes),
                            _ => None,
                        })
                        .ok_or(Error::NotFound)?;
                    let anchor_desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(
                        anchor_desc_blob.as_ref(),
                    );
                    drop(catalog);
                    let mut anchor_rids: Vec<UInt64Array> = Vec::new();
                    let mut get_keys: Vec<BatchGet> = Vec::new();
                    let mut anchor_metas: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
                    for m in crate::store::descriptor::DescriptorIterator::new(
                        self.pager.as_ref(),
                        anchor_desc.head_page_pk,
                    ) {
                        let mm = m?;
                        if mm.row_count == 0 {
                            continue;
                        }
                        anchor_metas.push(mm);
                        get_keys.push(BatchGet::Raw { key: mm.chunk_pk });
                    }
                    if !anchor_metas.is_empty() {
                        let res = self.pager.batch_get(&get_keys)?;
                        for r in res {
                            if let GetResult::Raw { bytes, .. } = r {
                                let any = deserialize_array(bytes)?;
                                let arr = any
                                    .as_any()
                                    .downcast_ref::<UInt64Array>()
                                    .ok_or_else(|| Error::Internal("anchor rid not u64".into()))?
                                    .clone();
                                anchor_rids.push(arr);
                            }
                        }
                    }
                    // Build present rids from metas_rid using buffered handles
                    let mut present_rids: Vec<UInt64Array> = Vec::new();
                    for mr in &metas_rid {
                        let any = deserialize_array(
                            buffers
                                .rid_by_pk(mr.chunk_pk)
                                .ok_or(Error::NotFound)?
                                .clone(),
                        )?;
                        let arr = any
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .ok_or_else(|| Error::Internal("rid not u64".into()))?
                            .clone();
                        present_rids.push(arr);
                    }
                    // Two-pointer set-difference to get null rids (ascending)
                    let mut ai = 0usize;
                    let mut aj = 0usize; // anchor chunk/idx
                    let mut pi = 0usize;
                    let mut pj = 0usize; // present chunk/idx
                    let mut buf: Vec<u64> = Vec::new();
                    let mut flush = |v: &mut Vec<u64>| -> Result<()> {
                        if v.is_empty() {
                            return Ok(());
                        }
                        let arr = UInt64Array::from(std::mem::take(v));
                        let len = arr.len();
                        pv.null_run(&arr, 0, len);
                        Ok(())
                    };
                    // Emit order: nulls_first => nulls then values; else values then nulls
                    if opts.nulls_first {
                        // Emit nulls first
                        while ai < anchor_rids.len() {
                            let a = &anchor_rids[ai];
                            while aj < a.len() {
                                let av = a.value(aj);
                                // Advance present to >= av
                                while pi < present_rids.len() {
                                    let p = &present_rids[pi];
                                    if pj >= p.len() {
                                        pi += 1;
                                        pj = 0;
                                        continue;
                                    }
                                    let pv2 = p.value(pj);
                                    if pv2 < av {
                                        pj += 1;
                                    } else {
                                        break;
                                    }
                                }
                                let present_eq = if pi < present_rids.len() {
                                    let p = &present_rids[pi];
                                    if pj < p.len() {
                                        p.value(pj) == av
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                };
                                if !present_eq {
                                    buf.push(av);
                                }
                                if present_eq {
                                    pj += 1;
                                }
                                aj += 1;
                                if buf.len() >= 4096 {
                                    flush(&mut buf)?;
                                }
                            }
                            ai += 1;
                            aj = 0;
                        }
                        flush(&mut buf)?;
                        vals_res
                    } else {
                        vals_res.and_then(|_| {
                            while ai < anchor_rids.len() {
                                let a = &anchor_rids[ai];
                                while aj < a.len() {
                                    let av = a.value(aj);
                                    while pi < present_rids.len() {
                                        let p = &present_rids[pi];
                                        if pj >= p.len() {
                                            pi += 1;
                                            pj = 0;
                                            continue;
                                        }
                                        let pv2 = p.value(pj);
                                        if pv2 < av {
                                            pj += 1;
                                        } else {
                                            break;
                                        }
                                    }
                                    let present_eq = if pi < present_rids.len() {
                                        let p = &present_rids[pi];
                                        if pj < p.len() {
                                            p.value(pj) == av
                                        } else {
                                            false
                                        }
                                    } else {
                                        false
                                    };
                                    if !present_eq {
                                        buf.push(av);
                                    }
                                    if present_eq {
                                        pj += 1;
                                    }
                                    aj += 1;
                                    if buf.len() >= 4096 {
                                        flush(&mut buf)?;
                                    }
                                }
                                ai += 1;
                                aj = 0;
                            }
                            flush(&mut buf)?;
                            Ok(())
                        })
                    }
                } else {
                    vals_res
                }
            } else if opts.include_nulls && opts.nulls_first {
                let anchor_fid = opts.anchor_row_id_field.ok_or_else(|| {
                    Error::Internal("anchor_row_id_field required when include_nulls=true".into())
                })?;
                let catalog = self.catalog.read().unwrap();
                let anchor_desc_pk = *catalog.map.get(&anchor_fid).ok_or(Error::NotFound)?;
                let anchor_desc_blob = self
                    .pager
                    .batch_get(&[BatchGet::Raw {
                        key: anchor_desc_pk,
                    }])?
                    .pop()
                    .and_then(|r| match r {
                        GetResult::Raw { bytes, .. } => Some(bytes),
                        _ => None,
                    })
                    .ok_or(Error::NotFound)?;
                let anchor_desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(
                    anchor_desc_blob.as_ref(),
                );
                drop(catalog);
                let mut anchor_rids: Vec<UInt64Array> = Vec::new();
                let mut get_keys: Vec<BatchGet> = Vec::new();
                let mut anchor_metas: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
                for m in crate::store::descriptor::DescriptorIterator::new(
                    self.pager.as_ref(),
                    anchor_desc.head_page_pk,
                ) {
                    let mm = m?;
                    if mm.row_count == 0 {
                        continue;
                    }
                    anchor_metas.push(mm);
                    get_keys.push(BatchGet::Raw { key: mm.chunk_pk });
                }
                if !anchor_metas.is_empty() {
                    let res = self.pager.batch_get(&get_keys)?;
                    for r in res {
                        if let GetResult::Raw { bytes, .. } = r {
                            let any = deserialize_array(bytes)?;
                            let arr = any
                                .as_any()
                                .downcast_ref::<UInt64Array>()
                                .ok_or_else(|| Error::Internal("anchor rid not u64".into()))?
                                .clone();
                            anchor_rids.push(arr);
                        }
                    }
                }
                let mut present_rids: Vec<UInt64Array> = Vec::new();
                for mr in &metas_rid {
                    let any = deserialize_array(
                        buffers
                            .rid_by_pk(mr.chunk_pk)
                            .ok_or(Error::NotFound)?
                            .clone(),
                    )?;
                    let arr = any
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| Error::Internal("rid not u64".into()))?
                        .clone();
                    present_rids.push(arr);
                }
                // Emit nulls first
                let mut ai = 0usize;
                let mut aj = 0usize;
                let mut pi = 0usize;
                let mut pj = 0usize;
                let mut buf: Vec<u64> = Vec::new();
                let mut flush = |v: &mut Vec<u64>| -> Result<()> {
                    if v.is_empty() {
                        return Ok(());
                    }
                    let arr = UInt64Array::from(std::mem::take(v));
                    let len = arr.len();
                    visitor.null_run(&arr, 0, len);
                    Ok(())
                };
                while ai < anchor_rids.len() {
                    let a = &anchor_rids[ai];
                    while aj < a.len() {
                        let av = a.value(aj);
                        while pi < present_rids.len() {
                            let p = &present_rids[pi];
                            if pj >= p.len() {
                                pi += 1;
                                pj = 0;
                                continue;
                            }
                            let pv2 = p.value(pj);
                            if pv2 < av {
                                pj += 1;
                            } else {
                                break;
                            }
                        }
                        let present_eq = if pi < present_rids.len() {
                            let p = &present_rids[pi];
                            if pj < p.len() {
                                p.value(pj) == av
                            } else {
                                false
                            }
                        } else {
                            false
                        };
                        if !present_eq {
                            buf.push(av);
                        }
                        if present_eq {
                            pj += 1;
                        }
                        aj += 1;
                        if buf.len() >= 4096 {
                            flush(&mut buf)?;
                        }
                    }
                    ai += 1;
                    aj = 0;
                }
                flush(&mut buf)?;
                // Then emit values
                with_integer_arrow_type!(
                    first_any.data_type().clone(),
                    |ArrowTy| {
                        <ArrowTy as sorted::SortedDispatch>::visit_with_rids(
                            self.pager.as_ref(),
                            &metas_val,
                            &metas_rid,
                            &buffers,
                            visitor,
                        )
                    },
                    Err(Error::Internal("unsupported sorted dtype".into())),
                )
            } else {
                // Values first
                let res = with_integer_arrow_type!(
                    first_any.data_type().clone(),
                    |ArrowTy| {
                        <ArrowTy as sorted::SortedDispatch>::visit_with_rids(
                            self.pager.as_ref(),
                            &metas_val,
                            &metas_rid,
                            &buffers,
                            visitor,
                        )
                    },
                    Err(Error::Internal("unsupported sorted dtype".into())),
                );
                if opts.include_nulls {
                    res?;
                    let anchor_fid = opts.anchor_row_id_field.ok_or_else(|| {
                        Error::Internal(
                            "anchor_row_id_field required when include_nulls=true".into(),
                        )
                    })?;
                    let catalog = self.catalog.read().unwrap();
                    let anchor_desc_pk = *catalog.map.get(&anchor_fid).ok_or(Error::NotFound)?;
                    let anchor_desc_blob = self
                        .pager
                        .batch_get(&[BatchGet::Raw {
                            key: anchor_desc_pk,
                        }])?
                        .pop()
                        .and_then(|r| match r {
                            GetResult::Raw { bytes, .. } => Some(bytes),
                            _ => None,
                        })
                        .ok_or(Error::NotFound)?;
                    let anchor_desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(
                        anchor_desc_blob.as_ref(),
                    );
                    drop(catalog);
                    let mut anchor_rids: Vec<UInt64Array> = Vec::new();
                    let mut get_keys: Vec<BatchGet> = Vec::new();
                    let mut anchor_metas: Vec<crate::store::descriptor::ChunkMetadata> = Vec::new();
                    for m in crate::store::descriptor::DescriptorIterator::new(
                        self.pager.as_ref(),
                        anchor_desc.head_page_pk,
                    ) {
                        let mm = m?;
                        if mm.row_count == 0 {
                            continue;
                        }
                        anchor_metas.push(mm);
                        get_keys.push(BatchGet::Raw { key: mm.chunk_pk });
                    }
                    if !anchor_metas.is_empty() {
                        let res_get = self.pager.batch_get(&get_keys)?;
                        for r in res_get {
                            if let GetResult::Raw { bytes, .. } = r {
                                let any = deserialize_array(bytes)?;
                                let arr = any
                                    .as_any()
                                    .downcast_ref::<UInt64Array>()
                                    .ok_or_else(|| Error::Internal("anchor rid not u64".into()))?
                                    .clone();
                                anchor_rids.push(arr);
                            }
                        }
                    }
                    let mut present_rids: Vec<UInt64Array> = Vec::new();
                    for mr in &metas_rid {
                        let any = deserialize_array(
                            buffers
                                .rid_by_pk(mr.chunk_pk)
                                .ok_or(Error::NotFound)?
                                .clone(),
                        )?;
                        let arr = any
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .ok_or_else(|| Error::Internal("rid not u64".into()))?
                            .clone();
                        present_rids.push(arr);
                    }
                    let mut ai = 0usize;
                    let mut aj = 0usize;
                    let mut pi = 0usize;
                    let mut pj = 0usize;
                    let mut buf: Vec<u64> = Vec::new();
                    let mut flush = |v: &mut Vec<u64>| -> Result<()> {
                        if v.is_empty() {
                            return Ok(());
                        }
                        let arr = UInt64Array::from(std::mem::take(v));
                        let len = arr.len();
                        visitor.null_run(&arr, 0, len);
                        Ok(())
                    };
                    while ai < anchor_rids.len() {
                        let a = &anchor_rids[ai];
                        while aj < a.len() {
                            let av = a.value(aj);
                            while pi < present_rids.len() {
                                let p = &present_rids[pi];
                                if pj >= p.len() {
                                    pi += 1;
                                    pj = 0;
                                    continue;
                                }
                                let pv2 = p.value(pj);
                                if pv2 < av {
                                    pj += 1;
                                } else {
                                    break;
                                }
                            }
                            let present_eq = if pi < present_rids.len() {
                                let p = &present_rids[pi];
                                if pj < p.len() {
                                    p.value(pj) == av
                                } else {
                                    false
                                }
                            } else {
                                false
                            };
                            if !present_eq {
                                buf.push(av);
                            }
                            if present_eq {
                                pj += 1;
                            }
                            aj += 1;
                            if buf.len() >= 4096 {
                                flush(&mut buf)?;
                            }
                        }
                        ai += 1;
                        aj = 0;
                    }
                    flush(&mut buf)?;
                    Ok(())
                } else {
                    res
                }
            }
        } else if opts.reverse {
            if paginate {
                let mut pv = crate::store::scan::PaginateVisitor::new_with_reverse(
                    visitor,
                    opts.offset,
                    opts.limit,
                    true,
                );
                self.scan_sorted_visit_reverse(field_id, &mut pv)
            } else {
                self.scan_sorted_visit_reverse(field_id, visitor)
            }
        } else if paginate {
            let mut pv = crate::store::scan::PaginateVisitor::new_with_reverse(
                visitor,
                opts.offset,
                opts.limit,
                false,
            );
            self.scan_sorted_visit(field_id, &mut pv)
        } else {
            self.scan_sorted_visit(field_id, visitor)
        }
    }
}
