use super::*;

pub fn unsorted_visit<P: Pager<Blob = EntryHandle>, V: PrimitiveVisitor>(
    pager: &P,
    catalog: &FxHashMap<LogicalFieldId, PhysicalKey>,
    field_id: LogicalFieldId,
    visitor: &mut V,
) -> Result<()> {
    let descriptor_pk = *catalog.get(&field_id).ok_or(Error::NotFound)?;
    let desc_blob = pager
        .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
        .pop()
        .and_then(|r| match r {
            GetResult::Raw { bytes, .. } => Some(bytes),
            _ => None,
        })
        .ok_or(Error::NotFound)?;
    let desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

    // Gather metas and fetch blobs in one batch.
    let mut metas: Vec<ChunkMetadata> = Vec::new();
    for m in DescriptorIterator::new(pager, desc.head_page_pk) {
        let meta = m?;
        if meta.row_count > 0 {
            metas.push(meta);
        }
    }
    if metas.is_empty() {
        return Ok(());
    }
    let gets: Vec<BatchGet> = metas
        .iter()
        .map(|m| BatchGet::Raw { key: m.chunk_pk })
        .collect();
    let results = pager.batch_get(&gets)?;
    let mut blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    for r in results {
        if let GetResult::Raw { key, bytes } = r {
            blobs.insert(key, bytes);
        }
    }

    // Inspect dtype of first chunk to monomorphize the loop.
    let first_any = deserialize_array(
        blobs
            .get(&metas[0].chunk_pk)
            .ok_or(Error::NotFound)?
            .clone(),
    )?;
    match first_any.data_type() {
        DataType::UInt64 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast UInt64".into()))?;
                visitor.u64_chunk(a);
            }
            Ok(())
        }
        DataType::UInt32 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("downcast UInt32".into()))?;
                visitor.u32_chunk(a);
            }
            Ok(())
        }
        DataType::UInt16 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<UInt16Array>()
                    .ok_or_else(|| Error::Internal("downcast UInt16".into()))?;
                visitor.u16_chunk(a);
            }
            Ok(())
        }
        DataType::UInt8 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .ok_or_else(|| Error::Internal("downcast UInt8".into()))?;
                visitor.u8_chunk(a);
            }
            Ok(())
        }
        DataType::Int64 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Internal("downcast Int64".into()))?;
                visitor.i64_chunk(a);
            }
            Ok(())
        }
        DataType::Int32 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| Error::Internal("downcast Int32".into()))?;
                visitor.i32_chunk(a);
            }
            Ok(())
        }
        DataType::Int16 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<Int16Array>()
                    .ok_or_else(|| Error::Internal("downcast Int16".into()))?;
                visitor.i16_chunk(a);
            }
            Ok(())
        }
        DataType::Int8 => {
            for m in metas {
                let a_any = deserialize_array(blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
                let a = a_any
                    .as_any()
                    .downcast_ref::<Int8Array>()
                    .ok_or_else(|| Error::Internal("downcast Int8".into()))?;
                visitor.i8_chunk(a);
            }
            Ok(())
        }
        _ => Err(Error::Internal("unsorted_visit: unsupported dtype".into())),
    }
}

pub fn unsorted_with_row_ids_visit<P: Pager<Blob = EntryHandle>, V: PrimitiveWithRowIdsVisitor>(
    pager: &P,
    catalog: &FxHashMap<LogicalFieldId, PhysicalKey>,
    value_fid: LogicalFieldId,
    rowid_fid: LogicalFieldId,
    visitor: &mut V,
) -> Result<()> {
    let v_pk = *catalog.get(&value_fid).ok_or(Error::NotFound)?;
    let r_pk = *catalog.get(&rowid_fid).ok_or(Error::NotFound)?;

    let v_desc_blob = pager
        .batch_get(&[BatchGet::Raw { key: v_pk }])?
        .pop()
        .and_then(|r| match r {
            GetResult::Raw { bytes, .. } => Some(bytes),
            _ => None,
        })
        .ok_or(Error::NotFound)?;
    let r_desc_blob = pager
        .batch_get(&[BatchGet::Raw { key: r_pk }])?
        .pop()
        .and_then(|r| match r {
            GetResult::Raw { bytes, .. } => Some(bytes),
            _ => None,
        })
        .ok_or(Error::NotFound)?;
    let v_desc = ColumnDescriptor::from_le_bytes(v_desc_blob.as_ref());
    let r_desc = ColumnDescriptor::from_le_bytes(r_desc_blob.as_ref());

    let mut v_metas: Vec<ChunkMetadata> = Vec::new();
    for m in DescriptorIterator::new(pager, v_desc.head_page_pk) {
        let meta = m?;
        if meta.row_count > 0 {
            v_metas.push(meta);
        }
    }
    let mut r_metas: Vec<ChunkMetadata> = Vec::new();
    for m in DescriptorIterator::new(pager, r_desc.head_page_pk) {
        let meta = m?;
        if meta.row_count > 0 {
            r_metas.push(meta);
        }
    }
    if v_metas.len() != r_metas.len() {
        return Err(Error::Internal(
            "unsorted_with_row_ids: chunk count mismatch".into(),
        ));
    }
    if v_metas.is_empty() {
        return Ok(());
    }

    let mut gets: Vec<BatchGet> = Vec::with_capacity(v_metas.len() * 2);
    for (vm, rm) in v_metas.iter().zip(r_metas.iter()) {
        gets.push(BatchGet::Raw { key: vm.chunk_pk });
        gets.push(BatchGet::Raw { key: rm.chunk_pk });
    }
    let results = pager.batch_get(&gets)?;
    let mut vals_blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    let mut rids_blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    for r in results {
        if let GetResult::Raw { key, bytes } = r {
            if r_metas.iter().any(|m| m.chunk_pk == key) {
                rids_blobs.insert(key, bytes);
            } else {
                vals_blobs.insert(key, bytes);
            }
        }
    }

    let first_any = deserialize_array(
        vals_blobs
            .get(&v_metas[0].chunk_pk)
            .ok_or(Error::NotFound)?
            .clone(),
    )?;
    let r_first_any = deserialize_array(
        rids_blobs
            .get(&r_metas[0].chunk_pk)
            .ok_or(Error::NotFound)?
            .clone(),
    )?;
    let _rids = r_first_any
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| Error::Internal("row_id array must be UInt64".into()))?;
    match first_any.data_type() {
        DataType::UInt64 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast u64".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.u64_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::UInt32 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("downcast u32".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.u32_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::UInt16 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<UInt16Array>()
                    .ok_or_else(|| Error::Internal("downcast u16".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.u16_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::UInt8 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .ok_or_else(|| Error::Internal("downcast u8".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.u8_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::Int64 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Internal("downcast i64".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.i64_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::Int32 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| Error::Internal("downcast i32".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.i32_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::Int16 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<Int16Array>()
                    .ok_or_else(|| Error::Internal("downcast i16".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.i16_chunk_with_rids(v, r);
            }
            Ok(())
        }
        DataType::Int8 => {
            for (vm, rm) in v_metas.into_iter().zip(r_metas.into_iter()) {
                let va =
                    deserialize_array(vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
                let ra =
                    deserialize_array(rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
                let v = va
                    .as_any()
                    .downcast_ref::<Int8Array>()
                    .ok_or_else(|| Error::Internal("downcast i8".into()))?;
                let r = ra
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
                visitor.i8_chunk_with_rids(v, r);
            }
            Ok(())
        }
        _ => Err(Error::Internal(
            "unsorted_with_row_ids: unsupported dtype".into(),
        )),
    }
}

pub fn unsorted_with_row_ids_and_nulls_visit<
    P: Pager<Blob = EntryHandle>,
    V: PrimitiveWithRowIdsVisitor + PrimitiveSortedWithRowIdsVisitor,
>(
    pager: &P,
    catalog: &FxHashMap<LogicalFieldId, PhysicalKey>,
    value_fid: LogicalFieldId,
    rowid_fid: LogicalFieldId,
    anchor_rowid_fid: LogicalFieldId,
    _nulls_first: bool, // Anchor order interleave; nulls_first ignored for unsorted
    visitor: &mut V,
) -> Result<()> {
    let v_pk = *catalog.get(&value_fid).ok_or(Error::NotFound)?;
    let r_pk = *catalog.get(&rowid_fid).ok_or(Error::NotFound)?;
    let a_pk = *catalog.get(&anchor_rowid_fid).ok_or(Error::NotFound)?;

    // Load descriptors
    let v_desc = {
        let b = pager
            .batch_get(&[BatchGet::Raw { key: v_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        ColumnDescriptor::from_le_bytes(b.as_ref())
    };
    let r_desc = {
        let b = pager
            .batch_get(&[BatchGet::Raw { key: r_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        ColumnDescriptor::from_le_bytes(b.as_ref())
    };
    let a_desc = {
        let b = pager
            .batch_get(&[BatchGet::Raw { key: a_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        ColumnDescriptor::from_le_bytes(b.as_ref())
    };

    // Gather metas and blobs
    let mut v_metas: Vec<ChunkMetadata> = Vec::new();
    for m in DescriptorIterator::new(pager, v_desc.head_page_pk) {
        let meta = m?;
        if meta.row_count > 0 {
            v_metas.push(meta);
        }
    }
    let mut r_metas: Vec<ChunkMetadata> = Vec::new();
    for m in DescriptorIterator::new(pager, r_desc.head_page_pk) {
        let meta = m?;
        if meta.row_count > 0 {
            r_metas.push(meta);
        }
    }
    let mut a_metas: Vec<ChunkMetadata> = Vec::new();
    for m in DescriptorIterator::new(pager, a_desc.head_page_pk) {
        let meta = m?;
        if meta.row_count > 0 {
            a_metas.push(meta);
        }
    }
    if v_metas.is_empty() || r_metas.is_empty() || a_metas.is_empty() {
        return Ok(());
    }
    if v_metas.len() != r_metas.len() {
        return Err(Error::Internal(
            "unsorted_with_nulls: chunk count mismatch".into(),
        ));
    }

    // Batch get
    let mut gets: Vec<BatchGet> = Vec::with_capacity(v_metas.len() * 2 + a_metas.len());
    for (vm, rm) in v_metas.iter().zip(r_metas.iter()) {
        gets.push(BatchGet::Raw { key: vm.chunk_pk });
        gets.push(BatchGet::Raw { key: rm.chunk_pk });
    }
    for am in &a_metas {
        gets.push(BatchGet::Raw { key: am.chunk_pk });
    }
    let results = pager.batch_get(&gets)?;
    let mut vblobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    let mut rblobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    let mut ablobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
    for r in results {
        if let GetResult::Raw { key, bytes } = r {
            if v_metas.iter().any(|m| m.chunk_pk == key) {
                vblobs.insert(key, bytes);
            } else if r_metas.iter().any(|m| m.chunk_pk == key) {
                rblobs.insert(key, bytes);
            } else {
                ablobs.insert(key, bytes);
            }
        }
    }

    // Determine dtype
    let first_any = deserialize_array(
        vblobs
            .get(&v_metas[0].chunk_pk)
            .ok_or(Error::NotFound)?
            .clone(),
    )?;
    match first_any.data_type() {
        DataType::UInt64 => {
            let mut ai = 0usize;
            let mut aj = 0usize;
            let mut pi = 0usize;
            let mut pj = 0usize;
            // Materialize typed arrays
            let mut vals: Vec<UInt64Array> = Vec::with_capacity(v_metas.len());
            let mut prids: Vec<UInt64Array> = Vec::with_capacity(r_metas.len());
            let mut anchors: Vec<UInt64Array> = Vec::with_capacity(a_metas.len());
            for vm in &v_metas {
                let any =
                    deserialize_array(vblobs.get(&vm.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                vals.push(
                    any.as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| Error::Internal("downcast".into()))?
                        .clone(),
                );
            }
            for rm in &r_metas {
                let any =
                    deserialize_array(rblobs.get(&rm.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                prids.push(
                    any.as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| Error::Internal("downcast".into()))?
                        .clone(),
                );
            }
            for am in &a_metas {
                let any =
                    deserialize_array(ablobs.get(&am.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                anchors.push(
                    any.as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| Error::Internal("downcast".into()))?
                        .clone(),
                );
            }
            let mut null_buf: Vec<u64> = Vec::new();
            while ai < anchors.len() {
                let a = &anchors[ai];
                while aj < a.len() {
                    let av = a.value(aj);
                    // advance pres to >= av
                    while pi < prids.len() {
                        let p = &prids[pi];
                        if pj >= p.len() {
                            pi += 1;
                            pj = 0;
                            continue;
                        }
                        let pv = p.value(pj);
                        if pv < av {
                            pj += 1;
                        } else {
                            break;
                        }
                    }
                    let present_eq = if pi < prids.len() {
                        let p = &prids[pi];
                        pj < p.len() && p.value(pj) == av
                    } else {
                        false
                    };
                    if present_eq {
                        if !null_buf.is_empty() {
                            let arr = UInt64Array::from(std::mem::take(&mut null_buf));
                            visitor.null_run(&arr, 0, arr.len());
                        }
                        // emit single present value slice
                        let v = &vals[pi];
                        let r = &prids[pi];
                        let sref_v = v.slice(pj, 1);
                        let sref_r = r.slice(pj, 1);
                        let sv = sref_v.as_any().downcast_ref::<UInt64Array>().unwrap();
                        let sr = sref_r.as_any().downcast_ref::<UInt64Array>().unwrap();
                        visitor.u64_chunk_with_rids(sv, sr);
                        pj += 1;
                    } else {
                        null_buf.push(av);
                        if null_buf.len() >= 4096 {
                            let arr = UInt64Array::from(std::mem::take(&mut null_buf));
                            visitor.null_run(&arr, 0, arr.len());
                        }
                    }
                    aj += 1;
                }
                ai += 1;
                aj = 0;
            }
            if !null_buf.is_empty() {
                let arr = UInt64Array::from(std::mem::take(&mut null_buf));
                visitor.null_run(&arr, 0, arr.len());
            }
            Ok(())
        }
        // Fallback: emit present in chunk order and nulls interleaved as found; for other dtypes, we can map to u64 anchor
        _ => Err(Error::Internal(
            "unsorted_with_nulls: dtype not supported".into(),
        )),
    }
}
