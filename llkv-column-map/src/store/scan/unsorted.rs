use super::*;
use arrow::datatypes::DataType;

macro_rules! unsorted_visit_arm {
    ($array:ty, $method:ident, $err:literal, $metas:ident, $blobs:ident, $visitor:ident) => {{
        for m in $metas {
            let a_any = deserialize_array($blobs.remove(&m.chunk_pk).ok_or(Error::NotFound)?)?;
            let a = a_any
                .as_any()
                .downcast_ref::<$array>()
                .ok_or_else(|| Error::Internal($err.into()))?;
            $visitor.$method(a);
        }
        Ok(())
    }};
}

macro_rules! unsorted_with_rids_arm {
    (
        $array:ty,
        $method:ident,
        $err:literal,
        $v_metas:ident,
        $r_metas:ident,
        $vals_blobs:ident,
        $rids_blobs:ident,
        $visitor:ident
    ) => {{
        for (vm, rm) in $v_metas.into_iter().zip($r_metas.into_iter()) {
            let va = deserialize_array($vals_blobs.remove(&vm.chunk_pk).ok_or(Error::NotFound)?)?;
            let ra = deserialize_array($rids_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?)?;
            let v = va
                .as_any()
                .downcast_ref::<$array>()
                .ok_or_else(|| Error::Internal($err.into()))?;
            let r = ra
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?;
            $visitor.$method(v, r);
        }
        Ok(())
    }};
}

macro_rules! dispatch_unsorted_visit {
    ($dtype:expr, $metas:ident, $blobs:ident, $visitor:ident) => {{
        match $dtype {
            DataType::UInt64 => unsorted_visit_arm!(
                UInt64Array,
                u64_chunk,
                "downcast UInt64",
                $metas,
                $blobs,
                $visitor
            ),
            DataType::UInt32 => unsorted_visit_arm!(
                UInt32Array,
                u32_chunk,
                "downcast UInt32",
                $metas,
                $blobs,
                $visitor
            ),
            DataType::UInt16 => unsorted_visit_arm!(
                UInt16Array,
                u16_chunk,
                "downcast UInt16",
                $metas,
                $blobs,
                $visitor
            ),
            DataType::UInt8 => unsorted_visit_arm!(
                UInt8Array,
                u8_chunk,
                "downcast UInt8",
                $metas,
                $blobs,
                $visitor
            ),
            DataType::Int64 => unsorted_visit_arm!(
                Int64Array,
                i64_chunk,
                "downcast Int64",
                $metas,
                $blobs,
                $visitor
            ),
            DataType::Int32 => unsorted_visit_arm!(
                Int32Array,
                i32_chunk,
                "downcast Int32",
                $metas,
                $blobs,
                $visitor
            ),
            DataType::Int16 => unsorted_visit_arm!(
                Int16Array,
                i16_chunk,
                "downcast Int16",
                $metas,
                $blobs,
                $visitor
            ),
            DataType::Int8 => unsorted_visit_arm!(
                Int8Array,
                i8_chunk,
                "downcast Int8",
                $metas,
                $blobs,
                $visitor
            ),
            DataType::Float64 => unsorted_visit_arm!(
                Float64Array,
                f64_chunk,
                "downcast Float64",
                $metas,
                $blobs,
                $visitor
            ),
            DataType::Float32 => unsorted_visit_arm!(
                Float32Array,
                f32_chunk,
                "downcast Float32",
                $metas,
                $blobs,
                $visitor
            ),
            _ => Err(Error::Internal("unsorted_visit: unsupported dtype".into())),
        }
    }};
}

macro_rules! dispatch_unsorted_with_rids_visit {
    (
        $dtype:expr,
        $v_metas:ident,
        $r_metas:ident,
        $vals_blobs:ident,
        $rids_blobs:ident,
        $visitor:ident
    ) => {{
        match $dtype {
            DataType::UInt64 => unsorted_with_rids_arm!(
                UInt64Array,
                u64_chunk_with_rids,
                "downcast u64",
                $v_metas,
                $r_metas,
                $vals_blobs,
                $rids_blobs,
                $visitor
            ),
            DataType::UInt32 => unsorted_with_rids_arm!(
                UInt32Array,
                u32_chunk_with_rids,
                "downcast u32",
                $v_metas,
                $r_metas,
                $vals_blobs,
                $rids_blobs,
                $visitor
            ),
            DataType::UInt16 => unsorted_with_rids_arm!(
                UInt16Array,
                u16_chunk_with_rids,
                "downcast u16",
                $v_metas,
                $r_metas,
                $vals_blobs,
                $rids_blobs,
                $visitor
            ),
            DataType::UInt8 => unsorted_with_rids_arm!(
                UInt8Array,
                u8_chunk_with_rids,
                "downcast u8",
                $v_metas,
                $r_metas,
                $vals_blobs,
                $rids_blobs,
                $visitor
            ),
            DataType::Int64 => unsorted_with_rids_arm!(
                Int64Array,
                i64_chunk_with_rids,
                "downcast i64",
                $v_metas,
                $r_metas,
                $vals_blobs,
                $rids_blobs,
                $visitor
            ),
            DataType::Int32 => unsorted_with_rids_arm!(
                Int32Array,
                i32_chunk_with_rids,
                "downcast i32",
                $v_metas,
                $r_metas,
                $vals_blobs,
                $rids_blobs,
                $visitor
            ),
            DataType::Int16 => unsorted_with_rids_arm!(
                Int16Array,
                i16_chunk_with_rids,
                "downcast i16",
                $v_metas,
                $r_metas,
                $vals_blobs,
                $rids_blobs,
                $visitor
            ),
            DataType::Int8 => unsorted_with_rids_arm!(
                Int8Array,
                i8_chunk_with_rids,
                "downcast i8",
                $v_metas,
                $r_metas,
                $vals_blobs,
                $rids_blobs,
                $visitor
            ),
            DataType::Float64 => unsorted_with_rids_arm!(
                Float64Array,
                f64_chunk_with_rids,
                "downcast f64",
                $v_metas,
                $r_metas,
                $vals_blobs,
                $rids_blobs,
                $visitor
            ),
            DataType::Float32 => unsorted_with_rids_arm!(
                Float32Array,
                f32_chunk_with_rids,
                "downcast f32",
                $v_metas,
                $r_metas,
                $vals_blobs,
                $rids_blobs,
                $visitor
            ),
            _ => Err(Error::Internal(
                "unsorted_with_row_ids: unsupported dtype".into(),
            )),
        }
    }};
}

macro_rules! dispatch_unsorted_nulls {
    ($dtype:expr) => {{
        match $dtype {
            DataType::UInt64 => {
                emit_unsorted_nulls!(UInt64Array, u64_chunk_with_rids, "downcast u64")
            }
            DataType::UInt32 => {
                emit_unsorted_nulls!(UInt32Array, u32_chunk_with_rids, "downcast u32")
            }
            DataType::UInt16 => {
                emit_unsorted_nulls!(UInt16Array, u16_chunk_with_rids, "downcast u16")
            }
            DataType::UInt8 => emit_unsorted_nulls!(UInt8Array, u8_chunk_with_rids, "downcast u8"),
            DataType::Int64 => {
                emit_unsorted_nulls!(Int64Array, i64_chunk_with_rids, "downcast i64")
            }
            DataType::Int32 => {
                emit_unsorted_nulls!(Int32Array, i32_chunk_with_rids, "downcast i32")
            }
            DataType::Int16 => {
                emit_unsorted_nulls!(Int16Array, i16_chunk_with_rids, "downcast i16")
            }
            DataType::Int8 => emit_unsorted_nulls!(Int8Array, i8_chunk_with_rids, "downcast i8"),
            DataType::Float64 => {
                emit_unsorted_nulls!(Float64Array, f64_chunk_with_rids, "downcast f64")
            }
            DataType::Float32 => {
                emit_unsorted_nulls!(Float32Array, f32_chunk_with_rids, "downcast f32")
            }
            _ => Err(Error::Internal(
                "unsorted_with_nulls: dtype not supported".into(),
            )),
        }
    }};
}

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
    dispatch_unsorted_visit!(first_any.data_type(), metas, blobs, visitor)
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
    dispatch_unsorted_with_rids_visit!(
        first_any.data_type(),
        v_metas,
        r_metas,
        vals_blobs,
        rids_blobs,
        visitor
    )
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
    macro_rules! emit_unsorted_nulls {
        ($array_ty:ty, $visit:ident, $err:literal) => {{
            let mut ai = 0usize;
            let mut aj = 0usize;
            let mut pi = 0usize;
            let mut pj = 0usize;
            let mut vals: Vec<$array_ty> = Vec::with_capacity(v_metas.len());
            let mut prids: Vec<UInt64Array> = Vec::with_capacity(r_metas.len());
            let mut anchors: Vec<UInt64Array> = Vec::with_capacity(a_metas.len());
            for vm in &v_metas {
                let any =
                    deserialize_array(vblobs.get(&vm.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                vals.push(
                    any.as_any()
                        .downcast_ref::<$array_ty>()
                        .ok_or_else(|| Error::Internal($err.into()))?
                        .clone(),
                );
            }
            for rm in &r_metas {
                let any =
                    deserialize_array(rblobs.get(&rm.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                prids.push(
                    any.as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| Error::Internal("downcast row_id u64".into()))?
                        .clone(),
                );
            }
            for am in &a_metas {
                let any =
                    deserialize_array(ablobs.get(&am.chunk_pk).ok_or(Error::NotFound)?.clone())?;
                anchors.push(
                    any.as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| Error::Internal("downcast anchor u64".into()))?
                        .clone(),
                );
            }
            let mut null_buf: Vec<u64> = Vec::new();
            while ai < anchors.len() {
                let a = &anchors[ai];
                while aj < a.len() {
                    let av = a.value(aj);
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
                        let v = &vals[pi];
                        let r = &prids[pi];
                        let sref_v = v.slice(pj, 1);
                        let sref_r = r.slice(pj, 1);
                        let sv = sref_v.as_any().downcast_ref::<$array_ty>().unwrap();
                        let sr = sref_r.as_any().downcast_ref::<UInt64Array>().unwrap();
                        visitor.$visit(sv, sr);
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
        }};
    }

    dispatch_unsorted_nulls!(first_any.data_type())
}
