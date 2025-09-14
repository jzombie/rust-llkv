//! Minimal manual codecs (little-endian). All large vectors are paged.

use crate::column_index::{
    Bootstrap, ColumnEntry, ColumnIndex, IndexSegment, IndexSegmentRef, Manifest, ValueDirL2,
    ValueIndex,
};
use crate::column_store::{ColumnarChunkRef, ColumnarDescriptor, ColumnarRegistry};
use crate::layout::{KeyLayout, ValueLayout};
use crate::types::{TypedKind, TypedValue};
use std::io::{self, Error, ErrorKind};

/// Small helpers

fn put_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_u32(out, b.len() as u32);
    out.extend_from_slice(b);
}

fn get_u8(inp: &mut &[u8]) -> io::Result<u8> {
    if inp.len() < 1 {
        return Err(Error::new(ErrorKind::UnexpectedEof, "u8"));
    }
    let v = inp[0];
    *inp = &inp[1..];
    Ok(v)
}

fn get_u32(inp: &mut &[u8]) -> io::Result<u32> {
    if inp.len() < 4 {
        return Err(Error::new(ErrorKind::UnexpectedEof, "u32"));
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&inp[..4]);
    *inp = &inp[4..];
    Ok(u32::from_le_bytes(b))
}

fn get_u64(inp: &mut &[u8]) -> io::Result<u64> {
    if inp.len() < 8 {
        return Err(Error::new(ErrorKind::UnexpectedEof, "u64"));
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&inp[..8]);
    *inp = &inp[8..];
    Ok(u64::from_le_bytes(b))
}

fn get_bytes(inp: &mut &[u8]) -> io::Result<Vec<u8>> {
    let n = get_u32(inp)? as usize;
    if inp.len() < n {
        return Err(Error::new(ErrorKind::UnexpectedEof, "bytes"));
    }
    let v = inp[..n].to_vec();
    *inp = &inp[n..];
    Ok(v)
}

/// KeyLayout

fn enc_key_layout(out: &mut Vec<u8>, kl: &KeyLayout) {
    match kl {
        KeyLayout::FixedWidth { width } => {
            put_u8(out, 0);
            put_u32(out, *width as u32);
        }
        KeyLayout::Variable { key_offsets } => {
            put_u8(out, 1);
            put_u32(out, key_offsets.len() as u32);
            for &o in key_offsets {
                put_u32(out, o as u32);
            }
        }
        KeyLayout::VariablePaged {
            offsets_pk,
            n_entries,
        } => {
            put_u8(out, 2);
            put_u64(out, *offsets_pk);
            put_u32(out, *n_entries as u32);
        }
    }
}

fn dec_key_layout(inp: &mut &[u8]) -> io::Result<KeyLayout> {
    match get_u8(inp)? {
        0 => Ok(KeyLayout::FixedWidth {
            width: get_u32(inp)?,
        }),
        1 => {
            let n = get_u32(inp)? as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(get_u32(inp)?);
            }
            Ok(KeyLayout::Variable { key_offsets: v })
        }
        2 => {
            let offsets_pk = get_u64(inp)?;
            let n_entries = get_u32(inp)?;
            Ok(KeyLayout::VariablePaged {
                offsets_pk,
                n_entries,
            })
        }
        _ => Err(Error::new(ErrorKind::InvalidData, "KeyLayout tag")),
    }
}

/// ValueLayout

fn enc_value_layout(out: &mut Vec<u8>, vl: &ValueLayout) {
    match vl {
        ValueLayout::FixedWidth { width } => {
            put_u8(out, 0);
            put_u32(out, *width as u32);
        }
        ValueLayout::Variable { value_offsets } => {
            put_u8(out, 1);
            put_u32(out, value_offsets.len() as u32);
            for &o in value_offsets {
                put_u32(out, o as u32);
            }
        }
        ValueLayout::VariablePaged {
            offsets_pk,
            n_entries,
        } => {
            put_u8(out, 2);
            put_u64(out, *offsets_pk);
            put_u32(out, *n_entries as u32);
        }
    }
}

fn dec_value_layout(inp: &mut &[u8]) -> io::Result<ValueLayout> {
    match get_u8(inp)? {
        0 => Ok(ValueLayout::FixedWidth {
            width: get_u32(inp)?,
        }),
        1 => {
            let n = get_u32(inp)? as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(get_u32(inp)?);
            }
            Ok(ValueLayout::Variable { value_offsets: v })
        }
        2 => {
            let offsets_pk = get_u64(inp)?;
            let n_entries = get_u32(inp)?;
            Ok(ValueLayout::VariablePaged {
                offsets_pk,
                n_entries,
            })
        }
        _ => Err(Error::new(ErrorKind::InvalidData, "ValueLayout tag")),
    }
}

/// Columnar registry/descriptor

fn enc_columnar_registry(out: &mut Vec<u8>, r: &ColumnarRegistry) {
    put_u32(out, r.columns.len() as u32);
    for (fid, pk) in &r.columns {
        put_u64(out, *fid as u64);
        put_u64(out, *pk);
    }
}

fn dec_columnar_registry(inp: &mut &[u8]) -> io::Result<ColumnarRegistry> {
    let n = get_u32(inp)? as usize;
    let mut cols = Vec::with_capacity(n);
    for _ in 0..n {
        let fid = get_u64(inp)?;
        let pk = get_u64(inp)?;
        cols.push((fid, pk));
    }
    Ok(ColumnarRegistry { columns: cols })
}

fn enc_columnar_descriptor(out: &mut Vec<u8>, d: &ColumnarDescriptor) {
    put_u64(out, d.field_id as u64);
    put_u32(out, d.chunks.len() as u32);
    for c in &d.chunks {
        put_u64(out, c.chunk_pk);
        match c.perm_pk {
            Some(pk) => {
                put_u8(out, 1);
                put_u64(out, pk);
            }
            None => put_u8(out, 0),
        }
        put_u64(out, c.row_count);
        put_u32(out, c.vector_len);
        put_u64(out, c.epoch);
        put_u64(out, c.min_val_u64);
        put_u64(out, c.max_val_u64);
    }
}

fn dec_columnar_descriptor(inp: &mut &[u8]) -> io::Result<ColumnarDescriptor> {
    let field_id = get_u64(inp)?;
    let n = get_u32(inp)? as usize;
    let mut chunks = Vec::with_capacity(n);
    for _ in 0..n {
        let chunk_pk = get_u64(inp)?;
        let has_perm = get_u8(inp)? != 0;
        let perm_pk = if has_perm { Some(get_u64(inp)?) } else { None };
        let row_count = get_u64(inp)?;
        let vector_len = get_u32(inp)?;
        let epoch = get_u64(inp)?;
        let min_val_u64 = get_u64(inp)?;
        let max_val_u64 = get_u64(inp)?;
        chunks.push(ColumnarChunkRef {
            chunk_pk,
            perm_pk,
            row_count,
            vector_len,
            epoch,
            min_val_u64,
            max_val_u64,
        });
    }
    Ok(ColumnarDescriptor { field_id, chunks })
}

/// Column index types

fn enc_manifest(out: &mut Vec<u8>, m: &Manifest) {
    put_u32(out, m.columns.len() as u32);
    for ColumnEntry {
        field_id,
        column_index_physical_key,
    } in &m.columns
    {
        put_u64(out, *field_id as u64);
        put_u64(out, *column_index_physical_key);
    }
}

fn dec_manifest(inp: &mut &[u8]) -> io::Result<Manifest> {
    let n = get_u32(inp)? as usize;
    let mut cols = Vec::with_capacity(n);
    for _ in 0..n {
        let fid = get_u64(inp)?;
        let pk = get_u64(inp)?;
        cols.push(ColumnEntry {
            field_id: fid,
            column_index_physical_key: pk,
        });
    }
    Ok(Manifest { columns: cols })
}

fn enc_column_index(out: &mut Vec<u8>, ci: &ColumnIndex) {
    put_u64(out, ci.field_id as u64);
    put_u32(out, ci.segments.len() as u32);
    for s in &ci.segments {
        put_u64(out, s.index_physical_key);
        put_u64(out, s.data_physical_key);
        put_bytes(out, &s.logical_key_min);
        put_bytes(out, &s.logical_key_max);
        put_u32(out, s.n_entries as u32);
    }
}

fn dec_column_index(inp: &mut &[u8]) -> io::Result<ColumnIndex> {
    let field_id = get_u64(inp)?;
    let n = get_u32(inp)? as usize;
    let mut segs = Vec::with_capacity(n);
    for _ in 0..n {
        let index_physical_key = get_u64(inp)?;
        let data_physical_key = get_u64(inp)?;
        let logical_key_min = get_bytes(inp)?;
        let logical_key_max = get_bytes(inp)?;
        let n_entries = get_u32(inp)?;
        segs.push(IndexSegmentRef {
            index_physical_key,
            data_physical_key,
            logical_key_min,
            logical_key_max,
            n_entries,
        });
    }
    Ok(ColumnIndex {
        field_id,
        segments: segs,
    })
}

fn enc_bootstrap(out: &mut Vec<u8>, b: &Bootstrap) {
    put_u64(out, b.manifest_physical_key);
}

fn dec_bootstrap(inp: &mut &[u8]) -> io::Result<Bootstrap> {
    Ok(Bootstrap {
        manifest_physical_key: get_u64(inp)?,
    })
}

fn enc_value_index(out: &mut Vec<u8>, vi: &ValueIndex) {
    put_u64(out, vi.value_order_pk);
    for x in &vi.l1_dir {
        put_u32(out, *x);
    }
    put_u32(out, vi.l2_dirs.len() as u32);
    for d in &vi.l2_dirs {
        put_u8(out, d.first_byte);
        for v in &d.dir257 {
            put_u32(out, *v);
        }
    }
}

fn dec_value_index(inp: &mut &[u8]) -> io::Result<ValueIndex> {
    let value_order_pk = get_u64(inp)?;
    let mut l1_dir = [0u32; 256];
    for i in 0..256 {
        l1_dir[i] = get_u32(inp)?;
    }
    let m = get_u32(inp)? as usize;
    let mut l2_dirs = Vec::with_capacity(m);
    for _ in 0..m {
        let first_byte = get_u8(inp)?;
        let mut dir257 = [0u32; 257];
        for i in 0..257 {
            dir257[i] = get_u32(inp)?;
        }
        l2_dirs.push(ValueDirL2 { first_byte, dir257 });
    }
    Ok(ValueIndex {
        value_order_pk,
        l1_dir,
        l2_dirs,
    })
}

fn enc_index_segment(out: &mut Vec<u8>, s: &IndexSegment) {
    put_u64(out, s.data_physical_key);
    put_u32(out, s.n_entries as u32);
    put_u64(out, s.key_bytes_pk);
    enc_key_layout(out, &s.key_layout);
    enc_value_layout(out, &s.value_layout);
    match &s.value_index {
        Some(vi) => {
            put_u8(out, 1);
            enc_value_index(out, vi);
        }
        None => put_u8(out, 0),
    }
}

fn dec_index_segment(inp: &mut &[u8]) -> io::Result<IndexSegment> {
    let data_physical_key = get_u64(inp)?;
    let n_entries = get_u32(inp)?;
    let key_bytes_pk = get_u64(inp)?;
    let key_layout = dec_key_layout(inp)?;
    let value_layout = dec_value_layout(inp)?;
    let has_vi = get_u8(inp)? != 0;
    let value_index = if has_vi {
        Some(dec_value_index(inp)?)
    } else {
        None
    };
    Ok(IndexSegment {
        data_physical_key,
        n_entries,
        key_bytes_pk,
        key_layout,
        value_layout,
        value_index,
    })
}

/// Public API: strict payloads only (no leading kind tag).

pub fn encode_typed(v: &TypedValue) -> Vec<u8> {
    let mut out = Vec::new();
    match v {
        TypedValue::Bootstrap(b) => enc_bootstrap(&mut out, b),
        TypedValue::Manifest(m) => enc_manifest(&mut out, m),
        TypedValue::ColumnIndex(ci) => enc_column_index(&mut out, ci),
        TypedValue::IndexSegment(s) => enc_index_segment(&mut out, s),
        TypedValue::ColumnarRegistry(r) => enc_columnar_registry(&mut out, r),
        TypedValue::ColumnarDescriptor(d) => enc_columnar_descriptor(&mut out, d),
    }
    out
}

pub fn decode_typed(kind: TypedKind, mut bytes: &[u8]) -> io::Result<TypedValue> {
    let v = match kind {
        TypedKind::Bootstrap => TypedValue::Bootstrap(dec_bootstrap(&mut bytes)?),
        TypedKind::Manifest => TypedValue::Manifest(dec_manifest(&mut bytes)?),
        TypedKind::ColumnIndex => TypedValue::ColumnIndex(dec_column_index(&mut bytes)?),
        TypedKind::IndexSegment => TypedValue::IndexSegment(dec_index_segment(&mut bytes)?),
        TypedKind::ColumnarRegistry => {
            TypedValue::ColumnarRegistry(dec_columnar_registry(&mut bytes)?)
        }
        TypedKind::ColumnarDescriptor => {
            TypedValue::ColumnarDescriptor(dec_columnar_descriptor(&mut bytes)?)
        }
    };
    Ok(v)
}
