//! Descriptor pages for columnar metadata (paged, streamable).
//!
//! Raw little-endian format, page-chained. No legacy/typed fallbacks.
//! Appends touch only the tail page (or link a fresh page when full).
//!
//! Header (64 bytes):
//!   magic[8] = "LLKVCDPG"
//!   version: u16 = 1
//!   kind: u8 = 1
//!   endian: u8 = 1
//!   field_id: u64
//!   next_page_pk: u64  // 0 if tail
//!   entry_count: u32
//!   reserved[32] = 0
//!
//! Entry (52 bytes) repeated entry_count times:
//!   chunk_pk: u64
//!   perm_pk_or_0: u64
//!   row_count: u64
//!   vector_len: u32
//!   epoch: u64
//!   min_val_u64: u64
//!   max_val_u64: u64

use crate::storage::{BatchGet, BatchPut, GetResult, Pager};
use crate::types::{LogicalFieldId, PhysicalKey};
use std::io::{self, Error, ErrorKind};

const MAGIC: &[u8; 8] = b"LLKVCDPG";
const VERSION: u16 = 1;
const KIND: u8 = 1;
const ENDIAN_LE: u8 = 1;

const HEADER_SIZE: usize = 64;
const ENTRY_SIZE: usize = 52;

#[derive(Debug, Clone, Copy)]
pub struct DescPageHeader {
    pub field_id: LogicalFieldId,
    pub next_page_pk: PhysicalKey,
    pub entry_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescPageEntry {
    pub chunk_pk: PhysicalKey,
    pub perm_pk: Option<PhysicalKey>,
    pub row_count: u64,
    pub vector_len: u32,
    pub epoch: u64,
    pub min_val_u64: u64,
    pub max_val_u64: u64,
}

#[inline]
fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn get_u16(inp: &mut &[u8]) -> io::Result<u16> {
    if inp.len() < 2 {
        return Err(Error::new(ErrorKind::UnexpectedEof, "u16"));
    }
    let mut b = [0u8; 2];
    b.copy_from_slice(&inp[..2]);
    *inp = &inp[2..];
    Ok(u16::from_le_bytes(b))
}

#[inline]
fn get_u32(inp: &mut &[u8]) -> io::Result<u32> {
    if inp.len() < 4 {
        return Err(Error::new(ErrorKind::UnexpectedEof, "u32"));
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&inp[..4]);
    *inp = &inp[4..];
    Ok(u32::from_le_bytes(b))
}

#[inline]
fn get_u64(inp: &mut &[u8]) -> io::Result<u64> {
    if inp.len() < 8 {
        return Err(Error::new(ErrorKind::UnexpectedEof, "u64"));
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&inp[..8]);
    *inp = &inp[8..];
    Ok(u64::from_le_bytes(b))
}

#[inline]
fn encode_header(h: &DescPageHeader) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_SIZE);
    out.extend_from_slice(MAGIC);
    put_u16(&mut out, VERSION);
    out.push(KIND);
    out.push(ENDIAN_LE);
    put_u64(&mut out, h.field_id as u64);
    put_u64(&mut out, h.next_page_pk);
    put_u32(&mut out, h.entry_count);
    out.resize(HEADER_SIZE, 0);
    out
}

#[inline]
fn decode_header(mut s: &[u8]) -> io::Result<DescPageHeader> {
    if s.len() < HEADER_SIZE {
        return Err(Error::new(ErrorKind::UnexpectedEof, "header"));
    }
    let magic = &s[..8];
    if magic != MAGIC {
        return Err(Error::new(ErrorKind::InvalidData, "bad magic"));
    }
    s = &s[8..];
    let ver = get_u16(&mut s)?;
    if ver != VERSION {
        return Err(Error::new(ErrorKind::InvalidData, "bad version"));
    }
    let kind = get_u8(&mut s)?;
    if kind != KIND {
        return Err(Error::new(ErrorKind::InvalidData, "bad kind"));
    }
    let endian = get_u8(&mut s)?;
    if endian != ENDIAN_LE {
        return Err(Error::new(ErrorKind::InvalidData, "bad endian"));
    }
    let field_id = get_u64(&mut s)?;
    let next_page_pk = get_u64(&mut s)?;
    let entry_count = get_u32(&mut s)?;
    Ok(DescPageHeader {
        field_id,
        next_page_pk,
        entry_count,
    })
}

#[inline]
fn get_u8(inp: &mut &[u8]) -> io::Result<u8> {
    if inp.is_empty() {
        return Err(Error::new(ErrorKind::UnexpectedEof, "u8"));
    }
    let v = inp[0];
    *inp = &inp[1..];
    Ok(v)
}

#[inline]
fn encode_entries(es: &[DescPageEntry]) -> Vec<u8> {
    let mut out = Vec::with_capacity(es.len() * ENTRY_SIZE);
    for e in es {
        put_u64(&mut out, e.chunk_pk);
        match e.perm_pk {
            Some(pk) => {
                put_u64(&mut out, pk);
            }
            None => put_u64(&mut out, 0),
        }
        put_u64(&mut out, e.row_count);
        put_u32(&mut out, e.vector_len);
        put_u64(&mut out, e.epoch);
        put_u64(&mut out, e.min_val_u64);
        put_u64(&mut out, e.max_val_u64);
    }
    out
}

#[inline]
fn decode_entries(mut s: &[u8], n: usize) -> io::Result<Vec<DescPageEntry>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let chunk_pk = get_u64(&mut s)?;
        let p = get_u64(&mut s)?;
        let perm_pk = if p == 0 { None } else { Some(p) };
        let row_count = get_u64(&mut s)?;
        let vector_len = get_u32(&mut s)?;
        let epoch = get_u64(&mut s)?;
        let min_val_u64 = get_u64(&mut s)?;
        let max_val_u64 = get_u64(&mut s)?;
        out.push(DescPageEntry {
            chunk_pk,
            perm_pk,
            row_count,
            vector_len,
            epoch,
            min_val_u64,
            max_val_u64,
        });
    }
    Ok(out)
}

#[inline]
pub fn decode_page(bytes: &[u8]) -> io::Result<(DescPageHeader, Vec<DescPageEntry>)> {
    if bytes.len() < HEADER_SIZE {
        return Err(Error::new(ErrorKind::UnexpectedEof, "page too small"));
    }
    let hdr = decode_header(&bytes[..HEADER_SIZE])?;
    let need = HEADER_SIZE + (hdr.entry_count as usize) * ENTRY_SIZE;
    if bytes.len() < need {
        return Err(Error::new(ErrorKind::UnexpectedEof, "page body"));
    }
    let entries = decode_entries(&bytes[HEADER_SIZE..need], hdr.entry_count as usize)?;
    Ok((hdr, entries))
}

#[inline]
pub fn encode_page(header: &DescPageHeader, entries: &[DescPageEntry]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_SIZE + entries.len() * ENTRY_SIZE);
    out.extend_from_slice(&encode_header(header));
    out.extend_from_slice(&encode_entries(entries));
    out
}

#[inline]
fn load_page_raw<P: Pager>(pager: &P, pk: PhysicalKey) -> io::Result<Vec<u8>> {
    match pager
        .batch_get(&[BatchGet::Raw { key: pk }])?
        .pop()
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "missing get result"))?
    {
        GetResult::Raw { bytes, .. } => Ok(bytes.as_ref().to_vec()),
        GetResult::Missing { .. } => Err(Error::new(ErrorKind::NotFound, "page not found")),
        GetResult::Typed { .. } => Err(Error::new(
            ErrorKind::InvalidData,
            "typed where raw expected",
        )),
    }
}

#[inline]
fn init_empty_page<P: Pager>(
    pager: &P,
    pk: PhysicalKey,
    field_id: LogicalFieldId,
) -> io::Result<()> {
    let hdr = DescPageHeader {
        field_id,
        next_page_pk: 0,
        entry_count: 0,
    };
    let bytes = encode_page(&hdr, &[]);
    pager.batch_put(&[BatchPut::Raw { key: pk, bytes }])?;
    Ok(())
}

#[inline]
fn ensure_head_exists<P: Pager>(
    pager: &P,
    dkey: PhysicalKey,
    field_id: LogicalFieldId,
) -> io::Result<()> {
    match load_page_raw(pager, dkey) {
        Ok(bytes) => {
            decode_page(&bytes)?;
            Ok(())
        }
        Err(e) if e.kind() == ErrorKind::NotFound => init_empty_page(pager, dkey, field_id),
        Err(e) => Err(e),
    }
}

/// Append one entry. Creates the head page if missing. Links a new page
/// when the tail is full. Returns the page key that holds the appended
/// entry.
pub fn append_entry<P: Pager>(
    pager: &P,
    dkey: PhysicalKey,
    field_id: LogicalFieldId,
    entry: DescPageEntry,
    entries_per_page: usize,
) -> io::Result<PhysicalKey> {
    assert!(entries_per_page > 0, "entries_per_page must be > 0");
    ensure_head_exists(pager, dkey, field_id)?;

    // Find tail page.
    let mut page_pk = dkey;
    loop {
        let bytes = load_page_raw(pager, page_pk)?;
        let (hdr, _) = decode_page(&bytes)?;
        if hdr.next_page_pk == 0 {
            break;
        }
        page_pk = hdr.next_page_pk;
    }

    // Load tail entries to check capacity and rewrite.
    let bytes = load_page_raw(pager, page_pk)?;
    let (mut hdr, mut es) = decode_page(&bytes)?;
    if es.len() >= entries_per_page {
        // Allocate a new tail, link, and switch current tail pointer.
        let npk = pager.alloc_many(1)?.get(0).copied().unwrap();
        hdr.next_page_pk = npk;
        let bytes = encode_page(&hdr, &es);
        pager.batch_put(&[BatchPut::Raw {
            key: page_pk,
            bytes,
        }])?;
        page_pk = npk;
        hdr = DescPageHeader {
            field_id,
            next_page_pk: 0,
            entry_count: 0,
        };
        es.clear();
    }

    es.push(entry);
    hdr.entry_count = es.len() as u32;
    let bytes = encode_page(&hdr, &es);
    pager.batch_put(&[BatchPut::Raw {
        key: page_pk,
        bytes,
    }])?;
    Ok(page_pk)
}

/// Iterate entries in order.
pub fn for_each_entry<P: Pager, F>(pager: &P, head: PhysicalKey, mut f: F) -> io::Result<()>
where
    F: FnMut(DescPageEntry),
{
    let mut page_pk = head;
    loop {
        let bytes = load_page_raw(pager, page_pk)?;
        let (hdr, es) = decode_page(&bytes)?;
        for e in es {
            f(e);
        }
        if hdr.next_page_pk == 0 {
            break;
        }
        page_pk = hdr.next_page_pk;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::pager::MemPager;

    #[test]
    fn roundtrip_page_encode_decode() {
        let h = DescPageHeader {
            field_id: 11,
            next_page_pk: 22,
            entry_count: 2,
        };
        let e = vec![
            DescPageEntry {
                chunk_pk: 100,
                perm_pk: None,
                row_count: 1,
                vector_len: 2,
                epoch: 3,
                min_val_u64: 0,
                max_val_u64: 0,
            },
            DescPageEntry {
                chunk_pk: 200,
                perm_pk: Some(300),
                row_count: 4,
                vector_len: 5,
                epoch: 6,
                min_val_u64: 1,
                max_val_u64: 9,
            },
        ];
        let bytes = encode_page(&h, &e);
        let (h2, e2) = decode_page(&bytes).unwrap();
        assert_eq!(h.field_id, h2.field_id);
        assert_eq!(h.next_page_pk, h2.next_page_pk);
        assert_eq!(h.entry_count, h2.entry_count);
        assert_eq!(e.len(), e2.len());
        assert_eq!(e[0].chunk_pk, e2[0].chunk_pk);
        assert_eq!(e[1].perm_pk, e2[1].perm_pk);
    }

    #[test]
    fn append_and_iterate() {
        let pager = MemPager::default();
        let head = pager.alloc_many(1).unwrap()[0];
        let e1 = DescPageEntry {
            chunk_pk: 100,
            perm_pk: None,
            row_count: 1,
            vector_len: 2,
            epoch: 3,
            min_val_u64: 0,
            max_val_u64: 0,
        };
        let e2 = DescPageEntry {
            chunk_pk: 200,
            perm_pk: Some(300),
            row_count: 4,
            vector_len: 5,
            epoch: 6,
            min_val_u64: 1,
            max_val_u64: 9,
        };
        let p1 = append_entry(&pager, head, 11, e1, 1).unwrap();
        let p2 = append_entry(&pager, head, 11, e2, 1).unwrap();
        assert_ne!(p1, p2);

        let mut got = Vec::new();
        for_each_entry(&pager, head, |e| got.push(e.chunk_pk)).unwrap();
        assert_eq!(got, vec![100, 200]);
    }
}
