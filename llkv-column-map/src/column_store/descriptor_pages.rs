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

#![forbid(unsafe_code)]

use std::io::{self, Error, ErrorKind};

use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::types::{LogicalFieldId, PhysicalKey};

pub const MAGIC: &[u8; 8] = b"LLKVCDPG";
pub const VERSION: u16 = 1;
pub const KIND_DESC_PAGE: u8 = 1;
pub const ENDIAN_LE: u8 = 1;

pub const HEADER_SIZE: usize = 64;
pub const ENTRY_SIZE: usize = 52;

#[derive(Debug, Clone, Copy)]
pub struct DescPageHeader {
    pub field_id: LogicalFieldId,
    pub next_page_pk: PhysicalKey,
    pub entry_count: u32,
}

#[derive(Debug, Clone, Copy)]
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
fn encode_header(h: &DescPageHeader) -> [u8; HEADER_SIZE] {
    let mut b = [0u8; HEADER_SIZE];
    b[0..8].copy_from_slice(MAGIC);
    b[8..10].copy_from_slice(&VERSION.to_le_bytes());
    b[10] = KIND_DESC_PAGE;
    b[11] = ENDIAN_LE;
    b[12..20].copy_from_slice(&h.field_id.to_le_bytes());
    b[20..28].copy_from_slice(&h.next_page_pk.to_le_bytes());
    b[28..32].copy_from_slice(&h.entry_count.to_le_bytes());
    b
}

#[inline]
fn decode_header(bytes: &[u8]) -> io::Result<DescPageHeader> {
    if bytes.len() < HEADER_SIZE {
        return Err(Error::new(ErrorKind::InvalidData, "truncated header"));
    }
    if &bytes[0..8] != MAGIC {
        return Err(Error::new(ErrorKind::InvalidData, "bad magic"));
    }
    let ver = u16::from_le_bytes([bytes[8], bytes[9]]);
    if ver != VERSION {
        return Err(Error::new(ErrorKind::InvalidData, "bad version"));
    }
    if bytes[10] != KIND_DESC_PAGE {
        return Err(Error::new(ErrorKind::InvalidData, "bad kind"));
    }
    if bytes[11] != ENDIAN_LE {
        return Err(Error::new(ErrorKind::InvalidData, "bad endian"));
    }
    let mut u8_8 = [0u8; 8];
    u8_8.copy_from_slice(&bytes[12..20]);
    let field_id = u64::from_le_bytes(u8_8);
    u8_8.copy_from_slice(&bytes[20..28]);
    let next_page_pk = u64::from_le_bytes(u8_8);
    let mut u8_4 = [0u8; 4];
    u8_4.copy_from_slice(&bytes[28..32]);
    let entry_count = u32::from_le_bytes(u8_4);
    Ok(DescPageHeader {
        field_id,
        next_page_pk,
        entry_count,
    })
}

#[inline]
fn encode_entries(entries: &[DescPageEntry]) -> Vec<u8> {
    let mut out = Vec::with_capacity(entries.len() * ENTRY_SIZE);
    for e in entries {
        out.extend_from_slice(&e.chunk_pk.to_le_bytes());
        let perm = e.perm_pk.unwrap_or(0);
        out.extend_from_slice(&perm.to_le_bytes());
        out.extend_from_slice(&e.row_count.to_le_bytes());
        out.extend_from_slice(&e.vector_len.to_le_bytes());
        out.extend_from_slice(&e.epoch.to_le_bytes());
        out.extend_from_slice(&e.min_val_u64.to_le_bytes());
        out.extend_from_slice(&e.max_val_u64.to_le_bytes());
    }
    out
}

#[inline]
fn decode_entries(bytes: &[u8], n: usize) -> io::Result<Vec<DescPageEntry>> {
    let need = n
        .checked_mul(ENTRY_SIZE)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "overflow"))?;
    if bytes.len() < need {
        return Err(Error::new(ErrorKind::InvalidData, "truncated entries"));
    }
    let mut v = Vec::with_capacity(n);
    let mut off = 0usize;
    for _ in 0..n {
        let mut u8_8 = [0u8; 8];

        u8_8.copy_from_slice(&bytes[off..off + 8]);
        let chunk_pk = u64::from_le_bytes(u8_8);
        off += 8;

        u8_8.copy_from_slice(&bytes[off..off + 8]);
        let perm_raw = u64::from_le_bytes(u8_8);
        let perm_pk = if perm_raw == 0 { None } else { Some(perm_raw) };
        off += 8;

        u8_8.copy_from_slice(&bytes[off..off + 8]);
        let row_count = u64::from_le_bytes(u8_8);
        off += 8;

        let mut u8_4 = [0u8; 4];
        u8_4.copy_from_slice(&bytes[off..off + 4]);
        let vector_len = u32::from_le_bytes(u8_4);
        off += 4;

        u8_8.copy_from_slice(&bytes[off..off + 8]);
        let epoch = u64::from_le_bytes(u8_8);
        off += 8;

        u8_8.copy_from_slice(&bytes[off..off + 8]);
        let min_val_u64 = u64::from_le_bytes(u8_8);
        off += 8;

        u8_8.copy_from_slice(&bytes[off..off + 8]);
        let max_val_u64 = u64::from_le_bytes(u8_8);
        off += 8;

        v.push(DescPageEntry {
            chunk_pk,
            perm_pk,
            row_count,
            vector_len,
            epoch,
            min_val_u64,
            max_val_u64,
        });
    }
    Ok(v)
}

#[inline]
pub fn decode_page(bytes: &[u8]) -> io::Result<(DescPageHeader, Vec<DescPageEntry>)> {
    let hdr = decode_header(bytes)?;
    let start = HEADER_SIZE;
    let need = start + (hdr.entry_count as usize) * ENTRY_SIZE;
    if bytes.len() < need {
        return Err(Error::new(ErrorKind::InvalidData, "truncated page"));
    }
    let entries = decode_entries(&bytes[start..need], hdr.entry_count as usize)?;
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
        _ => Err(Error::new(ErrorKind::InvalidData, "unexpected typed")),
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
    let (mut hdr, mut entries) = decode_page(&bytes)?;
    if (entries.len() as usize) < entries_per_page {
        entries.push(entry);
        hdr.entry_count += 1;
        let bytes = encode_page(&hdr, &entries);
        pager.batch_put(&[BatchPut::Raw {
            key: page_pk,
            bytes,
        }])?;
        Ok(page_pk)
    } else {
        let new_pk = pager.alloc_many(1)?[0];
        // Link old tail to new page.
        hdr.next_page_pk = new_pk;
        let old_bytes = encode_page(&hdr, &entries);
        // New page with the single entry.
        let new_hdr = DescPageHeader {
            field_id,
            next_page_pk: 0,
            entry_count: 1,
        };
        let new_bytes = encode_page(&new_hdr, std::slice::from_ref(&entry));
        pager.batch_put(&[
            BatchPut::Raw {
                key: page_pk,
                bytes: old_bytes,
            },
            BatchPut::Raw {
                key: new_pk,
                bytes: new_bytes,
            },
        ])?;
        Ok(new_pk)
    }
}

/// Iterate all entries by following next_page_pk starting at `dkey`.
/// Calls `f` once per entry and returns the total count.
pub fn for_each_entry<P, F>(pager: &P, dkey: PhysicalKey, mut f: F) -> io::Result<usize>
where
    P: Pager,
    F: FnMut(&DescPageEntry),
{
    let mut count = 0usize;
    let mut pk = dkey;
    loop {
        let bytes = load_page_raw(pager, pk)?;
        let (hdr, entries) = decode_page(&bytes)?;
        for e in &entries {
            f(e);
            count += 1;
        }
        if hdr.next_page_pk == 0 {
            break;
        }
        pk = hdr.next_page_pk;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::pager::MemPager;

    #[test]
    fn roundtrip_page_encode_decode() {
        let h = DescPageHeader {
            field_id: 42,
            next_page_pk: 0,
            entry_count: 2,
        };
        let e = vec![
            DescPageEntry {
                chunk_pk: 1,
                perm_pk: None,
                row_count: 3,
                vector_len: 4,
                epoch: 5,
                min_val_u64: 6,
                max_val_u64: 7,
            },
            DescPageEntry {
                chunk_pk: 8,
                perm_pk: Some(9),
                row_count: 10,
                vector_len: 11,
                epoch: 12,
                min_val_u64: 13,
                max_val_u64: 14,
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
            max_val_u64: 2,
        };
        append_entry(&pager, head, 7, e1, 1).unwrap();
        append_entry(&pager, head, 7, e2, 1).unwrap();
        let mut got = Vec::new();
        for_each_entry(&pager, head, |e| got.push(e.chunk_pk)).unwrap();
        assert_eq!(got, vec![100, 200]);
    }
}
