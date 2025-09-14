use crate::storage::pager::{BatchPut, BatchGet, GetResult, Pager};
use crate::types::PhysicalKey;

const MAGIC: &[u8; 8] = b"LLKVCHNK";

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChunkHeader {
    magic: [u8; 8],
    pub(crate) version: u16,
    pub(crate) kind: u8,       // 3 = chunk
    pub(crate) endian: u8,     // 1 = little-endian
    pub(crate) width: u16,     // bytes per value
    pub(crate) vector_len: u32,
    pub(crate) row_count: u64,
    pub(crate) epoch: u64,
    // Optional value-range metadata for pruning value-bounded scans.
    pub(crate) min_val_u64: u64,
    pub(crate) max_val_u64: u64,
    _pad: [u8; 64 - (8 + 2 + 1 + 1 + 2 + 4 + 8 + 8 + 8 + 8)],
}

impl ChunkHeader {
    fn new_le(width: u16, vector_len: u32, row_count: u64, epoch: u64, min_val_u64: u64, max_val_u64: u64) -> Self {
        Self {
            magic: *MAGIC,
            version: 1,
            kind: 3,
            endian: 1,
            width,
            vector_len,
            row_count,
            epoch,
            min_val_u64,
            max_val_u64,
            _pad: [0u8; 64 - (8 + 2 + 1 + 1 + 2 + 4 + 8 + 8 + 8 + 8)],
        }
    }

    pub(crate) fn encode(&self, dst: &mut Vec<u8>) {
        dst.extend_from_slice(&self.magic);
        dst.extend_from_slice(&self.version.to_le_bytes());
        dst.push(self.kind);
        dst.push(self.endian);
        dst.extend_from_slice(&self.width.to_le_bytes());
        dst.extend_from_slice(&self.vector_len.to_le_bytes());
        dst.extend_from_slice(&self.row_count.to_le_bytes());
        dst.extend_from_slice(&self.epoch.to_le_bytes());
        dst.extend_from_slice(&self.min_val_u64.to_le_bytes());
        dst.extend_from_slice(&self.max_val_u64.to_le_bytes());
        dst.extend_from_slice(&self._pad);
    }

    pub(crate) fn decode(src: &[u8]) -> Option<(Self, usize)> {
        if src.len() < 64 { return None; }
        let mut off = 0;
        let mut magic = [0u8; 8]; magic.copy_from_slice(&src[off..off+8]); off += 8;
        if &magic != MAGIC { return None; }
        let mut u16le = |s: &[u8], o: &mut usize| { let mut b=[0u8;2]; b.copy_from_slice(&s[*o..*o+2]); *o+=2; u16::from_le_bytes(b) };
        let mut u32le = |s: &[u8], o: &mut usize| { let mut b=[0u8;4]; b.copy_from_slice(&s[*o..*o+4]); *o+=4; u32::from_le_bytes(b) };
        let mut u64le = |s: &[u8], o: &mut usize| { let mut b=[0u8;8]; b.copy_from_slice(&s[*o..*o+8]); *o+=8; u64::from_le_bytes(b) };
        let version = u16le(src, &mut off);
        let kind = src[off]; off+=1;
        let endian = src[off]; off+=1;
        let width = u16le(src, &mut off);
        let vector_len = u32le(src, &mut off);
        let row_count = u64le(src, &mut off);
        let epoch = u64le(src, &mut off);
        // For version 1, expect min/max fields to be present.
        let min_val_u64 = u64le(src, &mut off);
        let max_val_u64 = u64le(src, &mut off);
        let mut pad_arr = [0u8; 64 - (8 + 2 + 1 + 1 + 2 + 4 + 8 + 8 + 8 + 8)];
        let pad_len = pad_arr.len();
        pad_arr.copy_from_slice(&src[off..off+pad_len]);
        off += pad_len;
        Some((Self { magic, version, kind, endian, width, vector_len, row_count, epoch, min_val_u64, max_val_u64, _pad: pad_arr }, off))
    }
}

/// Write a single u64 chunk (little-endian stripe) at `key`.
pub fn write_u64_chunk<P: Pager>(pager: &P, key: PhysicalKey, values: &[u64], vector_len: u32, epoch: u64) -> Result<(), String> {
    let mut buf = Vec::with_capacity(64 + values.len() * 8);
    // Compute min/max for pruning.
    let (min_val_u64, max_val_u64) = if values.is_empty() {
        (u64::MAX, u64::MIN)
    } else {
        let mut minv = u64::MAX;
        let mut maxv = u64::MIN;
        for &v in values { if v < minv { minv = v; } if v > maxv { maxv = v; } }
        (minv, maxv)
    };
    let hdr = ChunkHeader::new_le(8, vector_len, values.len() as u64, epoch, min_val_u64, max_val_u64);
    hdr.encode(&mut buf);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    pager.batch_put(&[BatchPut::Raw { key, bytes: buf }]).map_err(|e| format!("pager put: {:?}", e))
}

/// Get raw bytes for a chunk at `key`.
pub fn get_chunk_blob<P: Pager>(pager: &P, key: PhysicalKey) -> Option<P::Blob> {
    match pager.batch_get(&[BatchGet::Raw { key }]).ok()?.pop()? {
        GetResult::Raw { bytes, .. } => Some(bytes),
        _ => None,
    }
}

pub struct U64ChunkView<'a> {
    pub values_le: &'a [u8], // little-endian u64 stripe
    pub row_count: usize,
    pub vector_len: usize,
    pub min_val_u64: u64,
    pub max_val_u64: u64,
}

impl<'a> U64ChunkView<'a> {
    pub fn from_blob_le(blob: &'a [u8]) -> Option<Self> {
        let (hdr, off) = ChunkHeader::decode(blob)?;
        if hdr.kind != 3 || hdr.width != 8 || hdr.endian != 1 { return None; }
        let stripe = &blob[off..];
        let need = hdr.row_count as usize * 8;
        if stripe.len() < need { return None; }
        Some(Self { values_le: &stripe[..need], row_count: hdr.row_count as usize, vector_len: hdr.vector_len as usize, min_val_u64: hdr.min_val_u64, max_val_u64: hdr.max_val_u64 })
    }
}

/// Sum a little-endian u64 stripe using unaligned loads (fast path on LE hosts).
#[inline(always)]
pub fn sum_u64_le_unaligned(stripe: &[u8]) -> u128 {
    let n = stripe.len() / 8;
    let mut acc0: u128 = 0;
    let mut acc1: u128 = 0;
    let mut acc2: u128 = 0;
    let mut acc3: u128 = 0;
    let mut i = 0;
    let un = 4;
    let bound = n - (n % un);
    let mut p = stripe.as_ptr();
    unsafe {
        while i < bound {
            let a = (p as *const u64).read_unaligned();
            let b = (p.add(8) as *const u64).read_unaligned();
            let c = (p.add(16) as *const u64).read_unaligned();
            let d = (p.add(24) as *const u64).read_unaligned();
            acc0 += u64::from_le(a) as u128;
            acc1 += u64::from_le(b) as u128;
            acc2 += u64::from_le(c) as u128;
            acc3 += u64::from_le(d) as u128;
            p = p.add(32);
            i += un;
        }
        while i < n {
            let w = (p as *const u64).read_unaligned();
            acc0 += u64::from_le(w) as u128;
            p = p.add(8);
            i += 1;
        }
    }
    acc0 + acc1 + acc2 + acc3
}

#[inline(always)]
pub fn sum_u64_le_simd(stripe: &[u8]) -> u128 {
    let n = stripe.len() / 8;
    let mut acc: u128 = 0;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0usize;
        let mut v0 = _mm256_setzero_si256();
        let mut v1 = _mm256_setzero_si256();
        let mut v2 = _mm256_setzero_si256();
        let mut v3 = _mm256_setzero_si256();
        while i + 16 <= n {
            let p = stripe.as_ptr().add(i * 8) as *const __m256i;
            let a = _mm256_loadu_si256(p);
            let b = _mm256_loadu_si256(p.add(1));
            let c = _mm256_loadu_si256(p.add(2));
            let d = _mm256_loadu_si256(p.add(3));
            v0 = _mm256_add_epi64(v0, a);
            v1 = _mm256_add_epi64(v1, b);
            v2 = _mm256_add_epi64(v2, c);
            v3 = _mm256_add_epi64(v3, d);
            i += 16;
        }
        let s01 = _mm256_add_epi64(v0, v1);
        let s23 = _mm256_add_epi64(v2, v3);
        let s = _mm256_add_epi64(s01, s23);
        let hi = _mm256_extracti128_si256(s, 1);
        let lo = _mm256_castsi256_si128(s);
        let s128 = _mm_add_epi64(lo, hi);
        let mut tmp = [0u64; 2];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, s128);
        acc = tmp[0] as u128 + tmp[1] as u128;
        while i < n {
            let w = (stripe.as_ptr().add(i * 8) as *const u64).read_unaligned();
            acc += u64::from_le(w) as u128;
            i += 1;
        }
        return acc;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        use core::arch::aarch64::*;
        let mut i = 0usize;
        let mut v0 = vdupq_n_u64(0);
        let mut v1 = vdupq_n_u64(0);
        let mut v2 = vdupq_n_u64(0);
        let mut v3 = vdupq_n_u64(0);
        while i + 8 <= n {
            let p = stripe.as_ptr().add(i * 8) as *const u64;
            let a = vld1q_u64(p);
            let b = vld1q_u64(p.add(2));
            let c = vld1q_u64(p.add(4));
            let d = vld1q_u64(p.add(6));
            v0 = vaddq_u64(v0, a);
            v1 = vaddq_u64(v1, b);
            v2 = vaddq_u64(v2, c);
            v3 = vaddq_u64(v3, d);
            i += 8;
        }
        let s01 = vaddq_u64(v0, v1);
        let s23 = vaddq_u64(v2, v3);
        let s = vaddq_u64(s01, s23);
        let lanes: [u64; 2] = core::mem::transmute(s);
        acc = lanes[0] as u128 + lanes[1] as u128;
        while i < n {
            let w = (stripe.as_ptr().add(i * 8) as *const u64).read_unaligned();
            acc += u64::from_le(w) as u128;
            i += 1;
        }
        return acc;
    }
    // Fallback
    sum_u64_le_unaligned(stripe)
}
