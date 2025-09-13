use super::*;

/// Opaque bytes codec with trailer length:
///   [bytes][be32 len_bytes]
///
/// Notes:
/// - Decoding is zero-copy via `decode_borrowed`.
/// - Trailer len allows parsing without a sentinel.
/// - If you require memcomparable ordering for raw bytes in composite keys,
///   add a dedicated "memcomparable bytes" codec (escape scheme) instead.
pub struct Bytes;

impl Bytes {
    #[inline]
    fn encode_impl(dst: &mut Vec<u8>, v: &[u8]) {
        dst.reserve_exact(v.len() + 4);
        dst.extend_from_slice(v);
        dst.extend_from_slice(&(v.len() as u32).to_be_bytes());
    }

    /// Borrow the original bytes without allocation.
    #[inline]
    pub fn decode_borrowed(src: &[u8]) -> Option<&[u8]> {
        if src.len() < 4 {
            return None;
        }
        let len_bytes: [u8; 4] = src[src.len() - 4..].try_into().ok()?;
        let n = u32::from_be_bytes(len_bytes) as usize;
        if n + 4 > src.len() {
            return None;
        }
        src.get(..n)
    }
}

impl Codec for Bytes {
    const WIDTH: usize = 0; // variable-width
    type Borrowed<'a> = &'a [u8];
    type Owned = Vec<u8>;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: &[u8]) -> Result<(), EncodeError> {
        Self::encode_impl(dst, v);
        Ok(())
    }

    #[inline]
    fn decode(src: &[u8]) -> Result<Vec<u8>, DecodeError> {
        Self::decode_borrowed(src)
            .map(|b| b.to_vec())
            .ok_or(DecodeError::InvalidFormat)
    }

    #[inline]
    fn decode_many_into(_dst: &mut [Vec<u8>], _src: &[u8]) -> Result<(), DecodeError> {
        panic!("Bytes::decode_many_into is not supported (variable-width codec)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_roundtrip() {
        let data = [0x00u8, 0xFF, 0x10, 0x00, 0x7A];
        let mut buf = Vec::new();
        Bytes::encode_into(&mut buf, &data[..]).unwrap();

        let dec = Bytes::decode(&buf).unwrap();
        assert_eq!(dec, data);

        let dec_b = Bytes::decode_borrowed(&buf).unwrap();
        assert_eq!(dec_b, &data[..]);
    }

    #[test]
    fn bytes_decode_borrowed_bounds() {
        let data = b"abc";
        let mut buf = Vec::new();
        Bytes::encode_into(&mut buf, &data[..]).unwrap();

        let dec_b = Bytes::decode_borrowed(&buf).unwrap();
        let range = buf.as_ptr_range();
        let start = dec_b.as_ptr() as usize;
        let end = start + dec_b.len();
        assert!(start >= range.start as usize);
        assert!(end <= range.end as usize);
    }

    #[test]
    fn bytes_invalid_input_rejected() {
        // Too short
        assert!(Bytes::decode_borrowed(&[]).is_none());

        // Bad length (claims 10 bytes, but only 5 + 4 trailer exist)
        let mut bad = b"12345".to_vec();
        bad.extend_from_slice(&(10u32.to_be_bytes()));
        assert!(Bytes::decode_borrowed(&bad).is_none());
        assert!(Bytes::decode(&bad).is_err());
    }
}
