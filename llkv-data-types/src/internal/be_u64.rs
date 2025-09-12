use super::*;

/// Big-endian u64 codec. Lexicographic order == numeric order.
pub struct BeU64;

impl Codec for BeU64 {
    type Borrowed<'a> = &'a u64;
    type Owned = u64;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: &u64) -> Result<(), EncodeError> {
        dst.extend_from_slice(&v.to_be_bytes());
        Ok(())
    }

    #[inline]
    fn decode(src: &[u8]) -> Result<u64, DecodeError> {
        if src.len() < 8 {
            return Err(DecodeError::NotEnoughData);
        }
        // This unwrap is safe due to the length check above.
        let bytes: [u8; 8] = src[..8].try_into().unwrap();
        Ok(u64::from_be_bytes(bytes))
    }
}

impl BeU64 {
    /// Decode `src` (concatenated big-endian u64) into `dst` in one pass.
    /// Length must match exactly: `src.len() == dst.len() * 8`.
    ///
    /// TODO: Experimental bulk decoder; consider moving into a generic
    /// codec trait extension and sharing impl across integer codecs.
    #[inline]
    pub fn decode_many_into(dst: &mut [u64], src: &[u8]) -> Result<(), DecodeError> {
        let n = dst.len();
        if src.len() != n.saturating_mul(8) {
            return Err(DecodeError::NotEnoughData);
        }
        let mut off = 0usize;
        for out in dst.iter_mut() {
            // Safe: we checked bounds above.
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&src[off..off + 8]);
            *out = u64::from_be_bytes(bytes);
            off += 8;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beu64_roundtrip_and_order() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();

        1u64.encode_into(&mut a);
        2u64.encode_into(&mut b);
        10u64.encode_into(&mut c);

        // Lex order == numeric
        assert!(a < b && b < c);

        // Round-trip
        assert_eq!(BeU64::decode(&a).unwrap(), 1);
        assert_eq!(BeU64::decode(&b).unwrap(), 2);
        assert_eq!(BeU64::decode(&c).unwrap(), 10);
    }
}
