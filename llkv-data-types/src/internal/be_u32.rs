use super::*;

/// Big-endian u32 codec. Lexicographic order == numeric order.
pub struct BeU32;

impl Codec for BeU32 {
    const WIDTH: usize = 4;
    type Borrowed<'a> = &'a u32;
    type Owned = u32;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: &u32) -> Result<(), EncodeError> {
        dst.extend_from_slice(&v.to_be_bytes());
        Ok(())
    }

    #[inline]
    fn decode(src: &[u8]) -> Result<u32, DecodeError> {
        if src.len() < 4 {
            return Err(DecodeError::NotEnoughData);
        }
        // This unwrap is safe due to the length check above.
        let bytes: [u8; 4] = src[..4].try_into().unwrap();
        Ok(u32::from_be_bytes(bytes))
    }

    // Specialized fast path to avoid per-item length checks in the hot loop.
    #[inline]
    fn decode_many_into(dst: &mut [u32], src: &[u8]) -> Result<(), DecodeError> {
        let n = dst.len();
        if src.len() != n.saturating_mul(4) {
            return Err(DecodeError::NotEnoughData);
        }
        let mut off = 0usize;
        for out in dst.iter_mut() {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&src[off..off + 4]);
            *out = u32::from_be_bytes(bytes);
            off += 4;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beu32_roundtrip_and_order() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();

        1u32.encode_into(&mut a);
        2u32.encode_into(&mut b);
        10u32.encode_into(&mut c);

        // Lex order == numeric
        assert!(a < b && b < c);

        // Round-trip
        assert_eq!(BeU32::decode(&a).unwrap(), 1);
        assert_eq!(BeU32::decode(&b).unwrap(), 2);
        assert_eq!(BeU32::decode(&c).unwrap(), 10);
    }
}
