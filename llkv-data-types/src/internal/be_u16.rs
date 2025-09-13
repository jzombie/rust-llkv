use super::*;

/// Big-endian u16 codec. Lexicographic order == numeric order.
pub struct BeU16;

impl Codec for BeU16 {
    const WIDTH: usize = 2;
    type Borrowed<'a> = &'a u16;
    type Owned = u16;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: &u16) -> Result<(), EncodeError> {
        dst.extend_from_slice(&v.to_be_bytes());
        Ok(())
    }

    #[inline]
    fn decode(src: &[u8]) -> Result<u16, DecodeError> {
        if src.len() < 2 {
            return Err(DecodeError::NotEnoughData);
        }
        // Safe due to the length check above.
        let bytes: [u8; 2] = src[..2].try_into().unwrap();
        Ok(u16::from_be_bytes(bytes))
    }

    // Specialized fast path to avoid per-item length checks in the hot loop.
    #[inline]
    fn decode_many_into(dst: &mut [u16], src: &[u8]) -> Result<(), DecodeError> {
        let n = dst.len();
        if src.len() != n.saturating_mul(2) {
            return Err(DecodeError::NotEnoughData);
        }
        let mut off = 0usize;
        for out in dst.iter_mut() {
            let mut bytes = [0u8; 2];
            bytes.copy_from_slice(&src[off..off + 2]);
            *out = u16::from_be_bytes(bytes);
            off += 2;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beu16_roundtrip_and_order() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();

        1u16.encode_into(&mut a);
        2u16.encode_into(&mut b);
        10u16.encode_into(&mut c);

        // Lex order == numeric order
        assert!(a < b && b < c);

        // Round-trip
        assert_eq!(BeU16::decode(&a).unwrap(), 1);
        assert_eq!(BeU16::decode(&b).unwrap(), 2);
        assert_eq!(BeU16::decode(&c).unwrap(), 10);
    }
}
