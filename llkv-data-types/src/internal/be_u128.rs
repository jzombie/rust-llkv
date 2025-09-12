use super::*;

/// Big-endian u128 codec. Lexicographic order == numeric order.
pub struct BeU128;

impl Codec for BeU128 {
    type Borrowed<'a> = &'a u128;
    type Owned = u128;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: &u128) -> Result<(), EncodeError> {
        dst.extend_from_slice(&v.to_be_bytes());
        Ok(())
    }

    #[inline]
    fn decode(src: &[u8]) -> Result<u128, DecodeError> {
        if src.len() < 16 {
            return Err(DecodeError::NotEnoughData);
        }
        // Safe due to the length check above.
        let bytes: [u8; 16] = src[..16].try_into().unwrap();
        Ok(u128::from_be_bytes(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beu128_roundtrip_and_order() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();

        1u128.encode_into(&mut a);
        2u128.encode_into(&mut b);
        10u128.encode_into(&mut c);

        // Lex order == numeric order
        assert!(a < b && b < c);

        // Round-trip
        assert_eq!(BeU128::decode(&a).unwrap(), 1);
        assert_eq!(BeU128::decode(&b).unwrap(), 2);
        assert_eq!(BeU128::decode(&c).unwrap(), 10);
    }
}
