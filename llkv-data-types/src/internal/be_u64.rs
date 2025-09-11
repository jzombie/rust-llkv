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

    /// This is more robust and can be faster than a separate copy.
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
