use super::*;

/// Little-endian u64 codec. Payload stored in native LE; ordering handled by caller policy.
pub struct LeU64;

impl Codec for LeU64 {
    type Borrowed<'a> = &'a u64;
    type Owned = u64;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: &u64) -> Result<(), EncodeError> {
        dst.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn decode(src: &[u8]) -> Result<u64, DecodeError> {
        if src.len() < 8 {
            return Err(DecodeError::NotEnoughData);
        }
        let bytes: [u8; 8] = src[..8].try_into().unwrap();
        Ok(u64::from_le_bytes(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leu64_roundtrip() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();

        1u64.encode_into(&mut a);
        2u64.encode_into(&mut b);
        10u64.encode_into(&mut c);

        // Round-trip only (ordering is policy-driven elsewhere)
        assert_eq!(LeU64::decode(&a).unwrap(), 1);
        assert_eq!(LeU64::decode(&b).unwrap(), 2);
        assert_eq!(LeU64::decode(&c).unwrap(), 10);
    }
}

