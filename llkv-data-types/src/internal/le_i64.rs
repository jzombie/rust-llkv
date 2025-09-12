use super::*;

/// Little-endian i64 codec. Payload stored in LE; ordering handled by caller policy.
pub struct LeI64;

impl Codec for LeI64 {
    type Borrowed<'a> = &'a i64;
    type Owned = i64;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: &i64) -> Result<(), EncodeError> {
        dst.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn decode(src: &[u8]) -> Result<i64, DecodeError> {
        if src.len() < 8 {
            return Err(DecodeError::NotEnoughData);
        }
        let bytes: [u8; 8] = src[..8].try_into().unwrap();
        Ok(i64::from_le_bytes(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lei64_roundtrip() {
        let vals = [i64::MIN + 1, -10, -1, 0, 1, 10, i64::MAX];
        for v in vals {
            let mut b = Vec::new();
            v.encode_into(&mut b);
            assert_eq!(LeI64::decode(&b).unwrap(), v);
        }
    }
}

