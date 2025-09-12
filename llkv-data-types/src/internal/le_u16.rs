use super::*;

/// Little-endian u16 codec. Payload stored in native LE; ordering handled by caller policy.
pub struct LeU16;

impl Codec for LeU16 {
    type Borrowed<'a> = &'a u16;
    type Owned = u16;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: &u16) -> Result<(), EncodeError> {
        dst.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn decode(src: &[u8]) -> Result<u16, DecodeError> {
        if src.len() < 2 {
            return Err(DecodeError::NotEnoughData);
        }
        let bytes: [u8; 2] = src[..2].try_into().unwrap();
        Ok(u16::from_le_bytes(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leu16_roundtrip() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();

        1u16.encode_into(&mut a);
        2u16.encode_into(&mut b);
        10u16.encode_into(&mut c);

        assert_eq!(LeU16::decode(&a).unwrap(), 1);
        assert_eq!(LeU16::decode(&b).unwrap(), 2);
        assert_eq!(LeU16::decode(&c).unwrap(), 10);
    }
}

