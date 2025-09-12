use super::*;

/// Little-endian u8 codec. Trivial 1-byte payload.
pub struct LeU8;

impl Codec for LeU8 {
    type Borrowed<'a> = &'a u8;
    type Owned = u8;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: &u8) -> Result<(), EncodeError> {
        dst.push(*v);
        Ok(())
    }

    #[inline]
    fn decode(src: &[u8]) -> Result<u8, DecodeError> {
        if src.is_empty() {
            return Err(DecodeError::NotEnoughData);
        }
        Ok(src[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leu8_roundtrip() {
        let mut a = Vec::new();
        7u8.encode_into(&mut a);
        assert_eq!(LeU8::decode(&a).unwrap(), 7);
    }
}

