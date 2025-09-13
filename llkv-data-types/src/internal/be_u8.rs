use super::*;

/// Big-endian u8 codec. Lexicographic order == numeric order.
pub struct BeU8;

impl Codec for BeU8 {
    const WIDTH: usize = 1;
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

    #[inline]
    fn decode_many_into(dst: &mut [u8], src: &[u8]) -> Result<(), DecodeError> {
        let n = dst.len();
        if src.len() != n {
            return Err(DecodeError::NotEnoughData);
        }
        dst.copy_from_slice(src);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beu8_roundtrip_and_order() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();

        1u8.encode_into(&mut a);
        2u8.encode_into(&mut b);
        10u8.encode_into(&mut c);

        // Lex order == numeric order
        assert!(a < b && b < c);

        // Round-trip
        assert_eq!(BeU8::decode(&a).unwrap(), 1);
        assert_eq!(BeU8::decode(&b).unwrap(), 2);
        assert_eq!(BeU8::decode(&c).unwrap(), 10);
    }
}
