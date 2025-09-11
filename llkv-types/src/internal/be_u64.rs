use super::*;

/// Big-endian u64 codec. Lexicographic order == numeric order.
pub struct BeU64;

impl Codec for BeU64 {
    type Borrowed<'a> = &'a u64;
    type Owned = u64;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: &u64) {
        dst.extend_from_slice(&v.to_be_bytes());
    }

    /// This is more robust and can be faster than a separate copy.
    #[inline]
    fn decode(src: &[u8]) -> u64 {
        u64::from_be_bytes(
            src[..8]
                .try_into()
                .expect("source slice has less than 8 bytes"),
        )
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
        assert_eq!(BeU64::decode(&a), 1);
        assert_eq!(BeU64::decode(&b), 2);
        assert_eq!(BeU64::decode(&c), 10);
    }
}
