use super::*;

/// Codec for bool. `false` -> `0u8`, `true` -> `1u8`.
pub struct Bool;

impl Codec for Bool {
    type Borrowed<'a> = &'a bool;
    type Owned = bool;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: &bool) {
        dst.push(if *v { 1 } else { 0 });
    }

    #[inline]
    fn decode(src: &[u8]) -> bool {
        src[0] != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_roundtrip_and_order() {
        let mut f_bytes = Vec::new();
        let mut t_bytes = Vec::new();

        false.encode_into(&mut f_bytes);
        true.encode_into(&mut t_bytes);

        // Check byte representation
        assert_eq!(f_bytes, &[0]);
        assert_eq!(t_bytes, &[1]);

        // Lex order: false < true
        assert!(f_bytes < t_bytes);

        // Round-trip
        assert!(!Bool::decode(&f_bytes));
        assert!(Bool::decode(&t_bytes));
    }
}
