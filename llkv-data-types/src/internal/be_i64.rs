use super::*;

/// Big-endian i64 codec with sign-bit flip so lexicographic order == numeric order.
pub struct BeI64;

impl BeI64 {
    #[inline]
    fn to_lex(v: i64) -> u64 {
        (v as u64) ^ 0x8000_0000_0000_0000
    }
    #[inline]
    fn from_lex(u: u64) -> i64 {
        (u ^ 0x8000_0000_0000_0000) as i64
    }
}

impl Codec for BeI64 {
    const WIDTH: usize = 8;
    type Borrowed<'a> = &'a i64;
    type Owned = i64;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: &i64) -> Result<(), EncodeError> {
        let x = Self::to_lex(*v);
        dst.extend_from_slice(&x.to_be_bytes());
        Ok(())
    }

    #[inline]
    fn decode(src: &[u8]) -> Result<i64, DecodeError> {
        if src.len() < 8 {
            return Err(DecodeError::NotEnoughData);
        }
        let bytes: [u8; 8] = src[..8].try_into().unwrap();
        Ok(Self::from_lex(u64::from_be_bytes(bytes)))
    }

    // Specialized fast path to avoid per-item length checks in the hot loop.
    #[inline]
    fn decode_many_into(dst: &mut [i64], src: &[u8]) -> Result<(), DecodeError> {
        let n = dst.len();
        if src.len() != n.saturating_mul(8) {
            return Err(DecodeError::NotEnoughData);
        }
        let mut off = 0usize;
        for out in dst.iter_mut() {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&src[off..off + 8]);
            let u = u64::from_be_bytes(bytes);
            *out = Self::from_lex(u);
            off += 8;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bei64_roundtrip_and_order() {
        // Include negatives, zero, positives, and extremes
        let vals = [i64::MIN + 1, -10, -1, 0, 1, 10, i64::MAX];

        // Encode all
        let encoded: Vec<Vec<u8>> = vals
            .iter()
            .map(|v| {
                let mut b = Vec::new();
                v.encode_into(&mut b);
                b
            })
            .collect();

        // Lex sort of bytes should match numeric sort of values
        let mut bytes_sorted = encoded.clone();
        bytes_sorted.sort(); // lexicographic (bytewise)

        let mut vals_sorted = vals.to_vec();
        vals_sorted.sort(); // numeric

        let decoded: Vec<i64> = bytes_sorted
            .iter()
            .map(|b| BeI64::decode(b).unwrap())
            .collect();

        assert_eq!(decoded, vals_sorted, "lexicographic != numeric order");

        // Spot round-trips
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(BeI64::decode(&encoded[i]).unwrap(), *v);
        }
    }
}
