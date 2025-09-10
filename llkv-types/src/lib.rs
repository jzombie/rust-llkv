#![forbid(unsafe_code)]

/// Case-insensitive, order-preserving UTF-8 codec for use with byte-ordered stores.
/// Layout: [be32(len_folded)] [folded_utf8_bytes] [original_utf8_bytes]
///
/// - For ASCII, we fold with `to_ascii_lowercase()` (fast path).
/// - For non-ASCII, we do NFKC then full Unicode lowercase (`char::to_lowercase`).
/// - Value-order compares by the folded prefix first, giving case-insensitive ordering,
///   and uses the original bytes only as a stable tie-breaker (so "APPLE" and "apple"
///   group together but keep a deterministic order).

pub struct Utf8CaseFold;

impl Utf8CaseFold {
    #[inline]
    pub fn encode_into(dst: &mut Vec<u8>, s: &str) {
        // ASCII fast path
        if s.is_ascii() {
            dst.reserve_exact(s.len() + 1 + s.len());
            for &b in s.as_bytes() {
                dst.push(b.to_ascii_lowercase());
            }
            dst.push(0);
            dst.extend_from_slice(s.as_bytes());
            return;
        }

        // Full Unicode: NFKC + lowercase
        use unicode_normalization::UnicodeNormalization;
        let folded: String = s.nfkc().flat_map(char::to_lowercase).collect();
        let f = folded.as_bytes();

        dst.reserve_exact(f.len() + 1 + s.len());
        dst.extend_from_slice(f);
        dst.push(0);
        dst.extend_from_slice(s.as_bytes());
    }

    #[inline]
    pub fn decode(bytes: &[u8]) -> String {
        let split = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        let orig = bytes.get(split.saturating_add(1)..).unwrap_or_default();
        String::from_utf8(orig.to_vec())
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }

    #[inline]
    pub fn folded_key(bytes: &[u8]) -> &[u8] {
        let split = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        &bytes[..split]
    }
}

#[cfg(test)]
mod tests {
    use super::Utf8CaseFold;

    #[test]
    fn ascii_roundtrip_and_order() {
        let mut buf_a = Vec::new();
        let mut buf_b = Vec::new();
        Utf8CaseFold::encode_into(&mut buf_a, "Large Words");
        Utf8CaseFold::encode_into(&mut buf_b, "lower words");

        // Decodes to original
        assert_eq!(Utf8CaseFold::decode(&buf_a), "Large Words");
        assert_eq!(Utf8CaseFold::decode(&buf_b), "lower words");

        // Folded keys compare case-insensitively: "large words" < "lower words"
        assert!(Utf8CaseFold::folded_key(&buf_a) < Utf8CaseFold::folded_key(&buf_b));
    }

    #[test]
    fn unicode_nfkc_and_lowercase() {
        // Angstrom sign Å (U+212B) NFKC→Å, lowercase→å; Kelvin sign K (U+212A)→K→k
        let s1 = "Å"; // compatibility equivalent of Å
        let s2 = "K"; // compatibility equivalent of K

        let mut a = Vec::new();
        let mut b = Vec::new();
        Utf8CaseFold::encode_into(&mut a, s1);
        Utf8CaseFold::encode_into(&mut b, s2);

        let fa = std::str::from_utf8(Utf8CaseFold::folded_key(&a)).unwrap();
        let fb = std::str::from_utf8(Utf8CaseFold::folded_key(&b)).unwrap();

        assert_eq!(fa, "å"); // folded result
        assert_eq!(fb, "k");
        assert!(fa > fb); // "å" (0xC3 A5) sorts after "k"
    }
}
