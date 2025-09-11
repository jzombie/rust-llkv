use super::*;
use icu_casemap::{CaseMapper, CaseMapperBorrowed};
use icu_normalizer::{ComposingNormalizer, ComposingNormalizerBorrowed};
use std::sync::OnceLock;

/* ----------------------- Text: case-insensitive UTF-8 ------------------- */

/// Case-insensitive, order-preserving UTF-8 codec using ICU4X:
///   NFKC → default case fold (full Unicode).
///
/// Layout: `[folded_utf8][original_utf8][be32 len_folded]`.
pub struct Utf8CaseFold;

impl Utf8CaseFold {
    // ICU singletons (compiled data; see Cargo features below).
    fn cm() -> &'static CaseMapperBorrowed<'static> {
        static CM: OnceLock<CaseMapperBorrowed<'static>> = OnceLock::new();
        CM.get_or_init(CaseMapper::new)
    }
    fn nfkc() -> &'static ComposingNormalizerBorrowed<'static> {
        static NFKC: OnceLock<ComposingNormalizerBorrowed<'static>> = OnceLock::new();
        NFKC.get_or_init(ComposingNormalizer::new_nfkc)
    }

    /// NFKC + default case fold (full Unicode, not just lowercase).
    #[inline]
    fn fold_nfkc(s: &str) -> String {
        let normalized = Self::nfkc().normalize(s);
        Self::cm().fold_string(&normalized).into_owned()
    }

    /// Encode layout:
    ///   \[folded_utf8\]\[original_utf8\]\[be32(len_folded)\]
    /// This keeps lex order correct while making decode trivial and NUL-safe.
    #[inline]
    fn encode_impl(dst: &mut Vec<u8>, s: &str) {
        // ASCII fast path: use vectorized operations instead of a byte-by-byte loop.
        if s.is_ascii() {
            let s_bytes = s.as_bytes();
            let f_len = s_bytes.len();
            dst.reserve_exact(f_len * 2 + 4);

            // This is faster than iterating and pushing bytes individually.
            dst.extend(s_bytes.iter().map(|b| b.to_ascii_lowercase()));
            dst.extend_from_slice(s_bytes);
            dst.extend_from_slice(&(f_len as u32).to_be_bytes());
            return;
        }

        // Non-ASCII path using ICU.
        let folded = Self::fold_nfkc(s);
        let f = folded.as_bytes();
        dst.reserve_exact(f.len() + s.len() + 4);
        dst.extend_from_slice(f);
        dst.extend_from_slice(s.as_bytes());
        dst.extend_from_slice(&(f.len() as u32).to_be_bytes());
    }

    /// Borrow just the folded key part.
    #[inline]
    pub fn folded_key(bytes: &[u8]) -> &[u8] {
        if bytes.len() < 4 {
            return &[];
        }
        let len_bytes: [u8; 4] = bytes[bytes.len() - 4..].try_into().unwrap();
        let len = u32::from_be_bytes(len_bytes) as usize;

        // Bounds check to prevent panics on malformed input.
        bytes.get(..len).unwrap_or(&[])
    }

    /// Decode the original string segment without allocating a new String.
    #[inline]
    pub fn decode_borrowed(src: &[u8]) -> Option<&str> {
        if src.len() < 4 {
            return None;
        }
        let len_bytes: [u8; 4] = src[src.len() - 4..].try_into().ok()?;
        let folded_len = u32::from_be_bytes(len_bytes) as usize;

        if folded_len + 4 > src.len() {
            return None;
        }

        let original_bytes = &src[folded_len..src.len() - 4];
        std::str::from_utf8(original_bytes).ok()
    }
}

impl Codec for Utf8CaseFold {
    type Borrowed<'a> = &'a str;
    type Owned = String;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, s: &str) -> Result<(), EncodeError> {
        Self::encode_impl(dst, s);
        Ok(())
    }

    /// Decode the original (post-fold) segment.
    /// This now calls the allocation-free `decode_borrowed` for max performance.
    #[inline]
    fn decode(src: &[u8]) -> Result<String, DecodeError> {
        Self::decode_borrowed(src)
            .map(|s| s.to_owned())
            .ok_or(DecodeError::InvalidFormat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_ascii_roundtrip_and_fold_order() {
        // "Large Words" vs "lower words": both fold to lower,
        // and "large..." < "lower..." so the folded prefix orders them.
        let mut a = Vec::new();
        let mut b = Vec::new();
        "Large Words".encode_into(&mut a);
        "lower words".encode_into(&mut b);

        // Round-trip original
        assert_eq!(Utf8CaseFold::decode(&a).unwrap(), "Large Words");
        assert_eq!(Utf8CaseFold::decode(&b).unwrap(), "lower words");

        // Compare folded prefixes
        assert!(Utf8CaseFold::folded_key(&a) < Utf8CaseFold::folded_key(&b));
    }

    #[test]
    fn utf8_unicode_nfkc_and_casefold() {
        // Å (U+212B) → Å → å ;
        // K (U+212A) → K → k
        let s1 = "Å";
        let s2 = "K";
        let mut a = Vec::new();
        let mut b = Vec::new();
        s1.encode_into(&mut a);
        s2.encode_into(&mut b);

        assert_eq!(Utf8CaseFold::decode(&a).unwrap(), s1);
        assert_eq!(Utf8CaseFold::decode(&b).unwrap(), s2);

        let fa = std::str::from_utf8(Utf8CaseFold::folded_key(&a)).unwrap();
        let fb = std::str::from_utf8(Utf8CaseFold::folded_key(&b)).unwrap();

        assert_eq!(fa, "å");
        assert_eq!(fb, "k");
        assert!(fa > fb);
    }

    #[test]
    fn utf8_simple_roundtrip() {
        let cases = vec![
            "Hello, World!", // Simple ASCII
            "👋🌎",          // Unicode Emoji
            "résumé",        // Unicode with accents
            "Straße",        // German character that case-folds
            "",              // Empty string
        ];
        for original in cases {
            let mut encoded = Vec::new();
            original.encode_into(&mut encoded);
            let decoded = Utf8CaseFold::decode(&encoded).unwrap();
            assert_eq!(original, decoded, "Failed roundtrip for: {}", original);
        }
    }

    #[test]
    fn utf8_embedded_nul_safe_with_trailer_len() {
        // Embedded NUL is fine;
        // we don't need a sentinel.
        let s = "A\0B";
        let mut buf = Vec::new();
        s.encode_into(&mut buf);

        assert_eq!(Utf8CaseFold::decode(&buf).unwrap(), s);

        // Folded starts with "a\0b" (ASCII lowercased)
        let fk = Utf8CaseFold::folded_key(&buf);
        assert_eq!(fk, b"a\0b");
    }

    #[test]
    fn utf8_decode_borrowed_works_with_pointer_checks() {
        let s = "A Test String";
        let mut buf = Vec::new();
        s.encode_into(&mut buf);

        // Test the zero-allocation path
        let decoded_borrowed = Utf8CaseFold::decode_borrowed(&buf).unwrap();
        assert_eq!(s, decoded_borrowed);

        // --- Advanced Pointer Assertions ---

        // Get the memory range of the buffer
        let buf_range = buf.as_ptr_range();

        // Get the start pointer and length of the decoded slice
        let decoded_start_ptr = decoded_borrowed.as_ptr();
        let decoded_len = decoded_borrowed.len();

        // Calculate the end pointer's address as a usize
        let decoded_end_addr = decoded_start_ptr as usize + decoded_len;

        // Assert that the decoded slice's memory is *within* the buffer's memory
        assert!(decoded_start_ptr >= buf_range.start);
        assert!(decoded_end_addr <= buf_range.end as usize);
    }
}
