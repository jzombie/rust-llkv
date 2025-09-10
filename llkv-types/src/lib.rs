#![forbid(unsafe_code)]

/// A tiny, order-preserving, case-insensitive UTF-8 codec.
///
/// Layout: lowercased_ascii(s) + 0x00 + original_bytes(s)
/// - OrderBy::Value uses the folded prefix; originals remain intact.
/// - Works for ASCII case folding. (ASCII only; extend later if you need full Unicode casefold.)
pub struct Utf8CaseFold;

impl Utf8CaseFold {
    /// Encode into `dst` without intermediate allocations for the folded key.
    #[inline]
    pub fn encode_into(dst: &mut Vec<u8>, s: &str) {
        // Reserve: folded + NUL + original
        dst.reserve_exact(s.len() + 1 + s.len());

        // Fold to ASCII lower directly into dst
        for &b in s.as_bytes() {
            dst.push(b.to_ascii_lowercase());
        }
        dst.push(0); // delimiter
        dst.extend_from_slice(s.as_bytes());
    }

    /// Decode original string back (the substring after first 0x00).
    #[inline]
    pub fn decode(bytes: &[u8]) -> String {
        let split = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        let orig = bytes.get(split.saturating_add(1)..).unwrap_or_default();
        // Safety: stored as original UTF-8; if ever invalid, fall back lossily.
        String::from_utf8(orig.to_vec()).unwrap_or_else(|e| {
            // If someone fed non-UTF8, keep it lossless-ish.
            String::from_utf8_lossy(e.as_bytes()).into_owned()
        })
    }

    /// Returns the folded key slice (before 0x00). Useful for debugging/tests.
    #[inline]
    pub fn folded_key<'a>(bytes: &'a [u8]) -> &'a [u8] {
        let split = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        &bytes[..split]
    }
}
