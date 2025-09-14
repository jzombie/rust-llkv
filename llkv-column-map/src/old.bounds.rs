use crate::constants::VALUE_BOUND_MAX;
use crate::types::ByteLen;

#[derive(Clone, Debug, PartialEq, Eq, bitcode::Encode, bitcode::Decode)]
pub struct ValueBound {
    /// Full byte length of the value this bound represents.
    pub total_len: ByteLen,
    /// First min(VALUE_BOUND_MAX, total_len) bytes of that value.
    pub prefix: Vec<u8>,
}

impl ValueBound {
    #[inline]
    pub fn from_bytes(b: &[u8]) -> Self {
        let take = core::cmp::min(b.len(), VALUE_BOUND_MAX);
        Self {
            total_len: b.len() as u32,
            prefix: b[..take].to_vec(),
        }
    }

    #[inline]
    pub fn is_truncated(&self) -> bool {
        (self.prefix.len() as u32) < self.total_len
    }

    #[inline]
    pub fn min_max_bounds(values: &[Vec<u8>]) -> (ValueBound, ValueBound) {
        debug_assert!(!values.is_empty());
        let mut min = &values[0];
        let mut max = &values[0];
        for v in &values[1..] {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        (ValueBound::from_bytes(min), ValueBound::from_bytes(max))
    }
}
