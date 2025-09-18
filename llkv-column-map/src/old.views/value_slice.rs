use crate::types::ByteOffset;

/// Zero-copy view into a subrange of an Arc-backed data blob.
/// Holds the Arc and byte offsets; derefs to `[u8]` without copying.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueSlice<B> {
    pub(crate) data: B,
    pub(crate) start: ByteOffset,
    pub(crate) end: ByteOffset, // exclusive
}

impl<B: AsRef<[u8]> + Clone> ValueSlice<B> {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.data.as_ref()[self.start as usize..self.end as usize]
    }

    #[inline]
    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn data(&self) -> B {
        self.data.clone()
    }

    #[inline]
    pub fn start(&self) -> ByteOffset {
        self.start
    }

    #[inline]
    pub fn end(&self) -> ByteOffset {
        self.end
    }
}

impl<B: AsRef<[u8]> + Clone> std::ops::Deref for ValueSlice<B> {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<B: AsRef<[u8]> + Clone> AsRef<[u8]> for ValueSlice<B> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}
