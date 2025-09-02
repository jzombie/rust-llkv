use std::ops::Range;

// ============================ Zero-copy value view ===========================

#[derive(Clone)]
pub struct ValueRef<Page>
where
    Page: Clone + core::ops::Deref<Target = [u8]>,
{
    page: Page,
    range: Range<usize>,
}

impl<Page> ValueRef<Page>
where
    Page: Clone + core::ops::Deref<Target = [u8]>,
{
    #[inline]
    pub(crate) fn new(page: Page, range: Range<usize>) -> Self {
        Self { page, range }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.page[self.range.clone()]
    }

    #[inline]
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let base = self.range.start;
        Self {
            page: self.page.clone(),
            range: (base + start)..(base + end),
        }
    }

    #[inline]
    pub fn slice_from(&self, offset: usize) -> Self {
        let start = self.range.start + offset;
        Self {
            page: self.page.clone(),
            range: start..self.range.end,
        }
    }
}

impl<Page> AsRef<[u8]> for ValueRef<Page>
where
    Page: Clone + core::ops::Deref<Target = [u8]>,
{
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl<Page> core::ops::Deref for ValueRef<Page>
where
    Page: Clone + core::ops::Deref<Target = [u8]>,
{
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
