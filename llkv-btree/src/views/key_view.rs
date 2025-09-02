// ============================ Zero-copy key view =============================

#[derive(Clone)]
pub struct KeyRef<Page>
where
    Page: Clone + core::ops::Deref<Target = [u8]>,
{
    page: Page,
    range: core::ops::Range<usize>,
}

impl<Page> KeyRef<Page>
where
    Page: Clone + core::ops::Deref<Target = [u8]>,
{
    // TODO: Use?
    // #[inline]
    // pub(crate) fn new(page: Page, range: core::ops::Range<usize>) -> Self {
    //     Self { page, range }
    // }

    #[inline]
    pub(crate) fn from_subslice(page: Page, sub: &[u8]) -> Self {
        let pb: &[u8] = &page;
        let base = pb.as_ptr() as usize;
        let start = sub.as_ptr() as usize - base;
        let end = start + sub.len();
        Self {
            page,
            range: start..end,
        }
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

impl<Page> AsRef<[u8]> for KeyRef<Page>
where
    Page: Clone + core::ops::Deref<Target = [u8]>,
{
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl<Page> core::ops::Deref for KeyRef<Page>
where
    Page: Clone + core::ops::Deref<Target = [u8]>,
{
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
