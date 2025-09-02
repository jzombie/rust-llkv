use crate::codecs::read_u32_at;
use crate::errors::Error;
use crate::pager::Pager;
use std::ops::Range;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NodeTag {
    Internal = 0,
    Leaf = 1,
}
impl NodeTag {
    pub(crate) fn from_u8(x: u8) -> Result<Self, Error> {
        match x {
            0 => Ok(NodeTag::Internal),
            1 => Ok(NodeTag::Leaf),
            _ => Err(Error::Corrupt("unknown node tag")),
        }
    }
}

#[derive(Clone)]
pub struct NodeView<P: Pager> {
    page: P::Page,
}

impl<P: Pager> NodeView<P> {
    pub(crate) fn new(page: P::Page) -> Result<Self, Error> {
        if page.len() < 9 {
            Err(Error::Corrupt("node too small"))
        } else {
            Ok(Self { page })
        }
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.page
    }

    #[inline]
    pub(crate) fn into_page(self) -> P::Page {
        self.page
    }

    #[inline]
    pub(crate) fn tag(&self) -> Result<NodeTag, Error> {
        NodeTag::from_u8(self.as_slice()[0])
    }

    #[inline]
    pub(crate) fn count(&self) -> usize {
        u32::from_le_bytes(self.as_slice()[1..5].try_into().unwrap()) as usize
    }

    #[inline]
    pub(crate) fn aux_len(&self) -> usize {
        u32::from_le_bytes(self.as_slice()[5..9].try_into().unwrap()) as usize
    }

    #[inline]
    fn hdr_len(&self) -> usize {
        9
    }

    #[inline]
    fn is_leaf(&self) -> bool {
        self.as_slice()[0] == NodeTag::Leaf as u8
    }

    #[inline]
    pub(crate) fn aux_range(&self) -> Range<usize> {
        let h = self.hdr_len();
        h..(h + self.aux_len())
    }

    #[inline]
    fn entries_start(&self) -> usize {
        self.hdr_len() + self.aux_len()
    }

    #[inline]
    fn index_entry_size(&self) -> usize {
        if self.is_leaf() { 16 } else { 4 }
    }

    #[inline]
    fn index_table_start(&self) -> usize {
        self.as_slice().len() - self.count() * self.index_entry_size()
    }

    // -------- Internal node entry decoding (row layout) --------

    #[inline]
    fn entry_offset_at(&self, i: usize) -> usize {
        let base = self.index_table_start();
        let off = u32::from_le_bytes(
            self.as_slice()[base + i * 4..base + (i + 1) * 4]
                .try_into()
                .unwrap(),
        );
        self.entries_start() + off as usize
    }

    #[inline]
    fn entry_slice(&self, i: usize) -> &[u8] {
        let start = self.entry_offset_at(i);
        let end = if i + 1 < self.count() {
            self.entry_offset_at(i + 1)
        } else {
            self.index_table_start()
        };
        &self.as_slice()[start..end]
    }

    pub(crate) fn internal_entry_slices(&self, i: usize) -> (&[u8], &[u8]) {
        let e = self.entry_slice(i);
        let (klen, mut pos) = read_u32_at(e, 0);
        let key = &e[pos..pos + klen as usize];
        pos += klen as usize;
        let (ilen, pos2) = read_u32_at(e, pos);
        let idb = &e[pos2..pos2 + ilen as usize];
        (key, idb)
    }

    // ------------- Leaf node decoding (columnar layout) -------------

    #[inline]
    fn leaf_keys_base(&self) -> usize {
        self.entries_start()
    }

    #[inline]
    fn leaf_keys_len(&self) -> usize {
        let n = self.count();
        if n == 0 {
            return 0;
        }
        let base = self.index_table_start() + (n - 1) * 16;
        let k_off =
            u32::from_le_bytes(self.as_slice()[base..base + 4].try_into().unwrap()) as usize;
        let k_len =
            u32::from_le_bytes(self.as_slice()[base + 4..base + 8].try_into().unwrap()) as usize;
        k_off + k_len
    }

    #[inline]
    fn leaf_values_base(&self) -> usize {
        self.leaf_keys_base() + self.leaf_keys_len()
    }

    #[inline(always)]
    pub(crate) fn leaf_index_fields(&self, i: usize) -> (usize, usize, usize, usize) {
        let base = self.index_table_start() + i * 16;
        let k_off =
            u32::from_le_bytes(self.as_slice()[base..base + 4].try_into().unwrap()) as usize;
        let k_len =
            u32::from_le_bytes(self.as_slice()[base + 4..base + 8].try_into().unwrap()) as usize;
        let v_off =
            u32::from_le_bytes(self.as_slice()[base + 8..base + 12].try_into().unwrap()) as usize;
        let v_len =
            u32::from_le_bytes(self.as_slice()[base + 12..base + 16].try_into().unwrap()) as usize;
        (k_off, k_len, v_off, v_len)
    }

    #[inline]
    pub(crate) fn leaf_entry_key_range(&self, i: usize) -> core::ops::Range<usize> {
        let (k_off, k_len, _, _) = self.leaf_index_fields(i);
        let start = self.leaf_keys_base() + k_off;
        let end = start + k_len;
        start..end
    }

    #[inline]
    pub(crate) fn leaf_entry_slices(&self, i: usize) -> (&[u8], &[u8]) {
        let (k_off, k_len, v_off, v_len) = self.leaf_index_fields(i);
        let kb = self.leaf_keys_base();
        let vb = self.leaf_values_base();
        let key = &self.as_slice()[kb + k_off..kb + k_off + k_len];
        let val = &self.as_slice()[vb + v_off..vb + v_off + v_len];
        (key, val)
    }

    /// Exact byte range of the value payload for entry `i`.
    #[inline(always)]
    pub(crate) fn leaf_entry_value_range(&self, i: usize) -> Range<usize> {
        let (_, _, v_off, v_len) = self.leaf_index_fields(i);
        let start = self.leaf_values_base() + v_off;
        let end = start + v_len;
        start..end
    }

    /// Aux area of a leaf: [u32 next_len][next_id_bytes...].
    #[inline]
    pub(crate) fn leaf_next_aux(&self) -> &[u8] {
        let aux = &self.as_slice()[self.aux_range()];
        if aux.len() < 4 {
            return &aux[0..0];
        }
        let (n, pos) = read_u32_at(aux, 0);
        &aux[pos..pos + n as usize]
    }

    /// Returns Some(width) if all keys are fixed-width and packed;
    /// else None.
    #[inline]
    pub(crate) fn leaf_fixed_key_width(&self) -> Option<usize> {
        if !self.is_leaf() {
            return None;
        }
        let n = self.count();
        if n == 0 {
            return None;
        }
        let keys_len = self.leaf_keys_len();
        if keys_len == 0 || keys_len % n != 0 {
            return None;
        }
        let w = keys_len / n;
        // Validate a few positions to avoid O(n) checking.
        let checks = [0usize, n / 2, n - 1].into_iter().collect::<Vec<_>>();
        for &i in checks.iter() {
            let (k_off, k_len, _, _) = self.leaf_index_fields(i);
            if k_len != w || k_off != i * w {
                return None;
            }
        }
        Some(w)
    }

    /// Returns the contiguous keys block (leaf only).
    #[inline]
    pub(crate) fn leaf_keys_block(&self) -> &[u8] {
        let kb = self.leaf_keys_base();
        &self.as_slice()[kb..kb + self.leaf_keys_len()]
    }
}
