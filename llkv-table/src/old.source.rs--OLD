//! Streaming sources of row-ids with zero allocation.
//!
//! Each Source yields sorted, unique row-ids as &[u8]. Returned
//! references remain valid until the next call to peek/next.

#![forbid(unsafe_code)]

use crate::types::RowIdCmp;
use core::cmp::Ordering;

/// Stream of row-ids. Implementations must be sorted and deduped.
pub trait Source<'a> {
    /// Peek the current row-id without advancing.
    fn peek(&mut self) -> Option<&'a [u8]>;

    /// Advance and return the current row-id.
    fn next(&mut self) -> Option<&'a [u8]>;
}

/// Intersection of two sorted Sources (A ∩ B).
pub struct Intersect2<'a, A: Source<'a>, B: Source<'a>> {
    a: A,
    b: B,
    cmp: RowIdCmp,
    pa: Option<&'a [u8]>,
    pb: Option<&'a [u8]>,
}

impl<'a, A: Source<'a>, B: Source<'a>> Intersect2<'a, A, B> {
    pub fn new(mut a: A, mut b: B, cmp: RowIdCmp) -> Self {
        let pa = a.peek();
        let pb = b.peek();
        Self { a, b, cmp, pa, pb }
    }
}

impl<'a, A: Source<'a>, B: Source<'a>> Source<'a> for Intersect2<'a, A, B> {
    fn peek(&mut self) -> Option<&'a [u8]> {
        loop {
            match (self.pa, self.pb) {
                (Some(x), Some(y)) => match (self.cmp)(x, y) {
                    Ordering::Equal => return self.pa,
                    Ordering::Less => {
                        self.a.next();
                        self.pa = self.a.peek();
                    }
                    Ordering::Greater => {
                        self.b.next();
                        self.pb = self.b.peek();
                    }
                },
                _ => return None,
            }
        }
    }

    fn next(&mut self) -> Option<&'a [u8]> {
        let cur = self.peek()?;
        self.a.next();
        self.b.next();
        self.pa = self.a.peek();
        self.pb = self.b.peek();
        Some(cur)
    }
}

/// Union of two sorted Sources with dedup (A ∪ B).
pub struct Union2<'a, A: Source<'a>, B: Source<'a>> {
    a: A,
    b: B,
    cmp: RowIdCmp,
    pa: Option<&'a [u8]>,
    pb: Option<&'a [u8]>,
}

impl<'a, A: Source<'a>, B: Source<'a>> Union2<'a, A, B> {
    pub fn new(mut a: A, mut b: B, cmp: RowIdCmp) -> Self {
        let pa = a.peek();
        let pb = b.peek();
        Self { a, b, cmp, pa, pb }
    }
}

impl<'a, A: Source<'a>, B: Source<'a>> Source<'a> for Union2<'a, A, B> {
    fn peek(&mut self) -> Option<&'a [u8]> {
        match (self.pa, self.pb) {
            (None, None) => None,
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (Some(x), Some(y)) => match (self.cmp)(x, y) {
                Ordering::Less => Some(x),
                Ordering::Greater => Some(y),
                Ordering::Equal => Some(x),
            },
        }
    }

    fn next(&mut self) -> Option<&'a [u8]> {
        let cur = self.peek()?;
        match (self.pa, self.pb) {
            (Some(x), Some(y)) => match (self.cmp)(x, y) {
                Ordering::Less => {
                    self.a.next();
                    self.pa = self.a.peek();
                }
                Ordering::Greater => {
                    self.b.next();
                    self.pb = self.b.peek();
                }
                Ordering::Equal => {
                    self.a.next();
                    self.b.next();
                    self.pa = self.a.peek();
                    self.pb = self.b.peek();
                }
            },
            (Some(_), None) => {
                self.a.next();
                self.pa = self.a.peek();
            }
            (None, Some(_)) => {
                self.b.next();
                self.pb = self.b.peek();
            }
            _ => {}
        }
        Some(cur)
    }
}

/// Difference of two sorted Sources (A \\ B).
pub struct Diff2<'a, A: Source<'a>, B: Source<'a>> {
    a: A,
    b: B,
    cmp: RowIdCmp,
    pa: Option<&'a [u8]>,
    pb: Option<&'a [u8]>,
}

impl<'a, A: Source<'a>, B: Source<'a>> Diff2<'a, A, B> {
    pub fn new(mut a: A, mut b: B, cmp: RowIdCmp) -> Self {
        let pa = a.peek();
        let pb = b.peek();
        Self { a, b, cmp, pa, pb }
    }
}

impl<'a, A: Source<'a>, B: Source<'a>> Source<'a> for Diff2<'a, A, B> {
    fn peek(&mut self) -> Option<&'a [u8]> {
        loop {
            match (self.pa, self.pb) {
                (Some(x), Some(y)) => match (self.cmp)(x, y) {
                    Ordering::Less => return Some(x),
                    Ordering::Greater => {
                        self.b.next();
                        self.pb = self.b.peek();
                    }
                    Ordering::Equal => {
                        self.a.next();
                        self.pa = self.a.peek();
                        self.b.next();
                        self.pb = self.b.peek();
                    }
                },
                (Some(x), None) => return Some(x),
                _ => return None,
            }
        }
    }

    fn next(&mut self) -> Option<&'a [u8]> {
        let cur = self.peek()?;
        self.a.next();
        self.pa = self.a.peek();
        Some(cur)
    }
}

/// Simple slice-backed Source for tests and adapters.
///
/// It expects `items` to be sorted and unique under the provided
/// comparator. It does not verify ordering.
pub struct SliceSource<'a> {
    items: &'a [&'a [u8]],
    i: usize,
}

impl<'a> SliceSource<'a> {
    pub fn new(items: &'a [&'a [u8]]) -> Self {
        Self { items, i: 0 }
    }
}

impl<'a> Source<'a> for SliceSource<'a> {
    fn peek(&mut self) -> Option<&'a [u8]> {
        self.items.get(self.i).copied()
    }

    fn next(&mut self) -> Option<&'a [u8]> {
        let cur = self.items.get(self.i).copied()?;
        self.i += 1;
        Some(cur)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmp(a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }

    #[test]
    fn slice_source_basic() {
        let a = [b"a".as_slice(), b"b".as_slice()];
        let mut s = SliceSource::new(&a);
        assert_eq!(s.peek(), Some(b"a".as_slice()));
        assert_eq!(s.next(), Some(b"a".as_slice()));
        assert_eq!(s.next(), Some(b"b".as_slice()));
        assert_eq!(s.next(), None);
    }

    #[test]
    fn intersect2_streams() {
        let a = [b"a".as_slice(), b"b".as_slice(), b"d".as_slice()];
        let b = [b"b".as_slice(), b"c".as_slice(), b"d".as_slice()];
        let mut i = Intersect2::new(SliceSource::new(&a), SliceSource::new(&b), cmp);
        assert_eq!(i.next(), Some(b"b".as_slice()));
        assert_eq!(i.next(), Some(b"d".as_slice()));
        assert_eq!(i.next(), None);
    }

    #[test]
    fn union2_streams() {
        let a = [b"a".as_slice(), b"b".as_slice(), b"d".as_slice()];
        let b = [b"b".as_slice(), b"c".as_slice(), b"d".as_slice()];
        let mut u = Union2::new(SliceSource::new(&a), SliceSource::new(&b), cmp);
        assert_eq!(u.next(), Some(b"a".as_slice()));
        assert_eq!(u.next(), Some(b"b".as_slice()));
        assert_eq!(u.next(), Some(b"c".as_slice()));
        assert_eq!(u.next(), Some(b"d".as_slice()));
        assert_eq!(u.next(), None);
    }

    #[test]
    fn diff2_streams() {
        let a = [b"a".as_slice(), b"b".as_slice(), b"d".as_slice()];
        let b = [b"b".as_slice(), b"c".as_slice()];
        let mut d = Diff2::new(SliceSource::new(&a), SliceSource::new(&b), cmp);
        assert_eq!(d.next(), Some(b"a".as_slice()));
        assert_eq!(d.next(), Some(b"d".as_slice()));
        assert_eq!(d.next(), None);
    }
}
