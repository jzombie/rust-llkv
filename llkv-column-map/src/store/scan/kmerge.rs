use super::*;

/// Generic k-way merge over sorted per-chunk arrays, coalescing runs.
/// `get` fetches the T value at index for an array, and `emit` sends runs.
pub fn kmerge_coalesced<T, A, FLen, FGet, FEmit>(
    arrays: &[A],
    mut len_of: FLen,
    mut get: FGet,
    mut emit: FEmit,
) where
    T: Ord + Copy,
    FLen: FnMut(&A) -> usize,
    FGet: FnMut(&A, usize) -> T,
    FEmit: FnMut(usize, usize, usize), // (chunk_idx, start, len)
{
    #[derive(Clone, Copy, Debug)]
    struct H<T> {
        v: T,
        c: usize,
        i: usize,
    }
    impl<T: Ord> PartialEq for H<T> {
        fn eq(&self, o: &Self) -> bool {
            self.v == o.v && self.c == o.c && self.i == o.i
        }
    }
    impl<T: Ord> Eq for H<T> {}
    impl<T: Ord> PartialOrd for H<T> {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    // Max-heap by value; break ties by chunk to keep deterministic.
    impl<T: Ord> Ord for H<T> {
        fn cmp(&self, o: &Self) -> Ordering {
            // Reverse value ordering to get a min-heap behavior via BinaryHeap
            o.v.cmp(&self.v).then_with(|| o.c.cmp(&self.c))
        }
    }

    let mut heap: BinaryHeap<H<T>> = BinaryHeap::new();
    for (ci, a) in arrays.iter().enumerate() {
        let al = len_of(a);
        if al > 0 {
            heap.push(H {
                v: get(a, 0),
                c: ci,
                i: 0,
            });
        }
    }

    while let Some(h) = heap.pop() {
        let c = h.c;
        let a = &arrays[c];
        let s = h.i;
        let mut e = s + 1;
        let thr = heap.peek().map(|x| x.v);
        let al = len_of(a);
        if let Some(t) = thr {
            while e < al && get(a, e) <= t {
                e += 1;
            }
        } else {
            e = al;
        }
        emit(c, s, e - s);
        if e < al {
            heap.push(H {
                v: get(a, e),
                c,
                i: e,
            });
        }
    }
}

/// Reverse (descending) k-way merge over sorted per-chunk arrays, coalescing runs.
pub fn kmerge_coalesced_rev<T, A, FLen, FGet, FEmit>(
    arrays: &[A],
    mut len_of: FLen,
    mut get: FGet,
    mut emit: FEmit,
) where
    T: Ord + Copy,
    FLen: FnMut(&A) -> usize,
    FGet: FnMut(&A, usize) -> T,
    FEmit: FnMut(usize, usize, usize), // (chunk_idx, start, len) but start..start+len iterates descending via get
{
    #[derive(Clone, Copy, Debug)]
    struct H<T> {
        v: T,
        c: usize,
        i: usize,
    }
    impl<T: Ord> PartialEq for H<T> {
        fn eq(&self, o: &Self) -> bool {
            self.v == o.v && self.c == o.c && self.i == o.i
        }
    }
    impl<T: Ord> Eq for H<T> {}
    impl<T: Ord> PartialOrd for H<T> {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    // Max-heap by value (natural ordering)
    impl<T: Ord> Ord for H<T> {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.v.cmp(&o.v).then_with(|| self.c.cmp(&o.c))
        }
    }

    let mut heap: BinaryHeap<H<T>> = BinaryHeap::new();
    for (ci, a) in arrays.iter().enumerate() {
        let al = len_of(a);
        if al > 0 {
            let idx = al - 1;
            heap.push(H {
                v: get(a, idx),
                c: ci,
                i: idx,
            });
        }
    }

    while let Some(h) = heap.pop() {
        let c = h.c;
        let a = &arrays[c];
        let e = h.i; // inclusive end
        let mut s = e; // inclusive start, will decrease
        let thr = heap.peek().map(|x| x.v);
        if let Some(t) = thr {
            while s > 0 {
                let p = s - 1;
                if get(a, p) >= t {
                    s = p;
                } else {
                    break;
                }
            }
        } else {
            // drain remaining
            s = 0;
        }
        // Emit as (start,len) using ascending indices; caller can iterate descending within the run
        emit(c, s, e - s + 1);
        if s > 0 {
            let next = s - 1;
            heap.push(H {
                v: get(a, next),
                c,
                i: next,
            });
        }
    }
}
