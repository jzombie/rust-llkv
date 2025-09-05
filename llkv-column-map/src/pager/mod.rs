use crate::types::PhysicalKey;
use bitcode::{Decode, Encode};

pub trait Pager {
    // -------- allocation (batched only) --------
    fn alloc_many(&mut self, n: usize) -> Vec<PhysicalKey>;

    // TODO: Combine raw & typed requests into a single batch operation for less physical storage calls

    // -------- raw bytes (batched only) --------
    fn batch_put_raw(&mut self, items: &[(PhysicalKey, Vec<u8>)]);

    fn batch_get_raw<'a>(&'a self, keys: &[PhysicalKey]) -> Vec<&'a [u8]>;

    // -------- typed (batched only) --------
    fn batch_put_typed<T: Encode>(&mut self, items: &[(PhysicalKey, T)]);

    fn batch_get_typed<T>(&self, keys: &[PhysicalKey]) -> Vec<T>
    where
        for<'a> T: Decode<'a>;
}
