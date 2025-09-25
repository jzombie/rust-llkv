use xxhash_rust::xxh3::xxh3_64_with_seed;

#[inline]
pub fn hash64(data: &[u8], seed: u64) -> u64 {
    xxh3_64_with_seed(data, seed)
}
