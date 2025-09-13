use super::*;

/// Fixed-length f32 array codec (length N). Encodes as N big-endian u32 IEEE-754 words.
///
/// TODO: Experimental. If adopted, consider a generic fixed-width vector trait.
pub struct F32x<const N: usize>;

/// Alias: clearer name if used externally.
pub type Float32Array<const N: usize> = F32x<N>;

impl<const N: usize> Codec for F32x<N> {
    const WIDTH: usize = N * 4;
    type Borrowed<'a> = &'a [f32]
    where
        Self: 'a;
    type Owned = Vec<f32>;

    #[inline]
    fn encode_into(dst: &mut Vec<u8>, v: Self::Borrowed<'_>) -> Result<(), EncodeError> {
        assert_eq!(v.len(), N, "F32x encode length mismatch: expected {}, got {}", N, v.len());
        dst.reserve(N * 4);
        for &x in v {
            dst.extend_from_slice(&x.to_bits().to_be_bytes());
        }
        Ok(())
    }

    #[inline]
    fn decode(src: &[u8]) -> Result<Self::Owned, DecodeError> {
        if src.len() < N * 4 { return Err(DecodeError::NotEnoughData); }
        let mut out = vec![0f32; N];
        f32x_decode_into::<N>(&mut out, src)?;
        Ok(out)
    }

    #[inline]
    fn decode_many_into(_dst: &mut [Vec<f32>], _src: &[u8]) -> Result<(), DecodeError> {
        panic!("F32x::decode_many_into is not supported; use f32x_decode_many_into");
    }
}

/// Decode exactly N f32 from `src` into `dst`.
#[inline]
pub fn f32x_decode_into<const N: usize>(dst: &mut [f32], src: &[u8]) -> Result<(), DecodeError> {
    debug_assert_eq!(dst.len(), N);
    if src.len() < N * 4 { return Err(DecodeError::NotEnoughData); }
    for i in 0..N {
        let a = i * 4;
        let mut b4 = [0u8; 4];
        b4.copy_from_slice(&src[a..a + 4]);
        dst[i] = f32::from_bits(u32::from_be_bytes(b4));
    }
    Ok(())
}

/// Decode many fixed-length f32 arrays from a concatenated buffer.
/// `rows` is the number of arrays, `N` is the length per row.
/// `dst` must have length == rows * N.
#[inline]
pub fn f32x_decode_many_into<const N: usize>(dst: &mut [f32], src: &[u8], rows: usize) -> Result<(), DecodeError> {
    if dst.len() != rows * N { return Err(DecodeError::InvalidFormat); }
    if src.len() < rows * N * 4 { return Err(DecodeError::NotEnoughData); }
    for r in 0..rows {
        let row_bytes = &src[(r * N * 4)..((r + 1) * N * 4)];
        for c in 0..N {
            let a = c * 4;
            let mut b4 = [0u8; 4];
            b4.copy_from_slice(&row_bytes[a..a + 4]);
            dst[r * N + c] = f32::from_bits(u32::from_be_bytes(b4));
        }
    }
    Ok(())
}

/// Parallel decode many fixed-length f32 arrays using Rayon.
/// Splits by rows and decodes each row independently.
pub fn f32x_decode_many_into_par<const N: usize>(dst: &mut [f32], src: &[u8], rows: usize) -> Result<(), DecodeError> {
    if dst.len() != rows * N { return Err(DecodeError::InvalidFormat); }
    if src.len() < rows * N * 4 { return Err(DecodeError::NotEnoughData); }
    use rayon::prelude::*;
    dst.par_chunks_mut(N)
        .zip(src.par_chunks_exact(N * 4))
        .try_for_each(|(row_dst, row_src)| f32x_decode_into::<N>(row_dst, row_src))
}
