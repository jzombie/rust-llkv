use crate::layout::KeyLayout;

pub fn slice_key_by_layout<'a>(bytes: &'a [u8], layout: &'a KeyLayout, i: usize) -> &'a [u8] {
    match layout {
        KeyLayout::FixedWidth { width } => {
            let w = *width as usize;
            let a = i * w;
            let b = a + w;
            &bytes[a..b]
        }
        KeyLayout::Variable { key_offsets } => {
            let a = key_offsets[i] as usize;
            let b = key_offsets[i + 1] as usize;
            &bytes[a..b]
        }
    }
}
