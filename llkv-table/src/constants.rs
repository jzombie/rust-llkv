use crate::types::ColumnInput;
use std::borrow::Cow;

// TODO: Rename to `EMPTY_CELL`? This shouldn't imply that the entire column is empty.
pub const EMPTY_COL: ColumnInput<'static> = Cow::Borrowed(&[]);
