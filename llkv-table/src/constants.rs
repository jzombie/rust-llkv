use crate::types::ColumnInput;
use std::borrow::Cow;

pub const EMPTY_COL: ColumnInput<'static> = Cow::Borrowed(&[]);
