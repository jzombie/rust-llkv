use crate::types::{ByteWidth, LogicalFieldId, LogicalKeyBytes};

pub struct Put {
    pub field_id: LogicalFieldId,
    pub items: Vec<(LogicalKeyBytes, Vec<u8>)>, // unordered; duplicates allowed (last wins)
}

#[derive(Clone, Copy, Debug)]
pub enum ValueMode {
    Auto,
    ForceFixed(ByteWidth),
    ForceVariable,
}

#[derive(Clone, Debug)]
pub struct AppendOptions {
    pub mode: ValueMode,
    pub segment_max_entries: usize,
    pub segment_max_bytes: usize, // data payload budget per segment
    pub last_write_wins_in_batch: bool,
}

impl Default for AppendOptions {
    fn default() -> Self {
        Self {
            mode: ValueMode::Auto,
            segment_max_entries: 65_536,
            segment_max_bytes: 8 * 1024 * 1024,
            last_write_wins_in_batch: true,
        }
    }
}
