use std::path::Path;

use llkv_result::Result;
use llkv_storage::pager::mem_pager::MemPager;

mod pager_harness;
use pager_harness::{Persistence, run_crud_roundtrip, run_reopen_behavior};

fn make_mem(_path: &Path) -> Result<MemPager> {
    // MemPager ignores paths and is purely in-memory.
    Ok(MemPager::new())
}

#[test]
fn mem_crud_roundtrip() {
    run_crud_roundtrip::<MemPager, _>(make_mem);
}

#[test]
fn mem_reopen_behavior_is_ephemeral() {
    run_reopen_behavior::<MemPager, _>(make_mem, Persistence::Ephemeral);
}
