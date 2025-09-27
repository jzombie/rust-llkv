use std::path::Path;

use llkv_result::Result;
use llkv_storage::pager::simd_r_drive_pager::SimdRDrivePager;

mod pager_harness;
use pager_harness::{
    Persistence, run_allocator_advances_after_reopen, run_crud_roundtrip, run_reopen_behavior,
};

fn make_simd(path: &Path) -> Result<SimdRDrivePager> {
    SimdRDrivePager::open(path)
}

#[test]
fn simd_crud_roundtrip() {
    run_crud_roundtrip::<SimdRDrivePager, _>(make_simd);
}

#[test]
fn simd_reopen_behavior_is_persistent() {
    run_reopen_behavior::<SimdRDrivePager, _>(make_simd, Persistence::Persistent);
}

#[test]
fn simd_allocator_advances_after_reopen() {
    run_allocator_advances_after_reopen::<SimdRDrivePager, _>(make_simd);
}
