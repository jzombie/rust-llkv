//! Shared test harness for all Pager implementations.
//!
//! Verifies for any Pager:
//! - CRUD roundtrip: after `batch_put`, keys are returned by `batch_get`;
//!   after delete, they're gone.
//! - Reopen behavior: if the pager is persistent, data survives reopen;
//!   if it's ephemeral (in-memory), reopen yields an empty view.
//! - (Persistent only) Allocator continuity: after reopen, newly
//!   allocated keys must start strictly after the previous max key.

use std::path::{Path, PathBuf};

use llkv_result::Result;
use llkv_storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use llkv_storage::types::PhysicalKey;
use tempfile::TempDir;

/// Whether a pager should persist across reopen.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[allow(dead_code)]
pub enum Persistence {
    Persistent,
    Ephemeral,
}

/// Run a CRUD roundtrip on any `Pager`.
pub fn run_crud_roundtrip<P, F>(make: F)
where
    P: Pager,
    F: FnOnce(&Path) -> Result<P>,
{
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("pager.db");
    let pager = make(&path).expect("open pager");

    // Allocate three keys and write three distinct payloads.
    let ks: Vec<PhysicalKey> = pager.alloc_many(3).expect("alloc_many");
    assert_eq!(ks.len(), 3);

    pager
        .batch_put(&[
            BatchPut::Raw {
                key: ks[0],
                bytes: b"alpha".to_vec(),
            },
            BatchPut::Raw {
                key: ks[1],
                bytes: b"bravo-123".to_vec(),
            },
            BatchPut::Raw {
                key: ks[2],
                bytes: vec![0u8; 1024],
            },
        ])
        .expect("batch_put");

    // Read them back.
    let got = pager
        .batch_get(&[
            BatchGet::Raw { key: ks[0] },
            BatchGet::Raw { key: ks[1] },
            BatchGet::Raw { key: ks[2] },
        ])
        .expect("batch_get");

    let found = got
        .iter()
        .filter(|r| matches!(r, GetResult::Raw { .. }))
        .count();
    assert_eq!(found, 3, "all inserted keys should be found");

    // Overwrite ks[1] then ensure it’s still found.
    pager
        .batch_put(&[BatchPut::Raw {
            key: ks[1],
            bytes: b"new".to_vec(),
        }])
        .expect("overwrite");

    let got2 = pager
        .batch_get(&[BatchGet::Raw { key: ks[1] }])
        .expect("get");
    let found2 = got2
        .iter()
        .filter(|r| matches!(r, GetResult::Raw { .. }))
        .count();
    assert_eq!(found2, 1, "overwritten key should still be found");

    // Delete ks[0] and ks[2], keep ks[1].
    pager.free_many(&[ks[0], ks[2]]).expect("free_many");

    let after_del = pager
        .batch_get(&[
            BatchGet::Raw { key: ks[0] },
            BatchGet::Raw { key: ks[1] },
            BatchGet::Raw { key: ks[2] },
        ])
        .expect("get after delete");

    let found_after = after_del
        .iter()
        .filter(|r| matches!(r, GetResult::Raw { .. }))
        .count();
    let missing_after = after_del
        .iter()
        .filter(|r| matches!(r, GetResult::Missing { .. }))
        .count();

    assert_eq!(found_after, 1, "only one key should remain after delete");
    assert_eq!(missing_after, 2, "two keys should be missing after delete");
}

/// Run reopen behavior checks on any `Pager`.
///
/// - If `persistence == Persistent`, data must survive reopen.
/// - If `persistence == Ephemeral`, reopen must *not* see prior data.
pub fn run_reopen_behavior<P, Make>(make: Make, persistence: Persistence)
where
    P: Pager,
    Make: Fn(&Path) -> Result<P>,
{
    let tmp = TempDir::new().expect("tempdir");
    let path: PathBuf = tmp.path().join("pager.db");

    // First handle: insert two keys.
    let pager1 = make(&path).expect("open pager");
    let ks = pager1.alloc_many(2).expect("alloc");
    assert_eq!(ks.len(), 2);
    pager1
        .batch_put(&[
            BatchPut::Raw {
                key: ks[0],
                bytes: b"a".to_vec(),
            },
            BatchPut::Raw {
                key: ks[1],
                bytes: b"bb".to_vec(),
            },
        ])
        .expect("put");
    drop(pager1);

    // Reopen and check according to capability.
    let pager2 = make(&path).expect("reopen pager");

    let got = pager2
        .batch_get(&[BatchGet::Raw { key: ks[0] }, BatchGet::Raw { key: ks[1] }])
        .expect("get after reopen");

    let found = got
        .iter()
        .filter(|r| matches!(r, GetResult::Raw { .. }))
        .count();
    let missing = got
        .iter()
        .filter(|r| matches!(r, GetResult::Missing { .. }))
        .count();

    match persistence {
        Persistence::Persistent => {
            assert_eq!(found, 2, "persistent pager should retain data after reopen");
            assert_eq!(
                missing, 0,
                "no keys should be missing in persistent pager after reopen"
            );
        }
        Persistence::Ephemeral => {
            assert_eq!(
                found, 0,
                "mem/ephemeral pager should NOT retain data after reopen"
            );
            assert_eq!(
                missing, 2,
                "all keys should be missing after reopen on ephemeral pager"
            );
        }
    }
}

/// Persistent-only: allocator must continue from max key after reopen.
#[allow(dead_code)] // Each test file is its own crate and this will only work for durable storage tests
pub fn run_allocator_advances_after_reopen<P, Make>(make: Make)
where
    P: Pager,
    Make: Fn(&Path) -> Result<P>,
{
    let tmp = TempDir::new().expect("tempdir");
    let path: PathBuf = tmp.path().join("pager.db");

    // Session 1: allocate and write some keys; record max.
    let pager1 = make(&path).expect("open pager #1");
    let ks1 = pager1.alloc_many(3).expect("alloc #1");
    let max1 = *ks1.iter().max().unwrap();
    pager1
        .batch_put(&[
            BatchPut::Raw {
                key: ks1[0],
                bytes: b"A".to_vec(),
            },
            BatchPut::Raw {
                key: ks1[1],
                bytes: b"BB".to_vec(),
            },
            BatchPut::Raw {
                key: ks1[2],
                bytes: b"CCC".to_vec(),
            },
        ])
        .expect("put #1");
    drop(pager1);

    // Session 2: reopen and allocate again; first new key must be > max1.
    let pager2 = make(&path).expect("open pager #2");
    let ks2 = pager2.alloc_many(2).expect("alloc #2");
    assert!(
        ks2[0] > max1,
        "allocator did not advance after reopen: new {} <= old max {}",
        ks2[0],
        max1
    );
}
