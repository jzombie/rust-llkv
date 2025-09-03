use llkv_btree::codecs::{BigEndianIdCodec, BigEndianKeyCodec};
use llkv_btree::define_mem_pager;
use llkv_btree::iter::ScanOpts;
use llkv_btree::pager::SharedPager;
use llkv_btree::prelude::*;
use llkv_btree::shared_bplus_tree::SharedBPlusTree;

define_mem_pager! {
    /// In-memory pager with u64 page IDs.
    name: MemPager64,
    id: u64,
    default_page_size: 256
}

// ---------- types ----------
type KC = BigEndianKeyCodec<u64>;
type IC = BigEndianIdCodec<u64>;
type SP = SharedPager<MemPager64>;
type SharedTree = SharedBPlusTree<SP, KC, IC>;

fn main() {
    // Wrap base storage with the shared adapter and build a shared B+Tree.
    let base = MemPager64::new(4096);
    let pager = SharedPager::new(base);
    let remote: SharedTree = SharedTree::create_empty(pager, None).expect("create_empty");

    // Seed 1..=10 with k.to_be_bytes()
    let items: Vec<(u64, Vec<u8>)> = (1u64..=10u64)
        .map(|k| (k, k.to_be_bytes().to_vec()))
        .collect();
    let owned: Vec<(u64, &[u8])> = items.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    remote.insert_many(&owned).expect("insert_many");

    // --- Three concurrent snapshot scans ---
    // 1) full forward scan (no bounds)
    let rx_all_fwd = remote.start_stream_with_opts(ScanOpts::<'static, KC>::forward());
    // 2) forward "range" [3, 7]  (implemented by consumer-side filter)
    let rx_rng_fwd = remote.start_stream_with_opts(ScanOpts::<'static, KC>::forward());
    // 3) reverse "range" (-inf, 6)  (implemented by consumer-side filter)
    let rx_rng_rev = remote.start_stream_with_opts(ScanOpts::<'static, KC>::reverse());

    // We’ll interleave events from all three streams in a single consumer thread.
    let mut rx_all_fwd = rx_all_fwd;
    let mut rx_rng_fwd = rx_rng_fwd;
    let mut rx_rng_rev = rx_rng_rev;

    // A receiver that never yields; used to disable a select arm once closed.
    let dead = crossbeam_channel::never::<(u64, Vec<u8>)>();

    let mut open = 3;
    while open > 0 {
        crossbeam_channel::select! {
            recv(rx_all_fwd) -> res => {
                match res {
                    Ok((k, v)) => {
                        // Verify payload == k.to_be_bytes()
                        assert_eq!(v.len(), 8, "value length must be 8 for key {k}");
                        let got = u64::from_be_bytes(v.as_slice().try_into().expect("len 8"));
                        assert_eq!(got, k, "decoded value mismatch for key {k}");
                        println!("all_fwd {k} -> {got}");
                    }
                    Err(_) => { open -= 1; rx_all_fwd = dead.clone(); } // disable this arm
                }
            },
            recv(rx_rng_fwd) -> res => {
                match res {
                    Ok((k, v)) => {
                        if !(3..=7).contains(&k) { continue; } // consumer-side range: [3,7]
                        assert_eq!(v.len(), 8, "value length must be 8 for key {k}");
                        let got = u64::from_be_bytes(v.as_slice().try_into().expect("len 8"));
                        assert_eq!(got, k, "decoded value mismatch for key {k}");
                        println!("rng_fwd[3..=7] {k} -> {got}");
                    }
                    Err(_) => { open -= 1; rx_rng_fwd = dead.clone(); } // disable this arm
                }
            },
            recv(rx_rng_rev) -> res => {
                match res {
                    Ok((k, v)) => {
                        if k >= 6 { continue; } // consumer-side range: (-inf, 6)
                        assert_eq!(v.len(), 8, "value length must be 8 for key {k}");
                        let got = u64::from_be_bytes(v.as_slice().try_into().expect("len 8"));
                        assert_eq!(got, k, "decoded value mismatch for key {k}");
                        println!("rng_rev[..6) {k} -> {got}");
                    }
                    Err(_) => { open -= 1; rx_rng_rev = dead.clone(); } // disable this arm
                }
            },
            default => std::thread::yield_now(),
        }
    }

    println!("done");
}
