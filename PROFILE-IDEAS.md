# Profiling

**Measure compile phases of the bench harness**

```sh
cargo +nightly rustc -p llkv-join --bench join_bench --release -- -Ztime-passes
```

That will print a phase breakdown for the whole bench harness.

**Self-profile with events**

```sh
cargo +nightly rustc -p llkv-join --bench join_bench --release -- \
  -Z self-profile -Z self-profile-events=default -Z self-profile-output=prof
```

Then run `flamegraph` / `summarize` on the prefix in `prof/`.

**Try compiling with Criterion**

Make a minimal `benches/manual_bench.rs` that just calls your join functions and prints elapsed time.
Build that with `cargo build --release --benches`. If it compiles fast, Criterion is the main culprit.

**llvm-lines**

https://github.com/dtolnay/cargo-llvm-lines

```sh
cargo llvm-lines --example reimplement_benches --release
```

Current debug for https://github.com/jzombie/rust-llkv/issues/63:

```sh
CARGO_PROFILE_RELEASE_LTO=fat cargo llvm-lines -p llkv-join --example reimplement_benches --release > debug-lines2.txt
```

**Related**

https://github.com/apache/datafusion/issues/13814
