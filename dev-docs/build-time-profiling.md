# Build Profiling

> Note: this document is a short reference for compile-time performance
> measurement and profiling of this repository. It collects commands and
> lightweight examples useful for measuring rustc phases, LTO/IR size
> (cargo-llvm-lines), and obtaining rustc self-profiles. Use the quick
> `compile_deps` example to compile dependencies fast, and run the heavier
> llvm-lines command when you need full IR-size/LTO analysis.

## Measure compile phases of the bench harness

```sh
cargo +nightly rustc -p llkv-join --bench join_bench --release -- -Ztime-passes
```

That will print a phase breakdown for the whole bench harness.

## Self-profile with events

```sh
cargo +nightly rustc -p llkv-join --bench join_bench --release -- \
  -Z self-profile -Z self-profile-events=default -Z self-profile-output=prof
```

Then run `flamegraph` / `summarize` on the prefix in `prof/`.

## Try compiling with Criterion

Make a minimal `benches/manual_bench.rs` that just calls your join functions and prints elapsed time.
Build that with `cargo build --release --benches`. If it compiles fast, Criterion is the main culprit.

## llvm-lines

https://github.com/dtolnay/cargo-llvm-lines

If you only want to compile dependencies quickly (no heavy bench runs), use the small example that exercises the public API:

```sh
# quick compile of dependencies for llkv-join
cargo build --example compile_deps -p llkv-join --release
```

For the full llvm-lines analysis (heavier, includes LTO), run the example that compiles dependencies and records LLVM lines:

```sh
# heavy llvm-lines run (LTO) using the compile_deps example
CARGO_PROFILE_RELEASE_LTO=fat cargo llvm-lines -p llkv-join --example compile_deps --release | head -n 100
```

## Related

Note: This project does not use DataFusion internally, but various build-time profiling techniques are listed by users in this thread: https://github.com/apache/datafusion/issues/13814
