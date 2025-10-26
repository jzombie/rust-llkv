# LLKV SQL Logic Tester

**Work in Progress**

`llkv-slt-tester` provides a test harness to provide [sqllogictest](https://sqlite.org/sqllogictest/doc/trunk/about.wiki) testing for the [LLKV](../) toolkit.

It treats .slt files like idiomatic Rust tests and can also execute remote tests.

## Overview

This crate provides a small harness and API for running sqllogictest (`.slt`) suites against the LLKV engine. It is intended to make SLT files feel like ordinary Rust test cases: they can be discovered and executed from a test binary, or fetched from remote locations and executed on demand using idiomatic Rust testing commands.

**Key points:**

- The harness source file (`tests/slt_harness.rs`) implements a standalone test runner using `libtest-mimic` so the harness controls discovery and reporting.
- Programmatic API is exposed via `LlkvSltRunner` for embedding SLT execution in other tooling.
- `.slturl` pointer files are supported: a `.slturl` file contains a URL which will be fetched (via `reqwest`) and executed.

## Usage

Run the included test harness:

From the workspace root:

```
cargo test -p llkv-slt-tester --test slt_harness
```

Or from inside the `llkv-slt-tester` crate directory:

```
cargo test --test slt_harness
```

The harness looks for SLT files under the crate `tests/slt` directory by default.

Programmatically you can use the `LlkvSltRunner` API. Examples:

```rust
// Run all .slt files under a directory using the in-memory factory
let runner = llkv_slt_tester::LlkvSltRunner::in_memory();
runner.run_directory("tests/slt")?;

// Run a single .slt or .slturl file
runner.run_file("tests/slt/example.slt")?;
runner.run_file("tests/slt/remote.slturl")?; // reads URL and fetches script
```

The library also exposes helpers to run SLT content from a string or reader, and to choose single-threaded vs multi-threaded runtimes.

## Remote (pointer) tests

Files with the `.slturl` extension are treated as pointers: the harness reads the file contents as a URL, fetches the script over the network, and executes it. Network fetches use `reqwest` (blocking) and therefore require network access at runtime.

## Internals / Implementation notes

- The test harness source file (`tests/slt_harness.rs`) provides a custom `main` that parses `libtest-mimic::Arguments` and calls into `run_slt_harness_with_args` so the crate can control discovery and reporting.
- The public API lives in `src/lib.rs` and delegates parsing and execution to `src/runner.rs`.

## Contributing

If you add new SLT suites, place them under `tests/slt` (or adjust the harness invocation). When adding remote pointer tests, add a `.slturl` file whose content is the URL to fetch.

Please run `cargo test -p llkv-slt-tester --test slt_harness` after changes to verify behavior.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
