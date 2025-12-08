# LLKV SQL Logic Tester

[![made-with-rust][rust-logo]][rust-src-page]
[![rust-docs][rust-docs-badge]][rust-docs-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

`llkv-slt-tester` provides a harness and library for running sqllogictest (`.slt`) suites against the [LLKV](../) SQL engine. It treats SLT files like idiomatic Rust tests and also supports pointer files (`.slturl`) that reference remote test content.

This README consolidates the previous `SLTURL-SUPPORT.md` and test-runner guidance into one place.

## Quick start

Run the included test harness from the workspace root:

```bash
cargo test -p llkv-slt-tester --test slt_harness
```

Or from inside the `llkv-slt-tester` crate directory:

```bash
cargo test --test slt_harness
```

The harness discovers SLT files under `tests/slt` by default.

### Run a specific SLT (harness filter)

To run a single file, use the test-filter argument (quote the path to avoid shell globbing):

```bash
cargo test --package llkv-slt-tester --test slt_harness -- "slt/sqlite/index/in/10/slt_good_0.slturl"
```

This is useful when iterating on failing SLT cases.

## File types: `.slt` vs `.slturl`

- `.slt` — local sqllogictest file containing SLT content in the repo.
- `.slturl` — pointer file containing a single HTTP(S) URL (one line) that references remote SLT content. The harness will fetch this content at test time and execute it as if it were a local `.slt`.

Example `.slturl` file content:

```
https://raw.githubusercontent.com/jzombie/sqlite-sqllogictest-corpus/refs/heads/main/test/index/in/10/slt_good_0.test
```

### Benefits of `.slturl`

- Keeps repository size small by avoiding checked-in large corpora.
- Lets tests be maintained in an authoritative external corpus.
- Tests run identically whether local or fetched remotely.

### Network requirements and CI guidance

- `.slturl` tests require network access during test execution. If your CI environment lacks internet access, mark network-dependent tests `#[ignore]` or run them on a runner with network access.
- Recommended pattern: provide a small smoke test in-tree and run large remote suites only on a designated benchmarking/validation runner with internet.

## Programmatic API (`LlkvSltRunner`)

`LlkvSltRunner` is the public API for programmatic test execution. Common usage patterns:

```rust
// Create an in-memory runner that uses the in-memory pager factory
let runner = llkv_slt_tester::LlkvSltRunner::in_memory();

// Run all .slt/.slturl files under a directory
runner.run_directory("tests/slt")?;

// Run a single file (works for both .slt and .slturl)
runner.run_file("tests/slt/example.slt")?;
runner.run_file("tests/slt/remote.slturl")?; // will fetch the URL

// Run SLT content from a string (useful for unit tests)
runner.run_reader("--slt-from-string--", std::io::Cursor::new(slt_text))?;
```

The runner supports single-threaded or multi-threaded execution and integrates with `libtest-mimic` for discovery when running the harness binary.

## Remote pointer implementation details

When the harness encounters a `.slturl` file it:

1. Reads the single-line URL from the pointer file (whitespace trimmed).
2. Fetches the remote content via HTTP(S) using `reqwest`.
3. Executes the fetched content exactly like a local `.slt` file.

Errors reported include context (source pointer file and URL) and fall into three classes: file read errors, network errors, and content/parse errors.

### Required format for `.slturl`

- A single line containing a valid HTTP(S) URL.
- Whitespace will be trimmed automatically.

## Migration guide (converting local `.slt` to `.slturl`)

To move large SLT files out of the repository:

1. Upload your `.slt` file to a remote host (e.g., GitHub raw URL).
2. Replace the local `.slt` with a `.slturl` file that contains the raw URL.
3. Commit the small `.slturl` pointer file instead of the large `.slt`.

Example:

```bash
# Before: tests/slt/sqlite/index/in/10/slt_good_0.slt (1.2MB)
# After:  tests/slt/sqlite/index/in/10/slt_good_0.slturl (1 line)
```

## Testing and CI

- A dedicated test (`slt_harness`) discovers both `.slt` and `.slturl` files.
- For network-dependent tests provide a separate `--ignored` suite or a CI job with internet access. Example command for ignored network tests:

```bash
cargo test --test slturl_test -- --ignored
```

- The harness supports collection of query statistics via `LLKV_SLT_STATS=1` if you want timings and per-query metrics.

## Internals / Implementation notes (for maintainers)

- Entry point: `tests/slt_harness.rs` builds the harness binary and delegates to `run_slt_harness_with_args` for discovery and execution.
- Public API: `src/lib.rs` exposes `LlkvSltRunner`.
- Runner internals: `src/runner.rs` contains the logic to detect `.slturl` files, fetch remote content, and feed it into the SLT parser/executor.
- Network fetches currently use blocking `reqwest` for simplicity. Consider async fetch or caching if network latency becomes a bottleneck.

## Troubleshooting

- If a remote `.slturl` fails, curl the URL locally to inspect raw content:

```bash
curl -sSf "$(cat tests/slt/path/to/file.slturl)" | less
```

- Normalized or last-failed SLT output is written to `target/last_failed_slt.tmp` when enabled by the harness to aid debugging.

## Contributing

When adding SLT suites, prefer pointer files for large corpora. For local iteration, keep a small subset of representative tests in-tree.

Please run the harness after changes:

```bash
cargo test -p llkv-slt-tester --test slt_harness
```

## License

Licensed under the [Apache-2.0 License](../LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[rust-docs-page]: https://docs.rs/llkv-slt-tester
[rust-docs-badge]: https://img.shields.io/docsrs/llkv-slt-tester

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
