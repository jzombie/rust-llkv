# Testing Guide

LLKV relies on a layered test strategy so regressions are caught where they originate. This document explains the required workflows for running the suite, with extra detail on the SQL Logic Tests (SLT) that gate SQL compatibility.

## 1. Run the Workspace Suite

```zsh
cargo test --workspace
```

- Run from the repository root so shared fixtures resolve correctly.
- Do not truncate or pipe away output; failures often include table diffs and SQL plans that are needed during review.
- Use feature flags sparingly. If a scenario only reproduces with a feature enabled, describe the flag combination in your commit message and in the associated test.

## 2. Targeted Crate and Doc Tests

- Every crate can be tested independently via `cargo test -p <crate-name>` which runs unit, integration, and doc tests.
- For doc comment examples, run `cargo test -p <crate-name> --doc` before committing. The `comment-style-guide` requires runnable snippets; this check guarantees they still compile.
- When iterating on a particular integration test, use `cargo test -p <crate-name> --test <file> -- --ignored <filter>` so unrelated trials do not mask failures.

## 3. SQL Logic Tests (SLT)

SQL support in LLKV is test-driven using upstream suites. The `llkv-slt-tester` crate wraps [sqllogictest](https://sqlite.org/sqllogictest/doc/trunk/about.wiki) files and curated DuckDB differential tests. Keeping these tests green is the highest priority for SQL work—changes that regress SLT coverage or performance are rejected until fixed.

### 3.1 Harness Layout

- SLT assets live under `llkv-slt-tester/tests/slt`. Both `.slt` files and `.slturl` pointer files are supported; pointer files let us reference external SLT scripts without copying them into the repo.
- The harness is implemented in `llkv-slt-tester/tests/slt_harness.rs` and delegates to `llkv_slt_tester::run_slt_harness_with_args`. It mirrors `cargo test` semantics via `libtest-mimic`, so the usual flags (`--ignored`, `--include`, `--exact`, `--list`) behave as expected.
- Statistics can be enabled by setting `LLKV_SLT_STATS=1`, and the last normalized failing script is persisted to `target/last_failed_slt.tmp` for debugging.

### 3.2 Running the Suite

```zsh
# Discover available trials
cargo test -p llkv-slt-tester --test slt_harness -- --list

# Focus on one script (pattern matches file names relative to tests/slt)
cargo test -p llkv-slt-tester --test slt_harness -- --include "generic/select/basic.slt"

# Collect per-query stats for performance tracking
LLKV_SLT_STATS=1 cargo test -p llkv-slt-tester --test slt_harness
```

- When a test fails, inspect `target/last_failed_slt.tmp` to see the normalized script that produced the error.
- Use `--ignored` to opt into slow or flaky upstream cases that we still track but exclude from the default suite.
- Keep an eye on runtime when modifying the planner or executor. If a patch degrades SLT timing, add benchmarks or notes in the PR describing the mitigation plan before landing.

### 3.3 Running SLT via the REPL

For interactive debugging, pipe SQL into the `llkv` binary or point it at an SLT directory:

```zsh
# Execute a standalone SLT file with the single-threaded runtime
cargo run -p llkv -- --slt path/to/tests/slt/basic/select.slt

# Feed SQL statements directly (handy for shrinking failing cases)
cargo run -p llkv -- <<'SQL'
CREATE TABLE t(col0 INTEGER);
INSERT INTO t VALUES (1), (2);
SELECT * FROM t ORDER BY col0 DESC;
SQL
```

## 4. Guarding Performance and Accuracy

- Every SQL-facing change must keep all SLT tests passing. When adding new syntax or semantics, land the SLT coverage in the same patch (red/green TDD).
- Record any new external `.slturl` sources inside `llkv-slt-tester/tests/slt` so reviewers can trace provenance.
- If you need to temporarily skip an upstream test, document the reason in-line with a TODO referencing a follow-up issue, and link that issue in your PR.
- Performance regressions spotted during SLT runs must be addressed immediately—either by optimizing before merge or by introducing a benchmark in `llkv-slt-tester/benches` that demonstrates parity with the previous behavior.

Following these steps keeps the SQL stack honest and ensures our upstream-derived test corpus remains a reliable safety net for both correctness and throughput.
