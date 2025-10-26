# Running SLT tests in llkv-slt-tester

This repository includes SQLite-compatible SLT (SQL Logic Test) files under `llkv-sql/tests/slt/` used by the test harness `slt_harness`.

To run the entire SLT harness testsuite:

```bash
cargo test --package llkv-slt-tester --test slt_harness
```

To run a single SLT file with the harness use the harness binary test filter. Example:

```bash
cargo test --package llkv-slt-tester --test slt_harness -- "slt/sqlite/index/in/10/slt_good_0.slturl"
```

Notes:
- The path is matched against the test input identifier used by the harness; quoting the path is recommended to avoid shell globbing.
- Running individual files is useful when iterating on failing SLT tests (e.g., the SQLite compatibility suite).
- You can run multiple files by passing multiple quoted paths in the same command.

- File extensions:
	- Files with the `.slturl` extension are pointer files: they contain a single HTTP(S) URL that points to a remote SLT (SQL Logic Test) file. The harness will fetch the remote SLT content at test time and execute it as if the test were local.
	- Files with the `.slt` extension contain the SLT test content directly in the repository and are executed from the local source file.

If the test harness or test names change, update this file accordingly.
