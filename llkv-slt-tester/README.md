# LLKV SLT Tester

`llkv-slt-tester` provides a harness and library for running [sqllogictest](https://sqlite.org/sqllogictest/doc/trunk/about.wiki) (`.slt`) suites against the LLKV SQL engine. It treats SLT files like idiomatic Rust tests and also supports pointer files (`.slturl`) that reference remote test content.

This crate is not intended for direct standalone use.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
