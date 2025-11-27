# LLKV Plan

`llkv-plan` defines the typed plan surface that bridges parsed SQL and LLKV [runtime](../llkv-runtime/) execution. It organizes statement plans, expression programs, and subquery metadata so lower layers can execute without re-parsing SQL.

This crate is not intended for direct standalone use.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
