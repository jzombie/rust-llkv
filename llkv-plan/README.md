# llkv-plan

**Work in Progress**

`llkv-plan` is the planner intermediate representation (IR) crate for the [LLKV](https://github.com/jzombie/rust-llkv) toolkit. It provides a stable `PlanGraph` model for expressing optimized query plans, deterministic DOT/JSON serialization for snapshot tests and visualization, and helpers to validate and hydrate plans into runtime executors.

## Features

- Stable [`PlanGraph`] data model with node/edge metadata.
- Validation helpers that guarantee the graph is acyclic and internally
  consistent.
- DOT and JSON serializers for visualization and golden tests.


## License

Licensed under the [Apache-2.0 License](../LICENSE).
