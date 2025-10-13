# LLKV Aggregate

**Work in Progress**

`llkv-aggregate` provides aggregate computation functions for the [LLKV](https://github.com/jzombie/rust-llkv) toolkit.

## Purpose

- Implement standard SQL aggregate functions (SUM, COUNT, AVG, MIN, MAX, etc.).
- Provide efficient streaming aggregation over Arrow data structures.
- Support grouped aggregations for query execution via [`llkv-executor`](../llkv-executor/).

## Design Notes

- The crate integrates with Arrow's columnar format for efficient computation.
- Aggregates are designed to work with [`llkv-executor`](../llkv-executor/) for pipelined evaluation in SELECT queries.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
