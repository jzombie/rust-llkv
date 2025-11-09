# LLKV Executor

[![made-with-rust][rust-logo]][rust-src-page]
[![rust-docs][rust-docs-badge]][rust-docs-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

`llkv-executor` evaluates `SELECT` plans for the [LLKV](../) stack. It produces streaming Arrow `RecordBatch`es, coordinating with table scans, joins, and aggregation primitives while remaining oblivious to transaction orchestration.

## Responsibilities

- Execute logical `SelectPlan`s emitted by [`llkv-plan`](../llkv-plan/) and dispatched by [`llkv-runtime`](../llkv-runtime/).
- Stream `RecordBatch` results through callbacks so callers can consume output incrementally.
- Evaluate projection, filtering, scalar expressions, aggregation, and join operations over Arrow arrays.
- Integrate with the table layer for predicate pushdown and efficient column gathering.

## Execution Pipeline

- `TableExecutor` builds a `RowStreamBuilder`, fetching projected columns via [`llkv-table`](../llkv-table/) and combining them with computed expressions.
- Filter evaluation uses vectorized expression kernels to avoid per-row branching; scalar subqueries bind correlated values supplied by the planner.
- Aggregations delegate to [`llkv-aggregate`](../llkv-aggregate/) for grouped and streaming operations.
- Joins rely on [`llkv-join`](../llkv-join/) implementations such as hash joins optimized for primitive key pairs.
- Results are emitted as soon as a batch is available, preventing full result materialization.

## Interaction with the Runtime

- Invoked exclusively by [`llkv-runtime`](../llkv-runtime/), which provides transaction context and MVCC-filtered row streams.
- Receives already-validated plans; error handling focuses on execution-time failures (e.g., overflow, unsupported expression forms).
- Remains agnostic to DDL/DML responsibilities so it can focus on efficient batch processing.

## License

Licensed under the [Apache-2.0 License](../LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[rust-docs-page]: https://docs.rs/llkv-executor
[rust-docs-badge]: https://img.shields.io/docsrs/llkv-executor

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
