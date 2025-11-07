# LLKV Query Expression AST

[![made-with-rust][rust-logo]][rust-src-page]
[![CodSpeed Badge][codspeed-badge]][codspeed-page]
[![Ask DeepWiki][deepwiki-badge]][deepwiki-page]

**Work in Progress**

`llkv-expr` defines the expression Abstract Syntax Tree (AST) used across the [LLKV](../) database stack. It provides type-aware, Arrow-native expression structures that decouple logical expressions from concrete evaluation strategies.

This crate intentionally ships without an evaluator or execution engine; it focuses on representation. Downstream crates like [`llkv-table`](../llkv-table/) and [`llkv-executor`](../llkv-executor/) consume these ASTs for compilation and evaluation.

## Purpose and Scope

- Define a shared expression language for logical predicates (`Expr`) and scalar operations (`ScalarExpr`).
- Support generic field identifiers (`Expr<F>`) so callers can use `FieldId`, `String`, or custom types as needed.
- Enable modular testing and independent evolution of planner and executor layers.
- Serve as the foundation for expression compilation into stack-based evaluation programs used by the table layer.

## Expression Types

### Logical Expressions (`Expr`)

_The following is a non-exhaustive list of supported expressions._

Logical expressions include boolean operations and predicates:

- Boolean composition: `And`, `Or`, `Not`
- Comparisons: `Compare` operations with standard relational operators
- Null checks: `IsNull` predicates
- Set membership: `InList` for testing values against a set
- Subquery evaluation: `Exists` for correlated subqueries
- Literal booleans for constant folding

### Scalar Expressions (`ScalarExpr`)

Scalar expressions represent arithmetic and data manipulation:

- Column references and literal values
- Binary arithmetic operations (addition, subtraction, multiplication, division, modulo)
- Type casts and coercion
- Aggregate function calls (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`)
- Null coalescing (`Coalesce`) and conditional logic (`Case`)
- Subquery evaluation for scalar results
- Struct field access (`GetField`)

## Integration with Other Crates

### Planner (`llkv-plan`)

- [`llkv-plan`](../llkv-plan/) uses `llkv-expr` to construct logical query plans.
- `SelectFilter` and other plan structures embed `Expr<String>` for predicates, ensuring the planner remains decoupled from execution details.
- Correlation helpers in the planner manage placeholder assignment for correlated subqueries, which are represented as `ScalarExpr::ScalarSubquery`.

### Executor (`llkv-executor`)

- [`llkv-executor`](../llkv-executor/) evaluates expressions by collecting aggregates, compiling filter programs, and streaming results.
- The executor applies MVCC visibility filters and evaluates `HAVING` clauses using the `Expr` and `ScalarExpr` structures.

### Table Layer (`llkv-table`)

- [`llkv-table`](../llkv-table/) compiles `Expr<FieldId>` into stack-based `EvalProgram` structures for efficient vectorized evaluation.
- Provides domain analysis and affine transformation extraction to optimize range scans and index selection.

## Design Principles

- **Generic field identifiers**: expressions work with any `F` type (e.g., `u32`, `&'static str`, `FieldId`), enabling flexibility across layers.
- **Zero-copy where possible**: expressions use borrowed references to avoid unnecessary allocations during AST construction.
- **Decoupling**: the crate avoids embedding evaluation logic so compilation strategies can be optimized independently in consuming crates.
- **Arrow-native**: expressions map naturally to Arrow data types and columnar operations, supporting efficient vectorized execution.

## License

Licensed under the [Apache-2.0 License](../LICENSE).

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black?&logo=Rust

[codspeed-page]: https://codspeed.io/jzombie/rust-llkv
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json

[deepwiki-page]: https://deepwiki.com/jzombie/rust-llkv
[deepwiki-badge]: https://deepwiki.com/badge.svg
