# Stack Flattening Refactor Plan

This document tracks the work required to remove recursive call stacks from the planner and related components.

## Goals

- Eliminate deep recursion in predicate evaluation, SQL translation, and numeric expression handling.
- Ensure extremely large sqllogictest workloads run without increasing thread stack sizes.
- Keep fast-path fusion optimizations intact while moving to iterative execution.

## Plan of Record

1. **Compile predicates into instruction buffers.**
   - When the planner receives an `Expr`, convert it once into a `Vec<EvalOp>` in post-order (think bytecode).
   - Each instruction holds opcode plus operand/child span metadata. Interpret the program with a small manual stack (`Vec<Vec<RowId>>`) so predicate evaluation is purely iterative.
   - Mirror the approach for domain collection using a distinct instruction stream.

2. **Iterative SQL translation before the planner.**
   - Replace recursive walkers such as `translate_condition_with_context`, `flatten_and`, and `flatten_or` with an explicit work stack.
   - As we consume the sqlparser AST, emit the corresponding planner instructions directly, avoiding huge intermediate ASTs.

3. **Iterative scalar evaluation.**
   - Teach `NumericKernels::simplify`, `collect_fields`, and `evaluate_value` to operate on the new instruction form (or at least on an explicit evaluation stack).
   - For row-level evaluation use reverse-polish evaluation: push literals/columns, pop/push for binary operations, never recurse.

4. **Cache the compiled programs.**
   - Because compilation is pure, cache instruction buffers per `Expr` so repeated scans reuse them.
   - Preserve existing fusion fast paths by compiling dedicated opcodes that short-circuit to the fused runtime when their shape is detected.

## Notes

- Ownership of these changes likely spans the `llkv-plan` and `llkv-table` crates, since the planner lives in `llkv-table` but expression construction flows from `llkv-plan` and `llkv-sql`.
- Each milestone should be validated with the sqllogictest suite without relying on enlarged stack sizes.
- Instrumentation (e.g., call depth counters) can help verify that recursive pathways are eliminated as work progresses.
