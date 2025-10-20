# Comment Style Guide

Our goal is to keep documentation blocks brief, accurate, and consistent across crates. Reference this guide whenever you add or update API comments.

## Core Principles
- Write comments only when they add value—prefer expressive names over redundant narration.
- Keep tone direct and technical; avoid marketing phrasing or speculative language.
- When documenting one parameter, document them all; otherwise omit the parameter section entirely.
- Favor short sentences and bulleted lists for clarity.

## Structuring API Docs
1. **Summary line:** One sentence that states what the item does.
2. **Details:** Optional paragraphs describing invariants, concurrency expectations, or performance characteristics.
3. **Inputs/Outputs:** Use `# Arguments`, `# Returns`, and `# Errors` headings only when you need to explain behavior not obvious from the signature. When used, cover every parameter or variant under that heading.
4. **Examples:** Include only runnable snippets. Avoid `ignore` or non-Rust code fences.

## Examples Policy
- Examples must compile and pass `cargo test --doc`.
- Prefer minimal examples that demonstrate the primary use. If setup is heavy, link to integration tests instead.
- Remove examples that can no longer be verified or are redundant with tests.

## Maintaining Consistency
- When touching a file, skim nearby comments and align them with this guide. Treat outdated comments as lint violations to fix on sight.
- Keep terminology synchronized with the high-level crate overview in `docs/crate-linkage.md`.
- Use consistent headings and casing (`# Arguments`, `# Returns`, `# Errors`, `# Safety`).

## Process Checklist
- [ ] Verified that summary line is accurate after your change.
- [ ] Confirmed parameters/returns/errors sections are complete or intentionally omitted.
- [ ] Ran `cargo test --doc` if examples were added or modified.
- [ ] Updated neighboring comments that drift from the style.
- [ ] Removed stale examples or prose that no longer reflects current behavior.

Following this guide keeps future refactors lean and makes rustdoc outputs dependable for both internal and external consumers.
