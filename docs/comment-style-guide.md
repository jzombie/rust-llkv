# Comment Style Guide

Our goal is to keep documentation blocks brief, accurate, and consistent across crates. Reference this guide whenever you add or update API comments. Treat it as an overlay on top of idiomatic Rust documentation practices.

Large modules and crates must start with a module-level doc comment (`//!`) that summarizes what the component does, why it exists, and how it is used. Mirror the consistent structure shown in the transaction crate header: summary, key concepts, reserved values or constants, visibility rules, and architecture roles. Tailor sections to the module’s needs while keeping the narrative concise.

## Core Principles
- Write comments only when they add value—prefer expressive names over redundant narration.
- Keep tone direct and technical; avoid marketing phrasing or speculative language.
- When documenting one parameter, document them all; otherwise omit the parameter section entirely.
- Favor short sentences and bulleted lists for clarity.
- Large modules must open with a clear summary paragraph that orients readers before diving into details.

## Structuring API Docs
1. **Summary line:** One sentence that states what the item does.
2. **Details:** Optional paragraphs describing invariants, concurrency expectations, or performance characteristics.
3. **Inputs/Outputs:** Use `# Arguments`, `# Returns`, `# Errors`, and `# Safety` headings only when needed. When you introduce one heading, document every item underneath it. Avoid partially documented argument lists; either cover them all or omit the section.
4. **Examples:** Include only runnable snippets. Avoid `ignore` or non-Rust code fences.

## Linking and References
- Use Rust intra-doc links (`[Type]`, `[module::Item]`, ``[`crate::path::Item`]``) so references stay checked by the compiler.
- When linking across crates, reference the crate-qualified path (for example ``[`llkv_storage::ColumnStore`]``) to make relationships explicit and keep rustdoc navigation working.
- Prefer mentioning related crates or modules directly in the summary or architecture sections so readers can see inter-crate responsibilities at a glance.
- Keep structural docs (such as `docs/crate-linkage.md`) updated when dependencies change, and sync comment links with those updates.

## Unsafe Code
- Document `unsafe` sections with inline comments of the form `// SAFETY: ...`, explaining the invariants that make the block sound.
- Keep the safety rationale specific and actionable (e.g., memory aliasing requirements, threading guarantees).
- For public `unsafe` APIs, still include a `# Safety` rustdoc section that mirrors the inline comment and informs callers of their obligations.

## Examples Policy
- Examples must compile and pass `cargo test --doc`.
- Prefer minimal examples that demonstrate the primary use. If setup is heavy, link to integration tests instead.
- Remove examples that can no longer be verified or are redundant with tests.

## Maintaining Consistency
- When touching a file, skim nearby comments and align them with this guide. Treat outdated comments as lint violations to fix on sight.
- Keep terminology synchronized with the high-level crate overview in `docs/crate-linkage.md`.
- Use consistent headings and casing (`# Arguments`, `# Returns`, `# Errors`, `# Safety`).
- Leave existing TODO/FIXME-style notes in place unless you can resolve them immediately, and never promote those markers into `///` doc comments.

## Process Checklist
- [ ] Verified that summary line is accurate after your change.
- [ ] Confirmed parameters/returns/errors sections are complete or intentionally omitted.
- [ ] Ran `cargo test --doc` if examples were added or modified.
- [ ] Updated neighboring comments that drift from the style.
- [ ] Removed stale examples or prose that no longer reflects current behavior.

Following this guide keeps future refactors lean and makes rustdoc outputs dependable for both internal and external consumers.
