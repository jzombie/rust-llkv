# Agent Operating Guide

Use this guide as the first stop before making changes. It ties together the repo’s required instructions and the key internal docs you must follow.

## Required Reading
- Always read every file under `.github/instructions/` before starting work; treat them as authoritative constraints.
- Refer to `dev-docs/feature-placement-guide.md` for where new code belongs. Follow the placement process and guardrails—no shortcuts or in-memory hacks.
- Follow `dev-docs/comment-style-guide.md` whenever writing or updating comments. Keep summaries concise, document safety clearly, and fix nearby drift while you are there.

## Working Rules
- Apply the `.github/instructions` guidance to your workflow (test commands, no truncated output, and the no-shortcut policy).
- When planning a feature, identify the lowest crate that owns the responsibility, keep dependencies acyclic, and update linkage docs if architecture changes.
- Clean up comment style in touched modules and ensure any examples stay runnable with `cargo test --doc` when added.
