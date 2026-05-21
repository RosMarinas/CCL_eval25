# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Layout

This is a single-context repo:

- `CONTEXT.md` at the repo root
- `docs/adr/` at the repo root

## Before Exploring

Read these when they exist:

- `CONTEXT.md` for project vocabulary and domain concepts
- `docs/adr/` for architectural decisions that touch the area being changed

If these files do not exist yet, proceed silently. Do not create them just because they are missing. Producer workflows such as `grill-with-docs` can create them later when project terms or decisions need to be captured.

## Vocabulary

When output names a domain concept in an issue title, refactor proposal, hypothesis, or test name, use the term as defined in `CONTEXT.md`.

If the concept is not in the glossary yet, note the gap rather than inventing new project language.

## ADR Conflicts

If a proposed change contradicts an existing ADR, surface the conflict explicitly instead of silently overriding the decision.
