# 2. Adopt Thin Vertical Slices + Golden Path

Date: 2026-02-04

## Status
Accepted

## Context
The project needs to deliver value incrementally while maintaining a "always green" state. Large refactors or "layer-by-layer" builds (e.g. "build the whole perfect Model layer first") lead to integration hell and broken intermediate states. We need a way to add features (like new instruments or engines) safely.

## Decision
We will adopt **Thin Vertical Slices** as the primary unit of work.

A **Thin Vertical Slice** is defined as one user-facing capability that spans:
1.  **Public API**: A minimal surface (import + call).
2.  **Logic**: The implementation in the engine/model.
3.  **Verification**: At least one deterministic test.
4.  **Golden Path**: Inclusion in (or validation by) the `examples/` scripts that are known to work.

**Acceptance Criteria** for a slice:
- It must run deterministically on CI.
- It must not break the existing Golden Path.
- It must be "complete" enough to be used (no half-implemented functions that raise NotImplementedError for the happy path).

## Consequences
- **Tradeoff**: We may duplicate some code initially (e.g. boilerplate in engines) rather than creating the "perfect" abstraction upfront.
- **Benefit**: We can ship usable features (e.g. "PDE Greeks") independently.
- **Process**: Every task starts by defining the slice (API + Test).

## Golden Path Evolution
The Golden Path (`examples/bs_analytic.py` and friends) must remain runnable. New slices should either:
- Be added to an existing example if they fit naturally.
- Create a new, small example script if they are a distinct capability (e.g. `examples/binary_options.py`).
