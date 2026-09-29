# Design Philosophy

## Pre-1.0 exploratory mode

`qpl` is intentionally in pre-1.0 mode:

- API churn is acceptable when it improves clarity.
- Labs and educational workflows drive package design.
- Backward compatibility is not guaranteed yet.

## Engineering invariants

Even in exploratory mode, a few constraints stay fixed:

- deterministic stochastic workflows (explicit seeds),
- smoke-friendly notebooks and examples,
- always-green checks (`ruff` and `pytest`),
- clean working tree at slice boundaries.

## Public learning surface

The publication surface is centered on:

- `src/qpl/cases` and `tests/` for the checked claims,
- `docs/CURRICULUM.md` and `docs/notes/` for derivations and measurements,
- `examples/` for quick scripts,
- `notebooks/` for concise narrative workflows,
- this docs site for conceptual orientation.

The `labs/` folder is private by design (gitignored) and is never published. It drives package
design as a textbook-driven workbook set; the public cargo is `src/`, `tests/`, the `cases` layer,
and short derivation notes.
