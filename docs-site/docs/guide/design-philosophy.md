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

For v0.2.0, the publication surface is centered on:

- `examples/` for quick scripts,
- `notebooks/` for concise narrative workflows,
- this docs site for conceptual orientation.

The `labs/` folder remains internal and withheld from publication.
