# ADR-0003: Pre-1.0 Lab Authority and API Churn

Status: accepted
Date: 2026-02-19
Supersedes: ADR-0001

Context
- The project is in exploratory pre-1.0 mode.
- Labs are the primary learning and product-shaping mechanism.
- Strict API stability slows iteration and fights lab-driven discovery.

Decision
- Labs are the authoritative design driver.
- API churn is expected and acceptable in pre-1.0.
- Refactors are encouraged when they improve clarity and keep slices thin.
- Backward compatibility is not guaranteed.
- The mandatory invariants remain:
  - always-green workflow (`ruff check .` and `pytest -q` after each commit),
  - deterministic notebook behavior (`QPL_LAB_SMOKE=1` support),
  - clean working tree before reporting completion.

Consequences
- Contributors may change interfaces when a lab slice requires it, provided tests and smoke checks are updated.
- ADRs are reserved for major architecture/process policy decisions rather than routine pre-1.0 API churn.
- Documentation should describe current interfaces as provisional maps, not frozen contracts.

Supersedes (optional)
- ADR-0001.
