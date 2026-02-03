# ADR-0001: Public API truth source

Status: accepted
Date: 2026-02-03

Context
- Agents and contributors need a clear, stable definition of what counts as "public API".
- The repo exposes both top-level package exports and submodule imports in examples.
- Docs are minimal, so compatibility rules must be explicit.

Decision
Public API in this repo is the union of:
1) Symbols exported via `__all__` in `src/qpl/__init__.py` and subpackage `__init__.py` files for `qpl.instruments`, `qpl.market`, `qpl.models`, `qpl.engines`, and `qpl.engines.analytic`.
2) The dispatcher functions `qpl.pricing.price` and `qpl.pricing.greeks`.
3) Import paths used in `examples/` and README code blocks (for example, `qpl.engines.mc.pricers.MCConfig`, `qpl.engines.pde.pricers.PDEConfig`, and `qpl.market.curves.FlatRateCurve`).

Everything else under `src/qpl/` is internal and may change without notice.

Alternatives considered
- "Everything under src is public": rejected, too broad and blocks refactors.
- "Docs define public API": rejected, docs do not enumerate all imports.
- "Only top-level qpl exports are public": rejected, examples import submodules directly.

Consequences
- Any change to public API requires a new ADR (or an update that supersedes this one) and an update to `AGENT/brain.md`.
- Examples must be updated alongside any public API change.
- Internal refactors outside the defined public API do not require an ADR.

Supersedes (optional)
- None.
