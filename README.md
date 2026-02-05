# Quant Pricing Lab (qpl)

A small, well-tested Python codebase for option pricing with:
- **Analytic Black–Scholes**: European price + Greeks
- **Monte Carlo (GBM)**: European pricing with stderr (terminal sampling; optional multi-step discretization via `n_steps`)
- **PDE / Finite Differences**: European Black–Scholes pricing via theta scheme (call/put only)

MC Greeks supported for European options via finite differences (CRN).
Planned: variance reduction, benchmarking.

## Release: v0.1.0

Quant Pricing Lab is a compact Python lab for European option pricing under the Black–Scholes
assumptions, with analytic pricing/Greeks plus Monte Carlo and PDE (theta-scheme) engines for
vanilla calls and puts. It is intentionally narrow in scope: no non-European or path-dependent
instruments, no models beyond Black–Scholes, and no multi-asset or production trading framework.

Golden Path:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python examples/bs_analytic.py
pytest
```

Stability / Compatibility: v0.x — the public API may change; see
`AGENT/adr/0001-public-api-truth-source.md` for the current stable surface.

## License
MIT

## Project Brain / Contribution workflow

See `AGENT/brain.md` (architecture + invariants), `AGENT/adr/` (decision records), and
`AGENT/steering-brief.md` (recent changes). Read `AGENT/brain.md` before structural or API changes.

## Notebook hygiene

This repo uses nbstripout to keep notebooks deterministic.
After clone:
```bash
python -m nbstripout --install --attributes .gitattributes
```
