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

## What's New In v0.2.0

`v0.2.0` is currently unreleased and summarized in `CHANGELOG.md` (see the `v0.2.0 (Unreleased)` section).
It consolidates the lab-driven Chapters 1-8 buildout into new `qpl` modules for MC sampling/simulation,
dynamic programming, numerical linear solvers, quadrature, Laplace inversion, and copulas.

Labs are currently withheld from publication pending legal review; the code improvements are intended to ship.

Development phase: exploratory pre-1.0. API churn is expected, labs are authoritative,
and backward compatibility is not guaranteed. See `AGENTS.md` and
`.agents/brain/adr/0003-pre-1-0-lab-authority-and-api-churn.md`.

## License
MIT

## Project Brain / Contribution workflow

See `.agents/brain/brain.md` (architecture + invariants), `.agents/brain/adr/` (decision records), and
`.agents/brain/steering-brief.md` (recent changes). Read `.agents/brain/brain.md` before structural or API changes.

## Notebook hygiene

This repo uses nbstripout to keep notebooks deterministic.
After clone:
```bash
python -m nbstripout --install --attributes .gitattributes
```

## Documentation site (local)

```bash
cd docs-site
npm install
npm run docs:dev
```

Docs are published on GitHub Pages via GitHub Actions.
