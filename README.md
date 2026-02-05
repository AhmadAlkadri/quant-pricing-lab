# Quant Pricing Lab (qpl)

A small, well-tested Python codebase for option pricing with:
- **Analytic Black–Scholes**: European price + Greeks
- **Monte Carlo (GBM)**: European pricing with stderr (terminal sampling; optional multi-step discretization via `n_steps`)
- **PDE / Finite Differences**: European Black–Scholes pricing via theta scheme (call/put only)

MC Greeks supported for European options via finite differences (CRN).
Planned: variance reduction, benchmarking.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
./scripts/setup_git.sh
pytest
```

## Project brain

This repository maintains lightweight architectural context and decision history in `AGENT/`:

- `brain.md` — current architecture, invariants, and contributor/agent contract
- `adr/` — Architecture Decision Records (why key design choices were made)
- `steering-brief.md` — short summaries of recent changes and next steps

Contributors and AI agents should read `AGENT/brain.md` before making structural or API changes.

## Notebook hygiene

This repo uses nbstripout to keep notebooks deterministic.
After clone:
```bash
python -m nbstripout --install --attributes .gitattributes
```
