# Quant Pricing Lab (qpl)

A small, well-tested Python codebase for option pricing via:
- **Monte Carlo (Milestone A)**: path simulation, variance reduction, MC Greeks
- **PDE / Finite Differences (Milestone B)**: Black–Scholes PDE solver, Greeks from grids, comparisons vs analytics/MC

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
