# Quant Pricing Lab (qpl)

A small, well-tested Python codebase for option pricing with:
- **Analytic Black–Scholes**: European price + Greeks
- **Monte Carlo (GBM)**: European pricing with stderr (terminal sampling; optional multi-step discretization via `n_steps`)

Planned: MC Greeks + variance reduction, PDE/finite-difference pricer, benchmarking.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
