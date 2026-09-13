# Steering Brief

What changed in Slice 3 (files + bullets)
- `examples/README.md`:
  - Established the public v0.2.0 example set and de-emphasized legacy scripts.
- `examples/bs_analytic_greeks.py`:
  - Added deterministic analytic pricing/Greeks sanity script.
- `examples/mc_pricing_and_stderr.py`:
  - Added deterministic MC pricing + stderr + z-score script.
- `examples/pde_theta_scheme.py`:
  - Added compact theta-comparison script against analytic baseline.
- `examples/american_put_binomial_dp.py`:
  - Added American put DP script with early-exercise premium diagnostics.
- `examples/quadrature_demo.py`:
  - Added rule-comparison script for smooth-function integration error.
- `examples/laplace_inversion_demo.py`:
  - Added Stehfest inversion sanity script with max/mean error output.
- `examples/copula_gaussian_demo.py`:
  - Added Gaussian copula dependence/tail script with empirical vs theoretical metrics.
- `tests/test_examples_smoke.py`:
  - Reworked smoke harness to execute only curated public scripts and assert deterministic outputs.
- `notebooks/README.md`:
  - Marked three new notebooks as public v0.2.0 and existing ones as legacy.
- `notebooks/01_pricing_overview.ipynb`:
  - Added cross-method pricing notebook (analytic/MC/PDE) with deterministic setup.
- `notebooks/02_numerical_toolkit.ipynb`:
  - Added quadrature + linear-solver + Laplace inversion notebook.
- `notebooks/03_dependence_and_copulas.ipynb`:
  - Added copula/dependence notebook with tail-risk mini metric.
- `tests/test_notebooks_smoke.py`:
  - Added headless smoke execution for the three public notebooks.
- `src/qpl/pricing.py`, `src/qpl/engines/pde/pricers.py`, `src/qpl/engines/dp/american_put_binomial.py`, `src/qpl/engines/mc/pricers.py`, `src/qpl/utils/labs.py`, `src/qpl/engines/base.py`, `src/qpl/instruments/options.py`, `src/qpl/market/market.py`:
  - Upgraded core public-facing docstrings with consistent NumPy-style structure.
- `docs-site/package.json` + `docs-site/docs/**` + `README.md` + `.gitignore`:
  - Added minimal pinned VitePress skeleton and local run instructions.

What got better
- Public learning surface no longer depends on internal `labs/` content.
- Public examples and notebooks are deterministic, smoke-friendly, and data-source independent.
- Examples/notebooks now cover pricing, numerics, transforms, DP, and dependence in a compact way.
- Public API ergonomics improved via clearer docstrings at key entry points.
- Docs-site bootstrapping is in place for iterative documentation expansion.

Deferred (intentional)
- Automatic API reference generation (manual conceptual reference only for now).
- Docs-site deployment and CI integration for Node build checks.
- Broader migration/cleanup of legacy notebooks and legacy market-data examples.

How to validate quickly
- `ruff check .`
- `pytest -q`
- `QPL_LAB_SMOKE=1 MPLBACKEND=Agg pytest -q tests/test_labs_smoke.py`
- `PYTHONPATH=src MPLBACKEND=Agg QPL_LAB_SMOKE=1 pytest -q tests/test_notebooks_smoke.py`
