# Project Brain (AGENT)

How to use this document
- Read the Agent Contract before changing behavior or APIs.
- Follow evidence tags to validate claims; update tags when code moves.
- Keep updates short; link existing docs instead of duplicating.

Agent Contract
- Read-first files: `AGENT/brain.md`, `AGENT/steering-brief.md`, and relevant ADRs in `AGENT/adr/`.
- Do-not-touch without ADR + review: public API boundaries (ADR-0001), CI contract, Python compatibility, and numerical invariants in tests.
- Definition of done: tests pass, docs updated, Golden Path still runs, and no TODOs in the critical path.
- If >3 plausible causes exist, write a quick experiment or add instrumentation before changing code.
- If still uncertain after 2 iterations, produce a minimal repro and stop.
- Complexity receipts: any new abstraction must state why it exists, the bug it prevents, its cost, and what happens if omitted.
- Numerical engines must be accompanied by *empirical convergence evidence*. (Examples/notebooks must be reproducibility and interpretable).
- Public API or architectural changes require a new ADR (or update + supersede).
- Preserve determinism expectations (MC seed, PDE determinism) unless an ADR says otherwise.
- Keep error types consistent (`InvalidInputError`, `NotSupportedError`) and validate at boundaries.
- Prefer minimal disruption: avoid refactors unless they unlock the task.

0) Repo at a glance
- Purpose: small Python lab for European option pricing with analytic Black-Scholes, Monte Carlo, and PDE engines. (source: README.md; src/qpl/engines/analytic/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py)
- Primary language/toolchain: Python package `qpl`, Python >=3.10, numpy/scipy/matplotlib, pytest. (source: pyproject.toml)
- Primary entry points: `qpl.pricing.price`/`qpl.pricing.greeks` and example scripts in `examples/`. (source: src/qpl/pricing.py; examples/bs_analytic.py; examples/bs_mc_vs_analytic.py)
- How to run tests: `pytest`. (source: pyproject.toml; .github/workflows/ci.yml)
- Golden Path (clean checkout):
  ```bash
  python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && PYTHONPATH=src python examples/bs_analytic.py
  ```
  (source: README.md; examples/bs_analytic.py; pyproject.toml)

1) Purpose and non-goals
- Purpose: provide a compact, well-tested reference for European option pricing under Black-Scholes via analytic, MC, and PDE methods. (source: README.md; src/qpl/engines/analytic/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py)
- Non-goal: non-European or path-dependent instruments (only `EuropeanOption` and plain call/put payoffs exist). (source: src/qpl/instruments/options.py; src/qpl/instruments/payoffs.py)
- Non-goal: models beyond Black-Scholes (only `BlackScholesModel` is implemented). (source: src/qpl/models/black_scholes.py; src/qpl/models/__init__.py)
- Non-goal: multi-asset or stochastic rate/dividend frameworks (single-spot market with flat curves). (source: src/qpl/market/market.py; src/qpl/market/curves.py)
- Non-goal: CLI or service interface (no console scripts defined). (source: pyproject.toml)

2) Public API surface (current) - truth source: ADR-0001
- Stable: `qpl.pricing.price` and `qpl.pricing.greeks` dispatcher APIs. (source: src/qpl/pricing.py)
- Stable: `qpl.instruments` exports `EuropeanOption`, `call_payoff`, `put_payoff`. (source: src/qpl/instruments/__init__.py)
- Stable: `qpl.market` exports `Market`, `FlatRateCurve`, `FlatDividendCurve`. (source: src/qpl/market/__init__.py)
- Stable: `qpl.models` exports `BlackScholesModel`, `bs_price`. (source: src/qpl/models/__init__.py)
- Stable: `qpl.engines` exports `PriceResult`, `GreeksResult`; `qpl.engines.analytic` exports `price_european`, `greeks_european`. (source: src/qpl/engines/__init__.py; src/qpl/engines/analytic/__init__.py)
- Stable: `qpl.exceptions` module and its error types (`QPLError`, `InvalidInputError`, `ModelAssumptionError`, `NotSupportedError`). (source: src/qpl/exceptions.py; src/qpl/__init__.py)
- Experimental (public by example usage): `qpl.engines.mc.pricers.MCConfig`, `price_european`, `greeks_european`. (source: src/qpl/engines/mc/pricers.py; examples/bs_mc_vs_analytic.py)
- Experimental (public by example usage): `qpl.engines.pde.pricers.PDEConfig`, `price_european`. (source: src/qpl/engines/pde/pricers.py)
- Internal: anything else under `src/qpl/` may change without notice. (source: AGENT/adr/0001-public-api-truth-source.md)

3) Architecture (text-only diagram + bullets)

Instruments + Market + Model
        |
        v
pricing.price / pricing.greeks (dispatcher)
        |
        v
analytic engine | MC engine | PDE engine
        |
        v
PriceResult / GreeksResult

- Domain objects: `EuropeanOption`, `BlackScholesModel`, `Market` with flat curves. (source: src/qpl/instruments/options.py; src/qpl/models/black_scholes.py; src/qpl/market/market.py; src/qpl/market/curves.py)
- Dispatcher: `qpl.pricing` validates types and routes by method to engine functions. (source: src/qpl/pricing.py)
- Engines: analytic uses closed-form BS, MC uses GBM sampling (terminal or multi-step), PDE uses theta-scheme FD grid. (source: src/qpl/engines/analytic/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py)
- Results: `PriceResult` and `GreeksResult` normalize outputs across engines. (source: src/qpl/engines/base.py)
- Key entry points (paths): `src/qpl/pricing.py`, `src/qpl/__init__.py`, `src/qpl/engines/base.py`, `src/qpl/engines/analytic/black_scholes.py`, `src/qpl/engines/mc/pricers.py`, `src/qpl/engines/pde/pricers.py`, `src/qpl/instruments/options.py`, `src/qpl/market/market.py`, `src/qpl/market/curves.py`, `src/qpl/models/black_scholes.py`, `examples/bs_analytic.py`, `examples/bs_mc_vs_analytic.py`, `tests/test_pricing_analytic.py`, `tests/test_mc_pricing.py`, `tests/test_pde_pricing.py`, `.github/workflows/ci.yml`, `pyproject.toml`, `docs/ROADMAP.md`.

4) Key invariants and assumptions
- `EuropeanOption` requires kind in {"call","put"}, strike > 0, expiry >= 0; enforced at init. (source: src/qpl/instruments/options.py)
- `Market` requires spot > 0; enforced at init. (source: src/qpl/market/market.py)
- `BlackScholesModel` requires sigma >= 0; enforced at init. (source: src/qpl/models/black_scholes.py)
- Flat curves require non-negative rate/yield unless `allow_negative=True`. (source: src/qpl/market/curves.py)
- Dispatcher only supports `EuropeanOption` + `BlackScholesModel` + `Market` for methods {analytic, mc, pde}; otherwise `NotSupportedError`. (source: src/qpl/pricing.py)
- MC config requires n_paths >= 2 and n_steps >= 1; stderr uses ddof=1; results deterministic for a fixed seed. (source: src/qpl/engines/mc/pricers.py; tests/test_mc_pricing.py)
- PDE config requires n_s >= 3, n_t >= 1, theta in [0,1]; deterministic for fixed inputs. (source: src/qpl/engines/pde/pricers.py; tests/test_pde_pricing.py)
- At T=0 or sigma=0, pricing returns intrinsic or discounted-forward intrinsic (analytic/MC/PDE). (source: src/qpl/models/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py; tests/test_pricing_analytic.py; tests/test_mc_pricing.py)
- Units: T in years; r and q are continuously compounded. (source: src/qpl/models/black_scholes.py; src/qpl/market/curves.py)

5) Error handling & validation policy
- All domain validation errors raise `InvalidInputError`; unsupported combos raise `NotSupportedError`. (source: src/qpl/exceptions.py; src/qpl/pricing.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py; src/qpl/instruments/options.py)
- Validation happens at object construction and at engine entry points, not via return codes. (source: src/qpl/instruments/options.py; src/qpl/market/market.py; src/qpl/models/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py)
- `ModelAssumptionError` exists as a typed hook but is currently unused. (source: src/qpl/exceptions.py)

6) Configuration & defaults
- MC defaults: `MCConfig(n_paths=50_000, n_steps=1, seed=123)`. (source: src/qpl/engines/mc/pricers.py)
- PDE defaults: `PDEConfig(n_s=200, n_t=200, theta=0.5, s_max=None, s_max_multiplier=4.0)`. (source: src/qpl/engines/pde/pricers.py)
- Market defaults: flat rate/dividend curves with `allow_negative=False`. (source: src/qpl/market/curves.py)
- No repo-level config files beyond `pyproject.toml`; behavior is code-driven. (source: pyproject.toml)

Top 10 cheapest checks
- Check Python version >=3.10. (source: pyproject.toml)
- Check `pip install -e ".[dev]"` succeeds. (source: pyproject.toml)
- Check `pytest -q` runs. (source: pyproject.toml; .github/workflows/ci.yml)
- Check `python -m compileall src` for syntax errors. (source: src/)
- Check `PYTHONPATH=src python examples/bs_analytic.py` prints price/greeks. (source: examples/bs_analytic.py)
- Check `PYTHONPATH=src python examples/bs_mc_vs_analytic.py` prints analytic/mc lines. (source: examples/bs_mc_vs_analytic.py; tests/test_examples_smoke.py)
- Check `qpl.__version__` matches `pyproject.toml` version. (source: src/qpl/__init__.py; pyproject.toml)
- Check MC defaults (n_paths, n_steps, seed) match expectations in code. (source: src/qpl/engines/mc/pricers.py)
- Check PDE defaults (n_s, n_t, theta) match expectations in code. (source: src/qpl/engines/pde/pricers.py)
- Check CI runs on Python 3.11 with `pytest`. (source: .github/workflows/ci.yml)

7) Testing & CI contract
- Test runner: `pytest` with tests in `tests/`. (source: pyproject.toml)
- CI: GitHub Actions runs on ubuntu-latest, sets up Python 3.11, installs `.[dev]`, then runs `pytest`. (source: .github/workflows/ci.yml)
- Example smoke: `examples/bs_mc_vs_analytic.py` must run successfully (tested via subprocess). (source: tests/test_examples_smoke.py)

8) Decisions log (index)
- ADRs live in `AGENT/adr/` (see `AGENT/adr/0000-template.md`).
- Accepted ADRs: `AGENT/adr/0001-public-api-truth-source.md`.
- ADR rules: one decision per ADR, keep under 1 page, include status and supersedes links. (source: AGENT/adr/0000-template.md)

9) Roadmap: next 3 increments (vertical slices only)
- [Done] Enable PDE Greeks (Delta/Gamma): `pricing.greeks(..., method='pde')`. Implemented in `src/qpl/engines/pde/pricers.py`.
- [Done] Fix MC Theta: `greeks(..., method='mc')` returns non-zero Theta. Validated in `tests/test_mc_greeks.py`.
- [Next] Binary/Digital Options (Analytic): new instrument `BinaryOption`.
  - Criteria: `BinaryOption` class, analytic price formula ($e^{-rT} N(d_2)$ for Call), `pricing` dispatcher update (analytic only), tests, and `examples/binary_demo.py`.
- [Queued] Benchmark Harness: standard perf tracking.
- [Queued] MC Variance Reduction (Antithetic).

10) Next Thin Vertical Slices: Implied Volatility & Time-Series Grounding

1) Implied Volatility Solver (Analytic, European)
   - User story: As a quant, I want to compute implied volatility from an observed option price so I can invert the Black–Scholes model.
   - Notes:
     - Analytic Black–Scholes only.
     - Robust root-finding.
     - Deterministic tests and example script.
   - Rationale: Core quant concept; small surface area; unlocks calibration workflows.

2) Implied Volatility Teaching Notebook
   - User story: As a learner, I want to understand what implied volatility means and how it behaves as option prices change.
   - Notes:
     - Demonstrate monotonicity of price vs σ.
     - ATM vs ITM/OTM sensitivity.
     - Repricing consistency (σ → price → implied σ).
   - Rationale: High pedagogical value; pure vertical polish on the solver.

3) Historical Volatility Estimation from Time Series
   - User story: As a quant, I want to estimate σ from historical returns so I can compare realized vs implied volatility.
   - Notes:
     - Load historical prices.
     - Compute log returns and annualized realized volatility.
     - Minimal statistics, no option pricing yet.
   - Rationale: Bridges theory to data; thin, self-contained slice.

4) Implied vs Realized Volatility Comparison
   - User story: As a practitioner, I want to compare implied volatility to realized volatility to see when Black–Scholes assumptions break down.
   - Notes:
     - Rolling realized volatility.
     - Sample implied volatility.
     - Visual comparison over time.
   - Rationale: Ties together inversion + data; intuitive and compelling.

5) Simple Time-Series Model Fit for Returns (μ, σ)
   - User story: As a quant, I want to fit μ and σ from historical data and evaluate how well BS assumptions hold.
   - Notes:
     - Estimate drift and volatility.
     - Compare empirical return distribution to Gaussian.
   - Rationale: Explicitly introduces model-vs-reality pressure; prepares ground for richer models later.


11) Open questions / risks
- README Quickstart code block appears unclosed; verify and fix if needed. Verify: re-open `README.md` and confirm markdown renders cleanly. (source: README.md)
- Version sync risk: `pyproject.toml` and `qpl.__version__` must stay aligned. Verify: compare values and define an update rule. (source: pyproject.toml; src/qpl/__init__.py)

12) Execution Principle: Thin Vertical Slices
- We simply do not build "layers". We build **slices**.
- A slice = Public API + Engine Logic + Test + Golden Path update.
- See `AGENT/adr/0002-thin-vertical-slices.md`.
