# Changelog

All notable changes to `qpl` are documented in this file.

## Unreleased

> Slice 0 of the Textbook-Driven Development curriculum campaign (ADR-0004; see
> `docs/CURRICULUM.md`). None of this has reached `main` yet.

### Added

- `qpl.validation`: `fit_convergence_order`/`refinement_errors` (empirical convergence-order
  measurement) and the `EvidenceClass`/`BenchmarkRow` evidence taxonomy.
- `PDEConfig.strike_alignment` (`"none"` default, bit-identical; `"midpoint"`), fixing the
  strike-on-a-grid-node convergence pathology. Measured on `S=K=100, r=5%, q=0, sigma=20%, T=1`,
  European call, Crank-Nicolson, `n_s=n_t=n` for `n` in `{50,100,200,400,800}`: unaligned fitted
  order 1.126 (log-residual 0.860, non-monotone), aligned fitted order 1.997 (residual 0.022).
  Implicit-Euler temporal order, isolated on a fixed aligned grid: 0.997 (residual 3e-04). Full
  derivation and tables: `docs/notes/pde_strike_alignment.md`.
- `qpl.cases.european_black_scholes`: an importable, tested benchmark-case layer (parity, limits,
  known values, monotonicity) with a cross-engine (analytic/PDE/MC) test.
- `[data]` and `[oracle]` optional extras, split out of core dependencies (numpy/scipy/matplotlib
  only); optional imports (pandas/yfinance/QuantLib) are guarded at call time with a clear
  `NotSupportedError` + install hint. `tests/oracle/` collects only when QuantLib is installed.
- CI now runs two jobs, `tests-core` (`.[dev]`) and `tests-full` (`.[dev,data,oracle]`), both with
  `ruff check .` then `pytest -q`.

### Changed

- `labs/` is documented as private by design and gitignored (never published), not withheld
  pending review; `docs/labs_to_library_map.md` renamed to `docs/curriculum_provenance.md`.
- `docs/ROADMAP.md` superseded by `docs/CURRICULUM.md`.

### Docs

- Added `docs/CURRICULUM.md`, `docs/notes/pde_strike_alignment.md`, and ADR-0004
  (`.agents/brain/adr/0004-textbook-driven-development.md`).

## v0.2.0 (Unreleased)

> Labs 01-08 are private, textbook-driven workbooks under `labs/` (gitignored, never published).
> The underlying package improvements they drove ship in v0.2.0.

### Highlights

- Completed a lab-driven implementation track for Fusai Chapters 1-8 with deterministic notebook smoke execution.
- Added Monte Carlo sampling utilities (LCG, stratified, inverse-transform, and accept-reject) used by Chapter 1.
- Added GBM exact and Euler path simulators plus terminal-pricing helpers for Chapter 2 experiments.
- Added dynamic-programming optimal stopping utilities and a CRR American put binomial pricer for Chapter 3.
- Added reusable numerical modules for iterative linear solvers (Chapter 5) and quadrature rules (Chapter 6).
- Added Laplace inversion utilities (Stehfest) for Chapter 7 and Gaussian copula/dependence utilities for Chapter 8.
- Expanded smoke coverage to execute labs `01` through `08` headlessly in CI-style mode.

### Added

- `src/qpl/engines/mc/sampling.py`:
  - `lcg_uniform`
  - `stratified_uniform`
  - `inverse_exponential`
  - `sample_exponential_inverse`
  - `sample_beta22_accept_reject`
- `src/qpl/engines/mc/processes.py`:
  - `simulate_gbm_exact`
  - `simulate_gbm_euler`
  - `price_european_from_terminal`
- `src/qpl/engines/dp/optimal_stopping.py`:
  - `backward_induction_optimal_stopping`
- `src/qpl/engines/dp/american_put_binomial.py`:
  - `BinomialDPConfig`
  - `build_recombining_spot_tree`
  - `price_american_put_binomial`
- `src/qpl/numerics/linear_systems.py`:
  - `LinearSolveResult`
  - `jacobi_solve`
  - `gauss_seidel_solve`
  - `sor_solve`
- `src/qpl/numerics/quadrature.py`:
  - `composite_trapezoid`
  - `composite_simpson`
  - `gauss_legendre`
- `src/qpl/transforms/laplace.py`:
  - `stehfest_coefficients`
  - `inverse_laplace_stehfest`
  - `inverse_laplace_grid_stehfest`
- `src/qpl/dependence/copulas.py`:
  - `gaussian_copula_sample`
  - `empirical_kendall_tau`
  - `empirical_spearman_rho`
  - `gaussian_copula_kendall_tau`
  - `gaussian_copula_spearman_rho`
- New chapter support notebooks:
  - `labs/01_static_monte_carlo.ipynb`
  - `labs/02_dynamic_monte_carlo.ipynb`
  - `labs/03_dynamic_programming_stochastic_optimization.ipynb`
  - `labs/04_finite_difference_methods.ipynb`
  - `labs/05_numerical_linear_systems.ipynb`
  - `labs/06_quadrature_methods.ipynb`
  - `labs/07_laplace_transform_inversion.ipynb`
  - `labs/08_copula_functions.ipynb`

### Changed

- Monte Carlo flow now exposes Chapter 2 process-level simulation primitives and routes pricing internals through them (`src/qpl/engines/mc/pricers.py` + `src/qpl/engines/mc/processes.py`).
- Dynamic-programming chapter utilities are re-exported through `src/qpl/engines/dp/__init__.py` for notebook/test ergonomics.
- Numerical utilities are re-exported through `src/qpl/numerics/__init__.py`.
- Lab smoke harness now executes all chapter labs (`tests/test_labs_smoke.py` includes `01`-`08`).

### Fixed

- Hardened input validation and edge-case handling across new numerical modules (shape/domain checks, finite-value checks, parameter bounds).
- Added no-arbitrage probability guarding for CRR binomial DP pricing in Chapter 3.
- Tightened deterministic behavior in stochastic workflows with explicit seed plumbing in new chapter utilities and labs.

### Tests

- Added chapter-specific deterministic test suites:
  - `tests/test_mc_sampling.py`
  - `tests/test_mc_processes.py`
  - `tests/test_dp_optimal_stopping.py`
  - `tests/test_pde_ch4.py`
  - `tests/test_linear_systems_ch5.py`
  - `tests/test_quadrature_ch6.py`
  - `tests/test_laplace_ch7.py`
  - `tests/test_copula_ch8.py`
- Expanded `tests/test_labs_smoke.py` to headlessly execute labs `01` through `08` with `QPL_LAB_SMOKE=1`.

### Docs

- Added this release-oriented feature summary for `v0.2.0`.
- Added `docs/curriculum_provenance.md` (renamed from `docs/labs_to_library_map.md`) to map each private
  chapter lab to the library modules/tests it drove.
- Added a README pointer to the unreleased `v0.2.0` summary.

### Internal

- Continued pre-1.0 lab-driven workflow: thin vertical slices, deterministic notebooks, and always-green checks after each slice.
- Kept architecture growth incremental via chapter-scoped modules (`numerics`, `transforms`, `dependence`, and `dp`) rather than broad dispatcher expansion.
