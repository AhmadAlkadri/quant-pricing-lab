# Changelog

All notable changes to `qpl` are documented in this file.

## Unreleased

## v0.3.0 — 2026-09-29

A source/repository milestone, not a PyPI publication: install from a clone
(`pip install -e ".[dev]"`). It closes Phases 0–4 of the Textbook-Driven
Development curriculum (Slices 0–17; ADR-0004). A `0.2.0` was drafted in this file
as "Unreleased" but was never tagged or given a package version; its content (the
Part I numerical building blocks below) ships here.

Measured numbers are not repeated here: each slice's evidence is in the Delivered
section of `docs/CURRICULUM.md` and in the derivation notes under `docs/notes/`.

### Added

**Validation and cases**

- `qpl.validation`: `fit_convergence_order`, `refinement_errors`, and the
  `EvidenceClass` / `BenchmarkRow` evidence taxonomy.
- `qpl.cases`: importable benchmark cases, each carrying a `BenchmarkRow` with an
  evidence class and source — European, American, digital, Asian and barrier
  options under Black–Scholes, Monte Carlo variance reduction, SDE discretisation,
  Heston pricing and Heston calibration. Evaluated by `tests/cases/` and engine tests.
- `tests/oracle/`: QuantLib-Python cross-checks for trees, PDE, American, digital,
  Asian, barrier, Fourier, Heston, Heston Monte Carlo and calibration; collected only
  when the `[oracle]` extra is installed.

**Engine registry**

- `qpl.engines.registry`: `qpl.pricing.price` / `greeks` dispatch by
  `(instrument type, model type, method)` with a per-method `MethodSpec` keyword
  contract, replacing the `isinstance` ladder (ADR-0005).

**Trees**

- `method="tree"`: CRR and Leisen–Reimer (`TreeConfig(scheme=...)`) binomial lattices
  for European, American, digital and barrier options, with lattice Greeks.

**Finite differences**

- `PDEConfig(strike_alignment="midpoint")`, fixing the strike-on-a-node convergence
  pathology.
- `PDEConfig(time_stepping="rannacher")` and grid Greeks (`greeks_method="grid"`).
- American exercise by projected SOR on the linear complementarity problem.
- Cash-or-nothing digitals with `payoff_projection="cell_average"`.
- Non-uniform `grid="sinh"` meshes with anchors (a node on the barrier), and the
  barrier PDE.

**Monte Carlo**

- Variance reduction: `MCConfig(variance_reduction=...)` over antithetic, control
  variate, stratified and their valid combinations, each with its own standard error.
- Greek estimators: `MCConfig(greeks_estimator="bump" | "pathwise" |
  "likelihood_ratio")` with per-Greek standard errors and estimator names.
- `qpl.engines.mc.sde.simulate`: Euler, Milstein and exact schemes for scalar SDEs,
  with GBM and CIR models and full truncation.
- Least-squares Monte Carlo for American options, pricing an explicit Bermudan on
  `MCConfig(exercise_dates=...)` (reported in `meta["exercise_style"]`),
  out-of-sample (low-biased) by default.

**Instruments**

- `AmericanOption` (sibling of `EuropeanOption` under `VanillaOption`),
  `DigitalOption` (cash-or-nothing), `AsianOption` (fixed strike, arithmetic or
  geometric, discrete fixings; geometric closed form, Monte Carlo with the
  Kemna–Vorst control variate, Turnbull–Wakeman reachable by name only) and
  `BarrierOption` (single barrier, eight types, rebate, continuous or discrete
  monitoring; analytic, Monte Carlo with BGK / Brownian-bridge corrections, tree
  and PDE).

**Transforms, Heston and calibration**

- `method="fourier"` (`qpl.engines.fourier`): COS, Carr–Madan (FFT or quadrature),
  Lewis and Gil-Pelaez for European options and digitals, behind a
  characteristic-function interface; implied-volatility smile utilities.
- `HestonModel`: branch-cut-safe characteristic function and cumulants, priced by the
  Fourier methods; Heston Monte Carlo (`qpl.engines.mc.heston`) with Andersen's QE
  scheme, full-truncation Euler and an exact-variance hybrid, for European, digital,
  Asian and barrier options.
- `qpl.calibration.calibrate_heston`: bounded trust-region or Levenberg–Marquardt
  fits returning the Jacobian, singular values, condition number, Gauss–Newton
  covariance and Feller flag.

**Part I numerical building blocks** (the never-tagged 0.2.0 content, driven by
Fusai & Roncoroni chapters 1–8)

- Monte Carlo sampling (`qpl.engines.mc.sampling`) and GBM path simulators
  (`qpl.engines.mc.processes`).
- Dynamic-programming optimal stopping (`qpl.engines.dp`).
- Iterative linear solvers and quadrature rules (`qpl.numerics`).
- Stehfest Laplace inversion (`qpl.transforms`).
- Gaussian copula sampling and dependence measures (`qpl.dependence`).

**Examples and docs**

- Example scripts for each capability family (`examples/README.md`), smoke-tested.
- `docs/CURRICULUM.md` (supersedes `docs/ROADMAP.md`), 18 derivation notes in
  `docs/notes/`, `docs/curriculum_provenance.md` (renamed from
  `docs/labs_to_library_map.md`), ADR-0004 and ADR-0005.
- The docs site renders the curriculum and notes from `docs/` and adds a `qpl.cases`
  reference page.

### Changed (including breaking changes)

- Optional dependencies split into extras: core is NumPy/SciPy/Matplotlib only;
  pandas/yfinance/pyarrow moved to `[data]`, QuantLib is `[oracle]`. Guarded imports
  raise `NotSupportedError` with an install hint.
- `CRRLattice` renamed `BinomialLattice` (with a `spot_centred` flag) when the
  Leisen–Reimer scheme was added; CRR prices are bit-for-bit unchanged.
- American exercise is an instrument type (`AmericanOption`), priced through
  `qpl.pricing`; `qpl.engines.dp.price_american_put_binomial` is kept only as a
  deprecated keyword wrapper.
- `PDEConfig.greeks_method` defaults to `"grid"`, so `greeks(..., method="pde")`
  returns all five Greeks; the previous behaviour is `greeks_method="bump"`.
- `labs/` is private by design (gitignored, never published).
- Ruff pinned to 0.16 with an explicit rule selection.

### CI and test contract

- Two CI jobs, `tests-core` (`.[dev]`) and `tests-full` (`.[dev,data,oracle]`), each
  running `ruff check .` then `pytest -q`.
- A `slow` pytest marker (worst case above about 3 s) so `pytest -q -m "not slow"`
  is a quick inner loop; `pytest -q` still runs everything.
- Exact float pins relaxed to round-off tolerances for cross-platform CI.
- The `heston_calibration.py` example smoke test bounds the one-maturity
  identifiability spreads (kappa spread > 2, xi spread > 1.5, worst implied-vol RMSE
  < 1e-05) instead of pinning digits that move across platforms (4fe70fb); the note,
  curriculum entry and case row now say those endpoints are one build's values.

## v0.1.0 — 2026-02-04

Initial public release; see tag `v0.1.0`.
