# Project Brain (.agents)

How to use this document
- Read the Agent Contract before changing behavior or APIs.
- Follow evidence tags to validate claims; update tags when code moves.
- Keep updates short; link existing docs instead of duplicating.

Agent Contract
- Read-first files: `.agents/brain/brain.md`, `.agents/brain/steering-brief.md`, and relevant ADRs in `.agents/brain/adr/`.
- Pre-1.0 API churn acceptable when lab-driven and tested.
- Do-not-break invariants: CI contract, Python compatibility, numerical invariants in tests, determinism, and clean-tree hygiene.
- Definition of done: `ruff check .` passes, `pytest -q` passes, docs are updated, and no TODOs remain in the critical path.
- If >3 plausible causes exist, write a quick experiment or add instrumentation before changing code.
- If still uncertain after 2 iterations, produce a minimal repro and stop.
- Complexity receipts: any new abstraction must state why it exists, the bug it prevents, its cost, and what happens if omitted.
- Numerical engines must be accompanied by *empirical convergence evidence*. (Examples/notebooks must be reproducibility and interpretable).
- Use ADRs for major architecture/process decisions and policy shifts.
- Preserve determinism expectations (MC seed, PDE determinism) unless an ADR says otherwise.
- Keep error types consistent (`InvalidInputError`, `NotSupportedError`) and validate at boundaries.
- Refactors are encouraged when they improve clarity and still ship a thin, green slice.
- **Lab-first slices**: for educational work, ship one runnable lab notebook plus the minimum `src/` and tests needed to support it.
- **Thin vertical scope**: each slice should include behavior, deterministic verification, and proof-of-run evidence in one commit chain.
- **Proof of run**: after each commit, record local results for `ruff check .`, `pytest -q`, and notebook smoke execution whenever notebooks change.
- **Clean Working Tree**: Agents must not report completion unless `git status` is clean. When renaming files, always verify deletions are staged.

0) Repo at a glance
- Purpose: a numerical-methods lab built by Textbook-Driven Development (ADR-0004): textbook/literature results are re-derived, turned into cases, cross-checked, and backed by measured convergence or statistical evidence before becoming package capability. Covers analytic/MC/PDE European pricing under Black-Scholes, binomial trees (CRR and Leisen-Reimer) covering European and American exercise, and supporting numerics/transforms/dependence modules. (source: docs/CURRICULUM.md; src/qpl/engines/analytic/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py; src/qpl/engines/tree/american.py; src/qpl/numerics/__init__.py; src/qpl/transforms/__init__.py; src/qpl/dependence/__init__.py)
- Primary language/toolchain: Python package `qpl`, Python >=3.10, numpy/scipy/matplotlib, pytest. (source: pyproject.toml)
- Primary entry points: `qpl.pricing.price`/`qpl.pricing.greeks` and example scripts in `examples/`. (source: src/qpl/pricing.py; examples/bs_analytic.py; examples/bs_mc_vs_analytic.py)
- How to run tests: `pytest`. (source: pyproject.toml; .github/workflows/ci.yml)
- Golden Path (clean checkout):
  ```bash
  python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && PYTHONPATH=src python examples/bs_analytic.py
  ```
  (source: README.md; examples/bs_analytic.py; pyproject.toml)

1) Purpose and non-goals
- Purpose: provide a well-tested, evidence-backed reference for option pricing methods, growing along the phase plan in `docs/CURRICULUM.md` (foundations -> BS analytic -> trees/PDE/MC -> American exercise -> transforms/Heston/calibration -> exotics as method drivers -> multi-asset/rates/performance). (source: docs/CURRICULUM.md; README.md)
- Non-goal: market-data expansion beyond the frozen `[data]` extra (D9); the market-data strand is done but out of curriculum scope. (source: docs/CURRICULUM.md; pyproject.toml)
- Non-goal: a production trading framework or service/CLI interface. (source: docs/CURRICULUM.md; pyproject.toml)
- Non-goal: a universal pricing ontology built ahead of the cases that justify it — instruments and models are added only when a phase-plan case needs them, not speculatively. (source: docs/CURRICULUM.md)
- Non-goal: performance work before reference paths exist — profiling/vectorization/benchmark harness are Phase 5+, gated on having reference results to benchmark against. (source: docs/CURRICULUM.md)

## Notebook Hygiene
- Notebooks live in `notebooks/` and follow a strict numbering scheme: `NN_description.ipynb`.
- Numbers increase monotonically (`00`, `01`, `02`, ...).
- Fusai teaching labs live in `labs/` and follow the same naming convention (`NN_description.ipynb`).
- `labs/` is private by design and gitignored (see `.gitignore`); it is never published. It drives
  package changes as a textbook workbook set, not as public cargo.
- Output is stripped via `nbstripout` (enforced by `.gitattributes`).
- Dependencies: Must use repo environment; no cells should fail.

2) Current API map (pre-1.0; subject to change)
- Current dispatcher interfaces: `qpl.pricing.price` and `qpl.pricing.greeks`; both bind keyword arguments through a per-method `MethodSpec` and resolve the engine through the registry, rather than an `isinstance` ladder (ADR-0005). (source: src/qpl/pricing.py)
- Current registry exports: `qpl.engines.registry` exports `MethodSpec`, `EngineKey`, `register`, `resolve_price`, `resolve_greeks`, `method_spec`, `known_methods`. (source: src/qpl/engines/registry.py)
- Current instrument exports: `qpl.instruments` exports `VanillaOption`, `EuropeanOption`, `AmericanOption`, `call_payoff`, `put_payoff`. Exercise style is an instrument property: `EuropeanOption` and `AmericanOption` are sibling subclasses of the validation-only base `VanillaOption`, deliberately *not* parent and child, because registry lookup walks the MRO and a subclass would silently resolve to the European engines. (source: src/qpl/instruments/__init__.py; src/qpl/instruments/options.py)
- Current market exports: `qpl.market` exports `Market`, `FlatRateCurve`, `FlatDividendCurve`. (source: src/qpl/market/__init__.py)
- Current model exports: `qpl.models` exports `BlackScholesModel`, `bs_price`. (source: src/qpl/models/__init__.py)
- Current engine exports: `qpl.engines` exports `PriceResult`, `GreeksResult`; `qpl.engines.analytic` exports `price_european`, `greeks_european`. (source: src/qpl/engines/__init__.py; src/qpl/engines/analytic/__init__.py)
- Current exception exports: `qpl.exceptions` module and its error types (`QPLError`, `InvalidInputError`, `ModelAssumptionError`, `NotSupportedError`). (source: src/qpl/exceptions.py; src/qpl/__init__.py)
- Current example-level interfaces: `qpl.engines.mc.pricers.MCConfig`, `price_european`, `greeks_european`. (source: src/qpl/engines/mc/pricers.py; examples/bs_mc_vs_analytic.py)
- Current example-level interfaces: `qpl.engines.pde.pricers.PDEConfig`, `price_european`, `greeks_european`. (source: src/qpl/engines/pde/pricers.py; examples/pde_greeks_demo.py)
- Current tree engine exports: `qpl.engines.tree` exports `TreeConfig`, `BinomialLattice`, `Scheme`, `SCHEMES`, `crr_parameters`, `leisen_reimer_parameters`, `peizer_pratt_inversion`, `lattice_parameters`, `crr_spot_level`, `build_recombining_spot_tree`, `price_european`, `greeks_european`, `price_american`, `greeks_american`, `TREE_METHOD_SPEC`; reachable from the dispatcher as `method="tree"`. `TreeConfig(n_steps=200, scheme="crr")` takes `scheme` in `{"crr", "leisen-reimer"}`; `"leisen-reimer"` requires an **odd** `n_steps` and raises `InvalidInputError` on an even one (it has no even-n construction, and QuantLib's silent round-up is measured to be order 1 rather than order 2 -- see `docs/notes/leisen_reimer.md`). `lattice_parameters(scheme=...)` is the only place that branches on the scheme; every pricer, the Bellman step and the Greek estimators read only `up`, `down`, `p`, `discount`. `CRRLattice` was renamed `BinomialLattice` in Slice 3 and gained `spot_centred` (True only for CRR, where `u d == 1` by construction; the theta estimator needs to know). (source: src/qpl/engines/tree/__init__.py; src/qpl/engines/tree/lattice.py; src/qpl/pricing.py) Registry keys for `method="tree"`: `(EuropeanOption, BlackScholesModel, "tree")` for price and Greeks, and `(AmericanOption, BlackScholesModel, "tree")` for price and Greeks. The analytic, MC and PDE engines are registered for `EuropeanOption` only, so an `AmericanOption` sent to them raises `NotSupportedError` through the ordinary lookup rather than through a per-engine guard. (source: src/qpl/engines/tree/__init__.py; src/qpl/pricing.py)
- `PDEConfig` has a `strike_alignment: Literal["none", "midpoint"]` field; `"midpoint"` nudges the grid spacing so the strike sits exactly between two nodes, restoring measured order-2 convergence (see `docs/notes/pde_strike_alignment.md`). Default `"none"` is bit-identical to pre-Slice-0 output. (source: src/qpl/engines/pde/pricers.py)
- `PDEConfig` gained two fields in Slice 4, both defaulting to the previous behaviour. `time_stepping: Literal["theta", "rannacher"] = "theta"`: `"rannacher"` replaces the first **two** nominal steps by **four fully implicit steps of `dt/2`** (the module constant is `RANNACHER_STARTUP_STEPS = 4`) and then continues with `cfg.theta`; it requires `n_t >= 2`, takes `n_t + 2` steps, and covers exactly `n_t * dt` because halving is exact in binary floating point. `greeks_method: Literal["grid", "bump"] = "grid"`: `"grid"` reads delta and gamma from second-order central stencils on the finished grid, theta from the PDE identity, and vega/rho by bump-and-revalue on the same grid (`GRID_VEGA_BUMP = 1e-2`, `GRID_RHO_BUMP = 1e-4`), raising `InvalidInputError` at `T = 0` or `sigma = 0`; `"bump"` is the pre-Slice-4 three-solve spot bump (`BUMP_SPOT_FRACTION = 1e-2`) that still returns NaN for vega, theta and rho, kept as a named alternative and measured to stall at order 0.418 against the grid path's 2.001. `PriceResult.meta` adds `time_stepping`, `implicit_startup_steps` and `n_steps_taken`; `GreeksResult.meta` names the source of every Greek and carries `theta_backward_difference` as a first-order cross-check on the identity. Derivation and tables: `docs/notes/pde_greeks_and_rannacher.md`. (source: src/qpl/engines/pde/pricers.py)
- Current validation exports: `qpl.validation` exports `ConvergenceFit`, `fit_convergence_order`, `refinement_errors`, `BenchmarkRow`, `EvidenceClass`. (source: src/qpl/validation/__init__.py)
- Current cases exports: `qpl.cases` exports `EuropeanBSSpec`, `EuropeanBSCase`, `PARITY_CASES`, `LIMIT_CASES`, `KNOWN_VALUE_CASES`, `TREE_ORDER_CASES`, `MONOTONICITY_CASES`, `PDE_GREEK_CASES`, `ALL_CASES`, `parity_residual`, plus the tree study constants `TREE_ODD_LEVELS`, `TREE_EVEN_LEVELS`, `TREE_REFERENCE_N_STEPS`, `TREE_KNOWN_VALUE_TOLERANCE`. It also exports the American layer: `AmericanBSSpec`, `AmericanBSCase`, `LS2001_CASES`, `AMERICAN_IDENTITY_CASES`, `AMERICAN_PREMIUM_CASES`, `AMERICAN_REFERENCE_CASES`, `ALL_AMERICAN_CASES`, and the constants `LS2001_ROW1`, `LS2001_BERMUDAN_EXERCISES_PER_YEAR`, `LS2001_N_STEPS`, `PREMIUM_STRIKE_LADDER`, `AMERICAN_REFERENCE_SPEC`, `AMERICAN_REFERENCE_N_STEPS`, `AMERICAN_REFERENCE_VALUE`. `ALL_CASES` stays European-only; the two id spaces are asserted disjoint. Slice 3 adds `TREE_LR_ORDER_CASES`, `TREE_LR_REFERENCE_N_STEPS`, `TREE_LR_KNOWN_VALUE_TOLERANCE`, `AMERICAN_LR_CASES`, `AMERICAN_LR_LEVELS` and `AMERICAN_BRACKETED_LIMIT` (the converged American put value, distinct from `AMERICAN_REFERENCE_VALUE`, which pins this engine at n=8001). Slice 4 adds `PDE_GREEK_CASES` (ten `CLOSED_FORM` rows, five Greeks at the reference ATM call and put), the grid constants `PDE_GREEKS_N = 400`, `PDE_GREEKS_TIME_STEPPING = "rannacher"`, `PDE_GREEKS_STRIKE_ALIGNMENT = "midpoint"`, and exports the previously module-private `REFERENCE_ATM_CALL` / `REFERENCE_ATM_PUT`. (source: src/qpl/cases/__init__.py; src/qpl/cases/american_black_scholes.py; src/qpl/cases/european_black_scholes.py)
- Current DP engine exports: `qpl.engines.dp` exports `backward_induction_optimal_stopping` (the generic finite-horizon optimal-stopping solver) plus `BinomialDPConfig` and `price_american_put_binomial`, which are now a deprecated thin wrapper over `qpl.engines.tree.price_american` kept for lab notebooks written against the Slice 1 signature; the wrapper returns that engine's value bit for bit and only adds the optional `return_lattice=True` payload. American pricing itself is on the dispatcher. (source: src/qpl/engines/dp/__init__.py; src/qpl/engines/dp/american_put_binomial.py)
- Current numerics exports: `qpl.numerics` exports `LinearSolveResult`, `jacobi_solve`, `gauss_seidel_solve`, `sor_solve`, `composite_trapezoid`, `composite_simpson`, `gauss_legendre`. (source: src/qpl/numerics/__init__.py)
- Current transforms exports: `qpl.transforms` exports `stehfest_coefficients`, `inverse_laplace_stehfest`, `inverse_laplace_grid_stehfest`. (source: src/qpl/transforms/__init__.py)
- Current dependence exports: `qpl.dependence` exports `gaussian_copula_sample`, `empirical_kendall_tau`, `empirical_spearman_rho`, `gaussian_copula_kendall_tau`, `gaussian_copula_spearman_rho`. (source: src/qpl/dependence/__init__.py)
- Internal modules may be refactored freely when lab-driven and tested.

3) Architecture (text-only diagram + bullets)

Instruments + Market + Model
        |
        v
pricing.price / pricing.greeks (dispatcher)
        |
        v
engines.registry: MethodSpec (kwargs contract)
                  + (instrument type, model type, method) -> engine
        |
        v
analytic engine | MC engine | PDE engine | tree engine
        |
        v
PriceResult / GreeksResult

- Domain objects: `EuropeanOption` and `AmericanOption` (siblings under `VanillaOption`), `BlackScholesModel`, `Market` with flat curves. Exercise style is carried by the instrument type and resolved by the registry: no engine inspects an `is_american` flag, and an engine that cannot handle early exercise refuses by simply not being registered for `AmericanOption`. This is an application of ADR-0005's key rather than a change to it, so no new ADR was written. (source: src/qpl/instruments/options.py; src/qpl/models/black_scholes.py; src/qpl/market/market.py; src/qpl/market/curves.py)
- Dispatcher: `qpl.pricing` binds kwargs via the method's `MethodSpec`, resolves the engine from `qpl.engines.registry`, and calls it; the wiring table is `qpl.pricing._register_builtin_engines`. `Market` is type-checked but is not part of the key (ADR-0005). (source: src/qpl/pricing.py; src/qpl/engines/registry.py; .agents/brain/adr/0005-engine-registry.md)
- Engines: analytic uses closed-form BS, MC uses GBM sampling (terminal or multi-step), PDE uses a theta-scheme FD grid with optional Rannacher start-up and a banded (LAPACK) tridiagonal solve, tree uses a recombining binomial lattice -- CRR or Leisen-Reimer, selected by `TreeConfig.scheme` -- with vectorized backward induction. The lattice builder in `qpl.engines.tree.lattice` is the only scheme-aware code and is shared with the American tree engine and the American-put DP wrapper. (source: src/qpl/engines/analytic/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py; src/qpl/engines/tree/pricers.py; src/qpl/engines/tree/lattice.py)
- Results: `PriceResult` and `GreeksResult` normalize outputs across engines. (source: src/qpl/engines/base.py)
- Key entry points (paths): `src/qpl/pricing.py`, `src/qpl/engines/registry.py`, `src/qpl/__init__.py`, `src/qpl/engines/base.py`, `src/qpl/engines/analytic/black_scholes.py`, `src/qpl/engines/mc/pricers.py`, `src/qpl/engines/pde/pricers.py`, `src/qpl/instruments/options.py`, `src/qpl/market/market.py`, `src/qpl/market/curves.py`, `src/qpl/models/black_scholes.py`, `examples/bs_analytic.py`, `examples/bs_mc_vs_analytic.py`, `tests/test_pricing_analytic.py`, `tests/test_mc_pricing.py`, `tests/test_pde_pricing.py`, `.github/workflows/ci.yml`, `pyproject.toml`, `docs/CURRICULUM.md`.
- The dispatcher's `isinstance` ladder was replaced by the engine registry keyed by `(instrument type, model type, method)` (ADR-0005, accepted). Adding an engine means a `MethodSpec` next to its config object plus one `register(...)` call. (source: src/qpl/pricing.py; src/qpl/engines/registry.py; .agents/brain/adr/0005-engine-registry.md)

4) Key invariants and assumptions
- `VanillaOption` (and so `EuropeanOption` and `AmericanOption`) requires kind in {"call","put"}, strike > 0, expiry >= 0; enforced at init. (source: src/qpl/instruments/options.py)
- `Market` requires spot > 0; enforced at init. (source: src/qpl/market/market.py)
- `BlackScholesModel` requires sigma >= 0; enforced at init. (source: src/qpl/models/black_scholes.py)
- Flat curves require non-negative rate/yield unless `allow_negative=True`. (source: src/qpl/market/curves.py)
- Dispatcher supports `EuropeanOption` + `BlackScholesModel` + `Market` for methods {analytic, mc, pde, tree}, and `AmericanOption` + `BlackScholesModel` + `Market` for `tree` only; otherwise `NotSupportedError`. Keyword validation runs before engine lookup, so a bad `cfg` yields `InvalidInputError`, not `NotSupportedError`. (source: src/qpl/pricing.py; src/qpl/engines/registry.py; tests/test_engine_registry.py)
- MC config requires n_paths >= 2 and n_steps >= 1; stderr uses ddof=1; results deterministic for a fixed seed. (source: src/qpl/engines/mc/pricers.py; tests/test_mc_pricing.py)
- PDE config requires n_s >= 3, n_t >= 1, theta in [0,1], `strike_alignment` in {"none","midpoint"}, `time_stepping` in {"theta","rannacher"} with `n_t >= 2` for the latter, and `greeks_method` in {"grid","bump"}; deterministic for fixed inputs. PDE grid Greeks additionally require `T > 0` and `sigma > 0` (gamma is a point mass in both limits), matching the analytic engine; `price_european` still returns the correct price there. (source: src/qpl/engines/pde/pricers.py; tests/test_pde_pricing.py; tests/test_pde_greeks.py)
- Tree config requires n_steps >= 1 for prices and >= 2 for Greeks (gamma and theta read step-2 nodes), and scheme in {"crr", "leisen-reimer"}; `scheme="leisen-reimer"` additionally requires an odd n_steps, checked after the n_steps bound so `n_steps=0` still reports the size. The same applies to the American tree engine, whose Greeks use the same estimators (`lattice_delta_gamma_theta`). The CRR lattice raises `InvalidInputError` when no-arbitrage fails; the Leisen-Reimer lattice cannot violate it (`p' > p` follows from `d1 > d2`, and that is exactly `d < growth < u`). `lattice_delta_gamma_theta` takes an optional `spot`: pass it on a lattice that is not spot-centred, where theta needs an off-centre correction, and omit it on CRR, where the correction would only add round-off. Deterministic for fixed inputs. (source: src/qpl/engines/tree/pricers.py; src/qpl/engines/tree/lattice.py; tests/test_tree_pricing.py; tests/test_tree_leisen_reimer.py)
- At T=0 or sigma=0, European pricing returns intrinsic or discounted-forward intrinsic (analytic/MC/PDE/tree). The American tree engine returns intrinsic at T=0, and at sigma=0 the supremum of the discounted intrinsic value over the deterministic forward path, which equals `max(intrinsic now, discounted forward intrinsic)` whenever `r >= q` but **not** in general when `q > r` (derived and pinned: `qpl.engines.tree.american._deterministic_exercise_times`). (source: src/qpl/models/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py; tests/test_pricing_analytic.py; tests/test_mc_pricing.py)
- Units: T in years; r and q are continuously compounded. (source: src/qpl/models/black_scholes.py; src/qpl/market/curves.py)

5) Error handling & validation policy
- All domain validation errors raise `InvalidInputError`; unsupported combos raise `NotSupportedError`. (source: src/qpl/exceptions.py; src/qpl/pricing.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py; src/qpl/instruments/options.py)
- Validation happens at object construction and at engine entry points, not via return codes. (source: src/qpl/instruments/options.py; src/qpl/market/market.py; src/qpl/models/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py)
- `ModelAssumptionError` exists as a typed hook but is currently unused. (source: src/qpl/exceptions.py)

6) Configuration & defaults
- MC defaults: `MCConfig(n_paths=50_000, n_steps=1, seed=123)`. (source: src/qpl/engines/mc/pricers.py)
- PDE defaults: `PDEConfig(n_s=200, n_t=200, theta=0.5, s_max=None, s_max_multiplier=4.0, strike_alignment="none", time_stepping="theta", greeks_method="grid")`. Every default except `greeks_method` reproduces the pre-Slice-4 price; `greeks_method` deliberately defaults to the new grid path, because the old one returned NaN for three of the five Greeks. (source: src/qpl/engines/pde/pricers.py)
- Tree defaults: `TreeConfig(n_steps=200, scheme="crr")` -- the default scheme is unchanged by Slice 3, and CRR prices and Greeks are bit-for-bit what they were; tree Greek bumps `VEGA_BUMP=1e-2`, `RHO_BUMP=1e-4`, both chosen from a measured scan. (source: src/qpl/engines/tree/pricers.py)
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
- Check PDE defaults (n_s, n_t, theta, strike_alignment, time_stepping, greeks_method) match expectations in code. (source: src/qpl/engines/pde/pricers.py)
- Check CI runs on Python 3.11 with `pytest`. (source: .github/workflows/ci.yml)

7) Testing & CI contract
- Test runner: `pytest` with tests in `tests/`. (source: pyproject.toml)
- CI runs two jobs on ubuntu-latest / Python 3.11: `tests-core` (`pip install -e ".[dev]"`) and `tests-full` (`pip install -e ".[dev,data,oracle]"`); both run `ruff check .` then `pytest -q`. (source: .github/workflows/ci.yml)
- Example smoke: `examples/bs_mc_vs_analytic.py` must run successfully (tested via subprocess). (source: tests/test_examples_smoke.py)
- Lab smoke: notebooks in `labs/` must execute headlessly via `jupyter nbconvert --execute` under deterministic env vars.
- Evidence discipline: any numerical claim added to `tests/` or `docs/notes/` states which `qpl.validation.EvidenceClass` justifies it; numerical engines require measured convergence-order evidence (`qpl.validation.fit_convergence_order`) before being treated as delivered, not just a passing tolerance check. (source: src/qpl/validation/benchmark.py; src/qpl/validation/convergence.py; .agents/brain/adr/0004-textbook-driven-development.md)
- `tests/oracle/` collects only when the optional `QuantLib` (`[oracle]` extra) is importable; otherwise it is skipped at collection time, including when targeted directly by path. (source: tests/oracle/conftest.py)

8) Decisions log (index)
- ADRs live in `.agents/brain/adr/` (see `.agents/brain/adr/0000-template.md`).
- Active ADRs: `.agents/brain/adr/0002-thin-vertical-slices.md`, `.agents/brain/adr/0003-pre-1-0-lab-authority-and-api-churn.md` (still active; compatible with ADR-0004), `.agents/brain/adr/0004-textbook-driven-development.md`, `.agents/brain/adr/0005-engine-registry.md`.
- Historical/superseded ADRs: `.agents/brain/adr/0001-public-api-truth-source.md`.
- ADR rules: one decision per ADR, keep under 1 page, include status and supersedes links. (source: .agents/brain/adr/0000-template.md)

9) Curriculum
- Full curriculum map (TDD loop, identity decision, literature spine, method dependency graph, evidence classes, provenance rules, phase plan, Slice 0 delivered results, reconciled old roadmap): `docs/CURRICULUM.md`. (source: docs/CURRICULUM.md)
- Slice 1 delivered (Phase 1, first half): the engine registry (ADR-0005) and the CRR binomial tree with measured order-1 convergence, the odd/even oscillation documented rather than hidden, lattice Greeks, and a QuantLib oracle that disproved the expected 1e-10 agreement and pinned the reason instead. Derivation: `docs/notes/crr_tree_convergence.md`. (source: docs/CURRICULUM.md; .agents/brain/steering-brief.md)
- Slice 2 delivered (Phase 1, second half): American exercise as an instrument type (`AmericanOption`, sibling of `EuropeanOption` under `VanillaOption`), American pricing and Greeks on the CRR lattice with the early-exercise boundary, measured order-1 convergence with the odd/even bracketing confirmed, and four findings that contradicted the slice's written expectations (the Longstaff-Schwartz Table 1 value is a 50-exercise-date Bermudan, the sigma=0 put formula needs r >= q, the extracted boundary is monotone only within a node-grid parity, and Richardson extrapolation does not restore order 2). Derivation: `docs/notes/american_exercise_on_trees.md`. (source: docs/CURRICULUM.md; .agents/brain/steering-brief.md)
- Slice 3 delivered (Phase 1, completing it): `TreeConfig(scheme="leisen-reimer")` for European and American vanillas and their Greeks, built from the Peizer-Pratt **method-2** inversion of the binomial tail at `d1` and `d2`. Measured European order 1.9840 at the money and 1.9709-1.9842 at the two off-money points where CRR's constant is erratic, with no oscillation and errors 507x to 5229x smaller than CRR's. Four findings that contradicted the slice's written expectations: the lattice Greeks are still order 1 (the estimators read levels 1 and 2, an `O(dt)` substitution no lattice fixes); the Greek *constants* improve for theta/vega/rho but not for delta/gamma, which land between CRR's two parities; the Leisen-Reimer vega error is flat at 3.06e-03 because `VEGA_BUMP`'s own `O(h**2)` bias is now the binding term; and the American order is 1.06, not 2, with Richardson still failing (1.34, residual 0.72). One real bug found: the shared theta estimator assumed `u d == 1` and does not converge on a Leisen-Reimer lattice without an off-centre correction. Derivation: `docs/notes/leisen_reimer.md`. (source: docs/CURRICULUM.md; .agents/brain/steering-brief.md)
- Slice 4 delivered (Phase 2, first item): `PDEConfig(time_stepping="rannacher")` and Greeks read off the grid (`greeks_method="grid"`), with delta/gamma/theta measured at order ~2 on an aligned grid and four contradicted expectations encoded -- the gamma pathology does *not* appear on the `n_s = n_t = n` refinement path (it needs `dt` large relative to `ds**2`, and on `n_s = 80 n_t` plain Crank-Nicolson fits a **negative** gamma order of -1.043 against Rannacher's 1.933); QuantLib's `FdBlackScholesVanillaEngine` does **not** damp by default and is 180x-1353x out on the same grids; theta from the last two time levels is first order, not second; and four half steps replace the first *two* nominal steps, not one. Also a `perf(pde)` commit moving the tridiagonal solve to `scipy.linalg.solve_banded` (7.4x-28.7x, worst price difference 7.2e-13). Derivation: `docs/notes/pde_greeks_and_rannacher.md`. (source: docs/CURRICULUM.md; .agents/brain/steering-brief.md)
- Next slice: the rest of Phase 2 -- a non-uniform grid concentrated at the strike, PSOR for American exercise, and a digital option. Phase 1's two remaining items are deferred rather than scheduled: a Bermudan instrument (no second case wants one yet) and trinomial trees (no case forces one). (source: docs/CURRICULUM.md)

10) Open questions / risks
- Version sync risk: `pyproject.toml` `[project].version` and `src/qpl/__init__.py:__version__` currently match (`0.1.0`, verified 2026-09-13), but no CI step enforces that they stay aligned — a future bump to one without the other would drift silently. Verify: compare the two values directly. (source: pyproject.toml; src/qpl/__init__.py)

11) Execution Principle: Thin Vertical Slices
- We simply do not build "layers". We build **slices**.
- A slice = Public API + Engine Logic + Test + Golden Path update.
- See `.agents/brain/adr/0002-thin-vertical-slices.md`.
