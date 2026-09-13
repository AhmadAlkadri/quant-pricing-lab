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
- Purpose: a numerical-methods lab built by Textbook-Driven Development (ADR-0004): textbook/literature results are re-derived, turned into cases, cross-checked, and backed by measured convergence or statistical evidence before becoming package capability. Covers analytic/MC/PDE European pricing under Black-Scholes, a CRR binomial tree covering European and American exercise, and supporting numerics/transforms/dependence modules. (source: docs/CURRICULUM.md; src/qpl/engines/analytic/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py; src/qpl/engines/tree/american.py; src/qpl/numerics/__init__.py; src/qpl/transforms/__init__.py; src/qpl/dependence/__init__.py)
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
- Current example-level interfaces: `qpl.engines.pde.pricers.PDEConfig`, `price_european`. (source: src/qpl/engines/pde/pricers.py)
- Current tree engine exports: `qpl.engines.tree` exports `TreeConfig`, `CRRLattice`, `crr_parameters`, `crr_spot_level`, `build_recombining_spot_tree`, `price_european`, `greeks_european`, `price_american`, `TREE_METHOD_SPEC`; reachable from the dispatcher as `method="tree"`. Registry keys for `method="tree"`: `(EuropeanOption, BlackScholesModel, "tree")` for price and Greeks, and `(AmericanOption, BlackScholesModel, "tree")` for price. The analytic, MC and PDE engines are registered for `EuropeanOption` only, so an `AmericanOption` sent to them raises `NotSupportedError` through the ordinary lookup rather than through a per-engine guard. (source: src/qpl/engines/tree/__init__.py; src/qpl/pricing.py)
- `PDEConfig` has a `strike_alignment: Literal["none", "midpoint"]` field; `"midpoint"` nudges the grid spacing so the strike sits exactly between two nodes, restoring measured order-2 convergence (see `docs/notes/pde_strike_alignment.md`). Default `"none"` is bit-identical to pre-Slice-0 output. (source: src/qpl/engines/pde/pricers.py)
- Current validation exports: `qpl.validation` exports `ConvergenceFit`, `fit_convergence_order`, `refinement_errors`, `BenchmarkRow`, `EvidenceClass`. (source: src/qpl/validation/__init__.py)
- Current cases exports: `qpl.cases` exports `EuropeanBSSpec`, `EuropeanBSCase`, `PARITY_CASES`, `LIMIT_CASES`, `KNOWN_VALUE_CASES`, `TREE_ORDER_CASES`, `MONOTONICITY_CASES`, `ALL_CASES`, `parity_residual`, plus the tree study constants `TREE_ODD_LEVELS`, `TREE_EVEN_LEVELS`, `TREE_REFERENCE_N_STEPS`, `TREE_KNOWN_VALUE_TOLERANCE`. (source: src/qpl/cases/__init__.py)
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
- Engines: analytic uses closed-form BS, MC uses GBM sampling (terminal or multi-step), PDE uses theta-scheme FD grid, tree uses a CRR recombining lattice with vectorized backward induction. The CRR lattice builder in `qpl.engines.tree.lattice` is shared with the American-put DP engine. (source: src/qpl/engines/analytic/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py; src/qpl/engines/tree/pricers.py; src/qpl/engines/tree/lattice.py)
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
- PDE config requires n_s >= 3, n_t >= 1, theta in [0,1]; deterministic for fixed inputs. (source: src/qpl/engines/pde/pricers.py; tests/test_pde_pricing.py)
- Tree config requires n_steps >= 1 for prices and >= 2 for Greeks (gamma and theta read step-2 nodes), and scheme == "crr"; the CRR lattice raises `InvalidInputError` when no-arbitrage fails. Deterministic for fixed inputs. (source: src/qpl/engines/tree/pricers.py; src/qpl/engines/tree/lattice.py; tests/test_tree_pricing.py)
- At T=0 or sigma=0, pricing returns intrinsic or discounted-forward intrinsic (analytic/MC/PDE/tree). (source: src/qpl/models/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py; tests/test_pricing_analytic.py; tests/test_mc_pricing.py)
- Units: T in years; r and q are continuously compounded. (source: src/qpl/models/black_scholes.py; src/qpl/market/curves.py)

5) Error handling & validation policy
- All domain validation errors raise `InvalidInputError`; unsupported combos raise `NotSupportedError`. (source: src/qpl/exceptions.py; src/qpl/pricing.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py; src/qpl/instruments/options.py)
- Validation happens at object construction and at engine entry points, not via return codes. (source: src/qpl/instruments/options.py; src/qpl/market/market.py; src/qpl/models/black_scholes.py; src/qpl/engines/mc/pricers.py; src/qpl/engines/pde/pricers.py)
- `ModelAssumptionError` exists as a typed hook but is currently unused. (source: src/qpl/exceptions.py)

6) Configuration & defaults
- MC defaults: `MCConfig(n_paths=50_000, n_steps=1, seed=123)`. (source: src/qpl/engines/mc/pricers.py)
- PDE defaults: `PDEConfig(n_s=200, n_t=200, theta=0.5, s_max=None, s_max_multiplier=4.0)`. (source: src/qpl/engines/pde/pricers.py)
- Tree defaults: `TreeConfig(n_steps=200, scheme="crr")`; tree Greek bumps `VEGA_BUMP=1e-2`, `RHO_BUMP=1e-4`, both chosen from a measured scan. (source: src/qpl/engines/tree/pricers.py)
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
- Next slice: the rest of Phase 1 — Leisen-Reimer smoothing checked by measured order 2, American exercise as an instrument property (which is what would finally put `engines.dp` on the dispatcher), and tree Greeks for American payoffs. (source: docs/CURRICULUM.md)

10) Open questions / risks
- Version sync risk: `pyproject.toml` `[project].version` and `src/qpl/__init__.py:__version__` currently match (`0.1.0`, verified 2026-09-13), but no CI step enforces that they stay aligned — a future bump to one without the other would drift silently. Verify: compare the two values directly. (source: pyproject.toml; src/qpl/__init__.py)

11) Execution Principle: Thin Vertical Slices
- We simply do not build "layers". We build **slices**.
- A slice = Public API + Engine Logic + Test + Golden Path update.
- See `.agents/brain/adr/0002-thin-vertical-slices.md`.
