# Steering Brief

What changed in Slice 2 (files + bullets)

The rest of Phase 1's American half: exercise style becomes an instrument
property, and the CRR tree learns to price it. Still on `dev/curriculum`; not
pushed.

- `src/qpl/instruments/options.py`, `src/qpl/instruments/__init__.py`,
  `src/qpl/pricing.py`, `tests/test_tree_american.py`:
  - `VanillaOption(kind, strike, expiry)` carries the validation;
    `EuropeanOption` and `AmericanOption` are **siblings** under it, not parent
    and child. That is load-bearing: registry lookup walks the MRO, so an
    `AmericanOption` subclassing `EuropeanOption` would resolve to the European
    closed form and return a plausible wrong number instead of refusing.
    Pinned by a test.
  - The tree registers a second key, `(AmericanOption, BlackScholesModel,
    "tree")`, for price and Greeks. Analytic/MC/PDE stay European-only and
    refuse an `AmericanOption` through the ordinary lookup with the ordinary
    message — no engine grows an `if instrument.american: raise`. Tested for
    all three, for both `price` and `greeks`.
  - No change to the ADR-0005 contract (this is an application of its key), so
    **no ADR-0006**; brain.md sections 2/3/4 updated instead.
- `src/qpl/engines/tree/american.py`, `src/qpl/engines/tree/pricers.py`,
  `src/qpl/engines/dp/*`:
  - `price_american` / `greeks_american`: calls and puts, continuous yield,
    `meta["exercise_boundary"]` per level and `meta["early_exercise_node_count"]`,
    and delta/gamma/theta through `lattice_delta_gamma_theta`, now shared with
    the European path (same code, because the estimators do not care how the
    node values were produced). Vega/rho by bump at the same `n`.
  - Spot levels come from precomputed `u**i`/`d**i` tables: `O(n)` calls to
    `pow` instead of `O(n**2)`, same floating-point nodes, which is what makes
    an `n = 8001` reference cost 0.17 s.
  - `engines.dp.price_american_put_binomial` is now a **thin wrapper** over the
    dispatcher engine (kept rather than deleted because a private lab notebook
    calls it with `return_lattice=True`, and `labs/` is not mine to edit). Five
    Slice 1 values still pass with a **zero** tolerance.
- `tests/test_tree_american.py` (60 tests), `tests/test_tree_american_convergence.py`
  (14 tests, 1.3 s), `docs/notes/american_exercise_on_trees.md`:
  - **Measured** (S=K=100, r=5%, q=0, sigma=20%, T=1, American put, reference
    = this engine at n=8001): odd `n` order **1.0141** (residual 0.0174),
    above the reference at every level; even `n` **0.9722** (0.0242), below at
    every level. **The odd/even bracketing survives early exercise.** American
    call with q=6%: **1.0232** / **0.9674**.
  - The scaled constants do **not** settle like the European tree's 1.7529 /
    1.9994 — they drift (odd 1.391→1.313, even 0.822→0.924). Partly the
    reference's own 1.80e-04 error (quantified against a 64000/64001 average),
    partly real: the boundary is resolved only to the node spacing, an error
    with no parity structure. So the test pins the *sign structure*, not a
    constant.
  - **Richardson extrapolation does not restore order 2** for the American
    put: **0.2960** (residual 0.2712) odd, **0.6447** (0.3562) even, with the
    extrapolated error flattening at ~1.5e-04. It still cuts the error 10-220x
    against the raw sequence; both halves are asserted. Verified against the
    better reference that the floor is not just the reference's accuracy.
  - Lattice Greeks at order ~1 (delta/gamma/theta 1.164/1.185/1.259 odd,
    1.074/1.079/1.076 even), band [0.9, 1.4] with the reason stated.
  - Exact identities with **zero** tolerance: American call = European call bit
    for bit at q=0 (all five Greeks too), American put = European put at r=0.
- `src/qpl/cases/american_black_scholes.py`,
  `tests/cases/test_american_black_scholes_cases.py`:
  - **Contradicted expectation, and the best result of the slice.** The slice
    required the L&S (2001) Table 1 row-1 put at n=5000 within 2e-03 of
    **4.478**. Measured: **4.486710**, 8.71e-03 away. QuantLib's FD and
    binomial American engines land at ~4.4867 too, so the engine is not wrong
    — the *instrument* is different. Those options are exercisable **50 times
    per year**; restricting exercise to 50 dates on the same lattice gives
    **4.477922** (n=5000) and 4.477826 (n=40000). Carried as two rows on one
    spec: `PUBLISHED_BENCHMARK` at tolerance 5e-04 (the published figure's own
    three-decimal precision) and `NEGATIVE_FINDING` asserting the continuous
    gap *exceeds* 2e-03, pinned at 8.71e-03 ± 5e-05.
  - Nine rows total; the in-repo reference 6.0905564143067235 at n=8001 says in
    its own `source` that it is **not** a published table value.
- Two further contradictions, encoded rather than smoothed over:
  - At sigma=0 the American put is **not** always
    `max(intrinsic now, discounted forward intrinsic)` — that needs `r >= q`.
    For `q > r` the objective has an interior maximum at
    `t* = log(rK/(qS0))/(r−q)`; measured 83.9506 vs 81.1993 at
    S0=K=100, r=2%, q=50%, T=10.
  - The extracted boundary is **not** monotone level by level; it is monotone
    within each node-grid parity and zigzags by at most one grid offset
    (worst drop 1.3845 vs offset 1.4043 at n=200). At expiry it is the
    in-the-money node adjacent to K: `K/u²=97.2112` (put), `Ku²=102.8688`
    (call).
- `tests/oracle/test_american_vs_quantlib.py`:
  - `FdBlackScholesVanillaEngine` (tGrid=xGrid=3200) agrees with the tree
    (n=8001) to **3.71e-04** / 1.20e-04 / 3.10e-04 at three points, inside a
    1e-03 tolerance **derived** from both refinements — they approach from
    opposite sides, so the gap is the sum of their errors. QuantLib's FD order,
    fitted on successive differences so no reference value is needed:
    **1.0856** (residual 0.0211).
  - Slice 1's negative finding survives early exercise: `n·(qpl − QuantLib
    CRR)` constant to better than 0.5% over n∈{200,800,3200} at **−0.026413**
    (ATM put), **−0.013260** (L&S row 1), **+0.007104** (call with q=6%).
- `examples/american_put_binomial_dp.py`, `tests/test_examples_smoke.py`,
  `docs/CURRICULUM.md`, `docs/curriculum_provenance.md`,
  `docs-site/docs/reference/dp.md`, `README.md`, `.agents/brain/brain.md`:
  - The example now goes through the dispatcher and prints the premium and
    boundary samples; smoke keys extended. Curriculum "Delivered" gained Slice
    2 and Phase 1's remaining items were re-scoped (Leisen-Reimer is now
    better motivated: Richardson does not substitute for it on American
    payoffs).

Deferred (intentional, Slice 2)
- Leisen-Reimer, PSOR and Longstaff-Schwartz are later slices, as instructed.
- No Bermudan instrument: the 50-date restriction that reproduces the L&S
  benchmark lives in the test, on the shared lattice. If a second case wants
  it, that is when it earns a type.
- `engines.dp.price_american_put_binomial` kept as a deprecated wrapper rather
  than deleted, only because of the private lab notebook. It can go the moment
  that notebook is updated.

How to validate Slice 2 quickly
- `PYTHONPATH=src python examples/american_put_binomial_dp.py`
- `pytest -q tests/test_tree_american.py tests/test_tree_american_convergence.py tests/cases/test_american_black_scholes_cases.py`
- `pytest -q tests/oracle` (needs the `oracle` extra; skips cleanly without it)

---

What changed in Slice 1 (files + bullets)

Phase 1 of the curriculum: the CRR binomial tree, and the dispatcher refactor
it forced. Still on `dev/curriculum`; not pushed.

- `src/qpl/engines/registry.py`, `src/qpl/pricing.py`,
  `src/qpl/engines/{analytic,mc,pde}/*`, `tests/test_engine_registry.py`,
  `.agents/brain/adr/0005-engine-registry.md`:
  - Replaced the `isinstance` ladder (one branch per method, duplicated across
    `price` and `greeks`) with a registry keyed by
    `(instrument type, model type, method)` plus a per-method `MethodSpec`
    carrying the keyword contract (`cfg` type, MC's optional `bumps` and its
    validator). Public signatures, error types, error messages and the
    validation ordering are all unchanged and now pinned by a test. Lookup
    walks the MRO, preserving `isinstance` semantics. `Market` is type-checked
    but is not part of the key; ADR-0005 says why. Adding an engine is now one
    `register(...)` call.
- `src/qpl/engines/tree/{__init__,lattice,pricers}.py`,
  `src/qpl/engines/dp/american_put_binomial.py`, `tests/test_tree_pricing.py`:
  - Added `qpl.engines.tree`: `crr_parameters` (with the no-arbitrage check),
    `crr_spot_level`, `build_recombining_spot_tree` (moved out of
    `engines.dp`), `TreeConfig(n_steps=200, scheme="crr")`, `price_european`
    (vectorized backward induction, one numpy expression per time level) and
    `greeks_european` (delta/gamma/theta off the lattice, vega/rho by bump).
    Wired in as `method="tree"`. T=0 and sigma=0 behave as in the other
    engines.
  - The American-put DP engine now consumes the shared lattice; prices are
    **bit-identical** across 45 configurations, five of them pinned with a
    zero tolerance.
- `tests/test_tree_convergence.py`, `docs/notes/crr_tree_convergence.md`,
  `examples/tree_convergence.py`:
  - **Measured** (S=K=100, r=5%, q=0, sigma=20%, T=1, European call): odd `n`
    fitted order **1.0010** (residual 0.0005), tree **above** BS, constant
    `n·|err| -> 1.7529`; even `n` order **0.9987** (residual 0.0007), tree
    **below** BS, constant **1.9994**. The two subsequences bracket the true
    value — the textbook story held, and is asserted as measured rather than
    assumed. Richardson on consecutive odd pairs: order **1.9590** (residual
    0.0224), *not* 2, and the note says why; error at `n=401` drops from
    4.372e-03 to 5.710e-07.
  - Lattice Greeks converge at order ~1 too (delta/gamma/theta 1.004/1.002/
    1.001 odd, 0.999/1.010/1.010 even). Vega and rho carry the oscillation;
    bump sizes chosen from a measured scan, not habit.
  - One-step and two-step trees match hand-computed *replication* values
    (max residual 2.3e-14); put-call parity holds on the tree at every `n`
    (8.3e-12 at n=800) because `p` reprices the forward exactly.
- `tests/oracle/test_tree_vs_quantlib.py`:
  - **Contradicted expectation.** The slice expected 1e-10 agreement with
    QuantLib's `BinomialVanillaEngine<CoxRossRubinstein>`. QuantLib's
    `CoxRossRubinstein` uses the *log-space* probability
    `1/2 + (r−q−σ²/2)dt/(2σ√dt)` on the same lattice, not the
    forward-matching `p = (e^{(r−q)dt}−d)/(u−d)`. The two differ at
    `O(dt^1.5)` per step, i.e. `O(1/n)` in price: `n·(qpl − QuantLib)` is
    constant to better than 1% over `n` in {50, 200, 800}. A twelve-line
    reimplementation of QuantLib's probability reproduces its engine to
    8.3e-11, identifying the mechanism. Our closed form does match
    `AnalyticEuropeanEngine` to 1.1e-14.
  - Second negative finding: away from the money the order-1 constant is
    erratic (fitted order 1.21-1.48, log residual 0.37-1.52) for *both*
    engines, so the clean odd/even story is an at-the-money story.
- `src/qpl/cases/european_black_scholes.py`,
  `tests/cases/test_european_black_scholes_cases.py`:
  - Two `CONVERGENCE_ORDER` rows for the odd/even claims, plus
    `TREE_REFERENCE_N_STEPS=2000` and `TREE_KNOWN_VALUE_TOLERANCE=2.5e-3`
    derived from the measured constant (predicted 1.00e-3, measured 9.998e-04).
    The cross-engine test now has a fourth leg.
- `examples/README.md`, `tests/test_examples_smoke.py`,
  `docs/CURRICULUM.md`, `.agents/brain/brain.md`:
  - `examples/tree_convergence.py` added to the curated set and to both halves
    of the smoke harness; curriculum "Delivered" gained a Slice 1 entry with
    the measured numbers; brain.md sections 2/3/4/8 reconciled.

Deferred (intentional, Slice 1)
- Leisen-Reimer, American exercise on the European tree, and trinomial trees
  are later slices; `TreeConfig.scheme` exists so adding LR does not change
  this config's identity.
- `qpl.engines.dp.price_american_put_binomial` is still not on the dispatcher:
  it takes `strike`/`expiry` rather than an instrument, and American exercise
  becomes an instrument property in a later Phase 1 slice.

How to validate Slice 1 quickly
- `PYTHONPATH=src python examples/tree_convergence.py`
- `pytest -q tests/test_tree_convergence.py tests/test_tree_pricing.py tests/test_engine_registry.py`
- `pytest -q tests/oracle` (needs the `oracle` extra; skips cleanly without it)

---

What changed in Slice 0 (files + bullets)

This is the first slice of the Textbook-Driven Development curriculum
campaign (ADR-0004); none of it has reached `main` yet (`git log
main..dev/curriculum`), which will be fast-forwarded by the project owner at
the slice boundary.

- `AGENTS.md`, `.agents/brain/brain.md`, `.agents/brain/adr/0003-*.md`,
  `.agents/skills/**`, `.github/workflows/deploy-docs.yml`, `.gitignore`:
  - Adopted the lab-driven pre-1.0 policy (ADR-0003): `labs/` is gitignored
    and private, never public cargo.
- `src/qpl/engines/mc/processes.py`, `src/qpl/engines/mc/sampling.py`,
  `src/qpl/engines/mc/pricers.py`, `src/qpl/engines/pde/pricers.py`,
  `src/qpl/pricing.py`, `src/qpl/market/market.py`, `src/qpl/market/stats.py`,
  `src/qpl/utils/labs.py`:
  - Added exact/Euler GBM simulators, sampling primitives, MC Theta, PDE
    Delta/Gamma, an implied-volatility solver
    (`qpl.engines.analytic.black_scholes.implied_volatility`), and
    smoke-mode lab utilities.
- `src/qpl/engines/dp/*`, `src/qpl/numerics/*`, `src/qpl/transforms/*`,
  `src/qpl/dependence/*`:
  - Added the numerical-methods modules the private labs forced into
    existence: backward-induction optimal stopping and a CRR American put
    binomial pricer (`engines.dp`), Jacobi/Gauss-Seidel/SOR and
    trapezoid/Simpson/Gauss-Legendre (`numerics`), Stehfest Laplace inversion
    (`transforms`), and Gaussian copula sampling/concordance (`dependence`).
- `examples/*`, `notebooks/*`, `docs-site/**`, `CHANGELOG.md`,
  `tests/test_examples_smoke.py`, `tests/test_labs_smoke.py`,
  `tests/test_notebooks_smoke.py`:
  - Published the v0.2.0 public learning surface: example scripts, overview
    notebooks, a VitePress docs-site skeleton, and headless smoke tests for
    examples/notebooks/labs, including a clean skip (not a failure) when
    `labs/` is absent.
- `README.md`, `CHANGELOG.md`, `docs-site/docs/guide/design-philosophy.md`,
  `notebooks/README.md`, `docs/curriculum_provenance.md` (renamed from
  `docs/labs_to_library_map.md`):
  - Replaced "withheld pending legal review" language with the plain
    statement: `labs/` is private by design, gitignored, never published;
    public cargo is the package, tests, the cases layer, and derivation
    notes.
- `pyproject.toml`, `src/qpl/market/data.py`, `src/qpl/market/stats.py`,
  `tests/oracle/*`, `tests/test_market_data.py`, `tests/test_market_stats.py`:
  - Split optional extras out of core: `dev` (unchanged), `data`
    (pandas/yfinance/pyarrow), `oracle` (QuantLib, an independent pricing
    oracle per D3, never sole evidence). Core dependencies shrink to
    numpy/scipy/matplotlib. Optional imports are guarded at call time with a
    clear `NotSupportedError` + install hint; `tests/oracle/` collects only
    when QuantLib is importable, via `pytest_ignore_collect` (a module-level
    `importorskip` would hard-fail when the directory is targeted directly).
- `.github/workflows/ci.yml`:
  - Split into `tests-core` (`pip install -e ".[dev]"`) and `tests-full`
    (`pip install -e ".[dev,data,oracle]"`); both run `ruff check .` then
    `pytest -q`.
- `src/qpl/validation/__init__.py`, `src/qpl/validation/benchmark.py`,
  `src/qpl/validation/convergence.py`, `tests/test_validation_convergence.py`:
  - Added `qpl.validation`: `fit_convergence_order` (OLS slope of `log(err)`
    on `log(h)`, reporting order + log-space RMS residual — chosen over
    R-squared, which saturates near 1 for any plausible power law) and the
    `EvidenceClass`/`BenchmarkRow` taxonomy (exact identity, closed form,
    published benchmark, independent engine, convergence order, statistical,
    negative finding).
- `src/qpl/engines/pde/pricers.py`, `tests/test_pde_pricing.py`,
  `tests/test_pde_ch4.py`:
  - Added `PDEConfig.strike_alignment` (`"none"` default, bit-identical to
    prior output; `"midpoint"` nudges grid spacing so the strike sits exactly
    between two nodes). Replaced a grid-lucky monotone-decay check with three
    tests that name their evidence class: a pinned negative finding for the
    unaligned pathology, a measured convergence-order test for the aligned
    grid, and one isolating temporal order.
  - **Measured** (S=K=100, r=5%, q=0, sigma=20%, T=1, European call,
    Crank-Nicolson, `n_s=n_t=n` for `n` in `{50,100,200,400,800}`): unaligned
    fitted order **1.126** (log-residual **0.860**, non-monotone — error at
    `n=100` is 5x worse than at `n=50` because the strike lands exactly on a
    node); aligned fitted order **1.997** (residual **0.022**). Isolated
    implicit-Euler temporal order (`n_s=800` aligned, fixed, refining `n_t`):
    fitted order **0.997** (residual **3e-04**), with the spatial error floor
    measured at 1.95e-04, 27x below the smallest temporal error in the fit.
- `src/qpl/cases/__init__.py`, `src/qpl/cases/european_black_scholes.py`,
  `tests/cases/test_european_black_scholes_cases.py`:
  - Added `qpl.cases.european_black_scholes`: 18 benchmark rows across parity
    (exact identity), degenerate limits (closed form), a reference ATM
    call/put value (closed form, in-repo, not published), and comparative
    statics (exact identity, ordering). A cross-engine test prices each
    known-value row via analytic, PDE, and MC with per-engine tolerances
    justified against measurements. Superseded five redundant assertions in
    `tests/test_black_scholes_analytic.py` and `tests/test_pricing_analytic.py`.
- `docs/notes/pde_strike_alignment.md`, `docs/curriculum_provenance.md`:
  - Added the public derivation note for strike alignment (own words,
    citations to Pooley/Forsyth/Vetzal 2003 and Tavella/Randall 2000; no book
    prose/code/tables copied) and recorded it as a Chapter 4 follow-up.
- `docs/CURRICULUM.md`, `docs/ROADMAP.md`, `.agents/brain/adr/0004-*.md`,
  `.agents/brain/brain.md`:
  - Added the curriculum map (TDD loop, identity decision, literature spine,
    method dependency graph, evidence classes, provenance rules, phase plan,
    Slice 0 results, reconciled old roadmap); retired `docs/ROADMAP.md` to a
    pointer; recorded ADR-0004; reconciled `brain.md` sections 0-3, 7, 9-10 to
    match current reality.

What got better
- Numerical claims now state which evidence class justifies them, instead of
  an unlabelled tolerance in a test.
- The strike-on-a-node PDE pathology is measured, documented, and fixed
  (opt-in via `PDEConfig.strike_alignment="midpoint"`), not just tested
  against grids that happened to avoid it.
- `labs/` privacy is now stated as an intentional design choice, not a
  pending-review placeholder.
- Core install no longer pulls in pandas/yfinance/pyarrow/QuantLib; CI
  exercises both the minimal and fully-equipped install paths.
- The public curriculum has an importable, tested cases layer
  (`qpl.cases`) rather than only ad hoc assertions scattered across test
  files.

Deferred (intentional)
- `engines.dp` (American put) is not yet wired into the `qpl.pricing`
  dispatcher; that is Phase 1, alongside the `isinstance`-ladder-to-registry
  refactor (ADR-0005, pending).
- Rannacher start-up and non-uniform strike-concentrated grids (the other two
  standard remedies for the PDE strike pathology) are Phase 2.
- Market-data/historical-vol/implied-vs-realized work is done but frozen
  behind the `[data]` extra, outside this curriculum (D9).
- Automatic API reference generation and docs-site CI deployment wiring.

How to validate quickly
- `ruff check .`
- `pytest -q`
- `QPL_LAB_SMOKE=1 MPLBACKEND=Agg pytest -q tests/test_labs_smoke.py`
- `PYTHONPATH=src MPLBACKEND=Agg QPL_LAB_SMOKE=1 pytest -q tests/test_notebooks_smoke.py`
- `pip install -e ".[dev,data,oracle]" && pytest -q` to exercise the full-extras path (`tests/oracle/` included).
