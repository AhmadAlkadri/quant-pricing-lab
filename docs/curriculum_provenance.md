# Curriculum Provenance (Chapters 1-8)

Labs are private, textbook-driven workbooks under `labs/` (gitignored, never published); this
document is the public record of which private lab chapter drove which public module. It contains
no book prose, only provenance: lab notebook -> `src/qpl` changes -> test coverage.

## Chapter 1 - Static Monte Carlo

- Lab notebook:
  - `labs/01_static_monte_carlo.ipynb`
- New/modified `src/` modules:
  - `src/qpl/engines/mc/sampling.py`
- Key functions/classes added:
  - `lcg_uniform`
  - `stratified_uniform`
  - `inverse_exponential`
  - `sample_exponential_inverse`
  - `sample_beta22_accept_reject`
- New tests added:
  - `tests/test_mc_sampling.py`

## Chapter 2 - Dynamic Monte Carlo

- Lab notebook:
  - `labs/02_dynamic_monte_carlo.ipynb`
- New/modified `src/` modules:
  - `src/qpl/engines/mc/processes.py`
  - `src/qpl/engines/mc/pricers.py` (internal refactor to consume process-level utilities)
- Key functions/classes added:
  - `simulate_gbm_exact`
  - `simulate_gbm_euler`
  - `price_european_from_terminal`
- New tests added:
  - `tests/test_mc_processes.py`

## Chapter 3 - Dynamic Programming For Stochastic Optimization

- Lab notebook:
  - `labs/03_dynamic_programming_stochastic_optimization.ipynb`
- New/modified `src/` modules:
  - `src/qpl/engines/dp/optimal_stopping.py`
  - `src/qpl/engines/dp/american_put_binomial.py`
  - `src/qpl/engines/dp/__init__.py`
- Key functions/classes added:
  - `backward_induction_optimal_stopping`
  - `BinomialDPConfig` (Slice 2: deprecated alias for `TreeConfig`)
  - `build_recombining_spot_tree` (Slice 1: moved to `qpl.engines.tree.lattice`)
  - `price_american_put_binomial` (Slice 2: thin wrapper over
    `qpl.engines.tree.price_american`)
- New tests added:
  - `tests/test_dp_optimal_stopping.py`

## Chapter 4 - Finite Difference Methods

- Lab notebook:
  - `labs/04_finite_difference_methods.ipynb`
- New/modified `src/` modules:
  - No new `src/` module added in this slice.
  - Lab uses existing PDE solver in `src/qpl/engines/pde/pricers.py` (`PDEConfig`, `price_european`).
- Key functions/classes added:
  - None (lab-only slice over existing PDE engine surface)
- New tests added:
  - `tests/test_pde_ch4.py`

### Chapter 4 follow-up - Measured convergence and strike alignment

- Driven by: the Chapter 4 lab's refinement experiment, which showed a
  non-monotone error sequence on the default grid.
- New/modified `src/` modules:
  - `src/qpl/validation/convergence.py`, `src/qpl/validation/benchmark.py`
  - `src/qpl/cases/european_black_scholes.py`
  - `src/qpl/engines/pde/pricers.py` (`PDEConfig.strike_alignment`)
- Key functions/classes added:
  - `fit_convergence_order`, `ConvergenceFit`, `refinement_errors`
  - `EvidenceClass`, `BenchmarkRow`
  - `EuropeanBSSpec`, `EuropeanBSCase` and the European Black-Scholes case lists
- New tests added:
  - `tests/test_validation_convergence.py`
  - `tests/cases/test_european_black_scholes_cases.py`
  - rewritten convergence tests in `tests/test_pde_ch4.py`
- Derivation note (own words, citations only):
  - `docs/notes/pde_strike_alignment.md`

### Phase 1 Slice 1 - Engine registry and the CRR binomial tree

- Driven by: the Phase 1 tree slice, whose fourth engine made the duplicated
  `isinstance` ladder in `qpl.pricing` untenable (ADR-0005).
- New/modified `src/` modules:
  - `src/qpl/engines/registry.py`, `src/qpl/pricing.py`
  - `src/qpl/engines/tree/{__init__,lattice,pricers}.py`
  - `src/qpl/engines/dp/american_put_binomial.py` (now consumes the shared
    lattice builder; prices bit-identical)
  - `src/qpl/cases/european_black_scholes.py` (tree convergence-order rows)
- Key functions/classes added:
  - `MethodSpec`, `register`, `resolve_price`, `resolve_greeks`, `method_spec`
  - `CRRLattice` (Slice 3: renamed `BinomialLattice`, since it is no longer
    CRR-only), `crr_parameters`, `crr_spot_level`, `TreeConfig`,
    `price_european`, `greeks_european`
  - `TREE_ORDER_CASES`, `TREE_REFERENCE_N_STEPS`, `TREE_KNOWN_VALUE_TOLERANCE`
- New tests added:
  - `tests/test_engine_registry.py`, `tests/test_tree_pricing.py`,
    `tests/test_tree_convergence.py`, `tests/oracle/test_tree_vs_quantlib.py`

### Phase 1 Slice 2 - American exercise as an instrument property

- Driven by: the Phase 1 American-exercise slice, which required the Chapter 3
  American-put dynamic program to become reachable from `qpl.pricing`.
- New/modified `src/` modules:
  - `src/qpl/instruments/options.py` (`VanillaOption`, `AmericanOption`)
  - `src/qpl/engines/tree/american.py`, `src/qpl/engines/tree/pricers.py`
  - `src/qpl/pricing.py` (second registry key on `method="tree"`)
  - `src/qpl/engines/dp/american_put_binomial.py` (now a thin deprecated
    wrapper over the dispatcher engine; values bit-identical)
  - `src/qpl/cases/american_black_scholes.py`
- Key functions/classes added:
  - `VanillaOption`, `AmericanOption`
  - `price_american`, `greeks_american`, `lattice_delta_gamma_theta`
  - `AmericanBSSpec`, `AmericanBSCase`, `ALL_AMERICAN_CASES` and the four
    American case lists
- New tests added:
  - `tests/test_tree_american.py`, `tests/test_tree_american_convergence.py`,
    `tests/cases/test_american_black_scholes_cases.py`,
    `tests/oracle/test_american_vs_quantlib.py`
- Derivation note (own words, citations only):
  - `docs/notes/american_exercise_on_trees.md`
- Derivation note (own words, citations only):
  - `docs/notes/crr_tree_convergence.md` (Cox/Ross/Rubinstein 1979 for the
    lattice and the replication argument; Leisen & Reimer 1996 for the order-1
    result and the oscillation)

### Phase 1 Slice 3 - Leisen-Reimer, and what order 2 does not buy

- Driven by: the Phase 1 Leisen-Reimer slice, motivated by Slice 2's finding
  that Richardson extrapolation does not restore order 2 for an American put.
- New/modified `src/` modules:
  - `src/qpl/engines/tree/lattice.py` (the Peizer-Pratt inversion, the
    Leisen-Reimer parameters, the scheme dispatcher; `CRRLattice` renamed
    `BinomialLattice` and given `spot_centred`)
  - `src/qpl/engines/tree/pricers.py` (`TreeConfig.scheme` widened; odd-`n`
    validation; the off-centre theta correction)
  - `src/qpl/engines/tree/american.py`, `src/qpl/engines/tree/__init__.py`
  - `src/qpl/cases/european_black_scholes.py`,
    `src/qpl/cases/american_black_scholes.py`
- Key functions/classes added:
  - `peizer_pratt_inversion`, `leisen_reimer_parameters`, `lattice_parameters`
  - `BinomialLattice`, `Scheme`, `SCHEMES`
  - `TREE_LR_ORDER_CASES`, `TREE_LR_REFERENCE_N_STEPS`,
    `TREE_LR_KNOWN_VALUE_TOLERANCE`, `AMERICAN_LR_CASES`,
    `AMERICAN_LR_LEVELS`, `AMERICAN_BRACKETED_LIMIT`
- New tests added:
  - `tests/test_tree_leisen_reimer.py`, `tests/test_tree_lr_convergence.py`,
    `tests/oracle/test_lr_vs_quantlib.py`
- Derivation note (own words, citations only):
  - `docs/notes/leisen_reimer.md` (Leisen & Reimer 1996 for the construction
    and the odd-`n` requirement; Peizer & Pratt 1968 for the tail
    approximation that is inverted). The formulas are restated from the
    construction, and every number in the note was measured in this
    repository.

## Chapter 5 - Numerical Solution Of Linear Systems

- Lab notebook:
  - `labs/05_numerical_linear_systems.ipynb`
- New/modified `src/` modules:
  - `src/qpl/numerics/linear_systems.py`
  - `src/qpl/numerics/__init__.py`
- Key functions/classes added:
  - `LinearSolveResult`
  - `jacobi_solve`
  - `gauss_seidel_solve`
  - `sor_solve`
- New tests added:
  - `tests/test_linear_systems_ch5.py`

## Chapter 6 - Quadrature Methods

- Lab notebook:
  - `labs/06_quadrature_methods.ipynb`
- New/modified `src/` modules:
  - `src/qpl/numerics/quadrature.py`
  - `src/qpl/numerics/__init__.py`
- Key functions/classes added:
  - `composite_trapezoid`
  - `composite_simpson`
  - `gauss_legendre`
- New tests added:
  - `tests/test_quadrature_ch6.py`

## Chapter 7 - Laplace Transform

- Lab notebook:
  - `labs/07_laplace_transform_inversion.ipynb`
- New/modified `src/` modules:
  - `src/qpl/transforms/laplace.py`
  - `src/qpl/transforms/__init__.py`
- Key functions/classes added:
  - `stehfest_coefficients`
  - `inverse_laplace_stehfest`
  - `inverse_laplace_grid_stehfest`
- New tests added:
  - `tests/test_laplace_ch7.py`

## Chapter 8 - Copula Functions

- Lab notebook:
  - `labs/08_copula_functions.ipynb`
- New/modified `src/` modules:
  - `src/qpl/dependence/copulas.py`
  - `src/qpl/dependence/__init__.py`
- Key functions/classes added:
  - `gaussian_copula_sample`
  - `empirical_kendall_tau`
  - `empirical_spearman_rho`
  - `gaussian_copula_kendall_tau`
  - `gaussian_copula_spearman_rho`
- New tests added:
  - `tests/test_copula_ch8.py`

## Cross-Chapter Smoke Coverage

- `tests/test_labs_smoke.py` executes:
  - `01_static_monte_carlo.ipynb`
  - `02_dynamic_monte_carlo.ipynb`
  - `03_dynamic_programming_stochastic_optimization.ipynb`
  - `04_finite_difference_methods.ipynb`
  - `05_numerical_linear_systems.ipynb`
  - `06_quadrature_methods.ipynb`
  - `07_laplace_transform_inversion.ipynb`
  - `08_copula_functions.ipynb`
