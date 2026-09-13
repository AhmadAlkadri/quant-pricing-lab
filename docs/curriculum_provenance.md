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

### Phase 2 Slice 4 - Rannacher start-up and Greeks from the grid

- Driven by: the Chapter 4 follow-up's own closing paragraph, which recorded
  that strike alignment fixes the price but not the Greeks and named Rannacher
  time stepping as the missing remedy.
- New/modified `src/` modules:
  - `src/qpl/engines/pde/pricers.py` (`PDEConfig.time_stepping`,
    `PDEConfig.greeks_method`, `_time_levels`, `_solve_grid`, a rewritten
    `greeks_european`, and the tridiagonal solve moved to
    `scipy.linalg.solve_banded`)
  - `src/qpl/cases/european_black_scholes.py` (`PDE_GREEK_CASES` and the grid
    constants; `REFERENCE_ATM_CALL` / `REFERENCE_ATM_PUT` exported)
- Key functions/classes added:
  - `RANNACHER_STARTUP_STEPS`, `GRID_VEGA_BUMP`, `GRID_RHO_BUMP`,
    `BUMP_SPOT_FRACTION`
  - `PDE_GREEK_CASES`, `PDE_GREEKS_N`, `PDE_GREEKS_TIME_STEPPING`,
    `PDE_GREEKS_STRIKE_ALIGNMENT`
- New tests added:
  - `tests/test_pde_greeks.py` (rewritten from the ground up),
    `tests/oracle/test_pde_vs_quantlib.py`, and new cases in
    `tests/test_pde_ch4.py`, `tests/test_pde_pricing.py`,
    `tests/cases/test_european_black_scholes_cases.py`
- Example rewritten:
  - `examples/pde_greeks_demo.py` (`--case smooth`, `--case startup`), added to
    the curated smoke list
- Derivation note (own words, citations only):
  - `docs/notes/pde_greeks_and_rannacher.md` (Rannacher 1984 for the
    construction; Giles & Carter 2006 for the damping/accuracy trade-off;
    Pooley-Forsyth-Vetzal 2003 and Tavella & Randall 2000 for the
    non-smooth-payoff context and the stencils). The amplification-factor
    argument is restated from `R(z) = (1 + (1-theta)z)/(1 - theta z)`, and
    every number in the note was measured in this repository.

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

### Chapter 6 follow-up - Phase 4 Slice 14: the quadrature toolkit reaches the pricing core

- Driven by: Phase 4's first item. Chapter 6 §6.8 ("Pricing using characteristic
  functions") is the map of what the private lab covered; the four methods below
  are derived independently in `src/qpl/engines/fourier/*.py` and in
  `docs/notes/fourier_pricing_methods.md`, from the sources cited there
  (Carr & Madan 1999; Fang & Oosterlee 2008; Gil-Pelaez 1951; Heston 1993 for
  the `Pi_1`/`Pi_2` form; Lewis 2001; Schmelzle 2010 as a survey).
- **What closed the gap.** Until this slice the Chapter 6 rules
  (`composite_trapezoid`, `composite_simpson`, `gauss_legendre`) had only ever
  been measured on smooth finite-interval integrands, where they are order 2,
  order 4 and spectrally accurate. The Carr-Madan direct-quadrature variant is
  the first time they compute a price, and on that integrand -- real, even,
  analytic, Gaussian-decaying -- none of the three converges at its nominal
  order: the trapezoid rule is spectrally accurate because every
  Euler-Maclaurin boundary term vanishes, and composite Simpson, being exactly
  `(4 T_n - T_{n/2}) / 3`, is six decimal orders of magnitude *worse* than
  trapezoid at the same evaluation count. The lab's own smooth-function test
  could not have shown either.
- New/modified `src/` modules:
  - `src/qpl/engines/fourier/{__init__,charfn,cos,carr_madan,lewis,gil_pelaez,pricers,digital}.py`
  - `src/qpl/pricing.py` (two `register(...)` calls, `method="fourier"`)
  - `src/qpl/cases/european_black_scholes.py`,
    `src/qpl/cases/digital_black_scholes.py` (transform legs)
- Key functions/classes added:
  - `CharacteristicFunctionModel`, `LogReturnCumulants`,
    `characteristic_function_model`, `register_characteristic_function`,
    `black_scholes_characteristic_function`,
    `black_scholes_log_return_cumulants`, `BlackScholesCharacteristicFunction`
  - `FourierConfig`, `FOURIER_METHOD_SPEC`, `price_european`, `greeks_european`,
    `price_digital`, `greeks_digital`
  - `cos_price`, `cos_truncation_range`, `chi_coefficients`, `psi_coefficients`
  - `carr_madan_fft`, `carr_madan_quadrature`, `damped_call_transform`,
    `default_u_max`, `CarrMadanGrid`
  - `lewis_call`, `gil_pelaez_probabilities`, `GilPelaezProbabilities`
- New tests added:
  - `tests/test_fourier_cos.py`, `tests/test_fourier_carr_madan.py`,
    `tests/test_fourier_lewis_gil_pelaez.py`,
    `tests/oracle/test_fourier_vs_quantlib.py`, `tests/fourier_points.py`
    (shared specification points, not a test module)
- Derivation note (own words, citations only):
  - `docs/notes/fourier_pricing_methods.md`

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
- **Still not reached by the pricing core.** Chapter 6's quadrature rules met a
  price in Slice 14; the Stehfest inversion has not. Its intended case is
  Phase 4's last item, the Fusai & Roncoroni chapter 15 Laplace approach to
  arithmetic Asians, where the transform of the average's law is known and the
  price is an inversion rather than a simulation. Until that slice,
  `qpl.transforms` remains a tested numerical building block with no instrument
  behind it -- which is exactly the condition ADR-0004 was written about.

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
