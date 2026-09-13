# Labs To Library Delta Map (Chapters 1-8)

This map summarizes how each lab notebook drove `src/qpl` changes and test coverage.

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
  - `BinomialDPConfig`
  - `build_recombining_spot_tree`
  - `price_american_put_binomial`
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
