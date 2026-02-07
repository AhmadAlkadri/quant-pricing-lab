# Steering Brief

What changed since last brief (files + bullets)
- `src/qpl/engines/mc/pricers.py`: Implemented MC Theta via finite difference.
- `tests/test_mc_greeks.py`: Added verification for MC vs Analytic Theta.
- `examples/bs_mc_vs_analytic.py`: Updated to display Theta.

Current architecture (8-12 lines)
- Core: Dispatcher `pricing.price`/`greeks` routes to {Analytic, MC, PDE} engines.
- Domain: `EuropeanOption`, `BlackScholesModel`, `Market` (flat curves).
- Engines: 
  - Analytic: Closed-form BS.
  - MC: GBM (terminal/multi-step) with Greeks via bumps.
  - PDE: Theta-scheme FD (Call/Put only).
- Status: European vanilla pricing/greeks complete across all 3 engines.

Public API status (stable vs experimental)
- Stable: `qpl.pricing` dispatcher, `EuropeanOption`, `Market`, `BlackScholesModel`.
- Experimental: Engine config classes (`MCConfig`, `PDEConfig`) and their direct entry points.

Risks / unknowns
- Dispatcher complexity might grow with new Instrument types (Binary Options).

Next 3 recommended actions
- Implement `BinaryOption` (Analytic engine only).
- Add `Benchmark Harness` to track regression.
- Explore MC Variance Reduction (Antithetic).

One simplification / deletion candidate
- None currently (clean state).

Assumptions I'm making
- Binary Options will only support Analytic engine initially (as per roadmap).

How to validate quickly
- Run `pytest` for regressions.
- Run `examples/bs_mc_vs_analytic.py` to see MC Theta in action.
