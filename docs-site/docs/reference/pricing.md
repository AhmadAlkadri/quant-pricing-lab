# Pricing

Core dispatchers:

- `qpl.pricing.price`
- `qpl.pricing.greeks`

Both look up an engine by `(instrument type, model type, method)` in the engine
registry (`qpl.engines.registry`, ADR-0005); a pair that is not registered raises
`NotSupportedError`, and some registered pairs raise it on purpose with a message
naming the route that does work.

Methods and their configs:

| `method` | config | Black–Scholes instruments | Heston instruments |
|---|---|---|---|
| `analytic` | none | European, digital, geometric Asian, continuous barrier | — |
| `tree` | `TreeConfig` (`scheme="crr"` or `"leisen-reimer"`) | European, American, digital, barrier | — |
| `pde` | `PDEConfig` (theta, Rannacher, strike alignment, `grid="sinh"`) | European, American (PSOR), digital, barrier | — |
| `mc` | `MCConfig` (seed, variance reduction, Greek estimator, LSM settings) | European, American (LSM, Bermudan), digital, Asian, discrete barrier | European, digital, Asian, barrier (QE scheme) |
| `fourier` | `FourierConfig` (`cos`, `carr_madan`, `lewis`, `gil_pelaez`) | European, digital | European, digital |

Typical workflow:

1. build an instrument (`EuropeanOption`, `AmericanOption`, `DigitalOption`,
   `AsianOption`, `BarrierOption`), a model (`BlackScholesModel`, `HestonModel`) and a
   `Market`;
2. choose the method and its config;
3. call the dispatcher and inspect `PriceResult` / `GreeksResult`, whose `meta` records
   what was actually computed (e.g. `exercise_style` for LSM, per-Greek estimators and
   standard errors for Monte Carlo).

Heston calibration is separate: `qpl.calibration.calibrate_heston`.

See `examples/bs_analytic_greeks.py`, `examples/american_put_cross_method.py` and
`examples/heston_smile.py`, and the [Cases](/reference/cases) page for checked values.
