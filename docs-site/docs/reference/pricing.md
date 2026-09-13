# Pricing

Core dispatchers:

- `qpl.pricing.price`
- `qpl.pricing.greeks`

Supported methods:

- `analytic`
- `mc` (requires `MCConfig`)
- `pde` (requires `PDEConfig`)

Typical workflow:

1. build `EuropeanOption`, `BlackScholesModel`, `Market`
2. choose method/config
3. call dispatcher and inspect `PriceResult` or `GreeksResult`

See `examples/bs_analytic_greeks.py` and `examples/mc_pricing_and_stderr.py`.
