# Cases

`qpl.cases` holds the curriculum's numerical claims as importable data. Each case
pairs a specification (market, model, instrument or method settings) with a
`qpl.validation.BenchmarkRow`: the expected value, the tolerance, the
`EvidenceClass` that justifies them and the source. The modules contain no
assertions and no pytest dependency; the tests under `tests/cases/` (and several
engine test files) parametrise over them.

| Module | What it holds | Aggregate |
|---|---|---|
| `european_black_scholes` | Put–call parity, degenerate limits, closed-form reference values, comparative statics, CRR / Leisen–Reimer tree orders, PDE and Monte Carlo Greek cases, Fourier known values | `ALL_CASES` |
| `american_black_scholes` | No-early-exercise identities (no-dividend call, zero-rate put), early-exercise premium rows, lattice reference and bracketed limit, Leisen–Reimer rows, tree/PDE/LSM cross-engine cases, the Longstaff–Schwartz Table 1 Bermudan row; `bermudan_value_on_lattice` | `ALL_AMERICAN_CASES` |
| `digital_black_scholes` | Cash-or-nothing identities and known values, the strike-derivative identity, tree and PDE order cases, Monte Carlo and Fourier settings | `ALL_DIGITAL_CASES` |
| `mc_variance_reduction` | Measured variance ratios for antithetic, control-variate and stratified estimators, and for the Monte Carlo Greek estimators | `MC_VARIANCE_REDUCTION_CASES` |
| `asian_black_scholes` | Geometric-average closed form and published values, continuous-limit and approximation rows, arithmetic-average Monte Carlo rows | `ALL_ASIAN_CASES` |
| `sde_discretization` | Strong and weak orders for Euler and Milstein on GBM, CIR mean, negativity and price-bias rows | `ALL_SDE_CASES` |
| `barrier_black_scholes` | Barrier identities and limits, published values, cross-engine rows, Monte Carlo monitoring-bias and tree/PDE order cases | `ALL_BARRIER_CASES` |
| `heston` | Published reference values, cross-method agreement, parity, the Black–Scholes limit, smile shape, QE Monte Carlo rows | `ALL_HESTON_CASES` |
| `heston_calibration` | Synthetic-recovery, noise, objective, conditioning and initialisation rows, including the one-maturity identifiability negative finding; `synthetic_quotes`, `noisy_quotes` | `ALL_HESTON_CALIBRATION_CASES` |

Every name is re-exported from `qpl.cases`:

```python
from qpl.cases import KNOWN_VALUE_CASES

case = KNOWN_VALUE_CASES[0]
case.row.id, case.row.expected, case.row.evidence   # ('atm_1y_call_reference', 10.450583572185565, <EvidenceClass.CLOSED_FORM: ...>)
```

The derivations and measurements behind each module are in the
[derivation notes](/curriculum/); the tests that evaluate them are in
[`tests/cases/`](https://github.com/AhmadAlkadri/quant-pricing-lab/tree/main/tests/cases).
