# Quant Pricing Lab (qpl)

`qpl` is a pre-1.0 numerical-methods lab for option pricing under Black–Scholes and
Heston. It is built by **Textbook-Driven Development**: a textbook or literature result
is re-derived independently, turned into an executable case, cross-checked by an
independent method, and backed by measured convergence or statistical evidence before
it becomes package surface. Every case lives in `qpl.cases` as data with its evidence
class and source; the derivations, measured numbers and known failure modes are written
up in `docs/notes/`.

It is **not** a production trading framework. Correctness and measured convergence come
before instrument breadth, API polish and speed.

Current milestone: **v0.3.0** (a source/repository milestone; see `CHANGELOG.md`). The
package is not published on PyPI; install from source.

## What is implemented

All engines are reached through one dispatcher, `qpl.pricing.price` / `qpl.pricing.greeks`,
keyed by `(instrument type, model type, method)` (ADR-0005).

- **Analytic Black–Scholes**: European price and Greeks, implied volatility; closed forms
  for cash-or-nothing digitals, geometric-average Asians and continuously monitored
  single barriers (Reiner–Rubinstein).
- **Binomial trees** (`method="tree"`): CRR and Leisen–Reimer lattices, with Greeks read
  from the lattice. Measured order 1 for CRR (with its odd/even oscillation documented)
  and order 2 for Leisen–Reimer on European payoffs.
- **Finite differences** (`method="pde"`): theta scheme (explicit / Crank–Nicolson /
  implicit) with Rannacher start-up, strike alignment to a cell midpoint, cell-average
  payoff projection for discontinuous payoffs, and a non-uniform `sinh` grid that can
  place a node on a barrier. Greeks are read from the grid.
- **Monte Carlo** (`method="mc"`): seeded, with a standard error on every estimate;
  antithetic, control-variate and stratified estimators; bump (common random numbers),
  pathwise and likelihood-ratio Greek estimators, each with its own standard error; a
  scalar SDE simulator (Euler, Milstein, exact) for GBM and CIR.
- **American exercise**, three ways. The tree and the PDE (projected SOR on the linear
  complementarity problem) allow exercise only at their time steps, so each computes a
  Bermudan approximation that converges to the American value as the grid is refined
  (order about 1 on the lattice, about 1.85 fitted for PSOR, both measured against a
  lattice-bracketed limit). Least-squares Monte Carlo (Longstaff–Schwartz) prices an
  explicit Bermudan on `exercise_dates` and says so in its metadata; its default
  out-of-sample estimate is low-biased (a fitted, suboptimal exercise policy), the
  in-sample option is high-biased, and it has no Greeks.
- **Exotics as method drivers**: cash-or-nothing digitals (all five methods), fixed-strike
  arithmetic and geometric Asians (Monte Carlo with the Kemna–Vorst geometric control
  variate), and single barriers, eight types with rebate (analytic continuous monitoring;
  discrete monitoring by Monte Carlo with optional BGK / Brownian-bridge corrections;
  tree and PDE).
- **Fourier pricing** (`method="fourier"`): COS, Carr–Madan (FFT or direct quadrature),
  Lewis and Gil-Pelaez, behind a characteristic-function interface shared by
  Black–Scholes and Heston; implied-volatility smile utilities.
- **Heston**: branch-cut-safe characteristic function, validated against published
  reference values; Monte Carlo by Andersen's QE scheme (with full-truncation Euler and
  an exact-variance hybrid for comparison) and a bias study against a transform
  reference; calibration (`qpl.calibration.calibrate_heston`, bounded trust-region or
  Levenberg–Marquardt) that reports the Jacobian, its singular values and condition
  number, a Gauss–Newton covariance and the Feller flag, because a fit that reproduces a
  smile does not identify its parameters (see
  `docs/notes/heston_calibration_identifiability.md`).
- **Validation**: `qpl.validation` measures convergence orders and tags every benchmark
  with an `EvidenceClass` (exact identity, closed form, published benchmark, independent
  engine, convergence order, statistical, negative finding). QuantLib-Python is an
  optional independent oracle (`tests/oracle/`), never the sole evidence.
- **Earlier numerical building blocks** (from the Fusai & Roncoroni Part I chapters):
  random sampling, dynamic-programming optimal stopping, iterative linear solvers,
  quadrature, Stehfest Laplace inversion and Gaussian copulas.

The full map — the loop, the literature spine, what each slice delivered and what it
measured — is `docs/CURRICULUM.md`; the 18 derivation notes are in `docs/notes/`.

## Known limitations / out of scope

- Single-asset equity models only (Black–Scholes and Heston, flat rate and dividend
  curves). No multi-asset products, no interest-rate models or term structures, no local
  or stochastic-local volatility.
- No quasi-Monte Carlo; stratification is terminal-only.
- American Greeks by simulation are refused; no Bermudan instrument type (the exercise
  grid is an engine setting).
- Performance is not a goal: engines are plain NumPy/SciPy and nothing is benchmarked for
  speed. The full test suite takes several minutes.
- The API is unstable before 1.0 and backward compatibility is not guaranteed
  (ADR-0003).

These are deferred until a case forces them; see the Phase 5 gate in `docs/CURRICULUM.md`.

## Install (from source)

```bash
git clone https://github.com/AhmadAlkadri/quant-pricing-lab.git
cd quant-pricing-lab
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The core depends only on NumPy, SciPy and Matplotlib. Optional extras: `.[oracle]` adds
QuantLib-Python for the oracle tests; `.[data]` adds pandas/yfinance for the frozen,
out-of-curriculum market-data scripts.

## Try it

One European call, five engines, one dispatcher; then an American put and a Heston price:

```python
from qpl.engines.fourier import FourierConfig
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.pde.pricers import PDEConfig
from qpl.engines.tree import TreeConfig
from qpl.instruments import AmericanOption, EuropeanOption
from qpl.market import FlatDividendCurve, FlatRateCurve, Market
from qpl.models import BlackScholesModel, HestonModel
from qpl.pricing import price

market = Market(spot=100.0, rate_curve=FlatRateCurve(0.05), dividend_curve=FlatDividendCurve(0.0))
bs = BlackScholesModel(sigma=0.20)
call = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
pde = PDEConfig(n_s=400, n_t=400, theta=0.5, strike_alignment="midpoint", time_stepping="rannacher")

print(price(call, bs, market, method="analytic").value)                  # 10.450583572185565
print(price(call, bs, market, method="tree",
            cfg=TreeConfig(n_steps=801, scheme="leisen-reimer")).value)  # 10.45058302...
print(price(call, bs, market, method="pde", cfg=pde).value)              # 10.45045...
mc = price(call, bs, market, method="mc", cfg=MCConfig(n_paths=200_000, seed=7))
print(mc.value, mc.stderr)                                               # 10.4479 0.0329
print(price(call, bs, market, method="fourier", cfg=FourierConfig(method="cos")).value)

put = AmericanOption(kind="put", strike=100.0, expiry=1.0)
print(price(put, bs, market, method="pde", cfg=pde).value)               # 6.08886...
heston = HestonModel(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
print(price(call, heston, market, method="fourier", cfg=FourierConfig(method="cos")).value)
```

The converged American value at this point is 6.090376 (`qpl.cases.AMERICAN_BRACKETED_LIMIT`);
the PDE result sits 1.5e-03 below it at this grid.

The cases are importable data. Check put–call parity on the Leisen–Reimer lattice against
each case's own tolerance:

```python
from qpl.cases import PARITY_CASES, parity_residual
from qpl.engines.tree import TreeConfig
from qpl.pricing import price

cfg = TreeConfig(n_steps=101, scheme="leisen-reimer")
for case in PARITY_CASES:
    s = case.spec
    call = price(s.option(), s.model(), s.market(), method="tree", cfg=cfg).value
    put = price(s.flipped().option(), s.model(), s.market(), method="tree", cfg=cfg).value
    print(case.row.id, case.row.evidence.value, abs(parity_residual(call, put, s)) <= case.row.tolerance)
```

Each line prints the case id, `exact_identity` and `True` (residuals are round-off, about
1e-12, against a 1e-10 tolerance).

Example scripts print their measurements as `key=value` lines (a few seconds each):

```bash
python examples/tree_convergence.py --scheme leisen-reimer   # order=1.9840; error 507x (n=101) to 3966x (n=801) below CRR
python examples/american_put_cross_method.py                 # PSOR grid vs lattice vs LSM, with the Bermudan gap
python examples/heston_calibration.py                        # one-maturity condition number 6.7e+07; kappa not identified
```

`examples/README.md` lists the rest; `tests/test_examples_smoke.py` runs them.

## Development

```bash
ruff check .
pytest -q                  # the CI contract: everything, several minutes
pytest -q -m "not slow"    # quick inner loop
```

CI runs two jobs: `tests-core` (`.[dev]`) and `tests-full` (`.[dev,data,oracle]`). Labs are
private, textbook-driven workbooks (Fusai & Roncoroni, Glasserman) that live under
`labs/` and are gitignored; they drive the design but are never published. The public
cargo is the package (`src/`), the tests, the `qpl.cases` layer and the derivation notes.

Development phase: exploratory pre-1.0. API churn is expected and backward compatibility
is not guaranteed. See `AGENTS.md` and
`.agents/brain/adr/0003-pre-1-0-lab-authority-and-api-churn.md`.

### Project brain / contribution workflow

See `.agents/brain/brain.md` (architecture and invariants), `.agents/brain/adr/`
(decision records) and `.agents/brain/steering-brief.md` (recent changes). Read
`.agents/brain/brain.md` before structural or API changes.

### Notebook hygiene

This repo uses nbstripout to keep notebooks deterministic. After clone:

```bash
python -m nbstripout --install --attributes .gitattributes
```

### Documentation site

The VitePress site in `docs-site/` renders `docs/CURRICULUM.md`, the derivation notes
and a short reference; `docs/` stays the single source. To build it locally:

```bash
cd docs-site
npm ci
npm run docs:dev      # or: BASE=/quant-pricing-lab/ npm run docs:build
```

It is built from `main` by `.github/workflows/deploy-docs.yml` for GitHub Pages
(intended at https://ahmadalkadri.github.io/quant-pricing-lab/; Pages is not enabled
yet).

## License

MIT
