# Heston by simulation: Andersen's QE scheme, and what it is worth

*Slice 16. Everything below is derived in `src/qpl/engines/mc/heston.py` and
measured in this repository; nothing is transcribed from any source. Sources
are cited where a construction is theirs.*

## The problem

Under the pricing measure, with `X_t = ln S_t`,

```
dX = (r - q - v/2) dt + sqrt(v) dW1
dv = kappa (theta - v) dt + xi sqrt(v) dW2,    d<W1, W2> = rho dt.
```

The variance is a square-root diffusion. An Euler step can send it below zero,
after which `sqrt(v)` is undefined and the spot step cannot be taken at all.
Slice 9 measured how bad that is on the CIR process alone: **68-73% of terminal
variances are negative** in the Feller-violating regime at `h = 1/4 ... 1/32`,
and Andersen's full truncation — evaluate the coefficients at `max(v, 0)`, leave
the state unfloored — keeps the recursion *defined* without making it positive.
Lord, Koekkoek & van Dijk (2010) find full truncation the least biased of the
Euler repairs, so it is the Euler comparison used throughout this note.

The variance transition is known exactly (a scaled noncentral chi-square:
Broadie & Kaya 2006, section 2.1; implemented in `qpl.engines.mc.sde`). What is
*not* cheap is the spot: exact simulation needs the conditional law of
`int_0^T v ds` given the endpoints, which Broadie and Kaya obtain by inverting a
characteristic function once per step per path. That cost, not the variance, is
what Andersen's scheme replaces.

## The QE scheme, in five steps

Andersen, L. (2008), *Journal of Computational Finance* 11(3), sections 3.2-3.3
and 4.3. Re-derived here from the CIR conditional moments.

**1. Two conditional moments.** Taking expectations in the variance SDE (the
diffusion term is a martingale),

```
m  = theta + (v_t - theta) e^{-kappa dt}
s2 = v_t (xi^2/kappa) e^{-kappa dt}(1 - e^{-kappa dt})
     + theta (xi^2/(2 kappa))(1 - e^{-kappa dt})^2
```

These are the same pair `qpl.engines.mc.sde.cir_moments` derives from the
noncentral chi-square's degrees of freedom and non-centrality — a completely
different route — and they agree to **1e-16 relative** on both branches. That
agreement is the slice's one EXACT_IDENTITY.

**2. One shape parameter.** `psi = s2 / m^2`, the squared coefficient of
variation. Large `psi` is the transition piled up near zero with a long right
tail; small `psi` is a bump around `m`. No two-parameter family covers both, so
the scheme uses two and switches.

**3a. `psi <= psi_c`: a scaled, shifted square.** `v' = a (b + Z)^2`. Matching
`E[(b+Z)^2] = 1 + b^2` and `Var = 2 + 4b^2` gives
`psi = (2 + 4b^2)/(1 + b^2)^2`, whose non-negative root is

```
b^2 = 2/psi - 1 + sqrt(2/psi) sqrt(2/psi - 1),   a = m / (1 + b^2),
```

real exactly when `psi <= 2`.

**3b. `psi > psi_c`: an atom plus an exponential.** `P(v' = 0) = p` with density
`(1-p) beta e^{-beta x}` above. Its moments are `(1-p)/beta` and
`(1-p)(1+p)/beta^2`, so `(1+p)/(1-p) = psi` and

```
p = (psi - 1)/(psi + 1),   beta = 2 / (m (psi + 1)),
```

which needs `psi >= 1`. Any switching level in `[1, 2]` is therefore
admissible; this package uses Andersen's own **`psi_c = 1.5`**, the midpoint,
and the moment match is checked on both sides of it.

Both branches match two moments **exactly** and nothing else. Measured against
the exact noncentral chi-square at 200,000 draws:

| set | QE skew | exact skew | QE kurtosis | exact kurtosis |
|---|---|---|---|---|
| reference (`v=0.04, dt=1/4, kappa=4, theta=0.25, xi=1`) | 1.103 | 1.421 | 4.620 | 6.183 |
| Feller-violating (`kappa=0.5, theta=0.04, xi=1`) | 3.542 | 3.674 | 19.646 | 21.128 |

QE's skewness is **22% low** and its kurtosis **25% low** on the first set; on
the second the gaps fall to 4% and 7%, because there the exponential branch's
own heavy tail is already close to the truth. That is a property of a
two-moment fit, not a defect, and it is reported rather than repaired.

**4. The spot step is not an Euler step.** Integrating the variance SDE over
one step,

```
int sqrt(v) dW2 = (v' - v - kappa theta dt + kappa int v ds) / xi,
```

so the correlated part of the spot increment is a *function of the sampled
variance endpoints*. With `dW1 = rho dW2 + sqrt(1-rho^2) dW_perp` and
`int v ds ~ dt (gamma1 v + gamma2 v')`,

```
X' = X + (r-q) dt + K0 + K1 v + K2 v' + sqrt(K3 v + K4 v') Z,   Z independent,

K0 = -rho kappa theta dt / xi
K1 = gamma1 dt (kappa rho/xi - 1/2) - rho/xi
K2 = gamma2 dt (kappa rho/xi - 1/2) + rho/xi
K3 = gamma1 dt (1 - rho^2)
K4 = gamma2 dt (1 - rho^2)
```

with `gamma1 = gamma2 = 1/2` (the trapezoidal rule). Note what this buys: `rho`
enters through a *deterministic* function of the two variance values, and the
only normal drawn for the spot is orthogonal to the variance. Measured
`corr(dX, dv)` against `rho`, 400,000 paths:

| rho | dt = 1/16 | 1/64 | 1/256 | sampling stderr |
|---|---|---|---|---|
| -0.9 | -0.8406 | -0.8837 | -0.8958 | 2e-05 |
| -0.5 | -0.4696 | -0.4917 | -0.4980 | 3e-05 |
| 0.0 | -0.0005 | -0.0002 | -0.0003 | 4e-05 |
| +0.5 | +0.4751 | +0.4932 | +0.4981 | 3e-05 |

The residual is a *discretisation* effect — it is hundreds of standard errors
wide and it falls at measured **order ~1 in `dt`** (fitted 1.0 over
`dt = 1/8 ... 1/64`, log-space residual below 0.1). All three schemes behave the
same way here, full-truncation Euler slightly worse at the coarse end
(+0.0177 at `rho = -0.9, dt = 1/64` against QE's +0.0163).

**5. The martingale correction (Andersen §4.3).** Nothing above forces
`E[S_{t+dt} | S_t, v_t] = S_t e^{(r-q) dt}`. Conditional on the two variance
values the increment is Gaussian, so with `A = K2 + K4/2`,

```
E[e^{X' - X} | v] = e^{(r-q)dt + K0 + (K1 + K3/2) v} * E[e^{A v'} | v],
```

and `E[e^{A v'}]` is closed form on each branch because each branch is a named
law: `exp(A a b^2/(1 - 2 A a)) / sqrt(1 - 2 A a)` on the quadratic one
(`2 A a < 1`), `p + beta(1-p)/(beta - A)` on the exponential one (`A < beta`).
Replacing `K0` by

```
K0* = -ln E[e^{A v'} | v] - (K1 + K3/2) v
```

makes the conditional expectation exactly `e^{(r-q) dt}` at **any** step size.
With `rho < 0` both conditions hold automatically; where one fails the step
falls back to the plain `K0` and the count is reported in
`meta["martingale_correction_fallbacks"]`.

Measured, QE on the reference set, 1,000,000 antithetic paths:

| dt | defect with correction | stderr | defect without | z (without) |
|---|---|---|---|---|
| 1/4 | +0.0123 | 0.0173 | **+1.1206** | +64.2 |
| 1/8 | -0.0053 | 0.0188 | +0.2784 | +14.8 |
| 1/16 | +0.0167 | 0.0197 | +0.0883 | +4.5 |

`E[e^{-(r-q)T} S_T] - S_0` on a spot of 100. **The uncorrected scheme
misprices the forward by 1.1% at `dt = 1/4`**, 64 standard errors from zero,
and the defect falls at roughly order 2 — invisible on a fine grid, expensive on
a coarse one. This is the slice's pinned NEGATIVE_FINDING.

## Conditioning on the variance driver

In all three schemes `ln S_T` is **exactly Gaussian conditional on the variance
driver**:

- QE and the exact-variance hybrid, by construction — `Z` is drawn
  independently of everything that made the variance path, so the sum is
  Gaussian with `mean = sum(mu dt + K0 + K1 v_i + K2 v_{i+1})` and
  `variance = sum(K3 v_i + K4 v_{i+1})`;
- full-truncation Euler, because conditioning on the variance path conditions
  on every `Z_v` (the variance recursion is a function of them), leaving
  `variance = (1 - rho^2) sum v_i^+ dt` and the `rho Z_v` term inside the mean.

So a terminal payoff has a closed-form conditional expectation (a Black-Scholes
formula in `(mean, variance)`) and averaging it estimates *the same scheme
price* with the whole spot diffusion removed. This is Romano and Touzi's (1997)
conditioning argument applied to the discretisation rather than to the model;
`MCConfig(heston_conditional=True)` turns it on.

Measured factors at `dt = 1/16` on the reference set, against the plain
estimator at equal path count:

| estimator | variance factor |
|---|---|
| antithetic | 1.49 |
| conditional | 10.27 |
| antithetic + conditional | **82.07** |

This is what makes the bias tables below resolvable at all: without it the
standard error at 1,000,000 paths is 2.8e-02 and the bias being measured at
`dt = 1/16` is 8.4e-03.

**The control variate and conditioning are substitutes, not complements.** The
discounted terminal spot correlates **0.797** with the plain payoff and
**0.486** with the conditional one, because what it was correlated with was the
spot diffusion and conditioning has already integrated that out. And its mean is
the *model's* forward, not the *scheme's*, so on a scheme whose own forward is
off it silently removes the part of the bias collinear with that defect
(measured shift on full-truncation Euler at `dt = 1/4`: **-5.2e-03**). Every
bias number in this note is therefore measured **without** it.

## The bias ladder: reference set (Feller satisfied, number 4.0)

ATM call, `S = K = 100, T = 1, r = 1%, q = 2%, v0 = 0.04, kappa = 4,
theta = 0.25, xi = 1, rho = -0.5`. Reference: the **Lewis** transform integral
(16.070155), which Slice 15's oracle established as one of the two methods with
no parameter to get wrong. 1,000,000 antithetic conditional paths,
`examples/heston_mc_qe.py --paths 1000000`.

| dt | QE | stderr | Euler (full trunc.) | stderr | exact-var. hybrid | stderr |
|---|---|---|---|---|---|---|
| 1/4 | **-0.101749** | 0.00315 | **+0.260106** | 0.00579 | **-0.146548** | 0.00873 |
| 1/8 | -0.025566 | 0.00306 | +0.036927 | 0.00450 | -0.041894 | 0.00870 |
| 1/16 | -0.008395 | 0.00309 | -0.005925 | 0.00384 | -0.009681 | 0.00872 |
| 1/32 | -0.004918 | 0.00313 | -0.007892 | 0.00349 | +0.004203 | 0.00873 |
| 1/64 | +0.004173 | 0.00314 | +0.000548 | 0.00330 | +0.008319 | 0.00873 |

**Only the two coarsest levels clear their own noise** for any of the three
schemes, which is stated per row rather than hidden: the example prints a
`resolved=` flag and refuses to fit an order through fewer than three resolved
levels. What can be quoted honestly:

| scheme | bias(1/4)/bias(1/8) | implied order | five-level fit (order, constant, log-residual) |
|---|---|---|---|
| QE | 3.98 | +1.99 | +1.159, 0.336, 0.361 |
| Euler full truncation | 7.04 | +2.82 | +2.001, 3.072, 0.582 |
| exact variance + Euler log-spot | 3.50 | +1.81 | +1.160, 0.457, 0.591 |

The five-level fits have log-space residuals of 0.36-0.58, which is what a fit
through three noise-dominated points looks like; the ratio column is the
measurement. Read as a decay rate, not as the scheme's theoretical weak order.

### Three things the slice statement expected and the measurements refuse

1. **"QE's bias is far smaller than Euler's at coarse steps."** Measured ratio
   at `dt = 1/4`: **2.56**. Better, and not by an order of magnitude — on a set
   where the Feller condition *holds*. The dramatic separation is a
   Feller-regime effect, not a scheme effect in general (next section).
   The two schemes also err in **opposite directions**, so a bias measured on
   one says nothing about the other even qualitatively.
2. **"The exact-variance hybrid is in between."** It is not: at `dt = 1/4` it
   is **worse than QE** (-0.147 against -0.102), despite sampling the variance
   marginal exactly. The two error sources are not additive — QE's variance-law
   error partially cancels the trapezoidal integrated-variance error, and
   removing the first exposes the second.
3. **"With `rho = 0` the QE call agrees with the transform within noise at
   coarse dt."** It does not. At `rho = 0`, everything else unchanged, the bias
   at `dt = 1/4` is **-0.2439** (stderr 1.4e-03), -0.0652 at 1/8 and -0.0154 at
   1/16 — *larger* in absolute value than the `rho = -0.5` bias. At `rho = 0`
   the spot step reduces to the bare trapezoidal rule for `int v ds`, and with
   `kappa dt = 1` that rule's error is exactly this size. A price comparison at
   `rho = 0` is a test of the time discretisation with the correlation switched
   off, not a test of the correlation; the sharp correlation check is the
   `corr(dX, dv)` table above.

## The bias ladder: Feller violated (number 0.08)

`v0 = 0.04, kappa = 0.5, theta = 0.04, xi = 1, rho = -0.9` — this repository's
own Feller-violating variance parameters (shared with the Slice 9 CIR study)
plus `rho = -0.9`. Reference: Lewis, 3.591838. The COS call at the derived
`L = 28, N = 4096` (Slice 15's `HESTON_TRUNCATION_L_FELLER_VIOLATED`) agrees
with it to **1.2e-10** at `T = 1`, which is what licenses using Lewis without
re-arguing the range; at `T = 10` the COS call has no usable setting at all and
the put-then-parity route is the recorded recipe.

| dt | QE | stderr | Euler (full trunc.) | stderr | Euler negative-variance fraction |
|---|---|---|---|---|---|
| 1/4 | **-0.021444** | 0.00113 | **+3.407607** | 0.00624 | 0.540 |
| 1/8 | -0.011907 | 0.00113 | +2.131970 | 0.00388 | 0.586 |
| 1/16 | -0.006215 | 0.00112 | +1.145278 | 0.00240 | 0.611 |
| 1/32 | -0.001249 | 0.00112 | +0.538888 | 0.00162 | 0.611 |
| 1/64 | +0.001831 | 0.00112 | +0.231848 | 0.00130 | — |

- **Euler's error at `dt = 1/4` is 95% of the price.** Ratio against QE:
  **158.9**. This is the Lord-Koekkoek-van Dijk ordering at its most extreme —
  full truncation is the *least*-biased Euler fix and is still unusable here.
- **Euler's ladder is the only fully resolved one in this slice**: five levels,
  every one hundreds of standard errors from zero, fitted **order +0.974** with
  constant **15.0** and a log-space residual of 0.106. Below one, and with a
  constant three decimal orders above the reference set's — the boundary the
  scheme cannot respect costs it both. That echoes Slice 9's measured order 0.64
  on the CIR variance mean in the same regime.
- **QE's ladder has three resolved levels here** (the bias is small but the
  variance is small too): fitted **order +0.893**, constant 0.075, log-space
  residual **0.015**.
- **QE's variance is non-negative everywhere, at every step size, by
  construction** — a scaled square on one branch, an atom plus a positive
  exponential draw on the other. Measured frequency of a negative draw: exactly
  0. Euler's: 54-61%.

## The smile from simulated prices

A Monte Carlo price is a number with an error bar and an implied volatility is a
nonlinear function of it, so the error bar travels through vega:
`sd(sigma) = sd(price) / vega(sigma)`, with vega the Black-Scholes vega *at the
implied volatility* — Black-Scholes is being used as a quoting convention, not
as a model. Measured at 100,000 antithetic conditional paths, `dt = 1/32`,
`rho = -0.5`, `T = 1`:

| K | MC price | iv (MC) | iv (transform) | difference | iv stderr |
|---|---|---|---|---|---|
| 80 | 26.75794 ± 0.0133 | 0.445924 | 0.446474 | -5.50e-04 | 4.35e-04 |
| 90 | 20.91794 ± 0.0117 | 0.434197 | 0.434630 | -4.33e-04 | 3.28e-04 |
| 100 | 16.05615 ± 0.0099 | 0.424121 | 0.424485 | -3.64e-04 | 2.57e-04 |
| 110 | 12.11959 ± 0.0080 | 0.415483 | 0.415806 | -3.23e-04 | 2.04e-04 |
| 120 | 9.01369 ± 0.0062 | 0.408108 | 0.408406 | -2.97e-04 | 1.63e-04 |

Every cell is inside 1.9 propagated standard errors, both smiles are strictly
decreasing in strike (the `rho < 0` skew), and their slopes agree to 2%. But
**every difference is negative**, and at `dt = 1/64` with 400,000 paths the same
column reads -1.99e-04 / -1.72e-04 / -1.42e-04 / -1.13e-04 / -8.7e-05 — halved
with the step size. Five independent statistical errors would share a sign with
probability 1/16. The agreement is statistical at these settings and would
become a *measurable bias* at a larger path count; a calibrator fitting to
simulated prices would inherit it.

## The two path-dependent engines, and what they lost

**The Asian lost its control variate.** Kemna and Vorst's geometric average is
lognormal under Black-Scholes, which is what makes its closed form a control
variate with correlation 0.9996 and a measured variance factor of 1277 (Slice
8). Under Heston the geometric average is lognormal only *conditional on the
variance path*, so there is no closed form and no exact mean. The fallback is
the discounted terminal spot: measured correlation **0.63**, factor **1.7**.
Three decimal orders, and the engine reports which control is in play and why
the good one is not.

**The barrier's Brownian bridge stopped being exact.** Under Black-Scholes the
bridge estimator is unbiased for the *continuous* contract at any number of
sampling dates — Slice 12 measured it pricing the continuous barrier from a
**single** observation. That rests on the bridge having constant volatility
between its endpoints. Under Heston it does not, the engine feeds it the
trapezoidal integrated variance `dt (v_i + v_{i+1})/2`, and the estimator
becomes an approximation. Measured on `H = 95, K = 100, S_0 = 100, T = 1`
(survival ~8%, so the bridge decides nearly every path), 400,000 antithetic
paths with the terminal-spot control:

| n_steps | price | stderr | mean survival |
|---|---|---|---|
| 25 | 4.63838 | 0.02376 | 0.08711 |
| 50 | 4.51679 | 0.02424 | 0.08402 |
| 100 | 4.41176 | 0.02452 | 0.08228 |
| 200 | 4.38643 | 0.02496 | 0.08142 |
| 400 | 4.31406 | 0.02508 | 0.08025 |
| 800 | 4.31367 | 0.02528 | 0.07994 |

A residual of **+0.325 at `n_steps = 25`**, 7.5% of the price and thirteen
standard errors, falling at roughly first order. The coarse grid *overstates*
survival — it cannot see the crossings inside a step whose integrated variance
it is understating — so a knock-out is priced too high. `meta` carries the
caveat on every result that uses the route.

The discretely monitored contract is a different matter: `barrier_correction=
"none"` is unbiased for it *given the scheme*, and the discrete-versus-continuous
gap belongs to the contract — measured **+1.083** at `m = 50, H = 80`, which is
Slice 12's monitoring bias reappearing under a second model.

## Against an independent implementation

QuantLib's `MCEuropeanHestonEngine` on a `HestonProcess` carrying
`QuadraticExponentialMartingale` is the same algorithm in a different codebase,
with different random numbers and someone else's reading of the same paper.
Eight cells (both parameter sets, `dt = 1/8` and `1/32`, calls and puts,
`T = 1.0` exact under `Actual365Fixed`): worst `|z|` against the **combined**
standard error is **2.04**.

Sharper than the price comparison, and the check that would actually catch a
wrong QE: a scheme's bias is a property of the algorithm, so two correct
implementations must miss the transform price in the *same direction* at the
same step size. At `dt = 1/8`, ATM call, both are negative on both sets
(QuantLib -1.02e-01 against -2.20e-02 here on the reference set; -2.28e-02
against -8.18e-03 on the Feller-violating one), with the sizes inside each
other's standard errors.

One asymmetry is the practical difference between the two: at 100,000 QuantLib
samples against 200,000 paths here, QuantLib's standard error at `dt = 1/32` on
the reference set is 5.1e-02 and this package's is **7.0e-03** — and this
package's *plain* estimator is also 5.1e-02. The whole factor of ~50 is
`heston_conditional=True`, which QuantLib's engine has no equivalent of.

## What is refused, and why

| request | answer |
|---|---|
| `MCConfig(n_steps=1)` under Heston | `InvalidInputError`. `n_steps` is the *time discretisation* here, not a cost knob, and no scheme is exact in one step. |
| `variance_reduction="stratified"` | `NotSupportedError`. A Heston path is driven by `2 n_steps` normals with no single scalar to partition. |
| `antithetic` with `exact_variance_euler_log_spot` | `NotSupportedError`. Its variance draw is a noncentral chi-square, not a function of one normal, so a reflected pair would reflect half the driver. |
| `greeks_estimator="likelihood_ratio"` | `NotSupportedError`. The *model's* transition density is what Broadie & Kaya reach only by numerical inversion; the QE step's conditional law does have a score, but that differentiates the discretisation, which is not what an unbiasedness claim means. |
| Digital, Asian, barrier Greeks | `NotSupportedError`, with the per-instrument reason. A bumped Heston Greek additionally costs a full re-simulation per leg. |
| `barrier_correction="bgk"` | `NotSupportedError`. The shift `exp(∓ beta sigma sqrt(dt))` names a single constant volatility, which this model does not have. |
| `heston_conditional=True` on an Asian or a barrier | `NotSupportedError`. It prices a *terminal* payoff by its conditional expectation; the conditional expectation of an arithmetic average's payoff is the original Asian problem one dimension in. |
| A barrier already touched at inception | `NotSupportedError`. The knock-in's vanilla leg is a transform price, and returning it from a Monte Carlo engine would silently swap methods. |

Greeks that **are** supported: all five by common-random-numbers bump, with
vega taken in `d/d sqrt(v0)` so it is comparable with the transform engine's,
plus a pathwise **delta** (`S_T` is homogeneous of degree one in `S_0` because
the variance dynamics never see the spot, so `dS_T/dS_0 = S_T/S_0` exactly).
Measured against the COS closed forms at 20,000 antithetic paths and
`n_steps = 16`: delta 0.5997 / 0.5995, gamma 9.288e-03 / 9.288e-03, vega 4.079
/ 4.072, theta -9.409 / -9.433, rho 43.90 / 43.88.

## Sources

- Andersen, L. (2008), "Simple and efficient simulation of the Heston
  stochastic volatility model", *Journal of Computational Finance* 11(3),
  1-42 — sections 3.2-3.3 (the scheme, the switching rule) and 4.3 (the
  martingale correction).
- Broadie, M. and Kaya, O. (2006), "Exact simulation of stochastic volatility
  and other affine jump diffusion processes", *Operations Research* 54(2),
  217-231 — section 2.1: the variance transition is exactly a noncentral
  chi-square, and the integrated-variance inversion that makes exact log-spot
  simulation expensive.
- Lord, R., Koekkoek, R. and van Dijk, D. (2010), "A comparison of biased
  simulation schemes for stochastic volatility models", *Quantitative Finance*
  10(2), 177-194 — full truncation is the least-biased Euler fix.
- Glasserman, P. (2003), *Monte Carlo Methods in Financial Engineering*,
  section 3.4 (the square-root process), chapter 4 (variance reduction).
- Romano, M. and Touzi, N. (1997), "Contingent claims and market completeness
  in a stochastic volatility model", *Mathematical Finance* 7(4), 399-412 —
  the conditioning argument, applied here to the discretisation.
- Kemna, A.G.Z. and Vorst, A.C.F. (1990), *Journal of Banking and Finance* 14,
  113-129 — the geometric control variate that this model does not admit.

## Where the code and the evidence live

- Scheme and path generator: `src/qpl/engines/mc/heston.py`.
- Pricers and registry wiring: `src/qpl/engines/mc/heston_pricers.py`,
  `src/qpl/pricing.py`.
- Claims as data: `qpl.cases.heston.HESTON_MC_CASES` (10 rows).
- Tests: `tests/test_mc_heston.py` (scheme), `tests/test_mc_heston_pricing.py`
  (registry and bias), `tests/test_mc_heston_smile.py`,
  `tests/test_mc_heston_path_dependent.py`,
  `tests/oracle/test_heston_mc_vs_quantlib.py`.
- Example: `examples/heston_mc_qe.py --case bias | feller [--paths N]`. Every
  table in this note was produced with `--paths 1000000` except where the text
  says otherwise.
