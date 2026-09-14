# Heston calibration: what a fit is, and what it is not

A calibration that finds parameters is not evidence that the parameters are
identified. This note records what Slice 17 measured about the difference, and
the numbers are all produced in this repository — by
`tests/test_heston_calibration.py`,
`tests/cases/test_heston_calibration_cases.py`,
`tests/oracle/test_heston_calibration_vs_quantlib.py` and
`examples/heston_calibration.py`. Nothing below is taken from a source; the
citations say which effect to look for and why.

Two parameter sets throughout, both already in this repository: the **reference
set** `(v0, kappa, theta, xi, rho) = (0.04, 4, 0.25, 1, -0.5)` at which six
published Heston values are quoted (Feller number 4.0), and the
**Feller-violating set** `(0.04, 0.5, 0.04, 1, -0.9)` (number 0.08), shared
with the Slice 9 CIR study and the Slice 15/16 Heston studies. The market is
`S = 100, r = 1%, q = 2%`, and the grid is strikes 80/90/100/110/120 across up
to six maturities `(0.25, 0.5, 1, 1.5, 2, 3)`.

## 1. The pricer decides more than the price

The calibration prices by the Slice 14 cosine expansion, vectorised over the
strikes of one maturity: one characteristic-function evaluation per maturity
instead of one per quote. That is a factor of **70** against adaptive-quadrature
Lewis (1.2 ms against 85 ms for 15 quotes), and a calibration makes hundreds of
those calls.

Slice 15 warned that COS "needs a setting chosen per parameter set". It turned
out the warning is about the **payoff leg** first and the range second, and the
two halves of that warning were the largest single source of wrong answers in
this slice.

### The leg

A COS *call*'s payoff coefficient integrates `e^z` up to the range's upper end
`b`, so the cosine sum has to cancel `e^b` against a price of order `S_0`; its
round-off grows exponentially with `b`. A COS *put*'s coefficient integrates up
from `a` and carries only `e^{z*} = K / S_0` — immune to that, and exposed
instead to whatever tail mass sits below `a`, which for `rho < 0` is the fat
one. Measured, worst absolute error over the five strikes against a Lewis
integral of the same transform, at the settings the calibrator uses:

| `b` | set / `T` | call direct | put then parity |
|---:|---|---:|---:|
| 1.730 | reference 0.25 | **2.13e-13** | 3.61e-07 |
| 2.946 | violating 0.25 | **2.08e-12** | 1.67e-10 |
| 4.558 | reference 1 | **1.33e-12** | 2.70e-08 |
| 6.694 | violating 1 | **1.48e-10** | 4.75e-08 |
| 8.489 | reference 3 | 1.30e-10 | **1.55e-11** |
| 14.128 | violating 3 | 2.90e-05 | **1.85e-07** |
| 15.357 | reference 10 | 1.81e-08 | **4.97e-14** |
| 31.106 | violating 10 | 6.45e+03 | **7.57e-07** |

Slice 15 measured this asymmetry twice and got opposite answers. Both were
right; they were measured in different regimes. The crossover is between 6.694
and 8.489 and it is sharp, so `COS_PUT_LEG_THRESHOLD = 7.5` picks the better
leg in all eight cells and the worst module error over the eight falls from
6.45e+03 (call only) and 3.61e-07 (put only) to **1.85e-07**.

That rule also makes the residual function *total*. Before it, a search that
wandered to `(0.5, 0.1, 0.8, 3.0, -0.99)` at `T = 3` — inside
`DEFAULT_BOUNDS` — got a 100-strike call priced at **-3.2e+43**, whose implied
volatility clamps to the edge of the arbitrage strip, whose residual is
therefore flat, and on which a local solver stalls on its first step. Three of
fourteen starts died that way. The put leg's coefficients cannot overflow;
every price and gradient over all fourteen starts at six maturities and over
200 uniform draws inside the bounds is finite and inside the strip, with no
clip and no penalty on the pricer.

### The range

With the leg chosen that way the round-off side of the range failure is gone
and only truncation is left. Worst error over five strikes and `T` in
0.25–2:

| `L`, `N` | Feller-satisfying | Feller-violating |
|---|---:|---:|
| 10, 256 | 4.92e-12 | 1.43e-02 |
| 16, 1024 | 3.70e-11 | 3.53e-04 |
| 24, 2048 | 3.23e-11 | 1.22e-06 |
| 28, 4096 | 2.05e-12 | 7.74e-08 |
| 32, 8192 | 8.16e-12 | 4.88e-09 |
| 40, 16384 | 2.13e-11 | 1.86e-11 |

The left column is flat at round-off rather than turning round and diverging
(before the leg rule it read 5.17e-07, 1.26e-05, 2.30e-04, 4.07e-02 with no
clip). So a single setting would now serve both sets — `L = 40, N = 16384` is
2.1e-11 on each — and it costs **64x** the term count of `L = 10, N = 256`.
`default_cos_settings` therefore branches on the Feller number for run time and
not for accuracy, and resolves **once** per solve from the initial guess: a
per-evaluation rule would make the residual discontinuous exactly where a
search crosses `4 kappa theta = 2 xi^2`.

## 2. The analytic gradient

`heston_charfn_gradient` returns `phi(u)` with its five parameter derivatives
from the affine solution rather than by differencing. Two are free — neither
`beta` nor `d` depends on `v0` or `theta`, so

```
d phi / d v0    = phi D
d phi / d theta = phi kappa (m T - 2 Lh)
```

— and the other three follow from `d beta/dp` and `d d/dp` by the chain rule
through `m`, `g`, `E`, `Lh`, `C` and `D`. Strategy and the ill-conditioning
claim: Cui, del Baño Rollin and Germano (2017), EJOR 263(2), section 3; their
rearrangement of the transform has no `g` in it, so the five derivatives here
are derived for this repository's arrangement and checked against differences
rather than against their formulas.

Measured:

| claim | value |
|---|---|
| `phi` against `qpl.models.heston` | 2.5e-16 absolute |
| derivatives against central differences of `phi` | 1.3e-09 – 4.6e-08 relative |
| price Jacobian against central differences of the **full pricer** | 6.8e-07 relative |
| speed-up over a 5-parameter central-difference Jacobian | **9.4x** |
| speed-up end to end on a real solve (price objective) | **3.4x** (41 ms vs 142 ms) |

The price gradient holds the truncation range fixed while the difference
quotient lets it move with the parameters. The agreement at 6.8e-07 is what
says the omitted range-motion term is the derivative of a 5e-12 truncation
error and not something that matters. A 5-strike by 6-maturity calibration
takes 41 ms (price) or 80 ms (implied volatility).

## 3. Synthetic recovery, and what it does not prove

Quotes generated from a known model, recovered from a heavily perturbed start
(`v0` doubled, `kappa` halved, `theta` and `xi` cut by 40%, `rho` moved to
-0.2). Absolute errors:

| set | objective | `v0` | `kappa` | `theta` | `xi` | `rho` |
|---|---|---:|---:|---:|---:|---:|
| reference | price | 2.9e-09 | 5.4e-07 | 3.3e-09 | 4.1e-07 | 1.3e-07 |
| reference | implied vol | 5.4e-14 | 7.0e-12 | 1.5e-13 | 1.2e-12 | 2.4e-13 |
| violating | price | 4.2e-10 | 7.6e-08 | 7.7e-09 | 2.9e-08 | 2.3e-09 |
| violating | implied vol | 7.6e-13 | 6.2e-12 | 5.7e-13 | 3.5e-11 | 1.2e-11 |

The asserted tolerance is not any of those numbers: it is `||r|| / s_min`
computed from the fit's own Jacobian, which on the reference implied-volatility
row is 5.43e-11 against a worst error of 7.02e-12 — tight to a factor of 8.

The price rows are five decimal orders worse than the implied-volatility rows,
and it is **not the objective's fault**. The quotes are implied volatilities
produced by a Brent inversion at `xtol = 1e-07`, so converting them back to
prices displaces the price objective's minimum by that tolerance times vega.
Fed prices directly, the same objective on the same grid from the same start
recovers `kappa` to **4.45e-12**. The oracle proved it before the mechanism was
found: QuantLib's independent Levenberg-Marquardt lands on the same displaced
point to 1.30e-10, with the same -5.41e-07 sign in `kappa`. Two solvers sharing
no code do not agree on a wrong answer by accident.

**None of this is evidence of identifiability.** The quotes are exact and the
errors are the solver's tolerances. What identifiability costs is below.

## 4. Noise, and the error bars

Gaussian implied-volatility noise, 20 independent draws, six maturities. The
empirical spread of the fitted parameters *across* draws against the
Jacobian-based standard error computed *within* one draw — two different
computations of the same quantity:

| noise | | `v0` | `kappa` | `theta` | `xi` | `rho` |
|---|---|---:|---:|---:|---:|---:|
| 5 bp | ratio | 1.07 | 1.08 | 0.99 | 1.14 | 1.16 |
| 20 bp | ratio | 1.09 | 1.10 | 1.00 | 1.15 | 1.23 |
| 20 bp | standard error | 3.79e-03 | 1.90e-01 | 1.62e-03 | 8.56e-02 | 3.29e-02 |

They agree to about 20% and the predicted errors are consistently on the
**small** side, which is the expected direction: the Gauss-Newton covariance
drops the second-order term and the model is not linear in `kappa`. That
agreement is what licenses quoting `CalibrationResult.standard_errors` as an
error bar at all.

Read the last row as the headline. At 20 bp — two-tenths of a volatility
point, a narrow real spread — a fit that reproduces the surface pins `theta` to
0.65% of its value and `kappa` only to 4.7%, `xi` to 8.6% and `v0` to 9.5%.
Quadrupling the noise quadruples the errors (measured ratios 3.68, 3.69, 4.14,
3.96, 4.28), so the problem is locally linear at this scale; `rho` drifts
first, which is where the nonlinearity shows up.

## 5. Identifiability: the flat direction

Condition number of the residual Jacobian **at the true parameters** on five
strikes, as maturities are added:

| maturities | implied-vol objective | price objective |
|---:|---:|---:|
| 1 | 6.7137e+07 | 6.4804e+07 |
| 3 | 5.6614e+02 | 7.8250e+02 |
| 6 | 4.7754e+02 | 9.5237e+02 |

The implied-volatility column falls monotonically. The price column falls by
five decimal orders and then **rises 22%** going from three maturities to six,
which contradicts the ordering the slice statement asserted without naming an
objective. The reason is an identity: `d sigma_i / dp = (d V_i / dp) / vega_i`,
so a vega-weighted price Jacobian **is** the implied-volatility Jacobian —
measured agreement 2.2e-16, condition numbers equal to fifteen digits. An
unweighted price objective therefore over-weights the long-dated at-the-money
cells, inflating the largest singular value faster than the smallest. This is
Gatheral (2006) chapter 3's argument for the implied-volatility objective, and
conditioning is where it shows.

At one maturity the singular values are `1.98, 6.42e-02, 3.27e-03, 1.31e-05,
2.95e-08` and the flat direction is

```
v0 +0.2755   kappa -0.9259   theta -0.0750   xi -0.2474   rho +0.0009
```

— a `kappa` direction with a `v0`/`xi` admixture, and no `rho` in it at all.

### The negative finding, priced

Six starts, one smile, exact quotes:

| start `kappa` | fitted `v0` | `kappa` | `theta` | `xi` | `rho` | implied-vol RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.09183 | 2.91811 | 0.24942 | 0.80548 | -0.50118 | 2.27e-06 |
| 1.0 | 0.10195 | 3.11094 | 0.24110 | 0.83200 | -0.50063 | 1.67e-06 |
| 2.0 | 0.10977 | 3.50826 | 0.23230 | 0.89383 | -0.49991 | 6.36e-07 |
| 8.0 | 0.00000 | 6.07068 | 0.23766 | 1.35949 | -0.49740 | 2.24e-06 |
| 12.0 | 0.00000 | 7.28789 | 0.23021 | 1.57280 | -0.49608 | 2.88e-06 |
| 6.0 | 0.04499 | 4.79408 | 0.23858 | 1.13015 | -0.49877 | 1.22e-06 |

Every fit reproduces the smile to better than three hundredths of a basis
point. The objective values span 1.0e-12 to 2.1e-11, all at the numerical
floor. The fitted `kappa` spans a factor of **2.50** around a true 4.0 and `xi`
a factor of 1.95 around a true 1.0; two of the six drive `v0` to its lower
bound. Meanwhile `rho` lands in [-0.5012, -0.4961] and `theta` in [0.2302,
0.2494] from every start: the two parameters the smile's *slope and level* see
are recovered, and the two the *term structure* sees are not.

The same six starts on three maturities all return `kappa = 4.00000` and
`xi = 1.00000` to better than 1e-05. Nothing about the solver changed.

## 6. The objective choice

The same noisy surface (20 bp) fitted three ways, ten draws:

| objective | implied-vol RMSE | price RMSE |
|---|---:|---:|
| prices | 0.002029 | **0.079817** |
| vega-weighted prices | **0.001908** | 0.081344 |
| implied volatilities | **0.001908** | 0.081348 |

Each objective wins on its own metric, by 6.3% and 1.9%, and that ordering
holds on every seed set and grid tried. Vega-weighted prices and implied
volatilities agree to four significant figures on both metrics and to 0.4% on
every parameter — the exact Jacobian identity of section 5 surviving the noise.

What does **not** hold is any ordering of the *parameter* accuracy. The ratio
of the price objective's mean absolute error to the implied-volatility
objective's, over two disjoint blocks of eight seeds:

| block | `v0` | `kappa` | `theta` | `xi` | `rho` |
|---|---:|---:|---:|---:|---:|
| A (seeds 3000–3007) | 1.48 | 1.76 | 1.17 | 1.18 | 1.27 |
| B (seeds 3008–3015) | 0.92 | 0.84 | 0.79 | 0.71 | 1.18 |

Block A says the price objective is uniformly worse; block B says it is
uniformly better on four of five. At 20 bp on this grid the objective choice is
worth a measurable amount on the *fit* and nothing that survives a seed change
on the *answer*. The slice statement asked which objective "wins"; the honest
answer is that the metric is not where the choice matters, the conditioning is.

## 7. Initialisation, and what a success rate measures

Fourteen starts — the truth, a mild perturbation, four corners of
`DEFAULT_BOUNDS`, five with the **wrong sign** of `rho` and six outside the
Feller region. On 20 bp quotes, **14/14** reach the best objective found, at
one, three and six maturities alike, and at three and six they all return the
same parameters to 1e-04.

That number was **11/14** until the payoff-leg rule of section 1 went in. The
three failures were not local minima: all three sat where the fixed call leg
was wrong by three decimal orders, so the residual was flat and each solver
stalled within 1e-02 of its start. Fixing the pricer fixed the optimiser.

And 14/14 is not an identifiability statement. At **one** maturity all fourteen
also reach the best objective within 1%, and land on `kappa` anywhere from
**7.197 to 15.004** against a true 4.0. No stopping rule a solver has can see
that; the condition number can, before the fit is run.

The oracle generalises the lesson. On the same twenty starts and the same exact
quotes, this package reaches `kappa = 4` from all twenty and QuantLib's
`HestonModel` + `LevenbergMarquardt` from **eleven**, several of its failures
pinned against a parameter constraint at `xi = 0.00000`. That is not a
ranking — the two differ in at least three ways at once (bounded trust region
against unconstrained MINPACK on QuantLib's internal transformations, analytic
against finite-difference Jacobian, and QuantLib's `ImpliedVolError` being a
vega-divided price residual rather than a real inversion) and this slice
separates none of them. It is the point restated: a start-grid success rate is
a property of a solver-and-pricer pair on a data set, never of the problem.

`n_starts` is multi-start local search. Measured honestly: from the worst start
in the grid, six uniform draws inside the bounds reach the same objective every
single start reaches. It **finds** the best objective and does not **improve**
on a single start, because on this problem there is nothing left to improve. It
was worth something while the pricer was broken.

## 8. Constraints

`DEFAULT_BOUNDS` keeps `v0, kappa, theta, xi > 0` and `|rho| < 1` and imposes
nothing else — no Feller constraint, no regularisation, no prior. The
Feller-violating set is recovered to 1e-06 and reported with
`feller_satisfied = False` and `feller_number = 0.0800`; QuantLib does the same
(5.93e-07), so neither library carries the unstated prior.

`method="lm"` is the unconstrained route. It refuses `bounds=` rather than
ignoring them, and from a start in the low corner of the box it converges — by
its own `xtol` — to `rho = +1.0140`, a correlation above one. The result
reports `in_domain = False` and `model = None` rather than coercing it into a
`HestonModel` that cannot exist.

## 9. What contradicted the plan

1. The plan said the COS **put** is the reliable leg under Heston and the call
   should come by parity. Neither leg is uniformly reliable; the range endpoint
   decides, and the first commit of this slice picked the wrong half of that
   before the oracle showed the other half (section 1).
2. The plan asked to "show that the condition number falls with more
   maturities". It does under the implied-volatility objective and does **not**
   under the price objective (6.48e+07 → 782 → 952), for a reason that is an
   exact identity (section 5).
3. The plan asked which objective "wins on which metric", presuming the metric
   decides. Each wins on its own metric by a few percent, and the parameter
   accuracy does not separate them at all at 20 bp (section 6).
4. The plan expected a measurable start-grid failure rate and a multi-start
   that rescues it. The failure rate was **the pricer**, and once it was fixed
   multi-start had nothing to rescue (section 7).
5. Clean synthetic recovery was five decimal orders better under the
   implied-volatility objective than under the price objective, which looked
   like an objective effect and is an artefact of writing synthetic quotes down
   as implied volatilities (section 3).

## Sources

Nothing below is quoted or transcribed; every number above is measured here.

- Gatheral (2006), *The Volatility Surface*, chapter 3 — calibration in
  practice, the objective choice, the `kappa`/`xi` flat direction.
- Cui, del Baño Rollin and Germano (2017), "Full and fast calibration of the
  Heston stochastic volatility model", *European Journal of Operational
  Research* 263(2), 625–638, section 3 — the analytic gradient of the
  characteristic function and the ill-conditioning along `kappa`–`xi`.
- Mikhailov and Nögel (2003), "Heston's stochastic volatility model:
  implementation, calibration and some extensions", *Wilmott Magazine* (July),
  74–79 — Heston calibration as a least-squares problem and the role of the
  initial guess.
- Levenberg (1944), *Quarterly of Applied Mathematics* 2, 164–168; Marquardt
  (1963), *SIAM Journal on Applied Mathematics* 11(2), 431–441 — the damped
  Gauss-Newton step, reached here through `scipy.optimize.least_squares`.
- Nocedal and Wright (2006), *Numerical Optimization* 2nd ed., chapter 10 — the
  Gauss-Newton covariance `sigma^2 (J^T J)^{-1}` and what it assumes.
- Fang and Oosterlee (2008), *SIAM J. Sci. Comput.* 31(2), 826–848, appendix —
  the COS truncation range and the `c4 = 0` convention this slice pays for.
- QuantLib-Python (BSD-3) — the independent calibration oracle, never the sole
  evidence for a claim.
