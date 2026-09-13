# Least-squares Monte Carlo: what the regression is for, and what it costs

Simulation runs forward and early exercise is decided backwards. That is the
whole difficulty of pricing an American option by Monte Carlo, and the
least-squares method is one answer to it: keep the forward simulation, and
replace the conditional expectation that backward induction needs with a
regression fitted on the sample. This note derives the estimator, states the
two biases it has and which direction each points, and records what they
measure at.

The algorithm and the basis: Longstaff, F.A. and Schwartz, E.S. (2001),
"Valuing American options by simulation: a simple least-squares approach",
*Review of Financial Studies* 14(1), 113–147, §1 (the method, stated on a
two-date example) and §2 (their numerical results and the weighted Laguerre
basis). The bias analysis: Glasserman, P. (2003), *Monte Carlo Methods in
Financial Engineering*, §8.6 (regression-based methods; the in-sample bias of a
fitted continuation value) and §8.7 (high- and low-biased estimators).
Convergence in the number of paths and basis functions: Clément, E., Lamberton,
D. and Protter, P. (2002), "An analysis of a least squares regression method for
American option pricing", *Finance and Stochastics* 6, 449–471. Every
derivation below is written out independently and every number was measured in
this repository. Two figures are quoted from a source — Longstaff and Schwartz's
simulation value 4.472 and its standard error 0.010 — and both are labelled
where they appear.

## What is actually being priced

A simulation cannot exercise continuously: a path visits finitely many dates.
The engine therefore prices a **Bermudan** option with exercise dates
`0 < t_1 < … < t_m = T`, `t_i = i T / m`, and says so in
`meta["exercise_style"]`. Writing `g` for the intrinsic value and `D(a, b)` for
the discount factor from `b` back to `a`, its value function satisfies

```
V(t_m, S) = g(S)
V(t_j, S) = max( g(S), C(t_j, S) )
C(t_j, S) = E[ D(t_j, t_{j+1}) V(t_{j+1}, S_{t_{j+1}}) | S_{t_j} = S ]
```

which is the same Bellman recursion `qpl.engines.tree.american` solves at
lattice nodes and `qpl.engines.pde.american` solves as a linear complementarity
problem. What is different is the state: a path sample has `n_paths` points at
every date and no local structure, so `C` cannot be formed as an average over
children. There is nothing to average.

The Bermudan value is **below** the continuously-exercisable one, and the gap
is not negligible at the frequencies a simulation can afford. Measured on the
CRR lattice at `n = 5000`, at the Longstaff–Schwartz specification
(`S = 36, K = 40, T = 1, r = 6%, q = 0, σ = 20%`), against the continuous value
4.4866721476:

| exercise dates `m` | Bermudan value | gap |
|---|---|---|
| 10 | 4.442643 | 4.4029e-02 |
| 50 | 4.477922 | 8.750e-03 |
| 250 | 4.485000 | 1.672e-03 |

Successive ratios over a fivefold refinement are 5.03 and 5.23, and the fitted
order against `1/m` is **1.0176** with a log-space RMS residual of 0.0080. So
the gap is first order in the exercise count — not the half order a barrier's
discrete-monitoring bias would give, which is why it is measured rather than
assumed. At the money (`S = K = 100, r = 5%, σ = 20%, T = 1`) the same study
gives gaps 5.694e-02, 1.185e-02 and 2.459e-03, fitted order 0.976 with a
residual of 0.0007.

This is the first correction that has to be applied before an LSM value is
compared with a lattice or a grid, and it is the reason the published
Longstaff–Schwartz figures are not continuous American values — a point Slice 2
already pinned for their finite-difference column and which applies equally to
their simulation column.

## The estimator

Following Longstaff and Schwartz §1, the continuation value is replaced by the
projection of the **realised** discounted cashflow onto a finite set of basis
functions of the current spot:

1. Start with the terminal cashflow `g(S_T)` on every path.
2. At each earlier date `t_j`, take the paths with `g(S_{t_j}) > 0` and regress
   their current cashflow, discounted back to `t_j`, on `φ(S_{t_j})`.
3. Exercise on a path where `g(S_{t_j})` exceeds the **fitted** continuation
   value; there, replace the cashflow by `g(S_{t_j})` and redate it to `t_j`.
4. The price is the sample mean of the discounted realised cashflows.

Three details carry the method.

**The in-the-money filter.** Out of the money the intrinsic value is zero and
the exercise decision needs no regression at all. Including those paths spends
a four-parameter basis describing a region where the answer is known, and makes
the fit worse where it matters. Longstaff and Schwartz make the restriction
explicitly.

**The regression decides; it does not value.** The reported price is the mean of
realised cashflows, never the mean of fitted continuation values at `t_1`. The
fitted value is a projection onto a small function space and its sample mean
absorbs the fit's own error; the realised cashflow under the resulting policy is
the value of an actual — suboptimal — stopping rule, which is a quantity with a
meaning. The three-path worked example in `tests/test_mc_american_lsm.py` makes
the difference concrete: a path whose fitted continuation is `0.4 e^{-0.05}`
contributes its actual `0.4 e^{-0.1}`.

**Which sample the policy is fitted on.** See the next section.

**The basis.** Both families are built on the moneyness `x = S / K`, never on
the raw spot, and at degree `d` both have `d + 1` columns:

```
polynomial   1, x, x², …, x^d
laguerre     1, w L_0(x), …, w L_{d-1}(x)        w = exp(-x/2)
```

with `L_k` from the standard recurrence `(k+1) L_{k+1} = (2k+1-x) L_k - k
L_{k-1}`. The scaling is not cosmetic. At a spot of 100 the unscaled weight is
`exp(-50)`; over a five-point sample the largest non-constant entry of the
unscaled design is `1.3e-14` against a constant column of 1, its condition
number is `5.0e+23`, and a regression on it returns the sample mean plus three
coefficients of noise. Scaled, the same design has condition number `8.5e+04`.
The default `lsm_degree=3, lsm_basis="laguerre"` is exactly the basis
Longstaff and Schwartz report: a constant plus the first three weighted Laguerre
functions.

`numpy.linalg.lstsq` solves the design matrix directly rather than the normal
equations, so the conditioning that matters is `cond(A)` and not its square.
Measured `cond(A)` at the Longstaff–Schwartz point with 100 000 paths and 50
dates: median `2.1e+05`, maximum `1.5e+07` — reported in `meta` because a
degree-5 fit on a sample that has collapsed into a narrow band of moneyness is
exactly the case where the price still looks plausible.

## Two biases, pointing opposite ways

**In-sample, biased high.** If the policy is fitted and applied on the same
paths, each path's exercise decision is informed by that path's own realised
future through the fitted coefficients. This is the estimator Longstaff and
Schwartz's table reports and it is what `lsm_in_sample=True` reproduces
(Glasserman §8.6).

**Out-of-sample, biased low.** Fit on one path set, value on an independent one.
The policy is then fixed before the valuation paths are seen, so the valuation
average is the value of a genuine suboptimal stopping rule and is a lower bound
in expectation (Glasserman §8.7). This is the default.

Measuring the first one takes care. The naive comparison — run in-sample, run
out-of-sample, subtract — carries the full sampling noise of two independent
valuations, and over 20 seeds at 50 000 paths it reads **-0.0022 ± 0.0028**:
the wrong sign, and not significant, on an effect that is really +0.0021. What
is measured instead is **paired in the valuation sample**: simulate two path
sets `A` and `B`, fit a policy on each, and value both policies on `A`. The
in-sample estimator is `A`'s policy on `A`, the out-of-sample one is `B`'s
policy on `A`, and their difference shares the valuation noise.

In-sample minus out-of-sample, paired, at the Longstaff–Schwartz point with 50
dates, over 24 seeds:

| paths | laguerre d=2 | d=3 | d=5 |
|---|---|---|---|
| 2 000 | +0.03794 ± 0.00663 | +0.04585 ± 0.00614 | +0.07413 ± 0.00729 |

and for the polynomial basis at the same sizes, +0.04396, +0.05219, +0.07436,
with comparable standard errors. Every entry is positive at better than five of
its own standard errors.

Two structural facts follow from that table and are asserted separately:

- The bias **grows with the number of basis functions**. Going from four
  coefficients to six adds `+0.02827 ± 0.00680` (laguerre) and
  `+0.02218 ± 0.00679` (polynomial). This is the `k` in an `O(k/N)` overfitting
  bias, and it is Glasserman §8.6 read as a measurement. The three-to-four
  coefficient step is *not* resolved at this seed count (+0.00791 ± 0.00776).
- The bias decays like `1/N`, not `1/√N`: +0.04585 at 2 000 paths and
  +0.01114 ± 0.00212 at 8 000, a fitted order of **1.020**. That different rate
  is why the in-sample estimator is unusable at two thousand paths and
  invisible at a hundred thousand — at 2 000 the bias is 0.046 against a
  single-run standard error of about 0.043; at 100 000 it is 0.0021 against
  0.006.

## What a finite sample costs the out-of-sample estimator

The low bias has two causes: too few basis functions to describe the
continuation value, and too few paths to fit the ones there are. The second is
why a **richer** basis can make it worse.

Out-of-sample bias against the 50-date lattice Bermudan 4.477922, 2 000 paths,
24 seeds:

| basis | d=2 | d=3 | d=5 |
|---|---|---|---|
| laguerre | -0.02630 ± 0.00905 | -0.01946 ± 0.00814 | -0.03416 ± 0.00791 |
| polynomial | -0.02716 ± 0.00866 | -0.02447 ± 0.00751 | -0.03370 ± 0.00832 |

Paired seed by seed, degree 3 beats degree 5 by `+0.01470 ± 0.00544` (laguerre)
and `+0.00923 ± 0.00457` (polynomial): at this sample size the richer basis
produces a **worse policy**. The penalty is gone by 20 000 paths, where the same
contrast reads `-0.00016 ± 0.00111`. Read with the previous section, degree 5 at
2 000 paths raises the in-sample bias by 0.028 *and* lowers the out-of-sample
value by 0.015; no choice of `lsm_in_sample` rescues a basis too rich for its
sample.

The basis **family**, at a fixed size, is not resolvable here: laguerre minus
polynomial out of sample reads `+0.00086 ± 0.00299`, `+0.00500 ± 0.00328` and
`-0.00046 ± 0.00212` at degrees 2, 3 and 5, while the degree contrast within a
family is two to three standard errors. QuantLib says the same thing with its
own two families on the same point — Monomial 4.473001 against Laguerre
4.470848, a difference of 0.36 of a standard error. The number of functions is
the knob; which spanning set they come from is not.

The residual low bias at usable path counts is **strongly point-dependent**, and
this is the practical warning of the whole note. At 100 000 antithetic paths:

| point | dates | mean out-of-sample bias vs the lattice Bermudan |
|---|---|---|
| Longstaff–Schwartz row 1 | 50 | -1.5e-03 ± 1.4e-03 (10 seeds) |
| at the money | 50 | -1.05e-02 ± 2.6e-03 (10 seeds) |
| at the money | 250 | -1.88e-02 ± 4.0e-03 (10 seeds) |

An order of magnitude between two vanilla puts at the same settings. It decays
like `N^{-1/2}` — at the money with 250 dates, 3.29e-02 at 20 000 paths,
2.30e-02 at 50 000, 1.88e-02 at 100 000 — which is the same rate as the standard
error, so it does not disappear inside the noise as the sample grows. An LSM
tolerance derived at one specification does not transfer to another.

It is also not an artefact of this implementation. QuantLib's
`MCAmericanEngine`, which is an independent least-squares implementation with
its own calibration pass, sits `-9.01e-03 ± 3.50e-03` below the same lattice
Bermudan over five seeds at the ATM point with 250 dates. Same sign, same order
of magnitude; the two do **not** agree on the size (they differ by
`9.8e-03 ± 5.3e-03`, 1.9 standard errors), and closing that would require
calibration samples matched path for path.

## Reproducing the published row

Longstaff and Schwartz's Table 1 row 1 is `S = 36, K = 40, T = 1, r = 6%,
q = 0, σ = 20%` at 50 exercise dates, 100 000 paths drawn as 50 000 antithetic
pairs, a constant plus three weighted Laguerre functions, policy and valuation
on the same paths. Their simulation value is 4.472 with a standard error of
0.010.

| estimator | value | stderr | reference | gap |
|---|---|---|---|---|
| this engine, in-sample | 4.467550 | 6.032e-03 | published 4.472 | -4.45e-03 (0.74 σ) |
| this engine, out-of-sample | 4.472996 | 6.057e-03 | lattice Bermudan 4.477922 | -4.93e-03 (0.81 σ) |
| QuantLib, Monomial order 3 | 4.473001 | 6.091e-03 | lattice Bermudan 4.477922 | -4.92e-03 |

with `z = -1.11` between QuantLib's run and this engine's out-of-sample one.
The in-sample run is compared with the published number because that is the
estimator the paper reports; the out-of-sample run is compared with the lattice
Bermudan, because comparing a low-biased estimator with a published high-biased
one would be comparing two different estimators of the same quantity.

What is *not* claimed is equality. Two Monte Carlo runs at different seeds that
agreed to four figures would be evidence of a copied answer, not of a working
estimator.

## Three engines on one number

At the money, from three discretisations that share only the model:

```
Leisen-Reimer lattice, n = 8001        6.09033758
PSOR grid, n_s = n_t = 800             6.08995244
LSM, 250 dates, 100 000 paths          6.05330   (stderr 1.31e-02)
```

The slice this was written for expected the simulation's budget to be "its
standard error plus the Bermudan gap". Measured, that is not enough, and the
missing term is the largest of the three:

```
Bermudan gap at m = 250                     2.46e-03
finite-sample low bias at 100 000 paths     1.89e-02
three standard errors                       3.93e-02
--------------------------------------------------
budget                                      6.07e-02
```

Six hundredths on a value of 6.09 is 1.0%, against the 1.5e-03 the
lattice-and-grid row of `qpl.cases` gets. That is the honest price of a third
discretisation that is a sample rather than a mesh, and it is written down
rather than rounded into a tolerance.

## The exercise boundary

The boundary implied by the fitted policy is the largest sampled spot at which
exercising beat the fitted continuation (for a put; the smallest, for a call).
Three things make it a loose estimator, and all three are one-sided or
noise-increasing: it is an **upper order statistic** of a sample; it is a
**Bermudan** boundary, so exercise is worthwhile a little higher than under
continuous exercise, an `O(dt)` effect; and near the boundary intrinsic and
continuation differ by very little, which is exactly where a small fit error
moves the crossing a long way.

Measured at the Longstaff–Schwartz point, 100 000 paths, 50 dates, degree-3
laguerre, over seeds 1, 2, 3, against the PSOR boundary at `n_s = n_t = 800`:

| t | PDE | tree | LSM (3 seeds) |
|---|---|---|---|
| 0.10 | 33.079 | 32.973 | 33.704 / 33.676 / 33.726 |
| 0.25 | 33.438 | 33.200 | 33.740 / 33.517 / 33.741 |
| 0.50 | 33.978 | 33.782 | 34.285 / 34.468 / 34.349 |
| 0.75 | 35.056 | 34.841 | 35.198 / 35.192 / 35.215 |
| 0.90 | 36.315 | 36.034 | 35.951 / 36.086 / 36.030 |

Worst deviation 0.65 on a boundary running 33 to 36, i.e. 2.0%; the lattice's
own boundary sits 0.10 to 0.28 below the grid's, which sets the scale of what
agreement can mean. The sign is informative rather than random: the LSM boundary
is **above** the true one early in the option's life, so the fitted policy
exercises too eagerly there — the same suboptimality the low bias in the price
measures, seen directly.

## The no-dividend American call, and why a fraction is not a diagnostic

An American call on a non-dividend-paying stock is never exercised early: the
continuation value dominates `S - K` by `K (1 - e^{-r(T-t)})` at every date. A
lattice reproduces that as an exact identity — the Bellman maximum is a no-op at
every node and the two engines agree bit for bit. LSM cannot, because it
compares intrinsic against a *fitted* continuation.

Measured at `S = K = 100, r = 5%, σ = 20%, T = 1`, 50 000 antithetic paths,
degree-3 laguerre:

| dates | early-exercise fraction | price | Black–Scholes | gap |
|---|---|---|---|---|
| 50 | 1.9e-03 | 10.4383 | 10.4506 | -1.2e-02 (0.27 σ) |
| 250 | 0.219 | 10.4570 | 10.4506 | +6.4e-03 (0.14 σ) |

A fifth of the paths stop early at 250 dates while the price stays inside one
standard error. The mechanism: near expiry the true early-exercise premium is
`K (1 - e^{-r dt}) ≈ K r dt`, which at `dt = 1/250` is 0.02 on a value of ten —
the same size as the regression's own approximation error over the in-the-money
range. The comparison at the last few dates is therefore close to a coin flip,
and losing it costs almost nothing, because exercising at `T - dt` instead of
`T` is worth about `K r dt`.

So the early-exercise fraction measures how often the fit's error exceeds a
premium that shrinks like `dt`. It is a statement about the basis and the date
spacing, not about whether the policy is right, and it is reported in `meta`
with that caveat attached rather than used as a health check.

## What this engine does not do

- **Greeks.** The price depends on the spot through the payoff *and* through an
  exercise policy that is a step function of coefficients estimated from the
  sample. A bump moves the policy, so a difference quotient of two LSM prices
  is not an estimate of `dV/dS`, and the pathwise derivative of the stopped
  payoff is valid only for a policy held fixed. `NotSupportedError`, with that
  message.
- **Upper bounds.** The out-of-sample estimator is low-biased and there is no
  high-biased companion here, so the value is reported without a bracket. The
  dual/martingale construction that supplies one (Andersen–Broadie, Rogers,
  Haugh–Kogan) is a separate method and a separate slice.
- **Control variates and stratification.** Refused, with reasons: a control
  needs a known mean *under the stopping rule the regression produced*, and the
  discounted spot is a martingale only up to a fixed date; stratification needs
  a single scalar to partition and a Bermudan payoff is driven by one normal per
  exercise date. Antithetic sampling does compose, because the pair average is
  taken on the realised cashflow *after* each path's own exercise decision.

## Where the numbers live

- Engine and derivation: `src/qpl/engines/mc/american.py`.
- Mechanics, degenerate limits, determinism, the call: `tests/test_mc_american_lsm.py`.
- Published row, exercise-frequency study, three-engine agreement, boundary:
  `tests/test_lsm_american.py`.
- The two biases, the basis and the degree: `tests/test_lsm_american_bias.py`.
- Benchmark rows with their evidence classes and citations:
  `qpl.cases.american_black_scholes` (`AMERICAN_LSM_CASES`).
- QuantLib comparison: `tests/oracle/test_lsm_vs_quantlib.py`.
- Runnable comparison: `examples/american_put_cross_method.py`.
