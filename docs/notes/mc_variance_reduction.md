# Variance reduction: three estimators, and what each is actually worth

A Monte Carlo price is a random variable. Variance reduction replaces the
estimator with a different one that has the same mean and less spread, so the
interesting question is never "does it work" — all three methods here are
unbiased by construction — but "how much, at what cost, and on which payoff".
This note derives the three estimators, states what each cannot do, and records
what they measure at on a European call, an out-of-the-money call and a
cash-or-nothing digital.

Ideas and their standard analysis: Glasserman (2003), *Monte Carlo Methods in
Financial Engineering*, §4.1 (control variates), §4.2 (antithetic variates),
§4.3 (stratified sampling). Every derivation below is written out
independently and every number was measured in this repository; nothing is
quoted from that book.

## The rules of the comparison

**Cost is counted in standard normal draws.** Antithetic sampling evaluates the
payoff twice per draw, so an antithetic run with `2N` paths and a plain run
with `N` paths cost the same number of normals, and that is the pairing used
throughout. Quoting the gain at equal *paths* silently gives antithetic free
randomness; both conventions appear below wherever they differ.

**The variance compared is the variance of the estimator**, measured over 50
independent seeds — not inferred from the reported standard errors, since those
formulas are themselves part of what is being checked. A variance estimated on
50 seeds has relative standard deviation `sqrt(2/49) = 20%`, so a *ratio* of
two of them carries roughly 29%; the asserted bands are a factor of 1.7 either
way, which is inside the [0.47, 2.11] 99% band of such a ratio.

**The points.** `S = 100`, `r = 5%`, `q = 0`, `sigma = 20%`, `T = 1`
throughout. `atm_call` is `K = 100`, `otm_call` is `K = 120` (about 19% of the
sample finishes in the money), `digital_call` is a cash-or-nothing call at
`K = 100` paying 1. The plain estimator runs 20 480 draws.

## Antithetic variates (§4.2)

Draw `m` normals and use each twice, as `Z` and `-Z`. The estimator's unit is
the **pair average**

    P_i = (Y(Z_i) + Y(-Z_i)) / 2,   Y(z) = e^{-rT} f(S_T(z)),

and the `P_i` are i.i.d. even though the `2m` payoffs are not. Writing
`rho_a = corr(Y(Z), Y(-Z))`,

    Var(P) = Var(Y) (1 + rho_a) / 2,

so `m` draws buy `Var(Y)/m` plain against `Var(Y)(1 + rho_a)/(2m)` antithetic:

    factor at equal normal draws      = 2 / (1 + rho_a)
    factor at equal payoff evaluations = 1 / (1 + rho_a)

The method helps exactly when `rho_a < 0`. For a monotone payoff that is
guaranteed: `z -> -z` reverses the order of the driving normal, and a monotone
transformation of a reversed order is negatively correlated with itself.
Nothing stronger is guaranteed: the second convention's floor is `1/2` at
`rho_a = 1` (a payoff symmetric in `Z`, where pairing does nothing but waste
half the evaluations) and its ceiling is unbounded only in the limit where the
pair average is constant.

| point | `rho_a` | predicted (draws) | predicted (evals) | measured |
|---|---:|---:|---:|---:|
| atm_call | −0.5009 | 4.01 | 2.00 | 4.59 |
| otm_call | −0.1399 | 2.33 | 1.16 | 3.04 |
| digital_call | −0.7868 | 9.38 | 4.69 | 9.32 |

**The digital gets the largest gain, which is the opposite of what convexity
suggests.** The mechanism is not convexity at all. For a cash-or-nothing call
the in-the-money boundary sits at `z* = -0.15`, and

    1{Z > z*} + 1{-Z > z*} = 1   unless |Z| < 0.15,

so the pair average is *exactly* `1/2` on 88% of the sample and the estimator
only has to resolve the remaining band. A bounded payoff whose reflection
nearly complements it is the best case for the method; an out-of-the-money call
— where both legs of a pair are usually zero, so there is little order to
reverse — is close to the worst.

## Control variates (§4.1)

The discounted terminal spot `X = e^{-rT} S_T` is a martingale under the
pricing measure, so its mean is known in closed form and needs no simulation:
`E[X] = e^{-rT} S_0 e^{(r-q)T} = S_0 e^{-qT}`. The estimator is

    V = mean(Y) - b (mean(X) - E[X]),

unbiased for every `b`, with variance minimised at `b* = Cov(Y,X)/Var(X)` and

    factor = 1 / (1 - rho^2),   rho = corr(Y, X).

| point | `rho` | predicted | measured |
|---|---:|---:|---:|
| atm_call | +0.9246 | 6.89 | 7.58 |
| otm_call | +0.7522 | 2.30 | 2.71 |
| digital_call | +0.7697 | 2.45 | 2.11 |

`1/(1 - rho^2)` is violently sensitive near 1: the correlation falls by 0.17
from the ATM call to the OTM call and the gain falls by two thirds. The
discounted spot is a good control for a payoff that is affine in `S_T` over the
region that matters, which is why it works best at the money and only
moderately on a step payoff — it explains *where* `S_T` lands, not *which side
of the strike*.

### The coefficient is estimated, so the estimator is biased

`b*` is unknown. This engine estimates it by ordinary least squares **on the
same sample it then corrects**, which makes the result a ratio of sample
moments and therefore biased. The honest treatment is to measure the bias
rather than wave at it.

First, how far the fitted coefficient is from one fitted on an independent
pilot sample of the same size (mean over 20 seed pairs, ATM call):

| N | mean \|b_same − b_pilot\| |
|---:|---:|
| 1 000 | 0.017186 |
| 4 000 | 0.008595 |
| 16 000 | 0.003800 |
| 64 000 | 0.002354 |
| 256 000 | 0.001061 |

Fitted decay **0.4952** (log-space RMS residual 0.0675): the `N^{-1/2}` of an
ordinary regression slope, with no floor.

Second, what that costs in price. Mean over 20 seeds of
`value(b fitted here) − value(b fitted on a pilot)`:

| N | mean difference | mean \|difference\| |
|---:|---:|---:|
| 2 000 | −4.830e−03 | 6.929e−03 |
| 8 000 | −1.191e−03 | 1.550e−03 |
| 32 000 | −2.610e−04 | 3.545e−04 |
| 128 000 | −5.848e−05 | 6.701e−05 |

A 64-fold increase in `N` shrinks the gap 82-fold: **order 1.06**, i.e. the
`O(1/N)` bias against an `O(N^{-1/2})` standard error. At `N = 128 000` the
bias is about 1% of one standard error and cannot move a confidence interval.
The sign is consistently negative — fitting `b` on the same sample removes
slightly more than the true regression would, so the estimator sits a little
low.

The reported standard error is the sample standard deviation of the regression
residuals with `ddof = 2`: one degree of freedom for the mean, one for the
fitted slope.

## Stratified sampling (§4.3)

For terminal sampling (`n_steps = 1`) the terminal spot is a monotone function
of a single standard normal, so stratifying that normal stratifies the payoff.
`K` equal-probability strata with `m = n_paths/K` draws each — proportional
allocation, which for equal-probability strata means equal allocation — give

    V = (1/K) sum_i mean_i(Y),    Var(V) = (1/K^2) sum_i s_i^2 / m,

and the gain is `Var(Y) / Var_within`: whatever variance survives *inside* a
stratum is what is left.

### On a vanilla the gain is linear in `K`, not quadratic

| K | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| atm_call | 14.0 | 33.2 | 58.3 | 110.4 | 249.2 | 506.5 |
| otm_call | 4.9 | 12.0 | 22.1 | 42.4 | 94.2 | 193.8 |

Fitted exponent in `K`: **1.0157** (residual 0.0652) and **1.0384** (0.0581).
Doubling the strata doubles the gain.

The textbook heuristic for a *smooth* integrand — a stratum of width `O(1/K)`
in probability leaves a within variance of `O(1/K^2)`, hence a gain of
`O(K^2)` — does not apply, and the reason is the tail. Equal-probability strata
make the outermost one `Z > Phi^{-1}(1 - 1/K)`, unbounded on the right; a call
payoff grows exponentially there, its conditional variance does not shrink with
`K` anywhere near fast enough, and that single stratum sets the floor. The gain
is then `K` times `Var(Y)/Var(Y | tail)`, which is linear. The standard repair
is optimal rather than proportional allocation (§4.3.1), which puts draws where
the variance is; it is not implemented here, and the linear behaviour is the
honest description of what proportional allocation does to an unbounded payoff.

### On a digital the gain has a closed form, and is not monotone in `K`

A cash-or-nothing payoff is constant inside every stratum except the one
containing the in-the-money boundary `u* = Phi(z*)`. All the residual variance
is in that one stratum. If `f = frac(K u*)` locates the boundary inside it, the
conditional in-the-money probability there is `1 - f`, the stratum variance is
`f(1-f)`, and

    gain = K p (1 - p) / (f (1 - f)),   p = N(d2).

Measured at `u* = 0.440382`, `p = 0.559618`, 50 seeds, 20 480 draws:

| K | f | predicted | measured | ratio |
|---:|---:|---:|---:|---:|
| 8 | 0.5231 | 7.90 | 9.00 | 1.14 |
| 16 | 0.0461 | 89.64 | 108.82 | 1.21 |
| 20 | 0.8076 | 31.73 | 48.50 | 1.53 |
| 32 | 0.0922 | 94.19 | 115.32 | 1.22 |
| 40 | 0.6153 | 41.65 | 75.55 | 1.81 |
| 64 | 0.1845 | 104.84 | 101.26 | 0.97 |
| 80 | 0.2306 | 111.13 | 102.36 | 0.92 |
| 128 | 0.3689 | 135.49 | 150.07 | 1.11 |
| 160 | 0.4612 | 158.68 | 191.61 | 1.21 |
| 256 | 0.7379 | 326.19 | 385.85 | 1.18 |
| 320 | 0.9223 | 1100.97 | 1014.18 | 0.92 |
| 512 | 0.4757 | 505.91 | 491.01 | 0.97 |

Two consequences. **The gain is not monotone in `K`**: going from 16 strata to
20 is predicted to fall 89.64 → 31.73 and measures 108.82 → 48.50; from 320 to
512, predicted 1100.97 → 505.91 and measured 1014.18 → 491.01. More strata can
buy less. And **it is a property of the contract, not of the method**: move the
strike and the same `K` lands at a different `f`.

The fit is worst where the predicted gain is smallest (1.81 at `K = 40`, 1.53
at `K = 20`), which is why the asserted scan is the six powers of two and the
non-monotone pairs are asserted as orderings rather than levels.

### What stratification refuses to do

With `n_steps > 1` there is no single normal driving the terminal price, so
"stratify the normal" has no referent. The nearby wrong thing — stratify the
first increment and draw the rest freely — is a legal sampler and a nearly
useless one, since the terminal value is then stratified through one of
`n_steps` contributions. The construction that works stratifies a projection of
the Brownian path (usually its endpoint) and fills the rest in with a Brownian
bridge; that is a *different sampler*, not different bookkeeping on this one,
and `NotSupportedError` says so and names it. Antithetic and control variates
have no such restriction: both are transformations of whatever normals the path
sampler drew.

Antithetic and stratified do not compose here either: reflecting a stratified
draw sends it into the mirror stratum, so the pair unit and the stratum unit
partition the sample differently. Symmetric stratification is a third sampler,
and the combination is rejected rather than silently reinterpreted.

## The combinations

| point | antithetic | control | stratified | anti+ctrl | strat+ctrl |
|---|---:|---:|---:|---:|---:|
| atm_call | 4.59 | 7.58 | 110.40 | 90.80 | 577.80 |
| otm_call | 3.04 | 2.71 | 42.41 | 92.35 | 84.61 |
| digital_call | 9.32 | 2.11 | 101.26 | 9.73 | 82.32 |

None of these is the product of its parts, and the ordering is not stable.
Antithetic+control on the OTM call is 92.35 against a product of 8.24, because
the control variate correlates far better with the *pair-averaged* payoff
(+0.9643) than with the raw one (+0.7522) — pair-averaging removes most of the
payoff's asymmetry before the regression sees it. On the digital, adding the
control to stratification makes things **worse** (82.32 against 101.26): the
residual variance lives in one stratum, and a coefficient fitted over the whole
sample is fitted mostly on strata with no variance left to explain.

## The standard error is part of the estimator

Each method has its own formula, and reporting the plain one for a reduced
estimator is the standard way to publish an interval that does not cover:

- antithetic: `sd(pair averages, ddof=1) / sqrt(m)` — over `m` pairs, *not*
  over the `2m` dependent payoffs. On the ATM call the wrong formula is more
  than 1.4x larger, so it is not a rounding matter.
- control variate: `sd(Y - b(X - E X), ddof=2) / sqrt(N)`.
- stratified: `sqrt( sum_i s_i^2 / (K^2 m) )`, with `s_i^2` the within-stratum
  `ddof=1` variance, which is why at least two paths per stratum are required.

All three are recomputed from the realised sample in
`tests/test_mc_variance_reduction.py` and compared with `==`.

Calibration, measured rather than assumed — coverage of the reported 95%
interval over 30 seeds x 5 120 draws:

| point | none | anti | ctrl | strat | a+c | s+c |
|---|---:|---:|---:|---:|---:|---:|
| atm_call | 26 | 30 | 29 | 29 | 29 | 29 |
| otm_call | 29 | 30 | 30 | 29 | 30 | 29 |
| digital_call | 26 | 28 | 28 | 30 | 28 | 30 |

Aggregate 519/540 = 0.961. The per-cell tolerance is binomial: under true 95%
coverage, `P(X <= 23) = 5.7e-04` for `X ~ Bin(30, 0.95)`, so requiring at least
24 is a one-sided test at the 0.06% level. The stratified column is the
informative one — its intervals are 30x narrower than the plain ones, so
29/30 there is a much stronger statement about the formula than 29/30 on the
plain estimator.

Unbiasedness at 200 000 draws: `|z|` against the closed form runs 0.05 to 1.58
over all eighteen (point, estimator) cells.

## Greeks

`greeks_european` revalues at bumped inputs through `price_european` with the
same config, hence the same seed, hence the same normals: common random numbers
survive variance reduction untouched. Standard deviation of the CRN bump delta
over 20 seeds, ATM call, 20 480 draws, `h = 1e-2` (analytic delta 0.636830):

| estimator | sd(delta) | sd ratio | variance ratio |
|---|---:|---:|---:|
| none | 0.004481 | 1.0 | 1.0 |
| antithetic | 0.001175 | 3.8 | 14.5 |
| control_variate | 0.001639 | 2.7 | 7.5 |
| stratified | 0.000254 | 17.6 | 311.2 |
| antithetic+control | 0.001060 | 4.2 | 17.9 |
| stratified+control | 0.000367 | 12.2 | 149.2 |

Two things to read out of that table. Stratification helps the **Greek more
than the price** (311x against 110x at the same setting): under common random
numbers the difference quotient is driven by the paths near the strike, and
stratification is precisely control over where paths land. And adding the
control variate to the stratified estimator makes delta *worse* (0.000367
against 0.000254) while making the price five times better — the coefficient is
refitted on each bumped sample, so `b_up - b_dn` is noise that common random
numbers cannot cancel. Fitting `b` once and reusing it across the bumps would
remove that; it is a different estimator with a different bias and is not what
this slice ships.

## Summary: what the slice statement got wrong

1. Antithetic was expected to be weakest on the digital; it is **strongest**
   there (9.32), and the mechanism is complementarity under reflection, not
   convexity.
2. Stratification was expected to give "a large gain for the vanilla and a very
   large gain for the digital". At `K = 64` they are 110.4 and 101.3 — the
   same, near enough — and the two behave completely differently under `K`:
   the vanilla is a clean `O(K)`, the digital follows `K p(1-p)/(f(1-f))` and is
   not monotone.
3. The vanilla's stratified gain was expected to be `O(K^2)` by the smooth
   integrand argument. The unbounded tail stratum makes it `O(K^{1.02})`.
4. The combinations were expected to compose roughly multiplicatively. They do
   not, in either direction: antithetic+control on the OTM call beats the
   product elevenfold, and stratified+control on the digital is *worse* than
   stratification alone.
5. One measurement was briefly mis-attributed in this repository and is
   recorded rather than deleted: the digital's non-monotonicity was first
   pinned at `K = 16` beating `K = 64` (108.82 against 101.26), which is a real
   measurement but the *opposite* of what the law predicts there (89.64 against
   104.84) — a 50-seed sampling accident, now replaced by two pairs where the
   prediction and the measurement agree on the direction.

## Where this lives

- Estimators: `src/qpl/engines/mc/variance_reduction.py`; config fields
  `MCConfig.variance_reduction` and `MCConfig.n_strata`.
- Correctness: `tests/test_mc_variance_reduction.py`.
- Measurement: `tests/test_mc_variance_ratios.py`.
- Rows: `src/qpl/cases/mc_variance_reduction.py` (fifteen `STATISTICAL` rows,
  log-space bands).
- Example: `examples/mc_variance_reduction.py`, also `--case digital`.
