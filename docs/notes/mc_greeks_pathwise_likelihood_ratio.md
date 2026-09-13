# Monte Carlo Greeks: three estimators, and when each one is the right one

A discounted price is an integral with a parameter in it,

    V(theta) = e^{-rT} ∫ f(x) p(x; theta) dx,

and there are exactly three places the parameter can sit: inside the payoff
`f` (through the state), inside the density `p`, or inside the discount factor.
Each of the three estimators below differentiates one of them and pays for it
somewhere else. The choice between them is not a matter of taste — on one
payoff the gap is a factor of 5, on another it is a factor of 865, and on a
third one of them returns exactly the wrong answer with no warning at all.

Ideas and their standard analysis: Glasserman (2003), *Monte Carlo Methods in
Financial Engineering*, §7.1 (finite differences and the bias/variance
trade-off), §7.2 (pathwise derivatives and the conditions for unbiasedness),
§7.3 (the likelihood-ratio method), §7.4 (combining them); Broadie and
Glasserman (1996), "Estimating security price derivatives using simulation",
*Management Science* 42(2), 269–285, the paper that introduced both estimators
for option Greeks with the Black–Scholes examples; Chen and Glasserman (2007),
"Malliavin Greeks without Malliavin calculus", *Stochastic Processes and their
Applications* 117, 1689–1723, for combined estimators generally. Every
derivation below was written out independently and every number was measured in
this repository; nothing is quoted from those sources.

Implementation: `src/qpl/engines/mc/greeks.py`. Unbiasedness:
`tests/test_mc_greeks.py`. Variance: `tests/test_mc_greeks_variance.py`.
Digital: `tests/test_digital_mc.py`. Asian: `tests/test_asian_mc.py`. Rows:
`qpl.cases.european_black_scholes.MC_GREEK_CASES`,
`qpl.cases.digital_black_scholes.DIGITAL_MC_GREEKS_CASES`,
`qpl.cases.mc_variance_reduction.MC_GREEKS_VARIANCE_CASES`. Runnable:
`examples/mc_greeks_estimators.py`.

## The terminal law, written once

Everything below is against the exact Black–Scholes terminal law, with
`mu = r - q`:

    S_T = S_0 exp((mu - sigma²/2) T + sigma sqrt(T) Z),   Z ~ N(0, 1),
    Z   = (log(S_T / S_0) - (mu - sigma²/2) T) / (sigma sqrt(T)).          (1)

`Z` is **recovered** from the sample by (1) rather than carried out of the
sampler. That is deliberate. With `n_steps > 1` the sampler draws `n_steps`
normals per path, but the terminal distribution is still exactly (1) with a
single standard normal, so the derivations apply unchanged at any step count
and the recovered `Z` is the one they are written in. For an antithetic pair
the recovered `Z` of the mirrored path is exactly `-Z`, as it must be.

## Pathwise (§7.2)

Differentiate the *sample*. Write `S_T` as a function of the parameter and a
parameter-free random input `Z`, and estimate `E[d f(S_T)/d theta]`. Holding
`Z` fixed in (1):

    dS_T/dS_0    = S_T / S_0                                               (2)
    dS_T/dsigma  = S_T (sqrt(T) Z - sigma T)                               (3)
    dS_T/dr      = S_T T            (q held fixed, so dmu/dr = 1)          (4)
    dS_T/dT      = S_T ((mu - sigma²/2) + sigma Z / (2 sqrt(T)))           (5)

Equation (3) is the form Broadie and Glasserman write with `W_T` in it; (3) is
used here because it needs no division by `sigma`. With `Y = e^{-rT} f(S_T)`:

    delta_pw = e^{-rT} f'(S_T) S_T / S_0
    vega_pw  = e^{-rT} f'(S_T) S_T (sqrt(T) Z - sigma T)
    rho_pw   = T e^{-rT} (f'(S_T) S_T - f(S_T))                            (6)
    theta_pw = e^{-rT} (r f(S_T) - f'(S_T) S_T [(mu - sigma²/2) + sigma Z / (2 sqrt(T))])

**The second term in (6) is the part a naive reading drops.** It is the
discount factor's own `-T e^{-rT}`, and for a call it makes the bracket
`S_T - (S_T - K) = K` in the money and `0` out of it, so the whole estimator
collapses to `T e^{-rT} K 1{S_T > K}` — whose mean is `K T e^{-rT} N(d2)`, the
closed-form rho, exactly. That collapse is asserted path by path in
`tests/test_mc_greeks.py`: drop the `- f(S_T)` and the sample stops being a
constant times an indicator, which the assertion sees immediately.

`theta` here is `-dV/dT` with `T` the time to expiry, the same convention the
analytic engine uses.

**When it is legitimate.** The interchange of derivative and expectation needs
`f(S_T(theta))` to be almost surely differentiable in `theta` and Lipschitz
with an integrable Lipschitz constant (Glasserman's conditions in §7.2.2). A
call satisfies both: it fails to be differentiable only at `S_T = K`, a null
event. A **digital does not**, and the failure mode is the worst possible one
— see below.

**Pathwise gamma does not exist** for any payoff in this package, because it
needs `f''`, a Dirac mass for a call and worse for a digital. What is used
instead is the mixed LR–PW estimator of §7.4: write
`delta = e^{-rT} E[f'(S_T) S_T] / S_0`, differentiate that expectation with the
`S_0` score, and collect the explicit `1/S_0`:

    gamma_mixed = e^{-rT} f'(S_T) S_T (Z / (sigma sqrt(T)) - 1) / S_0²     (7)

One derivative of the payoff, one of the density. This is what
`greeks_estimator="pathwise"` returns for gamma, and the result labels it
`pathwise_lr_mixed` in `meta["estimator"]` rather than calling it pathwise.

## Likelihood ratio (§7.3)

Differentiate the *density* and leave the payoff alone:

    dV/dtheta = e^{-rT} ∫ f(x) [d log p(x; theta)/dtheta] p(x; theta) dx.

The density of `S_T` is lognormal, `p(x) = phi(z) / (x sigma sqrt(T))` with `z`
from (1). Differentiating
`log p = -log x - log(sigma sqrt(T)) - log sqrt(2 pi) - z²/2` and using
`dz/dS_0 = -1/(S_0 sigma sqrt(T))`, `dz/dsigma = -(z/sigma) - sqrt(T)`,
`dz/dr = -sqrt(T)/sigma` and
`dz/dT = -z/(2T) - (mu - sigma²/2)/(sigma sqrt(T))`:

    d log p / dS_0     = z / (S_0 sigma sqrt(T))                           (8)
    d²p / dS_0² / p    = (z² - z sigma sqrt(T) - 1) / (S_0² sigma² T)      (9)
    d log p / dsigma   = (z² - 1)/sigma - z sqrt(T)                       (10)
    d log p / dr       = z sqrt(T) / sigma                                (11)
    d log p / dT       = (z² - 1)/(2T) + z (mu - sigma²/2)/(sigma sqrt(T)) (12)

The estimators are `e^{-rT} f(S_T)` times (8), (9), (10), `(11) - T` and
`r - (12)`, the extra `-T` and `r` coming from the discount factor.

Equation (9) is the **second-order score**
`d²p/dS_0²/p = (d log p/dS_0)² + d² log p/dS_0²`, which is what the second
derivative of the integral produces. It is *not* the square of (8), and
writing the square instead gives a plausible-looking estimator with the wrong
mean.

Everything here asks nothing of `f` beyond square integrability, which is why
this is the estimator for a discontinuous payoff — and it pays for that with
variance, because every score has mean zero and unbounded support.

### One identity worth recording

(9) and (10) were derived independently — one from the second `S_0` score, one
from the `sigma` score — and they turn out to be **proportional**:

    (z² - 1)/sigma - z sqrt(T)  =  S_0² sigma T · (z² - z sigma sqrt(T) - 1)/(S_0² sigma² T).

So the Black–Scholes relation `vega = S_0² sigma T gamma` is reproduced by the
estimator **path by path**, not merely in the mean. Consequence: the LR vega
and LR gamma carry identical noise up to that constant and are not two
independent pieces of evidence about one sample. Their coverage counts are
equal seed for seed, and their measured variance ratios below are equal to
three decimals. Asserted directly in `tests/test_mc_greeks.py`.

## Bump (§7.1)

Reprice at `theta ± h` on common random numbers and divide. It is biased at
`O(h²)` for a central difference. Its variance is
`Var(Y(theta+h) - Y(theta-h)) / (4 h² N)`, and the whole story is what the
numerator does as `h` shrinks:

- payoff Lipschitz in `theta` → numerator `O(h²)` → variance `O(1/N)`, and the
  estimator *is* the pathwise one plus `O(h²)`;
- payoff discontinuous → numerator `O(h)` (the paths that cross the jump, each
  contributing a full step) → standard deviation `O(1/sqrt(N h))`, growing
  without bound as `h -> 0`. The estimator is **inconsistent** in that limit at
  fixed `N`.

Slice 10 moved the implementation from "call `price_european` eight times" to
"draw the normals once, reprice the sample eight ways", so the paired per-path
differences exist and a standard error can be reported at all. The values are
**bit-for-bit** what they were, asserted with `==` over seven configurations
including `n_steps = 4`, all three variance-reduction samplers and `sigma = 0`;
what makes that possible rather than lucky is that the new sampler reproduces
the generator's *consumption pattern* (`n_paths` normals per step in a loop for
the plain path, the `(n_pairs, n_steps)` block for antithetic, the stratified
uniforms for stratified).

Two things about the reported standard error are worth stating because they are
easy to get wrong. First, a CRN difference's error bar has to come from the
**paired** per-unit differences; combining the two legs' independently reported
standard errors would overstate it by orders of magnitude, since the legs are
almost perfectly correlated. Second, the pre-slice theta is a **backward**
difference `(V(T - dt) - V(T))/dt`, not a central one, so it is first order in
`dt` where the other four are second order. Slice 10 reproduces it rather than
silently upgrading it, and records the discrepancy in `meta["fd_by_greek"]`.

## Unbiasedness: coverage, not closeness

"The estimate is close to the analytic value" is not a claim until *close on
what scale* is answered. What is measured instead is **confidence-interval
coverage over independent seeds**: 40 seeds, each producing an estimate and its
own reported standard error, and a count of how often the nominal 95% interval
covers the closed form. Under the null — unbiased *and* calibrated standard
error — the count is Binomial(40, 0.95), mean 38, standard deviation 1.38. A
bias and an understated standard error both show up here and the test cannot
tell them apart, which is honest: a confidence interval is a joint claim about
both.

European call, `S = K = 100`, `r = 5%`, `q = 1%`, `sigma = 20%`, `T = 1`,
20 000 paths:

| Greek | bump | pathwise | likelihood_ratio |
|-------|------|----------|------------------|
| delta | 35 | 35 | 38 |
| gamma | 37 | 38 | 38 |
| vega  | 38 | 38 | 38 |
| theta | 38 | 38 | 38 |
| rho   | 34 | 34 | 38 |

Cash-or-nothing digital at the same point: likelihood ratio 38 / 37 / 37 / 38 /
38. Geometric Asian (six fixings, `sigma = 25%`, `q = 3%`, 10 000 paths):
35–40 of 40 for every Greek and every estimator, and three of the five rows are
*identical* across estimators — `pathwise` supplies only delta and vega and
`likelihood_ratio` only delta, so gamma, rho and theta are the bumped estimates
in all three runs, seed for seed.

The pathwise and bump columns match almost cell for cell, which is the `O(h²)`
statement above seen from the other side. The likelihood-ratio column is 38
everywhere, which is what an exactly unbiased estimator with a calibrated
standard error looks like.

## Variance: what each estimator costs

Cost is counted in **standard normal draws**, the axis Slice 7 uses. Every cell
below spends 20 480 of them over 50 seeds, at the Slice 7 reference points
(`S = K = 100`, `r = 5%`, `q = 0`, `sigma = 20%`, `T = 1`). That is generous to
the bump, which draws once and evaluates the payoff eight times to produce five
Greeks where the other two evaluate it once; on a payoff-evaluation axis the
bump would look eight times worse.

| point | Greek | numerator / denominator | variance factor |
|-------|-------|-------------------------|-----------------|
| atm call | delta | likelihood_ratio / pathwise | **6.46** |
| atm call | gamma | likelihood_ratio / pathwise (mixed) | **13.03** |
| atm call | vega  | likelihood_ratio / pathwise | **13.03** |
| atm call | gamma | bump / pathwise (mixed) | **559.64** |
| digital  | delta | bump / likelihood_ratio | **864.73** |

The first and last rows are the same rule read in opposite directions, and they
are the headline of the slice: **differentiate the payoff when it is smooth,
the density when it is not.** The likelihood ratio pays 6.5× for ignoring that
on a call; the bump pays 865× for ignoring it on a digital.

The gamma row is what makes the mixed estimator worth having: a second
difference divides by `h²` rather than `h`, so its variance carries `1/(N h⁴)`,
and at the default `h = 0.01` that is a factor of 10 000 against a signal of
0.0189.

The vega row equalling the gamma row to three decimals is the pathwise identity
above, not a transcription error.

## The digital, in detail

### Pathwise returns exactly zero

`cash · 1{x > K}` is locally constant at every `x ≠ K`, so the
almost-everywhere derivative a program can evaluate is **identically zero** —
not noisy, not inaccurate, `0.0` at every path count and every seed. The
pathwise interchange is invalid here in the strongest possible way: it
converges, to the wrong number. `qpl.engines.mc.digital.digital_payoff_derivative`
computes that zero and is exported so the test can assert it; the engine
refuses the estimator rather than returning it.

### The bump is available and is the wrong tool

Held to the same 4-sigma budget the vanilla rows use, at 20 000 paths over 40
seeds:

| Greek | bump coverage | LR coverage | bump sd / abs(analytic Greek) |
|-------|---------------|-------------|-------------------------------|
| delta | 36/40 | 38/40 | 0.34 |
| gamma | 39/40 | 37/40 | 4665.55 |
| vega  | 36/40 | 37/40 | 0.67 |
| rho   | 26/40 | 38/40 | 1.35 |
| theta | **0/40** | 38/40 | 0.03 |

Three different failures sit in that table.

- **theta at 0/40.** The estimator is *not* biased: `E[(Y(T-dt) - Y(T))/dt]` is
  the exact difference quotient. But with `dt = 1e-04` the probability that a
  path crosses the strike when the maturity moves is about `2e-06`, so in a
  20 000-path run no path crosses at all. The sample is the discount factor's
  smooth `r V` part alone, and both the estimate and its reported standard
  error describe that part. The missing term is the entire density
  contribution. `rho` at 26/40 is the same mechanism one step less extreme
  (`dr = 1e-05`).
- **gamma at 39/40 passes for the wrong reason.** Its standard deviation is
  4666 times the Greek, so covering is trivial. A coverage test alone cannot
  see this; the ratio column is what sees it.
- **delta at 36/40 is the honest case.** With `h = 0.01` enough paths cross
  that the estimator behaves, and it is merely 34% relative noise.

**A single seed's z-score for these Greeks says nothing.** The estimator's
distribution is a rare-event lottery whose mean is right: the same bumped
digital theta reads `z = -0.60` in `examples/mc_greeks_estimators.py` and
`z = +531` in `examples/digital_option_cross_method.py`, at the same path count
and the same seed, differing only in the dividend yield. Only the spread over
seeds is informative. The engine says all of this in
`meta["estimator_caveat"]` so it is discoverable from a result.

### Bias against variance in `h`

ATM digital, 24 seeds, 20 000 paths per run:

| h | bias (deterministic) | sd (24 seeds) | rmse |
|---|---------------------|---------------|------|
| 10   | -6.5804e-04 | 1.6796e-04 | 6.7914e-04 |
| 3    | -6.0096e-05 | 3.6152e-04 | **3.6649e-04** |
| 1    | -6.6855e-06 | 5.4845e-04 | 5.4849e-04 |
| 0.3  | -6.0178e-07 | 9.6709e-04 | 9.6709e-04 |
| 0.1  | -6.6865e-08 | 1.3789e-03 | 1.3789e-03 |
| 0.03 | -6.0179e-09 | 2.1100e-03 | 2.1100e-03 |

Fitted orders: **-0.4263** for the standard deviation (log-space residual
0.0968, against the `-0.5` predicted — the shortfall is the `O(h²)` correction
to the crossing probability, still visible at `h = 10`) and **+1.9979** for the
bias (residual 0.0042, and exactly `2.0000` with residual `0.0000` over
`h ≤ 1`).

The bias column is computed from the closed form rather than from the runs, and
that is the honest way round: the Monte Carlo estimator is *exactly unbiased
for the finite difference*, so its bias **is** that difference's truncation
error, and measuring it from the sample would mean reading `O(h²)` through a
noise floor a hundred times larger at the useful end of the range.

The RMSE has an interior minimum at `h = 3`, between a bias falling at order 2
and a standard deviation *rising* at order 1/2. That is 300× the package
default of 0.01, and it moves with `N`: the variance branch carries
`1/sqrt(N h)` and the bias branch does not, so the optimum scales like
`N^{-1/5}`. **A default bump size cannot be right for a digital.**

### The 1/T blow-up, and what it actually means

§7.3 notes that the likelihood-ratio delta's variance grows like `1/T`. It
does, and the first reading of that is misleading. 40 seeds, 20 000 paths:

| K | T | analytic delta | sd | sd / delta |
|---|---|----------------|-----|------------|
| 100 | 1.00 | 0.018880 | 2.0176e-04 | 0.0107 |
| 100 | 0.05 | 0.088961 | 9.4218e-04 | 0.0106 |
| 110 | 1.00 | 0.017676 | 2.0207e-04 | 0.0114 |
| 110 | 0.05 | 0.009630 | 4.7264e-04 | **0.0491** |

The absolute standard deviation grows by 4.67 against the 4.47 that
`sqrt(1/0.05)` predicts — the blow-up is real. **But at the money the relative
precision does not move at all** (0.0107 against 0.0106), because an ATM
digital's delta carries the same `1/(S_0 sigma sqrt(T))` factor and grows by
4.71 over the same interval. Reporting "the variance grows 22-fold as maturity
shortens" without that second column would be true and misleading.

Where it bites is **off** the money, where the Greek shrinks while the noise
grows: a 4.3-fold loss of relative precision at `K = 110`. So the estimator
does degrade as maturity shortens, but the statement has to be made about the
ratio of the noise to the Greek, and it is about moneyness as much as about `T`.

## The Asian

Both averages are homogeneous of degree one in the spot — every fixing is
proportional to `S_0` — so

    dA/dS_0 = A / S_0   for the arithmetic mean and the geometric mean alike,

and `delta_pw = e^{-rT} f'(A) A / S_0`. For vega, recover the Brownian path
from the fixings,
`W_{t_i} = (log(S_{t_i}/S_0) - (mu - sigma²/2) t_i)/sigma`, differentiate each
fixing by (3), and let the average inherit it:

    arithmetic:  dA/dsigma = (1/n) Σ_i S_{t_i} (W_{t_i} - sigma t_i),
    geometric:   dG/dsigma = G · (1/n) Σ_i (W_{t_i} - sigma t_i),

the second because `log G` is the mean of the logs. The likelihood-ratio delta
is a one-line score, because `S_0` enters the joint density of the fixings
through the **first transition only**:

    delta_lr = e^{-rT} f(A) Z_1 / (S_0 sigma sqrt(t_1)).

Gamma, rho and theta come from a bump in every case, and `meta["estimator"]`
names the source per Greek because the result genuinely mixes families.

Measured, geometric Asian, 20 seeds at 10 000 paths, six fixings: pathwise
6.538e-03, bump 6.542e-03 (ratio 1.0006 — the same estimator to `O(h²)`),
likelihood ratio 1.287e-02 (ratio 1.968). **The likelihood-ratio penalty grows
with the density of the schedule**: over 12 seeds it is 1.567 at six fixings,
3.439 at twelve and 5.228 at twenty-six, because doubling the fixings halves
`t_1` while the pathwise estimator gets *quieter* (more fixings means a less
variable average). The worse estimator gets worse exactly where an Asian is
most likely to be monitored.

### Theta needs a convention, and it is not `-dV/dT`

The settlement date enters a geometric Asian price only through `e^{-rT}` — `m`
and `v` depend on the fixing times alone — so `-dV/dT = -r V`. That is a true
derivative and it is not time decay; it is what the contract loses by being
paid later while the average is unchanged.

Time decay is the derivative along the **roll**, where calendar time advances by
`s` and the settlement date and every fixing come closer together,
`T -> T - s` and `t_i -> t_i - s`. Along that path

    dtbar/ds = -1,     dv/ds = -(sigma²/n²) Σ_i (2(n-i)+1) = -sigma²,

the second because those weights sum to exactly `n²`. Hence `dF/ds = -mu F`,
`d(sqrt(v))/ds = -sigma²/(2 sqrt(v))` and

    theta = r V - e^{-rT} [mu F N(d1) + F phi(d1) sigma²/(2 sqrt(v))].

The convention is not chosen for tidiness. At `n = 1`, where the geometric
Asian **is** a European vanilla, it reproduces the Black–Scholes theta term for
term — asserted to 1e-12 in `tests/test_asian_analytic.py`, along with the
other four Greeks. A `-dV/dT` theta would read `-0.325` where the roll theta
reads `-8.028` at the 12-fixing point. The Monte Carlo bump uses the same
convention, which is what makes the three routes comparable.

Slice 10 also ships the exact geometric Asian Greeks
(`qpl.engines.analytic.asian.discrete_geometric_greeks`), reversing Slice 8's
blanket refusal. Arithmetic averaging still raises, but for a *mathematical*
reason rather than a scope one: the moment-matched approximations are
differentiable, and a derivative of an approximation whose error has no
rigorous control has no rigorous control either, so `method="analytic"` must
not hand one back.

## Composing with variance reduction

The estimator samples are per-path functions of the same draws the price uses,
so the Slice 7 machinery applies unchanged: an antithetic unit is the pair
average of the two per-path Greek samples, and a stratified estimator is the
strata-weighted mean of them. Nothing in either reduction knows whether the
quantity being averaged is a payoff or a derivative of one.

Delta of the ATM call, 40 seeds, 20 000 normal draws in every cell, variance
factors against the same estimator with no reduction:

| estimator | antithetic | control_variate | stratified |
|-----------|------------|-----------------|------------|
| pathwise | 20.43 | 3.92 | 84.42 |
| likelihood_ratio | 3.35 | 4.09 | 15.24 |
| bump | 20.20 | 3.88 | 86.34 |

Antithetic is worth six times more to the pathwise estimator than to the
likelihood-ratio one, and the reflection argument says why: the pathwise delta
sample `e^{-rT} 1{S_T>K} S_T/S_0` is monotone in `Z`, so `Z -> -Z`
anticorrelates it strongly, while the LR sample carries an explicit odd factor
`Z` that reflection leaves much less correlated.

**The control variate is applied to delta only.** Controlling a Greek needs the
estimator applied to the control *and* the exact value of the matching
derivative of `E[X]`. For delta with the Slice 7 control `X = e^{-rT} S_T` that
derivative is `e^{-rT} e^{mu T}` and the algebra is one line: the control
variable is the same estimator applied to `g(x) = x`. For vega, rho and gamma
the derivative of `E[X] = S_0 e^{-qT}` is identically zero, so the "control" is
a mean-zero variate worth only whatever correlation it happens to have, and for
theta it depends on how the curve is read at a shifted maturity. Rather than
ship four controls of unstated value, the other Greeks are left uncontrolled
and `meta["control_variate_greeks"]` says so — measured: the pathwise vega's
standard deviation is 5.1724e-01 with the control selected and 5.1724e-01
without it, the same number. Under `greeks_estimator="bump"` the control is
applied at the *price* level instead, which is what Slice 7 already did and
what keeps the bump numbers identical to the pre-slice ones.

## Summary: which estimator

| payoff | delta / vega / rho / theta | gamma |
|--------|----------------------------|-------|
| European vanilla | pathwise (bump is equivalent and 8× the payoff evaluations; LR costs 6.5× in variance) | mixed LR–PW (pure LR costs 13×, bump costs 560×) |
| cash-or-nothing digital | likelihood ratio (pathwise returns 0; bump costs 865× on delta and has no usable error bar on theta or rho) | likelihood ratio |
| discretely-monitored Asian | pathwise for delta and vega; bump for rho and theta; LR delta available and 2–5× noisier depending on the schedule | bump |

## What contradicted the expectation

1. **The pre-slice bump theta was never a central difference.** It is a
   backward one, first order in `dt` where the other four are second order.
   Reproduced rather than upgraded, and recorded in `meta["fd_by_greek"]`.
2. **The LR vega and LR gamma are the same estimator up to a constant.** The
   two weights are proportional path by path, so `vega = S_0² sigma T gamma`
   holds in the sample. Their coverage counts and variance ratios are equal,
   and they are one measurement rather than two.
3. **The digital LR delta's `1/T` blow-up does not degrade relative precision
   at the money** — the Greek grows with the noise. It degrades it 4.3-fold off
   the money. The one-line version of §7.3 is misleading without the second
   column.
4. **The digital bump's failure is not "noisy", it is "no error bar".** Its
   theta covers `0` times in 40 at a nominal 95%, not because it is biased —
   it is exactly unbiased for the difference it computes — but because the
   event that carries the signal has probability `2e-06` per path and never
   occurs in the sample. A single seed's z-score for it is meaningless: the
   same estimator reads `-0.60` and `+531` at the same path count.
5. **The "obvious" theta for an Asian is a different quantity.** `-dV/dT` is
   `-r V` and is not time decay; the roll derivative is, and it is the one that
   reduces to Black–Scholes at one fixing.
6. **Moving the bump to sample level cost nothing.** The expectation was
   agreement "to round-off"; the measured agreement is **bit-for-bit** over
   seven configurations, because the new sampler reproduces the generator's
   consumption pattern exactly.
