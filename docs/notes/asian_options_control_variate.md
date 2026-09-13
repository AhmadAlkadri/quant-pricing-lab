# Asian options: one average that has a closed form, and one that does not

An Asian option pays on an average of the underlying over a set of monitoring
dates rather than on its terminal value. That one change splits the contract in
two. The **geometric** average of lognormals is itself lognormal, so its option
price is Black (1976) in closed form. The **arithmetic** average is a sum of
lognormals, which is not lognormal and has no elementary density, so there is no
closed form at all — and the standard answer, from Kemna & Vorst (1990), is to
simulate the arithmetic payoff while using the geometric one, whose mean is
known exactly, as a control variate.

This note derives both, records what the control variate is measured to be
worth, and states three things this slice got wrong on the way in.

Sources. Kemna, A.G.Z. and Vorst, A.C.F. (1990), "A pricing method for options
based on average asset values", *Journal of Banking and Finance* 14, 113–129:
the continuous geometric closed form and the control-variate proposal. Turnbull,
S.M. and Wakeman, L.M. (1991), *Journal of Financial and Quantitative Analysis*
26(3), 377–389, and Levy, E. (1992), *Journal of International Money and
Finance* 11, 474–491: the two-moment lognormal approximation. Glasserman (2003),
*Monte Carlo Methods in Financial Engineering*, §3.2 (generating paths at given
dates) and §4.1 (control variates, with the Asian as the worked example). Fusai
& Roncoroni, *Implementing Models in Quantitative Finance*, chapter 15, was used
as a map of the problem only; its Laplace-transform and PDE routes are a later
slice. Every derivation below is written out independently and every number was
measured in this repository; the three *published values* used as fixtures are
cited where they appear.

## Convention

Fixing times `0 < t_1 < … < t_n ≤ T`, and everywhere in this package the uniform
schedule is `t_i = i T / n` — a right-endpoint rule with the last fixing at
expiry. `qpl.instruments.uniform_fixing_times` builds it with `linspace` rather
than a comprehension, because `(i + 1) * T / n` at `i = n - 1` is not `T` in
general: at `T = 90/365` and `n = 2560` it lands one ulp above, and the
instrument then correctly refuses a fixing after expiry.

All three published benchmarks are reproduced under this convention, which is
how it was chosen rather than assumed.

## The discrete geometric average is exactly lognormal

Under Black–Scholes with `mu = r - q`,

    S_{t_i} = S_0 exp( (mu - sigma²/2) t_i + sigma W_{t_i} ),

so the geometric average `G = (prod_i S_{t_i})^{1/n}` has

    log G = log S_0 + (mu - sigma²/2) tbar + (sigma/n) sum_i W_{t_i},
    tbar  = (1/n) sum_i t_i.

The last term is a linear combination of jointly Gaussian variables, so `log G`
is Gaussian. That single sentence is the whole reason one average has a closed
form and the other does not. Its mean is `m = log S_0 + (mu - sigma²/2) tbar`,
and from `Cov(W_s, W_t) = min(s, t)`,

    v = (sigma²/n²) sum_i sum_j min(t_i, t_j).

The double sum collapses. Fix `i` and split the inner sum at `j = i`:
`sum_j min(t_i, t_j) = sum_{j<i} t_j + (n - i + 1) t_i`. Summing over `i` and
collecting the coefficient of each `t_i` leaves

    v = (sigma²/n²) sum_{i=1}^{n} (2(n - i) + 1) t_i,

so the earliest fixing is weighted `2n - 1` and the last one `1` — earlier
fixings carry more of the variance, because every later Brownian increment is
still to come. That is `O(n)` work instead of `O(n²)`, and it is checked against
the double sum directly, on irregular schedules as well as uniform ones, where
an index slip would not cancel.

With `F = E[G] = exp(m + v/2)` the price is Black (1976) with total variance
`v`:

    C = e^{-rT}( F N(d1) - K N(d2) ),  d1 = (log(F/K) + v/2)/√v,  d2 = d1 - √v.

`T` enters only through the discount factor, while `m` and `v` depend only on
the fixing times. An average that stops before settlement is therefore priced
correctly by construction and not by a special case — verified by pricing one
schedule at two settlement dates and recovering `e^{-r Δ}` exactly.

**The strongest check available**: with `n = 1` and `t_1 = T`, `G = S_T` and the
whole apparatus must collapse onto Black–Scholes. It does, to 1e-13, against an
engine written years earlier that shares no code with it.

## Why the fixing count matters at order 1

For `t_i = iT/n`,

    tbar = (T/2)(1 + 1/n),     v = (sigma² T/3)(1 + 3/(2n) + 1/(2n²)),

whose `n → ∞` limits `T/2` and `sigma² T/3` are the Kemna–Vorst
continuous-averaging moments. Both corrections are `O(1/n)` and they do not
cancel, so the discrete price approaches the continuous one at **order 1**.

Measured against the closed-form continuous limit over `n` in (20, 40, …, 2560):

| point                                       | fitted order | log residual | err(n=20) | err(n=2560) |
|---------------------------------------------|--------------|--------------|-----------|-------------|
| S=K=100, r=6%, q=3%, σ=20%, T=1, call        | **1.0002**   | 0.00017      | 2.030e-01 | 1.584e-03   |
| S=80, K=85, r=5%, q=-3%, σ=20%, T=0.25, put  | **1.0096**   | 0.00917      | 1.622e-02 | 1.200e-04   |

Errors are one-signed and halve at every level; the 128-fold refinement divides
them by 128.1 and 135.2, against 16384 for order 2. The same order, 1.0002 with
residual 0.00024, is measured against **QuantLib's** continuous-averaging engine
in `tests/oracle/test_asian_vs_quantlib.py`, so the rate is a property of the
convention and not of either implementation.

This is a modelling fact, not a defect. A monthly-fixing Asian genuinely is not
a continuously averaged one, and the difference is `O(1/12)` of the continuous
correction. A midpoint fixing convention would be order 2; the convention that
puts the last fixing at expiry is a right-endpoint rule and is order 1.

## The published values, and two corrections to them

| value          | contract                                                     | measured residual |
|----------------|--------------------------------------------------------------|-------------------|
| 5.3425606635   | discrete geometric call, S=K=100, q=3%, r=6%, σ=20%, 10 fixings | **5.8e-11**    |
| 4.6922         | continuous geometric **put**, S=80, K=85, q=-3%, r=5%, σ=20%, 90 days | **3.7e-05** (20 000 fixings) |
| 19.5152        | Turnbull–Wakeman, S=100, K=80, r=q=5%, σ=20%, 26 fixings over 0.5y | **9.8e-06** |

Two of the three needed correcting before they could be reproduced.

**The 4.6922 is a put, and it is on a 90/360 year fraction.** The
geometric-average *call* at that point is 0.4714. And at `T = 90/365` the exact
continuous value is 4.6924339, which misses the published four decimals by
2.34e-04 — more than twice the tolerance a four-decimal quote deserves. At
`T = 90/360 = 0.25` it is 4.6922213, which rounds to the published figure. The
day count is load-bearing at this precision and is pinned by its own test.

**The Turnbull–Wakeman point needs `q = r = 5%`.** The dividend yield is not
usually quoted with it; at `q = 0` the same construction gives 20.7865. With
`q = r` every forward equals the spot and `E[A] = S` exactly, which is probably
why the point was chosen.

## The arithmetic average: no closed form, two ways to cope

The exact moments are available even though the density is not:

    M1 = E[A]  = (S_0/n) sum_i e^{mu t_i},
    M2 = E[A²] = (S_0²/n²) sum_i sum_j exp( mu(t_i + t_j) + sigma² min(t_i, t_j) ),

the second from `E[S_u S_w] = S_0² exp(mu(u+w) + sigma² min(u,w))`. `M1` alone
gives Asian put–call parity,

    C - P = e^{-rT}( E[A] - K ),

exactly and independently of volatility, because
`max(A - K, 0) - max(K - A, 0) = A - K` identically. In a Monte Carlo sample
that uses the same paths for both legs this holds **per sample** — measured to
1e-11, with no Monte Carlo error in the identity at all, only in the comparison
of the sample mean to `E[A]`.

### Turnbull–Wakeman: an approximation, labelled as one

Fit a lognormal to `M1` and `M2` and price that instead: total variance
`v_A = log(M2/M1²)`, forward `M1`, Black (1976) again. It is not a bound, not an
expansion, and has no error control. It is therefore **not registered** as
`method="analytic"` — `qpl.pricing.price` on an arithmetic Asian raises
`NotSupportedError` naming it and Monte Carlo — and it is reachable only as
`qpl.engines.analytic.asian.turnbull_wakeman_price`.

Measured against control-variate Monte Carlo (500 000 to 2 000 000 paths):

| point                        | σ²T   | CV-MC      | TW         | gap        | relative | resolved at |
|------------------------------|-------|------------|------------|------------|----------|-------------|
| ATM, 10 fixings, σ=20%       | 0.040 | 6.234154   | 6.252316   | **+0.018162** | +0.29% | 107 stderr |
| ATM, 52 fixings, σ=20%       | 0.040 | 5.853914   | 5.873245   | **+0.019331** | +0.33% | 122 stderr |
| K=80, 26 fixings, q=r        | 0.020 | 19.513387  | 19.515210  | **+0.001823** | +0.009% | 18 stderr |
| ATM, 52 fixings, σ=40%       | 0.160 | 10.289899  | 10.382623  | **+0.092724** | +0.89% | 143 stderr |

The sign is positive at every point: the fitted lognormal is more right-skewed
than the true law of the arithmetic average, so it puts too much mass where the
call pays. The size tracks `σ²T` — quadrupling the total variance roughly
triples the relative error — and collapses deep in the money, where the option
is nearly a forward on the average and the price is determined by the first
moment, which the fit matches *exactly*.

Read the third row next to the published 19.5152. The approximation reproduces
its own published value to 9.8e-06 and is 1.8e-03 from the truth: **a benchmark
for an approximation is not a benchmark for what it approximates**, and this is
the cheapest demonstration of that in the repository.

### Monte Carlo with the Kemna–Vorst control

Sample the path at the fixing times and nowhere else, by exact lognormal
stepping over a variable time grid. There is then **no time-discretisation
bias**: every fixing is an exact draw from its marginal with the correct joint
law, so the only error in the price is statistical and there is no refinement
study to run. (`simulate_gbm_exact` gained a `times=` grid for this; the uniform
`(t, n_steps)` branch is untouched and pinned bit-for-bit, because
`t_{i+1} - t_i` is not bit-for-bit `t/n_steps`.)

The estimator is `mean(Y) - b (mean(X) - E[X])` with

    Y = e^{-rT} max(A - K, 0),   X = e^{-rT} max(G - K, 0),
    E[X] = the discrete geometric closed form,

and `b` the OLS slope, all of it handled by the Slice 7 estimator layer
(`ddof=2` regression residual standard error, antithetic pair bookkeeping).
`TerminalSample` gained a `control_name` field so that a reported correlation
says what it is a correlation *with*.

Measured over 50 seeds at 41 600 normal draws (`4160 × 10 == 800 × 52`, so both
columns cost the same), with every prediction from an independent 40 000-path
pilot:

| fixings | estimator          | ρ            | predicted | measured | ratio |
|---------|--------------------|--------------|-----------|----------|-------|
| 10      | antithetic         | −0.5202      | 4.17      | 5.54     | 1.33  |
| 10      | control variate    | +0.999608    | 1276.7    | **1433.0** | 1.12 |
| 10      | antithetic+control | +0.999127 (pair) | 2387.7 | 2113.8  | 0.89  |
| 52      | antithetic         | −0.5172      | 4.14      | 6.31     | 1.52  |
| 52      | control variate    | +0.999604    | 1262.2    | **1196.2** | 0.95 |
| 52      | antithetic+control | +0.999122 (pair) | 2360.8 | 2584.0  | 1.09  |

Three orders of magnitude, against the 7.6 the discounted terminal spot buys on
a vanilla ATM call in Slice 7. The whole difference is `ρ`: 0.9996 against
0.9246, and `1/(1-ρ²)` is brutally sensitive up there. In practical terms the
control is worth `√1200 ≈ 35×` in accuracy at fixed cost, or 1200× in cost at
fixed accuracy — the 52-fixing column runs **800 paths** and still resolves the
price to four decimals.

`ρ` barely moves between 10 and 52 fixings (a difference of 4e-06, inside the
pilot's own noise). The control's quality is a property of the contract, not of
the monitoring frequency, so it does not need retuning when the schedule
changes.

Where it degrades (52 fixings, 40 000-path pilot):

| σ    | K   | ρ         | 1/(1-ρ²) |
|------|-----|-----------|----------|
| 10%  | 100 | 0.999891  | 4575.3   |
| 20%  | 100 | 0.999604  | 1262.2   |
| 40%  | 100 | 0.998382  | 309.3    |
| 80%  | 100 | 0.992153  | 64.0     |
| 20%  | 130 | 0.992562  | 67.5     |
| 40%  | 130 | 0.995199  | 104.4    |
| 80%  | 130 | 0.989342  | 47.2     |

At the money the ordering is exactly the prediction: quadrupling `σ²T` divides
the factor by about four, over a range of 70×. Out of the money it is **not
monotone** — 67.5 at σ=20%, 104.4 at σ=40%, 47.2 at σ=80% — because two things
move at once. Higher volatility widens `log A - log G`, which hurts; but at
`K = 130` it also moves the option out of the "never pays" region, and a payoff
that is zero on most of the sample is one the control cannot explain, because
the control is zero there too.

## Five things this slice got wrong

**1. The Haug 4.6922 is a put on a 90/360 year fraction, not a call on 90/365.**
Covered above; both halves were stated the other way in the slice brief and both
are now pinned by their own tests.

**2. The discrete-to-continuous rate is order 1, not something to be discovered
empirically.** It was derived first from the two `O(1/n)` moment corrections and
then measured at 1.0002 and 1.0096. Worth recording because a reader expecting
order 2 (as one does from most refinement studies in this repository) would
mis-size a fixing grid by a factor of `1/ε` rather than `1/√ε`.

**3. The combinations *do* compose — with the right second factor.** Slice 7
recorded that its combinations "do not compose multiplicatively", and by the
same arithmetic that is true here: the product of the two **marginal** factors
is 7933 and 7543 against measured 2114 and 2584, off by 3–4×. But the marginal
product is the wrong prediction. The combined estimator regresses the control on
the *pair-averaged* units, so the composite prediction is

    factor(antithetic) × 1/(1 - ρ_pair²)

with `ρ_pair` measured on those units — and that is right to 11% at both fixing
counts. Slice 7's non-composition was a statement about which correlation was
measured, not about an interaction between the methods. Note the direction:
`ρ_pair < ρ` (0.99913 against 0.99961), because pair-averaging has already
removed the part of the payoff the geometric control explains best.

**4. The control variate's value collapses exactly where plain Monte Carlo is
worst, and the confidence interval fails with it.** At σ=10%, K=130, 52 fixings
and 40 000 paths, **no** path's geometric average finishes in the money, so the
control has zero sample variance. `control_variate_coefficient` returns `0.0`
rather than dividing by a vanishing sum of squares, and the estimator silently
becomes the plain one: `β = 0`, `ρ = 0`, predicted factor `1.0`, no NaN. Exactly
one *arithmetic* path pays, so the answer is 5.275e-06 ± 5.275e-06 — while the
geometric closed form at the same point, a valid lower bound by AM–GM, is
3.533e-05. The reported interval does not reach the truth: `value + 4 stderr =
2.64e-05 < 3.53e-05`. A confidence interval built on one success does not cover,
and reading it carefully does not repair that. `meta` says so in machine-readable
form (`control_variance_factor_predicted == 1.0`), and the fix — reporting
`max(estimate, geometric_price)`, or refusing — is a design option this slice
records rather than takes.

**5. Antithetic sampling is not worth reaching for on an Asian.** 5.5 and 6.3
against the control variate's 1433 and 1196 at the same cost. The slice
statement asked for the control variate's factor to be "large"; the more useful
finding is how small everything else is next to it.

## What is refused, and why

- `method="analytic"` on an **arithmetic** Asian: `NotSupportedError` naming
  `turnbull_wakeman_price` and `method="mc"`. The registry has an entry for the
  key, so the message can be specific; registering nothing would report the
  generic "Unsupported instrument/model/market combination".
- `variance_reduction="stratified"`: `NotSupportedError` naming Brownian-bridge
  stratification. Stratifying one terminal normal stratifies a terminal price;
  an average over `n` fixings is driven by `n` normals with no single scalar to
  partition. Stated in terms of the fixing schedule rather than reusing Slice
  7's `n_steps > 1` rule, which a one-fixing Asian would slip through.
- `cfg.n_steps != 1`: `InvalidInputError`. The Asian's time grid *is* its fixing
  schedule; a config field read on one instrument and discarded on another is
  discovered by a wrong number, not by a message.
- Greeks, on both routes. The message says this is a **scope boundary and not an
  impossibility** — unlike a digital, the Asian pathwise estimator is well
  defined, because the payoff is Lipschitz in the average and the average is
  smooth in the path. Phase 3's Greeks item.
- `method="tree"` and `method="pde"`: nothing registers them, so the generic
  message is the correct one. Pricing an average needs a second state variable,
  which is a different discretisation rather than a branch inside the existing
  ones.

## Cross-checks against QuantLib

`tests/oracle/test_asian_vs_quantlib.py`, four legs of deliberately different
character:

| leg | what it checks | worst residual |
|-----|----------------|----------------|
| `AnalyticDiscreteGeometricAveragePriceAsianEngine` | the derivation | **2.487e-14** (≈4 ulp) |
| `TurnbullWakemanAsianEngine` | the approximation's moments | **3.055e-13** |
| `AnalyticContinuousGeometricAveragePriceAsianEngine` | convergence onto an independent limit | order **1.0002**, residual 0.00024 |
| `MCDiscreteArithmeticAPEngine(controlVariate=True)` | two implementations of one idea | \|z\| ≤ 0.72 |

QuantLib's Monte Carlo engine uses the *same* Kemna–Vorst control, arrived at
independently from the same paper, which makes that leg a check of two
implementations of one idea rather than of two different ideas; everything else
differs (path generator, stream, seed, standard-error formula).

The Turnbull–Wakeman leg is an order of magnitude looser than the geometric one
and the reason is visible in the data: the 5-fixing points agree to 1.2e-14 and
the 73-fixing ones to 3.1e-13, because `E[A²]` is an `n²` double sum (5329 terms
at 73 fixings) that the two implementations accumulate in different orders.

**Day count.** `Actual365Fixed` with 365 days gives `T = 1.0` exactly, and
fixings at 73-day (`n = 5`) or 5-day (`n = 73`) intervals — both divisors of 365
— give year fractions exactly `i/n`. The two *computations* of those year
fractions differ by at most **1.11e-16**, one ulp, three orders below the
analytic tolerance and not the dominant residual. Fixing counts that do not
divide 365 (the published 26-fixing Turnbull–Wakeman point, whose fixings would
fall 7.019 days apart) cannot be expressed exactly as calendar dates, which is
why that point is checked against its published value rather than against
QuantLib.

## Where to look

- Instrument and payoff reduction: `src/qpl/instruments/options.py`,
  `src/qpl/instruments/payoffs.py`.
- Closed forms and the approximation: `src/qpl/engines/analytic/asian.py`.
- Sampler and estimator: `src/qpl/engines/mc/processes.py`,
  `src/qpl/engines/mc/asian.py`.
- Claims: `src/qpl/cases/asian_black_scholes.py` (30 rows).
- Measurements: `tests/test_asian_analytic.py`, `tests/test_asian_mc.py`,
  `tests/test_asian_variance_ratios.py`,
  `tests/cases/test_asian_black_scholes_cases.py`,
  `tests/oracle/test_asian_vs_quantlib.py`.
- Runnable table: `examples/asian_option_control_variate.py` (also
  `--case fixings`).
