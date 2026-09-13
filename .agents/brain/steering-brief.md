# Steering Brief

What changed in Slice 10 (files + bullets)

Phase 3's pathwise/likelihood-ratio Greeks item. Three Greek estimators behind
one config field, each with a per-Greek standard error, checked by coverage
rather than closeness and chosen by measured variance. Still on
`dev/curriculum`; not pushed.

- `src/qpl/engines/mc/greeks.py` (new): the estimator layer. `PathSample`
  (terminal spots plus the normal **recovered** from them by inverting the
  terminal law, so the formulas hold at any `n_steps`), `draw_path_sample`
  (reproduces the price engine's generator consumption exactly),
  `pathwise_terminal_greeks`, `likelihood_ratio_terminal_greeks`,
  `estimate_greek` (routes through the Slice 7 `estimate_from_sample`, so the
  antithetic pair `ddof=1`, the control-variate `ddof=2` and the strata-weighted
  standard errors are not a second copy), `scenario` / `bump_greeks_terminal` /
  `bump_estimates` / `one_sided_estimate` / `bump_sizes`, `control_samples`,
  `greeks_result`.
- `MCConfig.greeks_estimator: Literal["bump","pathwise","likelihood_ratio"] =
  "bump"`. The default is **bit-for-bit** the pre-slice CRN bump (asserted with
  `==` over seven configurations including `n_steps=4`, all three
  variance-reduction samplers and `sigma=0`), even though the implementation
  moved from eight `price_european` calls to one sample repriced eight ways.
- `GreeksResult.meta` gains `stderr` and `estimator` as **per-Greek** dicts,
  plus `greeks_estimator`, `fd_by_greek`, `control_variate_greeks` and (digital
  bump only) `estimator_caveat`.
- `src/qpl/engines/mc/digital.py`: LR Greeks; `digital_payoff_derivative`
  exported so a test can compute the identically-zero pathwise sample; pathwise
  refused; bump available and measured to be the wrong tool. `T = 0` and
  `sigma = 0` now raise for Greeks.
- `src/qpl/engines/mc/asian.py`: `asian_fixings_sample` split out; pathwise
  delta/vega, LR delta, bump for the rest, theta by the **roll**.
- `src/qpl/engines/analytic/asian.py`: `discrete_geometric_greeks` and a real
  `greeks_asian`, reversing Slice 8's blanket refusal.
- Cases: `MC_GREEK_CASES` + `MC_GREEK_CASE_KEYS` (30 rows),
  `DIGITAL_MC_GREEKS_CASES` (15), `MC_GREEKS_VARIANCE_CASES` +
  `MCGreeksVarianceCase` + `MC_GREEKS_VARIANCE_BAND` (5).
- `tests/test_mc_greeks.py` (rewritten), `tests/test_mc_greeks_variance.py`
  (new), `examples/mc_greeks_estimators.py` (two curated smoke triples).

**Coverage of the nominal 95% interval, 40 seeds at 20 000 paths**, against a
Binomial(40, 0.95) mean of 38:

    Greek    call bump   call pw   call LR   digital LR
    delta       35          35        38         38
    gamma       37          38        38         37
    vega        38          38        38         37
    theta       38          38        38         38
    rho         34          34        38         38

**Variance at equal normal draws** (50 seeds, 20 480 draws, Slice 7 points):

    point         greek   numerator / denominator        factor
    atm call      delta   likelihood_ratio / pathwise      6.46
    atm call      gamma   likelihood_ratio / pathwise     13.03
    atm call      vega    likelihood_ratio / pathwise     13.03
    atm call      gamma   bump / pathwise                559.64
    digital       delta   bump / likelihood_ratio        864.73

The first and last rows are one rule read both ways: differentiate the payoff
when it is smooth, the density when it is not.

**Digital bump bias/variance in h** (24 seeds, 20 000 paths): sd order
**-0.4263** (r 0.0968), bias order **+1.9979** (r 0.0042; exactly 2.0000 for
`h <= 1`), RMSE minimum at **h = 3** -- 300x the package default, scaling like
`N^{-1/5}`. The bias is computed from the closed form, because the estimator is
exactly unbiased for the finite difference and its bias IS that difference's
truncation error.

**Composition** (ATM call delta, equal draws), antithetic / control_variate /
stratified: pathwise 20.43 / 3.92 / 84.42, LR 3.35 / 4.09 / 15.24, bump
20.20 / 3.88 / 86.34. The control variate is delta-only at sample level.

**Six contradicted expectations, all encoded.** (1) The pre-slice bump theta is
a *backward* difference, not central -- reproduced, not upgraded. (2) The LR
vega and LR gamma weights are proportional path by path, so
`vega = S_0^2 sigma T gamma` holds in the *sample*: one measurement, not two.
(3) The digital LR delta's `1/T` blow-up does not degrade relative precision at
the money (0.0107 -> 0.0106) because the Greek grows with the noise; it
degrades it 4.3x off the money. (4) The digital bump's failure is "no error
bar", not "noisy": theta covers **0/40** because the event carrying the signal
has probability 2e-06 per path, and the same estimator reads z = -0.60 and
z = +531 at the same path count. (5) `-dV/dT` is not an Asian's theta -- it is
`-r V` (-0.325 against -8.028); the roll derivative is, the variance weights sum
to exactly `n^2`, and at one fixing it reproduces Black-Scholes theta to 1e-12.
(6) The bump's move to sample level agrees **bit-for-bit**, not merely to
round-off.

Suite: **1322 tests, 125.6 s** (from 1224 / 112.2 s). Derivation and full
tables: `docs/notes/mc_greeks_pathwise_likelihood_ratio.md`.

---

What changed in Slice 9 (files + bullets)

Phase 3's Euler-vs-Milstein item, and the prerequisite for Heston Monte Carlo:
a scalar-SDE layer with measured strong and weak orders, plus the CIR variance
process with both an exact sampler and a discretisation whose boundary
behaviour is measured rather than assumed. Still on `dev/curriculum`; not
pushed.

- `src/qpl/engines/mc/sde.py` (new): `SDEModel` (drift, diffusion, diffusion
  derivative, optional Milstein *product*, optional exact transition, state
  floor) and `simulate(model, x0, t_grid, n_paths, scheme=, truncation=,
  seed=|rng=|normals=, db_dx=)` over `{'euler','milstein','exact'}`.
  `coarsen_normals` builds coarse standard normals by summing fine blocks and
  dividing by `sqrt(r)` -- without it the strong error is not measurable at
  all. `gbm_sde(mu, sigma)` and `cir_sde(kappa, theta, xi)`;
  `cir_transition_parameters`, `cir_moments`, `cir_expected_excess`.
  `simulate_gbm_exact` untouched, pinned to 1e-13 relative.
- `src/qpl/validation/stochastic.py` (new): `strong_error` / `weak_error`, each
  returning the estimate **with its own standard error**. `weak_error` takes
  either a coupled reference sample (common random numbers) or a known exact
  mean; which one you can use decides whether the bias is measurable.
- `src/qpl/cases/sde_discretization.py` (new): the sixth id space, 14 rows (10
  `CONVERGENCE_ORDER`, 4 `NEGATIVE_FINDING`), plus the study specs, levels,
  path counts and seeds that both test modules and the example now import.
- `tests/test_mc_sde.py`, `tests/test_sde_convergence.py`,
  `tests/cases/test_sde_discretization_cases.py`, `examples/sde_convergence.py`
  (two curated smoke triples).

**Measured orders on GBM** (S0 = 100, mu = 0.2, sigma = 0.2, T = 1, K = 100,
100 000 paths, h = 1/8 ... 1/128, coupled increments):

    claim                 Euler                        Milstein
    strong                0.5154 (r0.0062, C 2.9624)   0.9932 (r0.0025, C 4.2643)
    weak  f(x)=x          1.0041 (r0.0087, C 2.4142)   0.9944 (r0.0017, C 2.3916)
    weak  f(x)=(x-K)^+    1.0088 (r0.0109, C 2.5642)   0.9946 (r0.0018, C 3.1096)

The Milstein strong *constant* is larger: 1.90x better at h = 1/8, 7.12x at
h = 1/128. The whole gain is the exponent.

**The weak columns are an identity, not a coincidence.** Both schemes satisfy
`E[X^h_T] = S0 (1 + mu h)^{T/h}` exactly, because the Euler diffusion term and
the Milstein correction both have conditional mean zero. Closed-form constant
`S0 e^{mu T} mu^2 T / 2 = 2.4428`.

**European call bias** (read as r = 0.2, q = 0, analytic 19.6298871012): order
1.0088, constant 2.0994 = e^{-rT} x 2.5642, bias -0.2560 at n = 8 (-1.30%) to
-0.0155 at n = 128.

**CIR** (v0 = theta = 0.04, T = 1): exact transition
`v_{t+h} = c chi'^2(d, lambda)`, `c = xi^2(1-e^{-kappa h})/(4 kappa)`,
`d = 4 kappa theta/xi^2`, `lambda = v_t e^{-kappa h}/c`; `d` IS the Feller
number. Full-truncation Euler, 400 000 paths, h = 1/4 ... 1/32:

    regime            E[v_T]                          E[(v_T-0.05)^+]
    Feller satisfied  no measurable bias past h=1/4    0.9729 (r0.0705, C 3.8638e-03)
    Feller violated   0.6834 (r0.1261)                 0.9389 (r0.0606, C 0.1329)

Violating Feller costs a factor 34 in the constant and almost nothing in the
exponent.

**Three contradicted expectations, all encoded.**

1. **Milstein on GBM is NOT exp-of-Euler-on-log.** Euler applied to `log S` is
   the *exact* sampler (state-independent coefficients), and Milstein on the
   log is bit-for-bit Euler on the log. What holds is
   `Milstein factor = 1 + u + sigma^2 dW^2/2` against the exact `exp(u)` with
   `u = (mu - sigma^2/2)h + sigma dW` -- the exponential cut after the
   quadratic term in dW only. Pinned to 3.3e-16 relative.
2. **Full truncation does NOT keep CIR positive.** It produces exactly the same
   negative states as plain Euler (0.91174 of paths violated, 2.6e-04
   satisfied) and must, since the schemes are pathwise identical until the
   first negative state and plain Euler has no next state after it (0.90968
   become NaN). It keeps the recursion DEFINED. At these step sizes 68-73% of
   *terminal* variances are negative in the violated regime.
3. **The coupled weak-error estimator's noise floor is the scheme's own STRONG
   error.** Signal/noise goes as
   `(C_weak/c_strong) sqrt(N) h^{p_weak - p_strong}`: flat for Milstein
   (|z| 211 -> 207), falling for Euler (73 -> 20). Refining makes the Euler
   measurement worse; only N helps. At mu = 0.05 the Euler weak error is
   unmeasurable at 100 000 paths (|z| 5.8 -> 0.2, fitted 1.5742, residual
   0.4231), so the 20% drift is load-bearing, and the 5% failure is pinned.

**Trap for the next slice**: the Milstein correction `(1/2) b b'` for CIR is the
constant `xi^2/4` while its factors are 0 and inf at v = 0 -- a state full
truncation makes common. `cir_sde` supplies the product directly
(`SDEModel.milstein_coefficient`); building it from the factors gives NaN paths.

Suite: 1224 tests, 107.7 s. Full tables: `docs/notes/sde_discretization.md`.

---

What changed in Slice 8 (files + bullets)

Phase 3's Asian/control-variate item: the first path-dependent instrument, and
the control variate the Slice 7 entry said was waiting for it. Still on
`dev/curriculum`; not pushed.

- `src/qpl/instruments/options.py`: `AsianOption(kind, strike, expiry,
  fixing_times, averaging)`, outside the vanilla and digital hierarchies
  because its payoff reads the **path**. Validation: fixings strictly
  increasing, all in `(0, expiry]`, at least one, no repeats (a repeat is a
  real contract but makes the fixing grid and the simulation grid two different
  objects). `uniform_fixing_times(T, n)` builds `t_i = i T / n` with `linspace`
  -- `(i+1) * T / n` lands one ulp above `T` at `T = 90/365, n = 2560` and the
  instrument then correctly refuses it.
- `src/qpl/engines/analytic/asian.py` (new): the discrete geometric closed form
  derived from the lognormal transition density (`log G ~ N(m, v)` with the
  collapsed covariance sum `v = (sigma^2/n^2) sum_i (2(n-i)+1) t_i`), `E[A]` as
  a sum of forwards, the two arithmetic moments, and `turnbull_wakeman_price`
  -- **not registered**, so `method='analytic'` on an arithmetic Asian raises
  `NotSupportedError` naming it and Monte Carlo.
- `src/qpl/engines/mc/asian.py` (new): sample the path at the fixing times and
  nowhere else. `simulate_gbm_exact` gained an optional `times=` grid; the
  uniform branch is untouched and pinned bit-for-bit. Exact stepping means
  **no time-discretisation bias** -- the only error is statistical, and there
  is no refinement study to run. Control = the geometric-average discounted
  payoff with the closed form as its exact mean (Kemna-Vorst), falling back to
  the discounted last-fixing spot for a geometric Asian, where the geometric
  control would BE the payoff. Estimation routes through the Slice 7 layer;
  `TerminalSample` gained `control_name`.

**Fixing convention**: `t_i = i T / n`, last fixing at expiry. All three
published benchmarks are reproduced under it, which is how it was chosen.

**Measured variance factors** (50 seeds, 41 600 **normal draws**, so
`4160 x 10 == 800 x 52` is equal cost; predictions from a 40 000-path pilot):

    fixings   antithetic      control variate      antithetic+control
    10        5.54 (4.17)     1433.0 (1276.7)      2113.8 (2387.7)
    52        6.31 (4.14)     1196.2 (1262.2)      2584.0 (2360.8)

`rho = 0.999608` and `0.999604` -- a 4e-06 difference, inside the pilot's noise.
Three orders of magnitude, against the 7.6 the discounted terminal spot buys on
a vanilla ATM call in Slice 7; the whole difference is rho (0.9996 vs 0.9246)
and how steep `1/(1-rho^2)` is up there.

**Published anchors**: Clewlow-Strickland 5.3425606635 to **5.8e-11**; Haug
4.6922 to **3.7e-05** at 20 000 fixings; Turnbull-Wakeman 19.5152 to
**9.8e-06**.

**Discrete -> continuous is order 1**, derived first (`tbar = (T/2)(1+1/n)`,
`v = (sigma^2 T/3)(1 + 3/(2n) + 1/(2n^2))`, no cancellation) and then measured:
1.0002 (residual 0.00017) and 1.0096 (0.00917) in-repo, and 1.0002 (0.00024)
against QuantLib's continuous engine.

**Five contradicted expectations, all encoded.**

1. **The Haug 4.6922 is a PUT on a 90/360 year fraction**, not a call on 90/365.
   The call is 0.4714; at `T = 90/365` the exact value is 4.6924339, 2.34e-04
   from the published four decimals. Pinned in its own NEGATIVE_FINDING test.
2. **The Turnbull-Wakeman benchmark needs `q = r = 5%`** (recovered by matching;
   at `q = 0` the same construction gives 20.7865).
3. **The combinations DO compose -- with the right second factor.** Slice 7's
   "not multiplicative" holds for the product of the two *marginal* factors
   (7933 and 7543 against 2114 and 2584), but the combined estimator regresses
   the control on the **pair-averaged** units, and
   `factor(antithetic) x 1/(1-rho_pair^2)` is right to 11% at both fixing
   counts. Slice 7's finding was about which correlation was measured, not about
   an interaction between the methods. Note `rho_pair < rho` (0.99913 vs
   0.99961): pair-averaging has already removed what the control explains best.
4. **The control collapses exactly where plain Monte Carlo is worst, and the
   confidence interval fails with it.** `sigma = 10%, K = 130`, 52 fixings,
   40 000 paths: no path's geometric average is in the money, so the control has
   zero sample variance and `beta = 0`, `rho = 0`, factor 1.0, no NaN -- the
   estimator silently becomes the plain one. One arithmetic path pays, giving
   5.275e-06 +- 5.275e-06 against a geometric closed form (a valid AM-GM lower
   bound) of 3.533e-05: `value + 4 stderr = 2.64e-05` does **not** reach the
   truth. Recorded, not fixed; `control_variance_factor_predicted == 1.0` in
   `meta` is the machine-readable warning.
5. **Antithetic is not worth reaching for on an Asian**: 5.5 and 6.3 against
   1433 and 1196 at the same cost.

**The Turnbull-Wakeman gap, with its sign** (always positive: the fitted
lognormal is more right-skewed than the true law): +0.018162 (ATM 10f, +0.29%),
+0.019331 (ATM 52f, +0.33%), +0.001823 (K=80 26f q=r, +0.009%), +0.092724 (ATM
52f vol 40%, +0.89%). Resolved at 18 to 143 standard errors. Read the third next
to the published 19.5152: the approximation reproduces its own benchmark to
9.8e-06 and is 1.8e-03 from the truth. **A benchmark for an approximation is not
a benchmark for what it approximates.**

**Identities that are exact rather than statistical.** Asian put-call parity
holds **per sample** to 1e-11 (same paths both legs, so
`max(A-K,0) - max(K-A,0) = A - K` in the sample); AM-GM gives
`price_arith > price_geom` for **every** seed, reversed for puts.

**Refusals with reasons.** `stratified` raises `NotSupportedError` naming the
Brownian bridge -- stated in terms of the fixing schedule, because Slice 7's
`n_steps > 1` rule would let a one-fixing Asian through. `n_steps != 1` raises
rather than being silently ignored. Greeks raise on both routes with a message
saying it is a **scope boundary, not an impossibility** (unlike a digital, the
Asian pathwise estimator exists). Nothing registers `method='tree'` or
`'pde'`: an average needs a second state variable.

- `src/qpl/cases/asian_black_scholes.py` (new): a **fifth** id space, asserted
  disjoint from the other four, and the first whose two halves carry different
  evidence classes on purpose -- geometric rows `CLOSED_FORM` at 1e-13,
  arithmetic rows `STATISTICAL` against a 2 000 000-path reference with the
  tolerance at four times the combined standard error, and the
  Turnbull-Wakeman row a published benchmark **for the approximation** paired
  with four signed gap rows. A test asserts no arithmetic row can claim
  `CLOSED_FORM`.
- `tests/oracle/test_asian_vs_quantlib.py` (new, 27 tests, 2.4 s): four legs of
  different character -- analytic geometric at **2.487e-14**, Turnbull-Wakeman
  at 3.055e-13 (an order looser, and growing with `n` because `E[A^2]` is a
  5329-term double sum accumulated differently), the continuous engine as an
  independently-supplied convergence limit, and
  `MCDiscreteArithmeticAPEngine(controlVariate=True)` -- which uses the *same*
  Kemna-Vorst control -- at `|z| <= 0.72`. Day-count residual between the two
  fixing schedules: **1.11e-16**, one ulp. Fixing counts that do not divide 365
  cannot be expressed as calendar dates, which is why the 26-fixing published
  point is checked against its value rather than against QuantLib.
- `tests/test_asian_analytic.py` (new, 48 tests, 0.21 s),
  `tests/test_asian_mc.py` (new, 40 tests, 2.6 s),
  `tests/test_asian_variance_ratios.py` (new, 18 tests, 1.5 s),
  `tests/cases/test_asian_black_scholes_cases.py` (new, 33 tests, 1.4 s),
  `tests/test_examples_smoke.py`.
- `examples/asian_option_control_variate.py` (new, 0.67 s, curated smoke, also
  `--case fixings` at 0.21 s), `examples/README.md`.
- `docs/notes/asian_options_control_variate.md` (new), `docs/CURRICULUM.md`,
  `.agents/brain/brain.md`.
- Suite: 1174 tests, 99.1 s (from 1004 / 87.0 s); the five new Asian test files
  run in 7.5 s and the two new example smoke invocations in 3.4 s.

---

What changed in Slice 7 (files + bullets)

Phase 3's variance-reduction item: three estimators, each shipped with the
standard error that belongs to it, each measured. Still on `dev/curriculum`;
not pushed. Phase 2's last item (the non-uniform grid) was deferred by the
orchestrator on 2026-09-13 until the barrier case forces it, and this slice
went first.

- `src/qpl/engines/mc/variance_reduction.py` (new): antithetic (Glasserman
  4.2), control variate on the discounted terminal spot (4.1), stratified
  sampling of the terminal normal with proportional allocation (4.3). One
  entry point shared by the vanilla and digital MC engines; they differ only by
  the payoff callable.
- `MCConfig` gains `variance_reduction: str | tuple[str, ...] = "none"` and
  `n_strata: int = 64`. `"none"` takes the pre-existing code path untouched and
  is bit-for-bit what it was (four pinned `(value, stderr)` pairs asserted with
  `==`).
- **Each estimator's standard error is its own**, and each is recomputed from
  the realised sample in the tests: `ddof=1` over antithetic *pairs* (the naive
  all-paths version is over 1.4x larger), `ddof=2` regression residuals for the
  control variate, `sqrt(sum_i s_i^2/(K^2 m))` strata-weighted. Coverage of the
  reported 95% interval: 519/540 = 0.961 over 30 seeds x 5 120 draws, worst cell
  26/30 against a binomial floor of 24.
- **Refusals with reasons**: `stratified` with `n_steps > 1` raises
  `NotSupportedError` naming Brownian-bridge stratification; `antithetic` with
  `stratified` raises `InvalidInputError` (reflecting a stratified draw lands it
  in the mirror stratum).

**Measured variance factors** (50 seeds, 20 480 **normal draws**, K = 64;
predictions in brackets from a 400 000-draw pilot):

    point            antithetic   control      stratified  a+c     s+c
    ATM call         4.59 (4.01)  7.58 (6.89)     110.40   90.80   577.80
    OTM call K=120   3.04 (2.33)  2.71 (2.30)      42.41   92.35    84.61
    ATM digital      9.32 (9.38)  2.11 (2.45)     101.26    9.73    82.32

**Five contradicted expectations, all encoded.**

1. **Antithetic is worth MOST on the digital** (9.32), not least. Not
   convexity: with the in-the-money boundary at `z* = -0.15` the reflected pair
   sums to exactly 1 unless `|Z| < 0.15`, so the pair average is constant on
   88% of the sample and `rho_a = -0.7868`. The convexity story is right for
   the two calls (4.59 at the money, 3.04 off it) and is the wrong lens for the
   step payoff.
2. **Stratification gains `O(K)`, not `O(K**2)`**, on a vanilla: fitted
   exponents 1.0157 (ATM, residual 0.065) and 1.0384 (OTM, 0.058) over K in
   8...256. The outermost equal-probability stratum is unbounded and so is the
   payoff on it; its conditional variance does not shrink with K and sets the
   floor. Optimal allocation is the standard repair and is not implemented.
3. **For a digital the gain is `K p(1-p)/(f(1-f))`, `f = frac(K Phi(z*))`, and
   is NOT monotone in K**: 16 -> 20 strata measures 108.82 -> 48.50 (predicted
   89.64 -> 31.73) and 320 -> 512 measures 1014.18 -> 491.01 (predicted 1100.97
   -> 505.91). Which K is good is a property of the contract, not of the method.
4. **The combinations do not compose multiplicatively, in either direction.**
   Antithetic+control on the OTM call is 92.35 against a product of 8.24, and
   stratified+control on the digital is 82.32 -- *worse* than stratification
   alone (101.26).
5. **Variance reduction helps the CRN bump delta more than the price, except
   where the fitted coefficient interferes.** sd(delta) over 20 seeds at 20 480
   draws, h = 1e-2: none 0.004481, antithetic 0.001175, control 0.001639,
   stratified 0.000254 (variance factor **311** against the price's 110),
   anti+ctrl 0.001060, strat+ctrl 0.000367 -- worse than stratified alone,
   because `b` is refitted on each bumped sample and `b_up - b_dn` is noise CRN
   cannot cancel.

**The bias that is real and measured rather than excused.** The control
coefficient is fitted on the sample it corrects, so the estimator is biased.
`|b_same - b_pilot|` falls 0.017186 -> 0.001061 from N = 1 000 to 256 000
(fitted order 0.4952); the price gap it causes falls -4.830e-03 -> -5.848e-05
from N = 2 000 to 128 000 (fitted order **1.06**), consistently negative, about
1% of one standard error at the top end.

**Two traps found while writing the tests, both now written down.**

- `Market.rate(t)` recovers the rate from the curve's discount factor and
  returns `0.049999999999999996`, not the `0.05` passed in. Rebuilding a sample
  from the literal makes the terminal spots differ in the last bit and the
  stderr by one ulp, which breaks an exact-identity test. Read the market back.
- A 30-seed lag-1 autocorrelation of the stratified estimator reads +0.3274,
  which is 1.7 sampling standard deviations of nothing (+0.0256 at 200 seeds,
  -0.0069 at 1 000). The independence test uses 200 seeds for that reason.

**One claim this slice got wrong and then corrected in-branch.** The digital's
non-monotonicity was first pinned at `K = 16` beating `K = 64` (108.82 against
101.26) -- a real measurement, but the *opposite* of what the law predicts there
(89.64 against 104.84), i.e. a 50-seed sampling accident presented as a
mechanism. Replaced by two pairs where prediction and measurement agree on the
direction; the superseded reading is recorded in the test docstring rather than
deleted.

- `src/qpl/cases/mc_variance_reduction.py` (new): a **fourth id space**,
  asserted disjoint from the other three, and the first keyed by an *estimator*
  rather than an instrument. Fifteen `STATISTICAL` rows with `expected` and
  `tolerance` in **log space**, because an absolute tolerance on a ratio
  spanning 2.11 to 577.8 is either vacuous or impossible. `MC_VR_BAND = 1.7` is
  derived from the [0.47, 2.11] 99% range of a ratio of two 49-df variance
  estimates.
- The European cross-engine MC leg is now **stratified at 12 800 paths** rather
  than 200 000 plain ones: stderr 1.107e-02 against 3.283e-02, three times
  tighter for one sixteenth of the work, same four-stderr tolerance.
- `tests/test_mc_variance_reduction.py` (new, 71 tests, 0.83 s),
  `tests/test_mc_variance_ratios.py` (new, 26 tests, 2.5 s),
  `tests/cases/test_european_black_scholes_cases.py`,
  `tests/test_examples_smoke.py`.
- `examples/mc_variance_reduction.py` (new, 0.27 s, curated smoke, also
  `--case digital`), `examples/README.md`.
- `docs/notes/mc_variance_reduction.md` (new), `docs/CURRICULUM.md`,
  `.agents/brain/brain.md`.
- Suite: 1004 tests, 87.0 s (from 903 / 80.9 s).

---

What changed in Slice 6 (files + bullets)

Phase 2's fourth item: the cash-or-nothing digital as the discontinuous-payoff
case. Still on `dev/curriculum`; not pushed.

- `src/qpl/instruments/options.py`: `DigitalOption(kind, strike, expiry,
  cash=1.0)`, a frozen dataclass **outside** the `VanillaOption` hierarchy --
  registry lookup walks the MRO and a step payoff must not resolve to the
  vanilla engines. Shared validation moved into a module-level
  `_validated_kind` helper, so the rules and the messages have one copy without
  an inheritance edge. `src/qpl/instruments/payoffs.py`: `digital_payoff`, with
  **strict** inequalities on both sides (matching QuantLib's
  `CashOrNothingPayoff`, which is what makes the oracle the same contract).
- `src/qpl/engines/analytic/digital.py` (new): `cash e^{-rT} N(+-d2)`, derived
  from the risk-neutral probability in the docstring, citing Reiner & Rubinstein
  (1991), Risk 4(9). All five Greeks differentiated by hand. Note
  `gamma = -A phi(d2) d1 / (S^2 sigma^2 T)` **changes sign** at `d1 = 0`; every
  later gamma claim has to say where it is evaluated.
- `src/qpl/engines/tree/digital.py`, `src/qpl/engines/pde/digital.py`,
  `src/qpl/engines/mc/digital.py` (new), all registered for
  `(DigitalOption, BlackScholesModel, method)`. Each reuses the existing engine
  through a small hook rather than a copy: `_backward_induction` takes the
  terminal payoff as a callable, `_solve_grid` takes `payoff`/`dirichlet`
  callables, `price_european_from_terminal` takes a `payoff` callable. Vanilla
  output is bit-for-bit unchanged on all three paths.
- `src/qpl/engines/pde/pricers.py`: `PDEConfig.payoff_projection` in
  {`"none"`, `"cell_average"`}, read **only** by the digital engine -- the same
  arrangement `psor` has with the American one. `"cell_average"` puts the exact
  cell mean of the indicator at each node.
- **MC Greeks are refused, with a reason.** `greeks_digital` always raises
  `NotSupportedError` naming the Phase 3 likelihood-ratio estimator. It is
  registered (rather than left unregistered) so the caller gets that message
  instead of "Unsupported instrument/model/market combination". This is the
  first refusal in the package that is about a *quantity* rather than an
  instrument/method pair.

**Four contradicted expectations, all encoded.**

1. **Leisen-Reimer does not fail on a digital -- it is the best method here.**
   Measured order 2 at all three points (1.9844, 1.9839, 1.9836), one-signed and
   monotone, 4.55e+03 to 6.34e+05 times closer than CRR at n = 801. The
   mechanism is an exact identity, not a rate: the strike always falls strictly
   between the two central terminal nodes, so the lattice price **is**
   `cash e^{-rT} P(Bin(n, p) > n/2)` (asserted to 1e-13 against
   `scipy.stats.binom.sf`), and Peizer-Pratt chose `p` to make that `N(d2)`.
   CRR meanwhile is order 1 *only* at the money on odd n (1.0014, where `u d = 1`
   freezes the strike at the geometric mean of the two central nodes) and is
   **order 1/2** off it (block-RMS 0.5020 and 0.5037 over every odd n from 25 to
   801) with a sawtooth: 10 and 19 sign changes, errors spanning 5.11e-02 to
   3.35e-06.
2. **Midpoint alignment makes the projection a no-op.** With
   `K = (j + 1/2) ds` the strike lands on a *cell face*, so no cell straddles it
   and the cell average equals the point sample. Agreement 1e-16 at n = 100,
   bit-for-bit at 200/400/800. The two remedies are one remedy reached two ways;
   the projection exists for grids that cannot be aligned.
3. **The remedies that are orthogonal are the jump representation and
   Rannacher, and on a stressed grid both are required.** On `n_s = n_t`,
   Rannacher is nearly cosmetic (2.4e-06 correction against a 3.8e-02 error);
   on `n_s = 80 n_t` plain Crank-Nicolson **diverges** whatever the jump
   representation (delta order -1.00, gamma -2.00, gamma error reaching 52.19
   against a true gamma of -3.283e-04), and only Rannacher **plus** alignment or
   projection gives 2.004 / 2.058 / 2.059.
4. **The call + put identity breaks by 3.8% on an unaligned grid**, because a
   node sits exactly on the strike and the strict convention pays nothing there
   in either leg. Cell averaging repairs it exactly. What survives is the time
   scheme's own discounting error (-6.193e-10 for CN, +7.370e-08 for
   Rannacher's four implicit half steps), not round-off -- Rannacher is *less*
   exact on this identity and more accurate on everything else.

**The pathology, which did behave as predicted.** A jump is one derivative
worse than Slice 4's kink and the damage reaches the **price**: plain
Crank-Nicolson, unaligned, `n_s = n_t` in (100 ... 800) fits price order
**1.0012** (residual 0.0007), delta 0.8632, gamma 1.0153, with errors
-3.761e-02, -1.876e-02, -9.377e-03, -4.689e-03. At n = 800 that is 0.88% of the
value, against 2.229e-07 for the remedied grid. Remedied (Rannacher +
alignment): 1.9248 / 2.0360 / 2.0315. Projection on an unaligned grid: 2.0154 /
2.0061 / 1.9980.

**Monte Carlo does not notice the discontinuity.** stderr order **0.49990**
(residual 1.59e-04) over N in (5k, 20k, 80k, 320k) -- the cleanest power law in
the repository -- the reported `ddof=1` stderr matching the closed-form
Bernoulli one to 0.06%, |z| = 1.218 at 200k, and call + put exact *path by
path*. The bump-Greek refusal is measured, not asserted: CRN bump delta sd over
20 seeds at N = 40 000 is 0.00041 (h = 1), 0.00177 (0.1), 0.00474 (0.01),
0.01585 (0.001) -- 84.5% of the true delta, growing like `h^-1/2`.

**Gamma, reported rather than claimed.** Order 2.0315 at the money; at the
sign-change spot (true gamma -2.97e-19) the absolute error still falls at order
2.1226 but relative accuracy is undefined. Both pinned.

**QuantLib.** `ql.CashOrNothingPayoff` through `AnalyticEuropeanEngine` agrees
to 2.220e-16 on the price and 3.5e-18 or better on all five Greeks. Its FD
engine agrees at n = 800 to 3.514e-05 / 1.056e-06 / 9.922e-08, but its error
sequence is **not a power law** (residual 0.85 against 0.023 here, sign changes,
a meaningless fitted 4.85), for the same reason CRR is not: the log-spot mesher
places no node consistently relative to the jump. `dampingSteps = 2` does not
change it, so it is the mesher, not the time scheme. Honest caveat: QuantLib is
*ahead* at the money at n = 800 (1.80e-07 against 2.23e-07); what is asserted is
predictability under refinement, and at n = 100 it is 755x behind.

- `src/qpl/cases/digital_black_scholes.py` (new): nineteen rows over three
  points, a third id space asserted disjoint from the European and American
  ones. Cross-engine budgets, all derived -- Leisen-Reimer at n = 2001 within
  2e-08 (worst 5.284e-09), PDE at n = 800 within 8e-05 (worst 2.331e-05), MC at
  200k within 4 stderr. The two order-2 deterministic engines are three decimal
  orders apart and that gap is asserted, not glossed.
- `tests/test_digital_analytic.py`, `tests/test_digital_tree_convergence.py`,
  `tests/test_digital_pde.py`, `tests/test_digital_mc.py`,
  `tests/cases/test_digital_black_scholes_cases.py`,
  `tests/oracle/test_digital_vs_quantlib.py` (all new; 220 tests, 5.7 s).
- `examples/digital_option_cross_method.py` (new, 0.52 s, curated smoke, also
  `--case pde`): all four engines, the PDE table with and without each remedy,
  and the CRR sawtooth at consecutive odd `n` off the money.
- `docs/notes/digital_options_discontinuous_payoffs.md` (new),
  `docs/CURRICULUM.md`, `.agents/brain/brain.md`, `examples/README.md`.
- Suite: 903 tests, 80.6 s (from 679 / 72.9 s).

---

What changed in Slice 5 (files + bullets)

Phase 2's third item: American exercise by finite differences, the linear
complementarity problem solved with projected SOR. Still on `dev/curriculum`;
not pushed.

- `src/qpl/engines/pde/american.py` (new):
  - `price_american` / `greeks_american`, registered for `(AmericanOption,
    BlackScholesModel, "pde")`. The dispatcher now prices early exercise two
    ways; `method="analytic"` and `method="mc"` still refuse.
  - Each time step is the algebraic LCP `A x - b >= 0`, `x - g >= 0`,
    `(x - g)(A x - b) = 0`, with `A = I - theta dt L_h` and `L_h` the operator
    the European engine assembles. Solved by **red-black PSOR**: even indices,
    then odd, two vectorised numpy expressions per sweep instead of a Python
    loop over nodes. On a tridiagonal matrix that is exactly a Gauss-Seidel
    sweep in a permuted order, and consistent ordering (Young) makes the
    spectral radius and the optimal `omega` the same as the natural order's.
  - `PSORConfig(omega=1.2, tol=1e-8, max_iter=10_000, on_max_iter="raise")` on
    `PDEConfig.psor`, read only by this engine. `tol` is **absolute** on the max
    update; an exhausted cap raises naming the step, or is recorded in
    `meta["psor_unconverged_steps"]` with `psor_converged=False`. Never both
    silent and returned.
  - American Dirichlet values are the European ones lifted to the payoff:
    `max(g(0), V_eu(0))` and `max(g(s_max), V_eu(s_max))`. For a put that turns
    `K e^{-r tau}` into `K`; for a call nothing changes, which is why the
    no-dividend American call keeps its European boundary.
  - `meta` carries sweeps per step and their summary, the exercise boundary
    **with its times** (Rannacher makes the tau grid non-uniform), and the LCP
    measured rather than assumed: worst complementarity, worst slack, worst
    operator residual over the whole march.
  - `sigma = 0` and `T = 0` import `_degenerate_value` from
    `qpl.engines.tree.american`, because that is a fact about the model and not
    about a discretisation -- including the Slice 2 `q > r` interior turning
    point.
- `src/qpl/engines/pde/pricers.py`: `_build_grid`, `_payoff`, `_dirichlet` and
  `_operator` extracted so the American engine shares them; `_greeks_from_grid`
  and `_greeks_by_bump` take the solver as a parameter. **European prices and
  Greeks are bit-for-bit unchanged** over 160 configurations x 6 quantities.
  `_greeks_from_grid` gains an `obstacle` argument, used for one thing: theta.
- **omega = 1.2 is measured, not cited.** Mean sweeps per step on
  `n_s = n_t = n`, ATM put:

    n       w=1.00  w=1.05  w=1.10  w=1.15  w=1.20  w=1.30  w=1.50
    100      5.96    7.00    8.51   10.03   11.33   14.38   24.52
    400      9.72    7.98    8.07    9.28   10.65   13.50   22.49
    1600    21.12   18.97   16.78   14.65   12.75   12.38   20.13

  The per-grid optimum drifts 1.00 -> 1.30 because `dt / ds**2` grows like `n`.
  1.2 is the flattest column, never worse than 1.9x the optimum. On
  `n_s = 80 n_t, n_t = 20` the optimum is 1.8 or beyond (1367 sweeps at 1.0,
  173 at 1.8).
- `tests/test_pde_american.py` (new, 55 tests, 4.7 s),
  `tests/test_pde_american_convergence.py` (new, 11 tests, 5.6 s),
  `tests/oracle/test_pde_american_vs_quantlib.py` (new, 11 tests, 5.5 s),
  `tests/cases/test_american_black_scholes_cases.py`,
  `tests/test_tree_american.py`, `src/qpl/cases/american_black_scholes.py`,
  `examples/american_put_cross_method.py` (new),
  `docs/notes/pde_american_psor.md` (new):
  - **Price order 1.8502** (residual 0.0265) over `n = 50 ... 800` and
    **1.8506** (0.0261) over the disjoint `100 ... 1600`, against
    `AMERICAN_BRACKETED_LIMIT`. Errors -6.96e-02 ... -1.25e-04, all negative:
    from below, where CRR comes from above, so they bracket. Local orders
    1.739, 1.888, 1.895, 1.837, 1.766, 1.476 -- already falling toward 1.
  - **Unaligned is not a power law**: residual **0.2688** against 0.0265. The
    test pins the residual, not the order it reports.
  - **The LCP holds**: slack exactly 0.0, operator residual to -1.1e-07,
    complementarity at most 1.1e-07, and it falls three decades when `tol` falls
    four.
  - **Boundary**: monotone at every level with zero tolerance, no NaN prefix,
    `K - ds/2` at expiry. Against the `n = 2000` lattice at seven times: +0.19,
    +0.45, -0.04, -0.10, +0.72, +0.27, -0.10, worst 0.72 against a combined node
    spacing of 1.4412, both signs present.
  - **Three-engine agreement** (`qpl.cases`, two new INDEPENDENT_ENGINE rows):
    6.040e-04 at the ATM put (tolerance 1.5e-03) and 1.624e-04 at L&S row 1
    (5e-04). New `LS2001_BRACKETED_LIMIT = 4.4866721476`.
  - **Oracle**: gaps to QuantLib 2.331e-04, 4.209e-05, 1.215e-04 against 8e-04,
    at `T = 1.0` with `Actual365Fixed`.
- **Five contradicted expectations, all encoded rather than smoothed over:**
  1. No single optimal `omega` -- it is set by `dt / ds**2`, and the measured
     optimum is 1.00 at `n = 100`, not the 1.2-1.5 the slice suggested.
  2. Sweep counts do **not** grow on `n_s = n_t = n` (flat 100 -> 1600, fit
     residual 0.195, not a power law): the stiffer matrix and the better
     starting iterate cancel. Isolated on `n_s = 8 n_t` the growth is clean --
     exponent **0.787**, residual 0.0018, between optimal SOR's 0.5 and
     Gauss-Seidel's 1.0.
  3. **The projection makes PSOR faster.** The no-dividend American call, with
     an empty active set, needs 12, 12, 16, 33, 59 sweeps over `n = 50 ... 800`
     against the put's flat 11-12.
  4. The boundary has **no parity structure** to be monotone within: the grid is
     the same node set at every time level, so the whole sequence is monotone.
     The Slice 2 parity caveat is a lattice artefact, not a fact about American
     boundaries.
  5. **QuantLib's American FD is order 1.0328**, not comparable to this engine's
     1.8502. `dampingSteps` is not the cause (1.5e-05 at grid 3200, wrong
     direction). Candidates: the unaligned log-spot mesher and
     `FdmAmericanStepCondition`, which projects between steps instead of solving
     complementarity inside them. Not separated here, and said so.
- **Theta in the exercise region** is 0 exactly, not the PDE identity's
  `rK - qS` (`+5.0` at `S = 40, K = 100`). `meta["theta_source"]` says which.
- `qpl.numerics.linear_systems.sor_solve` could not be the solver (dense
  `O(n**2)` sweep, no projection hook) but is the **reference**: unprojected,
  the red-black sweep reproduces it and the LAPACK banded solve to 3.2e-13.
- Suite: 679 tests, 72.6 s (from 597 / 62.2 s). `ruff check .` clean.

---

What changed in Slice 4 (files + bullets)

Phase 2's first item: Rannacher start-up and Greeks read off the
finite-difference grid, with the Crank-Nicolson gamma pathology pinned,
repaired, and confirmed in a second engine. Still on `dev/curriculum`; not
pushed.

- `src/qpl/engines/pde/pricers.py`:
  - `PDEConfig` gains `time_stepping: Literal["theta", "rannacher"] = "theta"`
    and `greeks_method: Literal["grid", "bump"] = "grid"`. `"theta"` marches
    `n_t` steps of `dt` exactly as before and was **bit-for-bit** the previous
    engine when it landed (288 configurations diffed, 0 mismatches).
  - `"rannacher"` replaces the first **two** nominal steps by **four fully
    implicit steps of `dt/2`** and then continues with `theta`. Halving is
    exact in binary floating point, so the four half steps cover `2 dt` to the
    last bit, total time stays `n_t dt`, and `n_t + 2` steps are taken.
    `n_t >= 2` is required. Rannacher (1984), Numerische Mathematik 43;
    Giles & Carter (2006), JCF 9(4) -- their "two half steps, twice" is the
    same four steps counted as two pairs.
  - `greeks_european` now returns **all five** Greeks. Delta and gamma from
    second-order central stencils; theta from the PDE identity
    `V_t = -(1/2 sigma^2 S^2 V_SS + (r-q) S V_S - r V)`; vega and rho by
    bump-and-revalue on the same grid (`h = 1e-2` and `1e-4`, both re-measured
    here). `meta` names the source of each and carries
    `theta_backward_difference` as the cross-check.
  - The spot is usually **not a node** -- with `strike_alignment="midpoint"`
    and `S = K` it sits exactly halfway between two -- so the *Greeks* are
    interpolated between the two nearest nodes, which keeps `O(ds^2)`.
    Interpolating the price and differencing there would divide that error by
    `ds^2`.
  - `_solve_tridiagonal` moved from a Python Thomas loop to
    `scipy.linalg.solve_banded` in its own commit: **7.4x-28.7x** on a price
    call, `pytest -q` 109.1 s -> 60.6 s, worst price difference **7.2e-13**
    over 160 configurations. Round-off, not zero -- the earlier bit-identity
    claims are against the Thomas loop and stop holding at the 13th decimal.
- `tests/test_pde_greeks.py` (rewritten, 21 tests), `tests/test_pde_ch4.py`,
  `tests/test_pde_pricing.py`, `tests/cases/test_european_black_scholes_cases.py`,
  `tests/oracle/test_pde_vs_quantlib.py` (new, 13 tests, 3.0 s),
  `docs/notes/pde_greeks_and_rannacher.md`:
  - **Smooth case** (`S=K=100, r=5%, q=0, sigma=20%, T=1`, aligned,
    `n_s=n_t=n` over 50...800): delta **2.001**, gamma **2.064**, theta
    **2.024** with plain CN; 2.001 / 2.067 / 2.024 with Rannacher. Price order
    1.9972 and 1.9973, with Rannacher costing a flat **5.4%** on the constant
    (`n^2|err|` 19.61 -> 20.66) and nothing on the order.
  - **Pathology** (`T=0.05`, `n_s = 80 n_t`, unaligned): plain CN fits gamma at
    **-1.043** (residual 0.018) -- refinement makes it *worse* -- with relative
    errors -25.9%, +55.7%, +113.4%, +227.7%. Rannacher fits **1.933**. Aligned:
    **-0.915** against **1.888**, i.e. alignment buys a factor of nine and not
    the order.
  - **The price does not warn you**: at `n_s=1600, n_t=20` the CN price is
    0.12% out while its gamma is 113% out and its theta 99% out.
  - **Oracle**: agreement with QuantLib at `n=800` over three points and both
    kinds within 6e-04 (price), 1.5e-04 (delta), 4e-06 (gamma) -- tolerances
    derived from both engines' measured errors, each engine's own residual
    asserted separately. Both order two on price and delta.
  - `qpl.cases` gains ten `CLOSED_FORM` PDE-Greek rows at the reference ATM
    call and put on one fixed grid, tolerances derived with 3.5x-4.1x headroom.
- **Four contradicted expectations, all encoded rather than smoothed over:**
  1. The slice expected the gamma pathology at `n_s = n_t = n`. **It is not
     there.** On that path `dt` shrinks as fast as `ds`, `lambda dt` at the
     strike stays near one, and CN damps the stiff modes fine -- the two time
     steppings agree to three significant figures in all three Greeks. The
     pathology needs `dt` large relative to `ds^2`, which that refinement hides.
     It is pinned on `n_s = 80 n_t` instead.
  2. **QuantLib's `FdBlackScholesVanillaEngine` does not damp by default.** The
     slice said it used Rannacher. Its default is `FdmSchemeDesc.Douglas()`
     with `dampingSteps = 0`, pinned bit-for-bit, and on the same stressed
     grids its gamma is out by factors of 180 to **1353** -- two to three
     orders worse than ours, because its log-spot mesher packs more nodes near
     the strike and is stiffer there. `dampingSteps = 2` fixes it. Both
     undamped engines diverging is what proves the pathology belongs to the
     scheme and not to this package. (`FdmSchemeDesc.CrankNicolson()` is a
     different scheme *type* that agrees with `Douglas()` to 1.8e-15: Douglas
     splitting in 1D is the theta scheme, so the choice is not a choice.)
  3. **Theta from the last two time levels is first order**: measured 1.054
     (residual 0.0020) with the spatial grid fixed, 20x-200x worse than the
     identity at every level. The identity is reported; the difference is the
     cross-check.
  4. "Replace the first time step by four steps of `dt/2`" does not add up --
     four half steps cover `2 dt`, i.e. the first *two* steps. Implemented that
     way and counted in a test.
- **The old bump path stalls**, and for two reasons rather than one. Delta
  error -8.248e-05, -8.509e-05, -8.574e-05 at `n = 400, 800, 1600` against the
  grid path's -1.430e-04, -3.594e-05, -9.007e-06; orders **0.418**
  (residual 0.563) against **2.001** (0.021). The fixed `O(h^2)` bias of the 1%
  bump is the obvious cause; the one that was not anticipated is that
  `s_max = multiplier * spot`, so the three solves sit on three
  *differently-aligned* grids and their errors do not cancel -- which is why
  its gamma sequence is not a power law at all (residual 0.73 against 0.04).
- One thing the example surfaced that no test had asserted: **delta is not
  polluted by the ripple**, because the CN oscillation alternates in sign
  between neighbouring nodes and a first central difference cancels it while a
  second difference doubles it. Theta tracks gamma exactly, because it is built
  from gamma through the identity.
- `examples/pde_greeks_demo.py` rewritten from a legacy printout into two
  measured error tables with fitted orders (`--case smooth`, `--case startup`),
  both under a second, both in the curated smoke list with
  `startup_order theta_gamma=-1.` as a required key.
- Remaining in Phase 2: a non-uniform grid at the strike, PSOR for American
  exercise, and a digital option. The `qpl.numerics.linear_systems` question
  now has a recorded answer for the European solve (a direct banded solve wins)
  and moves to the PSOR slice, where projection forces an iteration.

What changed in Slice 3 (files + bullets)

Phase 1's last scheduled item: the Leisen-Reimer lattice, order 2 for European
payoffs, and a careful account of the three places that order 2 does *not*
reach. Still on `dev/curriculum`; not pushed.

- `src/qpl/engines/tree/lattice.py`, `src/qpl/engines/tree/pricers.py`,
  `src/qpl/engines/tree/american.py`, `src/qpl/engines/tree/__init__.py`:
  - `TreeConfig(scheme="leisen-reimer")` prices European and American vanillas
    and their Greeks through the existing `method="tree"` dispatch. The lattice
    builder is the only place that knows the scheme -- every pricer, the
    Bellman step and the Greek estimators read only `up`, `down`, `p`,
    `discount`.
  - New: `peizer_pratt_inversion` (**method 2**, the variant with the
    `0.1/(n+1)` term), `leisen_reimer_parameters`, and
    `lattice_parameters(scheme=...)`. `CRRLattice` -> `BinomialLattice`, plus a
    `spot_centred` field.
  - `p = h(d2, n)`, `p' = h(d1, n)`, `u = e^{(r-q)dt} p'/p`,
    `d = (e^{(r-q)dt} - p u)/(1 - p)`. The second form (rather than the equal
    `e^{(r-q)dt}(1-p')/(1-p)`) reprices the forward to round-off, which is what
    makes parity exact on the lattice.
  - No no-arbitrage check needed: `d1 > d2` and `h` increasing give `p' > p`,
    which *is* `d < growth < u`.
  - **CRR is bit-for-bit unchanged** -- 168 prices and Greek tuples diffed
    against the previous commit.
- **Even `n` is rejected**, not rounded up, and the reason is measured rather
  than asserted. `P(Bin(n,p) > n/2)` counts a whole number of outcomes only for
  odd `n`. QuantLib rounds up inside the tree but not inside the engine's time
  grid, and its even-`n` price is an **order-1** approximation: ATM call error
  -1.330e-02 at `n=800` against -5.518e-07 at `n=801`, a factor of 24 000. A
  20-line mirror reproduces ~95% of that error, which identifies the mechanism.
- `tests/test_tree_leisen_reimer.py` (123 tests, 0.5 s),
  `tests/test_tree_lr_convergence.py` (20 tests, 0.3 s),
  `tests/oracle/test_lr_vs_quantlib.py` (23 tests, 0.8 s),
  `docs/notes/leisen_reimer.md`:
  - **European order 1.9840** (residual 0.0087) at the money; **1.9842** and
    **1.9709** at the two off-money points where CRR's fit was 1.21-1.48 with
    residuals 0.37-1.52 in *both* engines. Errors 507x/35x/1137x smaller than
    CRR's at `n=101` and 3966x/2673x/5229x at `n=801`.
  - **No oscillation, measured**: over `n = 101..121` the CRR error changes
    sign 20 times; the LR error on the odd counts there is one-signed and
    monotone.
  - **American order 1.0641** (residual 0.0176) against CRR's 0.9872 -- order
    1, as expected, because the boundary error does not care about the terminal
    grid. Constant 3.79x-4.97x better, and *opposite sign*: LR below the limit
    at every `n`, CRR above, so the two schemes bracket it at a shared `n`.
  - **Richardson still fails on American**: 1.3423 with residual 0.7187 and
    non-monotone extrapolated errors. On European it works well (2.9577, error
    1.465e-09 at the (401,801) pair), which is the contrast worth keeping.
  - Oracle: European agreement with QuantLib's `leisenreimer` to **2.71e-11**,
    American to **7.3e-12**, and 1.52e-04 against its FD engine (half Slice 2's
    3.71e-04 for CRR at the same `n`). Two QuantLib quirks pinned: the even-`n`
    behaviour, and isolated `n` (501, 1601, 8001) where its American LR leaves
    its own convergence curve by up to 9.5e-03 while ours stays on it.
- **Four contradicted expectations, all encoded rather than smoothed over:**
  1. The slice expected lattice **delta at order ~2** at the money. Measured
     **1.00**, and so are gamma and theta, at three points and both kinds. The
     estimators read levels 1 and 2 and use the result at level 0; that `O(dt)`
     substitution dominates the `O(1/n^2)` price error. No lattice fixes it --
     an extended tree below the root would.
  2. The fallback -- "at least the Greek constants improve" -- is false for
     delta and gamma, whose LR errors sit **between** CRR's odd and even
     errors. Theta improves 4x-14x, vega 3x-32x, rho 190x-5000x. A first draft
     of this claim in a docstring said "twelve times better delta"; it was
     wrong, was corrected in its own commit, and is now asserted by a test
     rather than narrated.
  3. The LR **vega error is flat** at 3.06e-03 across `n` in {101..801}: the
     `O(h^2)` bias of `VEGA_BUMP=1e-2` is now the binding term, having been
     invisible under CRR's oscillation. `VEGA_BUMP` should be re-measured per
     scheme in a later slice.
  4. The American order is 1.06, not 2 -- anticipated by the slice statement,
     and confirmed rather than forced.
- **One real bug, surfaced by `u d != 1`.** `lattice_delta_gamma_theta` read
  theta as `(V(2,1) - V(0,0))/(2 dt)`, a pure time difference only when
  `S(2,1) = S0`. On an LR lattice the offset is 4.6e-02 at `S=100, K=120,
  n=801`; uncorrected the theta error is **5.235** at a fitted order of
  **-0.002** (it does not converge) while looking plausible. Subtracting the
  second-order Taylor expansion in spot restores order **1.000**. Applied only
  when `spot_centred` is False, so CRR theta is untouched.
- `src/qpl/cases/*`: three European order-2 rows, three American rows (order,
  bracketing, Richardson negative finding), `AMERICAN_BRACKETED_LIMIT`,
  `TREE_LR_REFERENCE_N_STEPS=2001`, `TREE_LR_KNOWN_VALUE_TOLERANCE=2.5e-7`.
  The cross-engine test is now **five** legs and asserts the order gap: at 2001
  steps against CRR's 2000, the LR error must be >= 1000x smaller (measured
  11 000x).
- `examples/tree_convergence.py --scheme {crr,leisen-reimer}`; default output
  byte-for-byte unchanged. `tests/test_examples_smoke.py` now carries
  `(script, args, keys)` triples so one example can have several curated
  invocations.
- Cross-platform hygiene (requested mid-slice): the Slice 1 pinned pre-refactor
  American values compared with `==` and failed on Linux CI by 2 ULP. Relaxed
  to `math.isclose(rel_tol=1e-12)`, and the `atm_1y_american_put_reference` row
  tolerance from 1e-12 to 1e-10 -- measured with `math.nextafter`, a one-ULP
  nudge to `u` moves the `n=8001` American put by 6.7e-13, so 1e-12 was inside
  the platform noise. Genuinely same-process identities were left exact.

## Phase 1 is complete

Remaining Phase 1 items are **deferred, not scheduled**: a Bermudan instrument
(no second case wants one) and trinomial trees (no case forces one -- LR gets
order 2 with no extra machinery). Next is Phase 2, PDE rigor.

---

What changed in Slice 2 (files + bullets)

The rest of Phase 1's American half: exercise style becomes an instrument
property, and the CRR tree learns to price it. Still on `dev/curriculum`; not
pushed.

- `src/qpl/instruments/options.py`, `src/qpl/instruments/__init__.py`,
  `src/qpl/pricing.py`, `tests/test_tree_american.py`:
  - `VanillaOption(kind, strike, expiry)` carries the validation;
    `EuropeanOption` and `AmericanOption` are **siblings** under it, not parent
    and child. That is load-bearing: registry lookup walks the MRO, so an
    `AmericanOption` subclassing `EuropeanOption` would resolve to the European
    closed form and return a plausible wrong number instead of refusing.
    Pinned by a test.
  - The tree registers a second key, `(AmericanOption, BlackScholesModel,
    "tree")`, for price and Greeks. Analytic/MC/PDE stay European-only and
    refuse an `AmericanOption` through the ordinary lookup with the ordinary
    message — no engine grows an `if instrument.american: raise`. Tested for
    all three, for both `price` and `greeks`.
  - No change to the ADR-0005 contract (this is an application of its key), so
    **no ADR-0006**; brain.md sections 2/3/4 updated instead.
- `src/qpl/engines/tree/american.py`, `src/qpl/engines/tree/pricers.py`,
  `src/qpl/engines/dp/*`:
  - `price_american` / `greeks_american`: calls and puts, continuous yield,
    `meta["exercise_boundary"]` per level and `meta["early_exercise_node_count"]`,
    and delta/gamma/theta through `lattice_delta_gamma_theta`, now shared with
    the European path (same code, because the estimators do not care how the
    node values were produced). Vega/rho by bump at the same `n`.
  - Spot levels come from precomputed `u**i`/`d**i` tables: `O(n)` calls to
    `pow` instead of `O(n**2)`, same floating-point nodes, which is what makes
    an `n = 8001` reference cost 0.17 s.
  - `engines.dp.price_american_put_binomial` is now a **thin wrapper** over the
    dispatcher engine (kept rather than deleted because a private lab notebook
    calls it with `return_lattice=True`, and `labs/` is not mine to edit). Five
    Slice 1 values still pass with a **zero** tolerance.
- `tests/test_tree_american.py` (60 tests), `tests/test_tree_american_convergence.py`
  (14 tests, 1.3 s), `docs/notes/american_exercise_on_trees.md`:
  - **Measured** (S=K=100, r=5%, q=0, sigma=20%, T=1, American put, reference
    = this engine at n=8001): odd `n` order **1.0141** (residual 0.0174),
    above the reference at every level; even `n` **0.9722** (0.0242), below at
    every level. **The odd/even bracketing survives early exercise.** American
    call with q=6%: **1.0232** / **0.9674**.
  - The scaled constants do **not** settle like the European tree's 1.7529 /
    1.9994 — they drift (odd 1.391→1.313, even 0.822→0.924). Partly the
    reference's own 1.80e-04 error (quantified against a 64000/64001 average),
    partly real: the boundary is resolved only to the node spacing, an error
    with no parity structure. So the test pins the *sign structure*, not a
    constant.
  - **Richardson extrapolation does not restore order 2** for the American
    put: **0.2960** (residual 0.2712) odd, **0.6447** (0.3562) even, with the
    extrapolated error flattening at ~1.5e-04. It still cuts the error 10-220x
    against the raw sequence; both halves are asserted. Verified against the
    better reference that the floor is not just the reference's accuracy.
  - Lattice Greeks at order ~1 (delta/gamma/theta 1.164/1.185/1.259 odd,
    1.074/1.079/1.076 even), band [0.9, 1.4] with the reason stated.
  - Exact identities with **zero** tolerance: American call = European call bit
    for bit at q=0 (all five Greeks too), American put = European put at r=0.
- `src/qpl/cases/american_black_scholes.py`,
  `tests/cases/test_american_black_scholes_cases.py`:
  - **Contradicted expectation, and the best result of the slice.** The slice
    required the L&S (2001) Table 1 row-1 put at n=5000 within 2e-03 of
    **4.478**. Measured: **4.486710**, 8.71e-03 away. QuantLib's FD and
    binomial American engines land at ~4.4867 too, so the engine is not wrong
    — the *instrument* is different. Those options are exercisable **50 times
    per year**; restricting exercise to 50 dates on the same lattice gives
    **4.477922** (n=5000) and 4.477826 (n=40000). Carried as two rows on one
    spec: `PUBLISHED_BENCHMARK` at tolerance 5e-04 (the published figure's own
    three-decimal precision) and `NEGATIVE_FINDING` asserting the continuous
    gap *exceeds* 2e-03, pinned at 8.71e-03 ± 5e-05.
  - Nine rows total; the in-repo reference 6.0905564143067235 at n=8001 says in
    its own `source` that it is **not** a published table value.
- Two further contradictions, encoded rather than smoothed over:
  - At sigma=0 the American put is **not** always
    `max(intrinsic now, discounted forward intrinsic)` — that needs `r >= q`.
    For `q > r` the objective has an interior maximum at
    `t* = log(rK/(qS0))/(r−q)`; measured 83.9506 vs 81.1993 at
    S0=K=100, r=2%, q=50%, T=10.
  - The extracted boundary is **not** monotone level by level; it is monotone
    within each node-grid parity and zigzags by at most one grid offset
    (worst drop 1.3845 vs offset 1.4043 at n=200). At expiry it is the
    in-the-money node adjacent to K: `K/u²=97.2112` (put), `Ku²=102.8688`
    (call).
- `tests/oracle/test_american_vs_quantlib.py`:
  - `FdBlackScholesVanillaEngine` (tGrid=xGrid=3200) agrees with the tree
    (n=8001) to **3.71e-04** / 1.20e-04 / 3.10e-04 at three points, inside a
    1e-03 tolerance **derived** from both refinements — they approach from
    opposite sides, so the gap is the sum of their errors. QuantLib's FD order,
    fitted on successive differences so no reference value is needed:
    **1.0856** (residual 0.0211).
  - Slice 1's negative finding survives early exercise: `n·(qpl − QuantLib
    CRR)` constant to better than 0.5% over n∈{200,800,3200} at **−0.026413**
    (ATM put), **−0.013260** (L&S row 1), **+0.007104** (call with q=6%).
- `examples/american_put_binomial_dp.py`, `tests/test_examples_smoke.py`,
  `docs/CURRICULUM.md`, `docs/curriculum_provenance.md`,
  `docs-site/docs/reference/dp.md`, `README.md`, `.agents/brain/brain.md`:
  - The example now goes through the dispatcher and prints the premium and
    boundary samples; smoke keys extended. Curriculum "Delivered" gained Slice
    2 and Phase 1's remaining items were re-scoped (Leisen-Reimer is now
    better motivated: Richardson does not substitute for it on American
    payoffs).

Deferred (intentional, Slice 2)
- Leisen-Reimer, PSOR and Longstaff-Schwartz are later slices, as instructed.
- No Bermudan instrument: the 50-date restriction that reproduces the L&S
  benchmark lives in the test, on the shared lattice. If a second case wants
  it, that is when it earns a type.
- `engines.dp.price_american_put_binomial` kept as a deprecated wrapper rather
  than deleted, only because of the private lab notebook. It can go the moment
  that notebook is updated.

How to validate Slice 2 quickly
- `PYTHONPATH=src python examples/american_put_binomial_dp.py`
- `pytest -q tests/test_tree_american.py tests/test_tree_american_convergence.py tests/cases/test_american_black_scholes_cases.py`
- `pytest -q tests/oracle` (needs the `oracle` extra; skips cleanly without it)

---

What changed in Slice 1 (files + bullets)

Phase 1 of the curriculum: the CRR binomial tree, and the dispatcher refactor
it forced. Still on `dev/curriculum`; not pushed.

- `src/qpl/engines/registry.py`, `src/qpl/pricing.py`,
  `src/qpl/engines/{analytic,mc,pde}/*`, `tests/test_engine_registry.py`,
  `.agents/brain/adr/0005-engine-registry.md`:
  - Replaced the `isinstance` ladder (one branch per method, duplicated across
    `price` and `greeks`) with a registry keyed by
    `(instrument type, model type, method)` plus a per-method `MethodSpec`
    carrying the keyword contract (`cfg` type, MC's optional `bumps` and its
    validator). Public signatures, error types, error messages and the
    validation ordering are all unchanged and now pinned by a test. Lookup
    walks the MRO, preserving `isinstance` semantics. `Market` is type-checked
    but is not part of the key; ADR-0005 says why. Adding an engine is now one
    `register(...)` call.
- `src/qpl/engines/tree/{__init__,lattice,pricers}.py`,
  `src/qpl/engines/dp/american_put_binomial.py`, `tests/test_tree_pricing.py`:
  - Added `qpl.engines.tree`: `crr_parameters` (with the no-arbitrage check),
    `crr_spot_level`, `build_recombining_spot_tree` (moved out of
    `engines.dp`), `TreeConfig(n_steps=200, scheme="crr")`, `price_european`
    (vectorized backward induction, one numpy expression per time level) and
    `greeks_european` (delta/gamma/theta off the lattice, vega/rho by bump).
    Wired in as `method="tree"`. T=0 and sigma=0 behave as in the other
    engines.
  - The American-put DP engine now consumes the shared lattice; prices are
    **bit-identical** across 45 configurations, five of them pinned with a
    zero tolerance.
- `tests/test_tree_convergence.py`, `docs/notes/crr_tree_convergence.md`,
  `examples/tree_convergence.py`:
  - **Measured** (S=K=100, r=5%, q=0, sigma=20%, T=1, European call): odd `n`
    fitted order **1.0010** (residual 0.0005), tree **above** BS, constant
    `n·|err| -> 1.7529`; even `n` order **0.9987** (residual 0.0007), tree
    **below** BS, constant **1.9994**. The two subsequences bracket the true
    value — the textbook story held, and is asserted as measured rather than
    assumed. Richardson on consecutive odd pairs: order **1.9590** (residual
    0.0224), *not* 2, and the note says why; error at `n=401` drops from
    4.372e-03 to 5.710e-07.
  - Lattice Greeks converge at order ~1 too (delta/gamma/theta 1.004/1.002/
    1.001 odd, 0.999/1.010/1.010 even). Vega and rho carry the oscillation;
    bump sizes chosen from a measured scan, not habit.
  - One-step and two-step trees match hand-computed *replication* values
    (max residual 2.3e-14); put-call parity holds on the tree at every `n`
    (8.3e-12 at n=800) because `p` reprices the forward exactly.
- `tests/oracle/test_tree_vs_quantlib.py`:
  - **Contradicted expectation.** The slice expected 1e-10 agreement with
    QuantLib's `BinomialVanillaEngine<CoxRossRubinstein>`. QuantLib's
    `CoxRossRubinstein` uses the *log-space* probability
    `1/2 + (r−q−σ²/2)dt/(2σ√dt)` on the same lattice, not the
    forward-matching `p = (e^{(r−q)dt}−d)/(u−d)`. The two differ at
    `O(dt^1.5)` per step, i.e. `O(1/n)` in price: `n·(qpl − QuantLib)` is
    constant to better than 1% over `n` in {50, 200, 800}. A twelve-line
    reimplementation of QuantLib's probability reproduces its engine to
    8.3e-11, identifying the mechanism. Our closed form does match
    `AnalyticEuropeanEngine` to 1.1e-14.
  - Second negative finding: away from the money the order-1 constant is
    erratic (fitted order 1.21-1.48, log residual 0.37-1.52) for *both*
    engines, so the clean odd/even story is an at-the-money story.
- `src/qpl/cases/european_black_scholes.py`,
  `tests/cases/test_european_black_scholes_cases.py`:
  - Two `CONVERGENCE_ORDER` rows for the odd/even claims, plus
    `TREE_REFERENCE_N_STEPS=2000` and `TREE_KNOWN_VALUE_TOLERANCE=2.5e-3`
    derived from the measured constant (predicted 1.00e-3, measured 9.998e-04).
    The cross-engine test now has a fourth leg.
- `examples/README.md`, `tests/test_examples_smoke.py`,
  `docs/CURRICULUM.md`, `.agents/brain/brain.md`:
  - `examples/tree_convergence.py` added to the curated set and to both halves
    of the smoke harness; curriculum "Delivered" gained a Slice 1 entry with
    the measured numbers; brain.md sections 2/3/4/8 reconciled.

Deferred (intentional, Slice 1)
- Leisen-Reimer, American exercise on the European tree, and trinomial trees
  are later slices; `TreeConfig.scheme` exists so adding LR does not change
  this config's identity.
- `qpl.engines.dp.price_american_put_binomial` is still not on the dispatcher:
  it takes `strike`/`expiry` rather than an instrument, and American exercise
  becomes an instrument property in a later Phase 1 slice.

How to validate Slice 1 quickly
- `PYTHONPATH=src python examples/tree_convergence.py`
- `pytest -q tests/test_tree_convergence.py tests/test_tree_pricing.py tests/test_engine_registry.py`
- `pytest -q tests/oracle` (needs the `oracle` extra; skips cleanly without it)

---

What changed in Slice 0 (files + bullets)

This is the first slice of the Textbook-Driven Development curriculum
campaign (ADR-0004); none of it has reached `main` yet (`git log
main..dev/curriculum`), which will be fast-forwarded by the project owner at
the slice boundary.

- `AGENTS.md`, `.agents/brain/brain.md`, `.agents/brain/adr/0003-*.md`,
  `.agents/skills/**`, `.github/workflows/deploy-docs.yml`, `.gitignore`:
  - Adopted the lab-driven pre-1.0 policy (ADR-0003): `labs/` is gitignored
    and private, never public cargo.
- `src/qpl/engines/mc/processes.py`, `src/qpl/engines/mc/sampling.py`,
  `src/qpl/engines/mc/pricers.py`, `src/qpl/engines/pde/pricers.py`,
  `src/qpl/pricing.py`, `src/qpl/market/market.py`, `src/qpl/market/stats.py`,
  `src/qpl/utils/labs.py`:
  - Added exact/Euler GBM simulators, sampling primitives, MC Theta, PDE
    Delta/Gamma, an implied-volatility solver
    (`qpl.engines.analytic.black_scholes.implied_volatility`), and
    smoke-mode lab utilities.
- `src/qpl/engines/dp/*`, `src/qpl/numerics/*`, `src/qpl/transforms/*`,
  `src/qpl/dependence/*`:
  - Added the numerical-methods modules the private labs forced into
    existence: backward-induction optimal stopping and a CRR American put
    binomial pricer (`engines.dp`), Jacobi/Gauss-Seidel/SOR and
    trapezoid/Simpson/Gauss-Legendre (`numerics`), Stehfest Laplace inversion
    (`transforms`), and Gaussian copula sampling/concordance (`dependence`).
- `examples/*`, `notebooks/*`, `docs-site/**`, `CHANGELOG.md`,
  `tests/test_examples_smoke.py`, `tests/test_labs_smoke.py`,
  `tests/test_notebooks_smoke.py`:
  - Published the v0.2.0 public learning surface: example scripts, overview
    notebooks, a VitePress docs-site skeleton, and headless smoke tests for
    examples/notebooks/labs, including a clean skip (not a failure) when
    `labs/` is absent.
- `README.md`, `CHANGELOG.md`, `docs-site/docs/guide/design-philosophy.md`,
  `notebooks/README.md`, `docs/curriculum_provenance.md` (renamed from
  `docs/labs_to_library_map.md`):
  - Replaced "withheld pending legal review" language with the plain
    statement: `labs/` is private by design, gitignored, never published;
    public cargo is the package, tests, the cases layer, and derivation
    notes.
- `pyproject.toml`, `src/qpl/market/data.py`, `src/qpl/market/stats.py`,
  `tests/oracle/*`, `tests/test_market_data.py`, `tests/test_market_stats.py`:
  - Split optional extras out of core: `dev` (unchanged), `data`
    (pandas/yfinance/pyarrow), `oracle` (QuantLib, an independent pricing
    oracle per D3, never sole evidence). Core dependencies shrink to
    numpy/scipy/matplotlib. Optional imports are guarded at call time with a
    clear `NotSupportedError` + install hint; `tests/oracle/` collects only
    when QuantLib is importable, via `pytest_ignore_collect` (a module-level
    `importorskip` would hard-fail when the directory is targeted directly).
- `.github/workflows/ci.yml`:
  - Split into `tests-core` (`pip install -e ".[dev]"`) and `tests-full`
    (`pip install -e ".[dev,data,oracle]"`); both run `ruff check .` then
    `pytest -q`.
- `src/qpl/validation/__init__.py`, `src/qpl/validation/benchmark.py`,
  `src/qpl/validation/convergence.py`, `tests/test_validation_convergence.py`:
  - Added `qpl.validation`: `fit_convergence_order` (OLS slope of `log(err)`
    on `log(h)`, reporting order + log-space RMS residual — chosen over
    R-squared, which saturates near 1 for any plausible power law) and the
    `EvidenceClass`/`BenchmarkRow` taxonomy (exact identity, closed form,
    published benchmark, independent engine, convergence order, statistical,
    negative finding).
- `src/qpl/engines/pde/pricers.py`, `tests/test_pde_pricing.py`,
  `tests/test_pde_ch4.py`:
  - Added `PDEConfig.strike_alignment` (`"none"` default, bit-identical to
    prior output; `"midpoint"` nudges grid spacing so the strike sits exactly
    between two nodes). Replaced a grid-lucky monotone-decay check with three
    tests that name their evidence class: a pinned negative finding for the
    unaligned pathology, a measured convergence-order test for the aligned
    grid, and one isolating temporal order.
  - **Measured** (S=K=100, r=5%, q=0, sigma=20%, T=1, European call,
    Crank-Nicolson, `n_s=n_t=n` for `n` in `{50,100,200,400,800}`): unaligned
    fitted order **1.126** (log-residual **0.860**, non-monotone — error at
    `n=100` is 5x worse than at `n=50` because the strike lands exactly on a
    node); aligned fitted order **1.997** (residual **0.022**). Isolated
    implicit-Euler temporal order (`n_s=800` aligned, fixed, refining `n_t`):
    fitted order **0.997** (residual **3e-04**), with the spatial error floor
    measured at 1.95e-04, 27x below the smallest temporal error in the fit.
- `src/qpl/cases/__init__.py`, `src/qpl/cases/european_black_scholes.py`,
  `tests/cases/test_european_black_scholes_cases.py`:
  - Added `qpl.cases.european_black_scholes`: 18 benchmark rows across parity
    (exact identity), degenerate limits (closed form), a reference ATM
    call/put value (closed form, in-repo, not published), and comparative
    statics (exact identity, ordering). A cross-engine test prices each
    known-value row via analytic, PDE, and MC with per-engine tolerances
    justified against measurements. Superseded five redundant assertions in
    `tests/test_black_scholes_analytic.py` and `tests/test_pricing_analytic.py`.
- `docs/notes/pde_strike_alignment.md`, `docs/curriculum_provenance.md`:
  - Added the public derivation note for strike alignment (own words,
    citations to Pooley/Forsyth/Vetzal 2003 and Tavella/Randall 2000; no book
    prose/code/tables copied) and recorded it as a Chapter 4 follow-up.
- `docs/CURRICULUM.md`, `docs/ROADMAP.md`, `.agents/brain/adr/0004-*.md`,
  `.agents/brain/brain.md`:
  - Added the curriculum map (TDD loop, identity decision, literature spine,
    method dependency graph, evidence classes, provenance rules, phase plan,
    Slice 0 results, reconciled old roadmap); retired `docs/ROADMAP.md` to a
    pointer; recorded ADR-0004; reconciled `brain.md` sections 0-3, 7, 9-10 to
    match current reality.

What got better
- Numerical claims now state which evidence class justifies them, instead of
  an unlabelled tolerance in a test.
- The strike-on-a-node PDE pathology is measured, documented, and fixed
  (opt-in via `PDEConfig.strike_alignment="midpoint"`), not just tested
  against grids that happened to avoid it.
- `labs/` privacy is now stated as an intentional design choice, not a
  pending-review placeholder.
- Core install no longer pulls in pandas/yfinance/pyarrow/QuantLib; CI
  exercises both the minimal and fully-equipped install paths.
- The public curriculum has an importable, tested cases layer
  (`qpl.cases`) rather than only ad hoc assertions scattered across test
  files.

Deferred (intentional)
- `engines.dp` (American put) is not yet wired into the `qpl.pricing`
  dispatcher; that is Phase 1, alongside the `isinstance`-ladder-to-registry
  refactor (ADR-0005, pending).
- Rannacher start-up and non-uniform strike-concentrated grids (the other two
  standard remedies for the PDE strike pathology) are Phase 2.
- Market-data/historical-vol/implied-vs-realized work is done but frozen
  behind the `[data]` extra, outside this curriculum (D9).
- Automatic API reference generation and docs-site CI deployment wiring.

How to validate quickly
- `ruff check .`
- `pytest -q`
- `QPL_LAB_SMOKE=1 MPLBACKEND=Agg pytest -q tests/test_labs_smoke.py`
- `PYTHONPATH=src MPLBACKEND=Agg QPL_LAB_SMOKE=1 pytest -q tests/test_notebooks_smoke.py`
- `pip install -e ".[dev,data,oracle]" && pytest -q` to exercise the full-extras path (`tests/oracle/` included).
