# Barrier options: one displacement, measured from three sides

A barrier option's payoff depends on whether the underlying ever reached a
level `H`. "Ever" is the whole difficulty. Every numerical method has to decide
*when* it looks, and each of them ends up pricing a slightly different barrier
than the one in the contract:

- a **discretely monitored** contract only looks on its observation dates, so
  the path can dip past `H` and return unseen;
- a **Monte Carlo** simulation of the continuous contract looks only where it
  sampled, which is the same failure with a different name;
- a **lattice** looks at every time level but can only kill a node, so it kills
  at the first node *beyond* the barrier.

In all three the barrier is effectively displaced away from the spot by
`O(sigma sqrt(dt))`, and the price is locally linear in the barrier level. Half
an order, therefore, not one — and that single sentence explains the
measurements in this note. The exception is the Brownian-bridge estimator,
which does not displace the barrier at all because it never has to decide
whether a path crossed: it computes the probability that it did.

Sources. Merton, R.C. (1973), "Theory of rational option pricing", *Bell
Journal of Economics and Management Science* 4(1), 141–183, §8: the
down-and-out call. Reiner, E. and Rubinstein, M. (1991), "Breaking down the
barriers", *Risk* 4(8), 28–35: all eight single-barrier types with rebates.
Haug, E.G. (2007), *The Complete Guide to Option Pricing Formulas*, 2nd ed.,
§4.17: the standard tabulation, three rows of which are used as fixtures with
that citation. Shreve, S.E., *Stochastic Calculus for Finance II*, ch. 7: the
joint law of the terminal value and the running extremum. Broadie, M.,
Glasserman, P. and Kou, S. (1997), "A continuity correction for discrete
barrier options", *Mathematical Finance* 7(4), 325–348. Glasserman, P. (2003),
*Monte Carlo Methods in Financial Engineering*, §6.4 and §3.1. Boyle, P.P. and
Lau, S.H. (1994), "Bumping up against the barrier with the binomial method",
*Journal of Derivatives* 1(4), 6–14. Derman, E., Kani, I., Ergener, D. and
Bardhan, I. (1995), "Enhanced numerical methods for options with barriers",
*Risk* 8(6) — cited, and deliberately *not* implemented; §6 records what that
cost. Every derivation below is written out independently and every number was
measured in this repository.

## The reference specification

Unless a table says otherwise: a **down-and-out call**, `S = K = 100`,
`H = 95`, `T = 0.5`, `r = 8%`, `q = 4%`, `sigma = 25%`, zero rebate. The
continuous closed form is **4.512598607823680**. The barrier is close enough
that the knock-out probability is about 0.75, so everything below is a
first-order effect and not a rounding difference.

## 1. The closed form, in one paragraph

Write `S_t = S exp(sigma X_t)` with `X_t = nu t + W_t` and
`nu = (r - q - sigma²/2)/sigma`. A down barrier at `H` is the level
`a = log(H/S)/sigma < 0` for `X`, so the price needs the joint law of `X_T` and
`m_T = min_t X_t`. For a driftless Brownian motion the reflection principle
gives it directly: a path ending at `x` that has touched `a` is in bijection
with a path ending at `2a - x`, so `P(m_T ≤ a, W_T ≥ x) = P(W_T ≥ 2a - x)`.
Under a drift, Girsanov multiplies the reflected density by exactly
`exp(2 nu a)` — the two Radon–Nikodym factors `exp(nu X_T - nu² T/2)` differ
only through `X_T → 2a - X_T`. Every single-barrier value is then an integral
of a vanilla payoff against that law, and every such integral is a difference
of two normal tails: one from the ordinary terminal density, one from the
reflected term carrying `(H/S)^{2mu}` or `(H/S)^{2(mu+1)}`.

Collecting them gives the six blocks `A`–`F` written out in
`qpl.engines.analytic.barrier`, and the eight types are sums of them. Two
structural facts are worth naming because they are what the tests check and
what a sign error breaks:

- **`F → K` as `H → S`**, and every knock-out assembly collapses to `F` alone
  there — exactly, with no limit taken, measured at 1.42e-14 over three points
  and four types. The route differs by kind: for a call with a down barrier
  `A = C` and `B = D` block by block; for a put `A + C = B + D` instead and the
  assembly `A - B + C - D` vanishes as a difference of *sums*. A test that only
  checked the call would pass with a sign error in the put's `C`.
- **In–out parity is exact at zero rebate**: `knock-in + knock-out = vanilla`,
  measured at 1.42e-14 absolute (1.81e-15 relative) over twenty cells. With a
  rebate it is *false* by exactly `E + F`, because the two rebate legs are paid
  on complementary events at different times — `E` at expiry if the barrier was
  never touched, `F` at the touch time if it was.

Three published values reproduce (four-decimal figures, so agreement is
asserted to 1e-4):

| contract | published | in-repo | residual |
|---|---|---|---|
| down-and-out call, `K = 90` | 9.0246 | 9.024567694966874 | 3.23e-05 |
| down-and-out call, `K = 100` | 6.7924 | 6.792436575025215 | 3.66e-05 |
| up-and-out call, `K = 100`, `H = 105` | 2.3580 | 2.358019790844047 | 1.98e-05 |

all at `S = 100`, rebate 3, `r = 8%`, `q = 4%`, `sigma = 25%`, `T = 0.5`.
QuantLib's `AnalyticBarrierEngine` agrees to **2.886e-14** over 96 cells (eight
types × two kinds × three strikes × two rebates) at `T = 1.0`.

## 2. The monitoring bias, and its order

A knock-out monitored on `m` equally spaced dates is worth **more** than the
continuous one. Measured at 40 000 paths with antithetic sampling and the
vanilla control variate, seed 20 260 913:

| `m` | plain MC | bias vs closed form | stderr | bridge `z` | BGK − bridge (paired) |
|---:|---:|---:|---:|---:|---:|
| 25 | 5.633652 | +1.121054 | 0.027990 | +0.136 | +0.0346 ± 0.0109 |
| 50 | 5.328995 | +0.816397 | 0.029915 | +0.129 | +0.0028 ± 0.0098 |
| 100 | 5.134664 | +0.622065 | 0.030737 | −0.257 | −0.0010 ± 0.0083 |
| 200 | 4.961593 | +0.448995 | 0.031441 | +0.715 | −0.0036 ± 0.0067 |
| 400 | 4.775655 | +0.263057 | 0.032049 | −1.425 | −0.0041 ± 0.0063 |

**Fitted bias order 0.5045** (log-space residual 0.0687). At `m = 25` the bias
is 25% of the option's value; the plain estimator is not slightly off, it is
pricing a different contract.

The plain estimator is *unbiased for the contract it prices*. There is no
time-discretisation error in it at all — the monitored spots are exact
lognormal draws at the monitoring dates — and QuantLib's
`MCBarrierEngine(isBiased=True)`, which checks the barrier on its own grid,
agrees at `z = −0.202` (`m = 25`) and `−0.879` (`m = 50`) on the combined
standard error while both sit 1.4 to 2.0 away from the continuous price. Two
engines agreeing on 7.10 when the continuous answer is 5.08 is evidence that
they price the same *contract*, which is the only thing an MC-versus-MC
comparison can establish.

### Is the order really 1/2?

Over ten seeds the fit against the closed form gives **0.4764 ± 0.0329**
(min 0.4350, max 0.5288). A much lower-noise estimator is available: the plain
and bridge payoffs can be differenced *path by path*, since the three
corrections read the same sample. That paired fit gives **0.4603 ± 0.0125** over
the same ten seeds — significantly below 0.5, and not by noise. Fitting
`a/sqrt(m) + b/m` to the measured biases returns `b < 0`, which is exactly what
drags a finite-window fit under one half. So: the asymptotic order is 1/2, the
*measured* order over `m ∈ [25, 400]` is 0.46, and the gap is the
`o(1/sqrt(m))` term the Broadie–Glasserman–Kou expansion leaves behind.

### The two corrections

**BGK barrier shift.** BGK show that a discretely monitored barrier at `H`
behaves like a continuous one at `H exp(∓ beta sigma sqrt(dt))`,
`beta = -zeta(1/2)/sqrt(2 pi) = 0.5825971579390107`, with error `o(1/sqrt(m))`.
The constant is the asymptotic expected overshoot of a random walk past a
level, not a fit. Read forwards it turns the closed form into an approximation
of the discrete price; read backwards it says a simulation with the barrier
moved **toward the spot** estimates the continuous price. Both directions are
implemented (`bgk_continuity_corrected_price`, and
`MCConfig(barrier_correction="bgk")`), and the direction is a named argument
rather than a sign the caller supplies, because getting it wrong doubles the
bias instead of removing it.

**The Brownian bridge.** Between two sampled points the log-spot is a Brownian
bridge, whose minimum has an explicit law; the probability that it reached `H`
is

```
p_i = exp( -2 log(S_{t_i}/H) log(S_{t_{i+1}}/H) / (sigma² dt_i) )
```

(and 1 if the endpoints straddle `H`). The estimator weights each path by
`prod_i (1 - p_i)` instead of killing it. That is
`E[payoff · 1{no touch} | sampled points]`, so it is **unbiased for the
continuous contract at any number of sampling dates**.

## 3. Three things the Monte Carlo study contradicted

**(a) The bridge prices the continuous barrier at `m = 1`.** This was not
expected and it is the cleanest result in the slice. With one observation, at
expiry, the bridged estimate is 4.5199 against 4.5126, `z = +0.92` at 20 000
paths and `z = -0.10` at 200 000; the plain estimator at the same `m` returns
**7.849428**, the vanilla, because a single terminal observation can barely
knock anything out. Conditioning on more points reduces the estimator's
variance and cannot change its mean, and the measurement says so. Coverage over
40 seeds at 20 000 paths: 39/40 at `m = 25` and 38/40 at `m = 100`, against a
Binomial(40, 0.95) mean of 38, with mean `z` of −0.13 and −0.09.

**(b) BGK's *order* is not measurable at this cost.** Paired against the bridge
on the same paths, the residual is resolved only at `m = 25` (3.2 standard
errors, reproduced across ten seeds as +0.0008 to +0.0377) and lies inside two
standard errors from `m = 50` on, with a wandering sign. A fit through that
returns 0.578 with a log-space residual of **1.02** — an order of magnitude
worse than the plain fit's 0.069, which is the diagnostic saying the fit is
noise. What *is* measurable is the size of the drop: from `m = 25` to `m = 50`
the plain bias falls by 1.37× and the BGK residual by at least 12×. Resolving
the order would need roughly 16× the paths at `m = 400` — 256 million normals —
to halve an error bar already smaller than the quantity it measures. This is the
same shape of finding as Slice 9's: a coupled estimator's noise floor decides
what is measurable, not the theory.

**(c) At `sigma = 0` the discrete knock-out is worth *less*.** With `r > q` the
deterministic forward crosses an up barrier at `t* = log(H/S)/(r-q)`; a
contract observed at 0.25 and 0.5 only notices at 0.25. Both knock out, both
pay the same rebate, and the discrete one pays it **later** — 2.94060 against
2.97022, exactly `3 e^{-r/4}` against `3 e^{-r t*}`. The usual reading
("discrete monitoring favours the knock-out holder") is about *option* value; a
later knock-out preserves more option value and less rebate value, and at
`sigma = 0` there is no option value left to preserve.

## 4. The lattice, and the Boyle–Lau sawtooth

On a CRR lattice the node spots form one geometric ladder `S u^i`,
`u = exp(sigma sqrt(dt))`. The tree knocks out at the first node beyond the
barrier, so its effective barrier is

```
H_eff = S exp( -ceil(lambda) sigma sqrt(dt) ) ≤ H,
lambda = log(S/H) / (sigma sqrt(dt)) = lambda_0 sqrt(n).
```

`lambda` grows like `sqrt(n)`, so its fractional part *cycles* as `n`
increments — and the error cycles with it. The period is `2 sqrt(n)/lambda_0`
steps, which is why amplitudes below are measured over a full period at each
level: a fixed-width window would report decay that is only the period
stretching.

| `n0` | period | min err | max err | CRR amplitude | LR amplitude | LR/CRR |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 70 | +0.01314 | +0.94823 | 0.93510 | 0.91326 | 0.977 |
| 200 | 99 | +0.00494 | +0.64461 | 0.63967 | 0.62561 | 0.978 |
| 400 | 139 | +0.00336 | +0.49990 | 0.49654 | 0.48657 | 0.980 |

**Amplitude order 0.4566** (CRR, residual 0.030) and **0.4542** (Leisen–Reimer).
At `n0 = 100` the error swings over 21% of the option's value as `n` walks
through 70 consecutive step counts, and the best `n` in each window is three
decimal orders better than the worst.

### Leisen–Reimer does not help

It buys **2%** of the amplitude, and none of the order. The reason is
structural: the Peizer–Pratt inversion arranges the terminal grid around the
*strike*, the level a vanilla payoff bends at. A barrier is a second level the
construction never looks at. Worse, `u d ≠ 1` on an LR lattice, so its node
spots are not a single geometric ladder and **no** step count puts a layer on
the barrier by construction. This is the reverse of Slice 3 (order 2 on a
vanilla) and Slice 6 (order 2 on a digital, where the lattice price *is* the
binomial tail the construction matches); here there is nothing for the
construction to match.

### Choosing `n`: the Boyle–Lau subsequence

Setting `lambda_0 sqrt(n) = k` and rounding down gives
`n_k = floor(k² sigma² T / log(S/H)²)`, which puts layer `k` within a hair of
the barrier.

| `k` | `n_k` | error | `error × n` | `1 - H_eff/H` |
|---:|---:|---:|---:|---:|
| 4 | 190 | +1.381e-04 | +0.026 | 5.62e-06 |
| 7 | 582 | −6.380e-05 | −0.037 | 1.14e-07 |
| 10 | 1187 | +1.432e-03 | +1.700 | 1.64e-05 |
| 13 | 2007 | +4.919e-04 | +0.987 | 4.03e-06 |
| 16 | 3040 | +3.372e-04 | +1.025 | 5.62e-06 |

against **+0.64** for the unaligned `n = 200`. A block-RMS fit over three groups
of thirteen layers gives order **1.1258** (residual 0.0041) — first order, as
it should be.

**It is not monotone**, which the slice expected. The signs alone disprove it
(negative at `k = 7` and `k = 14`), and the reason is that `floor` leaves a
residual misalignment: `1 - H_eff/H` spans 1.14e-07 to 8.14e-05 across the
thirteen layers and behaves like a uniform draw on the fractional part. Since
that residual is itself `O(1/n)` with a coefficient in `[0, 1)`, the error is
`O(1/n)` with an **erratic constant**: `|error × n|` stays inside
`[0.025, 1.70]` across all thirteen levels, which is the assumption-free form
of the same statement. A per-`n` fit returns 0.5716 with a log-space residual
of 1.49 — it is not a power law at fixed `n` at all. `error / (1 - H_eff/H)`
spans a factor of 7 while the errors themselves span a factor of 460, which is
the mechanism accounting for the scatter.

The Boyle–Lau counts also do **nothing** for Leisen–Reimer (block order 0.4254,
errors three decimal orders worse than CRR's at the identical `n`). "Choose `n`
to align the barrier" is a statement about a lattice *geometry*, not about a
step count, and carrying the count across schemes carries nothing.

## 5. Three engines on one number

`qpl.cases.barrier_black_scholes` prices three points by the closed form, by a
Boyle–Lau lattice, and by a Brownian-bridge Monte Carlo on 50 monitoring dates
— which is only a comparison at all because the bridged estimator is unbiased
for the *continuous* contract the other two compute.

| point | lattice error | MC \|z\| |
|---|---:|---:|
| `atm_6m_down_out` (`n = 2328`) | −1.075e-05 | 0.008 |
| `atm_1y_up_out` (`n = 2436`) | +1.730e-04 | 0.055 |
| `otm_1y_down_in` (`n = 2075`) | +1.055e-04 | 0.440 |

The lattice budget is `8.0 / n` and not the error at the chosen layer, because
the Boyle–Lau constant is erratic: reading a tolerance off one layer would be
cherry-picking. 8.0 is the measured envelope of `|error| × n` across the three
points (worst 5.54).

## 6. What QuantLib's lattice does that this one does not

`BinomialCRRBarrierEngine` shows **no sawtooth**. At `T = 1.0` (the oracle
maturity, where `Actual365Fixed` over 365 days is exact in double precision and
no day-count residual confounds the comparison), over `n = 200 … 239`:

| engine | min err | max err | amplitude |
|---|---:|---:|---:|
| `qpl` (knock out at nodes) | +9.41e-03 | +1.239 | 1.229 |
| QuantLib `BinomialCRRBarrier` | +1.09e-05 | +1.99e-03 | 1.98e-03 |

a factor of **620**. It also beats this package's Boyle–Lau-*aligned* prices at
the same step counts (−7.33e-06 against +6.10e-04 at `n = 2375`) and is stable
across four layers where this one varies by a factor of 42.

That is the behaviour of the interpolation remedy — adjust the value at the
nodes straddling the barrier rather than knocking out at the first node beyond
it — whose standard reference is Derman, Kani, Ergener and Bardhan (1995). The
attribution is inferred from the measurement here, not read from QuantLib's
source; only the behaviour is asserted. Slice 12 cited that remedy and
implemented the other one so that one could be measured properly rather than
two badly, and this is the record of what that choice cost. It is the strongest
argument for the later slice that closes the gap.

## 7. Greeks: where the singularity is not

Analytic barrier Greeks are central differences of the closed form (steps
`spot 1e-3`, `sigma 1e-4`, `r 1e-4`, `T 1e-4`), checked against the vanilla
engine's closed-form Greeks in the `H → 0` limit to 1.95e-10 (delta), 1.44e-06
(gamma), 2.01e-08 (vega), 4.29e-09 (theta) and 2.29e-08 (rho), relative.

The slice said to document the blow-up of gamma as `S → H`. **There is none.**

| `S` | 110 | 105 | 100 | 96 | 95.5 | 95.1 | 95.001 |
|---|---:|---:|---:|---:|---:|---:|---:|
| value | 13.2972 | 8.9034 | 4.5126 | 0.9206 | 0.4618 | 0.0926 | 0.00093 |
| delta | 0.8843 | 0.8754 | 0.8850 | 0.9150 | 0.9205 | 0.9253 | 0.9265 |
| gamma | +0.0029 | +0.0003 | −0.0046 | −0.0107 | −0.0116 | −0.0123 | −0.0125 |

The value vanishes **linearly** in `S - H` and both derivatives converge to
finite limits. What the barrier changes is gamma's **sign**: the same vanilla at
`S = 100` has gamma +0.02168, so a knock-out call is *concave* where the vanilla
is convex, because the value is pinned toward the rebate from one side rather
than bending away from the strike.

The real singularity is at the **corner** `(S = H, t = T)`, reached by letting
`T → 0` near the barrier — and only when the terminal payoff is discontinuous
across `H`, i.e. `vanilla(H) ≠ rebate`. With `K = 90 < H = 95` (payoff jumps by
5) at `S = 96`:

| `T` | 0.5 | 0.1 | 0.02 | 0.005 | 0.001 |
|---|---:|---:|---:|---:|---:|
| delta | 1.376 | 1.586 | 2.150 | 2.980 | 3.185 |
| gamma | −0.018 | −0.029 | −0.123 | −0.721 | −3.842 |

With `K = 100 > H` the payoff is continuous across the barrier (zero on both
sides) and there is no growth at all on the same ladder. That second half is
what says *why*.

MC and tree Greeks are refused, each naming its obstruction: a barrier payoff's
pathwise derivative is a surface measure on the barrier rather than a function
(the digital's obstruction, one dimension up), and the lattice estimators
divide by node spacings, so they would report the sawtooth amplified by
`1/sqrt(dt)` in delta and `1/dt` in gamma under a Greek's name.

## 8. What is deliberately not here

- **The barrier PDE**, and with it the non-uniform grid Phase 2 has been
  deferring since Slice 4. This is the case that forces it: the mesh has to
  place a node on a boundary that is *not* the strike, and where the solution
  is cut off rather than merely kinked. Next slice.
- **The Derman–Kani–Ergener–Bardhan interpolation** (§6), which the oracle shows
  is worth two decimal orders over choosing `n`.
- **Double barriers and lookbacks**, which need the reflection argument iterated
  rather than applied once.
- **MC and tree Greeks for barriers** (§7).
- **Discrete monitoring on a lattice**, which needs `n` to be a multiple of `m`
  *and* barrier-aligned — two conditions on one integer.
