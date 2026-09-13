# American exercise on a binomial tree

Written in this repository's own words. Load-bearing results are cited;
nothing is copied from a source's text, tables, code, or example sequence. All
numbers below were measured by running this package and can be reproduced from
`tests/test_tree_american.py`, `tests/test_tree_american_convergence.py`,
`tests/cases/test_american_black_scholes_cases.py` and
`tests/oracle/test_american_vs_quantlib.py`.

## 1. Optimal stopping is backward induction

A European claim has one exercise date, so its value at a node is the
discounted expectation of its value one step later. An American claim adds a
decision at every node: take the intrinsic value now, or keep the option. The
value of holding is the discounted expectation as before, so the value of the
node is whichever is larger:

```
V(t_j, S) = max( g(S),  e^{-r dt} [ p V(t_{j+1}, S u) + (1-p) V(t_{j+1}, S d) ] )
```

with `g(S) = max(K - S, 0)` for a put and `max(S - K, 0)` for a call, and `p`
the forward-matching probability derived in
[`crr_tree_convergence.md`](crr_tree_convergence.md). At expiry there is no
continuation and `V(T, S) = g(S)`.

This is the Bellman equation of a finite-horizon optimal stopping problem, and
on a recombining lattice it is finite and exact: `n+1` terminal nodes rolled
back `n` times, one elementwise maximum per level. Nothing is approximated
except time itself.

Two consequences fall straight out and are worth stating because they are what
the tests check:

- **`V >= continuation` at every node**, so the American value dominates the
  European one on the *same* lattice at *every* `n`, coarse trees included.
  This is structural, not asymptotic.
- **`V >= g` at every node**, so the value never falls below intrinsic.

The implementation is `qpl.engines.tree.price_american`, reached through the
dispatcher because exercise style is a property of the instrument:

```python
price(AmericanOption(kind="put", strike=100.0, expiry=1.0),
      BlackScholesModel(sigma=0.2), market,
      method="tree", cfg=TreeConfig(n_steps=2000))
```

The analytic, Monte Carlo and PDE engines are registered for `EuropeanOption`
only, so handing them an `AmericanOption` raises `NotSupportedError` from the
ordinary registry lookup. `AmericanOption` is deliberately not a subclass of
`EuropeanOption`: the registry walks the MRO, and a subclass would silently
resolve to the European closed form and return a plausible wrong number.

## 2. Why a call on a non-dividend-paying stock is never exercised early

Merton (1973), *Theory of rational option pricing*, Bell Journal of Economics
and Management Science 4(1), 141-183, is the standard reference; Shreve,
*Stochastic Calculus for Finance II*, chapter 8, gives the continuous-time
treatment. Here is the argument as it appears on the lattice, which is the
version this package tests.

Fix a node at time `t` with spot `S`, and let `tau = T - t`. The CRR
probability is chosen so the tree reprices the forward exactly: the expected
terminal spot from that node is `S e^{(r-q) tau}`, with no discretisation
error at any `n`. So for a call, using `max(x, 0) >= x` inside the
expectation,

```
C_euro(S, tau) = e^{-r tau} E[ max(S_T - K, 0) ]
              >= e^{-r tau} ( S e^{(r-q) tau} - K )
               = S e^{-q tau} - K e^{-r tau}.
```

With `q = 0` the first term is just `S`, and with `r >= 0` the second is at
most `K`, so `C_euro >= S - K`. Together with `C_euro >= 0` this gives
`C_euro >= max(S - K, 0)`: the continuation value already dominates the
intrinsic value everywhere, and the Bellman maximum is a no-op at every node.
The American call equals the European call.

The economics behind the algebra: exercising early hands over `K` immediately
instead of at expiry, forfeiting the interest on it, and throws away the
remaining optionality. With no dividend there is nothing on the other side of
that trade.

It stops being true as soon as `q > 0`: the first term becomes `S e^{-q tau}`,
which falls below `S`, and holding now costs the dividend yield. Measured at
`S = K = 100, T = 1, r = 5%, q = 6%, sigma = 20%`, the American call exceeds
the European one by 1.93e-01 at `n = 50` and 1.81e-01 at `n = 201`.

Because the maximum is a *no-op* rather than a near-no-op, `max(a, b)` returns
`b` itself and not a rounded copy of it, so at `q = 0` the American and
European trees agree **bit for bit** -- and so do all five Greeks, since
bumping `sigma` or `r` leaves `q = 0` alone. The tests use `==`, not a
tolerance.

The put has no such argument, and that is the whole subject: early exercise of
a put is *paid for* by the interest earned on the strike received. At `r = 0`
that payment vanishes and the American put equals the European put exactly
(measured: difference exactly 0.0 at `n` in {51, 200, 501}, with `q = 3%`).

### An upper bound on the early-exercise premium

For `q = 0`, the same forward-repricing inequality applied to a put gives
`P_euro(S, tau) >= K e^{-r tau} - S`. Exercising at that node is worth
`K - S`, so the gain from exercising rather than waiting is at most
`K (1 - e^{-r tau}) <= K (1 - e^{-rT})`, and discounting it back to today only
shrinks it. Hence

```
0 <= P_amer - P_euro <= K (1 - e^{-rT}) .
```

The bound is attained, not merely approached: deep in the money the American
put is worth `K` and the European put `K e^{-rT} - S`. Measured at
`S = 10, K = 100, T = 1, r = 10%`: premium 9.516258 against a bound of
9.516258. Premium-to-bound ratios at the six points in
`qpl.cases.american_black_scholes`: 0.28, 0.11, 0.90, 0.86, 1.00, 0.05.

## 3. The `sigma = 0` limit, and where the obvious answer is wrong

With `sigma = 0` the spot is deterministic, `S(t) = S0 e^{(r-q)t}`, and the
American value collapses to a one-dimensional maximisation over the exercise
date:

```
V = max_{0 <= t <= T} e^{-rt} g(S(t)) = max( 0, sup_{0 <= t <= T} h(t) ),
h_put(t)  = K e^{-rt} - S0 e^{-qt},
h_call(t) = S0 e^{-qt} - K e^{-rt}.
```

`h` is a difference of two exponentials, so `h'(t) = 0` has at most one root:
`h` is either monotone on `[0, T]` or has exactly one interior turning point,
at

```
t* = log( r K / (q S0) ) / (r - q),      h_put''(t*) = (r - q) r K e^{-r t*}.
```

For a put, `r > q` makes `t*` a **minimum**, so the supremum is at an
endpoint and

```
V_put = max( intrinsic now,  discounted forward intrinsic ),
```

which is the European `sigma = 0` limit with an "exercise now" branch added.
Every no-dividend case is of this kind.

**But `q > r` makes `t*` a maximum**, and when it lands inside `(0, T)` the
best exercise date is neither today nor expiry. Measured at
`S0 = K = 100, r = 2%, q = 50%, T = 10`: `t* = 6.7060` and `h(t*) = 83.9506`,
against `max(h(0), h(T)) = 81.1993`. An endpoints-only rule would be 2.7513
too low -- 3.4% of the price, nothing like round-off. The engine takes the
maximum over `{0, T, t*}`, which is exhaustive because there is only ever one
turning point, and `tests/test_tree_american.py` pins the counterexample.

This is one of four places where this slice's own written expectation turned
out to be wrong; the note records what is true instead rather than the
expectation.

## 4. Extracting the exercise boundary

For a put the exercise region is `{S <= B(t)}`, so the boundary at a time
level is the **largest** node at which exercising is optimal; for a call it is
`{S >= B(t)}` and therefore the smallest. `price_american` reports one value
per time level in `meta["exercise_boundary"]`, `NaN` where no node exercises,
and the node count in `meta["early_exercise_node_count"]`.

Two details matter.

*A node counts as exercising only if it is in the money.* Far out of the
money, intrinsic and continuation are both zero (or underflow to it), and
`exercise >= continuation` there is a statement about `0 >= 0`. The engine
requires the intrinsic value to exceed `1e-12 * K` as well. Without that
floor, the node that should sit exactly on the strike -- computed as
`S0 u^k d^k`, which equals `K` only to round-off -- would count as in the
money for a call and not for a put, purely on the last bit of a power.

*The extracted boundary is not monotone level by level, and claiming it is
would be false.* Levels of the same parity share a node grid, `S0 u^{2i-j}`,
and consecutive levels use the two interleaved grids, so the reported boundary
alternates between them. What is true, and what the tests assert:

- each parity subsequence is monotone (non-decreasing for a put, non-increasing
  for a call), up to a round-off budget of 1e-9 -- a *flat* stretch of the
  boundary is the same economic level computed as `S0 u^i d^{j-i}` for
  different `(i, j)`, which agree only to about 3e-14;
- no level-to-level step falls by more than the offset between the two grids.
  Measured worst drop at `n = 200`: 1.3845, against `S (1 - 1/u) = 1.4043`.

At expiry the boundary is the in-the-money node adjacent to the strike. At
`S0 = K = 100, n = 200` the put's is `K / u^2 = 97.2112`, exactly one
terminal-level node spacing below `K`, and the call's is
`K u^2 = 102.8688`. That is "equals `K` up to node spacing", stated precisely.

Sample boundary for the put at `S = K = 100, T = 1, r = 5%, q = 1%,
sigma = 20%, n = 200` (from `examples/american_put_binomial_dp.py`):

| `t` | `B(t)` |
|---|---|
| 0.0000 - 0.0750 | NaN (no node low enough on this lattice) |
| 0.0800 | 79.7499 |
| 0.5000 | 82.0378 |
| 0.7500 | 84.3913 |
| 0.9900 | 97.2112 |
| 1.0000 | 97.2112 |

The `NaN` prefix is not a missing boundary: it is a boundary below the bottom
of the lattice at those early levels, which is exactly where the continuous
boundary is, far from expiry.

## 5. Measured convergence

Reference: this engine at `n = 8001`. There is no closed form to use instead,
and a same-engine reference is not free -- its own error is subtracted from
every measured error. That error is 1.80e-04 for the point below, obtained by
comparing against the average of `n = 64000` and `n = 64001`
(`6.090376463020103`), which brackets and so cancels the odd/even oscillation.

`S = K = 100, T = 1, r = 5%, q = 0, sigma = 20%`, American put, errors against
the `n = 8001` reference:

| `n` (odd) | error | `n·abs(err)` | `n` (even) | error | `n·abs(err)` |
|---|---|---|---|---|---|
| 25 | +5.563e-02 | 1.3907 | 26 | -3.162e-02 | 0.8222 |
| 51 | +2.740e-02 | 1.3973 | 50 | -1.683e-02 | 0.8414 |
| 101 | +1.401e-02 | 1.4151 | 100 | -8.202e-03 | 0.8202 |
| 201 | +6.965e-03 | 1.3999 | 200 | -4.174e-03 | 0.8347 |
| 401 | +3.432e-03 | 1.3763 | 400 | -2.151e-03 | 0.8602 |
| 801 | +1.640e-03 | 1.3134 | 800 | -1.155e-03 | 0.9238 |

Fitted orders (`h = 1/n`, least squares on `log err` against `log h`):

| point | odd order (log-RMS residual) | even order |
|---|---|---|
| ATM put, `r=5%`, `q=0` | 1.0141 (0.0174) | 0.9722 (0.0242) |
| ATM call, `r=5%`, `q=6%` | 1.0232 (0.0173) | 0.9674 (0.0213) |

Both inside the `[0.8, 1.2]` band this repository uses for order-1 claims.

**The odd/even bracketing survives early exercise.** Odd `n` is above the
reference at every level tested and even `n` is below it, at both points, for
the same reason as in the European case: at the money an even `n` places a
terminal node exactly on the strike and an odd `n` straddles it.

**The constants do not settle** the way the European tree's do (1.7529 and
1.9994 to four figures). Part of the drift above is the reference's own error
leaking in -- against the 64000/64001 reference the same fits read 0.9872
(residual 0.0034) and 1.0168 (0.0091), and the constants run 1.395 -> 1.458
and 0.818 -> 0.780 instead. Part of it is real: the American error has a
second source with no parity structure at all, namely that the exercise
boundary is resolved only to the node spacing.

Lattice Greeks converge at order ~1 too, against the same engine at
`n = 2001`:

| Greek | odd order (residual) | even order (residual) |
|---|---|---|
| delta | 1.1644 (0.0265) | 1.0743 (0.0043) |
| gamma | 1.1845 (0.0255) | 1.0787 (0.0104) |
| theta | 1.2586 (0.0478) | 1.0759 (0.0129) |

### Richardson extrapolation does not work here

For the European call, combining two same-parity levels,
`V_ext = (n2 V(n2) - n1 V(n1)) / (n2 - n1)`, lifts the measured order from
1.00 to 1.959. For the American put it does not:

| parity | fitted order | log-RMS residual | extrapolated errors |
|---|---|---|---|
| odd | 0.2960 | 0.2712 | 2.53e-04, 3.55e-04, 1.52e-04, 1.18e-04, 1.57e-04 |
| even | 0.6447 | 0.3562 | 8.00e-04, 4.24e-04, 1.45e-04, 1.27e-04, 1.59e-04 |

Both halves of the truth matter. It *does* help in magnitude -- a factor of
10 to 220 against the raw error at the same `n1` -- and it *does not*
converge: the extrapolated sequence flattens around 1.5e-04 and a log-log fit
of a flat sequence returns a slope near zero with a large residual, which is
what those numbers are.

The flat floor is not merely the reference's own accuracy. Repeating the
measurement against the 64000/64001 reference gives extrapolated errors
4.33e-04, 5.35e-04, 2.83e-05, 6.19e-05, 2.28e-05 (order 1.1613, residual
0.7097) -- non-monotone, and nothing like the clean order-2 sequence the
European call produces.

The reason is structural. Richardson assumes an asymptotic expansion in `1/n`
with fixed coefficients. Part of the American error comes from the exercise
boundary snapping to the nearest node, which moves in steps as `n` changes
which node is nearest, so there is no smooth leading coefficient to cancel.
No Richardson-extrapolated American value appears anywhere in this package's
public surface.

## 6. Cross-checks

### QuantLib finite differences (INDEPENDENT_ENGINE)

`FdBlackScholesVanillaEngine` with an `AmericanExercise` solves the same free
boundary problem by a different route. Neither engine has a closed form to
lean on, so the tolerance is derived from both refinements rather than picked.
At the ATM put point, against the 64000/64001 limit:

| QuantLib FD `g = tGrid = xGrid` | value | error | `qpl` tree `n` | value | error |
|---|---|---|---|---|---|
| 200 | 6.08701549 | -3.36e-03 | 2000 | 6.08998995 | -3.87e-04 |
| 400 | 6.08880946 | -1.57e-03 | 8001 | 6.09055641 | +1.80e-04 |
| 800 | 6.08961524 | -7.61e-04 | 16001 | 6.09046362 | +8.72e-05 |
| 1600 | 6.08999849 | -3.78e-04 | 32001 | 6.09041722 | +4.08e-05 |
| 3200 | 6.09018558 | -1.91e-04 | | | |

They approach from **opposite sides**, so the gap between them is the sum of
their errors, `1.80e-04 + 1.91e-04 = 3.71e-04`, not a cancellation. The
tolerance used is 1e-03, a factor of 2.7; below about 4e-04 the test would be
asserting a cancellation that is not there. Measured gaps at three points:
3.71e-04 (ATM put), 1.20e-04 (Longstaff-Schwartz row 1), 3.10e-04 (call with
`q = 6%`).

QuantLib's FD order, fitted on successive differences `|v(2g) - v(g)|` so that
no reference value is needed: **1.0856** with log-RMS residual 0.0211. First
rather than second order in a joint `tGrid`/`xGrid` refinement is expected for
an American payoff -- the free boundary is resolved only to the grid.

### QuantLib's CRR American (NEGATIVE_FINDING, preserved from Slice 1)

QuantLib's `CoxRossRubinstein` uses the log-space probability
`1/2 + (r - q - sigma^2/2) dt / (2 sigma sqrt(dt))`, not the forward-matching
`p = (e^{(r-q)dt} - d)/(u - d)`. Slice 1 identified this for European payoffs
and reproduced QuantLib's engine to 8.3e-11 with a twelve-line reimplementation
of its probability. Early exercise leaves it untouched -- the probability
enters the continuation value, and the Bellman maximum is applied afterwards
-- so the two engines still differ at `O(1/n)`. Measured
`n · (qpl - QuantLib)`:

| point | `n = 200` | `n = 800` | `n = 3200` |
|---|---|---|---|
| ATM put | -0.026367 | -0.026404 | -0.026413 |
| L&S row 1 | -0.013281 | -0.013237 | -0.013260 |
| call, `q = 6%` | +0.007084 | +0.007099 | +0.007104 |

Constant to better than 0.5% over a sixteen-fold refinement. Neither engine is
"the" CRR American answer; both are order-1 approximations of the same number.

## 7. Longstaff and Schwartz (2001) Table 1, row 1

Longstaff & Schwartz (2001), *Valuing American options by simulation: a simple
least-squares approach*, Review of Financial Studies 14(1), 113-147, Table 1,
first row: `S = 36, sigma = 0.20, T = 1`, with `K = 40, r = 0.06`, no
dividends. The reported finite-difference value is 4.478 (their own
least-squares Monte Carlo estimate is 4.472).

This slice was specified to check that the tree's American put at `n = 5000`
lands within 2e-03 of 4.478. **It does not.** The measured value is 4.486710,
which is 8.71e-03 away, and refining does not close the gap:

| engine | setting | value |
|---|---|---|
| `qpl` tree | `n = 5000` | 4.486710 |
| `qpl` tree | `n = 32001` | 4.486680 |
| QuantLib binomial American (`crr`) | `n = 5000` | 4.486712 |
| QuantLib FD | `tGrid = xGrid = 3200` | 4.486536 (still rising) |

Three independent routes land near 4.4867; none lands on 4.478.

The explanation is the instrument, not the engine. The options in that table
are exercisable 50 times per year, not continuously. Restricting exercise to
50 equally spaced dates on the *same* CRR lattice reproduces the published
figure:

| `n` | Bermudan(50) value | gap to 4.478 |
|---|---|---|
| 5000 | 4.477922 | 7.8e-05 |
| 10000 | 4.477870 | 1.30e-04 |
| 20000 | 4.477834 | 1.66e-04 |
| 40000 | 4.477826 | 1.74e-04 |

So both numbers are right and they price different things. The gap between
them, 8.7e-03, is the value of continuous rather than fortnightly exercise at
this specification.

`qpl.cases.american_black_scholes` carries this as two rows on one
specification: `ls2001_table1_row1_bermudan_50`
(`PUBLISHED_BENCHMARK`, tolerance 5e-04, which is the published figure's own
three-decimal quoting precision rather than anything about the engine) and
`ls2001_table1_row1_is_not_a_continuous_american_value` (`NEGATIVE_FINDING`,
asserting that the distance *exceeds* 2e-03 and pinning it at
8.71e-03 ± 5e-05). The 50-date restriction is twenty lines of backward
induction in the test, not a new instrument: `qpl` has no Bermudan type, and
adding one to check a citation would be building an instrument ahead of the
case that justifies it.

## Summary of what contradicted the written expectation

1. The Longstaff-Schwartz 4.478 is a 50-exercise-date Bermudan value, so the
   2e-03 tolerance for a continuous American put is unachievable; the
   continuous value is 4.48668 and the Bermudan reproduction is 4.477922.
2. At `sigma = 0` the American put is *not* always
   `max(intrinsic now, discounted forward intrinsic)`; that holds only for
   `r >= q`.
3. The extracted exercise boundary is *not* monotone level by level; it is
   monotone within each node-grid parity, and zigzags by at most one grid
   offset between them.
4. Richardson extrapolation, which lifts the European tree to measured order
   1.96, does not restore order 2 for the American put -- it reduces the error
   and then stalls.

## References

- Cox, Ross and Rubinstein (1979), "Option pricing: a simplified approach",
  Journal of Financial Economics 7, 229-263 -- the lattice.
- Merton (1973), "Theory of rational option pricing", Bell Journal of
  Economics and Management Science 4(1), 141-183 -- the no-early-exercise
  result for calls on non-dividend-paying stock.
- Shreve, *Stochastic Calculus for Finance II*, ch. 8 -- American derivative
  securities and optimal stopping, in continuous time.
- Leisen and Reimer (1996), "Binomial models for option valuation - examining
  and improving convergence", Applied Mathematical Finance 3(4), 319-346 --
  the order-1 behaviour and odd/even oscillation of the CRR tree (a later
  slice implements their construction).
- Longstaff and Schwartz (2001), "Valuing American options by simulation: a
  simple least-squares approach", Review of Financial Studies 14(1), 113-147
  -- Table 1 row 1, used as a cited fixture.
