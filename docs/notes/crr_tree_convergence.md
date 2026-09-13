# The CRR binomial tree: where it comes from, and why its error oscillates

## One period, no probability required

Take one period of length `dt`. The spot moves from `S` to either `S u` or
`S d`, and a claim on it pays `V_u` or `V_d`. Hold `Δ` shares, whose dividend
yield `q` is reinvested so the position grows by `e^{q dt}`, and `B` in the
money market, which grows by `e^{r dt}`. To reproduce the claim in both states:

```
Δ S u e^{q dt} + B e^{r dt} = V_u
Δ S d e^{q dt} + B e^{r dt} = V_d
```

Two equations, two unknowns, and `u ≠ d`:

```
Δ = (V_u − V_d) / (S (u − d) e^{q dt})
B = e^{−r dt} (V_u − Δ S u e^{q dt})
```

The claim is worth what the portfolio costs, `Δ S + B`. Substituting and
collecting terms turns that into

```
V = e^{−r dt} [ p V_u + (1 − p) V_d ],    p = (e^{(r−q) dt} − d) / (u − d)
```

so `p` is not an assumption about anybody's beliefs: it is what falls out of
the hedging equations. It is a probability exactly when `d ≤ e^{(r−q) dt} ≤ u`,
which is the no-arbitrage condition, and `qpl.engines.tree.lattice` raises
when it fails rather than returning a signed `p`.

The definition of `p` also says that the one-step expected spot is
`p S u + (1 − p) S d = S e^{(r−q) dt}`, exactly. The tree reprices the forward
with **no** discretisation error at any `n`, which is why put-call parity holds
on a two-step tree just as well as on a two-thousand-step tree.

## Why `u = e^{σ√dt}`, `d = 1/u`

Cox, Ross and Rubinstein (1979) pick the symmetric choice
`u = exp(σ √dt)`, `d = 1/u`. Two things follow.

*It recombines.* `u d = 1`, so an up-then-down move returns to the starting
spot and the level-`j` node count is `j + 1` rather than `2^j`. Backward
induction costs `O(n²)` arithmetic instead of `O(2^n)`.

*It matches the first two moments of the log-return.* One step moves
`log S` by `±σ√dt`. Expanding `p` in `x = σ√dt`,

```
p = 1/2 + (r − q) dt / (2x) − x/4 + O(x³)
```

so

```
E[Δ log S] = x (2p − 1) = (r − q − σ²/2) dt + O(dt²)
Var[Δ log S] = x² (1 − (2p − 1)²) = σ² dt + O(dt²)
```

These are the exact Black-Scholes log-drift and log-variance, with relative
error `O(dt)`. Over `n` steps the errors accumulate linearly, giving `O(dt)`
= `O(1/n)` overall — first order, and no better. That is the whole story of
the *rate*. The *constant* is a separate question, and it is where the
oscillation lives.

## Where the oscillation comes from

The tree does two things at once: it approximates a lognormal by a binomial,
and it evaluates a non-smooth payoff on a finite set of points. The first is
a central-limit-theorem statement and is smooth in `n`. The second is not.

The terminal nodes sit at `S u^{2j−n}`, `j = 0 … n` — a geometric grid whose
*position*, not just its spacing, changes with `n`. The call payoff has a kink
at `K`, and the tree's error picks up a term that depends on where `K` falls
between the two nodes straddling it. That fractional position is a function of
`n`, so the error constant is a function of `n` too.

At the money the dependence collapses to the parity of `n`, and can be read
off directly. With `d = 1/u`, level `n` contains the node `S u^{2j−n}`:

- `n` **even**: `j = n/2` gives `u^0 = 1`, so a terminal node sits *exactly on
  the strike*. The node carries the full one-sided discrepancy of the kink.
- `n` **odd**: `2j − n` is always odd, so the nearest nodes are `S u^{±1}` and
  the strike sits *exactly midway* between them in log-spot.

Two regimes, two constants, and — as measured below — opposite signs: the odd
sequence approaches Black-Scholes from above, the even sequence from below.
The true value is bracketed at every refinement level. This is the oscillation
identified in Leisen and Reimer (1996); the measurement below is this
repository's own.

Away from the money the parity story does not survive: `log(K/S)` is generally
not a multiple of `σ√dt`, the fractional position wanders with `n` instead of
alternating, and the constant becomes erratic rather than two-valued. That is
a property of the scheme, not of this implementation —
`tests/oracle/test_tree_vs_quantlib.py` shows QuantLib's binomial engine doing
exactly the same thing at the same points, and pins it as a negative finding.

## Measured

Case: `S = K = 100`, `r = 5%`, `q = 0`, `σ = 20%`, `T = 1`, European call.
Black-Scholes value `10.4505835722`. Error is `tree − closed form`, signed.
The put gives identical signed errors, because parity holds on the tree to
round-off and the analytic call and put satisfy it exactly.

| parity | n | tree | signed error | n · abs(error) |
|---|---:|---:|---:|---:|
| odd | 25 | 10.5209656240 | +7.038e-02 | 1.7596 |
| odd | 51 | 10.4850184901 | +3.443e-02 | 1.7562 |
| odd | 101 | 10.4679546748 | +1.737e-02 | 1.7545 |
| odd | 201 | 10.4593079285 | +8.724e-03 | 1.7536 |
| odd | 401 | 10.4549555011 | +4.372e-03 | 1.7531 |
| odd | 801 | 10.4527719805 | +2.188e-03 | 1.7529 |
| even | 26 | 10.3740581787 | −7.653e-02 | 1.9897 |
| even | 50 | 10.4106915407 | −3.989e-02 | 1.9946 |
| even | 100 | 10.4306116622 | −1.997e-02 | 1.9972 |
| even | 200 | 10.4405912599 | −9.992e-03 | 1.9985 |
| even | 400 | 10.4455858413 | −4.998e-03 | 1.9991 |
| even | 800 | 10.4480843149 | −2.499e-03 | 1.9994 |

Least-squares slope of `log(error)` on `log(1/n)`:

| subsequence | fitted order | log-space RMS residual |
|---|---:|---:|
| odd n | 1.0010 | 0.0005 |
| even n | 0.9987 | 0.0007 |

Both are order 1 to three decimal places, and the residuals say these are
essentially exact power laws — which they could not be if odd and even were
fitted together, since the scaled constants converge to two different limits,
about **1.7529** (odd) and **1.9994** (even).

### Richardson extrapolation

Within one parity the constant no longer oscillates, so the leading term can
be cancelled between two levels of the same parity. From `V(n₁)` and `V(n₂)`,

```
V_ext = (n₂ V(n₂) − n₁ V(n₁)) / (n₂ − n₁)
```

Applied to consecutive odd pairs:

| pair | extrapolated | error |
|---|---:|---:|
| (25, 51) | 10.4504539382 | 1.296e-04 |
| (51, 101) | 10.4505495833 | 3.399e-05 |
| (101, 201) | 10.4505747147 | 8.857e-06 |
| (201, 401) | 10.4505813116 | 2.261e-06 |
| (401, 801) | 10.4505830012 | 5.710e-07 |

Fitted order **1.9590**, log-space residual 0.0224. Close to 2, but not 2, and
the shortfall is real rather than noise: the pairs are `(n, 2n+1)` rather than
`(n, 2n)`, and the second-order coefficient within a parity is itself only
asymptotically constant, so a clean `C/n + D/n²` expansion is not quite what
the data obey. The gain is nonetheless large — the error at `n₁ = 401` drops
from 4.372e-03 to 5.710e-07, a factor of 7700.

### Greeks

Delta, gamma and theta are read from lattice nodes at steps 1 and 2, so they
inherit the price's order rather than improving on it. Measured on the same
case: order 1.004 / 1.002 / 1.001 on odd `n` and 0.999 / 1.010 / 1.010 on
even `n`. Vega and rho are bump-and-revalue; bumping `σ` moves `u` and `d`, so
the oscillating price error does not cancel between the two evaluations and
vega is two to three orders of magnitude coarser relative to its own size than
delta is. Bumping `r` leaves the lattice geometry alone, and rho is much
better behaved.

## What Leisen-Reimer will fix

Leisen and Reimer (1996) keep the recombining binomial structure but choose
`u`, `d` and `p` from an inversion of a normal approximation to the binomial
(Peizer-Pratt), with the tree built so that the strike sits centrally in the
terminal distribution by construction for odd `n`. That removes the term that
was oscillating and raises the measured order to 2. It is a later slice; this
note stops at CRR deliberately, because the oscillation is worth measuring
before it is engineered away.

## Reproducing

```bash
PYTHONPATH=src python examples/tree_convergence.py
```

Pinned by `tests/test_tree_convergence.py` (replication identities, parity,
the two order fits, the sign structure, Richardson),
`tests/test_tree_pricing.py` (lattice construction, degenerate limits, Greeks,
and the bit-for-bit invariance of the American-put DP engine under the shared
lattice) and `tests/oracle/test_tree_vs_quantlib.py` (QuantLib, and why its
`CoxRossRubinstein` is a different probability on the same lattice). The
fitting is `qpl.validation.fit_convergence_order`.

## Sources

- Cox, J., Ross, S. and Rubinstein, M. (1979). "Option pricing: a simplified
  approach". *Journal of Financial Economics* 7, 229-263. The lattice and the
  replication argument.
- Leisen, D. and Reimer, M. (1996). "Binomial models for option valuation --
  examining and improving convergence". *Applied Mathematical Finance* 3(4),
  319-346. The order-1 result for CRR, the oscillation, and the construction
  that removes it.

Everything above was re-derived and re-measured here; no prose, code, table or
example sequence is taken from either source.
