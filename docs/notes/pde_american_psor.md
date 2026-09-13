# American exercise by finite differences: the LCP and PSOR

## What is actually being solved

An American vanilla is not a PDE problem. Writing `L` for the Black-Scholes
operator and `g(S)` for the exercise payoff, the value function satisfies three
conditions at every `(S, t)`:

```
V - g >= 0
V_t + L V <= 0
(V - g) (V_t + L V) = 0
```

The first says the holder may always exercise, so the value dominates the
payoff. The second says holding is never better than the PDE allows. The third
- complementarity - says one of the two is tight at every point: in the
continuation region the PDE holds with equality and `V > g`, and in the
exercise region `V = g` and the PDE holds strictly. The curve separating them
is the free boundary, and it is not given in advance. That is the whole
difficulty: a linear solve would need to know where the boundary is before it
could impose the right condition.

Discretising in `tau = T - t` with the same theta scheme the European engine
uses turns one time step into an algebraic linear complementarity problem:

```
A x - b >= 0,    x - g >= 0,    (x - g) * (A x - b) = 0    componentwise
```

with `A = I - theta dt L_h` and `b = (I + (1 - theta) dt L_h) v_old` plus the
Dirichlet contributions. `L_h` is the tridiagonal operator built by
`qpl.engines.pde.pricers._operator` - the same one the European engine
assembles, not a second copy.

The Dirichlet values are the European ones lifted to the payoff:
`max(g(0), V_eu(0))` at `S = 0` and `max(g(s_max), V_eu(s_max))` at the top. For
a put that turns `K e^{-r tau}` into `K`, which is the correct American value at
`S = 0` - exercise now, receive `K`. For a call the discounted forward
intrinsic already dominates `s_max - K`, so nothing changes and a no-dividend
American call keeps its European upper boundary. Both are the boundary-node
instance of `V >= g`.

## Projected SOR

PSOR runs an SOR sweep and clips every updated component against the obstacle:

```
y_i  = (b_i - sum_{j != i} A_ij x_j) / A_ii          Gauss-Seidel value
x_i <- max( g_i , x_i + omega (y_i - x_i) )          relax, then project
```

It converges to the LCP solution, and the projection does the free boundary's
work: a node whose relaxed value falls below the payoff is pinned to the
payoff, and the set of pinned nodes at convergence *is* the exercise region at
that time level. Nothing has to be located in advance.

Two properties of the implementation are worth stating because they are not the
textbook ones.

**The sweep is red-black.** Even-indexed nodes are updated first, then
odd-indexed ones. On a tridiagonal matrix each node's only neighbours are of
the other colour, so a whole colour is one vectorised numpy expression and the
result is exactly a Gauss-Seidel/SOR sweep in the permuted order red-then-black.
A tridiagonal matrix is consistently ordered in Young's sense under both the
natural and the red-black ordering, so the two share a spectral radius and an
optimal `omega`; they reach the same limit by different iterates (measured: 21
sweeps against 19 at `omega = 1.2`). `tests/test_pde_american.py` checks the
vectorised sweep against a lexicographic PSOR written out as a Python loop.

**`qpl.numerics.linear_systems.sor_solve` is not reused, and it is used.** It
is not reused as the solver because it takes a dense `(n, n)` matrix - one
sweep costs `O(n^2)` where a tridiagonal sweep costs `O(n)` - and because it
has no hook at which to project an iterate. What is reused is its shape: the
`(1 - omega) x + omega * gauss_seidel` update, the `0 < omega < 2` validation
with the same message, and reporting iterations and convergence rather than
only an answer. And it is used as the *reference*: with the obstacle removed,
the red-black sweep reproduces both `sor_solve` and the European engine's
LAPACK banded solve to 3.2e-13 on values up to 92.

## The stopping criterion

The iteration stops at the first sweep whose largest single-component change is
at or below `PSORConfig.tol`, an **absolute** tolerance in price units with a
default of `1e-8`. Absolute and not relative: the values on the grid run from
`0` to about `K`, so a relative test would be dominated by the deep
out-of-the-money nodes where everything is zero.

What that default costs, measured against a `tol = 1e-13` solve on the same
grid (ATM put, aligned, Rannacher, `n_s = n_t = n`):

| `n` | price given up | discretisation error | ratio |
|---|---|---|---|
| 100 | 2.3e-08 | 2.09e-02 | 9.1e+05 |
| 200 | 2.7e-08 | 5.64e-03 | 2.1e+05 |
| 400 | 6.0e-08 | 1.52e-03 | 2.5e+04 |
| 800 | 1.1e-07 | 4.24e-04 | 3.7e+03 |
| 1600 | 1.5e-07 | 1.25e-04 | 8.3e+02 |

The margin narrows as the grid is refined, which is the right way round to be
wrong: the solver error grows slowly (more time steps to accumulate over) while
the discretisation error falls fast. Tightening to `1e-10` costs about 27% more
sweeps and `1e-12` about 55%. The criterion cannot usefully be tightened below
about `1e-14 * K`: at the fixed point the Gauss-Seidel value differs from the
iterate by roughly `eps * |x|`, the relaxation multiplies that by `omega`, and
the test then never fires (measured: at `tol = 1e-14` and `omega = 1.5` the
sweep runs 100000 times while sitting 1.4e-14 from the answer).

An exhausted `max_iter` is never returned as if it had converged.
`on_max_iter="raise"` (the default) raises `InvalidInputError` naming the time
step; `on_max_iter="flag"` continues and records every offending step in
`meta["psor_unconverged_steps"]` with `meta["psor_converged"] = False`.

## Choosing omega

For a consistently ordered matrix, Young's theory gives
`omega* = 2 / (1 + sqrt(1 - rho_J^2))` with `rho_J` the Jacobi spectral radius.
Here `rho_J` is set by how strongly `A = I - theta dt L_h` is diagonally
dominant, and that is governed by `theta dt sigma^2 S^2 / ds^2` against `1`,
i.e. by `dt / ds^2` - not by the size of the grid. There is therefore no single
optimal `omega` for this engine, and the default is a choice about which
refinement path to be good on.

Mean sweeps per time step, ATM American put, aligned, Rannacher, `tol = 1e-8`,
on `n_s = n_t = n`:

| `n` | 1.00 | 1.05 | 1.10 | 1.15 | **1.20** | 1.30 | 1.50 |
|---|---|---|---|---|---|---|---|
| 100 | 5.96 | 7.00 | 8.51 | 10.03 | **11.33** | 14.38 | 24.52 |
| 200 | 7.33 | 7.00 | 8.26 | 9.79 | **11.17** | 14.15 | 23.41 |
| 400 | 9.72 | 7.98 | 8.07 | 9.28 | **10.65** | 13.50 | 22.49 |
| 800 | 13.99 | 12.12 | 10.30 | 9.51 | **10.18** | 13.14 | 21.45 |
| 1600 | 21.12 | 18.97 | 16.78 | 14.65 | **12.75** | 12.38 | 20.13 |

The per-grid optimum moves from 1.00 at `n = 100` to 1.30 at `n = 1600`,
because `dt / ds^2` grows like `n` along this path. `omega = 1.2` is the
flattest column - 11.33 down to 10.18 and back to 12.75, never worse than 1.9x
the per-grid optimum - while `omega = 1.0` degrades by 3.5x across the same
span and `omega = 1.5` costs about twice the optimum everywhere. That is the
reason for the default. That it also falls inside the 1.2-1.5 range the
standard references suggest is the weakest of the reasons.

It is the wrong choice on a grid that is fine in space and coarse in time. At
`n_s = 80 n_t, n_t = 20` the same put needs 1367, 952, 640, 391, 173 mean
sweeps at `omega = 1.0, 1.2, 1.4, 1.6, 1.8`, still falling at 1.8. The knob is
`PSORConfig(omega=...)`.

## Why the boundary limits the order

Two error terms are present and they have different orders.

The Crank-Nicolson/Rannacher scheme on a smooth solution is `O(ds^2) + O(dt^2)`,
and `strike_alignment="midpoint"` is what keeps the payoff kink from spoiling
that constant (Slice 4). On `n_s = n_t = n` both are `O(1/n^2)`.

The free boundary, though, is located only to the node spacing. Whether a given
node is in the exercise region at a given time level is a yes/no decision, so
the boundary the grid represents differs from the true one by `O(ds)`, and the
value error that follows is first order - and untouched by how good the time
stepping is. The second term must eventually dominate.

Measured on the ATM American put against `AMERICAN_BRACKETED_LIMIT =
6.090376463020103` (the average of the CRR lattice at `n = 64000` and
`n = 64001`), aligned grid, Rannacher start-up:

| `n` | error | local order |
|---|---|---|
| 50 | -6.963e-02 | -- |
| 100 | -2.086e-02 | 1.739 |
| 200 | -5.637e-03 | 1.888 |
| 400 | -1.515e-03 | 1.895 |
| 800 | -4.240e-04 | 1.837 |
| 1600 | -1.246e-04 | 1.766 |
| 3200 | -4.480e-05 | 1.476 |

Fitted order **1.8502** (log-space RMS residual 0.0265) over `n = 50 ... 800`
and **1.8506** (0.0261) over the disjoint window `n = 100 ... 1600`. The local
orders are already falling. So the honest statement is "between 1 and 2,
currently 1.85 and drifting down", and `tests/test_pde_american_convergence.py`
encodes exactly that: a band of `1.85 +- 0.12` that excludes both 1 and 2, plus
a separate assertion that the tail order is below the best one measured, so the
drift cannot quietly disappear.

Every error is negative: this engine approaches from below. The CRR lattice at
odd `n` approaches from above, so the two bracket the value at a shared
accuracy - a free error bar.

**Alignment is load-bearing.** With `strike_alignment="none"` the same sequence
(-6.96e-02, -4.70e-02, -1.16e-02, -3.01e-03, -7.90e-04) fits order 1.6885 with
a log-space residual of **0.2688**, ten times the aligned fit's. It is not a
power law at all, because the strike and the boundary move relative to the
nodes at every `n`. The test pins the residual, not the order, because the
order that fit reports is not measuring anything.

## The complementarity condition, checked

The engine measures the LCP rather than assuming it. At every time step it
computes the residual `A x - b`, the slack `x - g`, and the elementwise
`min(slack, |residual|)`, and reports the worst of each over the whole march in
`meta`. Measured over both kinds, `q` in `{0, 6%}` and `n_s = n_t` in
`50 ... 800`, at the default `tol = 1e-8`:

- `lcp_min_constraint_slack` is **exactly 0.0** everywhere. That is not a
  rounded zero: the projection assigns `np.maximum(obstacle, relaxed)`, which
  returns the obstacle value itself when the constraint binds, so an active
  node carries the payoff to the last bit.
- `lcp_min_operator_residual` reaches `-1.1e-07`, i.e. `A x >= b` is violated
  only at the solver's own tolerance.
- `lcp_max_complementarity` reaches `1.1e-07`, and that worst case is the
  no-dividend call, where the constraint is never active and the quantity is
  really the linear residual of a slowly converging solve. For the put it stays
  between 1.3e-09 and 1.8e-08.

Tightening `tol` by four orders of magnitude moves the complementarity residual
by three, which identifies it as the solver's and not the scheme's.

## The free boundary

The boundary is read off the converged grid at each time level: the node set
where the constraint is tight and the option is in the money, reduced to its
largest member for a put (the exercise region is `S <= B(t)`) and its smallest
for a call. Same convention as the lattice engine, which is what makes the two
comparable. `meta` carries `exercise_boundary` and `exercise_boundary_times`
together, because Rannacher's four half steps make the time grid non-uniform.

Slice 2 found that the *lattice* boundary is monotone only within a parity
subsequence: consecutive levels of a binomial tree live on two interleaved node
grids, so the reported boundary alternates between them and zigzags by up to
one fine-grid offset. That failure mode does not exist on a finite-difference
grid, for a structural reason rather than a lucky one: the node set is the same
at every time level, so there is one subsequence and it is the whole sequence.
Measured at `n_s = n_t = 400`: 402 steps, none negative, minimum step exactly
0.0. The call's boundary is non-increasing by the same argument.

Two further things the grid gets for free. It has **no NaN prefix**: the `n =
2000` lattice's boundary is NaN for its first 47 levels, up to `t = 0.0235`,
because no lattice node sits low enough that early, while the grid reaches down
to `S = 0` at every level and reports 80.5970 at `t = 0`. And at expiry the
boundary is exactly `K - ds/2 = 99.5025`, the node next to the strike - "equals
`K` up to node spacing", stated precisely.

Against the `n = 2000` lattice, `ds = 0.9950` and the lattice's node gap at the
strike is 0.4462, so the two can only be expected to agree to their sum,
1.4412:

| `t` | PDE | tree | difference |
|---|---|---|---|
| 0.10 | 81.5920 | 81.4062 | +0.1858 |
| 0.25 | 82.5871 | 82.1376 | +0.4494 |
| 0.50 | 83.5821 | 83.6202 | -0.0381 |
| 0.75 | 86.5672 | 86.6660 | -0.0989 |
| 0.90 | 90.5473 | 89.8228 | +0.7245 |
| 0.95 | 92.5373 | 92.2656 | +0.2717 |
| 0.99 | 95.5224 | 95.6264 | -0.1040 |

Worst 0.7245, about half the combined spacing, with both signs present - the
two boundaries interleave rather than one sitting consistently above the other,
which is what agreement inside a shared quantisation looks like.

## How the sweep count grows

Forsyth and Vetzal (2002) motivate their penalty method partly by the
observation that PSOR iteration counts grow as the grid is refined. That is
confirmed here, and qualified: what drives the growth is `dt / ds^2`, and a
refinement path can hold it down.

**Path A, `n_s = n_t = n`** (`dt / ds^2` grows like `n`):

| `n` | 100 | 200 | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|---|
| mean sweeps | 11.33 | 11.17 | 10.65 | 10.18 | 12.75 | 21.80 |

Flat to `n = 800` and rising after. Two effects nearly cancel there: the matrix
gets stiffer, but the previous time level - the starting iterate - gets closer
to the answer, because the solution moves by `O(dt)` over a step. A
least-squares fit over the whole range is not a power law (residual 0.195) and
the 0.10 exponent it reports should not be quoted.

**Path B, `n_s = 8 n_t`** (`dt / ds^2` grows like `n_t`, with a constant 64x
larger, so the starting-iterate effect is much weaker):

| `n_t` | 25 | 50 | 100 | 200 |
|---|---|---|---|---|
| `n_s` | 200 | 400 | 800 | 1600 |
| mean sweeps | 19.15 | 32.88 | 56.97 | 98.26 |

This one is a clean power law: exponent **0.787** in `n_t`, log-space RMS
residual **0.0018**. Optimal SOR on a consistently ordered matrix would give
0.5 and Gauss-Seidel 1.0; 0.787 at a fixed `omega = 1.2` sits between them,
which is where a fixed relaxation away from the per-grid optimum belongs.

One more finding in the same family, and it points the other way from the
intuition that a constraint is extra work: **the projection makes PSOR
faster**. The no-dividend American call has an empty active set, so the
iteration has to converge the whole linear system, including the poorly
dominant rows at the top of the grid. Its sweep counts are 12, 12, 16, 33, 59
over `n = 50 ... 800`, against 11-12 flat for the put on the same grids.

## Three engines, one number

Four discretisations of the same free-boundary problem now exist in or next to
this repository:

| engine | state space | early exercise | error at the ATM put |
|---|---|---|---|
| CRR lattice, `n = 8001` | geometric nodes | Bellman maximum | +1.800e-04 |
| Leisen-Reimer, `n = 8001` | geometric nodes | Bellman maximum | -3.888e-05 |
| this PDE, `n_s = n_t = 800` | uniform in spot, strike aligned | LCP inside each step | -4.240e-04 |
| QuantLib FD, grid 3200 | uniform in `log S` | step condition between steps | -1.909e-04 |

Worst pairwise gaps among the first three: 6.040e-04 at the ATM point (the PDE
and CRR errors have opposite signs, so that is a *sum*, not a cancellation) and
1.624e-04 at the Longstaff-Schwartz row-1 point, where all three sit below the
limit. The tolerances in `qpl.cases` are those sums, 1.5e-03 and 5e-04.

Against QuantLib, at `T = 1` with `Actual365Fixed` so 365 days is exactly 1.0:
measured gaps 2.331e-04, 4.209e-05 and 1.215e-04 at the three points, against a
budget of 8e-04 derived as the sum of the two engines' own measured errors.

**The two finite-difference engines do not share a rate**, which the slice
statement did not anticipate. On a joint `tGrid = xGrid` refinement QuantLib
fits order **1.0328** (residual 0.0220) against this engine's **1.8502**
(0.0265):

| grid / `n` | QuantLib FD | this PDE |
|---|---|---|
| 200 | -3.361e-03 | -5.637e-03 |
| 400 | -1.567e-03 | -1.515e-03 |
| 800 | -7.612e-04 | -4.240e-04 |
| 1600 | -3.780e-04 | -1.246e-04 |
| 3200 | -1.909e-04 | -- |

QuantLib halves its error per doubling; this engine divides it by about 3.5. At
`n_s = n_t = 800` this engine is within 12% of QuantLib's `1600` error, and at
`n = 1600` it is ahead of its `3200` outright, at a quarter of the node count.

The obvious explanation is wrong: `dampingSteps` is not the cause. At grid 3200
the American put moves 1.5e-05 between 0 and 2 damping steps - an eighth of the
engine gap, and in the wrong direction. The remaining candidates are QuantLib's
unaligned log-spot mesher (Slice 4 measured what the alignment is worth on a
European price) and its `FdmAmericanStepCondition`, which projects between time
steps rather than solving the complementarity conditions inside them. This
slice did not separate them; separating them would mean reimplementing one
engine on the other's grid.

## Greeks

The estimators are the European ones, reached through the same code with the
American solver passed in: a central stencil reads a slope off neighbouring
values and does not care how the values were produced. Measured against the
lattice at `n = 8001`, with the PDE at `n_s = n_t = 800`:

| Greek | PDE | tree (8001) | difference |
|---|---|---|---|
| delta | -0.411107 | -0.411062 | -4.5e-05 |
| gamma | 0.022990 | 0.022989 | +1.1e-06 |
| vega | 37.4874 | 37.4866 | +8.3e-04 |
| theta | -2.237925 | -2.237981 | +5.6e-05 |
| rho | -30.2211 | -30.2198 | -1.3e-03 |

Two American-specific points.

**Theta in the exercise region.** The PDE identity is the PDE, and inside the
exercise region the PDE does not hold - that is the content of complementarity.
There `V = g(S)` is independent of time, so theta is exactly `0`, while the
identity would return `rK - qS`. At `S = 40, K = 100, r = 5%` that is `+5.0`
for a quantity whose true value is zero. The engine detects the exercise region
from the obstacle, reports `theta = 0.0`, and names the source in
`meta["theta_source"]`. Delta and gamma need no such correction: on the payoff
the stencils return `-1` and `0` of their own accord (measured: `-1.0` and
`4.3e-15`).

**Gamma near the free boundary.** The American value has a genuine kink in `S`
there - value matching holds, smooth pasting only in the continuum limit - so a
second difference taken across it differences a function whose second
derivative is a delta in the limit. Away from the boundary it is as well
behaved as the European gamma, which is where the numbers above are measured.

## Degenerate limits

At `T = 0` the price is the payoff. At `sigma = 0` the spot is deterministic and
the American value is a one-dimensional maximisation over the exercise date,
`max_t e^{-rt} intrinsic(S0 e^{(r-q)t})`. That is a fact about the model, not
about either discretisation, so the PDE engine imports it from
`qpl.engines.tree.american` rather than restating it - including the Slice 2
result that the naive `max(intrinsic now, discounted forward intrinsic)` is
wrong when `q > r`, where the objective has an interior maximum at
`t* = log(rK / (q S0)) / (r - q)`.

The consequence for evidence is worth being blunt about: a test asserting that
the two engines agree there would be checking shared code. So
`tests/test_pde_american.py` re-derives the value by a 200001-point scan over
the forward path instead, and reproduces the pinned `q > r` case (`S0 = K =
100, r = 2%, q = 50%, T = 10`: value 83.9505861332, against 81.1993 for the
better endpoint) through the PDE entry point.

## Citations

- Cryer, C. W. (1971). *The solution of a quadratic programming problem using
  systematic overrelaxation*. SIAM Journal on Control 9(3), 385-392.
  (Convergence of projected SOR.)
- Wilmott, P., Dewynne, J., and Howison, S. (1993). *Option Pricing:
  Mathematical Models and Computation*. (The obstacle-problem formulation of
  American exercise and the PSOR treatment of it.)
- Forsyth, P. A., and Vetzal, K. R. (2002). *Quadratic convergence for valuing
  American options using a penalty method*. SIAM Journal on Scientific
  Computing 23(6), 2095-2122. (The penalty alternative, and the observation
  that PSOR iteration counts grow under refinement.)
- Young, D. M. (1971). *Iterative Solution of Large Linear Systems*.
  (Consistent ordering, the optimal relaxation formula, and why a red-black
  ordering of a tridiagonal matrix has the same spectral radius as the natural
  one.)
- Rannacher, R. (1984). Numerische Mathematik 43, 309-327; Giles, M. B., and
  Carter, R. (2006). Journal of Computational Finance 9(4), 89-112.
  (Start-up damping; see `docs/notes/pde_greeks_and_rannacher.md`.)
- Pooley, D. M., Forsyth, P. A., and Vetzal, K. R. (2003). Journal of
  Computational Finance 6(4). (Strike alignment; see
  `docs/notes/pde_strike_alignment.md`.)
- Merton, R. C. (1973). Bell Journal of Economics and Management Science 4(1),
  141-183. (An American call on a non-dividend-paying stock is never exercised
  early.)

Every number in this note was measured in this repository. None is quoted from
any of those sources.

## Reproducing

```
pytest -q tests/test_pde_american.py tests/test_pde_american_convergence.py
pytest -q tests/cases/test_american_black_scholes_cases.py
pytest -q tests/oracle/test_pde_american_vs_quantlib.py   # needs the [oracle] extra
PYTHONPATH=src python examples/american_put_cross_method.py
```
