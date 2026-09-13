# Non-uniform grids and the barrier PDE

Slice 13. Phase 2 deferred the non-uniform mesh three times, each time because
the motivating problem already had a cheaper remedy. The barrier is the case
that does not: a knock-out's boundary condition lives at a level that is not
the strike, and a grid that misses it prices a different contract. This note
derives the stencil such a grid needs, states its order honestly, gives the
mesh, and reports what was measured.

## 1. The three-point stencil on a general grid

Let `S_{i-1} < S_i < S_{i+1}` with

    h- = S_i - S_{i-1},      h+ = S_{i+1} - S_i.

Taylor about `S_i`:

    V_{i+1} = V_i + h+ V' + h+^2/2 V'' + h+^3/6 V''' + O(h^4),
    V_{i-1} = V_i - h- V' + h-^2/2 V'' - h-^3/6 V''' + O(h^4).

Multiply the first by `h-^2` and the second by `h+^2` and subtract: the `V''`
terms cancel and

    V'(S_i) = -h+/(h-(h-+h+)) V_{i-1}
              + (h+-h-)/(h- h+) V_i
              + h-/(h+(h-+h+)) V_{i+1}
              - (h+ h-)/6 V''' + O(h^3).

Multiply the first by `h-` and the second by `h+` and add: the `V'` terms
cancel and

    V''(S_i) = 2/(h-(h-+h+)) V_{i-1}
               - 2/(h- h+)   V_i
               + 2/(h+(h-+h+)) V_{i+1}
               + (h+-h-)/3 V''' + O(h^2).

At `h- = h+ = h` both collapse to the textbook central differences.

**The honest statement.** The first-derivative arm is second order on any grid.
The second-derivative arm is **first order pointwise** unless `h+ = h-`: its
leading error is proportional to the *difference* of the neighbouring spacings.
A naive reading therefore predicts a first-order scheme on any non-uniform
mesh, and that prediction is wrong for a reason worth spelling out.

**Why the global error is still second order.** Let the grid be the image of a
uniform grid under a smooth, strictly increasing map `S = g(xi)`,
`xi_i = i dxi`. Then

    h+ - h- = g(xi_{i+1}) - 2 g(xi_i) + g(xi_{i-1}) = g''(xi_i) dxi^2 + O(dxi^4),
    h+ + h- = 2 g'(xi_i) dxi + O(dxi^3),

so `(h+ - h-)/(h+ + h-) = O(dxi)` and the offending term is
`O(dxi) x O(h) = O(h^2)`. The pointwise first-order term is multiplied by a
spacing that is itself shrinking. Equivalently: the scheme is second order in
the transformed variable, in which the grid is uniform.

That is a statement about grids from *smooth maps*, not about arbitrary node
lists, and this repository measures both halves rather than asserting either.

| grid                             | 1st-derivative order | 2nd-derivative order |
|----------------------------------|---------------------:|---------------------:|
| spacings alternating `h, 2h`     | 2.0000               | **1.0000**           |
| `sinh`, `concentration = 0.05`   | ~2                   | ~2                   |

(`tests/test_pde_nonuniform_grid.py`; the alternating grid is the control that
makes the `sinh` row mean something.)

Sources: Duffy (2006), *Finite Difference Methods in Financial Engineering*,
states the consistency order of this stencil; Tavella and Randall (2000),
*Pricing Financial Instruments: The Finite Difference Method*, chapter 5, is the
coordinate-transformation view. The derivation above and every number below are
this repository's own.

## 2. The `sinh` mesh

Define the mesh by its *density*: nodes uniform in a coordinate whose
derivative is large where accuracy is wanted. Take

    xi(S) = sum_j asinh( (S - c_j) / alpha ),      alpha > 0,

so that

    dxi/dS = sum_j 1 / sqrt( alpha^2 + (S - c_j)^2 ),

a sum of bumps of height `1/alpha` and width `alpha` on the critical points
`c_j`. With one critical point the map inverts in closed form,

    S(xi) = c + alpha sinh(xi),

which is the standard mesh formula — In 't Hout and Foulon (2010), "ADI finite
difference schemes for option pricing in the Heston model with correlation",
*IJNAM* 7(2), 303–320, section 3, cited for the mesh alone (this repository has
no ADI scheme and no Heston model). With several critical points there is no
closed-form inverse and the nodes come from bisection on the monotone `xi(S)`.

`alpha` is supplied as the dimensionless `PDEConfig.concentration`:
`alpha = concentration * (s_max - s_min)`. Large values flatten the mesh toward
uniform, small values pile nodes onto the critical points and starve the tails.

### Anchors

Two placement requirements meet on one grid and they pull in opposite
directions.

*The barrier must be a node.* A knock-out's Dirichlet condition is imposed at
`S = H`; if `H` falls between two nodes the scheme imposes it at one of them,
which is a different contract. Each such point becomes a segment boundary of
the `xi` grid and is written back exactly, so `grid.s[i] == H` holds with `==`.

*The strike must not be a node.* The payoff kink costs the scheme a power of
`ds` when it lands on a node and does not when it lands at a cell midpoint
(Slice 0, `docs/notes/pde_strike_alignment.md`). The spacing of the segment
containing the strike is nudged until `K` is the arithmetic midpoint of its
cell, which is possible whenever that segment's far end is free to move — i.e.
`s_max` absorbs the adjustment, exactly as it does on the legacy uniform grid.
On a uniform grid the `xi` map is the identity and the half-integer rule *is*
the `S`-midpoint condition; on a `sinh` grid the two differ by `O(h^2)`, so the
spacing is refined by bisection until the `S`-midpoint condition holds to
round-off.

When both ends of the strike's segment are pinned — an up-and-out barrier,
whose domain is `[0, H]` — the midpoint cannot be imposed and the realised
offset is reported in `meta["strike_cell_offset"]` rather than silently
accepted.

`grid="uniform"` (the default) is bit-for-bit the pre-slice grid: the
construction, the operator coefficients and the Greek stencils are each
compared with `==` against the pre-slice expressions written out in the tests.

## 3. Orders on the `sinh` grid

`S = K = 100`, `r = 5%`, `q = 0`, `sigma = 20%`, `T = 1`, `n_s = n_t = n` over
`n = 50 … 800`, midpoint alignment, Rannacher start-up.

| quantity      | `sinh` order | uniform order | `sinh`/uniform error at `n = 800` |
|---------------|-------------:|--------------:|----------------------------------:|
| vanilla price | 1.9915       | 1.9973        | 0.86                              |
| digital price | 1.9907       | —             | —                                 |
| delta         | 1.9911       | 2.0014        | 0.098                             |
| gamma         | 1.9924       | 2.0668        | 0.171                             |

Order two throughout, which is the claim the stencil had to earn. The
*constants* barely move on a vanilla price — the mesh is worth about 15% here —
and move by a factor of ten on the Greeks. For a vanilla that is a nicety. The
barrier is where it is the method.

## 4. The barrier: on a node, and off it

`S = K = 100`, `H = 95`, `r = 8%`, `q = 4%`, `sigma = 25%`, `T = 0.5`,
down-and-out call, zero rebate. Closed form 4.5125986078 (checked to 3e-14
against QuantLib in Slice 12).

### Continuous monitoring

The domain is truncated at the barrier — `[H, s_max]` for a down type, `[0, H]`
for an up one — so the Dirichlet condition sits on `S = H` exactly and there is
no alignment arithmetic at all.

| `n` | on node (uniform) | off node (uniform) | on node (`sinh`) |
|----:|------------------:|-------------------:|-----------------:|
| 100 | −4.676e−03        | +8.515e−01         | −2.594e−04       |
| 200 | −8.861e−04        | +1.182e+00         | −6.838e−05       |
| 400 | −2.568e−04        | +3.924e−01         | −1.665e−05       |
| 800 | −5.959e−05        | +2.396e−01         | −4.108e−06       |
| fitted order | **2.0668** (res 0.0883) | **0.7079** (res 0.3068) | **1.9981** (res 0.0195) |

Three things in that table.

**Off a node the scheme is first order with an erratic constant.** The fit of
0.7079 has a log-space residual of 0.3068, which is the diagnostic saying it is
not a power law at fixed `n` — the constant is set by the fractional part of
`H / ds`. The assumption-free form is `|error| x n` in `[85.2, 245.7]` through
`n = 1600`: bounded and not shrinking. Every error is **positive**, because the
scheme kills at the largest node at or below `H` and therefore prices a barrier
further from the spot, which is worth more. Against the aligned grid at the
same node count the errors are 182x to 4021x larger. This is the
barrier-on-a-node requirement of Zvan, Vetzal and Forsyth (2000), "PDE methods
for pricing barrier options", *Journal of Economic Dynamics and Control* 24,
1563–1590.

**The mesh buys a factor of 13–18 in the constant**, not an order: 18.0, 13.0,
15.4, 14.5 at `n = 100, 200, 400, 800`. Concentration scan at `n = 100`:

| `concentration`   | 0.02      | 0.05      | 0.10      | 0.20      |
|-------------------|----------:|----------:|----------:|----------:|
| barrier error     | 2.114e−04 | 2.594e−04 | 4.355e−04 | 9.397e−04 |
| gain over uniform | 22.1      | 18.0      | 10.7      | 5.0       |
| vanilla error     | 2.491e−03 | 1.462e−03 | 9.261e−04 | 5.376e−04 |
| gain over uniform | **0.10**  | **0.18**  | **0.28**  | **0.49**  |

Reported, not tuned. The default `concentration = 0.05` is *not* the best cell
on the barrier, and every cell is a **loss** on a plain vanilla at the same
point: a mesh that starves the tails to feed the barrier pays for it. The mesh
is a placement remedy, not free accuracy.

**Order two is what neither Slice 12 discretisation could reach.** The lattice
and the simulation are each order one half on this contract, for the same
reason: both displace the barrier by `O(sigma sqrt(dt))`. A grid places a node
on the barrier for every time step at once.

### Discrete monitoring

The dead region is part of the domain — a path may dip past the barrier and
come back — so the grid is the full `[0, s_max]` and the barrier enters only at
the monitoring dates. **The time grid is built to contain them**: the levels are
the sorted union of the `n_t` uniform levels `j T / n_t` and the `m` monitoring
levels `T - t_j`, merged at a relative tolerance of `1e-12` with the monitoring
value kept when two coincide. A projection therefore lands on a time level and
is never interpolated onto one; the cost is up to `n_t + m` steps rather than
`n_t`. Rannacher still splits the first two intervals of the merged grid.

Against the plain Monte Carlo estimator — unbiased for the discrete contract,
so this is two routes to one number — at 400 000 paths: `z` = −0.57, −2.31,
−1.15, −1.25, −0.32 over `m = 10, 20, 40, 80, 160`.

The gap to the continuous price, on the `sinh` grid at `n_s = n_t = 400`:

| `m`  | gap        | BGK-predicted gap | ratio  |
|-----:|-----------:|------------------:|-------:|
| 10   | +1.625746  | +1.602161         | 1.0147 |
| 20   | +1.213371  | +1.215666         | 0.9981 |
| 40   | +0.900876  | +0.903237         | 0.9974 |
| 80   | +0.658822  | +0.661287         | 0.9963 |
| 160  | +0.475119  | +0.479176         | 0.9915 |
| fitted order in `1/m` | **0.4431** (res 0.0101) | **0.4361** | |

The order is below one half, and the reason is measured rather than assumed:
the Broadie–Glasserman–Kou closed-form shift, on the same `m` ladder, has
fitted order 0.4361. The deficit is the correction's own `o(1/sqrt(m))` at
finite `m`, not the grid's — the same effect Slice 12 saw from the simulation
side (0.4603 ± 0.0125, paired). And the continuity correction predicts *this
engine's* gap to better than 1% from `m = 20` on, which is a genuinely
independent check: nothing in the engine knows about
`beta = -zeta(1/2)/sqrt(2 pi)`.

## 5. What the slice statement got wrong: the projection must be cell-weighted

Observing a barrier turns the value function into `V · 1{S > H}`, which is
**discontinuous**. Slice 6 measured what a discontinuity costs a grid that
samples it at nodes: a full order. Here it is worse than usual, because on a
grid where `H` *is* a node, that node's cell is half alive and the node is
nonetheless set to the rebate — so the scheme discards half a cell of value at
every monitoring date, and is biased **low**.

Measured on one grid (`s_max = 380`, `n_s = n_t = n`, so `H = 95` is node `n/4`
for every `n`), `m = 10`, against a converged reference:

| `n` | node-sampled | cell-weighted |
|----:|-------------:|--------------:|
| 100 | −6.339e−01   | +3.639e−02    |
| 200 | −3.365e−01   | +7.067e−03    |
| 400 | −1.711e−01   | +1.304e−03    |
| 800 | −8.604e−02   | +3.261e−04    |
| fitted order | **0.9619** (res 0.0140) | **2.2844** (res 0.0748) |

Every node-sampled error is negative, which is the half-cell argument's
signature.

The implemented projection is therefore the `L2` one: each node takes the
average of the post-observation function over its own cell,

    V_i <- alive_fraction * (average of V over the live part of cell i)
           + (1 - alive_fraction) * (dead value at the dead part's midpoint),

with the live average read off the **pre-observation** `V` — smooth across `H`,
the previous jump having diffused away — by linear interpolation. It needs no
node placement at all, which is the point: `barrier_alignment` stops mattering
for the discrete contract once the projection is weighted.
`barrier_alignment="none"` keeps the crude node rule for both conventions and
is what the two negative findings above are measured on.

## 6. Knock-ins, and parity as a statement about the matrix

A knock-in is priced **directly**, not as "vanilla minus knock-out". On the
live side of a continuously monitored down-and-in:

    terminal:   V(S, T) = rebate                  (paid at expiry if never touched),
    at S = H:   V(H, t) = BS(H, K, T - t),        the vanilla the touch delivers,
    at s_max:   V(s_max, t) = rebate e^{-r(T-t)}.

With a zero rebate the terminal data is identically zero and the whole value
arrives through the barrier boundary, which is a sharp test of that condition.
Measured errors against the closed form on the `sinh` grid: −5.911e−05
(`n = 100`), −3.986e−06 (`n = 400`), order ~2.

In-out parity is then a statement about the **discrete system**. The knock-out
and the knock-in are two right-hand sides of the same matrix on the same grid,
so a third leg whose data is the sum of theirs satisfies

    knock-out + knock-in = that leg

exactly. Measured relative residuals 1.13e−16 and 3.40e−16 (uniform,
`n = 100, 400`), 3.40e−16 and 1.13e−15 (`sinh`) — the banded solve's round-off
and nothing else. This is stronger than "in + out = vanilla in the limit": it
holds at `n = 100`, where the knock-in is still 3.96e−03 from its own closed
form.

## 7. Greeks

Delta and gamma come from the same three-point stencil, interpolated to the
spot; theta from the PDE identity; vega and rho by bump-and-revalue on the same
grid. This is the first barrier engine in this package with real Greeks — the
lattice and the simulation both refuse, for reasons about *their*
discretisations (a sawtooth amplified by `1/dt`, and a pathwise derivative that
is a surface measure) that do not apply to a grid whose barrier is a boundary.

Residuals against central differences of the closed form, `n_s = n_t = 400`,
`sinh`:

| `S`  | delta residual | gamma residual |
|-----:|---------------:|---------------:|
| 110  | +2.83e−06      | −1.51e−07      |
| 105  | +3.87e−06      | −1.14e−07      |
| 100  | +5.66e−06      | −2.60e−07      |

Fitted orders at the reference spot: delta 1.9980, gamma 1.9558 on the `sinh`
grid; 2.0830 and 2.0833 on the uniform one, with errors 39x and 41x larger at
`n = 800`.

**Slice 12's gamma finding, reproduced on a grid.** Gamma does not blow up as
`S -> H`; it converges to a finite limit having changed **sign**:

| `S`   | 110       | 105       | 100       | 98        | 96        | 95.5      |
|------:|----------:|----------:|----------:|----------:|----------:|----------:|
| gamma | +0.002874 | +0.000322 | −0.004628 | −0.007407 | −0.010671 | −0.011563 |
| delta | +0.884249 | +0.875405 | +0.885040 | +0.896994 | +0.914988 | +0.920549 |

A knock-out near its barrier is concave where the vanilla is convex (the
vanilla's gamma at `S = 96` is positive), because the value is pinned toward
the rebate from one side rather than bending away from the strike. The real
singularity is at the corner `(S = H, t = T)`.

## 8. The oracle, and the finding it produced

QuantLib's `FdBlackScholesBarrierEngine`, 24 cells (all eight types, both
kinds, three strikes, zero rebate) at `T = 1.0` exactly. Worst error against
the closed form at `n = 400` for both: **2.663e−03** (QuantLib, 20 damping
steps) against **2.958e−05** here — a factor of 90. Worst difference between
the two engines 2.663e−03, i.e. QuantLib's error almost exactly, which is what
the derived 5.0e−3 budget is.

The finding: QuantLib's engine is **order one** on exactly the contracts whose
terminal payoff jumps *across* the barrier, and order two on the rest.

| cell                      | QuantLib | this engine |
|---------------------------|---------:|------------:|
| down-and-out call `K=100` | 1.8767   | 1.9983      |
| down-and-out put `K=110`  | **0.9502** | 1.9810    |

`K = 100 > H = 95` for a call has a payoff that is zero on both sides of the
barrier; `K = 110` for a *put* has a payoff of `K - H = 15` at the barrier
against a rebate of 0. On the second cell QuantLib's errors run 9.49e−03 down to
1.36e−03 against this engine's 3.13e−05 down to 2.05e−06, and the gap *widens*
under refinement, which is what a difference of orders means.

The mechanism is the domain, and it is the whole point of the slice: a grid
that carries the dead region must represent that jump at `t = T`, and a jump
sampled at nodes costs a full order; a grid truncated at the barrier never has
it in the interior, only a corner at `(S = H, t = T)` that four fully implicit
Rannacher half steps damp. The attribution is an inference from the measured
orders, not a claim about QuantLib's source. `dampingSteps` is ruled out as the
cause: zero beats twenty on both cells (1.9855 against 1.8767, 0.9945 against
0.9502) — the same shape of finding Slice 4 recorded for
`FdBlackScholesVanillaEngine`.

QuantLib has no finite-difference engine that accepts a monitoring schedule, so
the discrete half of this slice has no FD oracle; a test records that and fails
if a future QuantLib grows one.

## 9. Reproducing

- `tests/test_pde_nonuniform_grid.py` — the stencil, its two orders, the
  bit-for-bit uniform grid, the mesh geometry.
- `tests/test_pde_barrier.py` — everything in sections 4 to 7.
- `tests/cases/test_barrier_black_scholes_cases.py` — the four-engine
  continuous rows and the two-engine discrete ones.
- `tests/oracle/test_barrier_pde_vs_quantlib.py` — section 8.
- `examples/barrier_pde_grid.py` (also `--case mesh`, `--case discrete`) — the
  tables, printed.

Fitting throughout is `qpl.validation.fit_convergence_order`; every order is
reported with its log-space residual, which is what says whether the fit means
anything.
