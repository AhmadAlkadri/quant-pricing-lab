# Digital options: what a discontinuous payoff does to four methods

A cash-or-nothing digital pays a fixed amount if the spot finishes on one side
of the strike and nothing otherwise. It is trivial analytically and it is the
cleanest available stress test for a numerical method, because its payoff is
not merely non-smooth — it is *discontinuous*. Everything below follows from
that one fact, and every number was measured in this repository.

Contract and convention: `call: cash * 1{S_T > K}`, `put: cash * 1{S_T < K}`,
both indicators strict, so a point exactly on the strike pays nothing either
way. That event has probability zero under the model, so the convention is
free; it matches QuantLib's `CashOrNothingPayoff`, which is what makes the
oracle comparison a comparison of the same contract.

## The closed form, and two identities that pin it

The value is a discounted probability:

    V_call = e^{-rT} E^Q[cash * 1{S_T > K}] = cash e^{-rT} Q(S_T > K)
           = cash e^{-rT} N(d2),
    V_put  = cash e^{-rT} N(-d2),
    d2 = (log(S/K) + (r - q - sigma^2/2) T) / (sigma sqrt(T)).

Source for the closed forms: Reiner and Rubinstein (1991), *Unscrambling the
binary code*, Risk 4(9), 75–83. The derivation above and every expression in
`qpl.engines.analytic.digital` are written out independently; nothing is
reproduced from that article. Hull's exotics chapter was used as a concept map
only.

Two identities carry the whole analytic layer.

**Static replication.** `1{S_T > K} + 1{S_T < K} = 1` almost surely, so the two
digitals together are a zero-coupon bond: `V_call + V_put = cash e^{-rT}`,
exactly, with no dependence on the spot, the strike or the volatility. This is
strictly stronger than put–call parity, and the measured residual is `0.0`.

**The digital is the strike-derivative of the vanilla.** Differentiating
`C = S e^{-qT} N(d1) - K e^{-rT} N(d2)` in `K`, the `phi(d1)`/`phi(d2)` terms
cancel and `-dC/dK = e^{-rT} N(d2)`, which is the digital call per unit of
cash. Mechanically the digital is the limit of a short call spread. Checked by
a central difference in `K` against this package's *vanilla* engine:

| h | worst residual over five points, both kinds |
|---:|---:|
| 1e-1 | 1.375e-06 |
| 1e-2 | 1.375e-08 |
| 1e-3 | 1.424e-10 |
| 1e-4 | 1.367e-10 |

A clean `O(h^2)` decay onto a round-off floor near 1.4e-10; `h = 1e-2` is used,
with a 5e-08 budget.

The five Greeks are differentiated by hand in the engine's docstring and
checked against central differences of the price (worst relative residual
1.4e-06, on gamma). One of them matters downstream:

    gamma_call = -cash e^{-rT} phi(d2) d1 / (S^2 sigma^2 T),

which carries the factor `-d1` and therefore **changes sign** at
`S = K exp(-(r + sigma^2/2) T)`. A digital is convex on one side of the strike
and concave on the other. Any claim about the accuracy of a numerical gamma has
to say where it is evaluated.

## Trees: the slice's expectation was wrong in both directions

The expectation written into the slice was *order 1 with large oscillation for
both schemes, with Leisen–Reimer failing to cure it*. Measured over odd `n` in
(25, 51, 101, 201, 401, 801), against the closed form:

| point | CRR order (residual) | LR order (residual) | error ratio CRR/LR at n=801 |
|---|---:|---:|---:|
| `S=K=100, r=5%, q=0, sigma=20%, T=1` | 1.0014 (0.0008) | 1.9844 (0.0085) | 4.55e+03 |
| `S=100, K=110, T=0.75, r=3%, q=1%, sigma=25%` | 0.3569 (0.5064) | 1.9839 (0.0088) | 4.68e+05 |
| `S=120, K=90, T=1, r=3%, q=5%, sigma=35%` | -0.0415 (0.1436) | 1.9836 (0.0089) | 6.34e+05 |

### Leisen–Reimer is order 2, and the mechanism is exact

Not "empirically about 2". The Leisen–Reimer construction chooses the per-step
probability `p` by inverting the binomial tail so that
`P(Bin(n, p) > n/2) = N(d2)` — and a digital call's price *is* that tail,
discounted. Two facts make the connection exact rather than asymptotic, and
both are asserted in `tests/test_digital_tree_convergence.py`:

1. the strike falls **strictly between the two central terminal nodes** at
   every point and every odd `n` tested;
2. the lattice price therefore equals `cash e^{-rT} P(Bin(n, p) > n/2)` to
   within 1e-13 of an independent evaluation of that tail
   (`scipy.stats.binom.sf`).

So the digital is the contract this construction is *best* at. For a vanilla,
Leisen–Reimer matches two tails that the price then combines; for a digital the
price is one tail with nothing else in it.

### CRR is order 1/2 off the money, with a sawtooth

A CRR lattice records only which side of the strike each terminal node fell on.
Moving the strike across a node changes the discrete payoff there by a whole
`cash`, and the price by `cash` times that node's risk-neutral probability,
which is `O(1/sqrt(n))`. Block-RMS fit over *every* odd `n` from 25 to 801,
grouped into the windows (25–50, 51–100, 101–200, 201–400, 401–801):

| point | block-RMS order | residual | RMS at 25–50 | RMS at 401–801 | sign changes |
|---|---:|---:|---:|---:|---:|
| otm | 0.5020 | 0.1254 | 3.84e-02 | 8.39e-03 | 10 |
| itm | 0.5037 | 0.0728 | 3.38e-02 | 8.22e-03 | 19 |

Seen at consecutive odd `n` the error is a ramp with a jump in it. At the otm
point, `n` from 121 to 201:

    +2.72e-02 +2.82e-02 +2.93e-02 +3.03e-02 +3.13e-02 | -2.85e-02 -2.72e-02 ...
                                                      ^ strike crosses a node

Largest-to-smallest error in that window: 113. A single `n` tells you nothing,
which is why "order 1 with oscillation" is the wrong description — the
oscillation is not a parity effect that averaging odd and even `n` removes, but
a fractional position that drifts continuously with `n`.

At the money CRR *is* order 1 (1.0014, residual 0.0008, no sign change), and
that is a symmetry rather than a rule: `u d = 1` makes the strike the geometric
mean of the two central terminal nodes for every odd `n`, freezing the
fractional position.

One more thing worth recording, because the slice expected it to matter. At the
money with an **even** `n` the centre terminal node is algebraically the spot,
hence the strike, and the strict payoff convention would drop it. It does not
happen: `S u^j d^{n-j}` evaluates to `100 ± 1.4e-13`, so the node is always
counted exactly once, and which side of the strike it lands on is decided by
round-off and flips with `n` (below at `n = 50`, above at `n = 100` and `200`).

### Lattice Greeks

Order 2 on the price does not carry over. Measured at the reference ATM call on
a Leisen–Reimer lattice over the same odd levels: delta 1.003, gamma 1.002,
theta 1.005. The estimators read levels 1 and 2 of the lattice, which is an
`O(dt)` substitution no scheme repairs — the same finding Slice 3 recorded for
vanillas. Rho is 1.895 (bumping `r` does not move the lattice, so it inherits
the price's order) and vega is flat at -0.011, pinned at the `VEGA_BUMP` bias
floor.

## Finite differences: a jump costs a full order in the *price*

Slice 4's kink pathology damaged gamma and left the price alone; the price is
an average over the grid and barely notices a ripple. A jump is one derivative
worse and the damage reaches the price. Reference point `S = K = 100, r = 5%,
q = 0, sigma = 20%, T = 1`, `n_s = n_t = n` over (100, 200, 400, 800) — every
one of which puts the strike exactly *on* a node when the grid is unaligned:

| configuration | price | delta | gamma |
|---|---:|---:|---:|
| plain CN, unaligned, no projection | 1.0012 | 0.8632 | 1.0153 |
| CN, unaligned, `cell_average` | 2.0153 | 2.0059 | 1.9979 |
| Rannacher, unaligned, `cell_average` | 2.0154 | 2.0061 | 1.9980 |
| plain CN, midpoint-aligned | 1.9291 | 2.0353 | 2.0313 |
| Rannacher, midpoint-aligned | 1.9248 | 2.0360 | 2.0315 |

The first row's price errors are -3.761e-02, -1.876e-02, -9.377e-03,
-4.689e-03 — a clean halving, so genuinely first order rather than noise. At
`n = 800` that is 0.88% of a price of 0.5323, against 2.229e-07 for the
remedied grid: a factor of 21 000.

### The projection

`PDEConfig(payoff_projection="cell_average")` puts the *average* of the payoff
over each cell `[S_i - ds/2, S_i + ds/2]` at node `i` rather than the point
value. For an indicator that average has a closed form — the fraction of the
cell in the money, `cash * clip((S_i + ds/2 - K) / ds, 0, 1)` for a call — so
there is no quadrature and no smoothing width to choose. It is the `L2`
projection of the terminal data onto the space of grid functions, which is what
"put this function on the grid" should mean when the function is not
continuous. This is the standard remedy analysed in Pooley, Forsyth and Vetzal
(2003), *Convergence remedies for non-smooth payoffs in option pricing*,
Journal of Computational Finance 6(4), 25–40; the implementation and every
number here are this repository's own.

The default is `"none"`, the field is read only by the digital engine, and a
vanilla priced with `"cell_average"` returns the same number bit-for-bit — the
same arrangement `PDEConfig.psor` has with the American engine.

### The projection and alignment are the same remedy

With `strike_alignment="midpoint"` the spacing satisfies `K = (j + 1/2) ds`, so
node `j`'s cell ends exactly *at* the strike. No cell straddles `K`, every cell
is wholly in or wholly out of the money, and the cell average of the indicator
equals its point sample. Measured: the two configurations agree to 1.11e-16 at
`n = 100` and are bit-for-bit identical at 200, 400 and 800.

So they do not stack. The slice expected `alignment + Rannacher + projection`
to be the recipe; the truth is that alignment and projection are one remedy
reached two ways, and the projection's job is the grid that *cannot* be
aligned.

### Rannacher and the jump representation are orthogonal, and both are needed

On the `n_s = n_t` path above, Rannacher is nearly cosmetic: it moves the
unaligned price by 2.358e-06 at `n = 100` against an error of 3.761e-02, and
the fitted order stays 1.0012 with or without it. Damping a jump that is
already mis-represented removes ripples it never had.

On the stressed path `n_s = 80 n_t`, `n_t` in (10, 20, 40) — the regime Slice 4
used to expose undamped Crank–Nicolson — it is the binding remedy:

| configuration | price | delta | gamma |
|---|---:|---:|---:|
| plain CN, unaligned, no projection | +0.0106 | -1.0008 | -1.9986 |
| CN, unaligned, `cell_average` | +0.9915 | -1.0004 | -0.9996 |
| plain CN, midpoint-aligned | +0.9912 | -0.9978 | -0.9980 |
| Rannacher, midpoint-aligned | +2.0043 | +2.0583 | +2.0592 |
| Rannacher, unaligned, `cell_average` | +2.0044 | +2.0586 | +2.0586 |

Three configurations **diverge**: refining makes the Greeks worse. The plain-CN
unaligned gamma error reaches **52.19** against a true gamma of -3.283e-04, and
Rannacher plus a consistent jump representation brings it to 4.4e-08 — a factor
of 1.2e+09. Cell averaging (or alignment) fixes how the jump is represented in
*space*; Rannacher fixes the fact that Crank–Nicolson's amplification factor
tends to -1 for the stiffest modes and therefore does not damp it in *time*.
Neither alone is enough. Rannacher start-up is Rannacher (1984), *Finite
element solution of diffusion problems with irregular data*, Numerische
Mathematik 43, 309–327, analysed in Giles and Carter (2006), *Convergence
analysis of Crank–Nicolson and Rannacher time-marching*, Journal of
Computational Finance 9(4), 89–112.

### Gamma, reported honestly

At the money, the remedied grid gives gamma at order 2.0315 (residual 0.0239) —
the same order as delta, with no penalty for the sign change elsewhere. At the
sign-change spot itself (93.2394, where the true gamma is -2.97e-19) the
*absolute* error still falls at order 2.1226 (residual 0.0837, from 9.67e-06 to
1.09e-07), but the relative error is unbounded at every level because the
quantity being approximated is zero. "Second order" is a statement about the
absolute error and says nothing about relative accuracy near a zero.

### The identity breaks on an unaligned grid

On the unaligned, unprojected grid at `n = 200` the two grid solves sum to
`cash e^{-rT}` minus **3.8%**: a node sits exactly on the strike and the strict
convention pays nothing there in *either* leg, so that node's whole weight is
lost. Cell averaging repairs it exactly, because the call and put projections
sum to `cash` at every node by construction. This is an argument for the
projection that has nothing to do with convergence order.

What survives is the time scheme's own discounting error, not round-off: the
sum is constant in `S`, so the scheme integrates `V' = -rV` and returns a Padé
approximant of `e^{-r dt}`. Measured -6.193e-10 for Crank–Nicolson and
+7.370e-08 for Rannacher, whose four fully implicit half steps are only first
order on that ODE. Rannacher is *less* exact on this identity and more accurate
on everything else.

## Monte Carlo: the one method that does not notice

An indicator is bounded and square-integrable, so the sample proportion is
unbiased at every `N` and the central limit theorem applies unchanged. At the
reference point, terminal sampling, seed 123:

| N | price | stderr | (price − closed form) / stderr |
|---:|---:|---:|---:|
| 5 000 | 0.534971 | 6.674e-03 | +0.397 |
| 20 000 | 0.532641 | 3.339e-03 | +0.095 |
| 80 000 | 0.532522 | 1.669e-03 | +0.118 |
| 200 000 | 0.531038 | 1.056e-03 | -1.218 |
| 320 000 | 0.532778 | 8.347e-04 | +0.543 |

Fitted standard-error order in `N`: **0.49990**, log-space residual
**1.59e-04** — the cleanest power law in this repository, and it should be: the
standard error is `sigma_payoff / sqrt(N)` with a `sigma_payoff` that barely
moves, not a discretisation error. The reported `ddof=1` standard error matches
the closed-form Bernoulli one, `cash e^{-rT} sqrt(p(1-p)/N)`, to 0.06%. Call
plus put is exact *path by path* (1e-14), since every sample is in the money for
exactly one leg.

### Greeks are refused

The pathwise derivative of `cash * 1{S_T > K}` is a Dirac mass, so no pathwise
estimator exists. A common-random-numbers bump is not a substitute: the bumped
and unbumped payoffs differ only on the `O(h)` fraction of paths that cross the
strike, each by a full `cash`, so the difference quotient has variance
`O(cash^2 / (N h))`. Measured at `N = 40 000` over 20 seeds, against a true
delta of 0.018762:

| h | mean | sd across seeds | sd / true delta |
|---:|---:|---:|---:|
| 1 | 0.018751 | 0.00041 | 2.2% |
| 0.1 | 0.018781 | 0.00177 | 9.4% |
| 0.01 | 0.017060 | 0.00474 | 25.3% |
| 0.001 | 0.014859 | 0.01585 | 84.5% |

A factor of 1000 in `h` multiplies the noise by 38.6, against the 31.6 that
`h^{-1/2}` predicts. Note the trap in the first row: a *coarse* bump looks
accurate here, because delta happens to be smooth at this point and the
`O(h^2)` bias is tiny. That is an accident of the point, not a method — the
estimator has no limit as `h -> 0`. `qpl.engines.mc.digital.greeks_digital`
therefore raises `NotSupportedError` naming the likelihood-ratio (score
function) estimator, which differentiates the density instead of the payoff and
is scheduled for Phase 3 (Glasserman, *Monte Carlo Methods in Financial
Engineering*, ch. 7).

## Four engines on one number

`qpl.cases.digital_black_scholes` carries the cross-engine rows. At
`S = K = 100, r = 5%, q = 0, sigma = 20%, T = 1`, plus two off-money points:

| leg | setting | worst \|error\| | budget |
|---|---|---:|---:|
| analytic | closed form | — | 1e-15 |
| Leisen–Reimer tree | n = 2001 | 5.284e-09 | 2e-08 |
| PDE | n_s = n_t = 800, aligned, Rannacher | 2.331e-05 | 8e-05 |
| Monte Carlo | 200 000 paths, seed 123 | \|z\| = 1.644 | 4 stderr |

The two order-2 deterministic engines are three decimal orders apart, and that
gap is the finding rather than an embarrassment: the lattice's error constant
is tiny here because the digital price *is* the binomial tail its construction
matches, while the grid still has to represent a step function.

## QuantLib

`tests/oracle/test_digital_vs_quantlib.py`, `T = 1.0` via `Actual365Fixed` and
365 days.

**Analytic.** `ql.CashOrNothingPayoff` through `AnalyticEuropeanEngine` agrees
with this package's closed form to **2.220e-16** on the price and to 3.5e-18 or
better on all five Greeks, over three points and both kinds. The payoff
convention is checked first, directly on the payoff objects: QuantLib uses the
same strict inequalities, so the two are the same contract.

**Finite differences.** At `tGrid = xGrid = n_s = n_t = 800` the gaps are
3.514e-05 (price), 1.056e-06 (delta) and 9.922e-08 (gamma), against tolerances
1.2e-04 / 3.5e-06 / 3.5e-07 derived as about 3.4× the worst measured gap.

The finding is about shape. QuantLib's digital price error over
`n` in (100, 200, 400, 800) at the money runs -9.20e-03, +7.79e-05, -2.67e-05,
-1.80e-07: sign changes, five decimal orders, log-space residual 0.85 and a
meaningless fitted "order" of 4.85. This engine's runs +1.22e-05, +3.38e-06,
+8.80e-07, +2.23e-07 — monotone, one-signed, residual 0.023, order 1.92. Same
mechanism as CRR on a digital: the log-spot mesher places no node consistently
relative to the jump. `dampingSteps = 2` does not change it (4.79 / 1.84 / 2.15
with residuals 0.84 / 0.55 / 1.17), so it is the mesher and not the time
scheme.

Stated honestly: QuantLib is *ahead* at the money at `n = 800` (1.80e-07
against 2.23e-07). The claim asserted is predictability under refinement, not
accuracy at one grid — and at `n = 100` QuantLib is 755× behind.

## Reproducing

```bash
PYTHONPATH=src python examples/digital_option_cross_method.py
PYTHONPATH=src python examples/digital_option_cross_method.py --case pde
pytest -q tests/test_digital_analytic.py tests/test_digital_tree_convergence.py \
          tests/test_digital_pde.py tests/test_digital_mc.py \
          tests/cases/test_digital_black_scholes_cases.py
pytest -q tests/oracle/test_digital_vs_quantlib.py   # needs the [oracle] extra
```
