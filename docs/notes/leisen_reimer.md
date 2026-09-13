# The Leisen-Reimer tree: matching the tails instead of the moments

Companion to `docs/notes/crr_tree_convergence.md`, which measured the CRR
tree at order 1 with an error that changes sign with the parity of `n`. This
note derives the tree that removes both, measures it, and records the three
places where the improvement does **not** appear.

Everything below was re-derived and re-measured in this repository. The scheme
is Leisen and Reimer (1996); no prose, table, code or example sequence is
taken from that paper or any other.

## Where CRR's order-1 error actually comes from

Write the Black-Scholes price as

```
V = S e^{−qT} N(d1) − K e^{−rT} N(d2)
d1 = (log(S/K) + (r − q + σ²/2) T) / (σ √T),    d2 = d1 − σ √T
```

and the `n`-step binomial price of the same call in the same shape:

```
V_n = S e^{−qT} Φ' − K e^{−rT} Φ
```

where `Φ = P(Bin(n, p) > n/2 + m)` is the probability that the tree finishes
in the money, `m` being the (scheme-dependent) offset of the strike among the
terminal nodes, and `Φ'` is the same probability under the share measure —
the measure in which the up-probability is `p' = u p / e^{(r−q) dt}`.

Both prices have the *same* two terms. So the tree's error is, term by term,
the error of a **binomial tail against a normal tail**:

```
V_n − V = S e^{−qT} (Φ' − N(d1)) − K e^{−rT} (Φ − N(d2)).
```

That is the whole story of CRR's behaviour. A binomial tail is a step
function of its cut-off: it changes only when the cut-off crosses a lattice
node. CRR fixes the lattice (`u = e^{σ√dt}`, `d = 1/u`) and lets the strike
fall where it may among the terminal nodes, so `Φ − N(d2)` inherits a
sawtooth in `n`. At the money `d = 1/u` puts a node exactly on the strike for
even `n` and exactly midway between two nodes for odd `n`, which is the
measured odd/even oscillation; away from the money the offset follows no such
pattern and the error constant is simply erratic (measured in
`tests/oracle/test_tree_vs_quantlib.py`: fitted orders 1.21 to 1.48 with
log-space residuals 0.37 to 1.52, in QuantLib's engine as well as this one).

## The inversion: fix the tails, solve for the lattice

Leisen and Reimer turn the construction around. Instead of choosing `u` and
`d` and accepting whatever `Φ` results, **choose the probabilities that make
`Φ` and `Φ'` right**, and solve for `u` and `d`.

Concretely: put the strike exactly at the middle of the terminal grid, so the
cut-off is `n/2` with no offset, and pick

```
p   such that  P(Bin(n, p)  > n/2) = N(d2)
p'  such that  P(Bin(n, p') > n/2) = N(d1)
```

to the accuracy of a high-order normal approximation of the binomial tail. Two
remarks before the formula:

- `P(Bin(n, p) > n/2)` is the probability of **strictly more than** `n/2` up
  moves. That counts a whole number of outcomes only when `n` is odd. For even
  `n` the event "more than `n/2`" and the event "at least `n/2`" straddle a
  lattice node and there is no construction to invert. **The scheme requires
  an odd `n`**, and this package rejects an even one rather than rounding.
- Inverting a binomial tail exactly would need a root-find per step count.
  Peizer and Pratt (1968) give a closed-form normal approximation accurate
  enough that inverting *it* is as good; Leisen and Reimer use the inversion
  of their second (more accurate) variant, and so does this package.

### Peizer-Pratt, method 2

`qpl.engines.tree.lattice.peizer_pratt_inversion`, restated in words: the
probability whose binomial tail matches the normal tail at `z` is

```
h(z, n) = 1/2 + sign(z) · √( 1/4 − 1/4 · exp( −(n + 1/6) · t² ) )
t       = z / (n + 1/3 + 0.1/(n + 1))
```

The `0.1/(n + 1)` term is what distinguishes "method 2" from the simpler
"method 1"; dropping it costs accuracy at small `n`. Two properties the rest
of the construction leans on, both measured in
`tests/test_tree_leisen_reimer.py`:

- **Antisymmetry.** `z` enters only through `t²` and an explicit sign, so
  `h(−z, n) = 1 − h(z, n)` — measured to 5.6e-17, one ULP of 1/2, over `z` in
  {0.001 … 5} and `n` in {1 … 8001}.
- **Monotonicity.** `h` is strictly increasing in `z`.

### The multipliers

Set `p = h(d2, n)` and `p' = h(d1, n)`. Two constraints pin `u` and `d`:

1. the share measure must be the `p'`-measure: `u p / e^{(r−q) dt} = p'`;
2. the tree must reprice the one-step forward:
   `p u + (1 − p) d = e^{(r−q) dt}`.

Solving,

```
u = e^{(r−q) dt} · p' / p
d = (e^{(r−q) dt} − p u) / (1 − p)
```

The second is written in that form, rather than as the algebraically equal
`e^{(r−q) dt} (1 − p') / (1 − p)`, precisely so that constraint (2) holds to
round-off by construction. That is what makes put-call parity exact on the
lattice: measured residual at most 9.7e-12 at `n = 2001` across four
specification points, against the same 1e-10 budget CRR uses.

### Three consequences

- **`u d ≠ 1`.** The lattice is not centred on the spot. `u d = e^{2(r−q) dt}
  p'(1 − p') / (p (1 − p))`, which equals 1 only in the degenerate case `r = q`
  at the money forward. This matters downstream; see "The theta bug" below.
- **No no-arbitrage condition to check.** `d1 > d2` and `h` is increasing, so
  `p' > p`, and `p' > p` is exactly `d < e^{(r−q) dt} < u`. CRR can produce a
  `p` outside `[0, 1]` on a coarse tree with a large drift; this cannot.
- **The strike enters the geometry.** A call and a put on the *same* strike
  share a lattice (`d1` and `d2` do not know the option's kind), which is why
  parity is meaningful. A strike *ladder* does not: each strike gets its own
  tree.

### Symmetry at the money forward

"At the money" for this scheme means at the money **forward**,
`K = S e^{(r−q)T}`, which is where `d2 = −d1`. Antisymmetry then gives
`p' = 1 − p` and the multipliers become mirror images about the forward:

```
u d = e^{2(r−q) dt},      u / e^{(r−q) dt} = e^{(r−q) dt} / d
```

Measured: `p + p' − 1` is **exactly** 0.0 at every `n` tested and
`u d − e^{2(r−q) dt}` is at most 2.2e-16. With `r = q` this collapses to
`u d = 1`, i.e. the tree is spot-centred exactly where CRR always is. That is
the sense in which the scheme "reduces to a symmetric tree at the money".

## Measured: European

Reference point `S = K = 100, r = 5%, q = 0, σ = 20%, T = 1`, European call
(the put's signed errors are identical, because parity holds on the lattice).
Odd `n` throughout — there is no even branch. `h = 1/n`.

| `n` | Leisen-Reimer | signed error | `n²·abs(err)` | CRR `abs(err)` | ratio |
|---:|---:|---:|---:|---:|---:|
| 25 | 10.4500499402 | −5.336e-04 | 0.3335 | 7.038e-02 | 132 |
| 51 | 10.4504513209 | −1.323e-04 | 0.3440 | 3.443e-02 | 260 |
| 101 | 10.4505493366 | −3.424e-05 | 0.3492 | 1.737e-02 | 507 |
| 201 | 10.4505748602 | −8.712e-06 | 0.3520 | 8.724e-03 | 1001 |
| 401 | 10.4505813746 | −2.198e-06 | 0.3534 | 4.372e-03 | 1990 |
| 801 | 10.4505830203 | −5.519e-07 | 0.3541 | 2.188e-03 | 3966 |

Black-Scholes is 10.4505835722. `n²·abs(err)` settling at 0.354 *is* the order-2
constant. Fitted order **1.9840**, log-space RMS residual **0.0087**.

Off the money, at the two points where CRR's constant is erratic in both
engines:

| point | order | residual | `abs(err)` at `n=801` | CRR/LR at `n=101` | at `n=801` |
|---|---:|---:|---:|---:|---:|
| ATM (above) | 1.9840 | 0.0087 | 5.52e-07 | 507 | 3966 |
| `S=100, K=110, T=2, r=3%, q=1%, σ=25%` | 1.9842 | 0.0086 | 1.06e-06 | 35 | 2673 |
| `S=120, K=90, T=1, r=3%, q=5%, σ=35%` | 1.9709 | 0.0154 | 2.26e-07 | 1137 | 5229 |

**Order 2 survives off the money**, which is the central claim of the slice.
The weak cell (35x at `n = 101`) is CRR being accidentally good there, not
Leisen-Reimer being bad: off the money CRR's constant jumps around and at that
particular `n` it sits near a sign change.

Two honest caveats about the fitted orders:

- They sit consistently just **below** 2 (1.971 to 1.984), never above. The
  Peizer-Pratt inversion matches the binomial tail to high but *finite* order,
  so a slowly decaying correction rides on the `1/n²` term and drags the fitted
  slope down by 0.02-0.03 over a sixfold refinement.
- The log-space residuals (0.0086 to 0.0154) are an order of magnitude above
  CRR's on the same grid (0.0005), for the same reason. A clean power law fits
  at ~1e-03 here.

### No oscillation, measured rather than asserted

Over the 21 consecutive step counts `n = 101 … 121`, the CRR error changes
sign **20 times**: `+1.74e-02, −1.96e-02, +1.70e-02, …`. The Leisen-Reimer
error on the 11 odd counts in the same range is one-signed and monotone:
`−3.42e-05, −3.29e-05, …, −2.39e-05`, a total variation of 36% of its own
mean.

This is not the claim that the even branch has been fixed. There is no even
branch. It is the claim that within the sequence the scheme does define,
refinement moves the error smoothly, which is what makes a single `C/n²` fit
mean something — the CRR study could only get that by splitting the grid by
parity first.

### Richardson

Extrapolating the order-2 sequence with `n²` weights,
`V_ext = (n₂² V(n₂) − n₁² V(n₁)) / (n₂² − n₁²)`, over consecutive odd pairs
gives a fitted order of **2.9577** (residual 0.0271) and an error of
**1.465e-09** at the `(401, 801)` pair. So the expansion really does continue
`C/n² + D/n³ + …` with well-behaved coefficients. Contrast the American case
below.

## Measured: American

Same specification, American put, against the in-repo bracketed limit
`6.090376463020103` (the CRR engine's average of `n = 64000` and `n = 64001`,
so independent of the scheme being measured).

| scheme | order | residual | signed error at `n=801` |
|---|---:|---:|---:|
| leisen-reimer | 1.0641 | 0.0176 | −3.663e-04 |
| crr | 0.9872 | 0.0034 | +1.820e-03 |

**Order 1, and expected to be.** The European argument is about the terminal
distribution; the American value's dominant discretisation error is the
location of the early-exercise boundary, which a lattice resolves only to its
own node spacing. That error is `O(1/n)` in the value and is indifferent to
how the terminal grid was chosen.

Two things the scheme does buy here, both measured:

- **A better constant**, by a factor of 3.79, 4.05, 4.50, 4.55, 4.79, 4.97
  from `n = 25` to `n = 801`.
- **The opposite sign.** Leisen-Reimer approaches from *below* at every `n`
  tested, CRR from *above*. So the two schemes **bracket** the value at a
  shared `n` — a free error bar, where CRR alone needs an odd lattice and an
  even one to get the same thing.

**Richardson still fails**, exactly as Slice 2 found for CRR. On the
Leisen-Reimer lattice, `(n₂ V(n₂) − n₁ V(n₁)) / (n₂ − n₁)` over consecutive
odd pairs fits order **1.3423** with a log-space residual of **0.7187**, and
the extrapolated errors are non-monotone: 7.97e-04, 5.68e-04, 2.87e-05,
6.70e-05, 2.23e-05. Cancellation happens (one to two orders of magnitude) but
what is left is not a power law. The boundary error is not `C/n`; it is
`C(n)/n` with a constant that jumps as the node grid steps past the true
boundary. A better lattice does not change that, which is the useful negative
result of this slice.

## The theta bug, and the Greeks

### `u d ≠ 1` broke shared code

The lattice Greek estimator shared by the European and American engines read
theta as

```
θ ≈ (V(2,1) − V(0,0)) / (2 dt)
```

which is a pure difference **in time** only because `S(2,1) = S(0,0)` on a CRR
lattice. On a Leisen-Reimer lattice it is not: at `S = 100, K = 120, T = 1,
n = 801` the middle step-2 node sits 4.6e-02 above the spot, so the difference
also contains `offset · Δ`.

Measured consequence: uncorrected, the theta error at that point is **5.235**
with a fitted convergence order of **−0.002** — it does not converge at all,
while returning a plausible-looking number. Subtracting the second-order
Taylor expansion in spot,

```
θ = [ V(2,1) − V(0,0) − offset·Δ − offset²·Γ/2 ] / (2 dt),    offset = S(2,1) − S0
```

restores an error of 9.8e-03 at a fitted order of **1.000**. The correction is
applied only when the lattice is not spot-centred by construction, so CRR's
theta is unchanged in its last bits.

### The Greeks are still first order

The slice expected delta at about order 2 at the money. It is **1.00**.
Measured over `n` in {25, 51, 101, 201, 401}, at three points and both kinds:

| point | delta | gamma | theta |
|---|---:|---:|---:|
| ATM call | 0.996 | 1.003 | 0.995 |
| ATM put | 0.996 | 1.003 | 1.066 |
| `K=120` | 0.998 | 1.003 | 1.000 |
| `S=120, K=90, q=5%, σ=35%` | 0.999 / 1.000 | 1.008 | 1.002 / 1.005 |

The reason is not fixable by a better tree: delta, gamma and theta are read at
time levels 1 and 2 and used as estimates at time 0. That substitution is an
`O(dt)` error regardless of how good the lattice is, and `O(1/n)` dominates
the `O(1/n²)` price error. Order 2 would need a different *estimator* — an
extended lattice below the root, say — not a better tree.

### And the constants are better for only three of the five

The fallback expectation — "at least the constant improves" — is also false
for two Greeks. Absolute errors at the reference ATM call, with CRR shown at
the odd `n` and the even `n+1` that brackets it:

| `n` | scheme | delta | gamma | theta | vega | rho |
|---:|---|---:|---:|---:|---:|---:|
| 401 | leisen-reimer | 1.169e-04 | 2.774e-05 | 5.535e-04 | 3.065e-03 | 1.035e-05 |
| 401 | crr | 1.499e-04 | 1.738e-05 | 2.472e-03 | 2.273e-02 | 1.936e-02 |
| 402 | crr (even) | 7.935e-05 | 3.954e-05 | 7.724e-03 | 2.646e-02 | 2.963e-03 |
| 801 | leisen-reimer | 5.854e-05 | 1.388e-05 | 2.773e-04 | 3.055e-03 | 1.954e-06 |
| 801 | crr | 7.500e-05 | 8.697e-06 | 1.238e-03 | 9.853e-03 | 9.690e-03 |
| 802 | crr (even) | 3.978e-05 | 1.979e-05 | 3.867e-03 | 1.479e-02 | 1.486e-03 |

- **delta and gamma**: the Leisen-Reimer value sits *between* CRR's two
  parities, beating one and losing to the other. Same error, different source.
- **theta**: 4x to 14x better — it carries the `O(dt)` substitution *and* the
  price error at the step-2 node, and the second is what the order-2 lattice
  improves.
- **vega** (3x to 32x) and **rho** (190x to 5000x): bump-and-revalue, so they
  carry the price error divided by `2h`. CRR's oscillating error does not
  cancel between the two bumped evaluations and gets amplified by `1/(2h)`;
  the Leisen-Reimer error is smooth in both `σ` and `r`, so most of it
  cancels.

One consequence worth flagging for Phase 2: the Leisen-Reimer **vega error is
flat** at 3.06e-03 across `n` in {101, 201, 401, 801}. It has stopped being
discretisation error; what remains is the `O(h²)` bias of the central
difference at `VEGA_BUMP = 1e-2`, which was invisible under CRR's oscillation
and is now the binding term. A smaller bump would help this scheme and, per
that constant's own measurement note, hurt CRR.

## QuantLib

### The agreement

Unlike the CRR comparison — where the expected 1e-10 agreement turned out to
be impossible, because QuantLib's `CoxRossRubinstein` uses a different
probability on the same lattice — this one holds. Both engines invert the same
Peizer-Pratt method-2 formula at the same `d1`/`d2`.

- European call and put, `n` in {51, 201, 801}, at the money and at `K = 120`
  with a yield: worst residual against
  `BinomialVanillaEngine(process, "leisenreimer", n)` is **2.71e-11** on
  prices of 4.16 to 21.61.
- American put, `n` in {201, 801, 2001}: worst residual **7.3e-12**.
- American put at `n = 8001` against `FdBlackScholesVanillaEngine(3200, 3200)`:
  gap **1.52e-04**. Both sit below the limit (−3.888e-05 tree, −1.909e-04 FD),
  so that gap is a sum of same-signed errors. Slice 2 measured 3.71e-04 for
  the CRR tree at the same `n`; this more than halves it.

The 1e-11 floor rather than 1e-15 is QuantLib rolling back through its own
`TimeGrid` and accumulating round-off in a different order — the same floor
the CRR mirror test found.

### Quirk 1: QuantLib's even `n` is not a Leisen-Reimer price

QuantLib rounds even `n` up to the next odd number inside the `LeisenReimer`
tree, but the *engine's* time grid keeps the even count the caller asked for.
Measured ATM call error against Black-Scholes:

| `n` | error | `n` | error |
|---:|---:|---:|---:|
| 50 | −2.072e-01 | 51 | −1.323e-04 |
| 100 | −1.049e-01 | 101 | −3.424e-05 |
| 200 | −5.286e-02 | 201 | −8.712e-06 |
| 400 | −2.654e-02 | 401 | −2.198e-06 |
| 800 | −1.330e-02 | 801 | −5.518e-07 |

The odd column halves twice per doubling; the even column halves once. Order 1
against order 2, and a factor of **24 000** between asking for 800 steps and
asking for 801. A 20-line mirror that builds the parameters for `n + 1` steps
and rolls back over `n` reproduces about 95% of the even-`n` error (10.4379
against QuantLib's 10.4373 at `n = 800`), which identifies the mechanism: the
terminal distribution becomes an `n`-fold convolution of an `(n+1)`-step
parameterisation, so the strike is no longer where the construction put it.

**This is the measurement behind this package's decision to reject even `n`.**
Refusing costs the caller one character; rounding silently can cost four
orders of magnitude.

### Quirk 2: isolated `n` in QuantLib's American LR

At `n` in {501, 1601, 8001} QuantLib's American Leisen-Reimer value leaves the
smooth sequence it follows everywhere else, always *away* from the limit,
while this package's value stays between its own neighbours:

| `n` | qpl | QuantLib |
|---:|---:|---:|
| 451 | −6.606e-04 | −6.606e-04 |
| **501** | −5.918e-04 | **−9.524e-03** |
| 551 | −5.308e-04 | −5.308e-04 |
| 1501 | −1.916e-04 | −1.916e-04 |
| **1601** | −1.795e-04 | **−2.975e-03** |
| 1701 | −1.694e-04 | −1.694e-04 |

The direction identifies whose it is: a bug here would have to move this
package's value, and it does not. Pinned so it cannot later be mistaken for
one. Both quirks are observations about the QuantLib in the `[oracle]` extra
(1.43), not claims about the library in general; if a future version fixes
them, the pins should be deleted rather than loosened.

## What remains, for Phase 2

- **Order-2 Greeks.** The lattice estimators are the binding constraint, not
  the lattice. The standard remedy is an extended tree — two extra levels
  below the root, so delta and gamma are read *at* `t = 0` instead of at `dt`
  and `2 dt`. Worth a slice only if a case needs Greeks to better than
  `O(1/n)`; the PDE grid already gives them at order 2 and may be the better
  route.
- **The vega bump.** `VEGA_BUMP = 1e-2` was chosen against CRR's oscillation
  and is now the dominant vega error for this scheme. It should be re-measured
  per scheme, or vega should be obtained another way.
- **American order 2.** Nothing here gets it: the boundary error is the
  binding term, and neither a better lattice nor Richardson touches it. The
  Phase 2 candidates are PSOR on a non-uniform PDE grid concentrated at the
  strike, and Rannacher start-up for the kink — both of which attack the
  boundary directly rather than the terminal distribution.
- **Trinomial trees** are *not* scheduled. Leisen-Reimer gets order 2 for
  European payoffs with no extra machinery, and a trinomial lattice would be
  a third parameterisation with no case demanding it. Deferred until one does.

## Reproducing

```bash
PYTHONPATH=src python examples/tree_convergence.py --scheme leisen-reimer
PYTHONPATH=src python examples/tree_convergence.py            # the CRR comparison
```

Pinned by `tests/test_tree_leisen_reimer.py` (the inversion, the symmetry,
parity, the even-`n` refusal, the degenerate limits, the Greek orders and
constants), `tests/test_tree_lr_convergence.py` (the European and American
order fits, the no-oscillation measurement, the CRR comparison table,
Richardson), `tests/cases/*` (the same claims as data rows with evidence
classes) and `tests/oracle/test_lr_vs_quantlib.py` (QuantLib, and the two
quirks). The fitting is `qpl.validation.fit_convergence_order`.

## Sources

- Leisen, D. and Reimer, M. (1996). "Binomial models for option valuation --
  examining and improving convergence". *Applied Mathematical Finance* 3(4),
  319-346. The construction: matching the binomial tails at `d1` and `d2`, the
  choice of the Peizer-Pratt method-2 inversion, and the odd-`n` requirement.
- Peizer, D. and Pratt, J. (1968). "A normal approximation for binomial, F,
  beta, and other common, related tail probabilities, I". *Journal of the
  American Statistical Association* 63(324), 1416-1456. The tail
  approximation that is inverted.
- Cox, J., Ross, S. and Rubinstein, M. (1979). "Option pricing: a simplified
  approach". *Journal of Financial Economics* 7, 229-263. The lattice being
  improved on; derived in `docs/notes/crr_tree_convergence.md`.
