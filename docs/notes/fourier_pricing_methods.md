# Fourier pricing under Black-Scholes: four methods and what each one costs

Every engine in this package before Slice 14 discretises the **dynamics** — a
lattice, a grid, a path. A transform method discretises the **law** instead,
which it is allowed to do because a European payoff only ever sees `S_T`. Under
Black-Scholes that buys nothing a closed form does not already give, and that is
exactly why the methods are built here first: the reference is exact, so the
error measured against it is the method's own error with nothing else mixed in.

Everything below is measured in this repository. The sources named at the end
are cited for the methods; no expression, number or table is reproduced from any
of them.

## The interface

All four methods read the model through two functions and nothing else
(`qpl.engines.fourier.charfn`):

```
characteristic_function(u, expiry, *, rate, dividend)  ->  E[exp(i u ln(S_T/S_0))]
log_return_cumulants(expiry, *, rate, dividend)        ->  (c1, c2, c4)
```

Two choices in that signature are load-bearing.

**`u` may be complex.** Carr–Madan evaluates the transform at `u - (alpha+1) i`,
Lewis on the line `Im(u) = -1/2`, Gil-Pelaez's share-measure probability at
`u - i`. For Black-Scholes the continuation off the real axis is free (both sides
of `E[e^{wZ}] = e^{w^2/2}` are entire). For Heston it is where the branch cut of
a complex square root starts to matter, which is the next slice's problem and not
this one's.

**The transform is of the log return `ln(S_T/S_0)`, not of `ln S_T`.** The spot
is carried by the pricer. This keeps the COS truncation range independent of
`S_0`, which is what makes the COS delta and gamma *closed forms* rather than
bumps — see below. The transform of the log spot is one multiplication away:
`E[e^{iu ln S_T}] = e^{iu ln S_0} phi(u)`.

Under Black-Scholes, `phi(u) = exp(i u (r - q - sigma^2/2) T - sigma^2 u^2 T/2)`,
`c1 = (r - q - sigma^2/2) T`, `c2 = sigma^2 T`, `c4 = 0`. The fourth cumulant is
carried rather than dropped because it is the slot Heston fills.

## COS

Expand the density of `Z = ln(S_T/S_0)` in a cosine series on a truncated range
`[a, b]`. The series coefficients are *almost* the characteristic function:
extending the coefficient integral to the whole line (the error of doing so is
the range-truncation error and nothing else) gives

```
A_k ~= (2/(b-a)) Re{ phi(u_k) e^{-i u_k a} },      u_k = k pi / (b - a),
```

so the price of a payoff `g` is

```
V = e^{-rT} sum_k' Re{ phi(u_k) e^{-i u_k a} } V_k,
V_k = (2/(b-a)) Integral_a^b g(S_0 e^z) cos(u_k (z - a)) dz,
```

the prime halving the `k = 0` term. Two elementary integrals give every `V_k`
here: `chi_k(c,d) = Int_c^d e^z cos(u_k(z-a)) dz` and
`psi_k(c,d) = Int_c^d cos(u_k(z-a)) dz`. With `z* = ln(K/S_0)` clipped into the
range,

| payoff | `V_k (b-a)/2` |
|---|---|
| call | `S_0 chi_k(z*, b) - K psi_k(z*, b)` |
| put | `K psi_k(a, z*) - S_0 chi_k(a, z*)` |
| digital call | `cash * psi_k(z*, b)` |
| digital put | `cash * psi_k(a, z*)` |

Range: `[a, b] = [c1 - L w, c1 + L w]` with `w = sqrt(c2 + sqrt(c4))`.

### Convergence in N is Gaussian, not exponential, and the rate is derivable

For a Gaussian log return `|phi(u_k)| = exp(-sigma^2 T u_k^2 / 2)` and
`b - a = 2 L sigma sqrt(T)`, so the `k`-th coefficient is damped by
`exp(-k^2 pi^2 / (8 L^2))` — with **no** `sigma` and **no** `T` in it. The
truncation error is set by the first dropped coefficient, hence

```
log10 err(N) ~ - pi^2 N^2 / (8 L^2 ln 10).
```

Measured slope of `log10(err)` against `N^2` divided by that prediction:

| L | 10 | 20 | 40 |
|---|---|---|---|
| measured / predicted | 1.085 | 1.071 | 1.037 |

Error at the ATM reference point, `L = 10`:

| N | 8 | 12 | 16 | 20 | 24 | 32 | 48 | 64 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| err | 1.97e+00 | 3.49e-01 | 5.38e-02 | 6.34e-03 | 5.41e-04 | 1.37e-06 | 1.23e-13 | 2.66e-14 | 2.66e-14 |

**The slice statement's window was wrong and the tests say so.** It proposed
fitting the order over `N` in {16, 32, 64, 128} and asserting the error at
`N = 256` is below 1e-12. Both are true and neither is informative: the error is
at the floating-point floor by `N = 48`, and the value at `N = 256` is the *same
double* as the value at `N = 64`. Two of those four points measure round-off.
The fit is done on {8 ... 32} instead. A power-law fit there reports order 9.88
with a log-space residual of 1.41; a residual that size is what a least-squares
fit returns when the data is not a power law at all, and it is reported next to
the order rather than instead of it.

### The truncation range has a valley

At `N = 256`, ATM vanilla call:

| L | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 20 |
|---|---|---|---|---|---|---|---|---|
| err | 4.98e-01 | 6.25e-04 | 2.04e-08 | 2.66e-14 | 2.66e-14 | 1.60e-14 | 7.90e-13 | 5.88e-13 |

Two different failures bracket the usable band.

- **Too narrow**: the missing tail mass is an error no term count removes. At
  `L = 4` the price is the same double for `N = 32, 64, ..., 1024`.
- **Too wide**: the `L^2` in the decay denominator means more terms for the same
  accuracy, and the round-off floor itself rises with the range (measured floors
  2.7e-14 / 5.9e-13 / 5.5e-11 at `L = 10 / 20 / 40`, because the call's `chi_k`
  integrates `e^z` and so carries a factor `e^b`).

Fang–Oosterlee's `L = 10` sits in the middle of that valley rather than at its
edge, which is what makes it a good default rather than a lucky one.

### The jump costs the expansion nothing

Slice 6 measured that a cash-or-nothing payoff costs the finite-difference
scheme a **full order** of convergence, because a grid represents the payoff.
COS expands the *density*, which is smooth whatever the payoff is, and the
discontinuity enters through an exact integral. So the digital does not converge
more slowly — it converges **faster**, by a factor of 26 to 54 in the error at
the same `N` (its `V_k` is `cash * psi_k`, bounded, where the call's carries the
`e^z` weight). Measured at the ATM point: `N = 16`, 5.38e-02 against 9.95e-04;
`N = 24`, 5.41e-04 against 1.54e-05; `N = 32`, 1.37e-06 against 5.21e-08.

### Greeks in closed form

Because the range is a statement about the log return, `S_0` appears **only** in
`V_k`. Differentiating in `S_0`, the boundary terms cancel for a vanilla (the
payoff vanishes at `z*`) and

```
dV_k/dS_0  = +/- (2/(b-a)) chi_k(...)
d2V_k/dS_0^2 = (2/(b-a)) (K / S_0^2) cos(u_k (z* - a)).
```

Measured against the closed-form Greeks over 20 cells: delta worst 1.74e-14,
gamma worst 9.54e-18 — four orders of magnitude inside the `~1e-10` the slice
statement asked for. vega, theta and rho are *not* closed forms here (they would
need `d phi / d parameter`, which is model-specific and deliberately not part of
the interface); they are central differences at `h = 1e-05`, chosen from a scan
whose `h^2` regime and round-off floor cross there, and land at `~1e-08`.

## Carr–Madan

The call price is not square integrable in the log strike `k = ln K` — it tends
to `S_0 e^{-qT}` as `k -> -inf`. Damping it by `e^{alpha k}` fixes that, and the
damped price's transform is closed form:

```
psi_T(v) = e^{-rT} phi_S(v - (alpha+1) i) / (alpha^2 + alpha - v^2 + i(2alpha+1) v),
C_T(k)   = (e^{-alpha k} / pi) Integral_0^inf Re[ e^{-i v k} psi_T(v) ] dv.
```

The integral can be sampled on `v_j = j eta` and turned into one FFT, provided
the log strikes sit on the reciprocal grid `lambda = 2 pi / (N eta)`; or it can
be integrated at the strike actually asked for. This package does both.

### The FFT variant's error is interpolation, not integration

Same FFT output, two readings, at the recommended `N = 4096, eta = 0.25`:

| eta | lambda | on a grid node | interpolated |
|---|---|---|---|
| 0.05 | 3.07e-02 | 1.3e-13 | 6.1e-03 |
| 0.10 | 1.53e-02 | 1.0e-13 | 2.7e-03 |
| 0.25 | 6.14e-03 | 2.2e-07 | 6.3e-04 |
| 0.50 | 3.07e-03 | 2.7e-03 | 2.7e-03 |

At fixed `N`, `eta` sets the transform-grid resolution and `lambda = 2pi/(N eta)`
sets the strike-grid resolution, so they move in opposite directions and there is
no setting where both columns are small. `N` is the only way out and it costs an
FFT of length `N`. The engine reports `meta["on_grid"]` so a caller can tell
which of the two numbers they were handed.

### Carr and Madan's own Simpson weighting is the dominant error there

On-grid error at `atm_1y`, `N = 4096`, Simpson weighting against plain trapezoid
weighting of the same samples:

| eta | 0.10 | 0.25 | 0.50 | 1.00 |
|---|---|---|---|---|
| Simpson | 1.03e-13 | 2.17e-07 | 2.68e-03 | 2.89e-01 |
| trapezoid | 1.03e-13 | 7.46e-14 | 6.51e-07 | 8.07e-03 |
| ratio | 1.0 | 2.9e+06 | 4.1e+03 | 3.6e+01 |

Why is in the quadrature section below. Simpson stays the default because it is
what the method *is*; `FourierConfig.fft_weights` exposes both.

### The damping parameter: the predicted failure does not happen

The slice statement expected a failure at large `alpha` over
{0.5, 1, 1.5, 3, 10}. There is none: every cell in that range is at or below
5.5e-10 at all three moneyness levels, and all but two are at ~1e-14. Under
Black-Scholes `E[S_T^{alpha+1}]` is finite for **every** alpha, so the
integrability condition that usually limits the damping never binds. The two
real failures are outside the named range, and they are different failures.

**Small alpha is a resolution failure.** As `alpha -> 0` the denominator tends to
`-v^2 + i v`, which vanishes at `v = 0`: the integrand grows a spike of height
`~1/alpha` and width `~alpha` at the left endpoint, and a uniform grid cannot
see it. Measured at the ATM point: 7.4e+00 at 0.05, 4.7e-01 at 0.1, 1.5e-04 at
0.25, 2.3e-10 at 0.5.

**Large alpha is catastrophic cancellation, and it is moneyness-dependent.** The
price is `e^{-alpha ln K}` times an integral of size `~E[S_T^{alpha+1}]`, so the
relative round-off of the product is bounded by

```
(S_0/K)^alpha * exp(alpha^2 c2 / 2) * S_0 / price * eps.
```

Measured error, with that bound in brackets:

| point | alpha = 20 | alpha = 30 | alpha = 40 |
|---|---|---|---|
| `S/K = 1.000` | -1.03e-13 (6e-12) | -1.60e-09 (1e-07) | -3.83e-03 (2e-01) |
| `S/K = 0.909` | +1.63e-13 (7e-12) | -3.36e-09 (3e-07) | +3.14e-02 (2e+00) |
| `S/K = 1.333` | +3.04e-04 (1e-02) | +4.16e+10 (4e+12) | -2.41e+31 (3e+32) |

The in-the-money point is unusable by `alpha = 20` while the at-the-money one is
still exact. The bound predicts that ordering, and every measured error sits one
to two orders of magnitude inside it.

## Lewis

Transform the *payoff* rather than the price and there is no parameter to defend.
With `w(x) = (e^x - K)^+`, `w_hat(z) = -K^{iz+1}/(z^2 - iz)` for `Im z > 1`.
Parseval against `phi_S(-z)`, then push the contour down to `Im z = 1/2`, picking
up the pole at `z = i` whose residue is exactly the `S_0 e^{-qT}` term:

```
C = S_0 e^{-qT}
  - (sqrt(K) e^{-rT} / pi) Integral_0^inf Re[ e^{-i u ln K} phi_S(u - i/2) ] / (u^2 + 1/4) du.
```

On that line `z(z-i) = u^2 + 1/4`: real, positive and bounded below by 1/4. The
contrast with Carr–Madan is the point of having both — one has a denominator that
can be driven toward a pole by a badly chosen parameter, the other has no
parameter and a denominator that cannot vanish.

## Gil-Pelaez

Invert the characteristic function to a distribution function directly:

```
Q(X > k) = 1/2 + (1/pi) Integral_0^inf Re[ e^{-iuk} phi(u) / (i u) ] du,
C = S_0 e^{-qT} Pi_1 - K e^{-rT} Pi_2,
```

with `Pi_1` the same probability under the share measure, whose transform is the
argument shift `phi(u - i) / phi(-i)`. The apparent `1/(iu)` singularity is
removable: `Re[w/i] = Im[w]`, and the numerator vanishes linearly at the origin
with slope `c1 - k`. `phi(-i)` is the forward, which is a one-line arithmetic
check on any model claiming to satisfy the interface.

The cash-or-nothing digital is `cash * e^{-rT} * Pi_2` with no further work,
which is why this is the method that prices a digital as naturally as a vanilla.

## Accuracy, side by side

Worst absolute error over the five specification points and both kinds, at the
package defaults:

| method | vanilla | digital | notes |
|---|---|---|---|
| COS, `N = 256, L = 10` | 2.91e-12 | 2.78e-16 | delta 1.74e-14, gamma 9.54e-18 |
| Carr–Madan, FFT | 6.3e-04 | — | interpolation; 2.2e-07 on a grid node |
| Carr–Madan, quadrature | 1.93e-10 | — | trapezoid, 512 intervals |
| Lewis | 2.13e-14 | — | adaptive `quad` |
| Gil-Pelaez | 1.42e-14 | 1.11e-16 | `Pi_1`, `Pi_2` match `N(d1)`, `N(d2)` to 1.7e-16 |

Only COS computes a **put** from its own payoff coefficients; the other three
transform the call, so their put is `C - S e^{-qT} + K e^{-rT}` and put-call
parity holds by construction. The tests say so explicitly: parity is evidence
for COS (residual 2.4e-14 to 2.9e-12 across the points) and arithmetic for the
other three.

## The Chapter 6 quadrature rules, finally on a pricing integral

`qpl.numerics.quadrature` has held composite trapezoid, composite Simpson and
Gauss–Legendre since the Fusai Chapter 6 lab, tested only against smooth
finite-interval integrands where they are order 2, order 4 and spectrally
accurate. The Carr–Madan direct-quadrature variant is the first time they drive
a price. They behave nothing like the smooth-function test suggests.

Error at the ATM point, `alpha = 1.5`, reach 60:

| n | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| trapezoid | +2.64e+01 | +7.35e+00 | +6.39e-01 | +4.30e-03 | +1.85e-07 | +1.60e-14 |
| Simpson | +1.41e+01 | +9.91e-01 | -1.60e+00 | -2.07e-01 | -1.43e-03 | -6.18e-08 |
| Gauss–Legendre | +9.34e-01 | -1.00e-01 | -8.59e-05 | +2.22e-11 | +2.90e-12 | +9.68e-13 |

Fitted orders over the above-floor points: **6.49** (trapezoid, nominal 2),
**4.86** (Simpson, nominal 4), **9.31** (Gauss–Legendre) — each with a log-space
residual above 2.4, i.e. none of them is a power law.

**Why the trapezoid rule is spectral here.** `psi_T(-v) = conj(psi_T(v))`,
because the damped call price is real, so the integrand is an **even** function
of `v`. Every odd derivative vanishes at `v = 0`, and the Gaussian decay kills
the integrand and all its derivatives long before the truncation point. The
Euler–Maclaurin correction terms are exactly those two sets of boundary
derivatives, so every one of them is zero and only the aliasing term is left —
which for an analytic integrand decays faster than any power of the step. This
is a property of the integrand, not of the rule, and the Fusai lab's
smooth-function test could never have revealed it: `exp(x)` on `[0, 1]` is
perfectly smooth and perfectly non-even.

**Why Simpson's rule is six decimal orders worse.** On a uniform grid, composite
Simpson at `n` intervals is *exactly* the Romberg combination
`S_n = (4 T_n - T_{n/2}) / 3` (checked to 3.6e-15). So its error is
`(4(T_n - I) - (T_{n/2} - I))/3`. That combination is worth having when the
trapezoid error is `O(h^2)` and the leading terms cancel. Here `T_n - I` is
already negligible against `T_{n/2} - I`, so

```
S_n - I ~= -(T_{n/2} - I) / 3
```

— Simpson throws away the accurate rule and keeps a third of the inaccurate one.
Measured: -2.071e-01 against a prediction of -2.129e-01 at `n = 64`, and exact
agreement to three figures at 128 and 256. The same mechanism is what makes
Carr and Madan's Simpson-weighted FFT worse than a trapezoid-weighted one.

`FourierConfig.quadrature` therefore defaults to `"trapezoid"`, and that default
is a measurement rather than a habit.

## QuantLib

`AnalyticEuropeanEngine` is the price and Greek oracle: worst residual 2.91e-12
(COS), 2.32e-10 (Carr–Madan quadrature), 3.95e-14 (Lewis), 1.42e-14
(Gil-Pelaez); 6.66e-16 and 5.55e-16 for the two digital methods against
`CashOrNothingPayoff`; and for the five COS Greeks 1.74e-14, 1.39e-17, 5.61e-08,
1.18e-07, 1.38e-07.

The slice statement said QuantLib has no Fourier engine that can price a
Black-Scholes European. It does. **`AnalyticHestonEngine` on a Heston model with
`v0 = theta = sigma^2` and vol-of-vol driven to 1e-08 reproduces
`AnalyticEuropeanEngine` to 0.0** at every point tested — a genuine
transform-against-transform oracle, two contour integrals of two differently
written characteristic functions.

**`COSHestonEngine` cannot be used that way, and how it fails is the most useful
thing this slice learned for the next one.** Pushed toward the same limit it
*diverges*:

| vol-of-vol | 1e-02 | 1e-04 | 1e-06 | 1e-08 |
|---|---|---|---|---|
| error | -3.8e-04 | -5.0e-08 | +7.1e-05 | +8.9e-02 |

and 200, 1000 and 4000 cosine terms give the same price to 1e-15. A COS error
that does not respond to `N` is a **truncation-range** error, not a series
error — and Heston's cumulant formulas, which is where a COS engine gets its
range, carry the vol-of-vol in denominators. This package's COS, deriving its
range from the cumulants of the law it is actually pricing, is at 2.9e-12 on the
same problem. The consequence for the Heston slice is concrete: a Heston COS
implementation has to be tested at small vol-of-vol *specifically*, because that
is the corner where the range rule, not the expansion, decides the answer.

## Sources

- Carr, P. and Madan, D. (1999). "Option valuation using the fast Fourier
  transform". *Journal of Computational Finance* 2(4), 61–73. The damped call
  transform, the FFT strike grid, the Simpson weighting.
- Fang, F. and Oosterlee, C. W. (2008). "A novel pricing method for European
  options based on Fourier-cosine series expansions". *SIAM Journal on
  Scientific Computing* 31(2), 826–848. The COS method, the cumulant-based
  truncation range, the vanilla payoff coefficients.
- Gil-Pelaez, J. (1951). "Note on the inversion theorem". *Biometrika* 38(3–4),
  481–482. The inversion formula for the distribution function.
- Heston, S. L. (1993). "A closed-form solution for options with stochastic
  volatility". *Review of Financial Studies* 6(2), 327–343. The `Pi_1`/`Pi_2`
  form. Cited here for that decomposition only; the model itself is the next
  slice.
- Lewis, A. L. (2001). "A simple option formula for general jump-diffusions and
  other exponential Lévy processes". The fundamental transform and the strip.
- Schmelzle, M. (2010). "Option pricing formulae using Fourier transform: theory
  and application". Survey; cited for the view that every one of these methods
  touches a model only through its transform.
- Fusai, G. and Roncoroni, A. *Implementing Models in Quantitative Finance*,
  chapter 6 §6.8 ("Pricing using characteristic functions") and the chapter 6
  quadrature material, as the map of what the private lab covered. The rules in
  `qpl.numerics.quadrature` come from that chapter; the measurements of how they
  behave on a pricing integral are made here.
