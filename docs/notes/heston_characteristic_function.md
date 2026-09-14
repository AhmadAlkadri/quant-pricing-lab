# Heston through a characteristic function: the branch, the cumulants, the range

Slice 14 built four transform engines and one interface, and claimed that a
second model would cost nothing but the model. This slice is the test of that
claim. `HestonModel` implements two methods — `characteristic_function` and
`log_return_cumulants` — and two `register(...)` lines wire it into the
dispatcher; no pricer changed. Everything the slice learned is about the
*model*, and it is all about the three places a Heston transform can go wrong:
the branch of a complex logarithm, the cumulants that set a truncation range,
and the moment explosion that bounds a damping parameter.

Everything below is measured in this repository. The sources at the end are
cited for the model and for the failure modes; no expression, number or table is
reproduced from any of them.

## The model and its affine solution

Under the pricing measure, with `x_t = ln(S_t / S_0)`:

```
dx = (r - q - v/2) dt + sqrt(v) dW1
dv = kappa (theta - v) dt + xi sqrt(v) dW2,      d<W1, W2> = rho dt
```

The pair is affine, so `E[e^{i u x_T}] = exp(C(u,T) + D(u,T) v_0)`. Substituting
that ansatz into the backward equation and collecting powers of `v` gives two
ODEs with `C(0) = D(0) = 0`:

```
D' = (xi^2/2) D^2 + b D - c,     b = rho xi i u - kappa,  c = (u^2 + i u)/2
C' = (r - q) i u + kappa theta D
```

`D` is a constant-coefficient Riccati equation. Its stationary roots are
`D_pm = (-b +- d)/xi^2` with

```
d = sqrt(b^2 + 2 xi^2 c) = sqrt((kappa - rho xi i u)^2 + xi^2 (u^2 + i u)),
```

and the solution through `D(0) = 0`, with `beta = kappa - rho xi i u = -b` and
`g = (beta - d)/(beta + d)`, is

```
D(tau) = ((beta - d)/xi^2) (1 - e^{-d tau}) / (1 - g e^{-d tau})
C(tau) = (r - q) i u tau
         + (kappa theta / xi^2) [ (beta - d) tau - 2 ln((1 - g e^{-d tau})/(1 - g)) ]
```

## The little Heston trap, and why the usual demonstration does not work

`d` is a complex square root, so the formula above is two formulas. Heston's own
paper writes the equivalent expression with `+d`, `1/g` and `e^{+d tau}`; the
two are algebraically the same function and numerically are not.

Take `d` on the **principal branch**, so `Re(d) >= 0`. Then `|g| <= 1` (because
`Re(beta) = kappa > 0` for real `u`), `|e^{-d tau}| <= 1`, so
`|g e^{-d tau}| <= 1` and `1 - g e^{-d tau}` stays in the right half plane: its
argument never leaves `(-pi/2, pi/2)` and so the principal branch of `ln` is the
continuous one. Measured worst `|arg(1 - g e^{-dT})|` over four parameter sets,
`T` in `{1, 30}` and 4001 values of `u`: **0.0852** against `pi/2 = 1.5708`.

With the other sign the exponential grows, `|1/g| >= 1`, and the logarithm's
argument **winds** around the origin. Each winding adds `2 pi i`, and `C`
multiplies that by `-2 kappa theta / xi^2`, so the transform is multiplied by

```
exp(-4 pi i kappa theta / xi^2).
```

That factor is **1 whenever `2 kappa theta / xi^2` is an integer** — and on the
parameter set the six published reference values are quoted at
(`v0 = 0.04, kappa = 4, theta = 0.25, xi = 1, rho = -0.5`) it is exactly 2. The
trap is invisible on the reference set at every maturity out to 30 years:

| model | `2 kappa theta / xi^2` | T=1 | T=1.2 | T=2 | T=5 | T=10 | T=30 |
|---|---|---|---|---|---|---|---|
| reference | 2.00 | 1.1e-14 | 1.2e-13 | 1.2e-13 | 2.1e-11 | 3.4e-10 | 4.9e-07 |
| trap | 2.50 | 1.6e-08 | 2.2e-03 | **8.6e-01** | 5.3e+02 | 3.5e+04 | 8.6e+07 |

(relative error of the COS price computed with the original branch against the
COS price computed with the stable one, same range, same term count, same
model.) The two rows differ in `theta` **alone**, so `d`, `g` and the branch
windings are identical — only the multiplier changes. The first crossing sits at
the same `u` in both rows, and `u * T` is nearly constant: 19.4, 6.3, 5.6, 5.2,
5.1 at `T = 1, 2, 3, 5, 10`, which is why maturity is the axis the failure moves
along.

QuantLib's `AnalyticHestonEngine` offers `ComplexLogFormula.Gatheral` and
`ComplexLogFormula.BranchCorrection`, and it is tempting to read a difference
between them as a trap sighting. There is none: at matched Gauss-Legendre order
the two agree to **4.2e-14** on every cell tested. QuantLib's default is already
branch-safe, so the trap has to be demonstrated against a deliberately unstable
implementation — which is what
`qpl.models.heston.heston_characteristic_function_original_branch` is for.

## Writing the stable form so that it survives `xi -> 0`

The textbook transcription multiplies a bracket that is `O(xi^2)` by
`kappa theta / xi^2`, and computes `beta - d` as a literal subtraction that
loses `eps |beta|`. The transform therefore carries an error of order
`eps kappa^2 theta T / xi^2`. Measured `max_u |phi_xi(u) - phi_0(u)|` at
`v0 = theta = 0.04, kappa = 1, rho = 0, T = 1`:

| xi | 1e-03 | 1e-04 | 1e-05 | 1e-06 | 1e-07 | 1e-08 |
|---|---|---|---|---|---|---|
| literal | 1.15e-06 | **1.18e-08** | 1.87e-07 | 1.21e-05 | 1.30e-03 | 1.35e-01 |
| this package | 1.15e-06 | 1.15e-08 | 1.15e-10 | 1.15e-12 | 1.16e-14 | 2.24e-16 |

— a minimum and then divergence, which is the same *shape* as Slice 14's
QuantLib `COSHestonEngine` finding with a different cause (there the truncation
range, here the affine coefficients).

Two rewrites remove it. `beta^2 - d^2 = -xi^2 (u^2 + i u)` exactly, so

```
(beta - d)/xi^2 = -(u^2 + i u)/(beta + d),      g = xi^2 (beta - d)/(beta + d)^2
```

have no subtraction and no `1/xi^2`; and the logarithm is taken with `log1p`
(hand-written for complex arguments — `np.log1p` returns a zero real part for
`1e-18 + 1e-18j`), so the difference of two `O(xi^2)` logarithms divided by
`xi^2` stays bounded. The resulting column falls monotonically for six decades
and the order in `xi` is exactly what theory says: **2.00 at `rho = 0`** and
**1.00 at `rho != 0`**, because the leading correction is the covariance term
`rho xi` entering `beta`.

The price inherits that: order 1.996 / 1.997 / 1.991 at `K = 80 / 100 / 120`
with `rho = 0`, and 0.999 / 0.951 / 1.030 at `K = 80 / 90 / 120` with
`rho = -0.5`. The slice statement guessed `O(xi^2)`; that holds only at
`rho = 0`. The `rho xi` term is **odd in log-moneyness**, so it changes sign
across the strike (`+1.62e-01` at `K = 80`, `-1.75e-01` at `K = 110` at
`xi = 0.1`) and vanishes near the money, where a fitted order is 0.32 with a
log-space residual of 0.33 — meaningless, and pinned as such.

## Cumulants, and the one that is reported as zero

The COS truncation range `[c1 - L w, c1 + L w]` with `w = sqrt(c2 + sqrt(c4))`
needs the cumulants of `x_T`. They are derived here from the **variance
dynamics**, not from `ln phi`, which is what makes the finite-difference check
in the tests a real cross-check rather than a restatement.

Write `I = int_0^T v_s ds`, `M = int_0^T sqrt(v_s) dW1_s`, so
`x_T = (r-q)T - I/2 + M`. Integrating the variance dynamics and swapping the
order of integration,

```
I - E[I] = (xi/kappa) int_0^T (1 - e^{-kappa(T-s)}) sqrt(v_s) dW2_s,
```

so with `m(s) = E[v_s] = theta + (v0 - theta) e^{-kappa s}` and
`g(s) = 1 - e^{-kappa(T-s)}`, the Itô isometry gives `E[I] = int m`,
`Var(I) = (xi^2/kappa^2) int g^2 m`, `E[IM] = (xi rho/kappa) int g m` and
`Var(M) = E[I]`. Hence

```
c1 = (r - q) T - vbar/2
c2 = vbar - (xi rho/kappa) J1 + (xi^2/(4 kappa^2)) J2
```

with `vbar = int m`, `J1 = int g m`, `J2 = int g^2 m`, all elementary. **`xi`
appears only in numerators**, which is the direct repair of the Slice 14 oracle
finding: QuantLib's `COSHestonEngine` diverges as `xi -> 0` because its range
formulas carry the vol-of-vol below the line, and at `xi = 1e-06` this package
reproduces the Black-Scholes price to better than 1e-09 at the defaults.

`c4` is reported as **zero**, following Fang and Oosterlee's own treatment, and
that is not harmless. Measured by a fourth difference of `ln phi` at `T = 1`:

| set | Feller number | `c2` | `c4` | `sqrt(c4)/c2` | range too narrow by |
|---|---|---|---|---|---|
| reference | 4.00 | 0.21786 | 5.26e-02 | 1.05 | 1.43 |
| Feller-violating | 0.08 | 0.05767 | 1.56e-01 | 6.85 | **2.80** |

The consequence is a **range** error, which is a flat column in the term count:

| `L` | N=512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| 10 | 9.12e-04 | 9.13e-04 | 9.13e-04 | 9.13e-04 | 9.13e-04 |
| 14 | 4.33e-05 | 2.75e-05 | 2.75e-05 | 2.75e-05 | 2.75e-05 |
| 20 | 3.84e-04 | 1.32e-06 | 1.28e-07 | 1.28e-07 | 1.28e-07 |
| 28 | 2.42e-03 | 1.56e-04 | 1.30e-07 | 1.22e-10 | 1.22e-10 |

(Feller-violating set, `T = 1`, ATM call, against a Lewis integral of the same
transform.) The repair is the derived factor: `L = 10 x 2.80 = 28`, which is
what `HESTON_TRUNCATION_L_FELLER_VIOLATED` publishes. QuantLib's own
`COSHestonEngine` has the same failure on the same set — -9.92e-03 / -1.096e-02
/ -1.096e-02 at 200 / 800 / 3200 terms for `L = 10`, repaired to -3.0e-09 only
at `L = 32` — so this is a property of the range rule, not of this
implementation.

## The COS call and the COS put are not the same problem

This is the slice's least expected result. The call's payoff coefficient is
`chi_k(z*, b)`, which carries `e^b` with `b = c1 + L sqrt(c2)`; the put's is
`chi_k(a, z*)`, which carries `e^{z*} = K/S_0` and nothing else. So widening the
range amplifies round-off in the **call** exponentially and does nothing at all
to the put, while narrowing it costs the **put** left-tail mass (with `rho < 0`
the density is left-skewed) and costs the call almost nothing.

Under Black-Scholes both errors are tiny and the asymmetry is invisible; Slice
14 measured a COS parity residual of 2.4e-14. Under Heston at the package
defaults it is **2.70e-08**, identical at every strike — the put's missing mass
— and falls 2.28e-06 → 2.70e-08 → 3.13e-10 → 1.34e-11 as `L` goes 8 → 10 → 12
→ 14.

There is normally a window between the two errors. At
`v0 = 0.04, kappa = 0.5, theta = 0.04, xi = 1, rho = -0.9, T = 10` there is
**not**:

| `L` | 8 | 10 | 14 | 20 | 28 | 36 | 50 |
|---|---|---|---|---|---|---|---|
| call | 7.2e+00 | 1.3e+01 | 4.9e+01 | 4.1e+02 | 1.6e+05 | 9.4e+08 | 4.4e+15 |
| put | 3.1e-02 | 1.0e-02 | 1.2e-03 | 4.6e-05 | 1.5e-06 | 2.1e-06 | 2.1e-06 |

The call is monotone **upward** with no cell below 1. The recipe is therefore:
**price the put by COS at a generous `L` and take the call by parity.** Worst
residual against QuantLib over 24 cells that way: 7.66e-07, against 4.4e+15 for
the direct call at the same settings.

The engine is deliberately **not** changed to do that automatically. COS
computing its put from its own payoff coefficients is the only reason put-call
parity is a *check* on those coefficients here rather than an identity of the
implementation (the other three methods transform the call and build the put by
parity, so for them it proves nothing). That evidence is worth more than the
convenience, and the recipe is one line in caller code.

## Carr–Madan is now bounded by moment explosion

Under Black-Scholes `E[S_T^{alpha+1}]` is finite for every `alpha`, so Slice 14
could not produce the integrability failure its own slice statement predicted;
the failures it measured were resolution at small `alpha` and cancellation at
large `alpha`. Under Heston the constraint binds, and it is computable from the
same `d` and `g`.

Substituting `u = -i w` gives `d^2 = Delta(w) = (kappa - rho xi w)^2 + xi^2 w(1-w)`,
a downward parabola in `w`. Where `Delta >= 0` the moment is finite for all `T`.
Where `Delta < 0`, `d = i D` with `D = sqrt(-Delta)`, `|g| = 1`, and
`1 - g e^{-d tau} = 1 - e^{-i(2 psi + D tau)}` with `psi = atan2(D, beta)` first
vanishes at

```
T*(w) = 2 (pi - psi) / D.
```

`critical_moment(T)` is the root of `T*(w) = T`. On the reference set at
`T = 1` it is `w = 11.6905`, i.e. `alpha_max = 10.6905`. Measured error of the
direct-quadrature Carr–Madan against a Lewis integral:

| alpha | 0.5 | 1.5 | 3 | 5 | 8 | 10 | 10.6 | 10.65 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| `T*(alpha+1)` | inf | inf | inf | inf | inf | 1.2169 | 1.0233 | 1.0103 | 0.9287 | 0.7597 |
| error | 1.6e-10 | 1.0e-09 | 1.2e-09 | 6.0e-09 | 2.3e-08 | 1.7e-08 | 3.1e-08 | **2.7e+02** | 6.1e+03 | 6.1e+03 |

The method is exact while the pole stays off the contour and fails once the
explosion time is within about 1% of the maturity. Unlike Slice 14's
cancellation failure, the onset is the **same** `alpha` at every strike, because
a moment explosion is a property of the law and not of the contract.

Separately, the default integration reach is wrong for a fat-tailed law:
`u_max = 12/sqrt(c2)` is a statement about a Gaussian, and on the
Feller-violating set it gives 50 where the integrand needs 500. Default 4.9e-02;
`u_max = 200` gives 8.3e-05, 500 gives 3.6e-09 and 1000 gives 1.0e-13.

## The smile

`qpl.engines.fourier.smile` composes the transform price with the Brent inverter
already in `qpl.engines.analytic.black_scholes`. Measured on the reference set:

- **`rho` is the sign of the skew.** At `rho = -0.5` the implied volatility is
  strictly decreasing over `K = 70 ... 140` at every maturity; the
  at-the-forward skew `d sigma / d ln(K/F)` is -0.2235 / -0.0943 / -0.0231 at
  `T = 0.25 / 1 / 5`, against +0.2248 / +0.0972 / +0.0245 at `rho = +0.5` — the
  same magnitudes within 6%, sign reversed. At `rho = 0` the skew is exactly 0
  and the smile is symmetric but not flat (`xi` makes curvature, `rho` makes
  slope).
- **It flattens with maturity.** |skew| 0.2235, 0.1508, 0.0943, 0.0537, 0.0231,
  0.0118 at `T = 0.25 ... 10`. A power-law fit gives exponent 0.804 with a
  log-space residual of 0.097 — reported as a summary, not as a theorem, and
  measurably not the `1/T` of a large-maturity expansion.
- **At-the-forward implied variance runs from `v0` to `theta`**: 0.0434 → 0.2326
  with `v0 = 0.04, theta = 0.25`, and 0.2448 → 0.0387 with the two swapped, both
  monotone over `T = 0.01 ... 30`. It **undershoots `theta` from both
  directions**, which is the smile's own curvature rather than incomplete
  convergence.

One practical warning belongs with the smile and not with the model: a long
maturity is the same axis as a large `L` for the COS call, so the package
default `L = 10` is off by 1.2e+02 on a price of 13.3 at `T = 100` while
`L = 8` is at 2.0e-03.

## Against QuantLib

`AnalyticHestonEngine` with adaptive Gauss-Lobatto at 1e-13, two parameter sets
either side of Feller, `T = 1` and `T = 10` exact under `Actual365Fixed`, three
strikes, calls and puts — 24 cells. Worst residual:

| method | worst over 24 cells | Feller-satisfying only |
|---|---|---|
| Lewis (defaults) | 2.38e-10 | 2.3e-14 |
| Gil-Pelaez (defaults) | 1.24e-11 | 2.8e-14 |
| Carr–Madan (`u_max = 1000`) | **1.00e-13** | 1.0e-13 |
| COS put + parity (`L = 28`) | 7.66e-07 | 2.1e-14 |
| COS call, best `L` | > 1e+00 on one cell | 1.4e-12 |

The slice statement hoped for ~1e-08. The practical conclusion is that **Lewis
and Gil-Pelaez are the two methods with no parameter to get wrong** — no
truncation range, no damping, just an adaptive contour integral — and they are
the two that need no per-parameter tuning anywhere in this table.

QuantLib's own default engine (`AnalyticHestonEngine(model, 144)`, a fixed
144-point Gauss-Laguerre rule) is 2.9e-08 off on the Feller-violating set and is
not used as the reference here.

## Sources

- Heston (1993), "A closed-form solution for options with stochastic
  volatility", *Review of Financial Studies* 6(2), 327–343 — the model and the
  original branch choice.
- Albrecher, Mayer, Schoutens and Tistaert (2007), "The little Heston trap",
  *Wilmott Magazine* (January 2007), 83–92, section 3 — the stable form and the
  `|g| <= 1` argument.
- Lord and Kahl (2010), "Complex logarithms in Heston-like models",
  *Mathematical Finance* 20(4), 671–694, section 2 and Theorem 3.1 — the
  rotation count, and the proof that the principal branch suffices for Heston.
- Fang and Oosterlee (2008), *SIAM J. Sci. Comput.* 31(2), 826–848, appendix —
  the rule that the COS range is built from `c1`, `c2` and `c4`.
- Andersen and Piterbarg (2007), "Moment explosions in stochastic volatility
  models", *Finance and Stochastics* 11, section 3 — the criterion re-derived
  above for when `E[S_T^w]` is infinite.
- Gatheral (2006), *The Volatility Surface*, chapter 2 — the same transform in
  his notation, and the qualitative smile statements measured above.
- The six reference prices are attributed by the QuantLib test suite to Alan
  Lewis' posting on the Wilmott forums; they are used as fixtures with that
  citation and nothing else is taken from either.

## Where the numbers live

`tests/test_heston_model.py` (the transform and its cumulants),
`tests/test_heston_fourier.py` (prices, the range, the trap, the damping bound),
`tests/test_heston_smile.py` (the smile), `tests/cases/test_heston_cases.py`
(the rows in `qpl.cases.heston`),
`tests/oracle/test_heston_vs_quantlib.py`, `examples/heston_smile.py`
(`--case smile | cos | trap | alpha`).
