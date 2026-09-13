# PDE Greeks from the grid, and Rannacher start-up

## Where each Greek comes from

`qpl.pricing.greeks(..., method="pde")` returns all five Greeks. With the
default `PDEConfig(greeks_method="grid")` they come from three different places
and the `meta` dictionary says which.

**Delta and gamma** are the usual second-order central stencils on the finished
grid, `(V[i+1] - V[i-1]) / (2 ds)` and `(V[i+1] - 2 V[i] + V[i-1]) / ds^2`.
They are `O(ds^2)` wherever the solution is smooth, which at `tau = T` it is:
whatever the payoff did at `tau = 0`, diffusion has since smoothed it.

**The spot is usually not a node.** With `strike_alignment="midpoint"` the
strike sits at a half-integer number of spacings from the origin by
construction, so an at-the-money spot sits at a half-integer position too -
exactly between two nodes, the worst case for interpolation. The fix is to
interpolate the *Greek*, not the price. `delta_i` approximates `V'(S_i)` to
`O(ds^2)`; linear interpolation of a smooth function between two nodes `ds`
apart adds a term proportional to `ds^2` times its second derivative. Both are
second order, so nothing is lost. Interpolating the price to shifted points and
differencing there would lose it, because the interpolation error would then be
divided by `ds^2`.

**Theta** is the PDE identity at the spot,

```
V_t = -( 1/2 sigma^2 S^2 V_SS + (r - q) S V_S - r V ).
```

Everything on the right is already second order, so theta is too. The obvious
alternative - difference the last two time levels - is a one-sided difference
in calendar time and is `O(dt)` however accurate the scheme is. It is computed
anyway and returned as `meta["theta_backward_difference"]`, as a cross-check
rather than a second opinion.

**Vega and rho** are bump-and-revalue, because a one-factor spot grid knows
nothing about `sigma` or `r`. They inherit the *grid's* error rather than a
stencil's: they are no better than the price. They are much better behaved than
the tree's, though, because bumping `sigma` or `r` does not move the grid -
`ds`, `s_max` and the strike alignment depend only on the spot and the strike -
so the two solves share their discretisation error and most of it cancels.

## Measured orders, smooth case

`S = K = 100, r = 5%, q = 0, sigma = 20%, T = 1`, European call, aligned grid,
`n_s = n_t = n` over `(50, 100, 200, 400, 800)`. Fitted slope of `log|error|`
on `log(1/n)`, with the log-space RMS residual in brackets.

| Greek | `time_stepping="theta"` | `time_stepping="rannacher"` |
|---|---:|---:|
| delta | 2.001 (0.021) | 2.001 (0.021) |
| gamma | 2.064 (0.035) | 2.067 (0.036) |
| theta | 2.024 (0.018) | 2.024 (0.019) |

Error at `n = 800`: delta 3.59e-05, gamma 4.31e-07, theta 2.67e-04.

**The two time steppings are indistinguishable here.** That was not expected;
see "What contradicted the slice statement" below.

Theta, both ways, with the spatial grid held fixed and fine (`n_s = 800`,
aligned) and only `n_t` refined, Rannacher:

| `n_t` | PDE identity | last two time levels |
|---:|---:|---:|
| 25 | -1.076e-03 | -4.432e-02 |
| 50 | -8.362e-05 | -2.130e-02 |
| 100 | +1.789e-04 | -1.031e-02 |
| 200 | +2.440e-04 | -4.944e-03 |

The one-sided difference fits order **1.054** with residual **0.0020** - a
clean first-order sequence, 20x to 200x worse than the identity at every level.
The identity's own column is not fitted: by `n_t = 50` it has already hit the
spatial floor of that grid and is no longer measuring anything temporal.

## The Crank-Nicolson start-up pathology

Write the semi-discrete problem in the eigenbasis of the space operator. One
theta step multiplies a mode by

```
R(z) = (1 + (1 - theta) z) / (1 - theta z),    z = -lambda dt.
```

At `theta = 1/2` this is `(1 + z/2) / (1 - z/2)`. Its modulus is below one for
every `z < 0`, which is why Crank-Nicolson is unconditionally stable - but it
tends to **-1** as `z -> -infinity`. The stiffest modes are not damped at all;
they are flipped in sign at every step and kept forever. A vanilla payoff's
kink is exactly high-frequency content in that basis, so it survives the whole
march as a sign-alternating ripple.

The price is an average over the grid and hardly notices. Gamma is a *second*
difference of neighbouring values, so it differences the ripple against itself
and is dominated by it. Delta, a first difference, mostly cancels it.

A fully implicit step has `R(z) = 1 / (1 - z) -> 0`. Four such steps of `dt/2`
damp a mode by `(1 + |z|/2)^-4`, which kills the ripple while contributing only
`O(dt^2)` to the global error. That trade-off - enough damping to remove the
ripple, little enough to keep second order - is the content of the analysis in
Giles and Carter (2006). Rannacher (1984) is the original construction.

**What `time_stepping="rannacher"` does here:** the first *two* nominal steps
are replaced by *four* fully implicit steps of `dt/2`, then `cfg.theta`
resumes. Halving is exact in binary floating point, so `4 * (0.5 dt)` is
`2 dt` to the last bit and the march resumes at exactly the level it would
otherwise have reached; total time is still `n_t dt`, and `n_t + 2` steps are
taken. Giles and Carter describe the same four steps as "two half steps,
twice". Two half steps would damp by `(1 + |z|/2)^-2` only.

### When it bites

It needs `dt` large relative to `ds^2` *at the strike*. On the usual
`n_s = n_t = n` path it never happens: `dt` shrinks as fast as the spacing, so
`lambda dt` at the strike stays around one and Crank-Nicolson damps the stiff
modes perfectly well. It happens whenever the spatial grid is the expensive
resolution and the time grid is not - which is the normal situation when
someone refines space to sharpen the Greeks.

`T = 0.05`, `S = K = 100`, `n_s = 80 n_t`, closed-form gamma 0.08893343.
Relative error `(gamma_pde - gamma_bs) / gamma_bs`:

| `n_t` | `n_s` | CN, unaligned | Rannacher, unaligned | CN, aligned | Rannacher, aligned |
|---:|---:|---:|---:|---:|---:|
| 5 | 400 | -2.591e-01 | +1.303e-02 | -3.949e-02 | +2.159e-03 |
| 10 | 800 | +5.570e-01 | +4.161e-03 | +5.887e-02 | +7.916e-04 |
| 20 | 1600 | +1.134e+00 | +9.893e-04 | +1.258e-01 | +1.850e-04 |
| 40 | 3200 | +2.277e+00 | +2.416e-04 | +2.538e-01 | +4.467e-05 |

Fitted orders: **-1.043** (residual 0.018) and **-0.915** (0.089) for plain
Crank-Nicolson - *negative*, i.e. refinement makes it worse at rate `n` - and
**1.933** (0.076) and **1.888** (0.118) for Rannacher.

Strike alignment helps by a factor of about nine but does not cure it. It fixes
where the kink sits; it does nothing about what the scheme does to it.

Theta inherits the whole thing, because it is built from gamma through the PDE
identity: at `n_t = 20, n_s = 1600` unaligned its error is **-20.16** against
an analytic theta of **-20.35**. Delta does not: its plain-CN errors on the
unaligned sequence are -3.34e-04, -3.78e-05, -2.74e-06, +2.73e-06 - small, but
changing sign, so the fitted order (2.460) carries a residual of 0.643 and is a
cancellation rather than a rate. Rannacher's delta fits 2.009 with residual
0.003.

### The price does not warn you

At `n_s = 1600, n_t = 20, T = 0.05` unaligned, plain Crank-Nicolson returns

| | price | relative error | gamma | relative error |
|---|---:|---:|---:|---:|
| closed form | 1.90937494 | | 0.08893343 | |
| theta | 1.90699917 | 1.24e-03 | 0.189743 | 1.134 |
| rannacher | 1.90825869 | 5.85e-04 | 0.089021 | 9.89e-04 |

A factor of about 900 between the two relative errors. A test suite that
checked prices only would pass.

## Rannacher's cost on the price

Four implicit half steps are only first-order accurate over the `2 dt` they
cover, so they add an `O(dt^2)` term of the same sign as Crank-Nicolson's own.
Scaled price error `n^2 |error|` on the smooth case:

| `n` | theta | rannacher | ratio |
|---:|---:|---:|---:|
| 50 | 19.0195 | 20.0666 | 1.055 |
| 100 | 20.3437 | 21.3925 | 1.052 |
| 200 | 19.2799 | 20.3290 | 1.054 |
| 400 | 19.5074 | 20.5566 | 1.054 |
| 800 | 19.6144 | 20.6636 | 1.054 |

Fitted orders 1.9972 and 1.9973. A flat 5.4% on the constant, and order two
kept - which is exactly what a same-order perturbation should look like.

## `greeks_method="bump"`: what it cost

The pre-Slice-4 path is kept under that name: three full solves at `S`,
`S(1+h)` and `S(1-h)` with `h = 1%` of spot, each read through a cubic spline,
differenced centrally, with NaN for vega, theta and rho. Delta error against
the closed form, smooth case, aligned, Rannacher:

| `n` | grid | bump |
|---:|---:|---:|
| 50 | -9.038e-03 | +4.165e-04 |
| 100 | -2.394e-03 | -6.222e-05 |
| 200 | -5.658e-04 | -7.176e-05 |
| 400 | -1.430e-04 | -8.248e-05 |
| 800 | -3.594e-05 | -8.509e-05 |
| 1600 | -9.007e-06 | -8.574e-05 |

Fitted orders over `n` in `(50 ... 800)`: **2.001** (residual 0.021) against
**0.418** (0.563). The bump path is *better* on the coarse grids, because its
bias happens to have the opposite sign to the discretisation error there, and
then it stops. Two things cause the floor and only one of them is the bump
size:

1. the central difference's own `O(h^2)` bias, with `h` fixed at 1% of spot;
2. `s_max` defaults to `s_max_multiplier * spot`, so the three solves sit on
   three *different* grids with different `ds` and a different strike position
   - the one thing `strike_alignment` exists to control. Their discretisation
   errors do not cancel, which is why the bump gamma sequence is not a power
   law at all: log-space residual **0.73** against the grid path's **0.04**,
   with three sign changes over five refinements.

## Against QuantLib

`tests/oracle/test_pde_vs_quantlib.py`. Two genuinely different
discretisations: this package is uniform in spot over `[0, 4S]` with the strike
placed between nodes; QuantLib's `FdBlackScholesVanillaEngine` is uniform in
`log S` over a fixed number of standard deviations with nothing aligned to the
strike (on an 11-node mesh at `S = K = 100`, the nearest node to the strike is
at 102.53).

At `n = 800` (`tGrid = xGrid = n_s = n_t`), three points and both kinds:

| quantity | worst \|qpl - BS\| | worst \|QL - BS\| | worst gap | tolerance used |
|---|---:|---:|---:|---:|
| price | 2.00e-04 | 4.00e-04 | 2.00e-04 | 6e-04 |
| delta | 3.59e-05 | 1.22e-05 | 4.28e-05 | 1.5e-04 |
| gamma | 7.70e-07 | 2.71e-07 | 1.31e-06 | 4e-06 |

Both engines fit order two on price and delta: QuantLib 2.008-2.009 with
residuals of 0.002 across the board, this package 1.832-2.016. QuantLib's fits
are the cleaner ones because its mesher has no strike-alignment effect to
wobble with `n`.

**QuantLib's FD engine does not damp by default.** The default `schemeDesc` is
`FdmSchemeDesc.Douglas()` and the default `dampingSteps` is `0`;
`FdBlackScholesVanillaEngine(process, n, n)` is bit-for-bit
`(process, n, n, 0, FdmSchemeDesc.Douglas())`. `FdmSchemeDesc.CrankNicolson()`
is a different scheme *type* that agrees with `Douglas()` to 1.8e-15 on the
price, because Douglas splitting in one dimension is the theta scheme.

Which makes the strongest cross-check available. On `T = 18/365`,
`xGrid = 80 * tGrid`, relative gamma error:

| `tGrid` | `xGrid` | QL damping 0 | QL damping 2 | qpl theta | qpl rannacher |
|---:|---:|---:|---:|---:|---:|
| 5 | 400 | +1.799e+02 | +1.733e-02 | -2.462e-01 | +1.313e-02 |
| 10 | 800 | -2.361e+02 | -2.970e-04 | +5.311e-01 | +4.181e-03 |
| 20 | 1600 | -6.217e+02 | -1.110e-03 | +1.081e+00 | +9.945e-04 |
| 40 | 3200 | -1.353e+03 | -1.065e-03 | +2.172e+00 | +2.429e-04 |

Both undamped columns diverge under refinement, so the pathology is a property
of Crank-Nicolson and not of this package's stencils. QuantLib's is two to
three orders of magnitude worse, for a reason visible in its mesher: equal
spacing in `log S` packs far more nodes near the strike, `lambda dt` is
correspondingly larger, and `R(z)` sits closer to -1. Finer is stiffer. Its own
remedy - `dampingSteps = 2`, two fully implicit start-up steps - and ours are
the same idea.

## What contradicted the slice statement

1. **The gamma pathology does not appear at `n_s = n_t = n`.** The slice
   expected plain Crank-Nicolson to show start-up pollution on an aligned grid
   at the money and Rannacher to repair it. Measured, both are order two and
   the two columns agree to three significant figures. The pathology needs a
   time step large relative to `ds^2`, which that refinement path hides. It is
   pinned on `n_s = 80 n_t` instead.
2. **QuantLib's FD engine does not use Rannacher damping by default.** The
   slice said it did. It is undamped Crank-Nicolson, and on stressed grids its
   default gamma is wrong by a factor of up to 1353.
3. **"Replace the first time step by four steps of `dt/2`" does not add up.**
   Four half steps cover `2 dt`, so they replace the first *two* steps. That is
   both the standard Rannacher construction and what Giles and Carter call two
   half steps twice; it is what is implemented, and `_time_levels` is tested
   for taking `n_t + 2` steps covering exactly `T`.
4. **Reading theta from the last two time levels is first order**, not "a
   measured order" comparable to the identity's. The identity is what is
   reported; the difference is the cross-check.

## Citations

- Rannacher, R. (1984). *Finite element solution of diffusion problems with
  irregular data*. Numerische Mathematik 43, 309-327.
- Giles, M. B., and Carter, R. (2006). *Convergence analysis of Crank-Nicolson
  and Rannacher time-marching*. Journal of Computational Finance 9(4), 89-112.
- Pooley, D. M., Forsyth, P. A., and Vetzal, K. R. (2003). *Convergence
  remedies for non-smooth payoffs in option pricing*. Journal of Computational
  Finance 6(4). (Strike alignment; see also
  `docs/notes/pde_strike_alignment.md`.)
- Tavella, D., and Randall, C. (2000). *Pricing Financial Instruments: The
  Finite Difference Method*. (Grid construction and the standard stencils.)

Every number in this note was measured in this repository. None is quoted from
any of those sources.

## Reproducing

```
pytest -q tests/test_pde_greeks.py tests/test_pde_ch4.py
pytest -q tests/oracle/test_pde_vs_quantlib.py     # needs the [oracle] extra
PYTHONPATH=src python examples/pde_greeks_demo.py
PYTHONPATH=src python examples/pde_greeks_demo.py --case startup
```
