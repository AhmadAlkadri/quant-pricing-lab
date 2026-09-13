# SDE discretisation: strong and weak order, and a boundary Euler cannot see

This note derives what `qpl.engines.mc.sde` implements and records what was
measured. Everything numeric here was produced in this repository; nothing is
quoted from a source. Sources are cited where a result is not mine:
Kloeden & Platen (1992), *Numerical Solution of Stochastic Differential
Equations*, chapters 9-10 and 14 (strong and weak convergence, and the orders
of the schemes below); Glasserman (2003), *Monte Carlo Methods in Financial
Engineering*, sections 6.1-6.3 (discretisation) and 3.4 (exact CIR sampling);
Higham (2001), SIAM Review 43(3), section 5 (the measurement design);
Cox, Ingersoll & Ross (1985), Econometrica 53(2) (the process);
Broadie & Kaya (2006), Operations Research 54(2), section 2.1 (exact
transition sampling); Andersen (2008), Journal of Computational Finance 11(3),
section 3 (full truncation).

## 1. Two different questions

For `dX = a(X,t) dt + b(X,t) dW` on a grid of step `h`, a scheme is **strongly**
convergent of order `p` if `E|X_T^h - X_T| = O(h^p)` and **weakly** convergent
of order `q` if `|E f(X_T^h) - E f(X_T)| = O(h^q)` for smooth enough `f`.

The strong error compares *the same path*: it is only defined once the scheme
and the reference are driven by the same Brownian motion. The weak error
compares *laws*, and does not need a coupling at all. That distinction is the
whole note.

Euler-Maruyama:

    X_{i+1} = X_i + a h + b dW_i

Milstein adds the next term of the stochastic Taylor expansion, which is what
you get by applying Ito's lemma to `b` inside the diffusion integral and
keeping the leading piece:

    X_{i+1} = X_i + a h + b dW_i + (1/2) b (db/dx) (dW_i^2 - h)

The added term has conditional mean zero. That is exactly why it buys a strong
order and not a weak one: it corrects each path, not the law's low moments. The
rest of this note measures that sentence rather than asserting it.

## 2. Measuring the strong error at all

A coarse run and a fine run seeded independently differ by roughly
`sqrt(2) * std(X_T)` at *every* step size, and the fitted order is zero. The
coupling (Higham 2001) is to draw the increments on the finest grid and build
every coarser one by **summing blocks** of them. In units of standard normals
with fine step `h` and coarse step `r h`,

    dW_coarse = sum_{j<r} sqrt(h) Z_j = sqrt(r h) * (sum_j Z_j / sqrt(r)),

so the coarse normal is the block sum over `sqrt(r)`. That is
`qpl.engines.mc.sde.coarsen_normals`. For GBM the exact terminal value depends
only on the *total* Brownian increment, which the construction preserves
exactly, so the reference is literally the same array at every level (checked
to 1e-12 relative in `tests/test_mc_sde.py`).

## 3. GBM: the measured orders

Point: `S0 = 100`, `mu = 0.2`, `sigma = 0.2`, `T = 1`, `K = 100`, 100 000
paths, `h = 1/8 ... 1/128`, seed 20250913.

| n | Euler strong | Milstein strong | ratio | Euler weak call | Milstein weak call |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.0228e+00 | 5.3875e-01 | 1.90 | -3.1262e-01 | -3.9214e-01 |
| 16 | 7.0559e-01 | 2.7224e-01 | 2.59 | -1.5591e-01 | -1.9760e-01 |
| 32 | 4.9327e-01 | 1.3681e-01 | 3.61 | -7.8579e-02 | -9.9190e-02 |
| 64 | 3.4650e-01 | 6.8572e-02 | 5.05 | -3.9159e-02 | -4.9711e-02 |
| 128 | 2.4464e-01 | 3.4342e-02 | 7.12 | -1.8907e-02 | -2.4891e-02 |

Fitted (order, log-space residual, constant):

| claim | Euler | Milstein |
|---|---|---|
| strong | **0.5154**, 0.0062, 2.9624 | **0.9932**, 0.0025, 4.2643 |
| weak, `f(x) = x` | **1.0041**, 0.0087, 2.4142 | **0.9944**, 0.0017, 2.3916 |
| weak, `f(x) = (x-K)^+` | **1.0088**, 0.0109, 2.5642 | **0.9946**, 0.0018, 3.1096 |

Two things in that table are worth saying out loud.

**The Milstein strong constant is larger.** At `h = 1/8` Milstein is only 1.90x
better; by `h = 1/128` it is 7.12x. The whole gain is the exponent, and
"Milstein is more accurate" is not a statement about any one grid.

**The weak columns agree between the schemes.** They have to. Taking
conditional expectations through either recursion gives
`E[X_{i+1}] = (1 + mu h) E[X_i]`, because both the Euler diffusion term and the
Milstein correction have conditional mean zero. So

    E[X_T^h] = S0 (1 + mu h)^{T/h}    for BOTH schemes,

and the weak error in `f(x) = x` is the deterministic

    S0 ((1 + mu h)^{T/h} - e^{mu T}) = -S0 e^{mu T} mu^2 T h / 2 + O(h^2),

constant **2.4428** at this point, against measured fitted constants 2.4142 and
2.3916. This is the sharpest available form of "a strong order is not a weak
order": there is nothing to measure, the two schemes are *identical* in the
first moment, and the Monte Carlo only confirms an identity.

## 4. Milstein on GBM is not exp-of-Euler-on-log

The slice that produced this note expected "Milstein on GBM = exp of Euler
applied to `log X`". It is not, and the reason is more interesting than the
claim.

`d log S = (mu - sigma^2/2) dt + sigma dW` has **state-independent**
coefficients, so the Euler step on the log reproduces the log increment exactly
and its exponential is the exact lognormal transition. Euler on the log is not
order one; it is exact. (And Milstein on the log is bit-for-bit Euler on the
log, since `db/dx = 0` makes the correction exactly `0`.)

What is true is a truncation identity. Write `u = (mu - sigma^2/2) h + sigma dW`
for the exact log increment, so the exact transition multiplies by `exp(u)`.
Then

    Milstein step = X (1 + mu h + sigma dW + (sigma^2/2)(dW^2 - h))
                  = X (1 + u + (sigma^2/2) dW^2),

which is `exp(u)` truncated after the quadratic term **in dW only**: the genuine
second-order Taylor polynomial `1 + u + u^2/2` also carries the `h dW` and `h^2`
pieces, and Milstein drops them as `O(h^{3/2})` and `O(h^2)`. Pinned to
3.3e-16 relative in `tests/test_mc_sde.py`; the per-step residual
`exp(u) - (1 + u + sigma^2 dW^2/2)` is measured to shrink by 0.35 per halving,
consistent with `2^{-3/2} = 0.354`, which is the local error behind the global
strong order 1.

## 5. The finding that cost the most work: the noise floor

Measuring a weak error as the difference of two independently estimated means
puts an `O(1)` standard deviation on an `O(h)` signal. The fix is common random
numbers: estimate `mean(f(X^h_T) - f(X_T))` on coupled paths. Unbiased, and the
standard error is now set by the spread of the *difference*.

But the spread of the difference **is the scheme's strong error**. So

    signal / noise  =  (C_weak / c_strong) * sqrt(N) * h^{p_weak - p_strong}.

For Milstein `p_weak = p_strong = 1` and the ratio is flat in `h`: measured
|z| = 211, 209, 207, 207, 207 down the ladder. For Euler
`p_weak - p_strong = 1/2` and the ratio **falls**: measured |z| = 73, 55, 40,
29, 20. Refining the grid makes the Euler measurement worse. Only `N` helps,
and it helps at `sqrt(N)`.

That has a blunt consequence for the choice of study point. `C_weak` is set by
the drift (`S0 e^{mu T} mu^2 T / 2`) and `c_strong` by the diffusion
(`sqrt(2/pi) sigma^2 S0 e^{(mu + sigma^2/2) T} sqrt(T/2)`, derived from
`X^h - X ~ -(sigma^2/2) sum X_i (dW_i^2 - h)`, which predicts 2.812 against the
measured 2.9624 -- 5.1%). At `mu = 0.05, sigma = 0.2` -- an ordinary Black-Scholes
point -- the ratio is 16x smaller and the Euler weak error is simply not
measurable at 100 000 paths: z-scores -5.8, -3.3, -2.8, -1.3, -0.2, with a
fitted order of **1.5742** and a log-space residual of **0.4231**. That fit is
noise, and it is pinned as a `NEGATIVE_FINDING` so that nobody reads a slope
off a table like it. The `mu = 0.2` used throughout section 3 is load-bearing.

## 6. Pricing bias

Read the same GBM study as a risk-neutral market (`r = mu = 0.2`, `q = 0`): the
analytic Black-Scholes call is 19.6298871012, and the Euler price is
`e^{-rT} mean((X^h_T - K)^+)`.

| n | Euler price | bias | % of price |
|---:|---:|---:|---:|
| 8 | 19.375050 | -0.254837 | -1.298% |
| 16 | 19.502641 | -0.127247 | -0.648% |
| 32 | 19.564618 | -0.065269 | -0.333% |
| 64 | 19.597268 | -0.032619 | -0.166% |
| 128 | 19.614314 | -0.015573 | -0.079% |

(from `examples/sde_convergence.py` at 50 000 paths; the 100 000-path test run
gives -0.2560 ... -0.0155.) Fitted order **1.0088**, constant **2.0994**,
which is `e^{-rT} * 2.5642` -- the discounted weak-error constant, as it must
be. The bias is negative at every level: Euler under-prices this call.

The experiment is measurable only because the coupled exact payoff is used as a
control variate with its *known* mean, the analytic price. Its standard error
runs 3.5e-03 down to 7.8e-04 across the ladder against the plain estimator's
flat 5.7e-02 -- a variance factor of 250 to 5500 -- and at `n = 128` the plain
estimator's own standard error is **larger** than the bias it is trying to see.

## 7. CIR: the boundary

`dv = kappa (theta - v) dt + xi sqrt(v) dW`. Two parameter sets, both with
`v0 = theta = 0.04` and `T = 1`:

- **Feller satisfied**: `kappa = 2, xi = 0.2`, Feller number `4 kappa theta / xi^2 = 8`.
- **Feller violated**: `kappa = 0.5, xi = 1.0`, Feller number `0.08`.

### 7.1 The exact transition

The transition law is a **scaled noncentral chi-square**:

    v_{t+h} = c * X,   X ~ chi'^2(d, lambda),
    c = xi^2 (1 - e^{-kappa h}) / (4 kappa),
    d = 4 kappa theta / xi^2,
    lambda = v_t e^{-kappa h} / c.

Check rather than quote: with `E[chi'^2] = d + lambda` and
`Var = 2(d + 2 lambda)`,

    E[v_{t+h}]   = c d + v_t e^{-kappa h} = theta(1 - e^{-kappa h}) + v_t e^{-kappa h},
    Var[v_{t+h}] = 2 c^2 d + 4 c v_t e^{-kappa h}
                 = theta xi^2 (1-e^{-kappa h})^2/(2 kappa)
                   + v_t (xi^2/kappa)(e^{-kappa h} - e^{-2 kappa h}),

which are the CIR conditional moments. Note that `d` **is** the Feller number:
the condition `2 kappa theta >= xi^2` is exactly `d >= 2`, so "the sampler's
degrees of freedom" and "is zero attainable" are one quantity read two ways.
`scipy.stats.ncx2` needs only `d > 0`, so the sampler is exact on both sides.
Implemented as `cir_transition_parameters` / `cir_sde(...).exact_step`, checked
against the moments above at 200 000 paths and four steps (so the *composition*
of four transitions is checked, not just a one-shot draw) to within four
standard errors in both regimes.

`cir_expected_excess` gives `E[(v_T - K)^+]` deterministically by integrating
the ncx2 survival function (`E[(X-a)^+] = int_a^inf P(X > x) dx`). The slice
asked for a comparison "against the exact sampler"; integrating the same law
instead removes the reference's own Monte Carlo error, which is strictly better
evidence for the same claim.

### 7.2 What plain Euler does, and what full truncation does not fix

At 50 steps, 50 000 paths, seed 11:

| regime | paths reaching a negative state | paths becoming NaN | same under full truncation |
|---|---:|---:|---|
| Feller satisfied | 0.00026 | 0.00026 | 0.00026 negative, **0** NaN |
| Feller violated | 0.91174 | 0.90968 | 0.91174 negative, **0** NaN |

Three readings.

1. **The Feller condition is about the process, not the discretisation.** It
   holds in the first row and 13 paths in 50 000 still go negative.
2. **Plain Euler does not merely go negative, it stops.** `b(v) = xi sqrt(v)`
   cannot be evaluated at a negative state, so the *next* step is NaN. The NaN
   fraction is slightly below the negative fraction for the only reason it can
   be: a path whose first negative state is the terminal one is never evaluated
   again.
3. **Full truncation (Andersen 2008) produces exactly the same negative states.**
   This was written into the slice statement as "does not [go negative]", and it
   is false. The two schemes are pathwise identical while `v >= 0` (because
   `max(v,0) = v` there), and after the first negative state plain Euler has no
   next state at all, so the frequencies are *forced* to be equal. What full
   truncation buys is that the recursion stays **defined** -- which is what
   makes a convergence study possible at all. Positivity is not on offer and
   would need a different scheme (Andersen's QE, or the exact sampler).

The example goes further: in the Feller-violated regime, 68-73% of the
**terminal** variances under full truncation are negative at `h = 1/4 ... 1/32`.
This is not a scheme grazing a boundary.

One implementation trap worth recording: the Milstein correction `(1/2) b b'`
for CIR is the constant `xi^2/4`, but its two factors are `0` and `inf` at
`v = 0`, which full truncation makes a *common* state rather than a
measure-zero one. Building the correction from the factors gives NaN paths;
`cir_sde` supplies the product directly, and a test asserts both halves of that.

### 7.3 Full-truncation Euler weak orders

400 000 paths, `h = 1/4 ... 1/32`, seed 4242, reference = the exact law.

| quantity | Feller satisfied | Feller violated |
|---|---|---|
| `E[v_T]` | no measurable bias past `h = 1/4` | **0.6834** (residual 0.1261), constant 3.3e-02 |
| `E[(v_T - 0.05)^+]` | **0.9729** (residual 0.0705), constant 3.8638e-03 | **0.9389** (residual 0.0606), constant 0.1329 |

`v0 = theta` is chosen so that the exact `E[v_T]` is `theta` for every `T` and
the *untruncated* Euler recursion has the same fixed point
(`E[v_{i+1}] = (1 - kappa h) E[v_i] + kappa theta h`). Any measured bias in the
mean is therefore the truncation's alone. With the Feller condition satisfied
the truncation fires on 2.6e-04 of paths and there is nothing to see from
`h = 1/8` on: z-scores -2.78, +0.19, -0.37, +0.76, with a fitted order that
wanders 0.53 to 1.05 over five seeds and log-space residuals 0.17 to 0.94. The
honest report is "no order", not a slope.

With it violated the mean bias is large, negative and **degraded**: -28% of the
true mean at `h = 1/4`, fitting 0.6834 rather than 1 (0.6906 and 0.7037 at two
other seeds).

The call weak order is close to 1 in both regimes, and that is the point: the
damage of violating Feller shows up in the **constant** -- 0.1329 against
3.8638e-03, a factor of 34 -- far more than in the exponent. The two bands in
`qpl.cases.sde_discretization` differ for a measured reason: over seeds
{4242, 77, 5150} the Feller-satisfied fit moves 0.9729 / 1.0404 / 1.0850 while
the violated one moves 0.9389 / 0.9472 / 0.9407, so a 0.06 band on the latter is
a real claim that the order is below one and a 0.15 band on the former is an
admission that this budget cannot tell 0.97 from 1.09.

## 8. Where this is pinned

- `src/qpl/engines/mc/sde.py` -- schemes, models, exact samplers, coupling.
- `src/qpl/validation/stochastic.py` -- `strong_error` / `weak_error`, each
  reporting its own standard error, because that number is the floor a fitted
  order has to clear.
- `src/qpl/cases/sde_discretization.py` -- fourteen rows (ten orders, four
  negative findings) with every band derived in its `notes`.
- `tests/test_mc_sde.py`, `tests/test_sde_convergence.py`,
  `tests/cases/test_sde_discretization_cases.py`.
- `examples/sde_convergence.py` (also `--case cir`).
