"""Heston paths by simulation: Andersen's QE scheme and two comparisons.

What has to be discretised, and why the obvious thing fails
----------------------------------------------------------
Under the pricing measure, with `X_t = ln S_t`,

    dX_t = (r - q - v_t / 2) dt + sqrt(v_t) dW1_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2_t,   d<W1, W2> = rho dt.

The variance is a square-root diffusion, so an Euler step can send it below
zero, at which point `sqrt(v)` is undefined and the log-spot step cannot be
taken at all. `qpl.engines.mc.sde` already measured how bad that is: in the
Feller-violating regime 68-73% of terminal variances are negative at
`h = 1/4 ... 1/32`, and Andersen's **full truncation** (evaluate the
coefficients at `max(v, 0)`, leave the state unfloored) keeps the recursion
*defined* without making it positive. Lord, Koekkoek and van Dijk (2010),
Quantitative Finance 10(2), compare the Euler fixes and find full truncation
the least biased of them, which is why it is the Euler comparison here.

The variance transition is known exactly -- it is a scaled noncentral
chi-square (Broadie & Kaya 2006, Operations Research 54(2), section 2.1;
implemented in `qpl.engines.mc.sde.cir_sde`) -- so one could sample `v`
exactly. What is *not* cheap is the log-spot: the exact scheme needs the
conditional law of the integrated variance `int_0^T v_s ds` given the
endpoints, which Broadie and Kaya obtain by numerically inverting a
characteristic function once per step per path. That cost, not the variance,
is what makes exact Heston simulation impractical, and it is what Andersen's
scheme replaces.

The QE scheme (Andersen 2008, sections 3.2-3.3)
-----------------------------------------------
Andersen, L. (2008), "Simple and efficient simulation of the Heston stochastic
volatility model", *Journal of Computational Finance* 11(3), 1-42. The
derivation below is worked here from the CIR conditional moments; no formula,
table or line of code is transcribed from that paper.

**Step 1: the two conditional moments.** For the square-root process the first
two conditional moments of `v_{t+dt}` given `v_t` are elementary (take
expectations in the SDE; the diffusion term is a martingale):

    m  = E[v_{t+dt} | v_t] = theta + (v_t - theta) e^{-kappa dt}
    s2 = Var[v_{t+dt} | v_t]
       = v_t (xi^2 / kappa) e^{-kappa dt} (1 - e^{-kappa dt})
         + theta (xi^2 / (2 kappa)) (1 - e^{-kappa dt})^2

-- the same pair `qpl.engines.mc.sde.cir_moments` returns from the exact
noncentral chi-square transition, which is the cross-check
`tests/test_mc_heston.py` runs as an EXACT_IDENTITY.

**Step 2: one shape parameter.** Everything the scheme does is decided by

    psi = s2 / m^2,

the squared coefficient of variation. Large `psi` means the transition is
concentrated near zero with a long right tail (the Feller-violating,
small-`v_t` regime); small `psi` means it is a bump around `m`. No single
two-parameter family covers both shapes, so Andersen uses two and switches.

**Step 3a: the quadratic branch, `psi <= psi_c`.** Sample

    v_{t+dt} = a (b + Z)^2,    Z ~ N(0, 1),

a non-central chi-square with one degree of freedom, scaled and shifted.
Matching moments: `E[(b + Z)^2] = 1 + b^2` and `Var[(b + Z)^2] = 2 + 4 b^2`,
so `a (1 + b^2) = m` and `a^2 (2 + 4 b^2) = s2`. Dividing,

    psi = (2 + 4 b^2) / (1 + b^2)^2,

a quadratic in `b^2` whose root with `b^2 >= 0` is

    b^2 = 2/psi - 1 + sqrt(2/psi) sqrt(2/psi - 1),      a = m / (1 + b^2).

The root is real exactly when `psi <= 2`, which is why the switching level must
sit below 2.

**Step 3b: the exponential branch, `psi > psi_c`.** Sample from the law with
an atom at the origin and an exponential tail,

    P(v_{t+dt} = 0) = p,   density (1 - p) beta e^{-beta x} on x > 0,

i.e. by inverting the distribution function at a uniform `U`:

    v_{t+dt} = 0                        if U <= p,
             = beta^{-1} ln((1-p)/(1-U)) if U > p.

Its moments are `(1 - p)/beta` and `(1 - p)(2 - (1 - p))/beta^2`, so matching
gives `(1 + p)/(1 - p) = psi` and hence

    p = (psi - 1) / (psi + 1),     beta = 2 / (m (psi + 1)),

which is why this branch needs `psi >= 1` and the switching level must sit
above 1. Anywhere in `[1, 2]` therefore works; this module uses Andersen's own
`psi_c = 1.5` (`QE_PSI_C`), the midpoint of that window, and
`tests/test_mc_heston.py` checks the moment match on *both* sides of it.

Both branches match the first two moments **exactly** and nothing else: the QE
scheme is a two-moment fit, not an approximation of the transition density.
Its third and fourth moments differ from the exact law's, and that gap is
measured and reported rather than treated as a failure --
`qe_branch_moments` returns the branch's own analytic moments so that the claim
is checkable at the formula level, and the sampler is checked against them.

**Step 4: the log-spot, and why it is not an Euler step.** Integrating the
variance SDE over one step,

    int_t^{t+dt} sqrt(v_s) dW2_s
        = (v_{t+dt} - v_t - kappa theta dt + kappa int_t^{t+dt} v_s ds) / xi,

so the correlated part of the log-spot increment can be written in terms of the
*sampled* variance endpoints instead of being simulated. With
`dW1 = rho dW2 + sqrt(1 - rho^2) dW_perp` and the integrated variance
approximated by the two-point rule
`int v ds ~ dt (gamma_1 v_t + gamma_2 v_{t+dt})`,

    X_{t+dt} = X_t + (r - q) dt + K_0 + K_1 v_t + K_2 v_{t+dt}
               + sqrt(K_3 v_t + K_4 v_{t+dt}) Z,     Z ~ N(0,1) independent,

    K_0 = -rho kappa theta dt / xi
    K_1 = gamma_1 dt (kappa rho / xi - 1/2) - rho / xi
    K_2 = gamma_2 dt (kappa rho / xi - 1/2) + rho / xi
    K_3 = gamma_1 dt (1 - rho^2)
    K_4 = gamma_2 dt (1 - rho^2)

with `gamma_1 = gamma_2 = 1/2` here (`QE_GAMMA_1`, `QE_GAMMA_2`), the central
rule, which is the choice whose integrated-variance error is `O(dt^2)` rather
than `O(dt)`. Note what this buys: the correlation enters through a
*deterministic* function of the two variance values, so `rho` is reproduced
without ever simulating a correlated pair of increments, and the only normal
drawn for the spot is orthogonal to the variance. `tests/test_mc_heston.py`
measures the realised correlation of the increments against `rho`.

**Step 5: the martingale correction (Andersen section 4.3).** Nothing above
forces `E[S_{t+dt} | S_t, v_t] = S_t e^{(r-q) dt}`, and a scheme that fails it
prices a forward wrongly before it prices anything else. Conditional on the two
variance values the log-spot increment is Gaussian, so

    E[e^{X_{t+dt} - X_t} | v_t]
        = e^{(r-q) dt + K_0 + (K_1 + K_3/2) v_t} E[e^{A v_{t+dt}} | v_t],
        A = K_2 + K_4 / 2,

and `E[e^{A v_{t+dt}}]` is available in closed form for each branch because
each branch is a named law:

    quadratic:    exp(A a b^2 / (1 - 2 A a)) / sqrt(1 - 2 A a),   2 A a < 1
    exponential:  p + beta (1 - p) / (beta - A),                  A < beta.

Replacing the constant `K_0` by the path-dependent

    K_0* = -ln E[e^{A v_{t+dt}} | v_t] - (K_1 + K_3 / 2) v_t

makes the conditional expectation exactly `e^{(r-q) dt}`, hence
`E[e^{-(r-q)T} S_T] = S_0` up to sampling error at **any** step size. With
`rho < 0` -- the empirically relevant sign, and the sign of every parameter set
in this repository -- `A < 0` and both conditions hold automatically. Where a
condition fails the step falls back to the uncorrected `K_0` and the count is
reported in `meta["martingale_correction_fallbacks"]`, because silently
dropping the correction is exactly the failure it exists to prevent.
`tests/test_mc_heston.py` pins the defect **without** the correction as a
NEGATIVE_FINDING.

The three schemes
-----------------
`HESTON_SCHEMES` names them:

- `"qe"` -- the above.
- `"euler_full_truncation"` -- Euler on both state variables with the
  coefficients evaluated at `max(v, 0)` and the state left unfloored (Lord,
  Koekkoek & van Dijk 2010; the same truncation `qpl.engines.mc.sde` applies
  to the CIR process on its own). The log-spot step is the ordinary
  `(r - q - v^+/2) dt + sqrt(v^+) sqrt(dt) Z_x` with
  `Z_x = rho Z_v + sqrt(1 - rho^2) Z_perp`, so the correlation is imposed on
  the increments directly. It has **no** martingale correction: its drift is
  not built from a conditional moment generating function, so there is nothing
  to correct without changing the scheme into a different one.
- `"exact_variance_euler_log_spot"` -- the variance drawn from its exact
  noncentral chi-square transition (`qpl.engines.mc.sde`), the log-spot taken
  with the same `K_0 ... K_4` step as QE. It isolates the two sources of QE's
  bias: whatever remains here is the log-spot discretisation alone, because the
  variance marginal is exact. Its martingale correction uses the noncentral
  chi-square moment generating function.

Conditioning on the variance driver
-----------------------------------
All three schemes share one structural property, and it is what makes the bias
tables in this slice measurable at all: **conditional on the variance driver,
`ln S_T` is exactly Gaussian**. `ConditionalTerminalLaw` carries its two
moments, accumulated as the paths are built, and
`conditional_vanilla_values` / `conditional_digital_values` turn them into the
exact conditional expectation of a terminal payoff. Averaging that over
variance paths estimates *the same scheme price* with the whole spot-diffusion
variance removed -- measured factor **54x** for QE and **19-48x** for
full-truncation Euler on the reference set, which turns a 5e-02 standard error
into a 7e-03 one at the same path count. See `ConditionalTerminalLaw`.

Antithetic sampling reflects **both** driving normals. For the exponential
branch the uniform is taken as `U = Phi(Z_v)`, so a reflected `Z_v` gives
`1 - U`, which is again uniform: one normal per step per state variable drives
every branch, and the antithetic pair is an honest reflection of the whole
driver. The exact-variance scheme has no such representation -- a noncentral
chi-square draw is not a function of one normal -- so it refuses antithetic
sampling rather than reflecting half of its driver.

Sources (all re-derived above; nothing quoted):

- Andersen, L. (2008), *Journal of Computational Finance* 11(3), 1-42,
  sections 3.2-3.3 (the scheme), 4.3 (the martingale correction).
- Broadie, M. and Kaya, O. (2006), *Operations Research* 54(2), 217-231,
  section 2.1 (the exact variance transition; the integrated-variance cost).
- Lord, R., Koekkoek, R. and van Dijk, D. (2010), *Quantitative Finance*
  10(2), 177-194 (full truncation is the least-biased Euler fix).
- Glasserman, P. (2003), *Monte Carlo Methods in Financial Engineering*,
  section 3.4 (the square-root process).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.special import ndtr
from scipy.stats import ncx2

from ...exceptions import InvalidInputError, NotSupportedError
from ...models.heston import HestonModel
from .sde import cir_transition_parameters

__all__ = [
    "EULER_FULL_TRUNCATION",
    "EXACT_VARIANCE_EULER_LOG_SPOT",
    "HESTON_SCHEMES",
    "QE",
    "QE_GAMMA_1",
    "QE_GAMMA_2",
    "QE_PSI_C",
    "ConditionalTerminalLaw",
    "HestonPaths",
    "conditional_digital_values",
    "conditional_forward",
    "conditional_vanilla_values",
    "heston_variance_moments",
    "log_spot_coefficients",
    "qe_branch_moments",
    "qe_variance_step",
    "simulate_heston",
]

QE = "qe"
EULER_FULL_TRUNCATION = "euler_full_truncation"
EXACT_VARIANCE_EULER_LOG_SPOT = "exact_variance_euler_log_spot"

HESTON_SCHEMES: tuple[str, ...] = (QE, EULER_FULL_TRUNCATION, EXACT_VARIANCE_EULER_LOG_SPOT)
"""The three variance/log-spot discretisations, as data for validation."""

QE_PSI_C = 1.5
"""Andersen's switching level for `psi = s2 / m^2`.

The quadratic branch exists for `psi <= 2` (its `b^2` root is real there) and
the exponential branch for `psi >= 1` (its atom `p = (psi-1)/(psi+1)` is
non-negative there), so any level in `[1, 2]` is admissible and the scheme is
continuous in distribution across none of them -- the two families are
genuinely different laws that agree only in their first two moments. 1.5 is the
midpoint of the admissible window and Andersen's own choice; the moment match
is checked on both sides of it."""

QE_GAMMA_1 = 0.5
QE_GAMMA_2 = 0.5
"""Weights of the two-point rule for `int_t^{t+dt} v_s ds`.

`(1/2, 1/2)` is the trapezoidal rule, whose error on a smooth integrand is
`O(dt^3)` per step against the left rule's `O(dt^2)`. Andersen reports the
central choice as the better one and this module uses it; `(1, 0)` would make
the log-spot step explicit in `v_{t+dt}` and is not offered, because a
configuration knob whose only effect is to make the scheme worse is a way to
get a wrong answer, not a way to learn something."""

_TINY = float(np.finfo(float).tiny)

_PSI_FLOOR = 1e-16
"""Lower clip on `psi = s2 / m^2`, for a step so short that `e^{-kappa dt}`
rounds to 1.

Then `s2` evaluates to exactly zero, `psi` is `0/m^2 = 0`, and `2/psi` divides
by zero. A degenerate step of that kind is not an error -- it is the correct
limit, `v_{t+dt} = m` with no noise -- and clipping reproduces it: at
`psi = 1e-16` the quadratic branch has `b^2 ~ 2e16`, so `a (b + Z)^2` equals
`m` to within `1.4e-08 m |Z|` and its variance is `4 m^2 / b^2 ~ 2e-16 m^2`.
Sub-ulp grid intervals do occur in practice: `heston_time_grid` unions a
uniform grid with a contract's own schedule, and two ways of computing `i T /
n` can differ in the last bit."""


def _psi(m: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """`s2 / m^2`, clipped below; see `_PSI_FLOOR`."""
    return np.maximum(s2 / (m * m), _PSI_FLOOR)


@dataclass(frozen=True)
class ConditionalTerminalLaw:
    """`ln S_T` given the variance driver: Gaussian, with these two moments.

    Why it exists
    -------------
    In all three schemes here the terminal log-spot is, **conditional on the
    whole variance driver**, exactly Gaussian:

    - QE and the exact-variance hybrid step
      `X_{i+1} = X_i + mu dt + K_0 + K_1 v_i + K_2 v_{i+1} + sqrt(K_3 v_i +
      K_4 v_{i+1}) Z_i` with `Z_i` drawn independently of everything that made
      the variance path, so summing gives a Gaussian with
      `mean = sum(mu dt + K_0 + K_1 v_i + K_2 v_{i+1})` and
      `variance = sum(K_3 v_i + K_4 v_{i+1})`;
    - full-truncation Euler steps
      `X_{i+1} = X_i + (mu - v_i^+/2) dt + sqrt(v_i^+ dt)(rho Z_{v,i} +
      sqrt(1-rho^2) Z_{perp,i})`, and conditioning on the variance path
      conditions on every `Z_{v,i}` (the variance recursion is a function of
      them), leaving the orthogonal part: `mean` collects the drift and the
      `rho Z_v` term, `variance = (1 - rho^2) sum v_i^+ dt`.

    So the payoff of any instrument that reads only `S_T` has a **closed-form
    conditional expectation** -- a Black-Scholes formula in `(mean, variance)`
    -- and averaging that over variance paths is an unbiased estimator of
    *exactly the same scheme price* with the entire spot-diffusion variance
    removed. That is Romano and Touzi's (1997, Mathematical Finance 7(4))
    conditioning argument applied to the discretisation rather than to the
    model, and it is the reason the bias tables in
    `docs/notes/heston_monte_carlo_qe.md` resolve a 7e-03 weak error at a path
    count where the plain estimator's standard error is 5e-02.

    Complexity receipt: two extra `(n_paths,)` accumulators and about six
    flops per step, always computed because the cost is invisible next to the
    step itself and an optional buffer that half the callers forget to ask for
    is how a sharp measurement quietly becomes a blunt one. What it prevents:
    a convergence order fitted through levels whose errors are inside their
    own noise, which `qpl.validation.weak_error` warns about and which no
    amount of honest reporting turns into evidence.

    Parameters
    ----------
    mean, variance
        `(n_paths,)` conditional mean and variance of `ln S_T`. `variance` is
        zero only in the degenerate case of a variance path that is zero
        throughout.
    """

    mean: np.ndarray
    variance: np.ndarray


def conditional_forward(law: ConditionalTerminalLaw) -> np.ndarray:
    """`E[S_T | variance driver] = exp(mean + variance / 2)` per path."""
    return np.exp(law.mean + 0.5 * law.variance)


def _d2(law: ConditionalTerminalLaw, strike: float) -> tuple[np.ndarray, np.ndarray]:
    """`(sqrt(variance), d2)` with `d2 = (mean - ln K) / sqrt(variance)`."""
    root = np.sqrt(np.maximum(law.variance, 0.0))
    safe = np.where(root > 0.0, root, 1.0)
    return root, (law.mean - math.log(strike)) / safe


def conditional_vanilla_values(
    law: ConditionalTerminalLaw, *, strike: float, kind: str
) -> np.ndarray:
    """`E[(S_T - K)^+ | variance driver]` (or the put), per path, undiscounted.

    The Black-Scholes formula written in `(mean, variance)` rather than in
    `(forward, sigma sqrt(T))`, because that is the form the conditioning
    produces: with `ln S_T ~ N(m, V)`,

        E[(S_T - K)^+] = e^{m + V/2} Phi(d2 + sqrt(V)) - K Phi(d2),
        d2 = (m - ln K) / sqrt(V).

    At `V = 0` the law is a point mass and the value is the intrinsic
    `(e^m - K)^+`, which is the continuous limit of the line above.
    """
    if kind not in {"call", "put"}:
        raise InvalidInputError("kind must be 'call' or 'put'")
    root, d2 = _d2(law, strike)
    forward = conditional_forward(law)
    if kind == "call":
        smooth = forward * ndtr(d2 + root) - strike * ndtr(d2)
        degenerate = np.maximum(np.exp(law.mean) - strike, 0.0)
    else:
        smooth = strike * ndtr(-d2) - forward * ndtr(-d2 - root)
        degenerate = np.maximum(strike - np.exp(law.mean), 0.0)
    return np.where(root > 0.0, smooth, degenerate)


def conditional_digital_values(
    law: ConditionalTerminalLaw, *, strike: float, cash: float, kind: str
) -> np.ndarray:
    """`E[cash * 1{S_T > K} | variance driver]` (or `<`), per path, undiscounted.

    `P(S_T > K | m, V) = Phi(d2)`, the same `d2` as above -- the indicator is
    the one payoff whose conditional expectation is a single normal tail, which
    is why conditioning helps a digital even more than it helps a call.
    """
    if kind not in {"call", "put"}:
        raise InvalidInputError("kind must be 'call' or 'put'")
    root, d2 = _d2(law, strike)
    probability = np.where(
        root > 0.0,
        ndtr(d2) if kind == "call" else ndtr(-d2),
        (np.exp(law.mean) > strike).astype(float)
        if kind == "call"
        else (np.exp(law.mean) < strike).astype(float),
    )
    return cash * probability


@dataclass(frozen=True)
class HestonPaths:
    """Simulated log-spot and variance, plus what produced them.

    Parameters
    ----------
    times
        The time grid, starting at 0.
    log_spot, variance
        `(n_paths, len(times))` when the whole path was stored, and
        `(n_paths, 1)` holding the terminal column when it was not (see
        `simulate_heston`'s `store_paths`). `log_spot` is `ln S_t`, not
        `ln(S_t / S_0)`.
    normals
        `(z_variance, z_spot)`, each `(n_paths, n_steps)`, when the caller
        asked for them; `None` otherwise. They are not carried by default
        because a fine grid at a large path count doubles the memory for
        arrays most callers never read -- the opposite trade-off from
        `qpl.engines.mc.sde.SDEPaths`, which always carries its normals
        because every study there is a coupled one.
    meta
        Scheme, grid, path count and the diagnostics named in the module
        docstring.
    """

    times: np.ndarray
    log_spot: np.ndarray
    variance: np.ndarray
    conditional: ConditionalTerminalLaw | None = None
    normals: tuple[np.ndarray, np.ndarray] | None = None
    meta: Mapping[str, object] = field(default_factory=dict)

    @property
    def terminal_log_spot(self) -> np.ndarray:
        """`ln S_T` per path."""
        return self.log_spot[:, -1]

    @property
    def terminal_spot(self) -> np.ndarray:
        """`S_T` per path."""
        return np.exp(self.log_spot[:, -1])

    @property
    def terminal_variance(self) -> np.ndarray:
        """`v_T` per path."""
        return self.variance[:, -1]

    @property
    def spots(self) -> np.ndarray:
        """`exp(log_spot)`, whatever columns were stored."""
        return np.exp(self.log_spot)


def heston_variance_moments(
    v: np.ndarray | float, dt: float, *, kappa: float, theta: float, xi: float
) -> tuple[np.ndarray, np.ndarray]:
    """`(m, s2)`: the exact conditional mean and variance of `v_{t+dt}`.

    Derived in the module docstring by taking expectations in the square-root
    SDE. These are the two numbers the QE scheme matches, and they are the same
    pair `qpl.engines.mc.sde.cir_moments` obtains from the exact noncentral
    chi-square transition by a completely different route (its degrees of
    freedom and non-centrality), which is what makes the agreement a check
    rather than a restatement.
    """
    if dt <= 0.0 or not math.isfinite(dt):
        raise InvalidInputError("dt must be finite and > 0")
    v_arr = np.asarray(v, dtype=float)
    decay = math.exp(-kappa * dt)
    m = theta + (v_arr - theta) * decay
    s2 = v_arr * (xi * xi / kappa) * decay * (1.0 - decay) + theta * (
        xi * xi / (2.0 * kappa)
    ) * (1.0 - decay) ** 2
    return m, s2


def _quadratic_branch(m: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(a, b)` of `v' = a (b + Z)^2` matching `(m, psi m^2)`."""
    inv = 2.0 / psi
    b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
    return m / (1.0 + b2), np.sqrt(b2)


def _exponential_branch(m: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(p, beta)` of the atom-plus-exponential law matching `(m, psi m^2)`."""
    p = (psi - 1.0) / (psi + 1.0)
    beta = 2.0 / (m * (psi + 1.0))
    return p, beta


def qe_branch_moments(
    v: np.ndarray | float, dt: float, *, kappa: float, theta: float, xi: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(branch, mean, variance)` of the law QE samples, from its own parameters.

    `branch` is ``0`` for the quadratic branch and ``1`` for the exponential
    one. `mean` and `variance` are computed from `(a, b)` or `(p, beta)` --
    `a (1 + b^2)` and `a^2 (2 + 4 b^2)`; `(1-p)/beta` and
    `(1-p)(1+p)/beta^2` -- and **not** from the target moments, so comparing
    them with `heston_variance_moments` checks the moment-matching algebra
    itself rather than echoing it.
    """
    m, s2 = heston_variance_moments(v, dt, kappa=kappa, theta=theta, xi=xi)
    m = np.atleast_1d(np.asarray(m, dtype=float))
    s2 = np.atleast_1d(np.asarray(s2, dtype=float))
    psi = _psi(m, s2)

    branch = np.where(psi <= QE_PSI_C, 0, 1)
    mean = np.empty_like(m)
    variance = np.empty_like(m)

    lo = psi <= QE_PSI_C
    if lo.any():
        a, b = _quadratic_branch(m[lo], psi[lo])
        b2 = b * b
        mean[lo] = a * (1.0 + b2)
        variance[lo] = a * a * (2.0 + 4.0 * b2)
    hi = ~lo
    if hi.any():
        p, beta = _exponential_branch(m[hi], psi[hi])
        mean[hi] = (1.0 - p) / beta
        variance[hi] = (1.0 - p) * (1.0 + p) / (beta * beta)
    return branch, mean, variance


def qe_variance_step(
    v: np.ndarray,
    dt: float,
    *,
    kappa: float,
    theta: float,
    xi: float,
    z: np.ndarray,
) -> np.ndarray:
    """One QE variance step, driven by one standard normal per path.

    The quadratic branch reads `z` as a normal; the exponential branch reads it
    as the uniform `U = Phi(z)`, written through `1 - U = Phi(-z)` so that the
    upper tail keeps its precision (`Phi(z)` saturates at 1 for `z > 8.3`, and
    the inverse transform divides by `1 - U`).

    The output is non-negative by construction on both branches: a scaled
    square on one, an atom at zero plus a positive exponential draw on the
    other. That is the property Euler cannot have and is asserted directly in
    `tests/test_mc_heston.py`.
    """
    m, s2 = heston_variance_moments(v, dt, kappa=kappa, theta=theta, xi=xi)
    psi = _psi(m, s2)
    out = np.empty_like(m)

    lo = psi <= QE_PSI_C
    if lo.any():
        a, b = _quadratic_branch(m[lo], psi[lo])
        out[lo] = a * (b + z[lo]) ** 2
    hi = ~lo
    if hi.any():
        p, beta = _exponential_branch(m[hi], psi[hi])
        tail = np.maximum(ndtr(-z[hi]), _TINY)  # = 1 - U
        out[hi] = np.where(tail >= 1.0 - p, 0.0, np.log((1.0 - p) / tail) / beta)
    return out


def log_spot_coefficients(
    dt: float, *, kappa: float, theta: float, xi: float, rho: float
) -> tuple[float, float, float, float, float]:
    """`(K_0, K_1, K_2, K_3, K_4)` of the module docstring's log-spot step."""
    k0 = -rho * kappa * theta * dt / xi
    slope = kappa * rho / xi - 0.5
    k1 = QE_GAMMA_1 * dt * slope - rho / xi
    k2 = QE_GAMMA_2 * dt * slope + rho / xi
    orthogonal = 1.0 - rho * rho
    k3 = QE_GAMMA_1 * dt * orthogonal
    k4 = QE_GAMMA_2 * dt * orthogonal
    return k0, k1, k2, k3, k4


def _qe_martingale_k0(
    m: np.ndarray,
    psi: np.ndarray,
    *,
    a_coef: float,
    k1: float,
    k3: float,
    v: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    """`K_0*` per path for the QE branches, how many fell back, and which did not.

    `a_coef` is `A = K_2 + K_4/2`. Where the branch's moment generating
    function does not exist at `A` (`2 A a >= 1` on the quadratic branch,
    `A >= beta` on the exponential one) the uncorrected `K_0` is used instead,
    which is reported rather than hidden.
    """
    log_mgf = np.empty_like(m)
    usable = np.ones(m.shape, dtype=bool)

    lo = psi <= QE_PSI_C
    if lo.any():
        a, b = _quadratic_branch(m[lo], psi[lo])
        denom = 1.0 - 2.0 * a_coef * a
        ok = denom > 0.0
        safe = np.where(ok, denom, 1.0)
        log_mgf[lo] = a_coef * a * b * b / safe - 0.5 * np.log(safe)
        usable[lo] = ok
    hi = ~lo
    if hi.any():
        p, beta = _exponential_branch(m[hi], psi[hi])
        ok = beta > a_coef
        safe = np.where(ok, beta - a_coef, 1.0)
        log_mgf[hi] = np.log(p + beta * (1.0 - p) / safe)
        usable[hi] = ok

    corrected = -log_mgf - (k1 + 0.5 * k3) * v
    return corrected, int(np.count_nonzero(~usable)), usable


def _ncx2_martingale_k0(
    v: np.ndarray,
    dt: float,
    *,
    kappa: float,
    theta: float,
    xi: float,
    a_coef: float,
    k1: float,
    k3: float,
) -> tuple[np.ndarray, int, np.ndarray]:
    """`K_0*` for the exact variance transition, via the ncx2 m.g.f.

    For `v' = c X` with `X ~ chi'^2(d, lambda)`,
    `E[e^{A v'}] = exp(lambda A c / (1 - 2 A c)) (1 - 2 A c)^{-d/2}` when
    `2 A c < 1`; `(c, d, lambda)` come from
    `qpl.engines.mc.sde.cir_transition_parameters`.
    """
    df, nc, scale = cir_transition_parameters(v, dt, kappa=kappa, theta=theta, xi=xi)
    denom = 1.0 - 2.0 * a_coef * scale
    if denom <= 0.0:
        usable = np.zeros(np.shape(v), dtype=bool)
        return np.zeros(np.shape(v)), int(np.size(v)), usable
    log_mgf = nc * a_coef * scale / denom - 0.5 * df * math.log(denom)
    usable = np.ones(np.shape(v), dtype=bool)
    return -log_mgf - (k1 + 0.5 * k3) * v, 0, usable


def _draw_pair(
    rng: np.random.Generator, n_paths: int, n_pairs: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """One `(z_variance, z_spot)` column, antithetic when `n_pairs` is given."""
    if n_pairs is None:
        return rng.normal(size=n_paths), rng.normal(size=n_paths)
    base_v = rng.normal(size=n_pairs)
    base_s = rng.normal(size=n_pairs)
    return (
        np.concatenate([base_v, -base_v]),
        np.concatenate([base_s, -base_s]),
    )


def _clean_grid(t_grid: np.ndarray | list[float]) -> np.ndarray:
    grid = np.asarray(t_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise InvalidInputError("t_grid must be a 1D grid with at least two times")
    if not np.all(np.isfinite(grid)):
        raise InvalidInputError("t_grid must be finite")
    if grid[0] != 0.0:
        raise InvalidInputError("t_grid must start at 0")
    if np.any(np.diff(grid) <= 0.0):
        raise InvalidInputError("t_grid must be strictly increasing")
    return grid


def simulate_heston(
    model: HestonModel,
    *,
    s0: float,
    mu: float,
    t_grid: np.ndarray | list[float],
    n_paths: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    scheme: Literal[
        "qe", "euler_full_truncation", "exact_variance_euler_log_spot"
    ] = QE,
    martingale_correction: bool = True,
    antithetic: bool = False,
    store_paths: bool = True,
    return_normals: bool = False,
) -> HestonPaths:
    """Simulate `(ln S, v)` under Heston on `t_grid`.

    Parameters
    ----------
    model
        The Heston parameters. `v0` is the initial variance.
    s0
        Initial spot, `> 0`.
    mu
        The log-spot drift `r - q`.
    t_grid
        Strictly increasing times starting at `0.0`; one step per interval.
    n_paths
        Paths, `>= 1`, and even when `antithetic` is set.
    seed, rng
        Exactly one. Normals are drawn one column at a time so that a fine grid
        at a large path count costs `O(n_paths)` extra memory rather than
        `O(n_paths * n_steps)`.
    scheme
        One of `HESTON_SCHEMES`; see the module docstring.
    martingale_correction
        Whether to replace `K_0` by the value that makes
        `E[S_{t+dt} | S_t, v_t] = S_t e^{mu dt}` exactly. Read by `"qe"` and
        `"exact_variance_euler_log_spot"`; ignored by
        `"euler_full_truncation"`, whose drift has no moment generating
        function to invert, and `meta` says so.
    antithetic
        Reflect both driving normals for the second half of the paths. Refused
        for `"exact_variance_euler_log_spot"`, whose variance draw is not a
        function of a normal.
    store_paths
        Keep every column (needed by path-dependent payoffs) or only the
        terminal one.
    return_normals
        Also return the `(n_paths, n_steps)` normal blocks.

    Returns
    -------
    HestonPaths
    """
    if scheme not in HESTON_SCHEMES:
        raise InvalidInputError(f"scheme must be one of {HESTON_SCHEMES}")
    if not isinstance(model, HestonModel):
        raise InvalidInputError("model must be a HestonModel")
    if not math.isfinite(s0) or s0 <= 0.0:
        raise InvalidInputError("s0 must be finite and > 0")
    if not math.isfinite(mu):
        raise InvalidInputError("mu must be finite")
    if n_paths < 1:
        raise InvalidInputError("n_paths must be >= 1")
    if (seed is None) == (rng is None):
        raise InvalidInputError("pass exactly one of seed or rng")
    if antithetic:
        if scheme == EXACT_VARIANCE_EULER_LOG_SPOT:
            raise NotSupportedError(
                "antithetic sampling is not available for scheme "
                f"'{EXACT_VARIANCE_EULER_LOG_SPOT}': its variance transition is "
                "a noncentral chi-square draw, not a function of one normal per "
                "step, so reflecting the spot normal alone would reflect half "
                "the driver and the pair would not be an antithetic pair. Use "
                f"scheme='{QE}' or '{EULER_FULL_TRUNCATION}'."
            )
        if n_paths % 2 != 0:
            raise InvalidInputError(
                "n_paths must be even for antithetic sampling: the estimator's "
                "unit is a (Z, -Z) pair"
            )

    grid = _clean_grid(t_grid)
    n_steps = grid.size - 1
    dt_all = np.diff(grid)
    generator = rng if rng is not None else np.random.default_rng(seed)
    n_pairs = (n_paths // 2) if antithetic else None

    kappa, theta, xi, rho = model.kappa, model.theta, model.xi, model.rho
    correction_applies = scheme in (QE, EXACT_VARIANCE_EULER_LOG_SPOT)
    correcting = bool(martingale_correction) and correction_applies

    n_cols = n_steps + 1 if store_paths else 1
    log_spot = np.empty((n_paths, n_cols), dtype=float)
    variance = np.empty((n_paths, n_cols), dtype=float)
    x = np.full(n_paths, math.log(s0))
    v = np.full(n_paths, float(model.v0))
    if store_paths:
        log_spot[:, 0] = x
        variance[:, 0] = v

    z_variance = np.empty((n_paths, n_steps)) if return_normals else None
    z_spot = np.empty((n_paths, n_steps)) if return_normals else None

    fallbacks = 0
    negative_variance_steps = 0
    cond_mean = np.full(n_paths, math.log(s0))
    cond_var = np.zeros(n_paths)

    for i in range(n_steps):
        dt = float(dt_all[i])
        zv, zs = _draw_pair(generator, n_paths, n_pairs)
        if return_normals:
            z_variance[:, i] = zv  # type: ignore[index]
            z_spot[:, i] = zs  # type: ignore[index]

        if scheme == EULER_FULL_TRUNCATION:
            v_plus = np.maximum(v, 0.0)
            root = np.sqrt(v_plus)
            zx = rho * zv + math.sqrt(1.0 - rho * rho) * zs
            sqrt_dt = math.sqrt(dt)
            v_next = v + kappa * (theta - v_plus) * dt + xi * root * sqrt_dt * zv
            x = x + (mu - 0.5 * v_plus) * dt + root * sqrt_dt * zx
            # Conditioning on the variance driver conditions on `zv` as well,
            # so its share of the spot increment joins the conditional MEAN.
            cond_mean += (mu - 0.5 * v_plus) * dt + rho * root * sqrt_dt * zv
            cond_var += (1.0 - rho * rho) * v_plus * dt
            negative_variance_steps += int(np.count_nonzero(v_next < 0.0))
            v = v_next
        else:
            k0, k1, k2, k3, k4 = log_spot_coefficients(
                dt, kappa=kappa, theta=theta, xi=xi, rho=rho
            )
            a_coef = k2 + 0.5 * k4
            if scheme == QE:
                m, s2 = heston_variance_moments(
                    v, dt, kappa=kappa, theta=theta, xi=xi
                )
                psi = _psi(m, s2)
                v_next = qe_variance_step(
                    v, dt, kappa=kappa, theta=theta, xi=xi, z=zv
                )
                if correcting:
                    k0_vec, n_fallback, usable = _qe_martingale_k0(
                        m, psi, a_coef=a_coef, k1=k1, k3=k3, v=v
                    )
                    fallbacks += n_fallback
                    drift = np.where(usable, k0_vec, k0)
                else:
                    drift = k0
            else:
                df, nc, scale = cir_transition_parameters(
                    v, dt, kappa=kappa, theta=theta, xi=xi
                )
                v_next = np.asarray(
                    ncx2.rvs(df, nc, scale=scale, size=nc.shape, random_state=generator),
                    dtype=float,
                )
                if correcting:
                    k0_vec, n_fallback, usable = _ncx2_martingale_k0(
                        v, dt, kappa=kappa, theta=theta, xi=xi,
                        a_coef=a_coef, k1=k1, k3=k3,
                    )
                    fallbacks += n_fallback
                    drift = np.where(usable, k0_vec, k0)
                else:
                    drift = k0
            step_var = np.maximum(k3 * v + k4 * v_next, 0.0)
            spread = np.sqrt(step_var)
            step_mean = mu * dt + drift + k1 * v + k2 * v_next
            x = x + step_mean + spread * zs
            cond_mean = cond_mean + step_mean
            cond_var = cond_var + step_var
            v = v_next

        if store_paths:
            log_spot[:, i + 1] = x
            variance[:, i + 1] = v

    if not store_paths:
        log_spot[:, 0] = x
        variance[:, 0] = v

    meta: dict[str, object] = {
        "model": "Heston",
        "heston_scheme": scheme,
        "n_steps": n_steps,
        "n_paths": n_paths,
        "antithetic": antithetic,
        "martingale_correction": correcting,
        "martingale_correction_applies": correction_applies,
        "martingale_correction_fallbacks": fallbacks,
        "v0": model.v0,
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho": rho,
        "feller_number": model.feller_number,
    }
    if scheme == EULER_FULL_TRUNCATION:
        meta["negative_variance_draws"] = negative_variance_steps
        meta["negative_variance_fraction"] = negative_variance_steps / float(
            n_paths * n_steps
        )

    normals = (
        (z_variance, z_spot)  # type: ignore[arg-type]
        if return_normals
        else None
    )
    return HestonPaths(
        times=grid,
        log_spot=log_spot,
        variance=variance,
        conditional=ConditionalTerminalLaw(mean=cond_mean, variance=cond_var),
        normals=normals,
        meta=meta,
    )
