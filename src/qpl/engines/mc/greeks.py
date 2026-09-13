"""Monte Carlo Greeks: bump, pathwise and likelihood ratio, on the terminal law.

Three estimators of the same derivative, and they differ in *what* they
differentiate. Writing a discounted expectation as

    V(theta) = e^{-rT} integral f(x) p(x; theta) dx,

there are exactly three places a parameter can sit: in the payoff `f` (through
the state), in the density `p`, or in the discount factor. Each estimator picks
one of them and pays for it somewhere else.

- **Finite difference (Glasserman 2003, section 7.1).** Reprice at
  `theta +- h` on common random numbers and divide. It needs nothing from the
  model but it is biased at `O(h**2)` for a central difference, and the
  variance of the difference quotient is `Var(Y(theta+h) - Y(theta-h)) / (4 h**2 N)`.
  For a payoff that is Lipschitz in `theta` the numerator is `O(h**2)` and the
  ratio is `O(1/N)`; for a *discontinuous* payoff the numerator is `O(h)` (the
  paths that cross the jump, each contributing a full step) and the estimator's
  standard deviation grows like `h**-1/2`. That is the whole bias/variance
  trade-off and it is measured in `tests/test_mc_greeks_variance.py`.
- **Pathwise (Glasserman section 7.2; Broadie & Glasserman 1996, section 2).**
  Differentiate the *sample*: write `S_T` as a function of the parameter and a
  parameter-free random input `Z`, and estimate `E[d f(S_T)/d theta]`.
  Interchanging the derivative and the expectation is legitimate when
  `f(S_T(theta))` is almost surely differentiable in `theta` and Lipschitz in
  `theta` with an integrable Lipschitz constant (Glasserman's conditions in
  section 7.2.2). A call payoff satisfies both -- it fails to be differentiable
  only at `S_T = K`, a null event. A **digital does not**: its a.e. derivative
  is zero everywhere, so the interchange fails and the estimator converges to
  0, not to delta. That is not a technicality, it is a number, and
  `tests/test_mc_greeks.py` computes it.
- **Likelihood ratio (Glasserman section 7.3; Broadie & Glasserman 1996,
  section 3).** Differentiate the *density* and leave the payoff alone:

      dV/dtheta = e^{-rT} integral f(x) [d log p(x; theta) / d theta] p(x; theta) dx,

  so the estimator is `f(S_T)` times a score. It asks nothing at all of `f`
  beyond square integrability, which is why it is the estimator that works for
  a digital -- and it pays for that with variance, because the score has mean
  zero and unbounded support: multiplying the payoff by it strictly increases
  the variance relative to a pathwise estimator that exists.

The terminal law, once
----------------------
Every estimator below is written against the exact Black-Scholes terminal law,
with `mu = r - q`:

    S_T = S_0 exp( (mu - sigma**2/2) T + sigma sqrt(T) Z ),   Z ~ N(0, 1),
    Z   = ( log(S_T / S_0) - (mu - sigma**2/2) T ) / ( sigma sqrt(T) ).        (1)

`Z` is recovered from the sample by (1) rather than carried out of the sampler.
That is deliberate: with `n_steps > 1` the sampler draws `n_steps` normals per
path, but the *terminal* distribution is still exactly (1) with a single
standard normal, so the pathwise and likelihood-ratio derivations apply
unchanged and the recovered `Z` is the one they are written in. For an
antithetic pair the recovered `Z` of the mirrored path is exactly `-Z`, as it
must be.

Pathwise weights
----------------
From (1), holding `Z` fixed:

    dS_T/dS_0    = S_T / S_0                                                  (2)
    dS_T/dsigma  = S_T ( sqrt(T) Z - sigma T )                                (3)
    dS_T/dr      = S_T T          (q held fixed, so dmu/dr = 1)               (4)
    dS_T/dT      = S_T ( (mu - sigma**2/2) + sigma Z / (2 sqrt(T)) )          (5)

Equation (3) is the form Broadie & Glasserman write as `S_T (W_T - sigma T) / sigma`
after substituting `W_T = sigma sqrt(T) Z / sigma`; the two agree, and (3) is
used here because it needs no division by `sigma`.

With `Y = e^{-rT} f(S_T)` the estimators are

    delta_pw = e^{-rT} f'(S_T) S_T / S_0
    vega_pw  = e^{-rT} f'(S_T) S_T ( sqrt(T) Z - sigma T )
    rho_pw   = T e^{-rT} ( f'(S_T) S_T - f(S_T) )                             (6)
    theta_pw = e^{-rT} ( r f(S_T) - f'(S_T) S_T [ (mu - sigma**2/2) + sigma Z / (2 sqrt(T)) ] )

The second term in (6) is the discount factor's own `-T e^{-rT}` and is the
part a naive "differentiate the payoff" reading drops. For a call it makes the
estimator collapse to `T e^{-rT} K 1{S_T > K}`, whose mean is `K T e^{-rT} N(d2)`
-- the closed-form rho exactly, which is a useful check that (6) is right.
`theta` here is `-dV/dT` with `T` the time to expiry, the same convention the
analytic engine uses.

**Pathwise gamma does not exist** for any payoff in this package: it needs
`f''`, which is a Dirac mass for a call and worse for a digital. What is used
instead is the *mixed* LR-PW estimator (Glasserman section 7.4): differentiate
the pathwise delta's expectation with a score. Writing
`delta = e^{-rT} E[f'(S_T) S_T] / S_0` and applying the `S_0` score below,

    gamma_mixed = e^{-rT} f'(S_T) S_T ( Z / (sigma sqrt(T)) - 1 ) / S_0**2    (7)

which differentiates the payoff once and the density once. Measured against
pure LR gamma in `tests/test_mc_greeks_variance.py`.

Likelihood-ratio scores
-----------------------
The density of `S_T` is lognormal, `p(x) = phi(z) / (x sigma sqrt(T))` with `z`
from (1). Differentiating `log p = -log x - log(sigma sqrt(T)) - log sqrt(2 pi) - z**2 / 2`
and using `dz/dS_0 = -1/(S_0 sigma sqrt(T))`, `dz/dsigma = -(z/sigma) - sqrt(T)`,
`dz/dr = -sqrt(T)/sigma`, `dz/dT = -z/(2T) - (mu - sigma**2/2)/(sigma sqrt(T))`:

    d log p / dS_0   = z / ( S_0 sigma sqrt(T) )                              (8)
    d2 p / dS_0**2 / p = ( z**2 - z sigma sqrt(T) - 1 ) / ( S_0**2 sigma**2 T )  (9)
    d log p / dsigma = ( z**2 - 1 ) / sigma - z sqrt(T)                      (10)
    d log p / dr     = z sqrt(T) / sigma                                     (11)
    d log p / dT     = ( z**2 - 1 ) / (2 T) + z ( mu - sigma**2/2 ) / ( sigma sqrt(T) )  (12)

Equation (9) is the second-order score `d**2 p / dS_0**2 / p = (d log p/dS_0)**2 + d**2 log p / dS_0**2`,
which is what the second derivative of the integral produces; it is *not* the
square of (8). The estimators are `e^{-rT} f(S_T)` times (8), (9), (10),
`(11) - T` and `r - (12)` for delta, gamma, vega, rho and theta, the extra `-T`
and `r` coming from the discount factor in each case.

Every score above has mean zero (they are scores), so an LR estimator of a
Greek is the payoff -- an `O(1)` quantity -- multiplied by a mean-zero weight
whose scale is `1 / (S_0 sigma sqrt(T))` for delta. That factor is the
`1/sqrt(T)` blow-up Glasserman notes in section 7.3: as maturity shortens the
LR delta's variance grows like `1/T`, and it is measured at `T = 1` against
`T = 0.05` in `tests/test_mc_greeks_variance.py`.

Variance reduction composes, with one exception
-----------------------------------------------
The estimator samples above are per-path functions of the same draws the price
uses, so the Slice 7 machinery applies to them unchanged: an antithetic unit is
the pair average of the two per-path Greek samples, and a stratified estimator
is the strata-weighted mean of them. Both are reductions of the *sample*, and
they do not care that the sample is a Greek rather than a price.

The control variate is the exception, and it is implemented for **delta only**.
The Slice 7 control is `X = e^{-rT} S_T`, with the exact mean
`e^{-rT} S_0 e^{mu T}`. Controlling a Greek needs the estimator applied to the
control *and* the exact value of the corresponding derivative of `E[X]`. For
delta that derivative is `e^{-rT} e^{mu T}`, so the control variable is the
same estimator (pathwise, LR or bump) applied to the payoff `g(x) = x` and its
mean is known in closed form. For the other Greeks the derivative of
`E[X] = S_0 e^{-qT}` is either identically zero (vega, rho, and gamma) or
depends on how the curve is read at a shifted maturity (theta), and a control
whose mean is exactly zero buys only whatever correlation it happens to have;
rather than ship four controls of unstated value, the other Greeks are
estimated **without** the control and `meta["control_variate_greeks"]` says so.
Under `greeks_estimator="bump"` the control is applied at the *price* level
instead, which is what Slice 7 already did and what keeps this slice's bump
numbers identical to the pre-slice ones.

Reference: Glasserman, P. (2003), *Monte Carlo Methods in Financial
Engineering*, chapter 7 (7.1 finite differences, 7.2 pathwise, 7.3 likelihood
ratio, 7.4 combining them); Broadie, M. and Glasserman, P. (1996), "Estimating
security price derivatives using simulation", *Management Science* 42(2),
269-285; Chen, N. and Glasserman, P. (2007), "Malliavin Greeks without
Malliavin calculus", *Stochastic Processes and their Applications* 117,
1689-1723, for combined estimators generally. Every formula above was
re-derived here from the lognormal transition density; no expression, number or
table is transcribed from those sources.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtri

from ...exceptions import InvalidInputError, NotSupportedError
from ..base import GreeksResult
from .variance_reduction import (
    ANTITHETIC,
    CONTROL_VARIATE,
    STRATIFIED,
    TerminalSample,
    estimate_from_sample,
    terminal_spots_from_normals,
    validate_sampler,
)

__all__ = [
    "BUMP",
    "CONTROL_VARIATE_GREEKS",
    "GREEKS_ESTIMATORS",
    "GREEK_NAMES",
    "LIKELIHOOD_RATIO",
    "MIXED_PATHWISE_LR",
    "PATHWISE",
    "GreekEstimate",
    "PathSample",
    "bump_estimates",
    "control_samples",
    "draw_path_sample",
    "estimate_greek",
    "greeks_result",
    "likelihood_ratio_terminal_greeks",
    "normalise_greeks_estimator",
    "one_sided_estimate",
    "pathwise_terminal_greeks",
    "reduce_to_units",
    "scenario",
    "terminal_normal_from_spot",
]

BUMP = "bump"
PATHWISE = "pathwise"
LIKELIHOOD_RATIO = "likelihood_ratio"
MIXED_PATHWISE_LR = "pathwise_lr_mixed"
"""Name recorded in `meta["estimator"]` for the gamma of equation (7): the
payoff is differentiated once and the density once, so it belongs to neither
family and says so."""

GREEKS_ESTIMATORS = (BUMP, PATHWISE, LIKELIHOOD_RATIO)
GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")

CONTROL_VARIATE_GREEKS = ("delta",)
"""Greeks the `control_variate` reduction is applied to under `pathwise` and
`likelihood_ratio`; see the module docstring. Reported in
`meta["control_variate_greeks"]` so that a caller reading a suspiciously
unchanged vega knows the control was not applied to it."""


def normalise_greeks_estimator(value: object) -> str:
    """Validate `MCConfig.greeks_estimator` and return it.

    Raises
    ------
    InvalidInputError
        On anything that is not one of :data:`GREEKS_ESTIMATORS`.
    """
    if not isinstance(value, str):
        raise InvalidInputError("greeks_estimator must be a string")
    if value not in GREEKS_ESTIMATORS:
        raise InvalidInputError(
            f"unknown greeks_estimator '{value}'; expected one of {GREEKS_ESTIMATORS}"
        )
    return value


@dataclass(frozen=True)
class GreekEstimate:
    """One Greek: its value, the standard error of *that estimator*, its name."""

    value: float
    stderr: float
    estimator: str


@dataclass(frozen=True)
class PathSample:
    """Terminal spots and the recovered terminal normal, per path.

    Parameters
    ----------
    spots
        `(n_paths,)` terminal spots.
    z
        `(n_paths,)` standard normals recovered by equation (1). Recovered
        rather than carried, so the estimators are correct for any `n_steps`.
    stratum, n_strata
        Stratum labels and count, or `None`/`0` when unstratified.
    n_normal_draws, n_paths
        Cost bookkeeping, matching `TerminalSample`.
    """

    spots: np.ndarray
    z: np.ndarray
    stratum: np.ndarray | None
    n_strata: int
    n_normal_draws: int
    n_paths: int


def terminal_normal_from_spot(
    spots: np.ndarray, *, s0: float, mu: float, sigma: float, t: float
) -> np.ndarray:
    """Equation (1): recover `Z` from the terminal spot."""
    if sigma <= 0.0 or t <= 0.0:
        raise InvalidInputError("terminal normal needs sigma > 0 and t > 0")
    return (np.log(np.asarray(spots, dtype=float) / s0) - (mu - 0.5 * sigma * sigma) * t) / (
        sigma * math.sqrt(t)
    )


def draw_path_sample(
    *,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: int,
    methods: tuple[str, ...],
    n_strata: int,
) -> PathSample:
    """Draw the same normals the price engine would, and keep the paths.

    The plain branch draws `n_paths` normals per step in a loop, which is what
    `qpl.engines.mc.processes.simulate_gbm_exact` does; the antithetic and
    stratified branches reproduce
    `qpl.engines.mc.variance_reduction.simulate_terminal_sample`. Matching the
    generator's consumption pattern is what makes a bump estimator built on
    this sample **bit-for-bit** the pre-slice price-level one, and it is what
    lets a user compare a price and a Greek at the same seed and know they came
    from the same paths.
    """
    validate_sampler(methods=methods, n_paths=n_paths, n_steps=n_steps, n_strata=n_strata)
    rng = np.random.default_rng(seed)

    if ANTITHETIC in methods:
        n_pairs = n_paths // 2
        base = rng.normal(size=(n_pairs, n_steps))
        z = np.concatenate([base, -base], axis=0)
        stratum: np.ndarray | None = None
        n_strata_used = 0
        n_draws = n_pairs * n_steps
    elif STRATIFIED in methods:
        per_stratum = n_paths // n_strata
        stratum = np.repeat(np.arange(n_strata), per_stratum)
        u = (stratum + rng.random(n_paths)) / n_strata
        z = ndtri(u).reshape(n_paths, 1)
        n_strata_used = n_strata
        n_draws = n_paths
    else:
        z = np.empty((n_paths, n_steps), dtype=float)
        for step in range(n_steps):
            z[:, step] = rng.normal(size=n_paths)
        stratum = None
        n_strata_used = 0
        n_draws = n_paths * n_steps

    spots = terminal_spots_from_normals(z, s0=s0, mu=mu, sigma=sigma, t=t)
    return PathSample(
        spots=spots,
        z=terminal_normal_from_spot(spots, s0=s0, mu=mu, sigma=sigma, t=t),
        stratum=stratum,
        n_strata=n_strata_used,
        n_normal_draws=n_draws,
        n_paths=n_paths,
    )


def reduce_to_units(values: np.ndarray, *, methods: tuple[str, ...]) -> np.ndarray:
    """Per-path samples to per-estimator-unit samples.

    The antithetic unit is the pair average, exactly as it is for the price
    (Slice 7); everything else is one unit per path. Reducing the Greek sample
    the same way the price sample is reduced is what makes the reported
    standard error the standard error of the estimator actually used.
    """
    if ANTITHETIC not in methods:
        return values
    n_pairs = values.size // 2
    return 0.5 * (values[:n_pairs] + values[n_pairs:])


def estimate_greek(
    values: np.ndarray,
    *,
    sample: PathSample,
    methods: tuple[str, ...],
    estimator: str,
    control: np.ndarray | None = None,
    control_mean: float | None = None,
) -> GreekEstimate:
    """Reduce a per-path Greek sample to a value and its standard error.

    Routes through `qpl.engines.mc.variance_reduction.estimate_from_sample`, so
    the antithetic pair `ddof=1`, the control-variate residual `ddof=2` and the
    strata-weighted standard errors are the Slice 7 ones and not a second copy.
    """
    units = reduce_to_units(np.asarray(values, dtype=float), methods=methods)
    use_control = control is not None and control_mean is not None
    if use_control:
        control_units = reduce_to_units(np.asarray(control, dtype=float), methods=methods)
        active = methods
    else:
        control_units = np.zeros_like(units)
        active = tuple(name for name in methods if name != CONTROL_VARIATE)
    template = TerminalSample(
        y=units,
        x=control_units,
        x_mean=float(control_mean) if use_control else 0.0,
        stratum=sample.stratum,
        n_strata=sample.n_strata,
        n_normal_draws=sample.n_normal_draws,
        n_paths=sample.n_paths,
        control_name="pathwise_derivative_of_discounted_terminal_spot",
    )
    reduced = estimate_from_sample(template, methods=active)
    return GreekEstimate(value=reduced.value, stderr=reduced.stderr, estimator=estimator)


def pathwise_terminal_greeks(
    sample: PathSample,
    *,
    payoff: np.ndarray,
    payoff_derivative: np.ndarray,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    r: float,
    discount_factor: float,
) -> dict[str, np.ndarray]:
    """Per-path pathwise samples; gamma is the mixed estimator of equation (7).

    `payoff` and `payoff_derivative` are `f(S_T)` and `f'(S_T)` evaluated on the
    sample. The caller supplies both because "what is the derivative of this
    payoff" is a statement about the contract, not about the estimator -- and
    for a digital the honest answer is a zero array, which is exactly why the
    digital engine refuses this estimator rather than calling this function.
    """
    z = sample.z
    spots = sample.spots
    sqrt_t = math.sqrt(t)
    fp_s = payoff_derivative * spots

    return {
        "delta": discount_factor * fp_s / s0,
        # Equation (7): the mixed LR-PW estimator, because f'' does not exist.
        "gamma": discount_factor * fp_s * (z / (sigma * sqrt_t) - 1.0) / (s0 * s0),
        "vega": discount_factor * fp_s * (sqrt_t * z - sigma * t),
        "theta": discount_factor
        * (
            r * payoff
            - fp_s * ((mu - 0.5 * sigma * sigma) + sigma * z / (2.0 * sqrt_t))
        ),
        "rho": t * discount_factor * (fp_s - payoff),
    }


def likelihood_ratio_terminal_greeks(
    sample: PathSample,
    *,
    payoff: np.ndarray,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    r: float,
    discount_factor: float,
) -> dict[str, np.ndarray]:
    """Per-path likelihood-ratio samples: the payoff times the scores (8)-(12).

    Nothing here touches the payoff's smoothness, which is the point: the same
    five expressions serve a vanilla and a digital.
    """
    z = sample.z
    sqrt_t = math.sqrt(t)
    y = discount_factor * payoff

    score_s0 = z / (s0 * sigma * sqrt_t)
    score2_s0 = (z * z - z * sigma * sqrt_t - 1.0) / (s0 * s0 * sigma * sigma * t)
    score_sigma = (z * z - 1.0) / sigma - z * sqrt_t
    score_r = z * sqrt_t / sigma
    score_t = (z * z - 1.0) / (2.0 * t) + z * (mu - 0.5 * sigma * sigma) / (sigma * sqrt_t)

    return {
        "delta": y * score_s0,
        "gamma": y * score2_s0,
        "vega": y * score_sigma,
        "theta": y * (r - score_t),
        "rho": y * (score_r - t),
    }


def control_samples(
    sample: PathSample,
    *,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    discount_factor: float,
    estimator: str,
) -> tuple[np.ndarray, float]:
    """The delta control variable and its exact mean.

    The control payoff is `g(x) = x`, so the controlled quantity is
    `d/dS_0 E[e^{-rT} S_T] = e^{-rT} e^{mu T}`, known exactly. Under `pathwise`
    the control variable is `e^{-rT} S_T / S_0` (equation 2 with `g' = 1`);
    under `likelihood_ratio` it is `e^{-rT} S_T` times the score (8). Both have
    that same mean, which is what makes them controls rather than decorations.
    """
    known_mean = discount_factor * math.exp(mu * t)
    if estimator == PATHWISE:
        return discount_factor * sample.spots / s0, known_mean
    if estimator == LIKELIHOOD_RATIO:
        z = sample.z
        return (
            discount_factor * sample.spots * z / (s0 * sigma * math.sqrt(t)),
            known_mean,
        )
    raise NotSupportedError(
        f"no delta control variable is defined for greeks_estimator '{estimator}'"
    )


# ---------------------------------------------------------------------------
# The bump estimator, at sample level.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One repriced market state: its value and its per-unit sample.

    `value` is computed exactly the way `price_european` computes it (mean of
    the discounted payoff, or the Slice 7 reduced estimator), so a difference of
    two `value`s is **bit-for-bit** the pre-slice price-level bump. `units` is
    the same sample before it was averaged, which is what a paired difference
    needs in order to report a standard error at all.
    """

    value: float
    units: np.ndarray


def scenario(
    *,
    payoff: Callable[[np.ndarray], np.ndarray],
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    discount_factor: float,
    n_paths: int,
    n_steps: int,
    seed: int,
    methods: tuple[str, ...],
    n_strata: int,
) -> Scenario:
    """Price one market state and keep the sample that produced the price."""
    if sigma == 0.0:
        # The `price_european` degenerate branch: the path is deterministic, so
        # there is one number and no sampling error. Reproduced here rather than
        # simulated, down to the `math.exp` that produces the forward.
        forward = s0 * math.exp(mu * t)
        value = discount_factor * float(np.asarray(payoff(np.array([forward])))[0])
        return Scenario(value=value, units=np.full(2, value))

    sample = draw_path_sample(
        s0=s0,
        mu=mu,
        sigma=sigma,
        t=t,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        methods=methods,
        n_strata=n_strata,
    )
    y = discount_factor * np.asarray(payoff(sample.spots), dtype=float)
    if not methods:
        return Scenario(value=float(np.mean(y)), units=y)

    template = TerminalSample(
        y=reduce_to_units(y, methods=methods),
        x=reduce_to_units(discount_factor * sample.spots, methods=methods),
        x_mean=discount_factor * s0 * math.exp(mu * t),
        stratum=sample.stratum,
        n_strata=sample.n_strata,
        n_normal_draws=sample.n_normal_draws,
        n_paths=sample.n_paths,
    )
    reduced = estimate_from_sample(template, methods=methods)
    units = template.y
    if CONTROL_VARIATE in methods:
        from .variance_reduction import control_variate_coefficient

        beta = control_variate_coefficient(template.y, template.x)
        units = template.y - beta * (template.x - template.x_mean)
    return Scenario(value=reduced.value, units=units)


def bump_estimates(
    *,
    base: Scenario,
    up: Scenario,
    down: Scenario,
    step: float,
    order: int = 1,
    estimator: str = BUMP,
    stratum: np.ndarray | None = None,
    n_strata: int = 0,
    ddof: int = 1,
) -> GreekEstimate:
    """A central difference of two scenarios, with its paired standard error.

    `order=1` is `(up - down) / (2 h)` and `order=2` is
    `(up - 2 base + down) / h**2`. The *value* is formed from the scenario
    values (a difference of means) and the *standard error* from the per-unit
    differences (a mean of differences); the two agree to round-off and only the
    second one can be given an error bar, because common random numbers make
    the paths paired and a difference of two independent standard errors would
    be an overstatement of several orders of magnitude.
    """
    if order == 1:
        value = (up.value - down.value) / (2.0 * step)
        per_unit = (up.units - down.units) / (2.0 * step)
    elif order == 2:
        value = (up.value - 2.0 * base.value + down.value) / (step * step)
        per_unit = (up.units - 2.0 * base.units + down.units) / (step * step)
    else:  # pragma: no cover - guarded by the callers in this package
        raise InvalidInputError("order must be 1 or 2")
    return GreekEstimate(
        value=value,
        stderr=_sample_stderr(per_unit, stratum=stratum, n_strata=n_strata, ddof=ddof),
        estimator=estimator,
    )


def one_sided_estimate(
    *,
    base: Scenario,
    shifted: Scenario,
    step: float,
    estimator: str = BUMP,
    stratum: np.ndarray | None = None,
    n_strata: int = 0,
    ddof: int = 1,
) -> GreekEstimate:
    """`(shifted - base) / h`, paired. Used for the pre-slice backward theta."""
    return GreekEstimate(
        value=(shifted.value - base.value) / step,
        stderr=_sample_stderr(
            (shifted.units - base.units) / step,
            stratum=stratum,
            n_strata=n_strata,
            ddof=ddof,
        ),
        estimator=estimator,
    )


def _sample_stderr(
    values: np.ndarray, *, stratum: np.ndarray | None, n_strata: int, ddof: int = 1
) -> float:
    """Standard error of a paired difference sample, strata-weighted if stratified.

    `ddof=2` is passed when the scenario samples were control-variate adjusted:
    one degree of freedom went to the mean and one to the fitted slope, exactly
    as in `qpl.engines.mc.variance_reduction.estimate_from_sample`. The
    stratified branch keeps `ddof=1` *within* each stratum, which is the Slice 7
    convention: the strata-weighted variance is a sum of within-stratum
    variances and each of those loses one degree of freedom to its own mean.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    if stratum is None or n_strata <= 0:
        return float(np.std(arr, ddof=ddof) / math.sqrt(arr.size))
    per_stratum = arr.size // n_strata
    block = arr.reshape(n_strata, per_stratum)
    variance = float(block.var(axis=1, ddof=1).sum() / (n_strata * n_strata * per_stratum))
    return math.sqrt(max(variance, 0.0))


def greeks_result(
    estimates: dict[str, GreekEstimate], meta: dict[str, Any]
) -> GreeksResult:
    """Assemble a `GreeksResult`, recording each Greek's stderr and estimator.

    `meta["stderr"]` and `meta["estimator"]` are per-Greek dictionaries rather
    than single values, because a result can legitimately mix families: a
    pathwise call carries a mixed LR-PW gamma, and an Asian carries bumped
    rho and theta next to pathwise delta and vega.
    """
    meta = dict(meta)
    meta["stderr"] = {name: estimates[name].stderr for name in GREEK_NAMES}
    meta["estimator"] = {name: estimates[name].estimator for name in GREEK_NAMES}
    return GreeksResult(
        delta=estimates["delta"].value,
        gamma=estimates["gamma"].value,
        vega=estimates["vega"].value,
        theta=estimates["theta"].value,
        rho=estimates["rho"].value,
        meta=meta,
    )
