"""Variance reduction for the Monte Carlo engines.

Reference: Glasserman (2003), *Monte Carlo Methods in Financial Engineering*,
chapter 4 -- control variates (4.1), antithetic variates (4.2), stratified
sampling (4.3). Every estimator below was re-derived here; the measured
numbers live in `tests/test_mc_variance_ratios.py` and
`docs/notes/mc_variance_reduction.md`, and none of them is quoted from that
book.

What is shared by all three
---------------------------
The plain estimator draws `N` independent normals `Z_i`, maps each to a
terminal spot, and averages the discounted payoff `Y_i = e^{-rT} f(S_T(Z_i))`.
Its variance is `Var(Y)/N` and its standard error is the `ddof=1` sample
standard deviation over `sqrt(N)`. Each method below keeps the *mean* and
attacks the *variance*, and each therefore has its own standard-error formula.
Reporting the plain formula for a reduced estimator is the classic way to
report a confidence interval that does not cover; the whole point of putting
the estimators in one module is that the estimate and its standard error are
produced together.

**Antithetic (Glasserman 4.2).** Draw `m = N/2` normals and use both `Z` and
`-Z`. The unit of the estimator is then the *pair average*
`P_i = (Y(Z_i) + Y(-Z_i))/2`, and the `P_i` are i.i.d. -- so the standard error
is the ordinary `ddof=1` one over `m` pairs, and it is *not* the standard error
of the `2m` individual payoffs, which are not independent. At equal cost
measured in normal draws, the variance ratio is `2 / (1 + rho_a)` with
`rho_a = corr(Y(Z), Y(-Z))`: `m` draws buy `Var(Y)/m` plain and
`Var(Y)(1 + rho_a)/(2m)` antithetic. Measured at equal *payoff evaluations*
the same ratio reads `1 / (1 + rho_a)`; both conventions are reported in the
note, and the tests assert the normal-draw one. The method helps exactly when
`rho_a < 0`, which monotone `f` guarantees (Glasserman 4.2 derives this from
the fact that `Z -> -Z` reverses the order); it cannot lose more than a factor
of 2 and cannot gain unboundedly for a payoff whose antithetic image is not
close to its own reflection.

**Control variate (Glasserman 4.1).** The discounted terminal spot
`X = e^{-rT} S_T` is a martingale under the pricing measure, so its mean is
known exactly: `E[X] = e^{-rT} S_0 e^{(r-q)T} = S_0 e^{-qT}` (this module uses
the first form, which stays exact when the discount factor comes from a curve
rather than from a flat `r`). The estimator is
`mean(Y) - b (mean(X) - E[X])`, and the variance-minimising `b` is
`Cov(Y, X)/Var(X)`, giving a variance ratio of `1/(1 - rho**2)` with
`rho = corr(Y, X)`. That coefficient is unknown and is estimated here **on the
same sample**, which makes the estimator a ratio of sample moments and hence
**biased**; the bias is `O(1/N)` while the standard error is `O(1/sqrt(N))`,
so it vanishes relative to the reported uncertainty and is invisible at any `N`
where the confidence interval is meaningful. `tests/test_mc_variance_reduction.py`
measures the same-sample versus pilot-sample coefficient gap and its decay
rather than asserting the bias away. The reported standard error is the sample
standard deviation of the regression residuals with `ddof=2`, one degree of
freedom for the mean and one for the estimated slope.

**Stratified (Glasserman 4.3).** For `n_steps = 1` the terminal spot is a
monotone function of a single standard normal, so stratifying that normal
stratifies the payoff. `K` equal-probability strata with `n_paths / K` draws
each (proportional allocation, which for equal-probability strata is equal
allocation) give the estimator `(1/K) sum_i mean_i(Y)` and the variance
`(1/K**2) sum_i s_i**2 / m`. The gain is `Var(Y) / Var_within`, so it is
bounded by how much of the payoff's variance survives *inside* a stratum --
which is why the tail stratum, where the payoff is unbounded, is what limits
the gain for a vanilla call (measured; see the note).

Why stratification stops at `n_steps = 1`
-----------------------------------------
Stratifying the single normal that drives the terminal price is a statement
about the *terminal* distribution. With `n_steps > 1` the terminal spot is
driven by a sum of `n_steps` normals and there is no single scalar to
stratify; the standard construction is to stratify a linear projection of the
Brownian path (typically the terminal value) and fill in the rest with a
Brownian bridge, which is a different sampler, not a different bookkeeping of
this one. `NotSupportedError` says so rather than quietly stratifying the
first step. Antithetic and control variates have no such restriction: both are
transformations of whatever normals the path sampler used.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtri

from ...exceptions import InvalidInputError, NotSupportedError

__all__ = [
    "ANTITHETIC",
    "CONTROL_VARIATE",
    "NONE",
    "STRATIFIED",
    "TerminalSample",
    "VarianceReductionEstimate",
    "control_variate_coefficient",
    "estimate_from_sample",
    "normalise_variance_reduction",
    "price_with_variance_reduction",
    "simulate_terminal_sample",
    "terminal_spots_from_normals",
]

NONE = "none"
ANTITHETIC = "antithetic"
CONTROL_VARIATE = "control_variate"
STRATIFIED = "stratified"

_KNOWN = (NONE, ANTITHETIC, CONTROL_VARIATE, STRATIFIED)

# Canonical order inside the normalised tuple: the base sampler first (it
# decides which normals are drawn), then the control variate (which is a
# post-processing regression on whatever sample the base sampler produced).
_CANONICAL_ORDER = (ANTITHETIC, STRATIFIED, CONTROL_VARIATE)


def normalise_variance_reduction(value: object) -> tuple[str, ...]:
    """Validate `MCConfig.variance_reduction` and return it in canonical form.

    Accepts a single name or a tuple/list of names. Returns `()` for
    `"none"` (and for an empty tuple), so a caller tests
    `if methods:` rather than comparing against a string.

    Raises
    ------
    InvalidInputError
        On an unknown name, a repeated name, `"none"` mixed with a real
        method, or a combination that does not compose (see below).

    Notes
    -----
    `antithetic` and `stratified` are both *samplers* of the terminal normal
    and do not compose here: reflecting a stratified draw sends it out of its
    own stratum into the mirror stratum, so the pair-average unit and the
    per-stratum bookkeeping describe different partitions of the sample. The
    construction that does both (symmetric/antithetic stratification) is a
    third sampler, not a combination of these two, so the combination is
    rejected rather than silently interpreted.
    """
    if isinstance(value, str):
        names: tuple[str, ...] = (value,)
    elif isinstance(value, (tuple, list)):
        if any(not isinstance(name, str) for name in value):
            raise InvalidInputError("variance_reduction names must be strings")
        names = tuple(value)
    else:
        raise InvalidInputError(
            "variance_reduction must be a string or a tuple of strings"
        )

    if not names:
        return ()
    for name in names:
        if name not in _KNOWN:
            raise InvalidInputError(
                f"unknown variance_reduction '{name}'; expected one of {_KNOWN} "
                "or a tuple of them"
            )
    if len(set(names)) != len(names):
        raise InvalidInputError("variance_reduction names must not repeat")
    if NONE in names:
        if len(names) > 1:
            raise InvalidInputError(
                "variance_reduction 'none' cannot be combined with another method"
            )
        return ()
    if ANTITHETIC in names and STRATIFIED in names:
        raise InvalidInputError(
            "variance_reduction 'antithetic' and 'stratified' do not compose: "
            "reflecting a stratified draw moves it into the mirror stratum, so "
            "the pair-average unit and the per-stratum unit are different "
            "partitions of the same sample. Symmetric (antithetic) "
            "stratification is a separate sampler and is not implemented."
        )
    return tuple(name for name in _CANONICAL_ORDER if name in names)


def terminal_spots_from_normals(
    z: np.ndarray,
    *,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
) -> np.ndarray:
    """Terminal spots from a `(n_paths, n_steps)` array of standard normals.

    Same exact-lognormal recursion as
    `qpl.engines.mc.processes.simulate_gbm_exact`, applied to normals supplied
    by the caller instead of drawn internally, and keeping only the endpoint.
    With `n_steps > 1` the recursion is a product of `n_steps` factors rather
    than one factor in the summed normal; the two agree in exact arithmetic and
    this one is what the plain engine does, so the comparison between reduced
    and plain estimators is not confounded by a different rounding path.
    """
    z_arr = np.asarray(z, dtype=float)
    if z_arr.ndim != 2:
        raise InvalidInputError("z must be a 2D (n_paths, n_steps) array")
    n_steps = z_arr.shape[1]
    dt = t / n_steps
    drift_step = (mu - 0.5 * sigma * sigma) * dt
    vol_step = sigma * math.sqrt(dt)

    spots = np.full(z_arr.shape[0], float(s0))
    for step in range(n_steps):
        spots = spots * np.exp(drift_step + vol_step * z_arr[:, step])
    return spots


@dataclass(frozen=True)
class TerminalSample:
    """One realised sample, reduced to what every estimator below needs.

    Parameters
    ----------
    y
        Discounted payoff per estimator *unit*: one entry per path for the
        plain and stratified samplers, one entry per antithetic **pair** (the
        pair average) for the antithetic sampler. The unit is the thing that
        is i.i.d. (or, under stratification, i.i.d. within its stratum).
    x
        The control variable `e^{-rT} S_T`, reduced the same way as `y`, so
        that `y` and `x` are paired unit by unit.
    x_mean
        `E[x]`, known exactly: the discounted spot is a martingale, and a pair
        average of it has the same mean as a single draw.
    stratum
        Stratum index per unit, or `None` when the sample is unstratified.
    n_strata
        Number of strata, or `0` when unstratified.
    n_normal_draws
        Standard normals actually drawn. This is the cost axis the variance
        ratios are measured on: `n_paths * n_steps` plain and stratified,
        `(n_paths / 2) * n_steps` antithetic.
    n_paths
        Paths evaluated (`2 * len(y)` for antithetic, `len(y)` otherwise).
    """

    y: np.ndarray
    x: np.ndarray
    x_mean: float
    stratum: np.ndarray | None
    n_strata: int
    n_normal_draws: int
    n_paths: int


@dataclass(frozen=True)
class VarianceReductionEstimate:
    """A price estimate, its standard error, and the estimator's own metadata."""

    value: float
    stderr: float
    meta: dict[str, Any]


def control_variate_coefficient(y: np.ndarray, x: np.ndarray) -> float:
    """`Cov(y, x) / Var(x)`, the variance-minimising control coefficient.

    Ordinary least squares slope of `y` on `x` (Glasserman 4.1.2). Returns
    `0.0` when the control has no sample variance, which makes the control
    variate a no-op rather than a division by zero -- that happens only in
    degenerate samples (`sigma = 0`, or a single unit).
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    x_centred = x_arr - x_arr.mean()
    denom = float(np.dot(x_centred, x_centred))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x_centred, y_arr - y_arr.mean()) / denom)


def _stratified_normals(
    *, n_paths: int, n_strata: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-allocation stratified standard normals, with their stratum labels.

    Stratum `i` is the interval of probability `[i/K, (i+1)/K)`; the inverse
    normal CDF maps a uniform drawn inside it to a normal conditioned on that
    interval. `u = 0` is representable (with probability `2**-53`) and maps to
    `-inf`, i.e. a terminal spot of exactly zero -- a finite payoff, not a NaN,
    so it is left alone rather than clipped to a value that would bias the
    stratum.
    """
    per_stratum = n_paths // n_strata
    stratum = np.repeat(np.arange(n_strata), per_stratum)
    u = (stratum + rng.random(n_paths)) / n_strata
    return ndtri(u), stratum


def simulate_terminal_sample(
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
) -> TerminalSample:
    """Draw one sample under the requested sampler and reduce it to units.

    `methods` is the output of :func:`normalise_variance_reduction`; only its
    *sampler* part matters here (`antithetic`, `stratified`, or neither).
    `control_variate` changes nothing about the draw -- the control variable is
    computed unconditionally, because it costs one multiplication and makes the
    correlation reportable even when the control is not applied.
    """
    validate_sampler(methods=methods, n_paths=n_paths, n_steps=n_steps, n_strata=n_strata)
    rng = np.random.default_rng(seed)

    if ANTITHETIC in methods:
        n_pairs = n_paths // 2
        base = rng.normal(size=(n_pairs, n_steps))
        z = np.concatenate([base, -base], axis=0)
        stratum = None
        n_strata_used = 0
        n_draws = n_pairs * n_steps
    elif STRATIFIED in methods:
        z_flat, stratum = _stratified_normals(
            n_paths=n_paths, n_strata=n_strata, rng=rng
        )
        z = z_flat.reshape(n_paths, 1)
        n_strata_used = n_strata
        n_draws = n_paths
    else:
        z = rng.normal(size=(n_paths, n_steps))
        stratum = None
        n_strata_used = 0
        n_draws = n_paths * n_steps

    spots = terminal_spots_from_normals(z, s0=s0, mu=mu, sigma=sigma, t=t)
    y = discount_factor * np.asarray(payoff(spots), dtype=float)
    x = discount_factor * spots

    if ANTITHETIC in methods:
        n_pairs = n_paths // 2
        y = 0.5 * (y[:n_pairs] + y[n_pairs:])
        x = 0.5 * (x[:n_pairs] + x[n_pairs:])

    return TerminalSample(
        y=y,
        x=x,
        x_mean=discount_factor * s0 * math.exp(mu * t),
        stratum=stratum,
        n_strata=n_strata_used,
        n_normal_draws=n_draws,
        n_paths=n_paths,
    )


def validate_sampler(
    *, methods: tuple[str, ...], n_paths: int, n_steps: int, n_strata: int
) -> None:
    """Reject sampler settings that the requested methods cannot honour.

    Raises
    ------
    NotSupportedError
        Stratified sampling with `n_steps > 1`; see the module docstring.
    InvalidInputError
        An odd `n_paths` under `antithetic` (there is no half pair), a
        non-positive or oversized `n_strata`, a path count that is not a
        multiple of `n_strata` (proportional allocation would not be
        proportional), or fewer than two paths per stratum (the strata-weighted
        standard error needs a within-stratum `ddof=1` variance).
    """
    if ANTITHETIC in methods and n_paths % 2 != 0:
        raise InvalidInputError(
            "n_paths must be even for variance_reduction 'antithetic': the "
            "estimator's unit is a (Z, -Z) pair"
        )
    if STRATIFIED not in methods:
        return
    if n_steps > 1:
        raise NotSupportedError(
            "variance_reduction 'stratified' is implemented for n_steps=1 only: "
            "it stratifies the single normal that drives the terminal price. "
            "With n_steps>1 the terminal price is driven by a sum of normals "
            "and the standard construction stratifies a projection of the "
            "Brownian path and fills the rest in with a Brownian bridge, which "
            "is a different sampler; that is a later slice."
        )
    if n_strata < 1:
        raise InvalidInputError("n_strata must be >= 1")
    if n_strata > n_paths:
        raise InvalidInputError("n_strata must be <= n_paths")
    if n_paths % n_strata != 0:
        raise InvalidInputError(
            "n_paths must be a multiple of n_strata for proportional allocation"
        )
    if n_paths // n_strata < 2:
        raise InvalidInputError(
            "n_paths // n_strata must be >= 2 for a within-stratum stderr with ddof=1"
        )


def _stratified_value_and_variance(
    y: np.ndarray, n_strata: int, *, ddof: int
) -> tuple[float, float]:
    """Strata-weighted mean and variance of the mean, equal-probability strata.

    The sampler lays the strata out in contiguous blocks of equal size (see
    `_stratified_normals`), so the reshape below *is* the grouping; the stratum
    labels are carried on the sample for the tests and the metadata rather than
    for this computation.
    """
    per_stratum = y.size // n_strata
    block = y.reshape(n_strata, per_stratum)
    means = block.mean(axis=1)
    variances = block.var(axis=1, ddof=ddof)
    value = float(means.mean())
    variance = float(variances.sum() / (n_strata * n_strata * per_stratum))
    return value, variance


def estimate_from_sample(
    sample: TerminalSample, *, methods: tuple[str, ...]
) -> VarianceReductionEstimate:
    """Turn a sample into a price, a standard error, and the estimator's meta.

    The standard error is the one that belongs to the estimator actually used:
    the `ddof=1` one over antithetic *pairs*, the regression-residual one with
    `ddof=2` when a coefficient was fitted, and the strata-weighted one under
    stratification.
    """
    y = sample.y
    meta: dict[str, Any] = {
        "variance_reduction": methods if methods else NONE,
        "n_normal_draws": sample.n_normal_draws,
        "n_estimator_units": int(y.size),
    }
    if ANTITHETIC in methods:
        meta["n_antithetic_pairs"] = int(y.size)
    if STRATIFIED in methods:
        meta["n_strata"] = sample.n_strata
        meta["paths_per_stratum"] = int(y.size // sample.n_strata)

    ddof = 1
    if CONTROL_VARIATE in methods:
        beta = control_variate_coefficient(y, sample.x)
        y = y - beta * (sample.x - sample.x_mean)
        # One degree of freedom for the mean, one for the fitted slope.
        ddof = 2
        y_raw, x_raw = sample.y, sample.x
        sd_y = float(np.std(y_raw, ddof=1))
        sd_x = float(np.std(x_raw, ddof=1))
        rho = (
            float(np.cov(y_raw, x_raw, ddof=1)[0, 1] / (sd_y * sd_x))
            if sd_y > 0.0 and sd_x > 0.0
            else 0.0
        )
        meta["control_variate"] = "discounted_terminal_spot"
        meta["control_beta"] = beta
        meta["control_correlation"] = rho
        meta["control_mean"] = sample.x_mean
        meta["control_sample_mean"] = float(np.mean(x_raw))
        meta["control_variance_factor_predicted"] = (
            float("inf") if abs(rho) >= 1.0 else 1.0 / (1.0 - rho * rho)
        )

    if sample.stratum is None:
        value = float(np.mean(y))
        variance = float(np.var(y, ddof=ddof) / y.size)
    else:
        value, variance = _stratified_value_and_variance(y, sample.n_strata, ddof=1)
    return VarianceReductionEstimate(
        value=value, stderr=math.sqrt(max(variance, 0.0)), meta=meta
    )


def price_with_variance_reduction(
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
) -> VarianceReductionEstimate:
    """Simulate and estimate in one call; see the two functions it composes."""
    sample = simulate_terminal_sample(
        payoff=payoff,
        s0=s0,
        mu=mu,
        sigma=sigma,
        t=t,
        discount_factor=discount_factor,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        methods=methods,
        n_strata=n_strata,
    )
    return estimate_from_sample(sample, methods=methods)
