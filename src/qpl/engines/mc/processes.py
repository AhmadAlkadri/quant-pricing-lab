from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.payoffs import call_payoff, put_payoff
from ..base import PriceResult


def _validate_inputs(
    *,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
) -> None:
    if s0 <= 0.0:
        raise InvalidInputError("s0 must be > 0")
    if not math.isfinite(mu):
        raise InvalidInputError("mu must be finite")
    if sigma < 0.0:
        raise InvalidInputError("sigma must be >= 0")
    if t < 0.0:
        raise InvalidInputError("t must be >= 0")
    if n_steps < 1:
        raise InvalidInputError("n_steps must be >= 1")
    if n_paths < 1:
        raise InvalidInputError("n_paths must be >= 1")


def _initialize_paths(*, s0: float, t: float, n_steps: int, n_paths: int) -> np.ndarray:
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = s0
    if t == 0.0:
        paths[:, 1:] = s0
    return paths


def _clean_time_grid(times: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate a variable time grid: 1D, finite, strictly increasing, all > 0."""
    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise InvalidInputError("times must be a non-empty 1D sequence")
    if not np.all(np.isfinite(grid)):
        raise InvalidInputError("times must be finite")
    if np.any(grid <= 0.0):
        raise InvalidInputError("times must all be > 0")
    if np.any(np.diff(grid) <= 0.0):
        raise InvalidInputError("times must be strictly increasing")
    return grid


def gbm_paths_from_normals(
    z: np.ndarray,
    *,
    s0: float,
    mu: float,
    sigma: float,
    times: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Exact GBM path values at `times`, from caller-supplied normals.

    `z` is `(n_paths, len(times))` standard normals; the return is
    `(n_paths, len(times))` spot values, **without** an `s0` column. The
    recursion is the same exact lognormal transition
    `S_{t_{i+1}} = S_{t_i} exp((mu - sigma^2/2) dt_i + sigma sqrt(dt_i) Z)`
    used by :func:`simulate_gbm_exact`, with `dt_i = t_{i+1} - t_i` varying
    along the grid and `t_0 = 0`.

    Exactness matters here more than it does for a European. The Asian payoff
    reads the path at its fixing dates, so a scheme with a discretisation bias
    would bias every fixing; with exact lognormal stepping the *only* error in
    an Asian Monte Carlo price is statistical, and there is no time-step
    refinement study to run. That is why this function steps the grid directly
    rather than sub-dividing it.

    It takes the normals instead of drawing them because the variance-reduction
    layer needs the same draws twice (antithetic reflection) and needs the
    control variate computed on the *same* paths as the payoff. Same reason as
    `qpl.engines.mc.variance_reduction.terminal_spots_from_normals`, one
    dimension up.
    """
    z_arr = np.asarray(z, dtype=float)
    if z_arr.ndim != 2:
        raise InvalidInputError("z must be a 2D (n_paths, n_times) array")
    grid = _clean_time_grid(times)
    if z_arr.shape[1] != grid.size:
        raise InvalidInputError("z must have one column per time in the grid")
    if s0 <= 0.0:
        raise InvalidInputError("s0 must be > 0")
    if sigma < 0.0:
        raise InvalidInputError("sigma must be >= 0")

    dt = np.diff(grid, prepend=0.0)
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    increments = drift[None, :] + vol[None, :] * z_arr
    return s0 * np.exp(np.cumsum(increments, axis=1))


def simulate_gbm_exact(
    *,
    s0: float,
    mu: float,
    sigma: float,
    t: float | None = None,
    n_steps: int | None = None,
    n_paths: int,
    seed: int = 123,
    times: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Simulate GBM paths with exact lognormal transition stepping.

    Two ways to say what the time grid is, and exactly one of them must be
    given:

    - `t` and `n_steps`: a uniform grid of `n_steps` steps of `t / n_steps`.
      This is the original signature and this code path is **untouched**, so
      every existing caller gets bit-for-bit the same paths it always did. The
      arithmetic is deliberately not routed through the variable-grid branch:
      `t_{i+1} - t_i` for a uniform grid is not bit-for-bit `t / n_steps`, and
      a Monte Carlo price that moved in its last digit would invalidate the
      pinned `(value, stderr)` pairs in `tests/test_mc_pricing.py`. The two
      branches also consume the generator differently -- the uniform one draws
      `n_paths` normals per step, the variable one draws the whole
      `(n_paths, n_grid)` block at once -- so at the *same* seed and the same
      grid they produce different (equally valid) samples. Neither is a
      reference for the other path by path; only their laws agree.
    - `times`: an explicit strictly increasing grid of positive times, for a
      payoff whose monitoring dates are given rather than chosen (Slice 8's
      Asian options). The drift and volatility factors then vary step by step.

    Returns
    -------
    numpy.ndarray
        `(n_paths, n_grid + 1)`, column 0 being `s0`.
    """
    if (times is None) == (t is None):
        raise InvalidInputError(
            "simulate_gbm_exact needs exactly one of (t, n_steps) or times"
        )

    if times is not None:
        if n_steps is not None:
            raise InvalidInputError("n_steps is meaningless when times is given")
        grid = _clean_time_grid(times)
        _validate_inputs(
            s0=s0,
            mu=mu,
            sigma=sigma,
            t=float(grid[-1]),
            n_steps=grid.size,
            n_paths=n_paths,
        )
        rng = np.random.default_rng(seed)
        z = rng.normal(size=(n_paths, grid.size))
        paths = np.empty((n_paths, grid.size + 1), dtype=float)
        paths[:, 0] = s0
        paths[:, 1:] = gbm_paths_from_normals(
            z, s0=s0, mu=mu, sigma=sigma, times=grid
        )
        return paths

    if n_steps is None:
        raise InvalidInputError("n_steps is required when times is not given")
    _validate_inputs(s0=s0, mu=mu, sigma=sigma, t=t, n_steps=n_steps, n_paths=n_paths)

    paths = _initialize_paths(s0=s0, t=t, n_steps=n_steps, n_paths=n_paths)
    if t == 0.0:
        return paths

    dt = t / n_steps
    drift_step = (mu - 0.5 * sigma * sigma) * dt
    vol_step = sigma * math.sqrt(dt)
    rng = np.random.default_rng(seed)

    for i in range(n_steps):
        z = rng.normal(size=n_paths)
        paths[:, i + 1] = paths[:, i] * np.exp(drift_step + vol_step * z)

    return paths


def simulate_gbm_euler(
    *,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int = 123,
) -> np.ndarray:
    """Simulate GBM paths with Euler-Maruyama discretization."""
    _validate_inputs(s0=s0, mu=mu, sigma=sigma, t=t, n_steps=n_steps, n_paths=n_paths)

    paths = _initialize_paths(s0=s0, t=t, n_steps=n_steps, n_paths=n_paths)
    if t == 0.0:
        return paths

    dt = t / n_steps
    sqrt_dt = math.sqrt(dt)
    rng = np.random.default_rng(seed)

    for i in range(n_steps):
        z = rng.normal(size=n_paths)
        s_prev = paths[:, i]
        paths[:, i + 1] = s_prev + mu * s_prev * dt + sigma * s_prev * sqrt_dt * z

    return paths


def price_european_from_terminal(
    s_t: np.ndarray,
    *,
    strike: float,
    discount_factor: float,
    kind: Literal["call", "put"] = "call",
    payoff: Callable[[np.ndarray], np.ndarray] | None = None,
) -> PriceResult:
    """Estimate discounted European option value/stderr from terminal samples.

    `payoff` maps the terminal sample vector to the payoff vector and defaults
    to the vanilla call or put selected by `kind`. It exists so that
    `qpl.engines.mc.digital` reuses this estimator rather than writing a second
    copy of "discount, average, divide the sample standard deviation by
    `sqrt(N)`": the estimator is the same for any payoff that has a finite
    variance, and a cash-or-nothing indicator plainly does -- it is a Bernoulli
    variable, so its exact standard error is `cash * df * sqrt(p (1 - p) / N)`.
    `kind` is still validated when `payoff` is supplied, since the caller is
    describing the same contract either way.
    """
    s_t_arr = np.asarray(s_t, dtype=float)
    if s_t_arr.ndim != 1:
        raise InvalidInputError("s_t must be a 1D array of terminal prices")
    if s_t_arr.size < 2:
        raise InvalidInputError("Need at least 2 terminal samples for stderr with ddof=1")
    if strike <= 0.0:
        raise InvalidInputError("strike must be > 0")
    if discount_factor <= 0.0:
        raise InvalidInputError("discount_factor must be > 0")
    if kind not in {"call", "put"}:
        raise InvalidInputError("kind must be 'call' or 'put'")

    if payoff is None:
        values = call_payoff(s_t_arr, strike) if kind == "call" else put_payoff(s_t_arr, strike)
    else:
        values = payoff(s_t_arr)
    pv = discount_factor * values

    value = float(np.mean(pv))
    stderr = float(np.std(pv, ddof=1) / math.sqrt(s_t_arr.size))
    return PriceResult(value=value, stderr=stderr)
