from __future__ import annotations

import math
from collections.abc import Callable
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


def simulate_gbm_exact(
    *,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int = 123,
) -> np.ndarray:
    """Simulate GBM paths with exact lognormal transition stepping."""
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
