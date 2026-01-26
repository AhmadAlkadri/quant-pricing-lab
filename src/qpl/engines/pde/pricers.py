from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.options import EuropeanOption
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import PriceResult


@dataclass(frozen=True)
class PDEConfig:
    """Finite-difference grid configuration for Black–Scholes PDE."""

    n_s: int = 200
    n_t: int = 200
    theta: float = 0.5  # 1.0 = fully implicit, 0.5 = Crank–Nicolson
    s_max: float | None = None
    s_max_multiplier: float = 4.0


def _solve_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    n = len(diag)
    c_prime = np.empty(n, dtype=float)
    d_prime = np.empty(n, dtype=float)

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom

    x = np.empty(n, dtype=float)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def price_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: PDEConfig,
) -> PriceResult:
    """Price a European option by solving the Black–Scholes PDE via a theta scheme.

    Assumptions:
    - Constant volatility (sigma from BlackScholesModel)
    - Rates/dividend yields are implied from Market.df_r/df_q at each time step
    - European call/put only, single-asset 1D spatial grid
    """
    if cfg.n_s < 3:
        raise InvalidInputError("n_s must be >= 3")
    if cfg.n_t < 1:
        raise InvalidInputError("n_t must be >= 1")
    if not (0.0 <= cfg.theta <= 1.0):
        raise InvalidInputError("theta must be in [0, 1]")
    if cfg.s_max is not None and cfg.s_max <= 0:
        raise InvalidInputError("s_max must be > 0")
    if cfg.s_max_multiplier <= 0:
        raise InvalidInputError("s_max_multiplier must be > 0")

    s0 = market.spot
    k = option.strike
    t = option.expiry
    sigma = model.sigma
    theta = cfg.theta

    if t == 0.0:
        if option.kind == "call":
            value = max(s0 - k, 0.0)
        else:
            value = max(k - s0, 0.0)
        return PriceResult(value=float(value), meta={"method": "pde", "model": "BlackScholes"})

    if sigma == 0.0:
        r = market.rate(t)
        q = market.dividend_yield(t)
        forward = s0 * math.exp((r - q) * t)
        disc = market.df_r(t)
        if option.kind == "call":
            value = disc * max(forward - k, 0.0)
        else:
            value = disc * max(k - forward, 0.0)
        return PriceResult(value=float(value), meta={"method": "pde", "model": "BlackScholes"})

    s_max = cfg.s_max if cfg.s_max is not None else cfg.s_max_multiplier * s0
    n_s = cfg.n_s
    n_t = cfg.n_t
    ds = s_max / n_s
    dt = t / n_t

    s_grid = np.linspace(0.0, s_max, n_s + 1)
    if option.kind == "call":
        v = np.maximum(s_grid - k, 0.0)
    else:
        v = np.maximum(k - s_grid, 0.0)

    s_inner = s_grid[1:-1]

    for n in range(n_t):
        tau_n = n * dt
        tau_np1 = (n + 1) * dt

        df_r_n = market.df_r(tau_n)
        df_q_n = market.df_q(tau_n)
        df_r_np1 = market.df_r(tau_np1)
        df_q_np1 = market.df_q(tau_np1)

        if option.kind == "call":
            v0_n = 0.0
            v0_np1 = 0.0
            vmax_n = s_max * df_q_n - k * df_r_n
            vmax_np1 = s_max * df_q_np1 - k * df_r_np1
        else:
            v0_n = k * df_r_n
            v0_np1 = k * df_r_np1
            vmax_n = 0.0
            vmax_np1 = 0.0

        v[0] = v0_n
        v[-1] = vmax_n

        r = market.rate(tau_np1)
        q = market.dividend_yield(tau_np1)

        a = 0.5 * sigma * sigma * (s_inner**2) / (ds * ds) - (r - q) * s_inner / (2.0 * ds)
        b = -(sigma * sigma) * (s_inner**2) / (ds * ds) - r
        c = 0.5 * sigma * sigma * (s_inner**2) / (ds * ds) + (r - q) * s_inner / (2.0 * ds)

        lower = -theta * dt * a
        diag = 1.0 - theta * dt * b
        upper = -theta * dt * c

        rhs = (1.0 + (1.0 - theta) * dt * b) * v[1:-1] + (1.0 - theta) * dt * (
            a * v[:-2] + c * v[2:]
        )

        rhs[0] -= lower[0] * v0_np1
        rhs[-1] -= upper[-1] * vmax_np1
        lower[0] = 0.0
        upper[-1] = 0.0

        v[1:-1] = _solve_tridiagonal(lower, diag, upper, rhs)
        v[0] = v0_np1
        v[-1] = vmax_np1

    price = float(np.interp(s0, s_grid, v))
    meta = {
        "method": "pde",
        "model": "BlackScholes",
        "theta": theta,
        "n_s": n_s,
        "n_t": n_t,
        "s_max": s_max,
    }
    return PriceResult(value=price, meta=meta)
