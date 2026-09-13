from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.options import EuropeanOption
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult
from ..registry import MethodSpec


@dataclass(frozen=True)
class PDEConfig:
    """Finite-difference grid configuration for the Black-Scholes PDE.

    Parameters
    ----------
    n_s
        Number of spatial intervals (spot grid has `n_s + 1` nodes).
    n_t
        Number of time steps.
    theta
        Theta-scheme parameter in `[0, 1]`:
        - `0.0`: explicit
        - `0.5`: Crank-Nicolson
        - `1.0`: implicit
    s_max
        Optional maximum spot boundary. If `None`, `s_max_multiplier * spot` is used.
    s_max_multiplier
        Multiplier used when `s_max` is not explicitly provided.
    strike_alignment
        Placement of the strike relative to the uniform spot grid.

        - `"none"` (default): the grid is `linspace(0, s_max, n_s + 1)` and the
          strike falls wherever it happens to fall.
        - `"midpoint"`: the spacing is nudged so the strike sits exactly halfway
          between two adjacent nodes. See `price_european` for the derivation and
          for why this matters.
    """

    n_s: int = 200
    n_t: int = 200
    theta: float = 0.5  # 1.0 = fully implicit, 0.5 = Crank–Nicolson
    s_max: float | None = None
    s_max_multiplier: float = 4.0
    strike_alignment: Literal["none", "midpoint"] = "none"


PDE_METHOD_SPEC = MethodSpec(method="pde", cfg_type=PDEConfig)
"""Keyword contract for `method="pde"`: a required `PDEConfig`, nothing else."""


def _solve_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal linear system via Thomas algorithm."""
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

    Parameters
    ----------
    option
        European option (`call` or `put`).
    model
        Black-Scholes model with constant volatility.
    market
        Market object providing spot and discount/dividend curves.
    cfg
        PDE grid and theta-scheme settings.

    Returns
    -------
    PriceResult
        PDE price estimate with metadata for method and grid settings.

    Notes
    -----
    - Rates/dividend yields are read from `Market` at each time level.
    - This implementation is single-asset and 1D in spot.
    - Interpolation at spot uses cubic spline for smoother Greek estimates.

    Strike alignment (`cfg.strike_alignment`)
    -----------------------------------------
    The terminal condition for a vanilla option has a kink at `S = K`: the
    payoff is continuous but its first derivative jumps and its second
    derivative is a delta at the strike. A finite-difference operator is only
    second-order accurate where the function it differentiates is smooth
    enough for the Taylor expansion behind the stencil to be valid. Near the
    kink that expansion breaks down, so the local truncation error there is
    much larger than the formal `O(ds**2)` and it pollutes the global error
    constant. The effect is worst when the kink lands *exactly on a node*,
    because the node then carries the full one-sided discrepancy between the
    payoff and any smooth function through the neighbouring values.

    Placing the strike midway between two nodes instead makes the two
    neighbouring payoff values symmetric about the kink, so the leading part of
    the payoff-induced error cancels between them, and the observed order
    returns to the formal second order of the scheme. This is the cheapest of
    the standard remedies for non-smooth payoffs; see Pooley, Forsyth and
    Vetzal (2003), "Convergence remedies for non-smooth payoffs in option
    pricing", Journal of Computational Finance 6(4), and the grid-construction
    discussion in Tavella and Randall (2000), "Pricing Financial Instruments:
    The Finite Difference Method".

    The construction here preserves `n_s` and moves `s_max` slightly instead.
    Starting from the nominal spacing `ds0 = s_max / n_s`, the strike sits at
    `K / ds0` grid spacings from the origin. The nearest *half-integer*
    position is `j + 1/2` with `j = round(K / ds0 - 1/2)` (clamped at 0), and
    demanding `K = (j + 1/2) * ds` fixes the spacing as `ds = K / (j + 1/2)`.
    The grid is then `linspace(0, ds * n_s, n_s + 1)`, so `s_max` shifts by at
    most about half a spacing. The realised `s_max`, `ds` and alignment mode
    are reported in `PriceResult.meta`.

    What this does not fix: Crank-Nicolson still produces oscillatory Greeks
    near a non-smooth terminal condition because the scheme damps
    high-frequency modes only marginally. That needs Rannacher time stepping
    or a non-uniform grid, neither of which is implemented here.
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
    if cfg.strike_alignment not in {"none", "midpoint"}:
        raise InvalidInputError("strike_alignment must be 'none' or 'midpoint'")

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

    if cfg.strike_alignment == "midpoint":
        # Nearest half-integer node position for the strike; see the docstring.
        j = max(round(k / ds - 0.5), 0)
        ds = k / (j + 0.5)
        s_max = ds * n_s

    if not (0.0 < k < s_max):
        raise InvalidInputError(
            f"strike {k} must lie strictly inside the spot grid (0, {s_max})"
        )

    dt = t / n_t

    if cfg.strike_alignment == "midpoint":
        # Build from ds directly so the half-integer node position is exact.
        s_grid = ds * np.arange(n_s + 1, dtype=float)
    else:
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

    # Use CubicSpline for smoother interpolation (essential for Gamma via FD)
    # np.interp is piecewise linear -> 2nd derivative is 0 or undefined.
    from scipy.interpolate import CubicSpline

    cs = CubicSpline(s_grid, v)
    price = float(cs(s0))

    meta = {
        "method": "pde",
        "model": "BlackScholes",
        "theta": theta,
        "n_s": n_s,
        "n_t": n_t,
        "s_max": s_max,
        "ds": ds,
        "strike_alignment": cfg.strike_alignment,
    }
    return PriceResult(value=price, meta=meta)


def greeks_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: PDEConfig,
) -> GreeksResult:
    """Compute Delta and Gamma from PDE prices via central finite differences.

    Parameters
    ----------
    option
        European option (`call` or `put`).
    model
        Black-Scholes model.
    market
        Market object.
    cfg
        PDE grid and theta-scheme settings.

    Returns
    -------
    GreeksResult
        Delta and Gamma estimates. Vega/Theta/Rho are currently returned as NaN.
    """
    from dataclasses import replace

    # Finite difference bump size
    # Uses a larger bump (1%) to smooth out grid interpolation artifacts for Gamma
    s0 = market.spot
    h = max(0.01 * s0, 1e-4)

    # Prepare markets shifted up and down
    market_up = replace(market, spot=s0 + h)
    market_down = replace(market, spot=s0 - h)

    # Compute 3 prices: V(S+h), V(S-h), V(S)
    # Note: V(S) is strictly needed for Gamma. For Delta method-neutral,
    # central diff is (V(S+h) - V(S-h)) / 2h.
    # PDE grid alignment might introduce noise if h < ds, but for now we trust interp.
    res_up = price_european(option, model, market_up, cfg=cfg)
    res_down = price_european(option, model, market_down, cfg=cfg)
    res_mid = price_european(option, model, market, cfg=cfg)  # Needed for Gamma

    v_up = res_up.value
    v_down = res_down.value
    v_mid = res_mid.value

    delta = (v_up - v_down) / (2 * h)
    gamma = (v_up - 2 * v_mid + v_down) / (h * h)

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=math.nan,
        theta=math.nan,
        rho=math.nan,
        meta={
            "method": "pde",
            "bump_size": h,
            "pde_meta": res_mid.meta,
        },
    )
