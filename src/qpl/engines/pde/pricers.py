from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.options import EuropeanOption
from ...market.curves import FlatDividendCurve, FlatRateCurve
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
    time_stepping
        How the march in `tau = T - t` is started.

        - `"theta"` (default): every step uses `theta`. Bit-for-bit identical
          to the pre-Slice-4 engine.
        - `"rannacher"`: the first **two** steps are replaced by **four fully
          implicit steps of `dt / 2`**, and the remaining `n_t - 2` steps use
          `theta`. Requires `n_t >= 2`. Total time is exact: the four half
          steps cover `4 * (dt/2) = 2 dt` and the march resumes at `tau = 2 dt`.
          See `RANNACHER_STARTUP_STEPS` and `price_european` for why.
    greeks_method
        Where `greeks_european` gets delta, gamma and theta from.

        - `"grid"` (default): read off the finished finite-difference grid with
          second-order central stencils, and theta from the PDE identity. Vega
          and rho are still bump-and-revalue -- nothing on a one-factor spot
          grid knows about sigma or r.
        - `"bump"`: the pre-Slice-4 path -- three full solves at `S`, `S(1+h)`
          and `S(1-h)` with `h = 1%`, differenced through the cubic spline, and
          NaN for vega, theta and rho. Kept, named, and measured against
          `"grid"`; see `greeks_european` for where it is worse and why.
    """

    n_s: int = 200
    n_t: int = 200
    theta: float = 0.5  # 1.0 = fully implicit, 0.5 = Crank–Nicolson
    s_max: float | None = None
    s_max_multiplier: float = 4.0
    strike_alignment: Literal["none", "midpoint"] = "none"
    time_stepping: Literal["theta", "rannacher"] = "theta"
    greeks_method: Literal["grid", "bump"] = "grid"


PDE_METHOD_SPEC = MethodSpec(method="pde", cfg_type=PDEConfig)
"""Keyword contract for `method="pde"`: a required `PDEConfig`, nothing else."""


RANNACHER_STARTUP_STEPS = 4
"""Number of fully implicit start-up steps used by `time_stepping="rannacher"`.

Each is half the nominal step, so the four of them replace the **first two**
nominal steps and cover `4 * (dt/2) = 2 dt` of `tau`. This is the standard
choice: Rannacher (1984), "Finite element solution of diffusion problems with
irregular data", Numerische Mathematik 43, 309-327, and the analysis in Giles
and Carter (2006), "Convergence analysis of Crank-Nicolson and Rannacher
time-marching", Journal of Computational Finance 9(4), 89-112, which is where
the "two half steps, twice" phrasing comes from -- the same four half steps,
counted as two pairs.

Why it is needed. Write the scheme in terms of the eigenvalues `-lambda` of the
discrete space operator. One theta step multiplies the mode by the
amplification factor

    R(z) = (1 + (1 - theta) z) / (1 - theta z),      z = -lambda * dt.

At `theta = 1/2` this is the Cayley transform `(1 + z/2) / (1 - z/2)`, whose
modulus is below one for every `z < 0` -- Crank-Nicolson is unconditionally
stable -- but which tends to **-1** as `z -> -infinity`. The stiffest modes are
therefore not damped at all: they are merely flipped in sign at every step.
A vanilla payoff's kink at the strike is exactly high-frequency content in that
sense, so it survives the whole march as a sign-alternating ripple. The price is
an average over the grid and barely notices; the second difference that gives
gamma differences neighbouring ripples and is dominated by it.

Fully implicit stepping has `R(z) = 1 / (1 - z) -> 0`, so a few such steps kill
the stiff modes outright. Four half steps damp a mode of size `|z|` by
`(1 + |z| / 2)**-4`, which is `O(dt**2)` for the modes that matter, i.e. enough
to remove the ripple without spoiling the `O(dt**2)` accuracy of the
Crank-Nicolson steps that follow -- that trade-off is the content of Giles and
Carter (2006). Two half steps would damp by `(1 + |z|/2)**-2` only.
"""


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


def _validate(cfg: PDEConfig) -> None:
    """Reject unusable configurations, in the order the pre-registry code did."""
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
    if cfg.time_stepping not in {"theta", "rannacher"}:
        raise InvalidInputError("time_stepping must be 'theta' or 'rannacher'")
    if cfg.time_stepping == "rannacher" and cfg.n_t < 2:
        raise InvalidInputError(
            "time_stepping='rannacher' requires n_t >= 2: the four implicit "
            "start-up half steps replace the first two nominal steps"
        )
    if cfg.greeks_method not in {"grid", "bump"}:
        raise InvalidInputError("greeks_method must be 'grid' or 'bump'")


def _time_levels(t: float, cfg: PDEConfig) -> list[tuple[float, float, float, float]]:
    """The march in `tau`, as `(tau_start, tau_end, dt_step, theta_step)` per step.

    For `time_stepping="theta"` this is exactly `n_t` steps of `dt = t / n_t`,
    each with `cfg.theta`, and the endpoint arithmetic (`n * dt`) is the same
    expression the pre-Slice-4 engine evaluated inline -- which is what makes
    that path bit-for-bit unchanged.

    For `"rannacher"` the first two of those steps are replaced by
    `RANNACHER_STARTUP_STEPS = 4` fully implicit steps of `dt / 2`. Halving is
    exact in binary floating point, so `4 * (0.5 * dt)` is `2 * dt` to the last
    bit and the Crank-Nicolson march resumes at exactly the level it would
    otherwise have reached; the total time covered is `n_t * dt` either way.
    """
    dt = t / cfg.n_t
    if cfg.time_stepping == "theta":
        return [(n * dt, (n + 1) * dt, dt, cfg.theta) for n in range(cfg.n_t)]

    half = 0.5 * dt
    steps = [(i * half, (i + 1) * half, half, 1.0) for i in range(RANNACHER_STARTUP_STEPS)]
    steps += [(n * dt, (n + 1) * dt, dt, cfg.theta) for n in range(2, cfg.n_t)]
    return steps


@dataclass(frozen=True)
class _GridSolution:
    """A finished finite-difference solve, and everything read off it.

    `v` holds the value function at `tau = T`, i.e. at calendar time 0, which
    is where every Greek in this module is evaluated. `v_prev` holds the level
    one step earlier in `tau` -- calendar time `dt_last` -- and exists only so
    that `greeks_european` can compute the one-sided time difference it uses as
    a cross-check on the theta it reports.
    """

    s_grid: np.ndarray
    v: np.ndarray
    v_prev: np.ndarray
    dt_last: float
    ds: float
    price: float
    meta: dict[str, Any]


def _solve_grid(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    cfg: PDEConfig,
) -> _GridSolution:
    """March the grid from the payoff at `tau = 0` to `tau = T`.

    Assumes `_validate(cfg)` has run and that neither degenerate case
    (`T = 0`, `sigma = 0`) applies; `price_european` handles both before
    calling here. The arithmetic is exactly what `price_european` used to do
    inline, so prices are unchanged.
    """
    s0 = market.spot
    k = option.strike
    t = option.expiry
    sigma = model.sigma

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

    steps = _time_levels(t, cfg)

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

    n_steps = len(steps)
    v_prev = v.copy()
    dt_last = steps[-1][2]

    for index, (tau_n, tau_np1, dt, theta_step) in enumerate(steps):
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

        lower = -theta_step * dt * a
        diag = 1.0 - theta_step * dt * b
        upper = -theta_step * dt * c

        rhs = (1.0 + (1.0 - theta_step) * dt * b) * v[1:-1] + (1.0 - theta_step) * dt * (
            a * v[:-2] + c * v[2:]
        )

        rhs[0] -= lower[0] * v0_np1
        rhs[-1] -= upper[-1] * vmax_np1
        lower[0] = 0.0
        upper[-1] = 0.0

        if index == n_steps - 1:
            # Kept only at the last step, for the backward-difference theta
            # cross-check in `greeks_european`. Copying every step would be
            # pure waste.
            v_prev = v.copy()
            dt_last = dt

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
        "theta": cfg.theta,
        "n_s": n_s,
        "n_t": n_t,
        "s_max": s_max,
        "ds": ds,
        "strike_alignment": cfg.strike_alignment,
        "time_stepping": cfg.time_stepping,
        "implicit_startup_steps": (
            RANNACHER_STARTUP_STEPS if cfg.time_stepping == "rannacher" else 0
        ),
        "n_steps_taken": len(steps),
    }
    return _GridSolution(
        s_grid=s_grid,
        v=v,
        v_prev=v_prev,
        dt_last=dt_last,
        ds=ds,
        price=price,
        meta=meta,
    )


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
    high-frequency modes only marginally. That is what
    `cfg.time_stepping="rannacher"` is for; a non-uniform grid concentrated at
    the strike is the other standard remedy and is not implemented here.

    Time stepping (`cfg.time_stepping`)
    -----------------------------------
    `"theta"` marches `n_t` steps of `dt = T / n_t`, each with `cfg.theta`.

    `"rannacher"` replaces the first two of those by four fully implicit steps
    of `dt / 2` and then continues with `cfg.theta`. Crank-Nicolson's
    amplification factor tends to `-1` for the stiffest modes, so the payoff
    kink's high-frequency content is flipped rather than damped and survives
    the whole march; a few fully implicit steps, whose factor tends to `0`,
    remove it. The price barely notices -- it is an average over the grid --
    but gamma, which differences neighbouring values twice, is dominated by it.
    See `RANNACHER_STARTUP_STEPS` for the mechanism and the citations, and
    `docs/notes/pde_greeks_and_rannacher.md` for the measured effect.
    """
    _validate(cfg)

    s0 = market.spot
    k = option.strike
    t = option.expiry
    sigma = model.sigma

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

    sol = _solve_grid(option, model, market, cfg)
    return PriceResult(value=sol.price, meta=sol.meta)


GRID_VEGA_BUMP = 1e-2
"""Absolute volatility bump for the grid engine's vega (one volatility point).

Nothing on a spot grid knows about `sigma`, so vega has to be bumped. Unlike
the tree's, the PDE bump is well behaved: the grid does *not* move when `sigma`
changes -- `s_max`, `ds` and the strike alignment depend only on the spot and
the strike -- so the two solves share their discretisation error and much of it
cancels in the difference.

Measured on the reference ATM call (S=K=100, r=5%, q=0, sigma=20%, T=1) at
`n_s = n_t = 400`, aligned, Rannacher, against the closed-form vega of
37.524035: residual `+2.22e-03` at `h = 1e-3`, `-7.89e-04` at `h = 1e-2`, and
`-7.79e-02` at `h = 5e-2`. The sign change between the first two says the
grid's own error and the bump's `O(h**2)` bias are of opposite sign and cross
near `h = 1e-2`; by `h = 5e-2` the bias is plainly binding. `h = 1e-2` is
chosen there, and it is also the tree engine's `VEGA_BUMP`, which keeps the two
engines comparable. Do not read the `-7.89e-04` as an accuracy claim for vega:
it is a near-cancellation at one point, and the honest statement is that vega
is no better than the grid's own error, about 2e-03 here.
"""

GRID_RHO_BUMP = 1e-4
"""Absolute rate bump for the grid engine's rho, in rate units.

Same argument as vega, and tighter: `r` enters only the PDE coefficients and
the boundary values, never the grid geometry. On the same reference point the
residual against the closed-form rho of 53.232482 is `-5.6731e-03` at
`h = 1e-5` and `-5.6739e-03` at `h = 1e-4` -- identical to four figures, i.e.
entirely the grid's own error -- and `-1.3367e-02` at `h = 1e-2`, where the
bump's own bias has become visible. `h = 1e-4` sits safely inside the flat
region and matches the tree engine's `RHO_BUMP`.
"""

BUMP_SPOT_FRACTION = 1e-2
"""Relative spot bump used by `greeks_method="bump"`.

This is the pre-Slice-4 value, kept exactly so that the comparison in
`greeks_european` is against what this package actually shipped, not against a
re-tuned straw man.
"""


def _interpolation_bracket(s_grid: np.ndarray, s0: float, n_s: int) -> tuple[int, float]:
    """Return `(i, w)` with `s0 = (1 - w) s_grid[i] + w s_grid[i + 1]`.

    `i` is clamped to `[1, n_s - 2]` so that both `i` and `i + 1` are interior
    nodes, i.e. nodes at which a central stencil exists. On the grids this
    engine builds `s_max >= 4 * spot` by default, so the clamp only ever binds
    for a pathological `s_max`.
    """
    i = int(np.searchsorted(s_grid, s0, side="right")) - 1
    i = min(max(i, 1), n_s - 2)
    w = (s0 - s_grid[i]) / (s_grid[i + 1] - s_grid[i])
    return i, float(w)


def _interp_nodal(nodes: np.ndarray, i: int, w: float) -> float:
    """Linear interpolation of a full-grid quantity between nodes `i` and `i + 1`."""
    return float((1.0 - w) * nodes[i] + w * nodes[i + 1])


def _greeks_from_grid(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    cfg: PDEConfig,
) -> GreeksResult:
    """Delta, gamma and theta read off the grid; vega and rho by bump."""
    t = option.expiry
    sigma = model.sigma
    if t == 0.0:
        raise InvalidInputError("expiry must be > 0 for PDE grid Greeks")
    if sigma == 0.0:
        raise InvalidInputError("sigma must be > 0 for PDE grid Greeks")

    sol = _solve_grid(option, model, market, cfg)
    s0 = market.spot
    v = sol.v
    ds = sol.ds

    # Second-order central stencils at every interior node. Index `j` of these
    # two arrays is grid node `j + 1`.
    delta_nodes = (v[2:] - v[:-2]) / (2.0 * ds)
    gamma_nodes = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / (ds * ds)

    i, w = _interpolation_bracket(sol.s_grid, s0, cfg.n_s)
    spot_is_node = w == 0.0 or w == 1.0

    delta = _interp_nodal(delta_nodes, i - 1, w)
    gamma = _interp_nodal(gamma_nodes, i - 1, w)

    # Theta from the PDE identity, evaluated at the spot:
    #
    #   V_t = -(1/2 sigma^2 S^2 V_SS + (r - q) S V_S - r V).
    #
    # This is the reported theta, and it is the right choice: every term on the
    # right is already second-order accurate (the two stencils above, plus the
    # price), so theta inherits order 2. The obvious alternative -- differencing
    # the last two time levels -- is one-sided and therefore O(dt), first order,
    # however good the scheme is. It is computed below as a cross-check, not as
    # a second opinion of equal standing.
    #
    # `r` and `q` are read at `t`, which is exactly the pair the final time step
    # used (`tau_np1 = n_t * dt = t`), so the identity is evaluated with the
    # same coefficients the solver had just applied.
    r = market.rate(t)
    q = market.dividend_yield(t)
    theta = -(
        0.5 * sigma * sigma * s0 * s0 * gamma + (r - q) * s0 * delta - r * sol.price
    )

    # Cross-check only: (V at calendar time dt_last - V at calendar time 0)/dt_last.
    theta_backward = _interp_nodal((sol.v_prev - sol.v) / sol.dt_last, i, w)

    def _price(mdl: BlackScholesModel, mkt: Market) -> float:
        return _solve_grid(option, mdl, mkt, cfg).price

    sigma_up = BlackScholesModel(sigma=sigma + GRID_VEGA_BUMP)
    sigma_dn = BlackScholesModel(sigma=max(sigma - GRID_VEGA_BUMP, 0.0))
    vega = (_price(sigma_up, market) - _price(sigma_dn, market)) / (
        (sigma + GRID_VEGA_BUMP) - max(sigma - GRID_VEGA_BUMP, 0.0)
    )

    def _rate_market(rate: float) -> Market:
        return Market(
            spot=s0,
            rate_curve=FlatRateCurve(rate, allow_negative=True),
            dividend_curve=FlatDividendCurve(q, allow_negative=True),
        )

    rho = (
        _price(model, _rate_market(r + GRID_RHO_BUMP))
        - _price(model, _rate_market(r - GRID_RHO_BUMP))
    ) / (2.0 * GRID_RHO_BUMP)

    meta: dict[str, Any] = {
        "method": "pde",
        "greeks_method": "grid",
        "delta_source": "central stencil on the grid, linear interpolation in S",
        "gamma_source": "central stencil on the grid, linear interpolation in S",
        "theta_source": "PDE identity at the spot",
        "theta_backward_difference": float(theta_backward),
        "vega_source": "bump-and-revalue on the same grid",
        "rho_source": "bump-and-revalue on the same grid",
        "bumps": {"sigma": GRID_VEGA_BUMP, "r": GRID_RHO_BUMP},
        "spot_is_node": bool(spot_is_node),
        "spot_node_index": i,
        "spot_node_weight": w,
        "pde_meta": sol.meta,
    }
    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
        meta=meta,
    )


def _greeks_by_bump(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    cfg: PDEConfig,
) -> GreeksResult:
    """The pre-Slice-4 path: three full solves at three spots, differenced."""
    s0 = market.spot
    h = max(BUMP_SPOT_FRACTION * s0, 1e-4)

    market_up = replace(market, spot=s0 + h)
    market_down = replace(market, spot=s0 - h)

    res_up = price_european(option, model, market_up, cfg=cfg)
    res_down = price_european(option, model, market_down, cfg=cfg)
    res_mid = price_european(option, model, market, cfg=cfg)

    delta = (res_up.value - res_down.value) / (2 * h)
    gamma = (res_up.value - 2 * res_mid.value + res_down.value) / (h * h)

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=math.nan,
        theta=math.nan,
        rho=math.nan,
        meta={
            "method": "pde",
            "greeks_method": "bump",
            "bump_size": h,
            "pde_meta": res_mid.meta,
        },
    )


def greeks_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: PDEConfig,
) -> GreeksResult:
    """Greeks for a European option, read off the finite-difference grid.

    Parameters
    ----------
    option, model, market
        As for `price_european`.
    cfg
        Grid, time-stepping and `greeks_method` settings.

    Returns
    -------
    GreeksResult
        Delta, gamma, theta, vega and rho, with `meta` naming the source of
        each one.

    Raises
    ------
    InvalidInputError
        With `greeks_method="grid"`, if `T = 0` or `sigma = 0`: there is no
        grid to differentiate, and gamma is a distribution rather than a
        function in both limits. `price_european` still returns the correct
        price in both cases.

    Notes
    -----
    **What comes from where** (`greeks_method="grid"`, the default).

    Delta and gamma are the standard second-order central stencils on the
    finished grid,

        delta_i = (V[i+1] - V[i-1]) / (2 ds),
        gamma_i = (V[i+1] - 2 V[i] + V[i-1]) / ds**2,

    each accurate to `O(ds**2)` where `V` is smooth, which at `tau = T` it is:
    the payoff kink has diffused away.

    The spot is usually **not** a node. With `strike_alignment="midpoint"` the
    strike sits at a half-integer number of spacings by construction, so an
    at-the-money spot sits at a half-integer position too -- exactly halfway
    between two nodes, the worst case for interpolation. The remedy is to
    interpolate the *Greek*, not the price: `delta_i` approximates the first
    derivative at node `i` to `O(ds**2)`, and linear interpolation of a smooth
    function between two nodes a distance `ds` apart adds an error
    proportional to `ds**2` times its own second derivative. Both terms are
    second order, so the interpolated delta is second order; the same argument
    one derivative further up gives second-order gamma. Interpolating the
    *price* to shifted points and differencing there would not work: the
    interpolation error would be divided by `ds**2`.

    Theta is the PDE identity at the spot,

        V_t = -(1/2 sigma**2 S**2 V_SS + (r - q) S V_S - r V),

    because every term on the right is already second order, so theta is too.
    The alternative -- a difference of the last two time levels -- is one-sided
    and therefore `O(dt)`, first order, however accurate the scheme is. It is
    computed anyway and reported as `meta["theta_backward_difference"]`, and
    `tests/test_pde_greeks.py` measures both orders. It is a cross-check on the
    identity, not a second opinion of equal standing.

    Vega and rho are bump-and-revalue, because a one-factor spot grid carries
    no information about `sigma` or `r`. They therefore inherit the *grid's*
    error rather than the stencils': they are no better than the price. They
    are nonetheless much better behaved than the tree's, because bumping
    `sigma` or `r` does not move the grid -- `ds`, `s_max` and the strike
    alignment depend only on the spot and the strike -- so the two solves share
    their discretisation error and most of it cancels in the difference.

    **`greeks_method="bump"`** is the path this engine shipped before Slice 4:
    three full solves at `S`, `S(1 + h)` and `S(1 - h)` with `h = 1%` of spot,
    each read through a cubic spline, differenced centrally. It is kept as a
    named alternative and is measurably worse for three separate reasons:

    1. the central difference carries its own `O(h**2)` bias, and `h` is 1% of
       spot -- a *fixed* bias that does not shrink when the grid is refined. On
       the reference ATM call the bump delta error stalls at `-8.5e-05` from
       `n = 400` onwards (`-8.248e-05`, `-8.509e-05`, `-8.574e-05` at
       `n = 400, 800, 1600`) while the grid delta keeps halving twice per
       doubling (`-1.430e-04`, `-3.594e-05`, `-9.007e-06`). Fitted orders over
       `n` in (50 ... 800): **0.42** for the bump path against **2.00** for the
       grid path.
    2. the three solves are on three *different* grids. `s_max` defaults to
       `s_max_multiplier * spot`, so bumping the spot by 1% moves `s_max`,
       moves `ds`, and moves the strike relative to the nodes -- which is the
       one thing `strike_alignment` exists to control. Their discretisation
       errors therefore do not cancel, and the bump gamma sequence is not a
       power law at all: log-space RMS residual **0.73** against the grid
       path's **0.04**, with the sign of the error changing three times over
       five refinements.
    3. it spends three solves to produce two Greeks, where the grid path gets
       three out of one, and it still returns NaN for vega, theta and rho.

    See `docs/notes/pde_greeks_and_rannacher.md` for the measured comparison.
    """
    _validate(cfg)
    if cfg.greeks_method == "bump":
        return _greeks_by_bump(option, model, market, cfg)
    return _greeks_from_grid(option, model, market, cfg)
