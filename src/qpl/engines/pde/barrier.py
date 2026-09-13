"""Single barriers by finite differences: a boundary, or a projection.

This is the first engine in this package that prices **both** monitoring
conventions, and the reason is structural rather than clever: a grid carries
the whole value function at every time level, so "the contract dies at `H`" is
a boundary condition and "the contract is inspected at `t_j`" is something done
to the grid at a time level. Neither needs a different discretisation.

Continuous monitoring: truncate the domain at the barrier
---------------------------------------------------------
For a knock-out with a down barrier the value function is only defined on
`S > H` -- below it the contract is already dead and worth its rebate -- so the
PDE is solved on `[H, s_max]` with

    V(H, t) = rebate      (paid at the touch time, so undiscounted),
    V(s_max, t) = the vanilla's far-field value.

The barrier is then node 0 of the grid, **exactly**, with no alignment
arithmetic at all. An up barrier is the mirror image on `[0, H]`.

That is the whole point of the slice. A lattice knocks out at the first node
beyond the barrier and a uniform grid applied to `[0, s_max]` does the same, so
both price a barrier displaced by `O(ds)` and, because the price is locally
linear in the barrier level, both converge at order **one**. The observation is
Zvan, Vetzal and Forsyth (2000), "PDE methods for pricing barrier options",
*Journal of Economic Dynamics and Control* 24, 1563-1590; the numbers are
measured in `tests/test_pde_barrier.py` and the misaligned configuration is
kept reachable (`PDEConfig(barrier_alignment="none")`) so the negative finding
can be re-run rather than only described.

Discrete monitoring: project at the monitoring dates
-----------------------------------------------------
A discretely monitored contract is *not* absorbing between observations: a path
may dip past the barrier and come back, and the contract does not notice. The
dead region is therefore part of the domain, the grid is the full `[0, s_max]`,
and the barrier enters only at the monitoring dates, where the value function
is overwritten on the dead side:

    knock-out:  V(S, t_j) <- rebate                 for S on the dead side,
    knock-in:   V(S, t_j) <- vanilla(S, t_j)        for S on the dead side.

Between dates the plain Black-Scholes PDE holds everywhere. **The time grid is
built to contain the monitoring dates**: the levels are the sorted union of the
`n_t` uniform levels `j T / n_t` and the `m` monitoring levels `T - t_j`, with
duplicates merged at a relative tolerance of `1e-12` and the monitoring value
kept when two coincide. So a projection always lands on a time level and never
has to be interpolated onto one; the price of that is a time grid with up to
`n_t + m` steps rather than `n_t`, reported in `meta["n_steps_taken"]`. Rannacher
start-up still replaces the first two intervals by four fully implicit half
steps, whatever those intervals turned out to be.

The barrier is still placed on a node under discrete monitoring
(`barrier_alignment="node"` makes it an anchor of the mesh), because the
projection is applied at nodes and a barrier between two of them is again a
displaced barrier.

Knock-in: its own boundary problem, and the parity that checks it
------------------------------------------------------------------
A knock-in is priced **directly**, not as "vanilla minus knock-out". On the
live side of a continuously monitored down-and-in, the holder owns nothing
until the barrier is touched and owns a vanilla the moment it is, so

    terminal:   V(S, T) = rebate        (paid at expiry if never touched),
    at S = H:   V(H, t) = BS(H, K, T - t),   the vanilla the touch delivers,
    at s_max:   V(s_max, t) = rebate e^{-r (T - t)}.

With a zero rebate the terminal data is identically zero and the entire value
arrives through the barrier boundary, which is a pleasant way to see what a
knock-in *is*.

In-out parity is then a statement about the discrete system rather than about
the model. The knock-out and the knock-in are solved on the **same grid** with
the **same matrix**; only the terminal and boundary data differ, and the
solution is a linear function of that data. So a third leg whose data is the
*sum* of theirs -- terminal `vanilla payoff + rebate`, barrier value
`rebate + BS(H, K, tau)`, far value `vanilla far field + rebate e^{-r tau}` --
satisfies

    knock-out + knock-in = that leg

to the round-off of the banded solve, at any `n`, exactly. That leg is
`"vanilla"` below; with a zero rebate it is the vanilla option solved on the
truncated domain with the exact Black-Scholes value imposed at `H`, which is
itself a good approximation of the vanilla and is checked as one. Measured
parity residual: see `tests/test_pde_barrier.py`.

What this engine does not do
----------------------------
Double barriers, a rebate paid at a *discretely observed* touch time with more
care than "the observation date it was seen at", and any barrier that moves in
time. None of the three is a change to the machinery here; all three are out of
this slice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.options import BarrierOption
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel, bs_price
from ..analytic.barrier import barrier_price
from ..base import GreeksResult, PriceResult
from ..mc.barrier import _degenerate_value as _discrete_degenerate_value
from .grid import SpotGrid, build_spot_grid
from .pricers import (
    RANNACHER_STARTUP_STEPS,
    PDEConfig,
    _dirichlet,
    _greeks_by_bump,
    _greeks_from_grid,
    _GridSolution,
    _operator,
    _payoff,
    _solve_tridiagonal,
    _time_levels,
    _validate,
)

__all__ = [
    "BARRIER_LEGS",
    "KNOCK_IN",
    "KNOCK_OUT",
    "VANILLA_LEG",
    "greeks_barrier",
    "price_barrier",
    "solve_leg",
]

KNOCK_OUT = "knock_out"
KNOCK_IN = "knock_in"
VANILLA_LEG = "vanilla"

BARRIER_LEGS: tuple[str, ...] = (KNOCK_OUT, KNOCK_IN, VANILLA_LEG)
"""The three legs solved on one grid; their data adds, so their prices do."""

_MERGE_TOLERANCE = 1e-12
"""Relative tolerance at which a monitoring level and a uniform level merge."""


@dataclass(frozen=True)
class _Domain:
    """Where the grid lives and how the barrier is imposed on it."""

    lower: float
    upper: float | None
    node_points: tuple[float, ...]
    truncated: bool
    """`True` when the barrier IS the domain boundary (continuous monitoring on
    an aligned grid); `False` when the dead region is carried and masked."""


def _domain(option: BarrierOption, cfg: PDEConfig) -> _Domain:
    """Domain and anchors for this contract under this alignment."""
    barrier = float(option.barrier)
    aligned = cfg.barrier_alignment == "node"

    if option.is_continuous and aligned:
        if option.is_down:
            return _Domain(lower=barrier, upper=None, node_points=(), truncated=True)
        return _Domain(lower=0.0, upper=barrier, node_points=(), truncated=True)

    # The dead region is part of the domain: either because the contract is
    # monitored discretely and paths may come back, or because the caller asked
    # for the misaligned grid on purpose.
    return _Domain(
        lower=0.0,
        upper=None,
        node_points=(barrier,) if aligned else (),
        truncated=False,
    )


def _vanilla_on_grid(
    s: np.ndarray, option: BarrierOption, tau: float, r: float, sigma: float, q: float
) -> np.ndarray:
    """Black-Scholes value of the underlying vanilla at every node.

    `bs_price` refuses `S <= 0`, and `S = 0` is a real node on the full domain,
    so it is filled from the limit: a call is worth nothing there and a put is
    worth `K e^{-r tau}` (the spot can never leave zero).
    """
    out = np.empty_like(s, dtype=float)
    positive = s > 0.0
    out[positive] = np.asarray(
        bs_price(
            S=s[positive],
            K=option.strike,
            T=tau,
            r=r,
            sigma=sigma,
            q=q,
            kind=option.kind,
        ),
        dtype=float,
    )
    if not positive.all():
        out[~positive] = (
            0.0 if option.kind == "call" else option.strike * math.exp(-r * tau)
        )
    return out


def _time_levels_with_dates(
    t: float, cfg: PDEConfig, monitoring_taus: tuple[float, ...]
) -> tuple[list[tuple[float, float, float, float]], set[float]]:
    """The march in `tau`, with every monitoring level present as a time level.

    Returns `(steps, projection_levels)`; `steps` has the same
    `(tau_start, tau_end, dt, theta)` shape `_time_levels` returns, and
    `projection_levels` holds the `tau` values at which the barrier is
    observed, as exact floats that appear in `steps`.

    The uniform levels `j T / n_t` and the monitoring levels `T - t_j` are
    merged; when two fall within `1e-12 T` of each other the *monitoring* one
    survives, so a projection is never applied a hair away from its own date.
    Rannacher start-up then splits the first two intervals of the merged grid,
    whatever their lengths.
    """
    dt = t / cfg.n_t
    tol = _MERGE_TOLERANCE * t
    monitoring = {float(tau) for tau in monitoring_taus}
    merged: list[float] = []
    for value in sorted({n * dt for n in range(cfg.n_t + 1)} | monitoring):
        if merged and value - merged[-1] <= tol:
            if value in monitoring:
                merged[-1] = value
            continue
        merged.append(value)

    levels = set(merged)
    projection_levels = {tau for tau in monitoring if tau in levels}

    steps: list[tuple[float, float, float, float]] = []
    for index in range(len(merged) - 1):
        start, end = merged[index], merged[index + 1]
        if cfg.time_stepping == "rannacher" and index < 2:
            middle = 0.5 * (start + end)
            steps.append((start, middle, middle - start, 1.0))
            steps.append((middle, end, end - middle, 1.0))
        else:
            steps.append((start, end, end - start, cfg.theta))
    return steps, projection_levels


def _dead_mask(grid: SpotGrid, option: BarrierOption) -> np.ndarray:
    """Nodes at or beyond the barrier: where the contract has been touched.

    Weak inequalities, matching `BarrierOption.is_touched`: a node sitting
    exactly on the barrier has touched it. On an aligned grid the barrier node
    itself is therefore dead, which is what makes the effective barrier exactly
    `H` rather than the first node past it.
    """
    return grid.s <= option.barrier if option.is_down else grid.s >= option.barrier


def solve_leg(
    option: BarrierOption,
    model: BlackScholesModel,
    market: Market,
    cfg: PDEConfig,
    *,
    leg: str = KNOCK_OUT,
) -> _GridSolution:
    """Solve one of the three legs on the grid this contract asks for.

    `leg` is `"knock_out"`, `"knock_in"` or `"vanilla"`; see the module
    docstring for what each one's terminal and boundary data is and why the
    three add. Exposed so that `tests/test_pde_barrier.py` can measure in-out
    parity on **one** grid rather than comparing two independent solves, which
    would measure discretisation error instead of the identity.

    Assumes `_validate(cfg)` has run and that neither degenerate limit applies.
    """
    if leg not in BARRIER_LEGS:
        raise InvalidInputError(
            "leg must be one of " + ", ".join(repr(name) for name in BARRIER_LEGS)
        )

    s0 = market.spot
    k = option.strike
    t = option.expiry
    sigma = model.sigma
    rebate = float(option.rebate)

    domain = _domain(option, cfg)
    grid = build_spot_grid(
        strike=k,
        spot=s0,
        cfg=cfg,
        lower=domain.lower,
        upper=domain.upper,
        node_points=domain.node_points,
        concentration_points=(float(k), float(option.barrier)),
    )
    s = grid.s
    n_s = grid.n_s

    # Index of the end node that carries the barrier condition, and of the one
    # that carries the ordinary far-field condition.
    barrier_end = 0 if option.is_down else n_s
    far_end = n_s if option.is_down else 0

    dead = _dead_mask(grid, option)
    # On the truncated domain the dead region is exactly the one boundary node.
    interior_dead = dead.copy()
    interior_dead[barrier_end] = False
    if domain.truncated:
        interior_dead[:] = False

    def dead_values(tau: float, nodes: np.ndarray) -> np.ndarray | float:
        """The leg's value on the dead side at `tau = T - t`."""
        r_tau = market.rate(t)
        if leg == KNOCK_OUT:
            return rebate
        vanilla = _vanilla_on_grid(
            nodes, option, tau, r_tau, sigma, market.dividend_yield(t)
        )
        if leg == KNOCK_IN:
            return vanilla
        return vanilla + rebate

    def far_value(tau: float) -> float:
        df_r = market.df_r(tau)
        df_q = market.df_q(tau)
        v0, vmax = _dirichlet(option.kind, k, grid.s_max, df_r, df_q)
        vanilla_far = vmax if option.is_down else v0
        if leg == KNOCK_OUT:
            return float(vanilla_far)
        if leg == KNOCK_IN:
            return float(rebate * df_r)
        return float(vanilla_far + rebate * df_r)

    def barrier_value(tau: float) -> float:
        return float(np.atleast_1d(dead_values(tau, s[barrier_end : barrier_end + 1]))[0])

    # Cell faces, used by the monitoring projection below. Node `i`'s cell is
    # `[cell_lo[i], cell_hi[i]]`; the two end nodes' cells are completed by
    # reflecting their single face.
    faces = grid.cell_faces
    cell_lo = np.concatenate([[2.0 * s[0] - faces[0]], faces])
    cell_hi = np.concatenate([faces, [2.0 * s[-1] - faces[-1]]])
    if option.is_down:
        alive_fraction = np.clip(
            (cell_hi - option.barrier) / (cell_hi - cell_lo), 0.0, 1.0
        )
    else:
        alive_fraction = np.clip(
            (option.barrier - cell_lo) / (cell_hi - cell_lo), 0.0, 1.0
        )
    straddling = np.flatnonzero((alive_fraction > 0.0) & (alive_fraction < 1.0))
    fully_dead = alive_fraction <= 0.0

    def kill_dead_nodes(values: np.ndarray, tau: float) -> None:
        """Overwrite the dead side outright: the crude, node-counting rule.

        Used for continuous monitoring on an unaligned grid
        (`barrier_alignment="none"`), which is deliberately the naive scheme --
        it kills at the extreme dead *node*, so the barrier it prices is
        displaced by up to one spacing. That is the Zvan-Vetzal-Forsyth
        first-order configuration and it is kept crude on purpose.
        """
        if interior_dead.any():
            values[interior_dead] = dead_values(tau, s[interior_dead])

    def apply_projection(values: np.ndarray, tau: float) -> None:
        """Observe the barrier at one monitoring date, in the cell-average sense.

        The contract's value jumps at `H` the instant it is observed, and a
        jump put on a grid by node sampling costs a full order -- the Slice 6
        digital pathology, now repeated once per monitoring date. So the
        projection is the `L2` one: each node takes the **average** of the
        post-observation function over its own cell. Cells entirely on the dead
        side take the dead value, cells entirely alive are untouched, and the
        one cell the barrier straddles takes

            alive_fraction * (average of V over the live part)
            + (1 - alive_fraction) * (dead value at the dead part's midpoint),

        with the live average read off the **pre-observation** V, which is
        smooth across `H` (the previous jump has diffused away), by linear
        interpolation at the live part's midpoint. That is second order in the
        spacing and, unlike node sampling, it does not care where `H` falls
        between two nodes -- measured in `tests/test_pde_barrier.py`, where the
        node-sampling variant is order 1 and biased low by 0.09 at `n_s = 800`
        against this one's 3e-04.
        """
        before = values.copy()
        if fully_dead.any():
            values[fully_dead] = dead_values(tau, s[fully_dead])
        for index in straddling:
            weight = float(alive_fraction[index])
            lo, hi = float(cell_lo[index]), float(cell_hi[index])
            if option.is_down:
                live_mid, dead_mid = 0.5 * (option.barrier + hi), 0.5 * (lo + option.barrier)
            else:
                live_mid, dead_mid = 0.5 * (lo + option.barrier), 0.5 * (option.barrier + hi)
            live_value = float(np.interp(live_mid, s, before))
            dead_value = float(
                np.atleast_1d(dead_values(tau, np.array([dead_mid])))[0]
            )
            values[index] = weight * live_value + (1.0 - weight) * dead_value

    # Terminal condition at tau = 0.
    if leg == KNOCK_OUT:
        v = _payoff(option.kind, k, s)
    elif leg == KNOCK_IN:
        v = np.full(s.shape, rebate, dtype=float)
    else:
        v = _payoff(option.kind, k, s) + rebate

    if option.is_continuous:
        steps = _time_levels(t, cfg)
        projection_levels: set[float] = set()
        project_every_step = not domain.truncated
    else:
        monitoring_taus = tuple(float(t - date) for date in option.monitoring_times)
        steps, projection_levels = _time_levels_with_dates(t, cfg, monitoring_taus)
        project_every_step = False

    # `barrier_alignment` is one axis with two settings: resolve the barrier
    # exactly, or round it to the nodes. "Exactly" means a domain boundary
    # under continuous monitoring and a cell-weighted projection under
    # discrete monitoring; "rounded" means the crude node rule in both, which
    # is the configuration the two negative findings are measured on.
    observe = apply_projection if cfg.barrier_alignment == "node" else kill_dead_nodes

    n_projections = 0
    if project_every_step:
        kill_dead_nodes(v, 0.0)
    elif 0.0 in projection_levels:
        # A monitoring date at expiry: the barrier is observed before the
        # terminal payoff is ever discounted.
        observe(v, 0.0)
        n_projections += 1
    v[barrier_end] = barrier_value(0.0)

    n_steps = len(steps)
    v_prev = v.copy()
    dt_last = steps[-1][2]

    for index, (tau_n, tau_np1, dt, theta_step) in enumerate(steps):
        v[barrier_end] = barrier_value(tau_n)
        v[far_end] = far_value(tau_n)

        r = market.rate(tau_np1)
        q = market.dividend_yield(tau_np1)
        a, b, c = _operator(grid, sigma, r, q)

        lower = -theta_step * dt * a
        diag = 1.0 - theta_step * dt * b
        upper = -theta_step * dt * c

        rhs = (1.0 + (1.0 - theta_step) * dt * b) * v[1:-1] + (1.0 - theta_step) * dt * (
            a * v[:-2] + c * v[2:]
        )

        v0_np1 = barrier_value(tau_np1) if option.is_down else far_value(tau_np1)
        vmax_np1 = far_value(tau_np1) if option.is_down else barrier_value(tau_np1)
        rhs[0] -= lower[0] * v0_np1
        rhs[-1] -= upper[-1] * vmax_np1
        lower[0] = 0.0
        upper[-1] = 0.0

        if index == n_steps - 1:
            v_prev = v.copy()
            dt_last = dt

        v[1:-1] = _solve_tridiagonal(lower, diag, upper, rhs)
        v[0] = v0_np1
        v[-1] = vmax_np1

        if project_every_step:
            kill_dead_nodes(v, tau_np1)
        elif tau_np1 in projection_levels:
            observe(v, tau_np1)
            n_projections += 1

    from scipy.interpolate import CubicSpline

    price = float(CubicSpline(s, v)(s0))

    live = ~dead
    if domain.truncated or (
        not option.is_continuous and cfg.barrier_alignment == "node"
    ):
        # Truncated: the barrier IS the boundary. Discrete and aligned: the
        # projection is cell-weighted, so it knows where the barrier falls
        # between two nodes and does not round it to one of them.
        effective = float(option.barrier)
    elif dead.any():
        # The scheme killed at the extreme dead node, which is the barrier it
        # actually priced -- the Zvan-Vetzal-Forsyth displacement, reported
        # rather than left to be inferred.
        effective = float(s[dead].max() if option.is_down else s[dead].min())
    else:
        effective = math.nan
    barrier_index = grid.index_of(float(option.barrier))

    meta: dict[str, Any] = {
        "method": "pde",
        "model": "BlackScholes",
        "instrument": "barrier",
        "leg": leg,
        "barrier_type": option.barrier_type,
        "barrier": float(option.barrier),
        "rebate": rebate,
        "monitoring": "continuous" if option.is_continuous else "discrete",
        "n_monitoring": option.n_monitoring,
        "barrier_alignment": cfg.barrier_alignment,
        "barrier_on_node": bool(domain.truncated or barrier_index >= 0),
        "barrier_node_index": barrier_index,
        "effective_barrier": effective,
        "effective_barrier_ratio": effective / float(option.barrier),
        "domain_truncated_at_barrier": domain.truncated,
        "live_nodes": int(np.count_nonzero(live)),
        "n_projections": n_projections,
        "barrier_projection": (
            "cell_weighted" if cfg.barrier_alignment == "node" else "node_sampled"
        ),
        "projection_levels": len(projection_levels),
        "theta": cfg.theta,
        "n_s": cfg.n_s,
        "n_t": cfg.n_t,
        "time_stepping": cfg.time_stepping,
        "implicit_startup_steps": (
            RANNACHER_STARTUP_STEPS if cfg.time_stepping == "rannacher" else 0
        ),
        "n_steps_taken": n_steps,
        "touched_at_inception": option.is_touched(s0),
    }
    meta.update(grid.meta)
    return _GridSolution(
        grid=grid, v=v, v_prev=v_prev, dt_last=dt_last, price=price, meta=meta
    )


def _priced_leg(option: BarrierOption) -> str:
    return KNOCK_OUT if option.is_knock_out else KNOCK_IN


def _solve_priced_leg(
    option: BarrierOption,
    model: BlackScholesModel,
    market: Market,
    cfg: PDEConfig,
) -> _GridSolution:
    """The solver the shared grid-Greek code reads; signature fixed by it."""
    return solve_leg(option, model, market, cfg, leg=_priced_leg(option))


def _settled_at_inception(
    option: BarrierOption, model: BlackScholesModel, market: Market
) -> float:
    """A spot already at or beyond the barrier settles the contract.

    A fact about the contract, so every engine in this package returns the same
    number: a knock-out is dead and worth its rebate, undiscounted; a knock-in
    is alive and worth the vanilla. Under *discrete* monitoring the inception
    spot is not an observation date, so this applies only to the continuous
    contract -- which is why it is guarded on `is_continuous` at the call site.
    """
    t = option.expiry
    if option.is_knock_out:
        return float(option.rebate)
    return float(
        bs_price(
            S=market.spot,
            K=option.strike,
            T=t,
            r=market.rate(t),
            sigma=model.sigma,
            q=market.dividend_yield(t),
            kind=option.kind,
        )
    )


def _degenerate(
    option: BarrierOption, model: BlackScholesModel, market: Market
) -> float:
    """`T = 0` or `sigma = 0`: the path is deterministic and there is no grid.

    Both are facts about the model rather than about a discretisation, so this
    delegates: to `qpl.engines.analytic.barrier.barrier_price` for the
    continuous contract and to `qpl.engines.mc.barrier`'s deterministic branch
    for the discrete one, which differ in exactly one case -- a forward that
    crosses the barrier only *after* the last observation, where the discrete
    contract survives and the continuous one does not.
    """
    t = option.expiry
    if option.is_continuous:
        return float(
            barrier_price(
                S=market.spot,
                K=option.strike,
                T=t,
                r=market.rate(t),
                sigma=model.sigma,
                q=market.dividend_yield(t),
                H=option.barrier,
                rebate=option.rebate,
                barrier_type=option.barrier_type,
                kind=option.kind,
            )
        )
    if t == 0.0:
        touched = option.is_touched(market.spot)
        intrinsic = (
            max(market.spot - option.strike, 0.0)
            if option.kind == "call"
            else max(option.strike - market.spot, 0.0)
        )
        if option.is_knock_out:
            return float(option.rebate if touched else intrinsic)
        return float(intrinsic if touched else option.rebate)
    return float(
        _discrete_degenerate_value(option, market, barrier=float(option.barrier))
    )


def price_barrier(
    option: BarrierOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: PDEConfig,
) -> PriceResult:
    """Price a single-barrier option by finite differences.

    Unlike the other three barrier engines this one prices **both** monitoring
    conventions: continuous monitoring by truncating the domain at the barrier
    and imposing the rebate there, discrete monitoring by projecting onto the
    rebate at each monitoring date on a time grid built to contain those dates.
    Knock-ins are priced by their own boundary problem, not by parity; see the
    module docstring for the formulation and for the parity identity that
    checks it.

    Parameters
    ----------
    option, model, market
        As for `qpl.engines.pde.pricers.price_european`.
    cfg
        Grid and time-stepping settings. Two fields matter here beyond the
        usual ones: `grid` (`"uniform"` or `"sinh"`, the latter concentrating
        nodes at the strike and the barrier) and `barrier_alignment`
        (`"node"`, the default, or `"none"` -- the measured first-order
        negative finding).

    Returns
    -------
    PriceResult
        With `meta` reporting the grid, whether the barrier landed on a node,
        the `effective_barrier` the scheme actually priced and its ratio to the
        contractual one, the number of projections applied, and the time grid.

    Raises
    ------
    InvalidInputError
        A bad `cfg`, or a strike or barrier outside the grid.
    """
    _validate(cfg)

    t = option.expiry
    if option.is_continuous and option.is_touched(market.spot):
        return PriceResult(
            value=_settled_at_inception(option, model, market),
            meta={
                "method": "pde",
                "model": "BlackScholes",
                "instrument": "barrier",
                "barrier_type": option.barrier_type,
                "monitoring": "continuous",
                "degenerate": "touched_at_inception",
                "touched_at_inception": True,
            },
        )

    if t == 0.0 or model.sigma == 0.0:
        return PriceResult(
            value=_degenerate(option, model, market),
            meta={
                "method": "pde",
                "model": "BlackScholes",
                "instrument": "barrier",
                "barrier_type": option.barrier_type,
                "monitoring": "continuous" if option.is_continuous else "discrete",
                "degenerate": "expiry" if t == 0.0 else "zero_vol",
                "touched_at_inception": option.is_touched(market.spot),
            },
        )

    sol = _solve_priced_leg(option, model, market, cfg)
    return PriceResult(value=sol.price, meta=sol.meta)


def greeks_barrier(
    option: BarrierOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: PDEConfig,
) -> GreeksResult:
    """Barrier Greeks read off the finished grid.

    The estimators are the European engine's (`pricers._greeks_from_grid`)
    called with this module's solver: delta and gamma from the three-point
    stencil interpolated to the spot, theta from the PDE identity, vega and rho
    by bump-and-revalue on the same grid. This is the first barrier engine in
    this package with real Greeks -- the lattice and the simulation both
    register a callable that raises, for reasons that are about *their*
    discretisations (a sawtooth amplified by `1/dt`, and a pathwise derivative
    that is a surface measure) and do not apply to a grid whose barrier is a
    boundary.

    What to expect, measured in `tests/test_pde_barrier.py`: away from the
    barrier, agreement with central differences of the closed form within the
    grid's own error; and Slice 12's finding reproduced -- gamma does **not**
    blow up as `S -> H`, it converges to a finite limit having changed **sign**,
    because a knock-out near its barrier is concave where the vanilla is
    convex. The singularity is at the corner `(S = H, t = T)`, not on the
    barrier.

    Raises
    ------
    InvalidInputError
        At `T = 0` or `sigma = 0` (no grid, and gamma is a point mass in both
        limits), or when the spot has already touched a continuously monitored
        barrier, where the value is settled and its derivative in the spot is
        zero on one side and undefined on the other.
    """
    _validate(cfg)
    if option.is_continuous and option.is_touched(market.spot):
        raise InvalidInputError(
            "the spot has already touched the barrier, so a continuously "
            "monitored contract is settled at inception and there is no grid "
            "to differentiate: a knock-out is worth its rebate and a knock-in "
            "is the vanilla. Use method='analytic' for the settled value."
        )
    if cfg.greeks_method == "bump":
        return _greeks_by_bump(option, model, market, cfg, price_fn=price_barrier)
    return _greeks_from_grid(option, model, market, cfg, solve=_solve_priced_leg)
