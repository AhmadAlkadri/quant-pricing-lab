"""American option pricing by finite differences: the LCP solved with PSOR.

What is being solved
--------------------
Under Black-Scholes an American vanilla is not a PDE problem but a *linear
complementarity problem* (LCP). Writing `L` for the Black-Scholes operator and
`g(S)` for the exercise payoff, the value function satisfies, at every
`(S, t)`, all three of

    V - g >= 0,            (the holder may exercise, so V dominates the payoff)
    V_t + L V <= 0,        (holding is never better than the PDE allows)
    (V - g) (V_t + L V) = 0.   (complementarity: one of the two is tight)

The third line is the whole content: in the continuation region the PDE holds
with equality and `V > g`; in the exercise region `V = g` and the PDE holds as
a strict inequality. The free boundary between them is not given in advance,
which is why a plain linear solve cannot be used.

Discretised in `tau = T - t` with the same theta scheme the European engine
uses, one time step becomes the algebraic LCP

    A x - b >= 0,   x - g >= 0,   (x - g) * (A x - b) = 0   componentwise,

with `A = I - theta dt L_h`, `b = (I + (1 - theta) dt L_h) v_old` plus the
Dirichlet contributions, and `L_h` the tridiagonal operator built by
`qpl.engines.pde.pricers._operator` -- the same one the European engine
assembles, not a copy of it.

Projected SOR (PSOR)
--------------------
PSOR solves that system by running an SOR sweep and clipping every updated
component against the obstacle:

    y_i   = (b_i - sum_{j != i} A_ij x_j) / A_ii        (Gauss-Seidel value)
    x_i  <- max( g_i , x_i + omega (y_i - x_i) )        (relax, then project)

Cryer (1971), "The solution of a quadratic programming problem using
systematic overrelaxation", SIAM Journal on Control 9(3), 385-392, is the
convergence result for this iteration on a symmetric positive definite matrix;
the finance formulation -- obstacle problem, free boundary, PSOR -- is the one
in Wilmott, Dewynne and Howison (1993), "Option Pricing: Mathematical Models
and Computation". Forsyth and Vetzal (2002), "Quadratic convergence for
valuing American options using a penalty method", SIAM Journal on Scientific
Computing 23(6), 2095-2122, is the comparison point: they replace the
projection by a penalty term and report, among other things, that PSOR
iteration counts grow as the grid is refined. Both halves of that are measured
here and reported in `docs/notes/pde_american_psor.md`; the growth is real but
it is driven by `dt / ds**2`, not by `n` as such.

Why the sweep is red-black
--------------------------
`qpl.numerics.linear_systems.sor_solve` is the in-repo SOR and it is *not*
reused directly, for two reasons that are about its interface rather than its
correctness: it takes a dense `(n, n)` matrix, so one sweep costs `O(n**2)`
where a tridiagonal sweep costs `O(n)`, and it has no hook at which to project
an iterate onto a constraint set. What is reused is its shape -- the
`(1 - omega) x + omega * gauss_seidel` update, the `0 < omega < 2` validation,
and reporting iterations and convergence rather than only the answer -- and it
is used as the *reference* for this module's solver: with the obstacle removed,
`tests/test_pde_american.py` checks that this sweep reproduces both
`sor_solve`'s answer and the European engine's direct banded solve.

The sweep here updates even-indexed nodes first, then odd-indexed ones
("red-black"). On a tridiagonal matrix each node's only neighbours are of the
other colour, so a whole colour can be updated in one vectorised numpy
expression and the result is *exactly* a Gauss-Seidel/SOR sweep in the
permuted order red-then-black. A tridiagonal matrix is consistently ordered in
Young's sense under both the natural and the red-black ordering, so the two
have the same SOR spectral radius and the same optimal `omega`; they reach the
same limit, generally by different iterates. `tests/test_pde_american.py`
checks that against a natural-order PSOR written out in the test.

Degenerate limits
-----------------
At `T = 0` and at `sigma = 0` there is nothing to march and the answer is a
statement about the model, not about a grid; `_degenerate_value` from
`qpl.engines.tree.american` is that statement, including the `q > r` interior
turning point pinned in Slice 2, and it is imported rather than rewritten so
that the two engines cannot disagree about it. See `price_american` for what
that does and does not prove.

Derivation, measured tables and the three-engine comparison:
`docs/notes/pde_american_psor.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.options import AmericanOption
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult
from ..tree.american import _degenerate_value
from .pricers import (
    RANNACHER_STARTUP_STEPS,
    PDEConfig,
    _build_grid,
    _dirichlet,
    _greeks_by_bump,
    _greeks_from_grid,
    _GridSolution,
    _operator,
    _payoff,
    _time_levels,
    _validate,
)

__all__ = ["PSORConfig", "greeks_american", "price_american"]


@dataclass(frozen=True)
class PSORConfig:
    """Iteration settings for the projected SOR solve at each time step.

    Parameters
    ----------
    omega
        Over-relaxation parameter, `0 < omega < 2`. Default `1.2`; see
        `PSOR_OMEGA_DEFAULT` for the measurement behind that number.
    tol
        **Absolute** tolerance, in price units, on the largest single-component
        change over one full sweep: the iteration stops at the first sweep with
        `max_i |x_i^{k+1} - x_i^k| <= tol`. It is a *step* criterion, not a
        residual criterion, and not relative -- the values on the grid span
        `0` to about `K`, so a relative test would be dominated by the deep
        out-of-the-money nodes where everything is zero.

        Default `1e-8`. Measured against a `tol = 1e-13` solve on the same
        grid (ATM put, aligned, Rannacher, `n_s = n_t = n`), the price this
        default gives up is

            n        100       200       400       800      1600
            |err|  2.3e-08   2.7e-08   6.0e-08   1.1e-07   1.5e-07

        against discretisation errors of `2.1e-02` down to `1.2e-04` on the
        same grids -- between five and three orders of magnitude of headroom,
        narrowing as the grid is refined, which is the right way round to be
        wrong. Tightening to `1e-10` costs about 27% more sweeps and to
        `1e-12` about 55%. It cannot usefully be tightened below about
        `1e-14 * K`: that is the round-off scale of the values themselves, and
        the max-update criterion then never fires.
    max_iter
        Sweep cap per time step. Default `10_000`, about six times the worst
        count measured on any grid in this repository -- 1642 mean sweeps at
        `n_s = 3200, n_t = 40` with the default `omega`, where `dt / ds**2` is
        four orders of magnitude larger than on the `n_s = n_t` path.
    on_max_iter
        What happens when a time step exhausts `max_iter` without meeting
        `tol`:

        - `"raise"` (default): raise `InvalidInputError`, naming the step.
        - `"flag"`: carry on and record it in
          `PriceResult.meta["psor_unconverged_steps"]`, with
          `meta["psor_converged"] = False`.

        There is deliberately no third option in which an unconverged solve
        looks like a converged one.
    """

    omega: float = 1.2
    tol: float = 1e-8
    max_iter: int = 10_000
    on_max_iter: Literal["raise", "flag"] = "raise"


PSOR_OMEGA_DEFAULT = 1.2
"""Why the default over-relaxation is 1.2, and what it is not optimal for.

For a consistently ordered matrix Young's theory gives the optimal relaxation
as `omega* = 2 / (1 + sqrt(1 - rho_J**2))` with `rho_J` the Jacobi spectral
radius. Here `rho_J` is set by how strongly the step matrix
`A = I - theta dt L_h` is diagonally dominant, and that ratio is roughly
`theta dt sigma**2 S**2 / ds**2` against `1`: it depends on `dt / ds**2`, not
on the size of the grid. There is therefore no single optimal `omega` for this
engine, and the default is a choice about which refinement path to be good on.

Mean sweeps per time step, ATM American put (`S = K = 100, r = 5%, q = 0,
sigma = 20%, T = 1`), aligned grid, Rannacher start-up, `tol = 1e-8`, on the
path `n_s = n_t = n` -- the path the convergence study in
`tests/test_pde_american_convergence.py` uses:

    n       w=1.00  w=1.05  w=1.10  w=1.15  w=1.20  w=1.30  w=1.50
    100      5.96    7.00    8.51   10.03   11.33   14.38   24.52
    200      7.33    7.00    8.26    9.79   11.17   14.15   23.41
    400      9.72    7.98    8.07    9.28   10.65   13.50   22.49
    800     13.99   12.12   10.30    9.51   10.18   13.14   21.45
    1600    21.12   18.97   16.78   14.65   12.75   12.38   20.13

The per-grid minimum moves from `1.00` at `n = 100` to `1.30` at `n = 1600`,
because `dt / ds**2` grows like `n` along this path. `omega = 1.2` is the
flattest column over the whole range -- 11.33 down to 10.18 and back to 12.75,
never worse than 1.9x the per-grid optimum -- while `omega = 1.0` degrades by
3.5x across the same span and `omega = 1.5` costs about twice the optimum
everywhere. It is also inside the 1.2-1.5 band that the standard references
suggest, which is the weakest of the reasons to pick it.

It is **not** the right value on a grid that is fine in space and coarse in
time. On `n_s = 80 n_t` the same put needs, at `n_t = 20`:

    w=1.0   w=1.2   w=1.4   w=1.6   w=1.8
    1367     952     640     391     173

and the trend is still falling at 1.8. If a caller uses such a grid, raising
`omega` towards 1.8 is worth an order of magnitude and
`PSORConfig(omega=...)` is how to do it.
"""


def _validate_psor(psor: PSORConfig) -> None:
    """Reject unusable PSOR settings.

    The `0 < omega < 2` bound is the classical necessary condition for SOR
    (Kahan): outside it the iteration matrix has spectral radius at least
    `|omega - 1| >= 1` and cannot converge for any starting point. The message
    is the one `qpl.numerics.linear_systems.sor_solve` uses, so the two SOR
    implementations in this repository report the same failure the same way.
    """
    if not math.isfinite(psor.omega) or psor.omega <= 0.0 or psor.omega >= 2.0:
        raise InvalidInputError("omega must satisfy 0 < omega < 2")
    if not math.isfinite(psor.tol) or psor.tol <= 0.0:
        raise InvalidInputError("tol must be > 0")
    if not isinstance(psor.max_iter, int) or psor.max_iter < 1:
        raise InvalidInputError("max_iter must be >= 1")
    if psor.on_max_iter not in {"raise", "flag"}:
        raise InvalidInputError("on_max_iter must be 'raise' or 'flag'")


def _psor_sweeps(
    x: np.ndarray,
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    *,
    omega: float,
    tol: float,
    max_iter: int,
) -> tuple[int, float]:
    """Run red-black PSOR on a tridiagonal system, in place on `x`.

    `lower`, `diag`, `upper` are the three bands of `A` with `lower[0]` and
    `upper[-1]` already zeroed (their Dirichlet contribution having been folded
    into `rhs`), so the phantom neighbours at either end never contribute.

    Returns `(sweeps, last_update)` where `last_update` is the largest
    single-component change in the final sweep. The caller decides what an
    unconverged solve means; this function only reports.

    `obstacle` may be `-inf` everywhere, in which case the projection is a
    no-op and the iteration is plain SOR for `A x = b`. That is not a curiosity:
    it is the path `tests/test_pde_american.py` uses to check the solver
    against `sor_solve` and against the European engine's direct banded solve.
    """
    colours = (slice(0, None, 2), slice(1, None, 2))
    shift_down = np.empty_like(x)
    shift_up = np.empty_like(x)

    update = math.inf
    for sweep in range(1, max_iter + 1):
        update = 0.0
        for colour in colours:
            # Rebuilt per colour: the first colour's new values must be visible
            # to the second, which is what makes this Gauss-Seidel rather than
            # Jacobi.
            shift_down[0] = 0.0
            shift_down[1:] = x[:-1]
            shift_up[-1] = 0.0
            shift_up[:-1] = x[1:]

            gauss_seidel = (
                rhs[colour] - lower[colour] * shift_down[colour] - upper[colour] * shift_up[colour]
            ) / diag[colour]
            current = x[colour]
            relaxed = current + omega * (gauss_seidel - current)
            projected = np.maximum(obstacle[colour], relaxed)
            step = float(np.max(np.abs(projected - current))) if current.size else 0.0
            update = max(update, step)
            x[colour] = projected

        if update <= tol:
            return sweep, update
    return max_iter, update


_INTRINSIC_FLOOR_REL = 1e-12
"""Relative floor below which a node does not count as in the money.

Same role, and the same value, as the lattice engine's floor in
`qpl.engines.tree.american`: far out of the money the value and the payoff are
both zero, and `V <= g` there is a statement about `0 <= 0` rather than about
exercising. Scaled by the strike so it means the same thing at `K = 0.01` and
`K = 10_000`.
"""


def _boundary_node(s_grid: np.ndarray, active: np.ndarray, *, kind: str) -> float:
    """The exercise boundary at one time level, or NaN if nothing is active.

    For a put the exercise region is `S <= B(t)`, so the boundary is the
    largest active node; for a call it is `S >= B(t)` and therefore the
    smallest. Identical convention to the lattice engine, which is what makes
    the two boundaries comparable in
    `tests/test_pde_american_convergence.py`.
    """
    where = np.flatnonzero(active)
    if where.size == 0:
        return math.nan
    return float(s_grid[where[-1] if kind == "put" else where[0]])


def _active_set(v: np.ndarray, payoff: np.ndarray, floor: float) -> np.ndarray:
    """Nodes at which the obstacle constraint is tight and the option is ITM.

    The test is exact equality, not a tolerance, and that is a property of the
    solver rather than a convenience: the projection step assigns
    `np.maximum(obstacle, relaxed)`, which returns the obstacle value *itself*
    when the constraint binds, so an active node carries the payoff to the last
    bit. Nodes in the continuation region sit strictly above it.
    """
    return (v <= payoff) & (payoff > floor)


@dataclass(frozen=True)
class _PSORDiagnostics:
    """What the sweeps did, collected across the whole march."""

    iterations: np.ndarray
    unconverged_steps: tuple[int, ...]
    max_complementarity: float
    min_constraint_slack: float
    min_operator_residual: float


def _solve_grid_american(
    option: AmericanOption,
    model: BlackScholesModel,
    market: Market,
    cfg: PDEConfig,
) -> _GridSolution:
    """March the LCP from the payoff at `tau = 0` to `tau = T`.

    Assumes `_validate(cfg)` and `_validate_psor(cfg.psor)` have run and that
    neither degenerate case applies. The grid, the operator and the European
    Dirichlet values all come from `qpl.engines.pde.pricers`; what is new here
    is the obstacle, the boundary values that respect it, and the PSOR solve
    that replaces the banded one.

    Returns a `_GridSolution` so that the shared grid-Greek code in
    `pricers._greeks_from_grid` can read it, with the American-specific
    diagnostics carried in `meta`.
    """
    s0 = market.spot
    k = option.strike
    t = option.expiry
    sigma = model.sigma
    kind = option.kind
    psor = cfg.psor

    grid = _build_grid(k, s0, cfg)
    s_grid = grid.s
    ds = grid.ds
    s_max = grid.s_max
    payoff = _payoff(kind, k, s_grid)
    obstacle = payoff[1:-1]
    floor = _INTRINSIC_FLOOR_REL * k

    steps = _time_levels(t, cfg)
    n_steps = len(steps)
    v = payoff.copy()

    # Boundary and calendar time, one entry per time level including `tau = 0`.
    # Rannacher makes the tau grid non-uniform, so the times are reported
    # alongside rather than left to be reconstructed as `linspace`.
    taus = np.empty(n_steps + 1)
    taus[0] = 0.0
    boundary_by_tau = np.empty(n_steps + 1)
    boundary_by_tau[0] = _boundary_node(s_grid, _active_set(v, payoff, floor), kind=kind)
    early_exercise_nodes = 0

    iterations = np.empty(n_steps, dtype=int)
    unconverged: list[int] = []
    max_complementarity = 0.0
    min_slack = math.inf
    min_residual = math.inf

    v_prev = v.copy()
    dt_last = steps[-1][2]

    for index, (tau_n, tau_np1, dt, theta_step) in enumerate(steps):
        # American Dirichlet values: the European ones, lifted to the payoff
        # wherever exercising the boundary node is worth more. At `S = 0` this
        # turns the put's `K e^{-r tau}` into `K` (exercise now); at `S = s_max`
        # it leaves the call's discounted forward intrinsic alone, because that
        # already dominates `s_max - K`. Both are the boundary-node instance of
        # the same `V >= g` constraint the interior gets from the projection.
        v0_n, vmax_n = _dirichlet(kind, k, s_max, market.df_r(tau_n), market.df_q(tau_n))
        v0_np1, vmax_np1 = _dirichlet(kind, k, s_max, market.df_r(tau_np1), market.df_q(tau_np1))
        v0_n = max(v0_n, payoff[0])
        vmax_n = max(vmax_n, payoff[-1])
        v0_np1 = max(v0_np1, payoff[0])
        vmax_np1 = max(vmax_np1, payoff[-1])

        v[0] = v0_n
        v[-1] = vmax_n

        r = market.rate(tau_np1)
        q = market.dividend_yield(tau_np1)
        a, b, c = _operator(grid, sigma, r, q)

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
            v_prev = v.copy()
            dt_last = dt

        # Previous time level as the starting iterate. Over one step the
        # solution moves by `O(dt)`, so this is a much better guess than zero
        # and is where most of the "only ten sweeps" comes from.
        x = v[1:-1].copy()
        sweeps, _ = _psor_sweeps(
            x,
            lower,
            diag,
            upper,
            rhs,
            obstacle,
            omega=psor.omega,
            tol=psor.tol,
            max_iter=psor.max_iter,
        )
        iterations[index] = sweeps
        if sweeps >= psor.max_iter:
            if psor.on_max_iter == "raise":
                raise InvalidInputError(
                    f"PSOR did not converge within max_iter={psor.max_iter} sweeps at "
                    f"time step {index} (tau={tau_np1:.6g}); raise max_iter, loosen tol, "
                    f"or set PSORConfig(on_max_iter='flag') to continue and record it"
                )
            unconverged.append(index)

        # Complementarity, measured rather than assumed. `residual = A x - b`
        # should be >= 0 everywhere (`L V <= 0`), the slack `x - g` should be
        # >= 0 everywhere (exact, by the projection), and at every node one of
        # the two should be zero. The elementwise minimum of the two is the
        # direct test of that, and the worst one over the whole march is what
        # `tests/test_pde_american.py` asserts on.
        residual = diag * x - rhs
        residual[:-1] += upper[:-1] * x[1:]
        residual[1:] += lower[1:] * x[:-1]
        slack = x - obstacle
        max_complementarity = max(
            max_complementarity, float(np.max(np.minimum(slack, np.abs(residual))))
        )
        min_slack = min(min_slack, float(np.min(slack)))
        min_residual = min(min_residual, float(np.min(residual)))

        v[1:-1] = x
        v[0] = v0_np1
        v[-1] = vmax_np1

        taus[index + 1] = tau_np1
        active = _active_set(v, payoff, floor)
        early_exercise_nodes += int(np.count_nonzero(active))
        boundary_by_tau[index + 1] = _boundary_node(s_grid, active, kind=kind)

    from scipy.interpolate import CubicSpline

    price = float(CubicSpline(s_grid, v)(s0))

    # Report in calendar time, ascending, to match the lattice engine's
    # `meta["exercise_boundary"]`: index 0 is now, the last entry is expiry.
    boundary = boundary_by_tau[::-1].copy()
    times = (t - taus)[::-1].copy()
    boundary.setflags(write=False)
    times.setflags(write=False)
    iterations.setflags(write=False)

    diagnostics = _PSORDiagnostics(
        iterations=iterations,
        unconverged_steps=tuple(unconverged),
        max_complementarity=max_complementarity,
        min_constraint_slack=min_slack,
        min_operator_residual=min_residual,
    )

    meta: dict[str, Any] = {
        "method": "pde",
        "model": "BlackScholes",
        "exercise": "american",
        "scheme": "psor",
        "theta": cfg.theta,
        "n_s": cfg.n_s,
        "n_t": cfg.n_t,
        "s_max": s_max,
        "ds": ds,
        "strike_alignment": cfg.strike_alignment,
        "time_stepping": cfg.time_stepping,
        "implicit_startup_steps": (
            RANNACHER_STARTUP_STEPS if cfg.time_stepping == "rannacher" else 0
        ),
        "n_steps_taken": n_steps,
        "psor_omega": psor.omega,
        "psor_tol": psor.tol,
        "psor_max_iter": psor.max_iter,
        "psor_iterations": iterations,
        "psor_iterations_total": int(iterations.sum()),
        "psor_iterations_mean": float(iterations.mean()),
        "psor_iterations_max": int(iterations.max()),
        "psor_converged": not unconverged,
        "psor_unconverged_steps": diagnostics.unconverged_steps,
        "lcp_max_complementarity": max_complementarity,
        "lcp_min_constraint_slack": min_slack,
        "lcp_min_operator_residual": min_residual,
        "exercise_boundary": boundary,
        "exercise_boundary_times": times,
        "early_exercise_node_count": early_exercise_nodes,
        "exercise_time_level_count": int(np.count_nonzero(~np.isnan(boundary_by_tau[1:]))),
    }
    meta.update(grid.meta)
    return _GridSolution(
        grid=grid,
        v=v,
        v_prev=v_prev,
        dt_last=dt_last,
        price=price,
        meta=meta,
    )


def _degenerate_result(option: AmericanOption, market: Market, cfg: PDEConfig) -> PriceResult:
    """`T = 0` or `sigma = 0`, where there is no grid to march."""
    value = _degenerate_value(option, market)
    meta: dict[str, Any] = {
        "method": "pde",
        "model": "BlackScholes",
        "exercise": "american",
        "scheme": "psor",
        "degenerate": "expiry" if option.expiry == 0.0 else "zero_vol",
        "exercise_boundary": None,
        "exercise_boundary_times": None,
        "psor_iterations": None,
        "psor_converged": True,
        "n_steps_taken": 0,
        "strike_alignment": cfg.strike_alignment,
        "time_stepping": cfg.time_stepping,
    }
    return PriceResult(value=float(value), meta=meta)


def price_american(
    option: AmericanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: PDEConfig,
) -> PriceResult:
    """Price an American option by solving the Black-Scholes LCP with PSOR.

    Parameters
    ----------
    option
        American option (`call` or `put`).
    model
        Black-Scholes model with constant volatility.
    market
        Market object providing spot and flat rate/dividend curves.
    cfg
        Grid, time-stepping and PSOR settings. `cfg.psor` is read only here;
        the European engine ignores it.

    Returns
    -------
    PriceResult
        Price, with metadata carrying the grid description the European engine
        reports plus:

        - ``psor_iterations``: read-only array of sweeps per time step;
          ``psor_iterations_total`` / ``_mean`` / ``_max`` summarise it.
        - ``psor_converged`` and ``psor_unconverged_steps``: empty and `True`
          unless `PSORConfig(on_max_iter="flag")` let an unconverged step
          through.
        - ``lcp_max_complementarity``, ``lcp_min_constraint_slack``,
          ``lcp_min_operator_residual``: the LCP checked rather than assumed,
          worst case over the whole march. See `_solve_grid_american`.
        - ``early_exercise_node_count``: grid nodes strictly before expiry at
          which the constraint is tight and the option is in the money, summed
          over time levels, and ``exercise_time_level_count``, the number of
          time levels with a non-empty exercise region.
        - ``exercise_boundary`` and ``exercise_boundary_times``: read-only
          arrays of length `n_steps_taken + 1`, in ascending calendar time,
          holding the edge node of the exercise region at each time level --
          the largest active node for a put, the smallest for a call -- or
          `NaN` where nothing is active. The times are reported explicitly
          because Rannacher start-up makes the time grid non-uniform.

    Notes
    -----
    **Boundary conditions.** The Dirichlet values are the European ones lifted
    to the payoff: `max(g(0), V_eu(0))` at `S = 0` and `max(g(s_max),
    V_eu(s_max))` at the top. For a put that turns `K e^{-r tau}` into `K`,
    which is the correct American value at `S = 0` (exercise now, for `K`); for
    a call the discounted forward intrinsic already dominates `s_max - K`, so
    nothing changes and a no-dividend American call keeps its European upper
    boundary. Both are the boundary-node instance of `V >= g`.

    **Degenerate inputs.** At `T = 0` the price is the payoff. At `sigma = 0`
    it is the best discounted intrinsic value along the deterministic forward
    path, which is *not* in general `max(intrinsic now, discounted forward
    intrinsic)` -- when `q > r` there is an interior turning point. That result
    is `qpl.engines.tree.american._degenerate_value` and it is imported here
    rather than restated, because it is a fact about the model and not about
    either discretisation. The consequence for evidence is worth being blunt
    about: a test asserting that this engine and the lattice engine agree at
    `sigma = 0` is checking shared code, not two derivations, so
    `tests/test_pde_american.py` re-derives the value by brute-force
    maximisation over a fine time grid instead.

    Raises
    ------
    InvalidInputError
        If `cfg` or `cfg.psor` is out of range, if the strike falls outside the
        spot grid, or if a time step exhausts `cfg.psor.max_iter` while
        `on_max_iter="raise"`.
    """
    _validate(cfg)
    _validate_psor(cfg.psor)

    if option.expiry == 0.0 or model.sigma == 0.0:
        return _degenerate_result(option, market, cfg)

    sol = _solve_grid_american(option, model, market, cfg)
    return PriceResult(value=sol.price, meta=sol.meta)


def greeks_american(
    option: AmericanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: PDEConfig,
) -> GreeksResult:
    """Greeks for an American option, read off the finished PSOR grid.

    The estimators are the European ones -- `pricers._greeks_from_grid`, called
    with this module's solver -- for the same reason the lattice engine reuses
    its European estimators: a central stencil reads a slope off neighbouring
    values and does not care how the values were produced. Delta and gamma are
    second-order central differences interpolated to the spot; vega and rho are
    bump-and-revalue on the same grid; theta comes from the PDE identity.

    Two American-specific points, both encoded rather than left to the reader:

    - **Theta in the exercise region.** The identity
      `V_t = -(1/2 sigma^2 S^2 V_SS + (r-q) S V_S - r V)` is the PDE, and inside
      the exercise region the PDE does *not* hold -- that is the whole content
      of the complementarity condition. There `V = g(S)` is independent of time
      and the true theta is exactly `0`, while the identity would return
      `rK - qS` for a put. So when the spot sits in the exercise region this
      engine reports `theta = 0.0` and says so in
      `meta["theta_source"]`; `greeks_method="bump"` is unaffected, being a
      spot bump.
    - **Gamma near the free boundary.** The American value function has a kink
      in `S` at the boundary (value matching holds; smooth pasting only in the
      continuum limit), so a second difference taken across it is differencing
      a function whose second derivative is a delta in the limit. Away from the
      boundary gamma is as well behaved as the European one. The reference ATM
      put's spot sits well inside the continuation region, which is where
      `tests/test_pde_american.py` measures it.

    Parameters
    ----------
    option, model, market
        As for `price_american`.
    cfg
        As for `price_american`; `greeks_method` selects the estimator exactly
        as it does for European options.

    Returns
    -------
    GreeksResult
        Delta, gamma, vega, theta, rho, with `meta` naming the source of each
        and carrying the PSOR and LCP diagnostics under `meta["pde_meta"]`.

    Raises
    ------
    InvalidInputError
        With `greeks_method="grid"`, if `T = 0` or `sigma = 0`: there is no
        grid to differentiate and gamma is a distribution rather than a
        function in both limits, exactly as for European options.
        `price_american` still returns the correct price in both cases.
    """
    _validate(cfg)
    _validate_psor(cfg.psor)

    if cfg.greeks_method == "bump":
        # Three full American solves at three spots, differenced. Kept for the
        # same reason the European engine keeps it -- as a named, measured
        # alternative -- and it inherits the same two defects (a fixed O(h^2)
        # bias and three differently aligned grids), plus NaN vega/theta/rho.
        return _greeks_by_bump(option, model, market, cfg, price_fn=price_american)

    grid = _build_grid(option.strike, market.spot, cfg)
    return _greeks_from_grid(
        option,
        model,
        market,
        cfg,
        solve=_solve_grid_american,
        obstacle=_payoff(option.kind, option.strike, grid.s),
    )
