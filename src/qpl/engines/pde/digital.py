"""Cash-or-nothing digitals on the finite-difference grid.

The grid, the operator, the time march, the tridiagonal solve and the Greek
stencils are the European engine's, imported rather than copied
(`qpl.engines.pde.pricers`). Two things change, and between them they are the
entire slice:

1. the **terminal condition** is a step function rather than a hockey stick;
2. the **boundary data** is `cash e^{-r tau}` at the in-the-money end and `0`
   at the other, with no `e^{-q tau}` term -- the payoff is a fixed cash
   amount, not a share.

Why a jump is worse than a kink
-------------------------------
A finite-difference scheme is second order where the Taylor expansions behind
its stencils are valid. At a kink the first derivative jumps, so the expansion
fails at one point and the *local* truncation error there is `O(1)` instead of
`O(ds**2)`; the damage is confined and shows up mostly in gamma. At a jump the
*function* is discontinuous, and there is no expansion at all: the terminal
data itself is wrong by `O(cash)` at the node nearest the strike, for any
`ds`. That error is then diffused over the march, and what survives at `tau = T`
is `O(cash * ds / (sigma sqrt(T)))` in the price -- first order, not second.
This is the pathology the slice exists to pin, and it is measured below.

The remedies, and how they interact
-----------------------------------
Two are implemented here, and the interesting result is that they are the same
remedy seen from two sides.

**Cell averaging** (`PDEConfig(payoff_projection="cell_average")`). The node
value is meant to represent the solution near the node, so the consistent way
to put a discontinuous function on a grid is to project it onto the space of
grid functions -- in the `L2` sense, the average over the node's cell
`[S_i - ds/2, S_i + ds/2]`. For an indicator that average is exact and free:
it is the fraction of the cell that lies in the money, so the one cell that
straddles the strike gets a value strictly between `0` and `cash` instead of
being rounded to one of them. This is the standard construction; see Pooley,
Forsyth and Vetzal (2003), "Convergence remedies for non-smooth payoffs in
option pricing", Journal of Computational Finance 6(4), 25-40, which analyses
exactly this class of remedy (the implementation and the measurements here are
this repository's own).

**Strike-midpoint alignment** (`PDEConfig(strike_alignment="midpoint")`) nudges
the spacing so that `K = (j + 1/2) ds` for an integer `j`. That was introduced
in Slice 0 to make the *kink* symmetric between two nodes. For a jump it does
something sharper: `K` then lands exactly on a **cell face**, since node `j`'s
cell is `[(j - 1/2) ds, (j + 1/2) ds]`. No cell straddles the strike, every
cell is entirely in or entirely out of the money, and the cell average of the
indicator is therefore identical to its point sample.

So on a midpoint-aligned grid the projection is a **no-op**, and the two
remedies cannot be stacked -- alignment has already done what the projection
would do. That contradicts the slice's expectation that the combination would
be needed, and `tests/test_digital_pde.py` pins it: the two configurations
agree to 1e-16, i.e. to the round-off in `S_i + ds/2` against `(j + 1/2) ds`.
What the projection *is* for is the unaligned grid, where it is the only one of
the two available.

Rannacher start-up is still needed on top of either, and for the usual reason:
Crank-Nicolson's amplification factor tends to `-1` for the stiffest modes, and
a jump has far more high-frequency content than a kink for those modes to be
made of. See `RANNACHER_STARTUP_STEPS` in `qpl.engines.pde.pricers` and Giles
and Carter (2006), Journal of Computational Finance 9(4), 89-112.

Everything measured is in `tests/test_digital_pde.py` and
`docs/notes/digital_options_discontinuous_payoffs.md`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...instruments.options import DigitalOption
from ...instruments.payoffs import digital_payoff
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult
from .grid import SpotGrid
from .pricers import (
    PDEConfig,
    _greeks_by_bump,
    _greeks_from_grid,
    _GridSolution,
    _solve_grid,
    _validate,
)

__all__ = ["greeks_digital", "price_digital"]


def digital_payoff_on_grid(
    option: DigitalOption, grid: SpotGrid, projection: str
) -> np.ndarray:
    """The digital's terminal condition on a spot grid.

    With `projection="none"` this is `digital_payoff` evaluated at the nodes.

    With `projection="cell_average"` node `i` carries the mean of the payoff
    over its cell. On a **uniform** grid that cell is `[S_i - ds/2, S_i + ds/2]`
    and the mean is, for a call,

        cash * clip((S_i + ds/2 - K) / ds, 0, 1),

    the fraction of the cell above the strike, which is exact for a step
    function -- no quadrature and no smoothing parameter. The put is the
    mirror image, `cash * clip((K - S_i + ds/2) / ds, 0, 1)`, and the two sum
    to `cash` at every node, so the projection preserves the static
    replication identity exactly.

    On a **non-uniform** grid the cell is `[(S_{i-1} + S_i)/2, (S_i + S_{i+1})/2]`
    and the same fraction is taken over that width; the end nodes' cells are
    completed by reflecting their single face. Both expressions are written
    out separately rather than unified, because the uniform one is the
    pre-Slice-13 arithmetic and `S_i - ds/2` is not bit-for-bit
    `(S_{i-1} + S_i)/2` even when the two are equal in exact arithmetic.

    The two end nodes are overwritten by the Dirichlet data at the first time
    step, so their cells extending past the domain does not matter.
    """
    s_grid = grid.s
    if projection == "cell_average":
        if grid.uniform:
            ds = grid.ds
            half = 0.5 * ds
            lo = s_grid - half
            hi = s_grid + half
            width = ds
        else:
            faces = grid.cell_faces
            lo = np.concatenate([[2.0 * s_grid[0] - faces[0]], faces])
            hi = np.concatenate([faces, [2.0 * s_grid[-1] - faces[-1]]])
            width = hi - lo
        if option.kind == "call":
            fraction = (hi - option.strike) / width
        else:
            fraction = (option.strike - lo) / width
        # Clip the *fraction* rather than the overlap length: dividing first
        # and bounding second guarantees the result lies in [0, cash] exactly,
        # where `clip(overlap, 0, ds) / ds` can round to one ulp above 1.
        return option.cash * np.clip(fraction, 0.0, 1.0)
    return np.asarray(
        digital_payoff(s_grid, option.strike, option.cash, option.kind), dtype=float
    )


def _dirichlet(option: DigitalOption):
    """Boundary values at `S = 0` and `S = s_max`, as a function of the dfs.

    A digital call is certain to pay at `S = s_max` (the grid extends to four
    times the spot by default, far above any strike it will be asked about)
    and certain not to pay at `S = 0`, so its boundary values are `0` and
    `cash e^{-r tau}`. The put is the mirror. Unlike the vanilla's upper
    boundary there is no `e^{-q tau}` term: what is being delivered is cash,
    not a share, so the dividend yield does not enter.
    """

    def values(df_r: float, df_q: float) -> tuple[float, float]:
        paid = option.cash * df_r
        return (0.0, paid) if option.kind == "call" else (paid, 0.0)

    return values


def _solve_digital_grid(
    option: DigitalOption,
    model: BlackScholesModel,
    market: Market,
    cfg: PDEConfig,
) -> _GridSolution:
    """March the European grid with the digital's terminal and boundary data."""
    return _solve_grid(
        option,  # type: ignore[arg-type]
        model,
        market,
        cfg,
        payoff=lambda grid: digital_payoff_on_grid(
            option, grid, cfg.payoff_projection
        ),
        dirichlet=_dirichlet(option),
    )


def _degenerate_value(option: DigitalOption, market: Market) -> float:
    """Price at `T = 0` or `sigma = 0`; there is no grid in either limit."""
    t = option.expiry
    if t == 0.0:
        return float(digital_payoff(market.spot, option.strike, option.cash, option.kind))
    forward = market.spot * math.exp((market.rate(t) - market.dividend_yield(t)) * t)
    return market.df_r(t) * float(
        digital_payoff(forward, option.strike, option.cash, option.kind)
    )


def price_digital(
    option: DigitalOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: PDEConfig,
) -> PriceResult:
    """Price a cash-or-nothing digital by solving the Black-Scholes PDE.

    Parameters
    ----------
    option, model, market
        As for `qpl.engines.pde.pricers.price_european`.
    cfg
        Grid, time-stepping and `payoff_projection` settings. The measured
        recommendation is `strike_alignment="midpoint"` plus
        `time_stepping="rannacher"`; `payoff_projection="cell_average"` is the
        substitute for alignment on an unaligned grid and is provably a no-op
        on an aligned one (module docstring).

    Returns
    -------
    PriceResult
        With `meta` reporting the grid, the time stepping and the projection.

    Raises
    ------
    InvalidInputError
        As the European engine: a bad `cfg`, or a strike outside the grid.
    """
    _validate(cfg)

    t = option.expiry
    if t == 0.0 or model.sigma == 0.0:
        meta: dict[str, Any] = {
            "method": "pde",
            "model": "BlackScholes",
            "instrument": "digital",
        }
        return PriceResult(value=float(_degenerate_value(option, market)), meta=meta)

    sol = _solve_digital_grid(option, model, market, cfg)
    sol.meta["instrument"] = "digital"
    return PriceResult(value=sol.price, meta=sol.meta)


def greeks_digital(
    option: DigitalOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: PDEConfig,
) -> GreeksResult:
    """Digital Greeks off the grid, by the European engine's estimators.

    Delta and gamma are second-order central stencils on the finished grid,
    theta is the PDE identity at the spot, and vega and rho are
    bump-and-revalue on the same grid -- the same five sources the vanilla
    engine documents, reading a grid whose terminal data was a step function.

    What to expect, measured in `tests/test_digital_pde.py`: with midpoint
    alignment and Rannacher start-up, price and delta are second order; gamma
    is **not**, and its order depends on where it is read, because a digital's
    gamma changes sign at `d1 = 0` and passes through zero near the money. The
    engine reports it; the tests say what it is worth.

    Raises
    ------
    InvalidInputError
        With `greeks_method="grid"`, if `T = 0` or `sigma = 0`.
    """
    _validate(cfg)
    if cfg.greeks_method == "bump":
        return _greeks_by_bump(option, model, market, cfg, price_fn=price_digital)
    return _greeks_from_grid(option, model, market, cfg, solve=_solve_digital_grid)
