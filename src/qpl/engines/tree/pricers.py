"""European option pricing and Greeks on a Cox-Ross-Rubinstein binomial tree.

The lattice itself lives in `qpl.engines.tree.lattice`; this module adds the
terminal payoff, the backward induction, and the lattice Greek estimators.

Assumptions
-----------
- Flat curves: the rate and dividend yield are read from `Market` once, at the
  option's expiry, and held constant over every step of the tree.
- Black-Scholes dynamics: the lattice is the CRR discretisation of geometric
  Brownian motion with constant volatility.

Accuracy
--------
The scheme is first order in ``1 / n`` with a coefficient that oscillates with
the parity of ``n``, because the strike's position between the two terminal
nodes that straddle it changes as ``n`` increments. See
`docs/notes/crr_tree_convergence.md` and Leisen and Reimer (1996), "Binomial
models for option valuation -- examining and improving convergence", Applied
Mathematical Finance 3(4), 319-346, which both establishes the order-1
behaviour and constructs the tree that removes the oscillation (a later
slice; not implemented here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.options import EuropeanOption
from ...market.curves import FlatDividendCurve, FlatRateCurve
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult
from ..registry import MethodSpec
from .lattice import CRRLattice, crr_parameters, crr_spot_level

__all__ = [
    "TREE_METHOD_SPEC",
    "TreeConfig",
    "greeks_european",
    "lattice_delta_gamma_theta",
    "price_european",
]

VEGA_BUMP = 1e-2
"""Absolute volatility bump for the tree's vega (one volatility point).

Vega on a lattice is limited by the tree's oscillating price error, not by the
bump: shifting `sigma` moves every node, so the oscillation does not cancel
between the two evaluations and whatever survives is divided by `2h`.

Measured over `n` in `{1000..1029}` and `{2000..2029}` on five specification
points, the worst absolute residual against the closed-form vega is 0.75 at
`h = 0.002`, 0.28 at `h = 0.01`, and 0.074 at `h = 0.02`; but by `h = 0.02`
the `O(h**2)` bias of the central difference is already visible as a residual
that stops shrinking with `n`. `h = 0.01` is the compromise: large enough that
the amplified oscillation stays under about 1% of vega, small enough that the
bias is not the binding term.
"""

RHO_BUMP = 1e-4
"""Absolute rate bump for the tree's rho, in rate units.

Rho is far better behaved than vega: the rate enters only through `p` and the
discount factor, not through `u` and `d`, so the lattice geometry -- and hence
the position of the strike among the terminal nodes -- does not move when `r`
is bumped. The measured worst relative residual is 3.6e-04 and is essentially
the same for every bump from 1e-05 to 1e-02, so the bump is chosen small
enough for the truncation term to be negligible.
"""


@dataclass(frozen=True)
class TreeConfig:
    """Binomial lattice configuration.

    Parameters
    ----------
    n_steps
        Number of time steps in the lattice. Price error decays like
        ``1 / n_steps`` with an oscillating coefficient; Greeks read off the
        lattice inherit that oscillation.
    scheme
        Lattice parameterisation. Only ``"crr"`` exists so far; the field is
        present because Leisen-Reimer is a planned sibling and adding it must
        not change this config's identity.
    """

    n_steps: int = 200
    scheme: Literal["crr"] = "crr"


TREE_METHOD_SPEC = MethodSpec(method="tree", cfg_type=TreeConfig)
"""Keyword contract for `method="tree"`: a required `TreeConfig`, nothing else."""


def _validate(cfg: TreeConfig, *, min_steps: int) -> None:
    if cfg.scheme != "crr":
        raise InvalidInputError("scheme must be 'crr'")
    if cfg.n_steps < min_steps:
        raise InvalidInputError(f"n_steps must be >= {min_steps}")


def _payoff(spots: np.ndarray, *, kind: str, strike: float) -> np.ndarray:
    if kind == "call":
        return np.maximum(spots - strike, 0.0)
    return np.maximum(strike - spots, 0.0)


def _degenerate_value(option: EuropeanOption, market: Market) -> float:
    """Price at ``T = 0`` or at ``sigma = 0``.

    Matches the analytic, MC and PDE engines: intrinsic value at expiry, and
    the discounted intrinsic value of the forward when the spot is
    deterministic.
    """
    s0 = market.spot
    k = option.strike
    t = option.expiry

    if t == 0.0:
        if option.kind == "call":
            return max(s0 - k, 0.0)
        return max(k - s0, 0.0)

    r = market.rate(t)
    q = market.dividend_yield(t)
    forward = s0 * math.exp((r - q) * t)
    disc = market.df_r(t)
    if option.kind == "call":
        return disc * max(forward - k, 0.0)
    return disc * max(k - forward, 0.0)


def _meta(
    lattice: CRRLattice | None, cfg: TreeConfig, *, degenerate: str | None
) -> dict[str, object]:
    meta: dict[str, object] = {
        "method": "tree",
        "model": "BlackScholes",
        "scheme": cfg.scheme,
        "n_steps": cfg.n_steps,
    }
    if lattice is None:
        meta.update({"dt": 0.0, "u": 1.0, "d": 1.0, "p": 0.5})
    else:
        meta.update(
            {
                "dt": lattice.dt,
                "u": lattice.up,
                "d": lattice.down,
                "p": lattice.p,
            }
        )
    if degenerate is not None:
        meta["degenerate"] = degenerate
    return meta


def _backward_induction(
    option: EuropeanOption,
    market: Market,
    lattice: CRRLattice,
    *,
    capture: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Roll the terminal payoff back to the root, keeping the named levels.

    The payoff vector at one time level is updated in a single numpy
    expression, so the only Python-level loop is over time levels:
    ``V_j = e^{-r dt} (p V_{j+1}[1:] + (1 - p) V_{j+1}[:-1])``, where index
    ``j`` counts up moves.
    """
    n = lattice.n_steps
    values = _payoff(
        crr_spot_level(spot=market.spot, up=lattice.up, down=lattice.down, level=n),
        kind=option.kind,
        strike=option.strike,
    )

    kept: dict[int, np.ndarray] = {}
    if n in capture:
        kept[n] = values.copy()

    disc, p = lattice.discount, lattice.p
    for level in range(n - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if level in capture:
            kept[level] = values.copy()

    return kept


def lattice_delta_gamma_theta(
    *,
    v0: float,
    v1: np.ndarray,
    v2: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    dt: float,
) -> tuple[float, float, float]:
    """Delta, gamma and theta read off the first two levels of a lattice.

    Shared by the European and American tree engines: the estimators depend
    only on the node values and the node spots, not on how those values were
    produced, so early exercise changes the inputs and nothing else. The
    derivation of each formula is in `greeks_european`'s docstring.
    """
    delta = (v1[1] - v1[0]) / (s1[1] - s1[0])
    delta_up = (v2[2] - v2[1]) / (s2[2] - s2[1])
    delta_dn = (v2[1] - v2[0]) / (s2[1] - s2[0])
    gamma = (delta_up - delta_dn) / (0.5 * (s2[2] - s2[0]))
    theta = (v2[1] - v0) / (2.0 * dt)
    return float(delta), float(gamma), float(theta)


def price_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: TreeConfig,
) -> PriceResult:
    """Price a European option by backward induction on a CRR binomial tree.

    Parameters
    ----------
    option
        European option (`call` or `put`).
    model
        Black-Scholes model with constant volatility.
    market
        Market object providing spot and flat rate/dividend curves. The rate
        and dividend yield are sampled once, at the option's expiry.
    cfg
        Lattice settings.

    Returns
    -------
    PriceResult
        Price, with metadata reporting the method, scheme, `n_steps`, and the
        realised lattice parameters `u`, `d`, `p`, `dt`.

    Raises
    ------
    InvalidInputError
        If `cfg` is out of range, or if the lattice violates the no-arbitrage
        condition (see `qpl.engines.tree.lattice.crr_parameters`).
    """
    _validate(cfg, min_steps=1)

    t = option.expiry
    if t == 0.0 or model.sigma == 0.0:
        value = _degenerate_value(option, market)
        degenerate = "expiry" if t == 0.0 else "zero_vol"
        lattice = (
            None
            if t == 0.0
            else crr_parameters(
                sigma=0.0,
                expiry=t,
                rate=market.rate(t),
                dividend_yield=market.dividend_yield(t),
                n_steps=cfg.n_steps,
            )
        )
        return PriceResult(value=float(value), meta=_meta(lattice, cfg, degenerate=degenerate))

    lattice = crr_parameters(
        sigma=model.sigma,
        expiry=t,
        rate=market.rate(t),
        dividend_yield=market.dividend_yield(t),
        n_steps=cfg.n_steps,
    )
    kept = _backward_induction(option, market, lattice, capture=(0,))
    return PriceResult(value=float(kept[0][0]), meta=_meta(lattice, cfg, degenerate=None))


def greeks_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: TreeConfig,
) -> GreeksResult:
    """Greeks for a European option from the CRR lattice.

    Delta, gamma and theta are read off nodes the tree has already computed,
    so they cost nothing beyond the price. Vega and rho have no such
    representation on the lattice and are obtained by re-pricing on bumped
    inputs.

    Estimators
    ----------
    Write ``S(j, i)`` and ``V(j, i)`` for the spot and option value at time
    level ``j`` after ``i`` up moves. The standard lattice estimators, in the
    form used here (the idea is the usual one; see the binomial-tree chapter
    of Hull, *Options, Futures, and Other Derivatives*):

    - **Delta** is the slope across the two step-1 nodes,
      ``(V(1,1) - V(1,0)) / (S(1,1) - S(1,0))``. Both are one step from the
      root, so this is a centred difference in spot evaluated at time ``dt``
      rather than at ``0``; the resulting ``O(dt)`` bias is the same order as
      the price error itself.
    - **Gamma** differences two such slopes at step 2. Because ``d = 1/u`` the
      middle node ``S(2,1)`` equals the initial spot, so the two slopes
      ``(V(2,2) - V(2,1)) / (S(2,2) - S(2,1))`` and
      ``(V(2,1) - V(2,0)) / (S(2,1) - S(2,0))`` sit just above and just below
      the spot. Dividing their difference by the half-width
      ``(S(2,2) - S(2,0)) / 2`` gives the second derivative.
    - **Theta** uses that same middle node: ``S(2,1) = S(0,0)``, so
      ``(V(2,1) - V(0,0)) / (2 dt)`` compares the value at the *same* spot two
      steps later, which is exactly a forward difference in calendar time.
    - **Vega** and **rho** are central bump-and-revalue at fixed ``n_steps``:
      ``(V(sigma + h) - V(sigma - h)) / (2h)`` with ``h = VEGA_BUMP``, and the
      same in the rate with ``h = RHO_BUMP``. Bumping ``sigma`` moves ``u``
      and ``d``, hence the whole lattice, so the tree's oscillating
      discretisation error does *not* cancel between the two evaluations:
      vega carries that oscillation amplified by ``1 / (2h)`` and is
      correspondingly less accurate than delta and gamma. Bumping ``r`` leaves
      the lattice geometry alone and rho is much better behaved. See the
      module constants for the measurements behind the two bump sizes, and
      `tests/test_tree_convergence.py` for the tolerances they justify.

    Parameters
    ----------
    option, model, market
        As for `price_european`.
    cfg
        Lattice settings. `n_steps >= 2` is required, since gamma and theta
        read step-2 nodes.

    Returns
    -------
    GreeksResult
        Delta, gamma, vega, theta, rho, plus metadata reporting the bump sizes
        and the lattice parameters.
    """
    _validate(cfg, min_steps=2)

    t = option.expiry
    if t == 0.0:
        meta = _meta(None, cfg, degenerate="expiry")
        meta["bumps"] = {"sigma": VEGA_BUMP, "r": RHO_BUMP}
        return GreeksResult(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0, meta=meta)

    r = market.rate(t)
    q = market.dividend_yield(t)

    if model.sigma == 0.0:
        raise InvalidInputError("sigma must be > 0 for tree Greeks")

    lattice = crr_parameters(
        sigma=model.sigma, expiry=t, rate=r, dividend_yield=q, n_steps=cfg.n_steps
    )
    kept = _backward_induction(option, market, lattice, capture=(0, 1, 2))

    s0 = market.spot
    s1 = crr_spot_level(spot=s0, up=lattice.up, down=lattice.down, level=1)
    s2 = crr_spot_level(spot=s0, up=lattice.up, down=lattice.down, level=2)
    v0, v1, v2 = kept[0][0], kept[1], kept[2]

    delta, gamma, theta = lattice_delta_gamma_theta(
        v0=v0, v1=v1, v2=v2, s1=s1, s2=s2, dt=lattice.dt
    )

    def _price(mdl: BlackScholesModel, mkt: Market) -> float:
        return price_european(option, mdl, mkt, cfg=cfg).value

    sigma_up = BlackScholesModel(sigma=model.sigma + VEGA_BUMP)
    sigma_dn = BlackScholesModel(sigma=max(model.sigma - VEGA_BUMP, 0.0))
    vega = (_price(sigma_up, market) - _price(sigma_dn, market)) / (
        (model.sigma + VEGA_BUMP) - max(model.sigma - VEGA_BUMP, 0.0)
    )

    def _rate_market(rate: float) -> Market:
        return Market(
            spot=s0,
            rate_curve=FlatRateCurve(rate, allow_negative=True),
            dividend_curve=FlatDividendCurve(q, allow_negative=True),
        )

    rho = (_price(model, _rate_market(r + RHO_BUMP)) - _price(model, _rate_market(r - RHO_BUMP))) / (
        2.0 * RHO_BUMP
    )

    meta = _meta(lattice, cfg, degenerate=None)
    meta["fd"] = "central"
    meta["bumps"] = {"sigma": VEGA_BUMP, "r": RHO_BUMP}
    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=float(vega),
        theta=theta,
        rho=float(rho),
        meta=meta,
    )
