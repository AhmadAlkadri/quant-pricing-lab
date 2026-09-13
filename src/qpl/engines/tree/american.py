"""American option pricing on a recombining binomial lattice.

The lattice is the one in `qpl.engines.tree.lattice`, shared with the European
pricer, and either parameterisation it offers (`TreeConfig.scheme`, "crr" or
"leisen-reimer") works here unchanged: nothing below reads anything but
``up``, ``down``, ``p`` and ``discount``. The only difference from the
European pricer is the step. Where the European engine rolls back
the discounted expectation, the American engine takes

    V(t_j, S) = max( intrinsic(S),  e^{-r dt} E[ V(t_{j+1}, S') | S ] )

which is the Bellman equation of a finite-horizon optimal stopping problem.
The arithmetic per level is one numpy expression, exactly as in the European
pricer, with one extra elementwise maximum against the intrinsic value.

This module replaces the keyword-based `qpl.engines.dp.price_american_put_binomial`
of Slice 1. The value arithmetic is unchanged expression by expression -- the
continuation is still ``disc * (p * V[1:] + (1 - p) * V[:-1])`` and the Bellman
step is still ``np.maximum(intrinsic, continuation)`` -- so prices are
bit-identical to that engine's; `tests/test_tree_american.py` pins five of them
with a zero tolerance.

What is new here beyond dispatching on `AmericanOption`:

- calls as well as puts, and a continuous dividend yield read from `Market`;
- the early-exercise boundary, extracted per time level;
- exact degenerate limits at ``T = 0`` and ``sigma = 0`` (see
  `_degenerate_value`), rather than whatever a collapsed lattice happens to
  return at the caller's `n_steps`.

Derivation, measured convergence tables and the QuantLib comparisons:
`docs/notes/american_exercise_on_trees.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.options import AmericanOption
from ...market.curves import FlatDividendCurve, FlatRateCurve
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult
from .lattice import BinomialLattice, crr_spot_level, lattice_parameters
from .pricers import (
    RHO_BUMP,
    VEGA_BUMP,
    TreeConfig,
    _meta,
    _payoff,
    _validate,
    lattice_delta_gamma_theta,
)

__all__ = ["greeks_american", "price_american"]

_INTRINSIC_FLOOR_REL = 1e-12
"""Relative floor below which a node is not counted as in the money.

A lattice node that should sit exactly on the strike is computed as
``S0 * u**k * d**(k)`` and lands on `K` only up to round-off, so a bare
``intrinsic > 0`` test would make the reported boundary depend on the last bit
of a power: for a call the at-the-strike node would be "in the money" by 1e-13
and for a put it would not. The floor is scaled by the strike so it means the
same thing at ``K = 0.01`` and ``K = 10_000``; being wrong by one node at the
strike is inside the node spacing the boundary is reported to anyway.
"""


@dataclass(frozen=True)
class _Rollback:
    """What one backward induction sweep produced."""

    value: float
    levels: dict[int, np.ndarray]
    boundary: np.ndarray
    early_exercise_node_count: int


def _spot_levels(market: Market, lattice: BinomialLattice):
    """Return a callable giving the spot nodes at one time level.

    Precomputing ``u**i`` and ``d**i`` once turns the whole lattice into `O(n)`
    calls to `pow` instead of `O(n**2)`, which is what makes an `n = 8001`
    reference affordable. Element `i` of level `j` is still
    ``spot * u**i * d**(j - i)``, evaluated in that order, so the nodes are the
    same floating-point numbers `qpl.engines.tree.lattice.crr_spot_level`
    produces.
    """
    n = lattice.n_steps
    counts = np.arange(n + 1, dtype=float)
    up_powers = lattice.up**counts
    down_powers = lattice.down**counts
    spot = market.spot

    def level(j: int) -> np.ndarray:
        return spot * up_powers[: j + 1] * down_powers[j::-1]

    return level


def _boundary_node(spots: np.ndarray, mask: np.ndarray, *, kind: str) -> float:
    """The exercise boundary at one time level, or NaN if nobody exercises.

    For a put the exercise region is ``S <= B(t)``, so the boundary is the
    *largest* node at which exercising is optimal; for a call it is ``S >=
    B(t)`` and therefore the *smallest*.
    """
    where = np.flatnonzero(mask)
    if where.size == 0:
        return math.nan
    return float(spots[where[-1] if kind == "put" else where[0]])


def _rollback(
    option: AmericanOption,
    market: Market,
    lattice: BinomialLattice,
    *,
    capture: tuple[int, ...],
) -> _Rollback:
    """Bellman backward induction, keeping the named levels and the boundary.

    A node is recorded as an early-exercise node when exercising is at least as
    good as continuing *and* the option is in the money there by more than
    `_INTRINSIC_FLOOR_REL * strike`. The second condition is what makes the
    boundary meaningful: deep out of the money both sides of the comparison are
    zero (or underflow to it), and "exercise is weakly optimal" there is a
    statement about ``0 >= 0``, not about exercising.
    """
    n = lattice.n_steps
    kind, strike = option.kind, option.strike
    floor = _INTRINSIC_FLOOR_REL * strike
    spots_at = _spot_levels(market, lattice)

    spots = spots_at(n)
    values = _payoff(spots, kind=kind, strike=strike)

    boundary = np.full(n + 1, math.nan)
    # At expiry there is no continuation to compare against: every in-the-money
    # node is exercised, so the boundary is the in-the-money node nearest the
    # strike.
    boundary[n] = _boundary_node(spots, values > floor, kind=kind)

    levels: dict[int, np.ndarray] = {}
    if n in capture:
        levels[n] = values.copy()

    disc, p = lattice.discount, lattice.p
    early_count = 0
    for level in range(n - 1, -1, -1):
        continuation = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        spots = spots_at(level)
        intrinsic = _payoff(spots, kind=kind, strike=strike)
        values = np.maximum(intrinsic, continuation)

        mask = (intrinsic > floor) & (intrinsic >= continuation)
        early_count += int(np.count_nonzero(mask))
        boundary[level] = _boundary_node(spots, mask, kind=kind)

        if level in capture:
            levels[level] = values.copy()

    return _Rollback(
        value=float(values[0]),
        levels=levels,
        boundary=boundary,
        early_exercise_node_count=early_count,
    )


def _deterministic_exercise_times(option: AmericanOption, market: Market) -> tuple[float, ...]:
    """Candidate optimal exercise times when `sigma = 0`.

    With `sigma = 0` the spot is deterministic, ``S(t) = S0 e^{(r-q) t}``, so
    the American value is a one-dimensional maximisation over the exercise
    date:

        V = max_{0 <= t <= T} e^{-r t} * intrinsic(S(t))
          = max( 0, sup_{0 <= t <= T} h(t) ),
        h_put(t)  = K e^{-r t} - S0 e^{-q t},
        h_call(t) = S0 e^{-q t} - K e^{-r t}.

    `h` is a difference of two exponentials, so ``h'(t) = 0`` has at most one
    root: `h` is either monotone on `[0, T]` or has a single interior turning
    point. Its location is

        t* = log(r K / (q S0)) / (r - q),

    and differentiating twice at that point gives
    ``h_put''(t*) = (r - q) r K e^{-r t*}``: for a put the turning point is a
    *minimum* when ``r > q`` and a maximum when ``r < q`` (and the reverse for
    a call). The supremum is therefore attained in
    ``{0, T} union {t* if 0 < t* < T}``, which is what this function returns.

    Consequence worth stating plainly, because it is easy to get wrong: when
    ``r >= q`` -- which covers every no-dividend case -- the put's supremum is
    at an endpoint and the value collapses to
    ``max(intrinsic now, discounted forward intrinsic)``. When ``q > r`` it
    does not: there are specifications (deep in the money, large `q`, long
    `T`) whose zero-volatility American put is worth strictly more than either
    endpoint. `tests/test_tree_american.py` pins one.
    """
    t = option.expiry
    candidates = [0.0, t]

    r = market.rate(t)
    q = market.dividend_yield(t)
    if r > 0.0 and q > 0.0 and r != q:
        ratio = (r * option.strike) / (q * market.spot)
        if ratio > 0.0:
            turning = math.log(ratio) / (r - q)
            if 0.0 < turning < t:
                candidates.append(turning)
    return tuple(candidates)


def _degenerate_value(option: AmericanOption, market: Market) -> float:
    """Price at ``T = 0`` or at ``sigma = 0``.

    At ``T = 0`` the price is the payoff, as in every other engine. At
    ``sigma = 0`` it is the best discounted intrinsic value over the
    deterministic forward path; see `_deterministic_exercise_times` for why the
    candidate set below is exhaustive.
    """
    s0, k, t = market.spot, option.strike, option.expiry
    if t == 0.0:
        return max(s0 - k, 0.0) if option.kind == "call" else max(k - s0, 0.0)

    r = market.rate(t)
    q = market.dividend_yield(t)
    best = 0.0
    for time in _deterministic_exercise_times(option, market):
        forward = s0 * math.exp((r - q) * time)
        disc = math.exp(-r * time)
        payoff = (
            max(forward - k, 0.0) if option.kind == "call" else max(k - forward, 0.0)
        )
        best = max(best, disc * payoff)
    return best


def price_american(
    option: AmericanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: TreeConfig,
) -> PriceResult:
    """Price an American option by backward induction on a CRR binomial tree.

    Parameters
    ----------
    option
        American option (`call` or `put`).
    model
        Black-Scholes model with constant volatility.
    market
        Market object providing spot and flat rate/dividend curves. The rate
        and dividend yield are sampled once, at the option's expiry, and held
        constant across the lattice.
    cfg
        Lattice settings.

    Returns
    -------
    PriceResult
        Price, with metadata reporting the method, scheme, `n_steps`, the
        realised lattice parameters `u`, `d`, `p`, `dt`, and two early-exercise
        diagnostics:

        - ``exercise_boundary``: a read-only array of length ``n_steps + 1``
          holding, for each time level, the spot node at the edge of the
          exercise region -- the largest such node for a put, the smallest for
          a call -- or `NaN` at a level where exercising is never optimal.
          `None` in the degenerate cases below, where there is no lattice to
          read a boundary from.
        - ``early_exercise_node_count``: how many nodes strictly before expiry
          are early-exercise nodes.

    Notes
    -----
    Degenerate inputs are handled in closed form rather than on a collapsed
    lattice: at ``T = 0`` the price is the payoff, and at ``sigma = 0`` it is
    the best discounted intrinsic value along the deterministic forward path
    (`_deterministic_exercise_times`). Both are exact, whereas a lattice would
    only see the exercise dates its own `n_steps` happens to place.

    Raises
    ------
    InvalidInputError
        If `cfg` is out of range, or if the lattice violates the no-arbitrage
        condition (see `qpl.engines.tree.lattice.lattice_parameters`).
    """
    _validate(cfg, min_steps=1)

    t = option.expiry
    if t == 0.0 or model.sigma == 0.0:
        value = _degenerate_value(option, market)
        lattice = (
            None
            if t == 0.0
            else lattice_parameters(
                scheme=cfg.scheme,
                spot=market.spot,
                strike=option.strike,
                sigma=0.0,
                expiry=t,
                rate=market.rate(t),
                dividend_yield=market.dividend_yield(t),
                n_steps=cfg.n_steps,
            )
        )
        meta = _meta(lattice, cfg, degenerate="expiry" if t == 0.0 else "zero_vol")
        meta.update(
            {
                "exercise": "american",
                "exercise_boundary": None,
                "early_exercise_node_count": 0,
            }
        )
        return PriceResult(value=float(value), meta=meta)

    lattice = lattice_parameters(
        scheme=cfg.scheme,
        spot=market.spot,
        strike=option.strike,
        sigma=model.sigma,
        expiry=t,
        rate=market.rate(t),
        dividend_yield=market.dividend_yield(t),
        n_steps=cfg.n_steps,
    )
    rolled = _rollback(option, market, lattice, capture=())

    boundary = rolled.boundary
    boundary.setflags(write=False)
    meta = _meta(lattice, cfg, degenerate=None)
    meta.update(
        {
            "exercise": "american",
            "exercise_boundary": boundary,
            "early_exercise_node_count": rolled.early_exercise_node_count,
        }
    )
    return PriceResult(value=rolled.value, meta=meta)


def greeks_american(
    option: AmericanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: TreeConfig,
) -> GreeksResult:
    """Greeks for an American option from the CRR lattice.

    The estimators are exactly the European ones -- `lattice_delta_gamma_theta`
    for delta, gamma and theta off the step-1 and step-2 nodes, and central
    bump-and-revalue at a fixed `n_steps` for vega and rho -- because they
    depend only on the node values and node spots, not on how the values were
    produced. Early exercise changes the values; it does not change how a slope
    is read off two of them. The derivations are in
    `qpl.engines.tree.pricers.greeks_european`. On a Leisen-Reimer lattice the
    theta estimator takes its off-centre correction, for the reason set out in
    `lattice_delta_gamma_theta`; delta and gamma need none, being written in
    terms of the node spots already.

    Two things are worth stating about accuracy, since there is no closed form
    to compare an American Greek against:

    - Delta, gamma and theta inherit the price's `O(1 / n_steps)` error, as in
      the European case, plus the error in the location of the exercise
      boundary. `tests/test_tree_american_convergence.py` measures the order
      against a fine-`n` reference from this same engine.
    - Gamma is the worst-behaved of the three. The American value function has
      a genuine kink in `S` at the exercise boundary (value matching holds,
      smooth pasting only in the continuum limit), and a second difference
      taken across nodes near that kink is a second difference of a function
      whose second derivative is a delta function in the limit. Away from the
      boundary -- which is where the step-2 nodes sit for a
      not-deep-in-the-money option -- it is well behaved.

    Parameters
    ----------
    option, model, market
        As for `price_american`.
    cfg
        Lattice settings. `n_steps >= 2` is required, since gamma and theta
        read step-2 nodes.

    Returns
    -------
    GreeksResult
        Delta, gamma, vega, theta, rho, plus metadata reporting the bump sizes
        and the lattice parameters.

    Raises
    ------
    InvalidInputError
        If `cfg` is out of range, or if `sigma = 0` (a collapsed lattice has no
        second spot node to difference).
    """
    _validate(cfg, min_steps=2)

    t = option.expiry
    if t == 0.0:
        meta = _meta(None, cfg, degenerate="expiry")
        meta["exercise"] = "american"
        meta["bumps"] = {"sigma": VEGA_BUMP, "r": RHO_BUMP}
        return GreeksResult(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0, meta=meta)

    if model.sigma == 0.0:
        raise InvalidInputError("sigma must be > 0 for tree Greeks")

    r = market.rate(t)
    q = market.dividend_yield(t)
    lattice = lattice_parameters(
        scheme=cfg.scheme,
        spot=market.spot,
        strike=option.strike,
        sigma=model.sigma,
        expiry=t,
        rate=r,
        dividend_yield=q,
        n_steps=cfg.n_steps,
    )
    rolled = _rollback(option, market, lattice, capture=(0, 1, 2))

    s0 = market.spot
    s1 = crr_spot_level(spot=s0, up=lattice.up, down=lattice.down, level=1)
    s2 = crr_spot_level(spot=s0, up=lattice.up, down=lattice.down, level=2)
    delta, gamma, theta = lattice_delta_gamma_theta(
        v0=float(rolled.levels[0][0]),
        v1=rolled.levels[1],
        v2=rolled.levels[2],
        s1=s1,
        s2=s2,
        dt=lattice.dt,
        spot=None if lattice.spot_centred else s0,
    )

    def _price(mdl: BlackScholesModel, mkt: Market) -> float:
        return price_american(option, mdl, mkt, cfg=cfg).value

    sigma_dn = max(model.sigma - VEGA_BUMP, 0.0)
    vega = (
        _price(BlackScholesModel(sigma=model.sigma + VEGA_BUMP), market)
        - _price(BlackScholesModel(sigma=sigma_dn), market)
    ) / ((model.sigma + VEGA_BUMP) - sigma_dn)

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
    meta["exercise"] = "american"
    meta["fd"] = "central"
    meta["bumps"] = {"sigma": VEGA_BUMP, "r": RHO_BUMP}
    meta["early_exercise_node_count"] = rolled.early_exercise_node_count
    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=float(vega),
        theta=theta,
        rho=float(rho),
        meta=meta,
    )
