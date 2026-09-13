"""Cash-or-nothing digitals on a recombining binomial lattice.

Everything below the terminal condition is shared with the vanilla engine:
the same `lattice_parameters`, the same vectorised backward induction
(`_backward_induction`, which now takes the terminal payoff as a callable),
and the same lattice Greek estimators. Only the terminal values differ, and
that one difference is the whole story of this module.

Why a digital behaves differently on a lattice
----------------------------------------------
A vanilla payoff is Lipschitz: if the strike sits a fraction `x` of a node
spacing away from the nearest terminal node, the payoff values at the
neighbouring nodes are wrong by `O(x * ds)` and the resulting price error is
`O(1 / n)` with a coefficient that depends on `x`. That dependence is the
familiar CRR odd/even oscillation.

A digital payoff is not Lipschitz. The terminal value at a node is `cash` or
`0` with nothing in between, so the *only* thing the terminal condition
records about the strike is which side of it each node fell on. Moving the
strike across a node changes the discrete payoff by a whole `cash` at that
node, and the price by roughly `cash` times the risk-neutral probability of
landing there -- which is `O(1 / sqrt(n))` per node, spread over the `O(sqrt(n))`
nodes near the strike. The error therefore still decays like `1 / n`, but its
coefficient swings over the full range as `n` increments and the strike's
position between nodes drifts. The oscillation is not a parity effect that
averaging over odd and even `n` removes; it is driven by the fractional part
of a quantity that varies continuously with `n`.

Leisen-Reimer does **not** fix this, and the reason is worth stating because
it is easy to assume the opposite. The Leisen-Reimer construction chooses the
per-step probabilities so that the binomial tail `P(Bin(n, p) > n/2)` matches
`N(d2)` -- which is exactly the digital's price divided by `cash e^{-rT}`.
That looks like it should make the digital *exact*. It does not, because
matching the tail is not the same as pricing the tail: the tree's digital
price is the discounted probability of the terminal nodes that are strictly
above `K`, and that set is determined by where `K` falls among the node spots
`S u^j d^{n-j}`, not by the probability `p` the construction chose. The tail
identity binds the two only when the strike lands exactly at the median node.
So the question is empirical, and `tests/test_digital_tree_convergence.py`
answers it by measurement rather than by assumption.

References for the two schemes are in `qpl.engines.tree.lattice`; the digital
payoff and its closed form are in `qpl.engines.analytic.digital`.
"""

from __future__ import annotations

import math

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.options import DigitalOption
from ...instruments.payoffs import digital_payoff
from ...market.curves import FlatDividendCurve, FlatRateCurve
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult
from .lattice import crr_spot_level, lattice_parameters
from .pricers import (
    RHO_BUMP,
    VEGA_BUMP,
    TreeConfig,
    _backward_induction,
    _meta,
    _validate,
    lattice_delta_gamma_theta,
)

__all__ = ["greeks_digital", "price_digital"]


def _terminal(option: DigitalOption):
    """The digital's terminal payoff as a function of the node spots."""

    def payoff(spots: np.ndarray) -> np.ndarray:
        return np.asarray(
            digital_payoff(spots, option.strike, option.cash, option.kind), dtype=float
        )

    return payoff


def _degenerate_value(option: DigitalOption, market: Market) -> float:
    """Price at `T = 0` or at `sigma = 0`, matching the analytic engine.

    At `T = 0` the price is the payoff; at `sigma = 0` the terminal spot is its
    forward with certainty, so the digital is a zero-coupon bond paying `cash`
    exactly when that forward is in the money.
    """
    t = option.expiry
    if t == 0.0:
        return float(
            digital_payoff(market.spot, option.strike, option.cash, option.kind)
        )

    forward = market.spot * math.exp(
        (market.rate(t) - market.dividend_yield(t)) * t
    )
    return market.df_r(t) * float(
        digital_payoff(forward, option.strike, option.cash, option.kind)
    )


def _lattice(option: DigitalOption, model: BlackScholesModel, market: Market, cfg: TreeConfig):
    t = option.expiry
    return lattice_parameters(
        scheme=cfg.scheme,
        spot=market.spot,
        strike=option.strike,
        sigma=model.sigma,
        expiry=t,
        rate=market.rate(t),
        dividend_yield=market.dividend_yield(t),
        n_steps=cfg.n_steps,
    )


def price_digital(
    option: DigitalOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: TreeConfig,
) -> PriceResult:
    """Price a cash-or-nothing digital by backward induction on a binomial tree.

    The terminal condition is `cash * 1{S > K}` (call) or `cash * 1{S < K}`
    (put) evaluated at the terminal node spots, with the strict inequalities of
    `qpl.instruments.payoffs.digital_payoff`; a terminal node landing *exactly*
    on the strike therefore pays nothing. That is the convention QuantLib's
    `CashOrNothingPayoff` uses, and on a CRR lattice at the money with an even
    `n_steps` it is not a hypothetical: the centre node is the spot, so it sits
    exactly on the strike.

    Accuracy is measured, not assumed; see the module docstring and
    `tests/test_digital_tree_convergence.py`.

    Raises
    ------
    InvalidInputError
        As `qpl.engines.tree.price_european`: a bad `cfg`, or a lattice that
        violates no-arbitrage.
    """
    _validate(cfg, min_steps=1)

    t = option.expiry
    if t == 0.0 or model.sigma == 0.0:
        value = _degenerate_value(option, market)
        degenerate = "expiry" if t == 0.0 else "zero_vol"
        lattice = None if t == 0.0 else _lattice(option, model, market, cfg)
        meta = _meta(lattice, cfg, degenerate=degenerate)
        meta["instrument"] = "digital"
        return PriceResult(value=float(value), meta=meta)

    lattice = _lattice(option, model, market, cfg)
    kept = _backward_induction(
        option, market, lattice, capture=(0,), payoff=_terminal(option)
    )
    meta = _meta(lattice, cfg, degenerate=None)
    meta["instrument"] = "digital"
    return PriceResult(value=float(kept[0][0]), meta=meta)


def greeks_digital(
    option: DigitalOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: TreeConfig,
) -> GreeksResult:
    """Digital Greeks from the lattice, by the same estimators the vanilla uses.

    Delta, gamma and theta are read off levels 1 and 2 of the finished lattice
    (`lattice_delta_gamma_theta`); vega and rho are central bump-and-revalue at
    fixed `n_steps` with `VEGA_BUMP` and `RHO_BUMP`.

    These estimators were derived for a smooth value function and are first
    order even on a vanilla, where they carry an `O(dt)` bias from reading the
    slope one and two steps away from the root. On a digital they inherit, on
    top of that, the price's oscillation amplified by the node spacing: delta
    divides a difference of two node values by `S(1,1) - S(1,0)`, which is
    `O(sqrt(dt))`, so an oscillation of size `eps` in the prices becomes
    `eps / sqrt(dt)` in delta and `eps / dt` in gamma. They are reported
    because the estimators exist and are shared, and they are *measured* in
    `tests/test_digital_tree_convergence.py` rather than claimed.

    Raises
    ------
    InvalidInputError
        If `cfg.n_steps < 2` (gamma and theta read step-2 nodes) or
        `sigma = 0`.
    """
    _validate(cfg, min_steps=2)

    t = option.expiry
    if t == 0.0:
        meta = _meta(None, cfg, degenerate="expiry")
        meta["instrument"] = "digital"
        meta["bumps"] = {"sigma": VEGA_BUMP, "r": RHO_BUMP}
        return GreeksResult(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0, meta=meta)

    if model.sigma == 0.0:
        raise InvalidInputError("sigma must be > 0 for tree Greeks")

    r = market.rate(t)
    q = market.dividend_yield(t)
    lattice = _lattice(option, model, market, cfg)
    kept = _backward_induction(
        option, market, lattice, capture=(0, 1, 2), payoff=_terminal(option)
    )

    s0 = market.spot
    s1 = crr_spot_level(spot=s0, up=lattice.up, down=lattice.down, level=1)
    s2 = crr_spot_level(spot=s0, up=lattice.up, down=lattice.down, level=2)

    delta, gamma, theta = lattice_delta_gamma_theta(
        v0=kept[0][0],
        v1=kept[1],
        v2=kept[2],
        s1=s1,
        s2=s2,
        dt=lattice.dt,
        spot=None if lattice.spot_centred else s0,
    )

    def _price(mdl: BlackScholesModel, mkt: Market) -> float:
        return price_digital(option, mdl, mkt, cfg=cfg).value

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

    rho = (
        _price(model, _rate_market(r + RHO_BUMP)) - _price(model, _rate_market(r - RHO_BUMP))
    ) / (2.0 * RHO_BUMP)

    meta = _meta(lattice, cfg, degenerate=None)
    meta["instrument"] = "digital"
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
