"""Implied volatilities of transform prices: the smile a model actually makes.

This module is thin on purpose. It owns no root finder and no pricer: it calls
`qpl.pricing.price` with `method="fourier"` and hands the answer to
`qpl.engines.analytic.black_scholes.implied_volatility`, which is the Brent
inverter that has been in this package since before the curriculum started.
Everything here is the plumbing between the two, plus two summary statistics
(`atm_implied_variance` and `smile_skew`) that exist because the claims in
`tests/test_heston_smile.py` are about *shape*, and a shape claim needs a
number to be a claim about.

Why an implied volatility at all
--------------------------------
A Heston price is a number, and a number cannot be compared with a
Black-Scholes price at a different strike in any useful way. Quoting the same
price as the Black-Scholes volatility that reproduces it puts every strike and
every maturity on one axis, which is the axis the model's three shape
parameters act on:

- `rho` sets the **sign of the skew** (the slope in log-moneyness): a negative
  correlation makes down moves coincide with volatility up moves, which fattens
  the left tail and lifts low-strike implied vols;
- `xi` sets the **curvature** (the smile's convexity), because it is what makes
  the terminal law non-Gaussian at all;
- `kappa` and `theta` set the **term structure**: the average variance over
  `[0, T]` runs from `v0` at `T = 0` to `theta` as `T -> infinity`, at rate
  `kappa`, and the skew flattens over the same horizon because the variance of
  the averaged variance falls faster than the average itself.

All three are measured, not asserted, in `tests/test_heston_smile.py`.

Source for the qualitative statements: Gatheral (2006), "The Volatility
Surface", chapter 2 (the Heston smile and the term structure of skew). No
formula, number or figure is taken from it; every number this repository
publishes about the Heston smile is measured by this module.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.options import EuropeanOption
from ...market.market import Market
from ..analytic.black_scholes import implied_volatility
from .pricers import FourierConfig

__all__ = [
    "SKEW_LOG_MONEYNESS_BUMP",
    "atm_implied_variance",
    "implied_vol",
    "implied_vol_surface",
    "smile_skew",
]

SKEW_LOG_MONEYNESS_BUMP = 0.01
"""Half-width, in log-moneyness, of the central difference behind `smile_skew`.

The skew is `d sigma_imp / d ln(K / F)` at `K = F`. It is a difference quotient
because the implied volatility is defined by an inversion and has no closed
derivative here; `0.01` is one percent of the forward, small enough that the
`O(h^2)` truncation term is below the inverter's own `1e-07` tolerance on every
maturity tested and large enough that the two inversions differ in their
leading digits."""


def _forward(market: Market, expiry: float) -> float:
    """`S_0 e^{(r - q) T}`, the strike at which the skew is measured."""
    return market.spot * math.exp(
        (market.rate(expiry) - market.dividend_yield(expiry)) * expiry
    )


def implied_vol(
    model: Any,
    market: Market,
    *,
    strike: float,
    expiry: float,
    kind: str = "call",
    cfg: FourierConfig | None = None,
) -> float:
    """Black-Scholes implied volatility of the transform price of one option.

    Parameters
    ----------
    model
        Anything `method="fourier"` is registered for: `HestonModel`, or
        `BlackScholesModel` (where the answer must come back as the model's own
        `sigma`, which is exactly what `tests/test_heston_smile.py` checks
        first).
    cfg
        Transform configuration. `None` means `FourierConfig()` -- the COS
        method at the package defaults.

    Raises
    ------
    InvalidInputError
        Propagated from `implied_volatility` when the price is outside the
        arbitrage bounds, which is how a transform price that has gone wrong
        (a range too narrow, a damping parameter past the moment explosion)
        surfaces here rather than as a plausible-looking volatility.
    """
    from ...pricing import price as price_instrument

    option = EuropeanOption(kind=kind, strike=strike, expiry=expiry)
    value = price_instrument(
        option, model, market, method="fourier", cfg=cfg or FourierConfig()
    ).value
    return implied_volatility(value, option, market)


def implied_vol_surface(
    model: Any,
    market: Market,
    strikes,
    expiries,
    *,
    kind: str = "call",
    cfg: FourierConfig | None = None,
) -> np.ndarray:
    """Implied volatilities on the `(expiry, strike)` grid, shape `(n_T, n_K)`.

    Row `i` is the smile at `expiries[i]`; column `j` is the term structure at
    `strikes[j]`. Nothing is interpolated and nothing is cached: this is a
    double loop over `implied_vol`, which is the honest cost of a surface built
    from a transform price per point.
    """
    strike_grid = np.asarray(strikes, dtype=float)
    expiry_grid = np.asarray(expiries, dtype=float)
    if strike_grid.ndim != 1 or expiry_grid.ndim != 1:
        raise InvalidInputError("strikes and expiries must be one-dimensional")
    if strike_grid.size == 0 or expiry_grid.size == 0:
        raise InvalidInputError("strikes and expiries must be non-empty")
    return np.array(
        [
            [
                implied_vol(
                    model,
                    market,
                    strike=float(strike),
                    expiry=float(expiry),
                    kind=kind,
                    cfg=cfg,
                )
                for strike in strike_grid
            ]
            for expiry in expiry_grid
        ]
    )


def atm_implied_variance(
    model: Any,
    market: Market,
    expiry: float,
    *,
    cfg: FourierConfig | None = None,
) -> float:
    """`sigma_imp(F, T)^2` at the forward: the quantity whose limits are known.

    At the forward strike the implied *variance* is the natural object rather
    than the volatility, because `sigma_imp(F, T)^2 T` is an average of the
    instantaneous variance over `[0, T]` to leading order, so it runs from `v0`
    at `T -> 0` to `theta` as `T -> infinity`. `tests/test_heston_smile.py`
    measures both ends and the monotone approach between them.
    """
    vol = implied_vol(
        model,
        market,
        strike=_forward(market, expiry),
        expiry=expiry,
        cfg=cfg,
    )
    return vol * vol


def smile_skew(
    model: Any,
    market: Market,
    expiry: float,
    *,
    bump: float = SKEW_LOG_MONEYNESS_BUMP,
    cfg: FourierConfig | None = None,
) -> float:
    """`d sigma_imp / d ln(K / F)` at the forward, by a central difference.

    Negative for `rho < 0` and positive for `rho > 0`; its magnitude falls with
    maturity. Both statements are measured in `tests/test_heston_smile.py`
    rather than assumed here.
    """
    if bump <= 0.0:
        raise InvalidInputError("bump must be > 0")
    forward = _forward(market, expiry)
    up = implied_vol(
        model, market, strike=forward * math.exp(bump), expiry=expiry, cfg=cfg
    )
    down = implied_vol(
        model, market, strike=forward * math.exp(-bump), expiry=expiry, cfg=cfg
    )
    return (up - down) / (2.0 * bump)
