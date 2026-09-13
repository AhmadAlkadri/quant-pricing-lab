"""Shared pricing points for the Fourier test modules.

Not a test module (pytest does not collect it): the four transform methods are
measured at the same five specifications so that their tables can be read side
by side, and one copy of those five is better than four. The list deliberately
stays here rather than in `qpl.cases` -- the cases layer carries *claims* with
evidence classes, and these are only coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

from qpl.instruments.options import DigitalOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel

__all__ = ["KINDS", "PAYOFFS", "POINTS", "Point"]


@dataclass(frozen=True)
class Point:
    """One pricing point, as scalars."""

    name: str
    spot: float
    strike: float
    expiry: float
    rate: float
    dividend: float
    sigma: float

    def market(self) -> Market:
        return Market(
            spot=self.spot,
            rate_curve=FlatRateCurve(self.rate),
            dividend_curve=FlatDividendCurve(self.dividend),
        )

    def model(self) -> BlackScholesModel:
        return BlackScholesModel(sigma=self.sigma)

    def instrument(self, kind: str, payoff: str):
        if payoff == "digital":
            return DigitalOption(
                kind=kind, strike=self.strike, expiry=self.expiry, cash=1.0
            )
        return EuropeanOption(kind=kind, strike=self.strike, expiry=self.expiry)


POINTS: tuple[Point, ...] = (
    Point("atm_1y", 100.0, 100.0, 1.0, 0.05, 0.00, 0.20),
    Point("otm_9m_div", 100.0, 110.0, 0.75, 0.03, 0.01, 0.25),
    Point("itm_1y_div", 120.0, 90.0, 1.0, 0.03, 0.05, 0.35),
    Point("short_atm", 100.0, 100.0, 0.05, 0.01, 0.00, 0.30),
    Point("long_high_vol", 80.0, 120.0, 2.0, 0.02, 0.03, 0.40),
)
"""Five points spanning moneyness, maturity, dividend yield and volatility.

`short_atm` and `long_high_vol` are the two that move the derived integration
reach the most (`12 / sqrt(sigma^2 T)` is 178.9 and 21.2), which is what makes
them worth carrying: a method that quietly hardcoded a reach would pass at the
middle three."""

KINDS = ("call", "put")
PAYOFFS = ("vanilla", "digital")
