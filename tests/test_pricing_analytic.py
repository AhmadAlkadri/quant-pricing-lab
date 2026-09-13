import math

import pytest

from qpl.exceptions import InvalidInputError
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel, bs_price
from qpl.pricing import greeks, price


def _market(spot: float, r: float, q: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(r),
        dividend_curve=FlatDividendCurve(q),
    )


def test_invalid_inputs_raise():
    with pytest.raises(InvalidInputError):
        EuropeanOption(kind="call", strike=-1.0, expiry=1.0)
    with pytest.raises(InvalidInputError):
        EuropeanOption(kind="call", strike=100.0, expiry=-1.0)
    with pytest.raises(InvalidInputError):
        BlackScholesModel(sigma=-0.1)
    with pytest.raises(InvalidInputError):
        FlatRateCurve(rate=-0.01)
    with pytest.raises(InvalidInputError):
        FlatDividendCurve(yield_=-0.01)
    with pytest.raises(InvalidInputError):
        Market(
            spot=-100.0,
            rate_curve=FlatRateCurve(0.01),
            dividend_curve=FlatDividendCurve(0.0),
        )


def test_greeks_sanity_call():
    opt = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.0)

    g = greeks(opt, model, market, method="analytic")

    assert 0.0 <= g.delta <= 1.0
    assert g.gamma >= 0.0
    assert g.vega >= 0.0


class _CurveWithDf:
    def __init__(self, rate_attr: float, df_rate: float) -> None:
        self.rate = rate_attr
        self._df_rate = df_rate

    def df(self, t: float) -> float:
        return math.exp(-self._df_rate * t)


class _DivCurveWithDf:
    def __init__(self, yield_attr: float, df_yield: float) -> None:
        self.yield_ = yield_attr
        self._df_yield = df_yield

    def df(self, t: float) -> float:
        return math.exp(-self._df_yield * t)


def test_analytic_uses_curve_df():
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = Market(
        spot=100.0,
        rate_curve=_CurveWithDf(rate_attr=0.10, df_rate=0.01),
        dividend_curve=_DivCurveWithDf(yield_attr=0.05, df_yield=0.02),
    )

    res = price(option, model, market, method="analytic").value
    expected = bs_price(S=100.0, K=100.0, T=1.0, r=0.01, sigma=0.2, q=0.02, kind="call")
    assert res == pytest.approx(expected)
