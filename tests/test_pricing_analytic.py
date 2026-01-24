import math

import pytest

from qpl.exceptions import InvalidInputError
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price


def _market(spot: float, r: float, q: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(r),
        dividend_curve=FlatDividendCurve(q),
    )


def test_put_call_parity_analytic_dispatch():
    s = 100.0
    k = 110.0
    t = 0.75
    r = 0.03
    q = 0.01
    sigma = 0.25

    call = EuropeanOption(kind="call", strike=k, expiry=t)
    put = EuropeanOption(kind="put", strike=k, expiry=t)
    model = BlackScholesModel(sigma=sigma)
    market = _market(s, r, q)

    call_px = price(call, model, market, method="analytic").value
    put_px = price(put, model, market, method="analytic").value

    lhs = call_px - put_px
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert abs(lhs - rhs) < 1e-9


def test_price_T_zero_intrinsic():
    s = 105.0
    k = 100.0
    t = 0.0
    r = 0.05
    q = 0.01

    call = EuropeanOption(kind="call", strike=k, expiry=t)
    put = EuropeanOption(kind="put", strike=k, expiry=t)
    model = BlackScholesModel(sigma=0.2)
    market = _market(s, r, q)

    call_px = price(call, model, market, method="analytic").value
    put_px = price(put, model, market, method="analytic").value

    assert call_px == pytest.approx(max(s - k, 0.0))
    assert put_px == pytest.approx(max(k - s, 0.0))


def test_price_sigma_zero_discounted_forward_intrinsic():
    s = 100.0
    k = 105.0
    t = 1.0
    r = 0.05
    q = 0.02

    call = EuropeanOption(kind="call", strike=k, expiry=t)
    put = EuropeanOption(kind="put", strike=k, expiry=t)
    model = BlackScholesModel(sigma=0.0)
    market = _market(s, r, q)

    call_px = price(call, model, market, method="analytic").value
    put_px = price(put, model, market, method="analytic").value

    forward = s * math.exp((r - q) * t)
    disc = math.exp(-r * t)
    call_expected = disc * max(forward - k, 0.0)
    put_expected = disc * max(k - forward, 0.0)

    assert call_px == pytest.approx(call_expected, abs=1e-12)
    assert put_px == pytest.approx(put_expected, abs=1e-12)


def test_call_monotonicity_spot_strike_vol():
    r = 0.01
    q = 0.0
    t = 1.0

    strikes = [90.0, 100.0, 110.0]
    spots = [90.0, 100.0, 110.0]
    sigmas = [0.1, 0.2, 0.3]

    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, r, q)

    prices_spot = [
        price(
            EuropeanOption("call", strike=100.0, expiry=t),
            model,
            _market(s, r, q),
        ).value
        for s in spots
    ]
    assert prices_spot[0] <= prices_spot[1] + 1e-12
    assert prices_spot[1] <= prices_spot[2] + 1e-12

    prices_strike = [
        price(EuropeanOption("call", strike=k, expiry=t), model, market).value
        for k in strikes
    ]
    assert prices_strike[0] >= prices_strike[1] - 1e-12
    assert prices_strike[1] >= prices_strike[2] - 1e-12

    prices_sigma = [
        price(
            EuropeanOption("call", strike=100.0, expiry=t),
            BlackScholesModel(sigma=s),
            market,
        ).value
        for s in sigmas
    ]
    assert prices_sigma[0] <= prices_sigma[1] + 1e-12
    assert prices_sigma[1] <= prices_sigma[2] + 1e-12


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
