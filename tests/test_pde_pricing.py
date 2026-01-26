import math

from qpl.engines.pde.pricers import PDEConfig
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price


def _market(spot: float, r: float, q: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(r),
        dividend_curve=FlatDividendCurve(q),
    )


def test_pde_matches_analytic_call_put():
    s = 100.0
    k = 100.0
    t = 1.0
    r = 0.05
    q = 0.01
    sigma = 0.2

    cfg = PDEConfig(n_s=200, n_t=200, theta=0.5, s_max_multiplier=4.0)
    model = BlackScholesModel(sigma=sigma)
    market = _market(s, r, q)

    call = EuropeanOption(kind="call", strike=k, expiry=t)
    put = EuropeanOption(kind="put", strike=k, expiry=t)

    pde_call = price(call, model, market, method="pde", cfg=cfg).value
    pde_put = price(put, model, market, method="pde", cfg=cfg).value
    analytic_call = price(call, model, market, method="analytic").value
    analytic_put = price(put, model, market, method="analytic").value

    assert abs(pde_call - analytic_call) < 1e-2
    assert abs(pde_put - analytic_put) < 1e-2


def test_pde_determinism():
    cfg = PDEConfig(n_s=150, n_t=150, theta=1.0, s_max_multiplier=4.0)
    model = BlackScholesModel(sigma=0.3)
    market = _market(120.0, 0.04, 0.02)
    option = EuropeanOption(kind="call", strike=110.0, expiry=0.75)

    res_a = price(option, model, market, method="pde", cfg=cfg).value
    res_b = price(option, model, market, method="pde", cfg=cfg).value

    assert math.isfinite(res_a)
    assert res_a == res_b
