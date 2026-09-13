from __future__ import annotations

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


def test_ch4_theta_prices_close_to_analytic_call() -> None:
    s = 100.0
    k = 100.0
    t = 1.0
    r = 0.05
    q = 0.01
    sigma = 0.2

    option = EuropeanOption(kind="call", strike=k, expiry=t)
    model = BlackScholesModel(sigma=sigma)
    market = _market(s, r, q)
    analytic = price(option, model, market, method="analytic").value

    for theta in (0.0, 0.5, 1.0):
        cfg = PDEConfig(n_s=50, n_t=50, theta=theta, s_max_multiplier=4.0)
        pde = price(option, model, market, method="pde", cfg=cfg).value
        assert math.isfinite(pde)
        assert abs(pde - analytic) <= 0.05


def test_ch4_refinement_error_monotone_nonincreasing() -> None:
    s = 100.0
    k = 100.0
    t = 1.0
    r = 0.05
    q = 0.01
    sigma = 0.2

    model = BlackScholesModel(sigma=sigma)
    market = _market(s, r, q)

    for kind in ("call", "put"):
        option = EuropeanOption(kind=kind, strike=k, expiry=t)
        analytic = price(option, model, market, method="analytic").value

        errs: list[float] = []
        for n in (20, 30, 50):
            cfg = PDEConfig(n_s=n, n_t=n, theta=0.5, s_max_multiplier=4.0)
            pde = price(option, model, market, method="pde", cfg=cfg).value
            errs.append(abs(pde - analytic))

        assert errs[0] + 1e-12 >= errs[1]
        assert errs[1] + 1e-12 >= errs[2]


def test_ch4_call_monotonicity_in_spot_for_each_theta() -> None:
    k = 100.0
    t = 1.0
    r = 0.05
    q = 0.01
    sigma = 0.2

    option = EuropeanOption(kind="call", strike=k, expiry=t)
    model = BlackScholesModel(sigma=sigma)
    spots = (90.0, 100.0, 110.0)

    for theta in (0.0, 0.5, 1.0):
        prices: list[float] = []
        for s in spots:
            cfg = PDEConfig(n_s=50, n_t=50, theta=theta, s_max_multiplier=4.0)
            pde = price(option, model, _market(s, r, q), method="pde", cfg=cfg).value
            prices.append(pde)

        assert prices[0] <= prices[1] + 1e-12
        assert prices[1] <= prices[2] + 1e-12
