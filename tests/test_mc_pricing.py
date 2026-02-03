import math

import pytest

from qpl.engines.mc.pricers import MCConfig
from qpl.exceptions import InvalidInputError, NotSupportedError
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


def test_mc_seed_determinism():
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.01)

    cfg = MCConfig(n_paths=20_000, seed=123)
    res_a = price(option, model, market, method="mc", cfg=cfg)
    res_b = price(option, model, market, method="mc", cfg=cfg)

    assert res_a.value == res_b.value
    assert res_a.stderr == res_b.stderr

    cfg_other = MCConfig(n_paths=20_000, seed=124)
    res_c = price(option, model, market, method="mc", cfg=cfg_other)

    assert (res_a.value != res_c.value) or (res_a.stderr != res_c.stderr)


def test_mc_matches_analytic_within_ci():
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.01)

    cfg = MCConfig(n_paths=100_000, seed=7)
    mc = price(option, model, market, method="mc", cfg=cfg)
    analytic = price(option, model, market, method="analytic")

    assert math.isfinite(mc.stderr) and mc.stderr > 0.0
    assert abs(mc.value - analytic.value) <= 4.0 * mc.stderr


def test_mc_put_call_parity_within_noise():
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
    cfg_call = MCConfig(n_paths=100_000, seed=42)
    cfg_put = MCConfig(n_paths=100_000, seed=43)

    call_mc = price(call, model, market, method="mc", cfg=cfg_call)
    put_mc = price(put, model, market, method="mc", cfg=cfg_put)

    lhs = call_mc.value - put_mc.value
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    combined = math.sqrt(call_mc.stderr**2 + put_mc.stderr**2)

    assert abs(lhs - rhs) <= 6.0 * combined


def test_mc_invalid_n_paths_raises():
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.01)
    cfg = MCConfig(n_paths=1, seed=1)

    with pytest.raises(InvalidInputError):
        price(option, model, market, method="mc", cfg=cfg)


def test_mc_T_zero_intrinsic_and_zero_stderr():
    option = EuropeanOption(kind="call", strike=100.0, expiry=0.0)
    model = BlackScholesModel(sigma=0.3)
    market = _market(105.0, 0.05, 0.01)
    cfg = MCConfig(n_paths=2, seed=99)

    res = price(option, model, market, method="mc", cfg=cfg)
    assert res.value == pytest.approx(5.0)
    assert res.stderr == 0.0


def test_mc_sigma_zero_discounted_forward_intrinsic_and_zero_stderr():
    option = EuropeanOption(kind="put", strike=110.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.0)
    market = _market(100.0, 0.05, 0.02)
    cfg = MCConfig(n_paths=2, seed=100)

    res = price(option, model, market, method="mc", cfg=cfg)
    forward = 100.0 * math.exp((0.05 - 0.02) * 1.0)
    disc = math.exp(-0.05 * 1.0)
    expected = disc * max(110.0 - forward, 0.0)
    assert res.value == pytest.approx(expected, abs=1e-12)
    assert res.stderr == 0.0


def test_mc_greeks_supported():
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.01)
    cfg = MCConfig(n_paths=2, seed=1)

    res = greeks(option, model, market, method="mc", cfg=cfg)
    assert res.meta is not None
    assert res.meta["method"] == "mc"


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


def test_mc_uses_curve_df():
    option = EuropeanOption(kind="call", strike=90.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.0)
    market = Market(
        spot=100.0,
        rate_curve=_CurveWithDf(rate_attr=0.10, df_rate=0.01),
        dividend_curve=_DivCurveWithDf(yield_attr=0.05, df_yield=0.02),
    )
    cfg = MCConfig(n_paths=2, n_steps=1, seed=1)

    res = price(option, model, market, method="mc", cfg=cfg)
    forward = 100.0 * math.exp((0.01 - 0.02) * 1.0)
    expected = math.exp(-0.01) * max(forward - 90.0, 0.0)
    assert res.value == pytest.approx(expected, abs=0.0)
    assert res.stderr == 0.0


def test_mc_invalid_n_steps_raises():
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.01)
    cfg = MCConfig(n_paths=2, n_steps=0, seed=1)

    with pytest.raises(InvalidInputError):
        price(option, model, market, method="mc", cfg=cfg)


def test_mc_greeks_match_analytic_call():
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.01)
    cfg = MCConfig(n_paths=200_000, n_steps=1, seed=42)
    bumps = {"spot": 1e-2, "sigma": 1e-4, "r": 1e-5}

    g_mc = greeks(option, model, market, method="mc", cfg=cfg, bumps=bumps)
    g_an = greeks(option, model, market, method="analytic")

    assert abs(g_mc.delta - g_an.delta) < 5e-3
    assert abs(g_mc.gamma - g_an.gamma) < 5e-3
    assert abs(g_mc.vega - g_an.vega) < 2e-1
    assert abs(g_mc.rho - g_an.rho) < 2e-1
    assert g_mc.meta is not None
    assert g_mc.meta["method"] == "mc"
    assert "spot" in g_mc.meta["bumps"]
    assert "sigma" in g_mc.meta["bumps"]
    assert "r" in g_mc.meta["bumps"]


def test_mc_greeks_match_analytic_put():
    option = EuropeanOption(kind="put", strike=110.0, expiry=0.75)
    model = BlackScholesModel(sigma=0.25)
    market = _market(105.0, 0.04, 0.02)
    cfg = MCConfig(n_paths=200_000, n_steps=1, seed=42)
    bumps = {"spot": max(abs(market.spot) * 1e-4, 1e-6), "sigma": 1e-4, "r": 1e-5}

    g_mc = greeks(option, model, market, method="mc", cfg=cfg, bumps=bumps)
    g_an = greeks(option, model, market, method="analytic")

    assert abs(g_mc.delta - g_an.delta) < 5e-3
    assert abs(g_mc.gamma - g_an.gamma) < 5e-3
    assert abs(g_mc.vega - g_an.vega) < 2e-1
    assert abs(g_mc.rho - g_an.rho) < 2e-1


def test_mc_greeks_validation_errors():
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.01)
    cfg = MCConfig(n_paths=2, n_steps=1, seed=1)

    with pytest.raises(InvalidInputError):
        greeks(option, model, market, method="mc")

    with pytest.raises(InvalidInputError):
        greeks(option, model, market, method="mc", cfg=cfg, bumps=1.0)
