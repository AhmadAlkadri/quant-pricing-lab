import math

import pytest

from qpl.engines.pde.pricers import PDEConfig
from qpl.exceptions import InvalidInputError
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


def test_pde_strike_alignment_default_is_unaligned_and_bit_identical():
    # The default must not perturb existing results: `strike_alignment="none"`
    # is spelled out explicitly here and must reproduce the default bit for bit.
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.0)
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)

    default_cfg = PDEConfig(n_s=100, n_t=100, theta=0.5, s_max_multiplier=4.0)
    explicit_cfg = PDEConfig(
        n_s=100, n_t=100, theta=0.5, s_max_multiplier=4.0, strike_alignment="none"
    )

    res_default = price(option, model, market, method="pde", cfg=default_cfg)
    res_explicit = price(option, model, market, method="pde", cfg=explicit_cfg)

    assert default_cfg.strike_alignment == "none"
    assert res_default.value == res_explicit.value
    assert res_default.meta["s_max"] == 400.0
    assert res_default.meta["ds"] == 4.0
    assert res_default.meta["strike_alignment"] == "none"


def test_pde_midpoint_alignment_places_strike_between_nodes():
    # With ds reported in meta, the strike must sit at a half-integer number of
    # spacings from the origin, and n_s must be preserved (s_max moves instead).
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.0)
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)

    cfg = PDEConfig(
        n_s=100, n_t=100, theta=0.5, s_max_multiplier=4.0, strike_alignment="midpoint"
    )
    res = price(option, model, market, method="pde", cfg=cfg)

    ds = res.meta["ds"]
    offsets = option.strike / ds
    assert res.meta["strike_alignment"] == "midpoint"
    assert res.meta["n_s"] == 100
    assert abs(offsets - round(offsets - 0.5) - 0.5) < 1e-12
    assert res.meta["s_max"] == ds * 100
    # ds is rescaled by at most 0.5 / (j + 0.5) relative to the nominal spacing,
    # so s_max is rescaled by the same factor.
    j = round(offsets - 0.5)
    assert abs(res.meta["s_max"] - 400.0) <= 400.0 * 0.5 / (j + 0.5) + 1e-12


def test_pde_midpoint_alignment_is_deterministic():
    cfg = PDEConfig(
        n_s=120, n_t=90, theta=0.5, s_max_multiplier=4.0, strike_alignment="midpoint"
    )
    model = BlackScholesModel(sigma=0.25)
    market = _market(95.0, 0.03, 0.01)
    option = EuropeanOption(kind="put", strike=105.0, expiry=0.5)

    a = price(option, model, market, method="pde", cfg=cfg).value
    b = price(option, model, market, method="pde", cfg=cfg).value
    assert math.isfinite(a)
    assert a == b


def test_pde_rejects_unknown_alignment_and_strike_outside_grid():
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.0)
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)

    with pytest.raises(InvalidInputError):
        price(
            option,
            model,
            market,
            method="pde",
            cfg=PDEConfig(n_s=50, n_t=50, strike_alignment="left"),  # type: ignore[arg-type]
        )

    # s_max below the strike leaves the payoff kink outside the grid entirely.
    with pytest.raises(InvalidInputError):
        price(option, model, market, method="pde", cfg=PDEConfig(n_s=50, n_t=50, s_max=80.0))
