import pytest
from qpl.engines.mc.pricers import MCConfig, greeks_european
from qpl.engines.analytic.black_scholes import greeks_european as greeks_analytic
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel

def test_mc_theta_vs_analytic():
    """
    Verify that MC Theta is non-zero and matches Analytic Theta within tolerance.
    """
    # 1. Setup
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = Market(
        spot=100.0,
        rate_curve=FlatRateCurve(0.05),
        dividend_curve=FlatDividendCurve(0.01),
    )
    
    # Use enough paths for stability
    mc_cfg = MCConfig(n_paths=200_000, seed=42)
    
    # 2. Compute
    res_analytic = greeks_analytic(option, model, market)
    res_mc = greeks_european(option, model, market, cfg=mc_cfg)
    
    # 3. Assertions
    print(f"DEBUG: Analytic Theta={res_analytic.theta}")
    print(f"DEBUG: MC Theta={res_mc.theta}")
    
    # A) Check it's implemented (not 0.0)
    # Note: Analytic theta for this case should be around -6.xxx
    assert res_mc.theta != 0.0, "MC Theta should not be 0.0"
    
    # B) Check accuracy
    # Theta is usually negative for long calls.
    # Tolerance of 0.2 is roughly 3% error for a value around -6.0
    # With 200k paths, standard error should be small enough.
    assert abs(res_mc.theta - res_analytic.theta) < 0.2
