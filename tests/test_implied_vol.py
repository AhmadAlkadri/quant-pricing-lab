import pytest
import math
from qpl.engines.analytic.black_scholes import implied_volatility, bs_price
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatRateCurve, FlatDividendCurve
from qpl.market.market import Market
from qpl.exceptions import InvalidInputError

def test_implied_vol_call_recovery():
    """Verify we can recover sigma for a Call."""
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.01
    sigma_true = 0.25
    
    # 1. Compute price
    price = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma_true, kind="call")
    
    # 2. Recover vol
    option = EuropeanOption(kind="call", strike=K, expiry=T)
    market = Market(
        spot=S, 
        rate_curve=FlatRateCurve(r), 
        dividend_curve=FlatDividendCurve(q)
    )
    
    iv = implied_volatility(float(price), option, market)
    assert abs(iv - sigma_true) < 1e-6

def test_implied_vol_put_recovery():
    """Verify we can recover sigma for a Put."""
    S, K, T, r, q = 100.0, 110.0, 0.5, 0.02, 0.0
    sigma_true = 0.40
    
    price = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma_true, kind="put")
    
    option = EuropeanOption(kind="put", strike=K, expiry=T)
    market = Market(spot=S, rate_curve=FlatRateCurve(r), dividend_curve=FlatDividendCurve(q))
    
    iv = implied_volatility(float(price), option, market)
    assert abs(iv - sigma_true) < 1e-6

def test_implied_vol_bounds_error():
    """Verify errors raised for arbitrage violations."""
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.0, 0.0
    option = EuropeanOption(kind="call", strike=K, expiry=T)
    market = Market(spot=S, rate_curve=FlatRateCurve(r), dividend_curve=FlatDividendCurve(q))
    
    # Intrinsic value for ATM call (r=q=0) ~ 0? No, max(S-K, 0) = 0.
    # Upper bound = S.
    
    # Case 1: Negative Price
    with pytest.raises(InvalidInputError, match="non-negative"):
        implied_volatility(-1.0, option, market)
        
    # Case 2: Below Intrinsic
    # Deep ITM Call: S=120, K=100. Intrinsic = 20. Price=10.
    option_itm = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    market_itm = Market(spot=120.0, rate_curve=FlatRateCurve(0), dividend_curve=FlatDividendCurve(0))
    with pytest.raises(InvalidInputError, match="outside bounds"):
        implied_volatility(10.0, option_itm, market_itm)

    # Case 3: Above Max (S)
    # Price = 130 (S=120)
    with pytest.raises(InvalidInputError, match="outside bounds"):
        implied_volatility(130.0, option_itm, market_itm)

def test_implied_vol_solver_fail():
    """Verify reasonable behavior if bracketing fails (though bounds check should catch most)."""
    # If we pass a price that is technically valid but requires sigma > 5.0 (default upper), 
    # it should raise InvalidInputError from our bracketing check.
    S, K, T = 100.0, 100.0, 1.0
    # Very high price -> very high vol
    # Max price is S (100). Let's try 99 (almost max).
    # This requires massive Vol.
    option = EuropeanOption(kind="call", strike=K, expiry=T)
    market = Market(spot=S, rate_curve=FlatRateCurve(0), dividend_curve=FlatDividendCurve(0))
    
    with pytest.raises(InvalidInputError, match="Cannot bracket"):
        implied_volatility(99.0, option, market, upper=2.0) # Restrict upper to force fail
