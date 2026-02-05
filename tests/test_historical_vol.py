"""
Tests for historical volatility estimation.
"""

import math
import numpy as np
import pytest
from qpl.utils.statistics import historical_volatility

def test_historical_volatility_constant_growth():
    """
    If a price series grows at a constant geometric rate, returns are constant.
    The standard deviation of a constant sequence is 0.
    """
    # Create simple series: 100, 100*e^0.1, 100*e^0.2, ...
    rate = 0.01
    n_days = 10
    prices = [100.0 * math.exp(rate * i) for i in range(n_days)]
    
    vol = historical_volatility(prices)
    assert vol == pytest.approx(0.0, abs=1e-9)

def test_historical_volatility_known_series():
    """
    Test with a simple alternating series where we can hand-calculate the result.
    Prices: 100, 200, 100, 200
    Returns: ln(2), ln(0.5)=-ln(2), ln(2)
    Sequence of returns: [x, -x, x] where x = ln(2) approx 0.6931
    Mean return: (x - x + x) / 3 = x / 3
    
    Let's use a simpler one to get exact 0 mean if possible, or just trust the math.
    Sequence: 100, 200, 100
    Returns: [ln(2), ln(0.5)] -> [ln(2), -ln(2)]
    Mean: 0
    Variance (ddof=1): ( (x-0)^2 + (-x-0)^2 ) / (2-1) = 2x^2
    Std Dev: sqrt(2) * x
    Annualized: sqrt(2) * x * sqrt(252)
    """
    prices = [100, 200, 100]
    x = math.log(2)
    
    vol = historical_volatility(prices)
    
    expected_daily_vol = math.sqrt(2) * x
    expected_annual_vol = expected_daily_vol * math.sqrt(252)
    
    assert vol == pytest.approx(expected_annual_vol, rel=1e-9)

def test_demean_false():
    """
    Test the behavior when demean=False.
    If returns are [x, x], mean is x.
    std(demean=True) = 0.
    std(demean=False) = sqrt( (x^2 + x^2) / (N-1) ) = sqrt( 2x^2 / 1 ) = sqrt(2)*x.
    """
    # Prices: 100, 100*e^0.1, 100*e^0.2
    # Returns: [0.1, 0.1]
    rate = 0.1
    prices = [100, 100*math.exp(rate), 100*math.exp(2*rate)]
    
    # Check default behavior (demean=True) -> 0 vol
    assert historical_volatility(prices, demean=True) == pytest.approx(0.0, abs=1e-9)
    
    # Check demean=False
    # returns r = [0.1, 0.1]
    # sum_sq = 0.01 + 0.01 = 0.02
    # n-1 = 1
    # daily_vol = sqrt(0.02)
    vol_nodean = historical_volatility(prices, demean=False)
    expected_daily = math.sqrt(0.02)
    expected_annual = expected_daily * math.sqrt(252)
    
    assert vol_nodean == pytest.approx(expected_annual, rel=1e-9)

def test_insufficient_data():
    """Using fewer than 2 prices should raise ValueError."""
    with pytest.raises(ValueError, match="At least 2 prices"):
        historical_volatility([100])

def test_input_types():
    """Should handle numpy arrays and lists."""
    prices_list = [100, 101, 102]
    prices_arr = np.array(prices_list)
    
    v1 = historical_volatility(prices_list)
    v2 = historical_volatility(prices_arr)
    assert v1 == pytest.approx(v2)
