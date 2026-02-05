"""
Tests for market statistics utilities.
"""
import pytest
import numpy as np
from qpl.market.stats import log_returns, realized_volatility
from qpl.exceptions import InvalidInputError

def test_log_returns_basic():
    """Test log returns calculation on simple inputs."""
    prices = [100.0, 101.0, 102.01] # 1% increase each step exactly? No, ln(1.01) approx 0.00995
    # ln(101/100) = ln(1.01)
    # ln(102.01/101) = ln(1.01)
    
    rets = log_returns(prices)
    assert len(rets) == 2
    assert rets[0] == pytest.approx(np.log(1.01))
    assert rets[1] == pytest.approx(np.log(1.01))

def test_log_returns_errors():
    """Test error conditions for log returns."""
    with pytest.raises(InvalidInputError):
        log_returns([100.0]) # Too short
        
    with pytest.raises(InvalidInputError):
        log_returns([100.0, -50.0]) # Negative price

def test_realized_volatility_constant():
    """Constant returns -> 0 volatility."""
    rets = [0.01, 0.01, 0.01, 0.01]
    vol = realized_volatility(rets, demean=True)
    assert vol == pytest.approx(0.0)

def test_realized_volatility_known():
    """
    Test with returns [+x, -x].
    Mean = 0.
    Std(ddof=1) = sqrt( (x^2 + x^2) / (2-1) ) = sqrt(2)*x.
    """
    x = 0.01
    rets = [x, -x]
    
    # demean=True
    vol = realized_volatility(rets, annualization=1.0, demean=True)
    expected = np.std(rets, ddof=1)
    assert vol == pytest.approx(expected)
    assert vol == pytest.approx(np.sqrt(2)*x)
    
    # demean=False (Mean assumed 0)
    # sqrt( (x^2 + (-x)^2) / (2-1) ) = sqrt(2)*x
    vol_nodean = realized_volatility(rets, annualization=1.0, demean=False)
    assert vol_nodean == pytest.approx(np.sqrt(2)*x)

def test_realized_volatility_annualization():
    """Check scaling factor."""
    rets = [0.01, -0.01]
    daily_vol = np.std(rets, ddof=1)
    
    vol = realized_volatility(rets, annualization=252)
    assert vol == pytest.approx(daily_vol * np.sqrt(252))

def test_insufficient_data_vol():
    """Check behavior with minimal data."""
    # Empty -> Error
    with pytest.raises(InvalidInputError):
        realized_volatility([])
        
    # 1 data point -> 0.0 (by our convention)
    assert realized_volatility([0.01]) == 0.0

def test_rolling_realized_volatility_basics():
    """Test rolling volatility on synthetic data."""
    # Prices: 100, 101, 102, 103... constant log return?
    # ln(101/100) ~ 0.00995
    # If returns are constant, std dev is 0.
    
    # Create geometric brownian motion-ish path relative to constant return
    # p0=100
    # p1=100*e^r
    # p2=100*e^(2r)
    # returns are exactly r.
    r = 0.01
    prices = 100.0 * np.exp(np.arange(10) * r)
    
    # Window = 3
    # Returns: [r, r, r, r, r, r, r, r, r] (len 9)
    # Rolling (3) of constant array -> 0 std dev.
    
    vol = rolling_realized_volatility(prices, window=3, annualization=1.0)
    
    assert len(vol) == 10
    # First `window` + 1?
    # indices:
    # 0: NaN (no ret)
    # 1: NaN (ret 0 available, need 3 returns? No, pandas rolling usually requires `min_periods` or full window)
    # By default rolling(window=3) produces NaN until 3 items.
    # returns indices: 0, 1, 2. (corresponds to prices 1, 2, 3).
    # So at price index 3 (4th price), we have returns r0, r1, r2.
    # That is the first point we have 3 returns.
    # The result array aligns such that result[t] uses returns up to t.
    # So result[3] uses buffer ending at 3.
    
    # Let's check non-nan
    # index 0: Nan
    # index 1: (ret 0) -> NaN (len 1 < 3)
    # index 2: (ret 0, 1) -> NaN (len 2 < 3)
    # index 3: (ret 0, 1, 2) -> 0.0
    
    assert np.isnan(vol[0])
    assert np.isnan(vol[1])
    assert np.isnan(vol[2])
    assert vol[3] == pytest.approx(0.0)
    assert vol[-1] == pytest.approx(0.0)

def test_rolling_realized_volatility_value():
    """Check numerical correctness of rolling vol."""
    # Returns: 0.01, -0.01, 0.01, -0.01
    # prices: 100, 101.005, 100, ...
    # std( [.01, -.01] ) is specific value.
    
    prices = [100.0, 101.0050167, 100.0, 101.0050167, 100.0]
    # log(1.01005...) approx 0.01
    # returns approx: [0.01, -0.01, 0.01, -0.01]
    
    vol = rolling_realized_volatility(prices, window=2, annualization=1.0)
    
    # Expected std of [0.01, -0.01] is sqrt(2)*0.01 approx 0.01414
    expected = 0.0141421356
    
    # index 0: Nan
    # index 1: (ret 0) -> Nan (window=2)
    # index 2: (ret 0, 1) -> valid
    assert np.isnan(vol[1]) 
    assert vol[2] == pytest.approx(expected, rel=1e-4)

def test_fit_normal_returns_values():
    """Test parameter fitting."""
    from qpl.market.stats import fit_normal_returns
    
    # Create synthetic returns with mean 0.001 and std 0.01
    np.random.seed(42)
    # N=10000 large sample to converge
    rets = np.random.normal(0.001, 0.01, size=10000)
    
    params = fit_normal_returns(rets, annualization=1.0)
    
    # Check close to true values
    assert params.mu_daily == pytest.approx(0.001, abs=1e-3)
    assert params.sigma_daily == pytest.approx(0.01, abs=1e-3)
    # Annualization 1.0 -> same
    assert params.mu_annual == params.mu_daily
    assert params.sigma_annual == params.sigma_daily

def test_fit_normal_annualization():
    """Test annualization logic."""
    from qpl.market.stats import fit_normal_returns
    
    rets = [0.01, 0.01] # Mean 0.01, Std 0
    params = fit_normal_returns(rets, annualization=100)
    
    assert params.mu_daily == 0.01
    assert params.mu_annual == 1.0  # 0.01 * 100
    assert params.sigma_daily == 0.0
    assert params.sigma_annual == 0.0
