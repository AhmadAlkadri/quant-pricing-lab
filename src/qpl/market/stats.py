"""
Statistical utilities for market data analysis.
"""

from typing import Sequence, Union
import numpy as np
from qpl.exceptions import InvalidInputError

def log_returns(prices: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """
    Compute logarithmic returns from a price series.

    r_t = ln(p_t / p_{t-1})

    Parameters
    ----------
    prices : Sequence[float] | np.ndarray
        Sequence of prices. Must have at least 2 elements.

    Returns
    -------
    np.ndarray
        Array of log returns, length N-1.

    Raises
    ------
    InvalidInputError
        If prices has fewer than 2 elements.
    """
    prices_arr = np.asanyarray(prices, dtype=float).squeeze()
    if prices_arr.ndim != 1:
        raise InvalidInputError(f"Prices must be 1-dimensional, got {prices_arr.ndim}D")

    if len(prices_arr) < 2:
        raise InvalidInputError("At least 2 prices are required to compute returns.")
    
    # Handle zeros or negative prices which make log undefined
    if np.any(prices_arr <= 0):
         raise InvalidInputError("Prices must be strictly positive for log returns.")

    return np.log(prices_arr[1:] / prices_arr[:-1])


def realized_volatility(
    returns: Union[Sequence[float], np.ndarray],
    *,
    annualization: float = 252.0,
    demean: bool = True
) -> float:
    """
    Compute annualized realized volatility from returns.

    sigma = std(returns) * sqrt(annualization)

    Parameters
    ----------
    returns : Sequence[float] | np.ndarray
        Sequence of returns (log returns).
    annualization : float, default 252.0
        Annualization factor (e.g. 252 for daily data).
    demean : bool, default True
        If True, subtracts the sample mean (standard deviation).
        If False, assumes zero mean (sqrt(mean(x^2))).

    Returns
    -------
    float
        Annualized volatility.
    
    Raises
    ------
    InvalidInputError
        If returns array is empty.
    """
    r = np.asanyarray(returns, dtype=float)
    if len(r) == 0:
        raise InvalidInputError("Returns array cannot be empty.")
    
    if len(r) == 1:
        # Std of 1 point is technically 0 (or undefined depending on ddof), 
        # but let's return 0.0 for single return to be safe, or raise?
        # Standard deviation of 1 point with ddof=1 is NaN.
        # Let's enforce specific behavior: need at least 2 points for sample std?
        # If demean=False, 1 point is fine.
        if demean:
             return 0.0 # Or raise? 0.0 seems safer for degenerate case
        
    if demean:
        # std(ddof=1) is unbiased estimator for sample
        # If len=1, std yields nan.
        if len(r) < 2:
            return 0.0
        vol = np.std(r, ddof=1)
    else:
        # Root mean square (assuming mean=0)
        # We use N-1 to be consistent with sample variance definition? 
        # Or Just N?
        # Usually "Realized Volatility" often implies sum(r^2).
        # Let's stick to the previous logic: sqrt( sum(r^2) / (N-1) ) to generally match std behavior
        # but centered at 0.
        # If N=1, this would be infinite?
        n = len(r)
        if n < 2:
             return 0.0 # prevent div by zero
        sum_sq = np.sum(r * r)
        vol = np.sqrt(sum_sq / (n - 1))

    return vol * np.sqrt(annualization)


def rolling_realized_volatility(
    prices: Union[Sequence[float], np.ndarray],
    window: int,
    *,
    annualization: float = 252.0,
    demean: bool = True
) -> np.ndarray:
    """
    Compute rolling annualized realized volatility from a price series.
    
    The result array has the same length as the input `prices`.
    The first `window` elements will be NaN (returns are length N-1, plus windowing).
    
    Parameters
    ----------
    prices : Sequence[float] | np.ndarray
        Price series.
    window : int
        Size of the rolling window (in number of return periods).
    annualization : float, default 252.0
        Annualization factor.
    demean : bool, default True
        If True, calculate sample standard deviation (ddof=1).
        If False, calculate root mean squared return (assuming zero mean).
        
    Returns
    -------
    np.ndarray
        Array of annualized volatilities, aligned with `prices`.
        val[t] corresponds to volatility computed using returns up to time t.
    """
    import pandas as pd
    
    # 1. Compute returns
    # prices: [p0, p1, p2, ...] (len N)
    # returns: [r1, r2, ...] where r1 = ln(p1/p0) (len N-1)
    # rets array index i corresponds to p_{i+1}
    try:
        rets = log_returns(prices)
    except InvalidInputError:
        # If not enough data, return array of NaNs
        return np.full(len(prices), np.nan)
        
    # 2. Compute rolling stat
    s_rets = pd.Series(rets)
    
    if demean:
        # Standard deviation (unbiased, ddof=1 is pandas default)
        rolling_std = s_rets.rolling(window=window).std()
    else:
        # Root mean square (assume mean=0)
        # sum(x^2)/N or sum(x^2)/(N-1)?
        # To be consistent with realized_volatility logic for demean=False:
        # We used sqrt(sum(r^2)/(N-1)).
        # Pandas doesn't have a direct "rolling rms with ddof=1" method easily.
        # But we can do:
        # rolling_sum_sq = (s_rets**2).rolling(window).sum()
        # count = (s_rets.notna()).rolling(window).sum() (?) or just window if no nans.
        # Since we want to emulate the exact behavior of realized_volatility(..., demean=False):
        # vol = sqrt( sum(r^2) / (n - 1) )
        # Note: if n < 2, it handles it.
        
        sq = s_rets ** 2
        sum_sq = sq.rolling(window=window).sum()
        # For a full window, n = window.
        # We divide by window - 1 to match the 'sample proxy' scaling used in stats.py
        # Or should we just stick to 'demean=True' being the main path and keep this simple?
        # Let's match existing logic.
        denom = window - 1 if window > 1 else 1 # Avoid div by zero if window=1
        
        # If any NaNs in data, count might be less?
        # Let's assume clean data for now or let pandas handle NaNs propogation
        rolling_var = sum_sq / denom
        rolling_std = np.sqrt(rolling_var)

    # 3. Annualize
    rolling_vol = rolling_std * np.sqrt(annualization)
    
    # 4. Align with prices
    # rets has length N-1. rolling_vol has length N-1.
    # rolling_vol[i] is vol using returns ending at index i of rets.
    # rets[i] is return from p_i to p_{i+1}.
    # So rolling_vol[i] uses returns leading up to p_{i+1}.
    # We want result[t] to be vol at time t (using info up to t).
    # So result[i+1] = rolling_vol[i].
    # result[0] = NaN (no return).
    
    result = np.full(len(prices), np.nan)
    result[1:] = rolling_vol.values
    

    
    return result


from dataclasses import dataclass

@dataclass
class NormalParams:
    mu_daily: float
    sigma_daily: float
    mu_annual: float
    sigma_annual: float

def fit_normal_returns(
    returns: Union[Sequence[float], np.ndarray],
    *,
    annualization: float = 252.0
) -> NormalParams:
    """
    Fit a Normal distribution to the returns.
    
    Calculates sample mean and standard deviation (unbiased),
    both daily and annualized.
    
    mu_annual ~ mu_daily * annualization
    sigma_annual ~ sigma_daily * sqrt(annualization)
    
    Parameters
    ----------
    returns : array-like
        Log returns.
    annualization : float
        Annualization factor.
        
    Returns
    -------
    NormalParams
        Fitted parameters.
    """
    r = np.asanyarray(returns, dtype=float)
    if len(r) < 2:
        raise InvalidInputError("Need at least 2 returns to fit parameters.")
        
    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    
    return NormalParams(
        mu_daily=float(mu),
        sigma_daily=float(sigma),
        mu_annual=float(mu * annualization),
        sigma_annual=float(sigma * np.sqrt(annualization))
    )
