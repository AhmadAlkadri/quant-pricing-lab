"""
Statistical utilities for quantitative finance.
"""

from typing import Sequence, Union
import numpy as np

def historical_volatility(
    prices: Union[Sequence[float], np.ndarray],
    *,
    annualization_factor: float = 252.0,
    demean: bool = True
) -> float:
    """
    Compute the annualized historical (realized) volatility from a price series.

    The volatility is calculated as the sample standard deviation of the
    log returns, scaled by the square root of the annualization factor.

    Parameters
    ----------
    prices : Sequence[float] | np.ndarray
        A sequence of prices (e.g. daily closes). Must contain at least 2 prices
        to compute a return.
    annualization_factor : float, default 252.0
        The factor to annualize the volatility. Common values:
        - 252 (daily trading days)
        - 365 (calendar days)
    demean : bool, default True
        If True, calculates the standard deviation with the sample mean of returns.
        If False, assumes the mean return is zero (common in high-frequency or
        short-horizon estimates where drift is negligible).
        Note: When using np.std, this toggles between centering the data or not.
        However, standard np.std always centers. To strictly assume zero mean,
        we would compute sqrt(mean(x^2)).
        For this implementation:
        - if demean=True (default), we use standard sample std (ddof=1).
        - if demean=False, we compute sqrt(sum(r^2) / (N-1)).

    Returns
    -------
    float
        The annualized volatility (sigma).

    Raises
    ------
    ValueError
        If fewer than 2 prices are provided.
    """
    prices_arr = np.asanyarray(prices, dtype=float)
    if len(prices_arr) < 2:
        raise ValueError("At least 2 prices are required to compute volatility.")

    # Compute log returns: r_t = ln(S_t / S_{t-1})
    log_returns = np.log(prices_arr[1:] / prices_arr[:-1])

    if demean:
        # Standard sample standard deviation (ddof=1 gives an unbiased estimator of variance)
        daily_vol = np.std(log_returns, ddof=1)
    else:
        # Assume mean is 0. Sample variance = sum(x^2) / (N-1)
        # We use N-1 for consistency with the sample std definition, though N is also common
        # for zero-mean assumptions. Let's stick to N-1 (ddof=1 implied behavior).
        sum_sq = np.sum(log_returns * log_returns)
        n = len(log_returns)
        daily_vol = np.sqrt(sum_sq / (n - 1))

    return daily_vol * np.sqrt(annualization_factor)
