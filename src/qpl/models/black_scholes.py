from __future__ import annotations

import numpy as np
from typing import Union
from scipy.special import erf

ArrayLike = Union[float, int, np.ndarray]


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """
    Standard normal CDF using erf. Vectorized.
    """
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def bs_price(
    *,
    S: ArrayLike,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
) -> ArrayLike:
    """
    Black–Scholes price for a European call or put with continuous dividend yield q.

    Parameters
    ----------
    S : scalar or np.ndarray
        Spot price(s).
    K : float
        Strike.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualized).
    q : float, optional
        Continuous dividend yield (default 0.0).
    kind : {"call","put"}
        Option type.

    Returns
    -------
    scalar or np.ndarray
        Price(s) matching the shape of S.
    """
    if T <= 0:
        # At expiry: discounted payoff is just payoff at T=0 (no discounting needed)
        # but keep consistent with standard convention: price = intrinsic value.
        S_arr = np.asarray(S, dtype=float)
        if kind.lower() == "call":
            out = np.maximum(S_arr - K, 0.0)
        elif kind.lower() == "put":
            out = np.maximum(K - S_arr, 0.0)
        else:
            raise ValueError("kind must be 'call' or 'put'")
        return out.item() if np.isscalar(S) else out

    if sigma <= 0:
        # Zero vol: deterministic forward under q, price is discounted intrinsic of forward payoff
        S_arr = np.asarray(S, dtype=float)
        forward = S_arr * np.exp((r - q) * T)
        disc = np.exp(-r * T)
        if kind.lower() == "call":
            out = disc * np.maximum(forward - K, 0.0)
        elif kind.lower() == "put":
            out = disc * np.maximum(K - forward, 0.0)
        else:
            raise ValueError("kind must be 'call' or 'put'")
        return out.item() if np.isscalar(S) else out

    kind_l = kind.lower()
    if kind_l not in {"call", "put"}:
        raise ValueError("kind must be 'call' or 'put'")

    S_arr = np.asarray(S, dtype=float)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S_arr / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)

    if kind_l == "call":
        out = S_arr * df_q * Nd1 - K * df_r * Nd2
    else:
        # Put: N(-d) = 1 - N(d)
        Nmd1 = _norm_cdf(-d1)
        Nmd2 = _norm_cdf(-d2)
        out = K * df_r * Nmd2 - S_arr * df_q * Nmd1

    return out.item() if np.isscalar(S) else out
