from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy.special import erf

from ..exceptions import InvalidInputError

ArrayLike = Union[float, int, np.ndarray]


@dataclass(frozen=True)
class BlackScholesModel:
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise InvalidInputError("sigma must be >= 0")



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
    if T < 0:
        raise InvalidInputError("T must be >= 0")
    if sigma < 0:
        raise InvalidInputError("sigma must be >= 0")
    if K <= 0:
        raise InvalidInputError("K must be > 0")

    S_arr = np.asarray(S, dtype=float)
    if np.any(S_arr <= 0):
        raise InvalidInputError("S must be > 0")

    kind_l = kind.lower()
    if kind_l not in {"call", "put"}:
        raise InvalidInputError("kind must be 'call' or 'put'")

    if T == 0:
        # At expiry: discounted payoff is just payoff at T=0 (no discounting needed)
        # but keep consistent with standard convention: price = intrinsic value.
        if kind_l == "call":
            out = np.maximum(S_arr - K, 0.0)
        else:
            out = np.maximum(K - S_arr, 0.0)
        return out.item() if np.isscalar(S) else out

    if sigma == 0:
        # Zero vol: deterministic forward under q, price is discounted intrinsic of forward payoff
        forward = S_arr * np.exp((r - q) * T)
        disc = np.exp(-r * T)
        if kind_l == "call":
            out = disc * np.maximum(forward - K, 0.0)
        else:
            out = disc * np.maximum(K - forward, 0.0)
        return out.item() if np.isscalar(S) else out

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
