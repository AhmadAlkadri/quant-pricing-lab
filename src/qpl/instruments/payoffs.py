from __future__ import annotations

from typing import Union
import numpy as np

ArrayLike = Union[float, int, np.ndarray]


def call_payoff(S: ArrayLike, K: float) -> ArrayLike:
    """
    European call payoff: max(S - K, 0)

    Supports scalar or numpy array S.
    """
    S_arr = np.asarray(S)
    out = np.maximum(S_arr - K, 0.0)
    return out.item() if np.isscalar(S) else out


def put_payoff(S: ArrayLike, K: float) -> ArrayLike:
    """
    European put payoff: max(K - S, 0)

    Supports scalar or numpy array S.
    """
    S_arr = np.asarray(S)
    out = np.maximum(K - S_arr, 0.0)
    return out.item() if np.isscalar(S) else out
