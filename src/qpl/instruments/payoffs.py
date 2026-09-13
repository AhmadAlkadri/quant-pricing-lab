from __future__ import annotations

import numpy as np

ArrayLike = float | int | np.ndarray


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


def digital_payoff(
    S: ArrayLike, K: float, cash: float = 1.0, kind: str = "call"
) -> ArrayLike:
    """
    European cash-or-nothing digital payoff.

        call: cash * 1{S > K},        put: cash * 1{S < K}

    Both indicators are strict, so `S == K` pays nothing either way. That is a
    probability-zero event under the model and the convention is therefore
    free; it is chosen to match QuantLib's `CashOrNothingPayoff`, which tests
    the same contract in `tests/oracle/test_digital_vs_quantlib.py`, and to be
    symmetric between the two kinds. It is *not* free numerically: a lattice
    node or a grid node can land exactly on the strike, and which way that node
    is counted is one of the things that drives the convergence behaviour
    measured in `docs/notes/digital_options_discontinuous_payoffs.md`.

    Unlike `call_payoff` and `put_payoff` this function is discontinuous at
    `K`: it is not Lipschitz, has unbounded difference quotients across the
    strike, and its distributional derivative is a Dirac mass there. Every
    numerical consequence in this repository follows from that one sentence.

    Supports scalar or numpy array S.
    """
    S_arr = np.asarray(S)
    indicator = S_arr > K if kind == "call" else S_arr < K
    out = cash * np.asarray(indicator, dtype=float)
    return out.item() if np.isscalar(S) else out
