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


def arithmetic_average(fixings: np.ndarray) -> np.ndarray:
    """Arithmetic mean `(1/n) sum_i S_i` along the last axis.

    `fixings` is a `(..., n)` array of spot values at the monitoring dates, so
    a `(n_paths, n_fixings)` sample reduces to one average per path.
    """
    arr = np.asarray(fixings, dtype=float)
    return np.mean(arr, axis=-1)


def geometric_average(fixings: np.ndarray) -> np.ndarray:
    """Geometric mean `(prod_i S_i)^{1/n}` along the last axis.

    Computed as `exp(mean(log S_i))` rather than as an n-th root of a product.
    The product of 52 lognormal draws overflows or underflows long before the
    average does -- at `S ~ 100` a 52-fixing product is `O(10**104)`, and one
    path that dips to `S ~ 1e-3` makes the same product denormal -- whereas the
    mean of the logs is `O(log S)` by construction. The two agree in exact
    arithmetic; only one of them agrees in float64.

    `S_i > 0` almost surely under geometric Brownian motion, and exactly so for
    any path this package generates (`exp` of a finite number). A zero or
    negative sample would give `-inf` or `nan` here rather than a silently wrong
    positive number, which is the preferable failure.
    """
    arr = np.asarray(fixings, dtype=float)
    return np.exp(np.mean(np.log(arr), axis=-1))


def asian_payoff(
    average: ArrayLike, K: float, kind: str = "call"
) -> ArrayLike:
    """Fixed-strike Asian payoff `max(A - K, 0)` / `max(K - A, 0)`.

    A thin selector over `call_payoff` / `put_payoff`: once the path has been
    reduced to its average the payoff *is* a vanilla payoff in that average.
    That sentence is the whole content of the arithmetic-vs-geometric
    comparison below -- the contracts differ only in how the path is reduced,
    which is why `A_arith >= A_geom` (AM-GM) transfers directly to the payoffs
    and hence to the prices.
    """
    if kind not in {"call", "put"}:
        raise ValueError("kind must be 'call' or 'put'")
    return call_payoff(average, K) if kind == "call" else put_payoff(average, K)
