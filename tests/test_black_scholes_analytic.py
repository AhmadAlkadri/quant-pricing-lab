import math
import numpy as np
import pytest

from qpl.models.black_scholes import bs_price


def test_bs_call_put_known_value():
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    q = 0.0

    call = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind="call")
    put  = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind="put")

    # Standard benchmark values (roughly)
    assert math.isfinite(call) and call > 0.0
    assert math.isfinite(put) and put > 0.0

    assert abs(call - 10.4506) < 1e-3
    assert abs(put  - 5.5735) < 1e-3


def test_put_call_parity():
    S = 100.0
    K = 110.0
    T = 0.75
    r = 0.03
    sigma = 0.25
    q = 0.01

    call = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind="call")
    put  = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind="put")

    # Put-call parity with continuous dividend yield q:
    # C - P = S*exp(-qT) - K*exp(-rT)
    lhs = call - put
    rhs = S * math.exp(-q*T) - K * math.exp(-r*T)
    assert abs(lhs - rhs) < 1e-10


def test_vector_inputs_match_scalar():
    S = np.array([90.0, 100.0, 110.0])
    K = 100.0
    T = 1.0
    r = 0.02
    sigma = 0.3
    q = 0.01

    call_vec = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind="call")
    put_vec = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind="put")

    call_exp = np.array([bs_price(S=float(s), K=K, T=T, r=r, sigma=sigma, q=q, kind="call") for s in S])
    put_exp = np.array([bs_price(S=float(s), K=K, T=T, r=r, sigma=sigma, q=q, kind="put") for s in S])

    assert isinstance(call_vec, np.ndarray)
    assert isinstance(put_vec, np.ndarray)
    assert call_vec.shape == S.shape
    assert put_vec.shape == S.shape
    assert np.allclose(call_vec, call_exp, rtol=0.0, atol=1e-12)
    assert np.allclose(put_vec, put_exp, rtol=0.0, atol=1e-12)
