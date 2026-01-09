import math
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
