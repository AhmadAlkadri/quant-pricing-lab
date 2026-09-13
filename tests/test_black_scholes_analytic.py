import numpy as np

from qpl.models.black_scholes import bs_price


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
