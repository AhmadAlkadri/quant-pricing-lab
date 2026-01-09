import numpy as np
import pytest

from qpl.instruments.payoffs import call_payoff, put_payoff


def test_call_payoff_scalar():
    assert call_payoff(120.0, 100.0) == 20.0
    assert call_payoff(80.0, 100.0) == 0.0
    assert call_payoff(100.0, 100.0) == 0.0


def test_put_payoff_scalar():
    assert put_payoff(80.0, 100.0) == 20.0
    assert put_payoff(120.0, 100.0) == 0.0
    assert put_payoff(100.0, 100.0) == 0.0


def test_payoffs_vectorized():
    s = np.array([80.0, 100.0, 120.0])
    k = 100.0

    c = call_payoff(s, k)
    p = put_payoff(s, k)

    assert np.allclose(c, np.array([0.0, 0.0, 20.0]))
    assert np.allclose(p, np.array([20.0, 0.0, 0.0]))
