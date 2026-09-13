from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from qpl.exceptions import InvalidInputError
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.numerics.quadrature import composite_simpson, composite_trapezoid, gauss_legendre
from qpl.pricing import price


def test_composite_rules_known_integrals() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    exact = 2.0

    trap = composite_trapezoid(f, 0.0, math.pi, n_intervals=1_000)
    simp = composite_simpson(f, 0.0, math.pi, n_intervals=100)

    assert abs(trap - exact) <= 1e-4
    assert abs(simp - exact) <= 1e-7


def test_quadrature_accepts_scalar_integrand_output() -> None:
    def constant(_x: np.ndarray) -> float:
        return 3.0

    approx = composite_trapezoid(constant, 0.0, 2.0, n_intervals=100)
    assert abs(approx - 6.0) <= 1e-12


def test_gauss_legendre_exactness_for_polynomial_degree_2n_minus_1() -> None:
    n_nodes = 4

    def f(x: np.ndarray) -> np.ndarray:
        return x**7 - 2.0 * x**3 + 1.0
    # Integral over [-1, 1]: odd terms vanish, integral(1 dx)=2
    exact = 2.0

    approx = gauss_legendre(f, -1.0, 1.0, n_nodes=n_nodes)
    assert abs(approx - exact) <= 1e-12


def test_quadrature_call_price_close_to_analytic() -> None:
    s0 = 100.0
    k = 100.0
    t = 1.0
    r = 0.05
    q = 0.01
    sigma = 0.2

    option = EuropeanOption(kind="call", strike=k, expiry=t)
    model = BlackScholesModel(sigma=sigma)
    market = Market(
        spot=s0,
        rate_curve=FlatRateCurve(r),
        dividend_curve=FlatDividendCurve(q),
    )
    analytic = price(option, model, market, method="analytic").value

    mu_ln = math.log(s0) + (r - q - 0.5 * sigma * sigma) * t
    vol_ln = sigma * math.sqrt(t)
    disc = math.exp(-r * t)
    upper = s0 * math.exp((r - q - 0.5 * sigma * sigma) * t + 8.0 * vol_ln)

    def lognormal_density(st: np.ndarray) -> np.ndarray:
        return np.exp(-((np.log(st) - mu_ln) ** 2) / (2.0 * vol_ln * vol_ln)) / (
            st * vol_ln * math.sqrt(2.0 * math.pi)
        )

    def integrand(st: np.ndarray) -> np.ndarray:
        return np.maximum(st - k, 0.0) * lognormal_density(st)

    quad_price = disc * gauss_legendre(integrand, k, upper, n_nodes=64)

    assert abs(quad_price - analytic) <= 1e-3


def test_error_nonincreasing_with_refinement() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    exact = math.e - 1.0

    trap_grid = [16, 32, 64, 128]
    trap_errs = [abs(composite_trapezoid(f, 0.0, 1.0, n) - exact) for n in trap_grid]
    assert all(a + 1e-15 >= b for a, b in pairwise(trap_errs))

    simpson_grid = [16, 32, 64, 128]
    simpson_errs = [abs(composite_simpson(f, 0.0, 1.0, n) - exact) for n in simpson_grid]
    assert all(a + 1e-15 >= b for a, b in pairwise(simpson_errs))

    gauss_grid = [4, 8, 16, 32]
    gauss_errs = [abs(gauss_legendre(f, 0.0, 1.0, n) - exact) for n in gauss_grid]
    assert all(a + 1e-15 >= b for a, b in pairwise(gauss_errs))


def test_quadrature_validation_errors() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(InvalidInputError):
        composite_trapezoid(f, 1.0, 0.0, 10)
    with pytest.raises(InvalidInputError):
        composite_trapezoid(f, 0.0, 1.0, 0)
    with pytest.raises(InvalidInputError):
        composite_trapezoid(f, 0.0, 1.0, 10.0)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError):
        composite_simpson(f, 0.0, 1.0, 3)
    with pytest.raises(InvalidInputError):
        composite_simpson(f, 0.0, 1.0, 8.0)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError):
        gauss_legendre(f, 0.0, 1.0, 0)
    with pytest.raises(InvalidInputError):
        gauss_legendre(f, 0.0, 1.0, 6.0)  # type: ignore[arg-type]
