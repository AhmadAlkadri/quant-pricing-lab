from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ..exceptions import InvalidInputError


def _validate_n_terms(n_terms: int) -> int:
    if not isinstance(n_terms, int):
        raise InvalidInputError("n_terms must be an integer")
    if n_terms < 2 or n_terms > 18:
        raise InvalidInputError("n_terms must be in [2, 18] for Stehfest inversion")
    if n_terms % 2 != 0:
        raise InvalidInputError("n_terms must be even for Stehfest inversion")
    return n_terms


def _validate_positive_time(t: float) -> float:
    if not np.isfinite(t):
        raise InvalidInputError("t must be finite")
    if t <= 0.0:
        raise InvalidInputError("t must be strictly positive")
    return float(t)


def _validate_transform_value(value: object) -> float:
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise InvalidInputError("F(s) must return a scalar value")

    scalar = complex(arr.item())
    if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
        raise InvalidInputError("F(s) produced non-finite value during inversion")
    return float(scalar.real)


def _inverse_laplace_stehfest_with_coeffs(
    F: Callable[[float], float | complex],
    t: float,
    coeffs: np.ndarray,
) -> float:
    t_f = _validate_positive_time(t)
    ln2_over_t = math.log(2.0) / t_f
    acc = 0.0

    for k in range(1, coeffs.size + 1):
        s_k = k * ln2_over_t
        acc += coeffs[k - 1] * _validate_transform_value(F(s_k))

    return float(ln2_over_t * acc)


def stehfest_coefficients(n_terms: int) -> np.ndarray:
    """
    Compute Gaver-Stehfest coefficients for even n_terms.

    Notes:
        Numerical stability degrades for very large n_terms in double precision.
        We keep n_terms <= 18 to avoid pathological cancellation in this educational slice.
    """
    n = _validate_n_terms(n_terms)
    n2 = n // 2
    coeffs = np.zeros(n, dtype=float)

    for k in range(1, n + 1):
        j_min = (k + 1) // 2
        j_max = min(k, n2)
        acc = 0.0
        for j in range(j_min, j_max + 1):
            num = (j**n2) * math.factorial(2 * j)
            den = (
                math.factorial(n2 - j)
                * math.factorial(j)
                * math.factorial(j - 1)
                * math.factorial(k - j)
                * math.factorial(2 * j - k)
            )
            acc += num / den
        coeffs[k - 1] = ((-1) ** (k + n2)) * acc
    return coeffs


def inverse_laplace_stehfest(
    F: Callable[[float], float | complex],
    t: float,
    *,
    n_terms: int = 10,
) -> float:
    """Invert Laplace transform at a single positive time point using Stehfest."""
    if not callable(F):
        raise InvalidInputError("F must be callable")
    n = _validate_n_terms(n_terms)
    coeffs = stehfest_coefficients(n)
    return _inverse_laplace_stehfest_with_coeffs(F, t, coeffs)


def inverse_laplace_grid_stehfest(
    F: Callable[[float], float | complex],
    t_grid: np.ndarray,
    *,
    n_terms: int = 10,
) -> np.ndarray:
    """Invert Laplace transform on a 1D positive time grid using Stehfest."""
    if not callable(F):
        raise InvalidInputError("F must be callable")
    n = _validate_n_terms(n_terms)

    t_arr = np.asarray(t_grid, dtype=float)
    if t_arr.ndim != 1:
        raise InvalidInputError("t_grid must be a one-dimensional array")
    if t_arr.size == 0:
        raise InvalidInputError("t_grid must be non-empty")
    if not np.all(np.isfinite(t_arr)):
        raise InvalidInputError("t_grid must contain finite values")
    if np.any(t_arr <= 0.0):
        raise InvalidInputError("t_grid entries must be strictly positive")

    coeffs = stehfest_coefficients(n)
    values = [_inverse_laplace_stehfest_with_coeffs(F, float(t), coeffs) for t in t_arr]
    return np.asarray(values, dtype=float)
