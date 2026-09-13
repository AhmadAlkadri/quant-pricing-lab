from __future__ import annotations

import numpy as np
import pytest

from qpl.exceptions import InvalidInputError
from qpl.transforms.laplace import (
    inverse_laplace_grid_stehfest,
    inverse_laplace_stehfest,
    stehfest_coefficients,
)


def test_stehfest_coefficients_length_and_finiteness() -> None:
    coeffs = stehfest_coefficients(10)
    assert coeffs.shape == (10,)
    assert np.all(np.isfinite(coeffs))


def test_inverse_laplace_recovers_exp_decay() -> None:
    a = 0.75
    t_grid = np.linspace(0.2, 3.0, 16)

    def transform(s: float) -> float:
        return 1.0 / (s + a)

    approx = inverse_laplace_grid_stehfest(transform, t_grid, n_terms=10)
    truth = np.exp(-a * t_grid)

    max_err = np.max(np.abs(approx - truth))
    assert max_err <= 5e-3
    assert np.all(np.diff(approx) <= 2e-3)


def test_inverse_laplace_recovers_linear_function() -> None:
    t_grid = np.linspace(0.2, 2.0, 12)

    def transform(s: float) -> float:
        return 1.0 / (s * s)

    approx = inverse_laplace_grid_stehfest(transform, t_grid, n_terms=10)
    max_err = np.max(np.abs(approx - t_grid))
    assert max_err <= 1e-2


def test_inverse_laplace_scalar_entry_point() -> None:
    t = 0.9

    def transform(s: float) -> float:
        return 1.0 / (s + 2.0)

    approx = inverse_laplace_stehfest(transform, t, n_terms=10)
    truth = np.exp(-2.0 * t)
    assert abs(approx - truth) <= 5e-3


def test_grid_and_scalar_inversion_agree() -> None:
    t_grid = np.array([0.4, 0.8, 1.2, 1.6], dtype=float)

    def transform(s: float) -> float:
        return 1.0 / (s + 0.5)

    grid_vals = inverse_laplace_grid_stehfest(transform, t_grid, n_terms=10)
    scalar_vals = np.array(
        [inverse_laplace_stehfest(transform, float(t), n_terms=10) for t in t_grid],
        dtype=float,
    )

    assert np.allclose(grid_vals, scalar_vals, rtol=0.0, atol=1e-12)


def test_laplace_input_validation() -> None:
    def transform(s: float) -> float:
        return 1.0 / (s + 1.0)

    with pytest.raises(InvalidInputError):
        stehfest_coefficients(9)
    with pytest.raises(InvalidInputError):
        stehfest_coefficients(20)
    with pytest.raises(InvalidInputError):
        inverse_laplace_stehfest(transform, 0.0, n_terms=10)
    with pytest.raises(InvalidInputError):
        inverse_laplace_stehfest(transform, 1.0, n_terms=7)
    with pytest.raises(InvalidInputError):
        inverse_laplace_grid_stehfest(transform, np.array([[0.1, 0.2]]), n_terms=10)
    with pytest.raises(InvalidInputError):
        inverse_laplace_grid_stehfest(transform, np.array([]), n_terms=10)
    with pytest.raises(InvalidInputError):
        inverse_laplace_grid_stehfest(transform, np.array([0.1, -0.2]), n_terms=10)

    def bad_transform(_s: float) -> np.ndarray:
        return np.array([1.0, 2.0])

    with pytest.raises(InvalidInputError):
        inverse_laplace_stehfest(bad_transform, 1.0, n_terms=10)
