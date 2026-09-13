from __future__ import annotations

import numpy as np
import pytest

from qpl.exceptions import InvalidInputError
from qpl.numerics.linear_systems import gauss_seidel_solve, jacobi_solve, sor_solve


def _spd_tridiagonal_system(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    A = np.zeros((n, n), dtype=float)
    np.fill_diagonal(A, 2.0)
    i = np.arange(n - 1)
    A[i, i + 1] = -1.0
    A[i + 1, i] = -1.0
    b = np.ones(n, dtype=float)
    return A, b


def test_iterative_solvers_converge_to_low_residual() -> None:
    A, b = _spd_tridiagonal_system()

    jac = jacobi_solve(A, b, tol=1e-8, max_iter=8_000)
    gs = gauss_seidel_solve(A, b, tol=1e-8, max_iter=8_000)
    sor = sor_solve(A, b, omega=1.1, tol=1e-8, max_iter=8_000)

    assert jac.converged
    assert gs.converged
    assert sor.converged

    assert jac.residual_norm <= 1e-8
    assert gs.residual_norm <= 1e-8
    assert sor.residual_norm <= 1e-8


def test_residual_history_decreases_and_iteration_ordering() -> None:
    A, b = _spd_tridiagonal_system()

    jac = jacobi_solve(A, b, tol=1e-8, max_iter=8_000)
    gs = gauss_seidel_solve(A, b, tol=1e-8, max_iter=8_000)

    assert jac.residual_history[0] > jac.residual_history[-1]
    assert gs.residual_history[0] > gs.residual_history[-1]
    assert np.all(np.isfinite(jac.residual_history))
    assert np.all(np.isfinite(gs.residual_history))

    assert gs.iterations <= jac.iterations


def test_sor_converges_for_stable_omega_range() -> None:
    A, b = _spd_tridiagonal_system()
    for omega in (1.0, 1.1, 1.2):
        res = sor_solve(A, b, omega=omega, tol=1e-8, max_iter=8_000)
        assert res.converged
        assert res.residual_norm <= 1e-8


def test_linear_system_validation_errors() -> None:
    A, b = _spd_tridiagonal_system()

    with pytest.raises(InvalidInputError):
        jacobi_solve(np.ones((3, 2)), np.ones(3))
    with pytest.raises(InvalidInputError):
        gauss_seidel_solve(np.ones((3, 3)), np.ones(2))
    with pytest.raises(InvalidInputError):
        jacobi_solve(A, b, tol=0.0)
    with pytest.raises(InvalidInputError):
        jacobi_solve(A, b, tol=float("nan"))
    with pytest.raises(InvalidInputError):
        jacobi_solve(A, b, max_iter=0)
    with pytest.raises(InvalidInputError):
        jacobi_solve(A, b, max_iter=10.0)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError):
        sor_solve(A, b, omega=0.0)
    with pytest.raises(InvalidInputError):
        sor_solve(A, b, omega=2.0)
    with pytest.raises(InvalidInputError):
        sor_solve(A, b, omega=float("nan"))


def test_nonzero_diagonal_required() -> None:
    A = np.array([[1.0, -1.0], [-1.0, 0.0]])
    b = np.array([1.0, 1.0])
    with pytest.raises(InvalidInputError):
        jacobi_solve(A, b)
