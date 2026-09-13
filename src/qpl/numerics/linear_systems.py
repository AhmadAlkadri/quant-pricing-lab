from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import InvalidInputError


@dataclass(frozen=True)
class LinearSolveResult:
    """Result payload for iterative linear solves."""

    x: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    residual_history: np.ndarray


def _validate_system(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float).reshape(-1)

    if A_arr.ndim != 2:
        raise InvalidInputError("A must be a 2D array")
    n_rows, n_cols = A_arr.shape
    if n_rows != n_cols:
        raise InvalidInputError("A must be square")
    if b_arr.ndim != 1:
        raise InvalidInputError("b must be a 1D array")
    if b_arr.size != n_rows:
        raise InvalidInputError("b size must match A dimension")
    if not np.all(np.isfinite(A_arr)) or not np.all(np.isfinite(b_arr)):
        raise InvalidInputError("A and b must contain finite values")
    if np.any(np.diag(A_arr) == 0.0):
        raise InvalidInputError("A diagonal entries must be non-zero")

    return A_arr, b_arr


def _validate_iterative_params(
    *,
    x0: np.ndarray | None,
    n: int,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    if not np.isfinite(tol) or tol <= 0.0:
        raise InvalidInputError("tol must be > 0")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise InvalidInputError("max_iter must be >= 1")

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1)
        if x.size != n:
            raise InvalidInputError("x0 size must match A dimension")
        if not np.all(np.isfinite(x)):
            raise InvalidInputError("x0 must contain finite values")
        x = x.copy()

    return x


def _residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - A @ x, ord=2))


def jacobi_solve(
    A: np.ndarray,
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 2_000,
) -> LinearSolveResult:
    """Solve ``Ax=b`` with Jacobi iteration."""
    A_arr, b_arr = _validate_system(A, b)
    n = A_arr.shape[0]
    x = _validate_iterative_params(x0=x0, n=n, tol=tol, max_iter=max_iter)

    D = np.diag(A_arr)
    R = A_arr - np.diagflat(D)

    residuals: list[float] = [_residual_norm(A_arr, x, b_arr)]
    converged = residuals[-1] <= tol
    iterations = 0

    while (not converged) and (iterations < max_iter):
        x = (b_arr - R @ x) / D
        iterations += 1
        res = _residual_norm(A_arr, x, b_arr)
        residuals.append(res)
        converged = res <= tol

    return LinearSolveResult(
        x=x,
        residual_norm=residuals[-1],
        iterations=iterations,
        converged=converged,
        residual_history=np.asarray(residuals, dtype=float),
    )


def gauss_seidel_solve(
    A: np.ndarray,
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 2_000,
) -> LinearSolveResult:
    """Solve ``Ax=b`` with Gauss-Seidel iteration."""
    A_arr, b_arr = _validate_system(A, b)
    n = A_arr.shape[0]
    x = _validate_iterative_params(x0=x0, n=n, tol=tol, max_iter=max_iter)

    residuals: list[float] = [_residual_norm(A_arr, x, b_arr)]
    converged = residuals[-1] <= tol
    iterations = 0

    while (not converged) and (iterations < max_iter):
        for i in range(n):
            sigma_before = float(np.dot(A_arr[i, :i], x[:i]))
            sigma_after = float(np.dot(A_arr[i, i + 1 :], x[i + 1 :]))
            x[i] = (b_arr[i] - sigma_before - sigma_after) / A_arr[i, i]

        iterations += 1
        res = _residual_norm(A_arr, x, b_arr)
        residuals.append(res)
        converged = res <= tol

    return LinearSolveResult(
        x=x,
        residual_norm=residuals[-1],
        iterations=iterations,
        converged=converged,
        residual_history=np.asarray(residuals, dtype=float),
    )


def sor_solve(
    A: np.ndarray,
    b: np.ndarray,
    *,
    omega: float,
    x0: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 2_000,
) -> LinearSolveResult:
    """Solve ``Ax=b`` with Successive Over-Relaxation (SOR)."""
    if not np.isfinite(omega) or omega <= 0.0 or omega >= 2.0:
        raise InvalidInputError("omega must satisfy 0 < omega < 2")

    A_arr, b_arr = _validate_system(A, b)
    n = A_arr.shape[0]
    x = _validate_iterative_params(x0=x0, n=n, tol=tol, max_iter=max_iter)

    residuals: list[float] = [_residual_norm(A_arr, x, b_arr)]
    converged = residuals[-1] <= tol
    iterations = 0

    while (not converged) and (iterations < max_iter):
        for i in range(n):
            sigma_before = float(np.dot(A_arr[i, :i], x[:i]))
            sigma_after = float(np.dot(A_arr[i, i + 1 :], x[i + 1 :]))
            gs_value = (b_arr[i] - sigma_before - sigma_after) / A_arr[i, i]
            x[i] = (1.0 - omega) * x[i] + omega * gs_value

        iterations += 1
        res = _residual_norm(A_arr, x, b_arr)
        residuals.append(res)
        converged = res <= tol

    return LinearSolveResult(
        x=x,
        residual_norm=residuals[-1],
        iterations=iterations,
        converged=converged,
        residual_history=np.asarray(residuals, dtype=float),
    )
