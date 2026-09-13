from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..exceptions import InvalidInputError


def _validate_interval(a: float, b: float) -> tuple[float, float]:
    if not np.isfinite(a) or not np.isfinite(b):
        raise InvalidInputError("integration bounds must be finite")
    if b <= a:
        raise InvalidInputError("require b > a")
    return float(a), float(b)


def _validate_callable(f: Callable[[np.ndarray], np.ndarray | float]) -> None:
    if not callable(f):
        raise InvalidInputError("f must be callable")


def _validate_positive_int(name: str, value: int, *, minimum: int) -> int:
    if not isinstance(value, int) or value < minimum:
        raise InvalidInputError(f"{name} must be an integer >= {minimum}")
    return value


def _evaluate_on_grid(f: Callable[[np.ndarray], np.ndarray | float], x: np.ndarray) -> np.ndarray:
    y = np.asarray(f(x), dtype=float)
    if y.ndim == 0:
        y = np.full_like(x, float(y))
    if y.shape != x.shape:
        raise InvalidInputError("f must return either a scalar or an array matching the input shape")
    if not np.all(np.isfinite(y)):
        raise InvalidInputError("f evaluated on grid must be finite")
    return y


def composite_trapezoid(
    f: Callable[[np.ndarray], np.ndarray | float],
    a: float,
    b: float,
    n_intervals: int,
) -> float:
    """Composite trapezoidal-rule integral approximation on ``[a, b]``."""
    _validate_callable(f)
    a_f, b_f = _validate_interval(a, b)
    n = _validate_positive_int("n_intervals", n_intervals, minimum=1)

    x = np.linspace(a_f, b_f, n + 1, dtype=float)
    y = _evaluate_on_grid(f, x)
    h = (b_f - a_f) / n
    return float(h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]))


def composite_simpson(
    f: Callable[[np.ndarray], np.ndarray | float],
    a: float,
    b: float,
    n_intervals: int,
) -> float:
    """Composite Simpson-rule integral approximation on ``[a, b]``."""
    _validate_callable(f)
    a_f, b_f = _validate_interval(a, b)
    n = _validate_positive_int("n_intervals", n_intervals, minimum=2)
    if n % 2 != 0:
        raise InvalidInputError("n_intervals must be even and >= 2 for Simpson's rule")

    x = np.linspace(a_f, b_f, n + 1, dtype=float)
    y = _evaluate_on_grid(f, x)

    h = (b_f - a_f) / n
    odd_sum = np.sum(y[1:-1:2])
    even_sum = np.sum(y[2:-1:2])
    return float((h / 3.0) * (y[0] + y[-1] + 4.0 * odd_sum + 2.0 * even_sum))


def gauss_legendre(
    f: Callable[[np.ndarray], np.ndarray | float],
    a: float,
    b: float,
    n_nodes: int,
) -> float:
    """Gauss-Legendre quadrature on ``[a, b]`` with ``n_nodes`` nodes."""
    _validate_callable(f)
    a_f, b_f = _validate_interval(a, b)
    n = _validate_positive_int("n_nodes", n_nodes, minimum=1)

    nodes, weights = np.polynomial.legendre.leggauss(n)
    # affine transform from [-1, 1] to [a, b]
    x = 0.5 * (b_f - a_f) * nodes + 0.5 * (b_f + a_f)
    y = _evaluate_on_grid(f, x)

    return float(0.5 * (b_f - a_f) * np.sum(weights * y))
