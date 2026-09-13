from .linear_systems import (
    LinearSolveResult,
    gauss_seidel_solve,
    jacobi_solve,
    sor_solve,
)
from .quadrature import composite_simpson, composite_trapezoid, gauss_legendre

__all__ = [
    "LinearSolveResult",
    "jacobi_solve",
    "gauss_seidel_solve",
    "sor_solve",
    "composite_trapezoid",
    "composite_simpson",
    "gauss_legendre",
]
