from .linear_systems import (
    LinearSolveResult,
    gauss_seidel_solve,
    jacobi_solve,
    sor_solve,
)
from .quadrature import composite_simpson, composite_trapezoid, gauss_legendre

__all__ = [
    "LinearSolveResult",
    "composite_simpson",
    "composite_trapezoid",
    "gauss_legendre",
    "gauss_seidel_solve",
    "jacobi_solve",
    "sor_solve",
]
