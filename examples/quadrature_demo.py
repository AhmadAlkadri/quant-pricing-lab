from __future__ import annotations

import math

import numpy as np

from qpl.numerics.quadrature import composite_simpson, composite_trapezoid, gauss_legendre


def main() -> None:
    def integrand(x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    exact = math.e - 1.0

    trap = composite_trapezoid(integrand, 0.0, 1.0, n_intervals=128)
    simp = composite_simpson(integrand, 0.0, 1.0, n_intervals=128)
    gauss = gauss_legendre(integrand, 0.0, 1.0, n_nodes=16)

    print("example=quadrature_demo")
    print(f"exact={exact:.10f}")
    print(f"trap={trap:.10f} abs_error={abs(trap - exact):.3e}")
    print(f"simpson={simp:.10f} abs_error={abs(simp - exact):.3e}")
    print(f"gauss={gauss:.10f} abs_error={abs(gauss - exact):.3e}")


if __name__ == "__main__":
    main()
