from __future__ import annotations

import numpy as np

from qpl.transforms.laplace import inverse_laplace_grid_stehfest


def main() -> None:
    a = 0.75
    t_grid = np.linspace(0.2, 3.0, 24)

    def transform(s: float) -> float:
        return 1.0 / (s + a)

    approx = inverse_laplace_grid_stehfest(transform, t_grid, n_terms=10)
    truth = np.exp(-a * t_grid)

    max_abs_error = float(np.max(np.abs(approx - truth)))
    mean_abs_error = float(np.mean(np.abs(approx - truth)))

    print("example=laplace_inversion_demo")
    print(f"n_points={t_grid.size}")
    print(f"max_abs_error={max_abs_error:.6e}")
    print(f"mean_abs_error={mean_abs_error:.6e}")


if __name__ == "__main__":
    main()
