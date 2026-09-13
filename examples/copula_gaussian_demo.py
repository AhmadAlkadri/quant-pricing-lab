from __future__ import annotations

import numpy as np

from qpl.dependence.copulas import (
    empirical_kendall_tau,
    empirical_spearman_rho,
    gaussian_copula_kendall_tau,
    gaussian_copula_sample,
    gaussian_copula_spearman_rho,
)


def main() -> None:
    n_samples = 8_000
    rho = 0.5
    threshold = 0.95

    samples = gaussian_copula_sample(n_samples, rho, seed=123)
    independent = gaussian_copula_sample(n_samples, 0.0, seed=123)

    u1 = samples[:, 0]
    u2 = samples[:, 1]

    tau_emp = empirical_kendall_tau(u1, u2)
    tau_theory = gaussian_copula_kendall_tau(rho)
    rho_emp = empirical_spearman_rho(u1, u2)
    rho_theory = gaussian_copula_spearman_rho(rho)

    coex_corr = float(np.mean((samples[:, 0] > threshold) & (samples[:, 1] > threshold)))
    coex_ind = float(np.mean((independent[:, 0] > threshold) & (independent[:, 1] > threshold)))

    print("example=copula_gaussian_demo")
    print(f"n_samples={n_samples}")
    print(f"tau_emp={tau_emp:.4f} tau_theory={tau_theory:.4f}")
    print(f"spearman_emp={rho_emp:.4f} spearman_theory={rho_theory:.4f}")
    print(f"coexceedance_corr={coex_corr:.6f}")
    print(f"coexceedance_ind={coex_ind:.6f}")


if __name__ == "__main__":
    main()
