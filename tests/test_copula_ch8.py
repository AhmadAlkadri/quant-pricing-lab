from __future__ import annotations

import numpy as np
import pytest

from qpl.dependence.copulas import (
    empirical_kendall_tau,
    empirical_spearman_rho,
    gaussian_copula_kendall_tau,
    gaussian_copula_sample,
    gaussian_copula_spearman_rho,
)
from qpl.exceptions import InvalidInputError


def test_gaussian_copula_sampling_is_deterministic_with_seed() -> None:
    sample_1 = gaussian_copula_sample(2_000, 0.4, seed=77)
    sample_2 = gaussian_copula_sample(2_000, 0.4, seed=77)
    assert np.array_equal(sample_1, sample_2)


def test_gaussian_copula_uniform_marginal_sanity() -> None:
    samples = gaussian_copula_sample(20_000, 0.3, seed=123)
    x = samples[:, 0]
    y = samples[:, 1]

    expected_mean = 0.5
    expected_var = 1.0 / 12.0
    assert abs(np.mean(x) - expected_mean) <= 8e-3
    assert abs(np.mean(y) - expected_mean) <= 8e-3
    assert abs(np.var(x) - expected_var) <= 8e-3
    assert abs(np.var(y) - expected_var) <= 8e-3


def test_empirical_dependence_matches_theoretical_levels() -> None:
    rho = 0.6
    samples = gaussian_copula_sample(25_000, rho, seed=321)
    x = samples[:, 0]
    y = samples[:, 1]

    tau_emp = empirical_kendall_tau(x, y)
    tau_theory = gaussian_copula_kendall_tau(rho)
    rho_s_emp = empirical_spearman_rho(x, y)
    rho_s_theory = gaussian_copula_spearman_rho(rho)

    assert abs(tau_emp - tau_theory) <= 0.02
    assert abs(rho_s_emp - rho_s_theory) <= 0.02


def test_empirical_tau_increases_with_correlation() -> None:
    rho_grid = [-0.5, 0.0, 0.5]
    taus: list[float] = []
    for rho in rho_grid:
        samples = gaussian_copula_sample(15_000, rho, seed=100 + int((rho + 1.0) * 100))
        taus.append(empirical_kendall_tau(samples[:, 0], samples[:, 1]))

    assert taus[0] < taus[1] < taus[2]


def test_correlated_case_has_higher_joint_upper_tail_coexceedance() -> None:
    n = 20_000
    threshold = 0.95
    correlated = gaussian_copula_sample(n, 0.7, seed=11)
    independent = gaussian_copula_sample(n, 0.0, seed=11)

    corr_tail = np.mean((correlated[:, 0] > threshold) & (correlated[:, 1] > threshold))
    indep_tail = np.mean((independent[:, 0] > threshold) & (independent[:, 1] > threshold))

    assert corr_tail > indep_tail


def test_copula_validation_errors() -> None:
    with pytest.raises(InvalidInputError):
        gaussian_copula_sample(0, 0.2, seed=1)
    with pytest.raises(InvalidInputError):
        gaussian_copula_sample(10, 0.2, seed=-2)
    with pytest.raises(InvalidInputError):
        gaussian_copula_sample(10, 0.2, seed=2.5)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError):
        gaussian_copula_sample(10, 1.0, seed=1)
    with pytest.raises(InvalidInputError):
        gaussian_copula_kendall_tau(-1.0)
    with pytest.raises(InvalidInputError):
        gaussian_copula_spearman_rho(1.2)
    with pytest.raises(InvalidInputError):
        empirical_kendall_tau(np.array([0.1, 0.2]), np.array([0.1]))
    with pytest.raises(InvalidInputError):
        empirical_spearman_rho(np.array([[0.1, 0.2]]), np.array([0.3, 0.4]))


def test_theoretical_dependence_is_monotone_in_rho() -> None:
    rho_grid = np.array([-0.8, -0.2, 0.0, 0.4, 0.8])
    tau_vals = [gaussian_copula_kendall_tau(float(rho)) for rho in rho_grid]
    spearman_vals = [gaussian_copula_spearman_rho(float(rho)) for rho in rho_grid]

    assert all(a < b for a, b in zip(tau_vals, tau_vals[1:]))
    assert all(a < b for a, b in zip(spearman_vals, spearman_vals[1:]))
