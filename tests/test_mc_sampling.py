import math

import numpy as np
import pytest

from qpl.engines.mc.sampling import (
    inverse_exponential,
    lcg_uniform,
    sample_beta22_accept_reject,
    sample_exponential_inverse,
    stratified_uniform,
)
from qpl.exceptions import InvalidInputError


def test_lcg_uniform_deterministic_known_prefix() -> None:
    a = lcg_uniform(5, seed=1)
    b = lcg_uniform(5, seed=1)

    expected_states = np.array([16_807, 282_475_249, 1_622_650_073, 984_943_658, 1_144_108_930])
    expected = expected_states / 2_147_483_647

    assert np.allclose(a, expected, rtol=0.0, atol=1e-15)
    assert np.array_equal(a, b)
    assert np.all((a > 0.0) & (a < 1.0))


def test_lcg_uniform_mean_and_variance_sanity() -> None:
    u = lcg_uniform(20_000, seed=12_345)
    assert abs(float(np.mean(u)) - 0.5) < 0.02
    assert abs(float(np.var(u)) - (1.0 / 12.0)) < 0.01


def test_stratified_uniform_structure_and_mean() -> None:
    n = 1_000
    u = stratified_uniform(n, seed=7, shuffle=False)
    indices = np.floor(u * n).astype(int)

    assert np.array_equal(indices, np.arange(n))
    assert abs(float(np.mean(u)) - 0.5) < 0.01


def test_stratified_sampling_reduces_estimator_variance() -> None:
    n = 128
    plain_estimates = []
    strat_estimates = []
    for seed in range(100, 300):
        rng = np.random.default_rng(seed)
        plain = float(np.mean(np.exp(rng.random(n))))
        strat = float(np.mean(np.exp(stratified_uniform(n, seed=seed, shuffle=True))))
        plain_estimates.append(plain)
        strat_estimates.append(strat)

    plain_var = float(np.var(plain_estimates, ddof=1))
    strat_var = float(np.var(strat_estimates, ddof=1))
    assert strat_var < plain_var


def test_inverse_exponential_and_sampling_moments() -> None:
    x = inverse_exponential(np.array([0.0, 0.5]), rate=2.0)
    assert isinstance(x, np.ndarray)
    assert x[0] == pytest.approx(0.0)
    assert x[1] == pytest.approx(math.log(2.0) / 2.0)

    scalar = inverse_exponential(0.25, rate=2.0)
    assert isinstance(scalar, float)
    assert scalar == pytest.approx(-math.log(0.75) / 2.0)

    samples = sample_exponential_inverse(50_000, rate=2.0, seed=123)
    assert abs(float(np.mean(samples)) - 0.5) < 0.02
    assert abs(float(np.var(samples)) - 0.25) < 0.03


def test_beta22_accept_reject_moments_and_acceptance() -> None:
    samples, acceptance_rate = sample_beta22_accept_reject(50_000, seed=42)
    assert samples.shape == (50_000,)
    assert np.all((samples >= 0.0) & (samples <= 1.0))

    assert abs(float(np.mean(samples)) - 0.5) < 0.01
    assert abs(float(np.var(samples)) - 0.05) < 0.01
    assert acceptance_rate == pytest.approx(2.0 / 3.0, abs=0.05)


def test_sampling_validation_errors() -> None:
    with pytest.raises(InvalidInputError):
        lcg_uniform(-1)
    with pytest.raises(InvalidInputError):
        lcg_uniform(1, seed=0, increment=0)
    with pytest.raises(InvalidInputError):
        stratified_uniform(0)
    with pytest.raises(InvalidInputError):
        inverse_exponential(np.array([1.0]))
    with pytest.raises(InvalidInputError):
        inverse_exponential(0.5, rate=0.0)
    with pytest.raises(InvalidInputError):
        sample_exponential_inverse(-1)
    with pytest.raises(InvalidInputError):
        sample_beta22_accept_reject(-1)
