from __future__ import annotations

import numpy as np

from ...exceptions import InvalidInputError


def lcg_uniform(
    n: int,
    *,
    seed: int = 123,
    modulus: int = 2_147_483_647,
    multiplier: int = 16_807,
    increment: int = 0,
) -> np.ndarray:
    """Generate U(0,1) samples using a linear congruential generator."""
    if n < 0:
        raise InvalidInputError("n must be >= 0")
    if modulus <= 1:
        raise InvalidInputError("modulus must be > 1")
    if multiplier <= 0 or multiplier >= modulus:
        raise InvalidInputError("multiplier must satisfy 0 < multiplier < modulus")
    if increment < 0 or increment >= modulus:
        raise InvalidInputError("increment must satisfy 0 <= increment < modulus")
    if seed < 0 or seed >= modulus:
        raise InvalidInputError("seed must satisfy 0 <= seed < modulus")
    if increment == 0 and seed == 0 and n > 0:
        raise InvalidInputError("seed cannot be 0 when increment=0")

    state = seed
    out = np.empty(n, dtype=float)
    for i in range(n):
        state = (multiplier * state + increment) % modulus
        out[i] = state / modulus
    return out


def stratified_uniform(n: int, *, seed: int = 123, shuffle: bool = True) -> np.ndarray:
    """Generate one uniform sample per stratum over [0, 1)."""
    if n <= 0:
        raise InvalidInputError("n must be > 0")

    rng = np.random.default_rng(seed)
    samples = (np.arange(n, dtype=float) + rng.random(n)) / n
    if shuffle:
        rng.shuffle(samples)
    return samples


def inverse_exponential(u: np.ndarray | float, *, rate: float = 1.0) -> np.ndarray | float:
    """Inverse-CDF transform for the exponential distribution."""
    if rate <= 0.0:
        raise InvalidInputError("rate must be > 0")

    arr = np.asarray(u, dtype=float)
    if np.any(arr < 0.0) or np.any(arr >= 1.0):
        raise InvalidInputError("u values must satisfy 0 <= u < 1")

    out = -np.log1p(-arr) / rate
    return float(out) if np.isscalar(u) else out


def sample_exponential_inverse(n: int, *, rate: float = 1.0, seed: int = 123) -> np.ndarray:
    """Sample an exponential distribution using inverse-CDF sampling."""
    if n < 0:
        raise InvalidInputError("n must be >= 0")
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    return np.asarray(inverse_exponential(u, rate=rate))


def sample_beta22_accept_reject(n: int, *, seed: int = 123) -> tuple[np.ndarray, float]:
    """Sample Beta(2,2) on [0,1] via accept-reject with Uniform proposal."""
    if n < 0:
        raise InvalidInputError("n must be >= 0")
    if n == 0:
        return np.empty(0, dtype=float), 0.0

    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    n_accepted = 0
    n_draws = 0

    # For Beta(2,2), f(x)=6x(1-x), sup_x f(x)=1.5 over [0,1].
    while n_accepted < n:
        remaining = n - n_accepted
        batch = max(remaining * 2, 128)
        x = rng.random(batch)
        y = rng.random(batch)
        keep = y <= 4.0 * x * (1.0 - x)
        kept = x[keep]
        if kept.size:
            accepted.append(kept)
            n_accepted += kept.size
        n_draws += batch

    samples = np.concatenate(accepted)[:n]
    acceptance_rate = n_accepted / n_draws
    return samples, float(acceptance_rate)
