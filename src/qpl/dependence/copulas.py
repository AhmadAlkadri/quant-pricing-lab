from __future__ import annotations

import math

import numpy as np
from scipy.stats import kendalltau, norm, spearmanr

from ..exceptions import InvalidInputError


def _validate_corr(rho: float) -> float:
    if not np.isfinite(rho):
        raise InvalidInputError("corr must be finite")
    if abs(rho) >= 1.0:
        raise InvalidInputError("corr must satisfy |corr| < 1")
    return float(rho)


def _validate_seed(seed: int) -> int:
    if not isinstance(seed, int):
        raise InvalidInputError("seed must be an integer")
    if seed < 0:
        raise InvalidInputError("seed must be >= 0")
    return seed


def _validate_pseudo_observations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise InvalidInputError("x and y must be one-dimensional arrays")
    if x_arr.size != y_arr.size:
        raise InvalidInputError("x and y must have equal length")
    if x_arr.size < 2:
        raise InvalidInputError("x and y must contain at least 2 observations")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise InvalidInputError("x and y must contain finite values")
    return x_arr, y_arr


def gaussian_copula_sample(n_samples: int, corr: float, *, seed: int = 123) -> np.ndarray:
    """Sample bivariate uniforms from a Gaussian copula with scalar correlation."""
    if not isinstance(n_samples, int) or n_samples < 1:
        raise InvalidInputError("n_samples must be a positive integer")
    rho = _validate_corr(corr)
    seed_i = _validate_seed(seed)

    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    rng = np.random.default_rng(seed_i)
    z = rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=n_samples, check_valid="raise")
    u = norm.cdf(z)
    return np.asarray(u, dtype=float)


def empirical_kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate Kendall's tau from paired pseudo-observations."""
    x_arr, y_arr = _validate_pseudo_observations(x, y)
    stat = kendalltau(x_arr, y_arr, method="auto").statistic
    if stat is None or not np.isfinite(stat):
        raise InvalidInputError("kendall tau could not be computed for provided inputs")
    return float(stat)


def empirical_spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate Spearman's rho from paired pseudo-observations."""
    x_arr, y_arr = _validate_pseudo_observations(x, y)
    stat = spearmanr(x_arr, y_arr).statistic
    if stat is None or not np.isfinite(stat):
        raise InvalidInputError("spearman rho could not be computed for provided inputs")
    return float(stat)


def gaussian_copula_kendall_tau(rho: float) -> float:
    """Theoretical Kendall's tau for a Gaussian copula."""
    rho_f = _validate_corr(rho)
    return float((2.0 / math.pi) * math.asin(rho_f))


def gaussian_copula_spearman_rho(rho: float) -> float:
    """Theoretical Spearman's rho for a Gaussian copula."""
    rho_f = _validate_corr(rho)
    return float((6.0 / math.pi) * math.asin(rho_f / 2.0))
