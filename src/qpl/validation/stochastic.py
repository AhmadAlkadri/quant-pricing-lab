"""Strong and weak error estimators for a discretised stochastic process.

Two different questions are asked of a scheme that approximates an SDE, and
they have different answers (Kloeden & Platen 1992, chapters 9-10; Glasserman
2003, section 6.1):

- **strong** error, ``E|X_T^h - X_T|``: how far the *path* the scheme produces
  is from the path the same Brownian motion actually drives.  It is only
  defined when the approximation and the reference share their driving
  increments, so `strong_error` takes two aligned samples and never draws
  anything itself.
- **weak** error, ``|E f(X_T^h) - E f(X_T)|``: how far the *law* is, read
  through one test function.  The paths need not be coupled -- but coupling
  them anyway is what makes the number measurable, because then the estimator
  is the mean of a *difference* whose standard deviation is ``O(h^{1/2})``
  instead of the difference of two means whose standard deviations are
  ``O(1)``.

Both estimators report their own standard error, which is the noise floor a
fitted convergence order must sit above: an error level that is not several
standard errors clear of zero carries no information about the order, and
`fit_convergence_order` will happily fit a slope through noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..exceptions import InvalidInputError

__all__ = ["ErrorEstimate", "strong_error", "weak_error"]


@dataclass(frozen=True, eq=False)
class ErrorEstimate:
    """A Monte Carlo error measurement with its own sampling error.

    Parameters
    ----------
    value
        The estimate.  Non-negative for `strong_error`; **signed** for
        `weak_error`, because the sign of a discretisation bias is part of the
        finding.
    stderr
        Standard error of ``value`` over the sample that produced it.  This is
        the noise floor: a convergence study must keep ``abs(value)`` well
        above it at the coarsest *and* finest level used.
    n_samples
        Paths the estimate was computed from.
    """

    value: float
    stderr: float
    n_samples: int

    @property
    def z(self) -> float:
        """``value / stderr``: how many standard errors the estimate is from 0."""
        if self.stderr == 0.0:
            return math.inf if self.value != 0.0 else 0.0
        return self.value / self.stderr


def _aligned(approx: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(approx, dtype=float)
    b = np.asarray(reference, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise InvalidInputError("samples must be 1D arrays")
    if a.shape != b.shape:
        raise InvalidInputError("samples must have the same length (they must be coupled)")
    if a.size < 2:
        raise InvalidInputError("need at least 2 samples for a ddof=1 standard error")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise InvalidInputError("samples must be finite")
    return a, b


def strong_error(approx: np.ndarray, reference: np.ndarray) -> ErrorEstimate:
    """``E|X^h_T - X_T|`` estimated from coupled terminal samples.

    The two arrays must be path-aligned: ``approx[k]`` and ``reference[k]``
    have to be driven by the same Brownian path, which for a coarse grid means
    increments summed from the fine ones (see
    `qpl.engines.mc.sde.coarsen_normals`).  Differencing two independently
    seeded runs instead measures the spread of the *law*, which does not shrink
    with ``h`` at all; this function cannot detect that mistake, so the caller
    has to get the coupling right.

    Returns the sample mean of the absolute differences with the ``ddof=1``
    standard error of that mean.
    """
    a, b = _aligned(approx, reference)
    gaps = np.abs(a - b)
    return ErrorEstimate(
        value=float(np.mean(gaps)),
        stderr=float(np.std(gaps, ddof=1) / math.sqrt(gaps.size)),
        n_samples=int(gaps.size),
    )


def weak_error(
    approx_values: np.ndarray,
    *,
    reference_values: np.ndarray | None = None,
    reference_mean: float | None = None,
) -> ErrorEstimate:
    """``E f(X^h_T) - E f(X_T)``, signed, with its standard error.

    Exactly one of ``reference_values`` and ``reference_mean``:

    - ``reference_values``: the test function evaluated on a **coupled**
      reference sample.  The estimator is then ``mean(f_approx - f_ref)``,
      which is unbiased for the weak error and whose standard error is set by
      the spread of the *difference*.  Under a common Brownian path that
      spread is ``O(h^{1/2})`` while the payoffs themselves are ``O(1)``, so
      this form is one to three decimal orders quieter than the alternative at
      the same path count.  This is the common-random-numbers form.
    - ``reference_mean``: a known exact value of ``E f(X_T)``.  Then the
      estimator is ``mean(f_approx) - reference_mean`` and its standard error
      is the full ``std(f_approx)/sqrt(N)``, which is usually far larger than
      the bias being measured -- use this form only when the exact mean is
      known *and* the noise floor still clears the finest level.
    """
    if (reference_values is None) == (reference_mean is None):
        raise InvalidInputError(
            "pass exactly one of reference_values or reference_mean"
        )
    if reference_values is not None:
        a, b = _aligned(approx_values, reference_values)
        gaps = a - b
        return ErrorEstimate(
            value=float(np.mean(gaps)),
            stderr=float(np.std(gaps, ddof=1) / math.sqrt(gaps.size)),
            n_samples=int(gaps.size),
        )

    a = np.asarray(approx_values, dtype=float)
    if a.ndim != 1:
        raise InvalidInputError("samples must be 1D arrays")
    if a.size < 2:
        raise InvalidInputError("need at least 2 samples for a ddof=1 standard error")
    if not np.all(np.isfinite(a)):
        raise InvalidInputError("samples must be finite")
    if reference_mean is None or not math.isfinite(reference_mean):
        raise InvalidInputError("reference_mean must be finite")
    return ErrorEstimate(
        value=float(np.mean(a)) - float(reference_mean),
        stderr=float(np.std(a, ddof=1) / math.sqrt(a.size)),
        n_samples=int(a.size),
    )
