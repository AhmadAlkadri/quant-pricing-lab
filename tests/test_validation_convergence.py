from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qpl.exceptions import InvalidInputError
from qpl.validation import (
    BenchmarkRow,
    ConvergenceFit,
    ErrorEstimate,
    EvidenceClass,
    fit_convergence_order,
    refinement_errors,
    strong_error,
    weak_error,
)


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_exact_power_law_recovers_order(p: float) -> None:
    # err = C h^p with no noise: log(err) is exactly linear in log(h), so
    # least squares must recover p and the intercept to machine precision.
    c = 3.7
    h = np.array([1.0 / n for n in (10, 20, 40, 80, 160)])
    err = c * h**p

    fit = fit_convergence_order(h, err)

    assert isinstance(fit, ConvergenceFit)
    assert fit.order == pytest.approx(p, abs=1e-12)
    assert fit.log_constant == pytest.approx(math.log(c), abs=1e-12)
    assert fit.residual == pytest.approx(0.0, abs=1e-12)
    assert fit.n_points == 5
    assert fit.predict(h[0]) == pytest.approx(err[0], rel=1e-12)


def test_order_is_sign_consistent_for_increasing_h() -> None:
    # h ordered increasing instead of decreasing: the fit is orientation-free.
    h = np.array([1.0 / n for n in (160, 80, 40, 20, 10)])
    err = 2.0 * h**2
    assert fit_convergence_order(h, err).order == pytest.approx(2.0, abs=1e-12)


def test_noisy_data_has_positive_residual() -> None:
    h = np.array([1.0 / n for n in (10, 20, 40, 80, 160)])
    clean = 1.0 * h**2
    perturbed = clean * np.array([1.0, 1.4, 0.7, 1.3, 0.8])

    clean_fit = fit_convergence_order(h, clean)
    noisy_fit = fit_convergence_order(h, perturbed)

    assert clean_fit.residual == pytest.approx(0.0, abs=1e-12)
    assert noisy_fit.residual > 1e-2


def test_fit_inputs_are_read_only_copies() -> None:
    h = np.array([1.0 / n for n in (10, 20, 40)])
    err = h**2
    fit = fit_convergence_order(h, err)
    assert not fit.h.flags.writeable
    assert not fit.err.flags.writeable
    with pytest.raises(ValueError):
        fit.h[0] = 1.0


@pytest.mark.parametrize(
    "h, err",
    [
        ([1.0, 0.5], [1.0, 0.25]),  # fewer than 3 points
        ([1.0, 0.5, 0.25], [1.0, 0.25]),  # mismatched lengths
        ([1.0, 0.5, 0.0], [1.0, 0.25, 0.0625]),  # non-positive h
        ([1.0, 0.5, 0.25], [1.0, 0.0, 0.0625]),  # non-positive err
        ([1.0, 0.5, math.inf], [1.0, 0.25, 0.0625]),  # non-finite h
        ([1.0, 0.5, 0.25], [1.0, math.nan, 0.0625]),  # non-finite err
        ([1.0, 0.5, 0.5], [1.0, 0.25, 0.25]),  # h not strictly monotone
        ([0.5, 1.0, 0.25], [0.25, 1.0, 0.0625]),  # h not monotone
    ],
)
def test_invalid_inputs_raise(h: list[float], err: list[float]) -> None:
    with pytest.raises(InvalidInputError):
        fit_convergence_order(h, err)


def test_fit_rejects_two_dimensional_input() -> None:
    with pytest.raises(InvalidInputError):
        fit_convergence_order([[1.0, 0.5, 0.25]], [1.0, 0.25, 0.0625])


def test_predict_rejects_bad_h() -> None:
    fit = fit_convergence_order([1.0, 0.5, 0.25], [1.0, 0.25, 0.0625])
    with pytest.raises(InvalidInputError):
        fit.predict(0.0)


def test_refinement_errors_uses_reciprocal_levels() -> None:
    h, err = refinement_errors(lambda n: 5.0 / n**2, (4, 8, 16))
    assert np.allclose(h, [0.25, 0.125, 0.0625])
    assert fit_convergence_order(h, err).order == pytest.approx(2.0, abs=1e-12)


def test_refinement_errors_rejects_non_positive_levels() -> None:
    with pytest.raises(InvalidInputError):
        refinement_errors(lambda n: 1.0 / n, (4, 0))


def test_benchmark_row_is_frozen_and_labelled() -> None:
    row = BenchmarkRow(
        id="demo",
        description="demo row",
        expected=0.0,
        tolerance=1e-10,
        evidence=EvidenceClass.EXACT_IDENTITY,
        source="derived in-repo: tests/test_validation_convergence.py",
    )
    assert row.evidence is EvidenceClass.EXACT_IDENTITY
    assert row.notes == ""
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.expected = 1.0  # type: ignore[misc]


def test_evidence_class_members_are_stable() -> None:
    assert {member.name for member in EvidenceClass} == {
        "EXACT_IDENTITY",
        "CLOSED_FORM",
        "PUBLISHED_BENCHMARK",
        "INDEPENDENT_ENGINE",
        "CONVERGENCE_ORDER",
        "STATISTICAL",
        "NEGATIVE_FINDING",
    }


# ---------------------------------------------------------------------------
# strong_error / weak_error
# ---------------------------------------------------------------------------


def test_strong_error_is_the_mean_absolute_gap_with_its_own_stderr() -> None:
    """EXACT_IDENTITY: the estimator is mean|d| and stderr is std(|d|)/sqrt(N)."""
    rng = np.random.default_rng(1)
    reference = rng.normal(size=5_000)
    approx = reference + rng.normal(size=5_000) * 0.1
    est = strong_error(approx, reference)
    gaps = np.abs(approx - reference)
    assert isinstance(est, ErrorEstimate)
    assert math.isclose(est.value, float(np.mean(gaps)), rel_tol=1e-12)
    assert math.isclose(
        est.stderr, float(np.std(gaps, ddof=1) / math.sqrt(gaps.size)), rel_tol=1e-12
    )
    assert est.n_samples == 5_000
    assert est.z > 0.0


def test_weak_error_is_signed_and_coupling_shrinks_its_noise_floor() -> None:
    """STATISTICAL: the whole reason the coupled form exists.

    The same bias measured two ways on the same sample: differencing against a
    coupled reference and differencing two means. Both are unbiased; the
    coupled one has a standard error smaller by the ratio of the spread of the
    difference to the spread of the sample, which here is about 25x.
    """
    rng = np.random.default_rng(2)
    reference = rng.normal(size=20_000) * 10.0
    bias = -0.05
    approx = reference + bias + rng.normal(size=20_000) * 0.4

    coupled = weak_error(approx, reference_values=reference)
    uncoupled = weak_error(approx, reference_mean=0.0)
    assert coupled.value < 0.0  # signed, and the sign is the finding
    assert abs(coupled.value - bias) <= 4.0 * coupled.stderr
    assert uncoupled.stderr / coupled.stderr > 20.0


def test_error_estimate_z_handles_a_zero_stderr() -> None:
    assert strong_error(np.zeros(4), np.zeros(4)).z == 0.0
    assert math.isinf(weak_error(np.ones(4), reference_mean=0.0).z)


def test_stochastic_error_validation() -> None:
    with pytest.raises(InvalidInputError):
        strong_error(np.zeros(4), np.zeros(5))
    with pytest.raises(InvalidInputError):
        strong_error(np.zeros(1), np.zeros(1))
    with pytest.raises(InvalidInputError):
        strong_error(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(InvalidInputError):
        strong_error(np.array([1.0, np.nan]), np.zeros(2))
    with pytest.raises(InvalidInputError):
        weak_error(np.zeros(4))
    with pytest.raises(InvalidInputError):
        weak_error(np.zeros(4), reference_values=np.zeros(4), reference_mean=0.0)
    with pytest.raises(InvalidInputError):
        weak_error(np.zeros((2, 2)), reference_mean=0.0)
    with pytest.raises(InvalidInputError):
        weak_error(np.zeros(1), reference_mean=0.0)
    with pytest.raises(InvalidInputError):
        weak_error(np.array([1.0, np.nan]), reference_mean=0.0)
    with pytest.raises(InvalidInputError):
        weak_error(np.zeros(4), reference_mean=float("nan"))
