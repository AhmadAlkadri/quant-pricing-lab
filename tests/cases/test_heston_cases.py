"""Evaluate the Heston benchmark rows.

Every assertion reads its expected value, tolerance, evidence class and
citation from `qpl.cases.heston`; nothing numeric is hardcoded here. The
detailed studies live in `tests/test_heston_model.py`,
`tests/test_heston_fourier.py` and `tests/test_heston_smile.py`. This file
exists so that the cases layer carries the claims as *data*, including the
three shape claims -- which are predicates rather than numbers and are encoded
as indicator rows (`SHAPE_HOLDS`) so that they still carry an evidence class
and a citation.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import pairwise

import numpy as np
import pytest

from qpl.cases import (
    ALL_HESTON_CASES,
    HESTON_BS_LIMIT_CASES,
    HESTON_BS_LIMIT_ORDER_CASES,
    HESTON_BS_LIMIT_XI,
    HESTON_CALL_TOLERANCE,
    HESTON_CROSS_METHOD_CASES,
    HESTON_LEWIS_SPEC,
    HESTON_MC_CASES,
    HESTON_MC_FELLER_SPEC,
    HESTON_MC_ORDER_LEVELS,
    HESTON_MC_ORDER_PATHS,
    HESTON_MC_PATHS,
    HESTON_MC_REFERENCE_METHOD,
    HESTON_MC_SEED,
    HESTON_PARITY_CASES,
    HESTON_PUBLISHED_CASES,
    HESTON_PUT_TOLERANCE,
    HESTON_SMILE_CASES,
    HESTON_SMILE_MATURITIES,
    HESTON_SMILE_N_TERMS,
    HESTON_SMILE_STRIKES,
    HESTON_SMILE_TRUNCATION_L,
    HESTON_TERM_STRUCTURE_MATURITIES,
    HESTON_TRANSFORM_METHODS,
    SHAPE_HOLDS,
    HestonCase,
    black_scholes_limit_model,
    heston_parity_residual,
    method_config,
)
from qpl.engines.fourier import (
    FourierConfig,
    atm_implied_variance,
    implied_vol,
    smile_skew,
)
from qpl.engines.fourier.lewis import lewis_call
from qpl.engines.mc.heston import QE, simulate_heston
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.mc.sde import uniform_time_grid
from qpl.pricing import price
from qpl.validation import EvidenceClass, fit_convergence_order

SMILE_CFG = FourierConfig(
    truncation_l=HESTON_SMILE_TRUNCATION_L, n_terms=HESTON_SMILE_N_TERMS
)


def _ids(cases: tuple[HestonCase, ...]) -> list[str]:
    return [case.row.id for case in cases]


def _transform_price(spec, method: str = "cos") -> float:
    return price(
        spec.option(), spec.model(), spec.market(), method="fourier",
        cfg=method_config(method),
    ).value


def _lewis_reference(spec) -> float:
    """A Lewis contour integral of the same transform, for the COS rows.

    Not independent evidence about the model -- it reads the same
    characteristic function -- but it has no truncation range, which makes it
    the right reference for a claim about the COS range and the wrong one for a
    claim about the transform. The published rows carry the latter.
    """
    value, _ = lewis_call(
        spec.model(),
        s0=spec.spot,
        strike=spec.strike,
        expiry=spec.expiry,
        rate=spec.rate,
        dividend=spec.dividend,
        limit=800,
        tolerance=1e-13,
    )
    if spec.kind == "call":
        return value
    return (
        value
        - spec.spot * math.exp(-spec.dividend * spec.expiry)
        + spec.strike * math.exp(-spec.rate * spec.expiry)
    )


# --------------------------------------------------------------------------
# The published rows.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", HESTON_PUBLISHED_CASES, ids=_ids(HESTON_PUBLISHED_CASES))
@pytest.mark.parametrize("method", HESTON_TRANSFORM_METHODS)
def test_published_rows(case: HestonCase, method: str) -> None:
    assert case.row.evidence is EvidenceClass.PUBLISHED_BENCHMARK
    assert _transform_price(case.spec, method) == pytest.approx(
        case.row.expected, abs=case.row.tolerance
    )


@pytest.mark.parametrize(
    "case", HESTON_CROSS_METHOD_CASES, ids=_ids(HESTON_CROSS_METHOD_CASES)
)
def test_the_three_contour_methods_agree_far_inside_the_published_resolution(
    case: HestonCase,
) -> None:
    values = [
        _transform_price(case.spec, method)
        for method in ("lewis", "gil_pelaez", "carr_madan")
    ]
    assert max(values) - min(values) == pytest.approx(
        case.row.expected, abs=case.row.tolerance
    )


@pytest.mark.parametrize("case", HESTON_PUBLISHED_CASES, ids=_ids(HESTON_PUBLISHED_CASES))
def test_the_cos_call_and_the_cos_put_carry_different_errors(case: HestonCase) -> None:
    """The slice's most specific finding, carried by the cases layer as data.

    Two tolerances four decimal orders apart, because the two errors are four
    decimal orders apart: the COS call is at 1e-12 and the COS put at 2.7e-08,
    the latter identical at every strike because it is missing left-tail mass
    rather than anything strike-dependent.
    """
    spec = case.spec
    tolerance = (
        HESTON_CALL_TOLERANCE if spec.kind == "call" else HESTON_PUT_TOLERANCE
    )
    assert abs(_transform_price(spec, "cos") - _lewis_reference(spec)) < tolerance


def test_the_lewis_parameter_set_satisfies_the_feller_condition() -> None:
    """Recorded because the slice statement asserted the opposite and then
    computed that it holds: `2 kappa theta = 2.0` against `xi^2 = 1.0`."""
    assert HESTON_LEWIS_SPEC.feller_number == pytest.approx(4.0)
    assert HESTON_LEWIS_SPEC.feller_number >= 2.0
    assert HESTON_LEWIS_SPEC.model().feller_satisfied


# --------------------------------------------------------------------------
# The Black-Scholes limit.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", HESTON_BS_LIMIT_CASES, ids=_ids(HESTON_BS_LIMIT_CASES))
def test_black_scholes_limit_rows(case: HestonCase) -> None:
    spec = case.spec
    assert spec.v0 == spec.theta
    assert _transform_price(spec) == pytest.approx(
        case.row.expected, abs=case.row.tolerance
    )
    # The expected value is the Black-Scholes formula, recomputed rather than
    # read back from the transform.
    assert price(
        spec.option(), black_scholes_limit_model(spec), spec.market()
    ).value == pytest.approx(case.row.expected, abs=1e-12)


@pytest.mark.parametrize(
    "case", HESTON_BS_LIMIT_ORDER_CASES, ids=_ids(HESTON_BS_LIMIT_ORDER_CASES)
)
def test_black_scholes_limit_order_rows(case: HestonCase) -> None:
    assert case.row.evidence is EvidenceClass.CONVERGENCE_ORDER
    assert len(case.specs) == len(HESTON_BS_LIMIT_XI)
    reference = price(
        case.specs[0].option(),
        black_scholes_limit_model(case.specs[0]),
        case.specs[0].market(),
    ).value
    errors = np.array(
        [abs(_transform_price(spec) - reference) for spec in case.specs]
    )
    fit = fit_convergence_order(np.array(HESTON_BS_LIMIT_XI), errors)
    assert fit.order == pytest.approx(case.row.expected, abs=case.row.tolerance)
    assert fit.residual < 0.03


# --------------------------------------------------------------------------
# Parity.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", HESTON_PARITY_CASES, ids=_ids(HESTON_PARITY_CASES))
def test_parity_rows(case: HestonCase) -> None:
    assert case.row.evidence is EvidenceClass.EXACT_IDENTITY
    spec = case.spec
    call = _transform_price(spec.at(kind="call"))
    put = _transform_price(spec.at(kind="put"))
    assert heston_parity_residual(call, put, spec) == pytest.approx(
        case.row.expected, abs=case.row.tolerance
    )


# --------------------------------------------------------------------------
# Smile shape, evaluated as indicators.
# --------------------------------------------------------------------------


def _shape_indicator(case: HestonCase) -> float:
    """Evaluate one shape row and return 1.0 if the statement holds."""
    spec = case.spec
    model = spec.model()
    market = spec.market()

    if case.row.id == "heston_smile_negative_rho_is_decreasing_in_strike":
        assert spec.rho < 0.0
        holds = all(
            all(
                later < earlier
                for earlier, later in pairwise(
                    [
                        implied_vol(
                            model, market, strike=strike, expiry=expiry, cfg=SMILE_CFG
                        )
                        for strike in HESTON_SMILE_STRIKES
                    ]
                )
            )
            for expiry in HESTON_SMILE_MATURITIES
        )
        return float(holds)

    if case.row.id == "heston_smile_positive_rho_flips_the_skew":
        assert spec.rho > 0.0
        return float(
            all(
                smile_skew(model, market, expiry, cfg=SMILE_CFG) > 0.0
                for expiry in HESTON_SMILE_MATURITIES
            )
        )

    if case.row.id == "heston_smile_flattens_with_maturity":
        skews = [
            abs(smile_skew(model, market, expiry, cfg=SMILE_CFG))
            for expiry in HESTON_SMILE_MATURITIES
        ]
        return float(all(later < earlier for earlier, later in pairwise(skews)))

    variances = [
        atm_implied_variance(model, market, expiry, cfg=SMILE_CFG)
        for expiry in HESTON_TERM_STRUCTURE_MATURITIES
    ]
    rising = spec.v0 < spec.theta
    monotone = all(
        (later > earlier) == rising for earlier, later in pairwise(variances)
    )
    starts_at_v0 = abs(variances[0] - spec.v0) < 0.006
    approaches_theta = abs(variances[-1] - spec.theta) < abs(variances[0] - spec.theta)
    return float(monotone and starts_at_v0 and approaches_theta)


@pytest.mark.parametrize("case", HESTON_SMILE_CASES, ids=_ids(HESTON_SMILE_CASES))
def test_smile_shape_rows(case: HestonCase) -> None:
    """Shape claims as indicator rows: `expected = SHAPE_HOLDS`, tolerance 0."""
    assert case.row.expected == SHAPE_HOLDS
    assert case.row.tolerance == 0.0
    assert _shape_indicator(case) == case.row.expected


# --------------------------------------------------------------------------
# Monte Carlo rows (Slice 16).
# --------------------------------------------------------------------------

_MC_REFERENCE_CFG = FourierConfig(method=HESTON_MC_REFERENCE_METHOD)


def _mc_cfg(n_steps: int, *, scheme: str = QE, n_paths: int = HESTON_MC_PATHS) -> MCConfig:
    """The estimator every Monte Carlo row is measured with.

    Antithetic plus conditioning and deliberately no control variate; the
    reason is in `qpl.cases.heston.HESTON_MC_PATHS`.
    """
    return MCConfig(
        n_paths=n_paths,
        n_steps=n_steps,
        seed=HESTON_MC_SEED,
        variance_reduction="antithetic",
        heston_conditional=True,
        heston_scheme=scheme,
    )


@lru_cache(maxsize=None)
def _mc_bias(spec_id: str, scheme: str, n_steps: int, n_paths: int) -> tuple[float, float]:
    """`(bias, stderr)` of the simulated ATM call against the transform price."""
    spec = HESTON_LEWIS_SPEC if spec_id == "reference" else HESTON_MC_FELLER_SPEC
    option, model, market = spec.option(), spec.model(), spec.market()
    reference = price(
        option, model, market, method="fourier", cfg=_MC_REFERENCE_CFG
    ).value
    simulated = price(
        option,
        model,
        market,
        method="mc",
        cfg=_mc_cfg(n_steps, scheme=scheme, n_paths=n_paths),
    )
    return simulated.value - reference, float(simulated.stderr)


@lru_cache(maxsize=None)
def _martingale_defect(*, corrected: bool) -> tuple[float, float]:
    """`(E[e^{-(r-q)T} S_T] - S_0, stderr)` for QE at `dt = 1/4`."""
    spec = HESTON_LEWIS_SPEC
    drift = spec.rate - spec.dividend
    paths = simulate_heston(
        spec.model(),
        s0=spec.spot,
        mu=drift,
        t_grid=uniform_time_grid(spec.expiry, 4),
        n_paths=HESTON_MC_PATHS,
        seed=HESTON_MC_SEED,
        scheme=QE,
        antithetic=True,
        martingale_correction=corrected,
        store_paths=False,
    )
    discounted = math.exp(-drift * spec.expiry) * paths.terminal_spot
    half = HESTON_MC_PATHS // 2
    units = 0.5 * (discounted[:half] + discounted[half:])
    return (
        float(np.mean(units)) - spec.spot,
        float(np.std(units, ddof=1) / math.sqrt(units.size)),
    )


def _mc_row_value(case_id: str) -> float:
    """Evaluate the quantity one Monte Carlo row makes a claim about."""
    if case_id == "heston_mc_qe_bias_dt_quarter":
        return _mc_bias("reference", QE, 4, HESTON_MC_PATHS)[0]
    if case_id == "heston_mc_euler_bias_dt_quarter":
        return _mc_bias("reference", "euler_full_truncation", 4, HESTON_MC_PATHS)[0]
    if case_id == "heston_mc_euler_over_qe_bias_reference_set":
        qe = _mc_bias("reference", QE, 4, HESTON_MC_PATHS)[0]
        euler = _mc_bias("reference", "euler_full_truncation", 4, HESTON_MC_PATHS)[0]
        return abs(euler) / abs(qe)
    if case_id == "heston_mc_euler_over_qe_bias_feller_violated":
        qe = _mc_bias("feller", QE, 4, HESTON_MC_PATHS)[0]
        euler = _mc_bias("feller", "euler_full_truncation", 4, HESTON_MC_PATHS)[0]
        return abs(euler) / abs(qe)
    if case_id == "heston_mc_qe_feller_violated_bias_dt_quarter":
        return _mc_bias("feller", QE, 4, HESTON_MC_PATHS)[0]
    if case_id == "heston_mc_qe_weak_order_resolved_levels":
        return _fitted_bias_order("reference", QE, HESTON_MC_ORDER_PATHS)
    if case_id == "heston_mc_euler_feller_violated_order":
        return _fitted_bias_order(
            "feller", "euler_full_truncation", HESTON_MC_PATHS
        )
    if case_id == "heston_mc_martingale_defect_without_correction":
        return _martingale_defect(corrected=False)[0]
    if case_id == "heston_mc_martingale_defect_with_correction":
        return _martingale_defect(corrected=True)[0]
    if case_id == "heston_mc_conditional_variance_factor":
        return _conditional_variance_factor()
    raise AssertionError(f"no evaluator for row {case_id}")  # pragma: no cover


@lru_cache(maxsize=None)
def _fitted_bias_order(spec_id: str, scheme: str, n_paths: int) -> float:
    steps = np.array([1.0 / n for n in HESTON_MC_ORDER_LEVELS])
    errors = np.array(
        [
            abs(_mc_bias(spec_id, scheme, n, n_paths)[0])
            for n in HESTON_MC_ORDER_LEVELS
        ]
    )
    return fit_convergence_order(steps, errors).order


@lru_cache(maxsize=None)
def _conditional_variance_factor() -> float:
    spec = HESTON_LEWIS_SPEC
    option, model, market = spec.option(), spec.model(), spec.market()
    plain = price(
        option,
        model,
        market,
        method="mc",
        cfg=MCConfig(
            n_paths=HESTON_MC_PATHS,
            n_steps=16,
            seed=HESTON_MC_SEED,
            variance_reduction="antithetic",
            heston_conditional=False,
        ),
    )
    conditional = price(
        option, model, market, method="mc", cfg=_mc_cfg(16)
    )
    return (float(plain.stderr) / float(conditional.stderr)) ** 2


@pytest.mark.parametrize("case", HESTON_MC_CASES, ids=_ids(HESTON_MC_CASES))
def test_monte_carlo_rows(case: HestonCase) -> None:
    """Evaluate one simulation row at the settings its module records.

    Every row is deterministic at `HESTON_MC_SEED`, so a failure means the
    estimator moved, not that a draw was unlucky -- and the tolerances are
    multiples of the measured standard error (or, for the order rows, the
    measured spread across seeds), which the row's `notes` state.
    """
    assert case.row.evidence in {
        EvidenceClass.STATISTICAL,
        EvidenceClass.CONVERGENCE_ORDER,
        EvidenceClass.NEGATIVE_FINDING,
    }
    assert _mc_row_value(case.row.id) == pytest.approx(
        case.row.expected, abs=case.row.tolerance
    )


def test_the_martingale_correction_is_what_removes_the_defect() -> None:
    """The two martingale rows are only meaningful next to each other.

    Evidence class: NEGATIVE_FINDING for the uncorrected drift. The corrected
    defect has to be inside its own sampling error *and* a small fraction of
    the uncorrected one; either alone could be satisfied by a scheme that had
    simply become noisier.
    """
    uncorrected, uncorrected_stderr = _martingale_defect(corrected=False)
    corrected, corrected_stderr = _martingale_defect(corrected=True)
    assert uncorrected > 20.0 * uncorrected_stderr
    assert abs(corrected) < 4.0 * corrected_stderr
    assert abs(corrected) < 0.1 * uncorrected


def test_the_feller_violating_spec_is_the_repositorys_own() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the test data).

    Its variance parameters are `CIR_FELLER_VIOLATED`'s, so the simulation
    slice and the transform slice mean the same regime by "Feller violated".
    """
    assert HESTON_MC_FELLER_SPEC.feller_number == pytest.approx(0.08)
    assert HESTON_LEWIS_SPEC.feller_number == pytest.approx(4.0)


# --------------------------------------------------------------------------
# Meta.
# --------------------------------------------------------------------------


def test_every_row_states_its_evidence_and_its_source() -> None:
    ids = [case.row.id for case in ALL_HESTON_CASES]
    assert len(set(ids)) == len(ids)
    for case in ALL_HESTON_CASES:
        assert isinstance(case.row.evidence, EvidenceClass)
        assert case.row.source.strip()
        assert case.row.description.strip()
        assert case.row.tolerance >= 0.0
        assert case.specs


def test_the_published_rows_are_the_only_numbers_taken_from_outside() -> None:
    """Every other row's source says where it was derived or measured."""
    for case in ALL_HESTON_CASES:
        if case.row.evidence is EvidenceClass.PUBLISHED_BENCHMARK:
            assert "Wilmott" in case.row.source
        else:
            assert any(
                marker in case.row.source
                for marker in ("derived", "Gatheral", "measured")
            ), case.row.id
