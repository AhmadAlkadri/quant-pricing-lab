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
