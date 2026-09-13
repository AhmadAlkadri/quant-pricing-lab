"""Row-set hygiene for the sixth id space, `qpl.cases.sde_discretization`.

The measurements themselves are in `tests/test_mc_sde.py` and
`tests/test_sde_convergence.py`; this module checks only that the rows are
well-formed, that the id space is disjoint from the other five, and that the
study specifications say what the prose around them claims.
"""

from __future__ import annotations

import math

from qpl.cases import (
    ALL_SDE_CASES,
    CIR_LEVELS,
    GBM_LEVELS,
    SDE_CIR_CALL_CASES,
    SDE_CIR_MEAN_CASES,
    SDE_NEGATIVITY_CASES,
    SDE_PRICE_BIAS_CASES,
    SDE_STRONG_CASES,
    SDE_WEAK_CALL_CASES,
    SDE_WEAK_IDENTITY_CASES,
)
from qpl.cases.sde_discretization import CIR_FELLER_OK, CIR_FELLER_VIOLATED
from qpl.validation import EvidenceClass


def test_every_sde_row_states_its_evidence_and_source() -> None:
    """The metadata contract, and the sixth id space's disjointness.

    Six id spaces now: European, American, digital, variance-reduction, Asian
    and SDE discretisation. A duplicate id across two modules would make one
    row's failure report the other's description, so they are asserted
    pairwise disjoint the moment a new one appears.
    """
    from qpl.cases import (
        ALL_AMERICAN_CASES,
        ALL_ASIAN_CASES,
        ALL_CASES,
        ALL_DIGITAL_CASES,
        MC_VARIANCE_REDUCTION_CASES,
    )

    ids = [case.row.id for case in ALL_SDE_CASES]
    assert len(set(ids)) == len(ids)
    others = {
        case.row.id
        for case in (
            *ALL_CASES,
            *ALL_AMERICAN_CASES,
            *ALL_DIGITAL_CASES,
            *MC_VARIANCE_REDUCTION_CASES,
            *ALL_ASIAN_CASES,
        )
    }
    assert others.isdisjoint(ids)

    for case in ALL_SDE_CASES:
        row = case.row
        assert isinstance(row.evidence, EvidenceClass)
        assert row.description.strip()
        assert row.source.strip()
        assert row.notes.strip()
        assert row.tolerance > 0.0
        assert case.study in {"gbm", "gbm_price", "cir"}
        assert case.scheme in {"euler", "milstein", "exact"}
        assert case.truncation in {"none", "full"}


def test_the_row_selectors_partition_the_order_rows() -> None:
    """Every order row is reachable through exactly one selector."""
    selected = [
        *SDE_STRONG_CASES,
        *SDE_WEAK_IDENTITY_CASES,
        *SDE_WEAK_CALL_CASES,
        *SDE_PRICE_BIAS_CASES,
        *SDE_CIR_MEAN_CASES,
        *SDE_CIR_CALL_CASES,
    ]
    order_rows = [c for c in ALL_SDE_CASES if c not in SDE_NEGATIVITY_CASES]
    assert sorted(c.row.id for c in selected) == sorted(c.row.id for c in order_rows)
    assert len(selected) == 10
    assert len(SDE_NEGATIVITY_CASES) == 4

    for case in selected:
        assert case.row.evidence is EvidenceClass.CONVERGENCE_ORDER
    for case in SDE_NEGATIVITY_CASES:
        assert case.row.evidence is EvidenceClass.NEGATIVE_FINDING


def test_the_feller_number_is_the_feller_condition() -> None:
    """CLOSED_FORM: ``4 kappa theta / xi^2 >= 2`` is ``2 kappa theta >= xi^2``."""
    for spec, expected in ((CIR_FELLER_OK, 8.0), (CIR_FELLER_VIOLATED, 0.08)):
        assert math.isclose(spec.feller_number, expected, rel_tol=1e-12)
        assert spec.feller_satisfied == (
            2.0 * spec.kappa * spec.theta >= spec.xi * spec.xi
        )
    assert CIR_FELLER_OK.feller_satisfied
    assert not CIR_FELLER_VIOLATED.feller_satisfied
    # v0 = theta on purpose: the exact E[v_T] is then theta for every T, so any
    # measured bias in the mean is the truncation's own.
    assert CIR_FELLER_OK.v0 == CIR_FELLER_OK.theta
    assert CIR_FELLER_VIOLATED.v0 == CIR_FELLER_VIOLATED.theta


def test_the_refinement_ladders_are_nested_powers_of_two() -> None:
    """`coarsen_normals` sums whole blocks, which needs divisibility."""
    for levels in (GBM_LEVELS, CIR_LEVELS):
        finest = max(levels)
        assert all(finest % n == 0 for n in levels)
        assert list(levels) == sorted(levels)
        assert len(levels) >= 3  # fit_convergence_order needs three points
