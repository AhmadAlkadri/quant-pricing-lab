"""Evaluate the American Black-Scholes benchmark rows.

Every expected value, tolerance, evidence class and citation is read from
`qpl.cases.american_black_scholes`; the only numbers written in this file are
the ones that describe the *lattice* (step counts, the Bermudan exercise
restriction), not the answers.

The one row this file does not evaluate is
`atm_1y_american_put_reference_vs_quantlib_fd`, whose evidence is an
independent engine behind the optional `[oracle]` extra; it is evaluated in
`tests/oracle/test_american_vs_quantlib.py`.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from qpl.cases import (
    ALL_AMERICAN_CASES,
    AMERICAN_BRACKETED_LIMIT,
    AMERICAN_IDENTITY_CASES,
    AMERICAN_LR_CASES,
    AMERICAN_LR_LEVELS,
    AMERICAN_PREMIUM_CASES,
    AMERICAN_REFERENCE_CASES,
    AMERICAN_REFERENCE_N_STEPS,
    LS2001_BERMUDAN_EXERCISES_PER_YEAR,
    LS2001_CASES,
    LS2001_N_STEPS,
    AmericanBSCase,
)
from qpl.engines.tree import TreeConfig, crr_parameters
from qpl.pricing import price
from qpl.validation import EvidenceClass, fit_convergence_order

_LATTICE_N_STEPS = 500
"""Step count for the identity and premium rows.

Large enough that the premium is resolved to about 1e-03 and small enough that
the whole file stays under a second. The identity rows do not care: they hold
at every `n`.
"""


def _ids(cases: tuple[AmericanBSCase, ...]) -> list[str]:
    return [case.row.id for case in cases]


def _american(case_spec, n_steps: int, scheme: str = "crr") -> float:
    return price(
        case_spec.option(),
        case_spec.model(),
        case_spec.market(),
        method="tree",
        cfg=TreeConfig(n_steps=n_steps, scheme=scheme),  # type: ignore[arg-type]
    ).value


def _european(case_spec, n_steps: int) -> float:
    """The European twin, priced on the *same* lattice.

    Not the closed form: the premium rows are claims about early exercise, and
    comparing a lattice price against a closed form would fold the tree's own
    `O(1/n)` discretisation error into the premium. On the same lattice that
    error cancels, which is what lets the ordering rows use a zero tolerance.
    """
    return price(
        case_spec.european_option(),
        case_spec.model(),
        case_spec.market(),
        method="tree",
        cfg=TreeConfig(n_steps=n_steps),
    ).value


# --------------------------------------------------------------------------
# The Bermudan restriction, applied to the shared CRR lattice.
# --------------------------------------------------------------------------


def _bermudan_put(spec, *, n_steps: int, n_exercise: int) -> float:
    """Price a Bermudan put with `n_exercise` equally spaced exercise dates.

    Same lattice and same continuation step as `qpl.engines.tree.price_american`
    -- the only change is that the Bellman maximum against the intrinsic value
    is taken at every `n_steps / n_exercise`-th level instead of at every
    level. `n_steps` must be divisible by `n_exercise`, so the exercise dates
    land exactly on lattice levels and nothing has to be interpolated.

    This lives in the test rather than in `qpl` on purpose: the package has no
    Bermudan instrument, and adding one to check a citation would be building
    an instrument ahead of the case that justifies it. What it is here for is
    stated in `qpl.cases.american_black_scholes`: the Longstaff-Schwartz
    Table 1 options are exercisable 50 times per year, and this is what that
    sentence means numerically.
    """
    step, remainder = divmod(n_steps, n_exercise)
    assert remainder == 0, (n_steps, n_exercise)

    lattice = crr_parameters(
        sigma=spec.sigma,
        expiry=spec.expiry,
        rate=spec.rate,
        dividend_yield=spec.dividend,
        n_steps=n_steps,
    )
    counts = np.arange(n_steps + 1, dtype=float)
    up_powers = lattice.up**counts
    down_powers = lattice.down**counts

    def spots(level: int) -> np.ndarray:
        return spec.spot * up_powers[: level + 1] * down_powers[level::-1]

    exercise_levels = {n_steps - i * step for i in range(n_exercise)}
    values = np.maximum(spec.strike - spots(n_steps), 0.0)
    for level in range(n_steps - 1, -1, -1):
        values = lattice.discount * (
            lattice.p * values[1:] + (1.0 - lattice.p) * values[:-1]
        )
        if level in exercise_levels:
            values = np.maximum(np.maximum(spec.strike - spots(level), 0.0), values)
    return float(values[0])


def test_bermudan_restriction_brackets_the_two_exercise_styles() -> None:
    """Sanity check on the helper before it is used as evidence.

    More exercise dates cannot be worth less, so

        European <= Bermudan(50) <= American (continuous)

    on the same lattice, and the Bermudan value must move monotonically toward
    the American one as dates are added. Without this, a bug in the helper
    could make the published-benchmark row pass for the wrong reason.
    """
    spec = LS2001_CASES[0].spec
    n_steps = LS2001_N_STEPS
    european = _european(spec, n_steps)
    american = _american(spec, n_steps)

    values = [
        _bermudan_put(spec, n_steps=n_steps, n_exercise=m) for m in (1, 2, 5, 10, 50, 100, 500)
    ]
    assert values[0] == pytest.approx(european, abs=1e-12)
    assert all(b >= a - 1e-12 for a, b in pairwise(values)), values
    assert all(european - 1e-12 <= v <= american + 1e-12 for v in values), values


@pytest.mark.parametrize("case", LS2001_CASES, ids=_ids(LS2001_CASES))
def test_longstaff_schwartz_table1_row1(case: AmericanBSCase) -> None:
    """The one published benchmark in this slice, and what it is not.

    Two rows, two opposite assertions on the same specification:

    - `ls2001_table1_row1_bermudan_50` (PUBLISHED_BENCHMARK): restricting
      exercise to 50 dates per year reproduces the published 4.478 to
      7.8e-05 at `n = 5000`.
    - `ls2001_table1_row1_is_not_a_continuous_american_value`
      (NEGATIVE_FINDING): the continuously-exercisable American put at the
      same specification is 4.486710, which is 8.71e-03 away -- outside the
      2e-03 the slice statement expected, and the assertion is that it *stays*
      outside.

    Together they say something a single tolerance could not: the engine is
    right and the citation is right, and they refer to different instruments.
    """
    spec = case.spec
    assert "Longstaff" in case.row.source

    if case.row.evidence is EvidenceClass.PUBLISHED_BENCHMARK:
        value = _bermudan_put(
            spec,
            n_steps=LS2001_N_STEPS,
            n_exercise=LS2001_BERMUDAN_EXERCISES_PER_YEAR,
        )
        assert value == pytest.approx(case.row.expected, abs=case.row.tolerance), case.row.notes
    else:
        assert case.row.evidence is EvidenceClass.NEGATIVE_FINDING
        value = _american(spec, LS2001_N_STEPS)
        distance = abs(value - case.row.expected)
        assert distance > case.row.tolerance, (value, distance)
        # Pinned so that the size of the gap cannot drift either.
        assert distance == pytest.approx(8.71e-3, abs=5e-5), distance


# --------------------------------------------------------------------------
# Identities and premium bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", AMERICAN_IDENTITY_CASES, ids=_ids(AMERICAN_IDENTITY_CASES))
@pytest.mark.parametrize("n_steps", [7, 101, 500])
def test_american_identity_rows(case: AmericanBSCase, n_steps: int) -> None:
    """Both rows claim a difference of exactly zero, at every `n`."""
    assert case.row.evidence is EvidenceClass.EXACT_IDENTITY
    assert case.row.tolerance == 0.0

    spec = case.spec
    difference = _american(spec, n_steps) - _european(spec, n_steps)
    assert difference == case.row.expected


def test_premium_is_non_negative_row() -> None:
    case = AMERICAN_PREMIUM_CASES[0]
    assert case.row.evidence is EvidenceClass.EXACT_IDENTITY

    worst_violation = 0.0
    for spec in case.specs:
        premium = _american(spec, _LATTICE_N_STEPS) - _european(spec, _LATTICE_N_STEPS)
        worst_violation = max(worst_violation, -premium)
    assert worst_violation <= case.row.expected + case.row.tolerance, case.row.source


def test_premium_is_bounded_by_interest_on_the_strike_row() -> None:
    """Evidence class: CLOSED_FORM. `P_A - P_E <= K (1 - e^{-rT})` for `q = 0`.

    The bound is attained, not merely approached, deep in the money -- where
    the American put is worth `K` and the European put `K e^{-rT} - S` -- so
    the check below also asserts that at least one point comes within 1e-03 of
    it. A bound nothing gets close to would be no evidence that the premium is
    the right size.
    """
    case = AMERICAN_PREMIUM_CASES[1]
    assert case.row.evidence is EvidenceClass.CLOSED_FORM

    ratios = []
    worst_overshoot = 0.0
    for spec in case.specs:
        assert spec.dividend == 0.0, "the bound needs q = 0"
        premium = _american(spec, _LATTICE_N_STEPS) - _european(spec, _LATTICE_N_STEPS)
        bound = spec.strike * (1.0 - math.exp(-spec.rate * spec.expiry))
        worst_overshoot = max(worst_overshoot, premium - bound)
        ratios.append(premium / bound)

    assert worst_overshoot <= case.row.expected + case.row.tolerance, case.row.source
    assert max(ratios) > 0.999, ratios


def test_premium_increases_with_strike_row() -> None:
    case = AMERICAN_PREMIUM_CASES[2]
    assert case.row.evidence is EvidenceClass.EXACT_IDENTITY

    premia = [
        _american(spec, _LATTICE_N_STEPS) - _european(spec, _LATTICE_N_STEPS)
        for spec in case.specs
    ]
    worst_violation = max([0.0, *(a - b for a, b in pairwise(premia))])
    assert worst_violation <= case.row.expected + case.row.tolerance, premia
    # Not a vacuous ordering: the family actually moves.
    assert premia[-1] - premia[0] > 1.0


# --------------------------------------------------------------------------
# The in-repo reference
# --------------------------------------------------------------------------


def test_american_reference_value_row() -> None:
    """Evidence class: CONVERGENCE_ORDER.

    The row pins this engine's own output at `n = 8001` to 1e-12, which is a
    bit-equality budget rather than an accuracy claim. Its accuracy -- about
    1.8e-04 -- and the order-1 evidence behind it live in
    `tests/test_tree_american_convergence.py`; the independent-engine leg is in
    `tests/oracle/test_american_vs_quantlib.py`.
    """
    case = AMERICAN_REFERENCE_CASES[0]
    assert case.row.evidence is EvidenceClass.CONVERGENCE_ORDER
    assert "NOT a published table value" in case.row.source

    value = _american(case.spec, AMERICAN_REFERENCE_N_STEPS)
    assert value == pytest.approx(case.row.expected, abs=case.row.tolerance)


# --------------------------------------------------------------------------
# Early exercise on the Leisen-Reimer lattice
# --------------------------------------------------------------------------


def _lr_and_crr_signed_errors(spec) -> tuple[list[float], list[float]]:
    """Signed errors of both schemes against the bracketed limit, on the odd
    grid. One helper, because all three rows below read the same numbers."""
    lr = [
        _american(spec, n, "leisen-reimer") - AMERICAN_BRACKETED_LIMIT
        for n in AMERICAN_LR_LEVELS
    ]
    crr = [_american(spec, n) - AMERICAN_BRACKETED_LIMIT for n in AMERICAN_LR_LEVELS]
    return lr, crr


def test_american_lr_order_row() -> None:
    """Evidence class: CONVERGENCE_ORDER.

    The row's `expected` is the order and its `tolerance` the band; nothing
    numeric is written here. The point of the row is that it is 1, not 2: the
    European order-2 argument is about the terminal distribution and the
    American error is dominated by the early-exercise boundary instead.
    """
    case = AMERICAN_LR_CASES[0]
    assert case.row.evidence is EvidenceClass.CONVERGENCE_ORDER

    lr, _ = _lr_and_crr_signed_errors(case.spec)
    fit = fit_convergence_order(
        [1.0 / n for n in AMERICAN_LR_LEVELS], [abs(e) for e in lr]
    )
    assert abs(fit.order - case.row.expected) <= case.row.tolerance, case.row.notes
    assert fit.residual < 0.05, fit.residual
    # Explicitly not order 2, which is the claim the row exists to deny.
    assert fit.order < 1.5, fit.order


def test_american_lr_and_crr_bracket_the_limit_row() -> None:
    """Evidence class: CONVERGENCE_ORDER (the sign structure).

    Two schemes at the same `n` give an error bar for free, which is what
    CRR alone can only get by pairing an odd lattice with an even one. The
    expected value is the worst permitted violation of the ordering.
    """
    case = AMERICAN_LR_CASES[1]
    lr, crr = _lr_and_crr_signed_errors(case.spec)

    worst = max([0.0, max(lr), -min(crr)])
    assert worst <= case.row.expected + case.row.tolerance, (lr, crr)
    # And the bracket is not degenerate: LR is the tighter side at every n.
    ratios = [abs(c) / abs(x) for c, x in zip(crr, lr, strict=True)]
    assert min(ratios) > 3.0, ratios


def test_american_lr_richardson_negative_finding_row() -> None:
    """Evidence class: NEGATIVE_FINDING.

    The assertion is the failure: the fitted order must lie OUTSIDE the row's
    band and the log-space residual must stay large, so that the finding fails
    loudly if a future change quietly makes Richardson work here. The detailed
    study is in `tests/test_tree_lr_convergence.py`.
    """
    case = AMERICAN_LR_CASES[2]
    assert case.row.evidence is EvidenceClass.NEGATIVE_FINDING

    spec = case.spec
    h, errs = [], []
    for n1, n2 in pairwise(AMERICAN_LR_LEVELS):
        v1 = _american(spec, n1, "leisen-reimer")
        v2 = _american(spec, n2, "leisen-reimer")
        h.append(1.0 / n1)
        errs.append(abs((n2 * v2 - n1 * v1) / (n2 - n1) - AMERICAN_BRACKETED_LIMIT))

    fit = fit_convergence_order(h, errs)
    assert abs(fit.order - case.row.expected) > case.row.tolerance, fit.order
    assert fit.residual > 0.2, fit.residual


def test_every_american_row_states_its_evidence_and_source() -> None:
    seen: set[str] = set()
    for case in ALL_AMERICAN_CASES:
        row = case.row
        assert row.id not in seen, f"duplicate case id {row.id}"
        seen.add(row.id)
        assert isinstance(row.evidence, EvidenceClass)
        assert row.description.strip()
        assert row.source.strip()
        assert row.tolerance >= 0.0
        assert case.specs

    # The only zero tolerances belong to exact identities.
    for case in ALL_AMERICAN_CASES:
        if case.row.tolerance == 0.0:
            assert case.row.evidence is EvidenceClass.EXACT_IDENTITY, case.row.id


def test_american_and_european_case_ids_do_not_collide() -> None:
    from qpl.cases import ALL_CASES

    european = {case.row.id for case in ALL_CASES}
    american = {case.row.id for case in ALL_AMERICAN_CASES}
    assert not (european & american)
