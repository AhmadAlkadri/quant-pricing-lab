"""Evaluate the European Black-Scholes benchmark rows.

Every assertion here reads its expected value, tolerance, evidence class and
citation from `qpl.cases.european_black_scholes`; nothing numeric is hardcoded
in this file. The cross-engine test at the bottom is the one place where a
tolerance is chosen per engine rather than per row, and each of those choices
is justified in-line against a measurement.
"""

from __future__ import annotations

import pytest

from qpl.cases import (
    KNOWN_VALUE_CASES,
    LIMIT_CASES,
    MONOTONICITY_CASES,
    PARITY_CASES,
    EuropeanBSCase,
    parity_residual,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.pde.pricers import PDEConfig
from qpl.pricing import price
from qpl.validation import EvidenceClass


def _ids(cases: tuple[EuropeanBSCase, ...]) -> list[str]:
    return [case.row.id for case in cases]


def _analytic(case: EuropeanBSCase, index: int = 0) -> float:
    spec = case.specs[index]
    return price(spec.option(), spec.model(), spec.market(), method="analytic").value


@pytest.mark.parametrize("case", PARITY_CASES, ids=_ids(PARITY_CASES))
def test_put_call_parity_rows(case: EuropeanBSCase) -> None:
    assert case.row.evidence is EvidenceClass.EXACT_IDENTITY
    spec = case.spec
    call = price(
        spec.option(), spec.model(), spec.market(), method="analytic"
    ).value
    flipped = spec.flipped()
    put = price(
        flipped.option(), flipped.model(), flipped.market(), method="analytic"
    ).value

    residual = parity_residual(call, put, spec)
    assert abs(residual - case.row.expected) <= case.row.tolerance, case.row.source


@pytest.mark.parametrize("case", LIMIT_CASES, ids=_ids(LIMIT_CASES))
def test_degenerate_limit_rows(case: EuropeanBSCase) -> None:
    assert case.row.evidence is EvidenceClass.CLOSED_FORM
    assert _analytic(case) == pytest.approx(case.row.expected, abs=case.row.tolerance)


@pytest.mark.parametrize("case", KNOWN_VALUE_CASES, ids=_ids(KNOWN_VALUE_CASES))
def test_known_value_rows(case: EuropeanBSCase) -> None:
    assert case.row.evidence is EvidenceClass.CLOSED_FORM
    assert "not a published table" in case.row.source
    assert _analytic(case) == pytest.approx(case.row.expected, abs=case.row.tolerance)


@pytest.mark.parametrize("case", MONOTONICITY_CASES, ids=_ids(MONOTONICITY_CASES))
def test_monotonicity_rows(case: EuropeanBSCase) -> None:
    prices = [_analytic(case, i) for i in range(len(case.specs))]
    assert len(prices) >= 2

    decreasing = case.row.id.endswith("in_strike")
    steps = (
        [prices[i] - prices[i + 1] for i in range(len(prices) - 1)]
        if decreasing
        else [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    )
    worst_violation = max(0.0, -min(steps))
    assert worst_violation <= case.row.expected + case.row.tolerance, case.row.source
    # The family must actually move, otherwise the ordering check is vacuous.
    assert max(steps) > 1e-6


@pytest.mark.parametrize("case", KNOWN_VALUE_CASES, ids=_ids(KNOWN_VALUE_CASES))
def test_cross_engine_agreement_on_known_values(case: EuropeanBSCase) -> None:
    """Three independent routes to the same number.

    Evidence class: INDEPENDENT_ENGINE for the PDE leg (a different numerical
    method reaching the same value), STATISTICAL for the Monte Carlo leg (the
    tolerance is a multiple of the reported standard error, not an absolute
    accuracy claim).

    Per-engine tolerances:

    - analytic: `case.row.tolerance` (1e-12). The row's expected value was
      produced by this same closed form, so anything but agreement to round-off
      means the formula changed.
    - PDE, Crank-Nicolson, strike-aligned, n_s = n_t = 400: 5e-4. The measured
      error at this setting is 1.219e-4 for both the call and the put; 5e-4
      leaves roughly a factor of four of headroom, which is about one
      refinement level of the order-2 sequence measured in
      `tests/test_pde_ch4.py`. Tightening below ~2e-4 would make the test
      sensitive to harmless changes in the spline interpolation at spot.
    - Monte Carlo, terminal sampling, 200_000 paths, seed 123: 4 standard
      errors. The estimator is unbiased, so the only question is sampling
      noise; 4 sigma is a ~6e-5 false-failure rate for a fixed seed that is
      already known to land at 0.75 (call) and 0.13 (put) standard errors.
    """
    spec = case.spec
    option, model, market = spec.option(), spec.model(), spec.market()
    expected = case.row.expected

    analytic = price(option, model, market, method="analytic").value
    assert analytic == pytest.approx(expected, abs=case.row.tolerance)

    pde_cfg = PDEConfig(
        n_s=400, n_t=400, theta=0.5, s_max_multiplier=4.0, strike_alignment="midpoint"
    )
    pde = price(option, model, market, method="pde", cfg=pde_cfg).value
    assert pde == pytest.approx(expected, abs=5e-4)

    mc_res = price(
        option,
        model,
        market,
        method="mc",
        cfg=MCConfig(n_paths=200_000, n_steps=1, seed=123),
    )
    assert mc_res.stderr is not None
    assert abs(mc_res.value - expected) <= 4.0 * mc_res.stderr


def test_every_row_states_its_evidence_and_source() -> None:
    from qpl.cases import ALL_CASES

    seen_ids = set()
    for case in ALL_CASES:
        row = case.row
        assert row.id not in seen_ids, f"duplicate case id {row.id}"
        seen_ids.add(row.id)
        assert isinstance(row.evidence, EvidenceClass)
        assert row.description.strip()
        assert row.source.strip()
        assert row.tolerance > 0.0
        assert case.specs
