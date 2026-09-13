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
    TREE_EVEN_LEVELS,
    TREE_KNOWN_VALUE_TOLERANCE,
    TREE_ODD_LEVELS,
    TREE_ORDER_CASES,
    TREE_REFERENCE_N_STEPS,
    EuropeanBSCase,
    parity_residual,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.pde.pricers import PDEConfig
from qpl.engines.tree import TreeConfig
from qpl.pricing import price
from qpl.validation import EvidenceClass, fit_convergence_order


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
    """Four independent routes to the same number.

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
    - CRR tree, n_steps = 2000: `TREE_KNOWN_VALUE_TOLERANCE` (2.5e-3). This
      one is derived from the measured error constant rather than chosen:
      `n * |tree - closed form|` tends to 1.9994 on even `n`, so the predicted
      error at n = 2000 is 1.00e-3 and the measured error is 9.998e-04 for
      both rows. The tree is an order-1 scheme, so it is deliberately two
      orders of magnitude looser than the order-2 PDE leg at a comparable
      grid size; that difference is the point of having both.
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

    tree = price(
        option,
        model,
        market,
        method="tree",
        cfg=TreeConfig(n_steps=TREE_REFERENCE_N_STEPS),
    ).value
    assert tree == pytest.approx(expected, abs=TREE_KNOWN_VALUE_TOLERANCE)


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


@pytest.mark.parametrize("case", TREE_ORDER_CASES, ids=_ids(TREE_ORDER_CASES))
def test_tree_convergence_order_rows(case: EuropeanBSCase) -> None:
    """Evaluate the CRR convergence-order rows.

    The row's `expected` is the convergence order and its `tolerance` is the
    band the fitted slope must land in; nothing numeric is hardcoded here. The
    sign structure recorded in `row.notes` is checked too, because "order 1"
    alone would not distinguish a tree that brackets Black-Scholes from one
    that approaches it from a single side.

    The detailed study -- Richardson extrapolation, the scaled constants, the
    replication and parity identities -- lives in
    `tests/test_tree_convergence.py`. This test is here so that the cases
    layer carries the claim as data.
    """
    assert case.row.evidence is EvidenceClass.CONVERGENCE_ORDER

    spec = case.spec
    option, model, market = spec.option(), spec.model(), spec.market()
    analytic = price(option, model, market, method="analytic").value

    levels = TREE_ODD_LEVELS if case.row.id.endswith("odd_n") else TREE_EVEN_LEVELS
    signed = [
        price(option, model, market, method="tree", cfg=TreeConfig(n_steps=n)).value - analytic
        for n in levels
    ]

    fit = fit_convergence_order([1.0 / n for n in levels], [abs(e) for e in signed])
    assert abs(fit.order - case.row.expected) <= case.row.tolerance, case.row.source
    assert fit.residual < 0.01

    if "ABOVE" in case.row.notes:
        assert all(e > 0.0 for e in signed), signed
    else:
        assert all(e < 0.0 for e in signed), signed
