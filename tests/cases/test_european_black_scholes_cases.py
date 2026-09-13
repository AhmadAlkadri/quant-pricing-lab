"""Evaluate the European Black-Scholes benchmark rows.

Every assertion here reads its expected value, tolerance, evidence class and
citation from `qpl.cases.european_black_scholes`; nothing numeric is hardcoded
in this file. The cross-engine test at the bottom is the one place where a
tolerance is chosen per engine rather than per row, and each of those choices
is justified in-line against a measurement.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from qpl.cases import (
    KNOWN_VALUE_CASES,
    LIMIT_CASES,
    MC_CROSS_ENGINE_DRAWS,
    MC_CROSS_ENGINE_N_STRATA,
    MC_CROSS_ENGINE_SEED,
    MC_CROSS_ENGINE_STDERR_MULTIPLE,
    MC_CROSS_ENGINE_VARIANCE_REDUCTION,
    MONOTONICITY_CASES,
    PARITY_CASES,
    PDE_GREEK_CASES,
    PDE_GREEKS_N,
    PDE_GREEKS_STRIKE_ALIGNMENT,
    PDE_GREEKS_TIME_STEPPING,
    REFERENCE_ATM_CALL,
    TREE_EVEN_LEVELS,
    TREE_KNOWN_VALUE_TOLERANCE,
    TREE_LR_KNOWN_VALUE_TOLERANCE,
    TREE_LR_ORDER_CASES,
    TREE_LR_REFERENCE_N_STEPS,
    TREE_ODD_LEVELS,
    TREE_ORDER_CASES,
    TREE_REFERENCE_N_STEPS,
    EuropeanBSCase,
    parity_residual,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.pde.pricers import PDEConfig
from qpl.engines.tree import TreeConfig
from qpl.pricing import greeks, price
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
    """Five independent routes to the same number.

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
    - Monte Carlo, terminal sampling, **stratified** (K = 64), 12 800 paths,
      seed 123: 4 standard errors. The estimator is unbiased, so the only
      question is sampling noise; 4 sigma is a ~6e-5 false-failure rate for a
      fixed seed that lands at 0.954 (call) and 0.905 (put) standard errors.
      This leg was 200 000 plain paths until Slice 7. Stratifying the single
      normal that drives the terminal price buys a measured variance factor of
      about 110 at this point, so the leg now runs at **one sixteenth** of the
      paths and still reports a standard error three times *smaller* (1.107e-02
      against the plain run's 3.283e-02). The tolerance is deliberately
      unchanged -- four of the estimator's own standard errors is the only
      honest form for a Monte Carlo agreement -- but the interval it must fall
      inside shrank by a factor of three, so the leg is a strictly stronger
      check that costs less.
    - CRR tree, n_steps = 2000: `TREE_KNOWN_VALUE_TOLERANCE` (2.5e-3). This
      one is derived from the measured error constant rather than chosen:
      `n * |tree - closed form|` tends to 1.9994 on even `n`, so the predicted
      error at n = 2000 is 1.00e-3 and the measured error is 9.998e-04 for
      both rows. The tree is an order-1 scheme, so it is deliberately two
      orders of magnitude looser than the order-2 PDE leg at a comparable
      grid size; that difference is the point of having both.
    - Leisen-Reimer tree, n_steps = 2001: `TREE_LR_KNOWN_VALUE_TOLERANCE`
      (2.5e-7), likewise derived from the measurement -- the error at that
      lattice size is -8.853e-08 for both rows, so the tolerance keeps the
      same factor of 2.8 that the CRR leg keeps, four orders of magnitude
      tighter. The fifth leg exists to make that gap an assertion rather than
      a docstring: at essentially the same work (2001 steps against 2000) the
      order-2 scheme is 11_000 times closer, and the test below checks the
      ratio, not just the two tolerances.
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
        cfg=MCConfig(
            n_paths=MC_CROSS_ENGINE_DRAWS,
            n_steps=1,
            seed=MC_CROSS_ENGINE_SEED,
            variance_reduction=MC_CROSS_ENGINE_VARIANCE_REDUCTION,
            n_strata=MC_CROSS_ENGINE_N_STRATA,
        ),
    )
    assert mc_res.stderr is not None
    assert abs(mc_res.value - expected) <= MC_CROSS_ENGINE_STDERR_MULTIPLE * mc_res.stderr
    # The leg is tighter than the plain 200 000-path one it replaced, not just
    # cheaper: assert the standard error itself, so a run that had silently
    # fallen back to plain sampling (stderr 3.283e-02 at 200 000 paths, and
    # 0.13 at 12 800) would fail here rather than pass on a wider interval.
    assert mc_res.stderr < 0.02
    assert mc_res.meta is not None
    assert mc_res.meta["variance_reduction"] == ("stratified",)

    tree = price(
        option,
        model,
        market,
        method="tree",
        cfg=TreeConfig(n_steps=TREE_REFERENCE_N_STEPS),
    ).value
    assert tree == pytest.approx(expected, abs=TREE_KNOWN_VALUE_TOLERANCE)

    lr_tree = price(
        option,
        model,
        market,
        method="tree",
        cfg=TreeConfig(n_steps=TREE_LR_REFERENCE_N_STEPS, scheme="leisen-reimer"),
    ).value
    assert lr_tree == pytest.approx(expected, abs=TREE_LR_KNOWN_VALUE_TOLERANCE)
    # The order gap, as an assertion: at 2001 steps against 2000, the
    # Leisen-Reimer error is measured at 8.85e-08 and the CRR error at
    # 9.998e-04, a factor of 11_000. The floor keeps a factor of about ten.
    assert abs(lr_tree - expected) * 1_000.0 < abs(tree - expected)


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


@pytest.mark.parametrize("case", TREE_LR_ORDER_CASES, ids=_ids(TREE_LR_ORDER_CASES))
def test_leisen_reimer_convergence_order_rows(case: EuropeanBSCase) -> None:
    """Evaluate the Leisen-Reimer convergence-order rows.

    Same shape as `test_tree_convergence_order_rows`, with three differences
    that are the whole content of the scheme:

    - one sequence rather than two, since there is no even-`n` construction to
      fit separately;
    - the row's `expected` is 2, not 1;
    - the residual bound comes from the row rather than being a shared
      constant, because the measured residuals (0.0087 to 0.0154) are an order
      of magnitude above CRR's (0.0005) for a stated reason -- the
      Peizer-Pratt tail match is high but finite order -- and a single bound
      would either hide that or fail.

    The detailed study, including the comparison against CRR at matched `n`
    and the absence of parity oscillation, is in
    `tests/test_tree_lr_convergence.py`. This test is here so that the cases
    layer carries the claim as data.
    """
    assert case.row.evidence is EvidenceClass.CONVERGENCE_ORDER
    assert "measured in-repo" in case.row.source

    spec = case.spec
    option, model, market = spec.option(), spec.model(), spec.market()
    analytic = price(option, model, market, method="analytic").value

    signed = [
        price(
            option,
            model,
            market,
            method="tree",
            cfg=TreeConfig(n_steps=n, scheme="leisen-reimer"),
        ).value
        - analytic
        for n in TREE_ODD_LEVELS
    ]

    fit = fit_convergence_order(
        [1.0 / n for n in TREE_ODD_LEVELS], [abs(e) for e in signed]
    )
    assert abs(fit.order - case.row.expected) <= case.row.tolerance, case.row.source
    assert fit.residual < 0.05, fit.residual

    # "one-signed and monotone", as the row's notes claim.
    assert all(e > 0.0 for e in signed) or all(e < 0.0 for e in signed), signed
    magnitudes = [abs(e) for e in signed]
    assert all(a > b for a, b in pairwise(magnitudes)), magnitudes


def _pde_greeks_cfg() -> PDEConfig:
    """The one grid every PDE Greek row is evaluated on."""
    return PDEConfig(
        n_s=PDE_GREEKS_N,
        n_t=PDE_GREEKS_N,
        theta=0.5,
        s_max_multiplier=4.0,
        strike_alignment=PDE_GREEKS_STRIKE_ALIGNMENT,  # type: ignore[arg-type]
        time_stepping=PDE_GREEKS_TIME_STEPPING,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("case", PDE_GREEK_CASES, ids=_ids(PDE_GREEK_CASES))
def test_pde_grid_greek_rows(case: EuropeanBSCase) -> None:
    """Every Greek off the PDE grid, against the closed form.

    Evidence class: CLOSED_FORM. The reference is this repository's own
    analytic engine, so what this checks is that a completely different
    numerical route -- march a grid, difference it -- lands on the same five
    numbers to a tolerance derived from a measurement rather than chosen.

    Each row's tolerance keeps a factor of 3.5 to 4.1 over the error measured
    at this grid, which is roughly one refinement level of the order-2
    sequences in `tests/test_pde_greeks.py`; `row.notes` carries the
    measurement. The five tolerances span four orders of magnitude
    (7e-06 for gamma to 2e-02 for rho) because the Greeks themselves do: in
    relative terms they are all between 2.1e-05 and 3.9e-04.

    Delta and gamma come from second-order central stencils on the grid,
    theta from the PDE identity, vega and rho from bump-and-revalue on the
    same grid -- so the last two are no better than the price and are the
    loosest rows relative to their own size.
    """
    assert case.row.evidence is EvidenceClass.CLOSED_FORM
    assert "derived in-repo" in case.row.source
    assert case.row.expected == 0.0

    spec = case.spec
    option, model, market = spec.option(), spec.model(), spec.market()
    greek = case.row.id.split("_")[2]

    analytic = greeks(option, model, market, method="analytic")
    pde = greeks(option, model, market, method="pde", cfg=_pde_greeks_cfg())

    residual = getattr(pde, greek) - getattr(analytic, greek)
    assert abs(residual) <= case.row.tolerance, (case.row.id, residual, case.row.notes)
    # Not vacuous: the row would also pass if the engine returned the analytic
    # value, so pin that it is genuinely a discretisation and not a passthrough.
    assert residual != 0.0


def test_pde_grid_greeks_beat_the_bump_path_on_delta() -> None:
    """The cases layer carries the Slice 4 replacement as an assertion.

    Evidence class: NEGATIVE_FINDING for the bump leg. At the grid these rows
    use, the bump path's delta error is already sitting on its `O(h**2)` floor
    (measured -8.248e-05 at n = 400 and still -8.574e-05 at n = 1600) while
    the grid path's is -1.430e-04 and falling at order 2. At n = 400 the bump
    path is therefore *ahead*; by n = 800 it is behind, and by n = 1600 it is
    9.5x behind. This test pins the crossing rather than the endpoint, because
    the endpoint alone would read as "the old path was fine".
    """
    spec = REFERENCE_ATM_CALL
    option, model, market = spec.option(), spec.model(), spec.market()
    analytic = greeks(option, model, market, method="analytic").delta

    def _delta(n: int, greeks_method: str) -> float:
        cfg = PDEConfig(
            n_s=n,
            n_t=n,
            theta=0.5,
            s_max_multiplier=4.0,
            strike_alignment=PDE_GREEKS_STRIKE_ALIGNMENT,  # type: ignore[arg-type]
            time_stepping=PDE_GREEKS_TIME_STEPPING,  # type: ignore[arg-type]
            greeks_method=greeks_method,  # type: ignore[arg-type]
        )
        return greeks(option, model, market, method="pde", cfg=cfg).delta

    grid_400 = abs(_delta(400, "grid") - analytic)
    grid_1600 = abs(_delta(1600, "grid") - analytic)
    bump_400 = abs(_delta(400, "bump") - analytic)
    bump_1600 = abs(_delta(1600, "bump") - analytic)

    # The grid path converges; the bump path does not.
    assert grid_400 / grid_1600 > 10.0, (grid_400, grid_1600)
    assert 0.9 < bump_400 / bump_1600 < 1.1, (bump_400, bump_1600)
    # Which is why the crossing happens between the two levels.
    assert bump_400 < grid_400
    assert bump_1600 > 5.0 * grid_1600
