"""Evaluate the single-barrier benchmark rows.

Every assertion reads its expected value, tolerance, evidence class and
citation from `qpl.cases.barrier_black_scholes`; nothing numeric is hardcoded
here except the per-engine budgets in the cross-engine test, and each of those
is a named constant from the cases layer with its derivation in the constant's
docstring.

The detailed studies live in `tests/test_barrier_analytic.py`,
`tests/test_barrier_mc.py` and `tests/test_barrier_tree_convergence.py`. This
file exists so that the cases layer carries the claims as *data*, so that the
three published values are checked in one place, and so that the three engines
are compared on one number.
"""

from __future__ import annotations

import pytest

from qpl.cases import (
    ALL_AMERICAN_CASES,
    ALL_ASIAN_CASES,
    ALL_BARRIER_CASES,
    ALL_CASES,
    ALL_DIGITAL_CASES,
    ALL_SDE_CASES,
    BARRIER_BOYLE_LAU_ENVELOPE,
    BARRIER_CROSS_ENGINE_CASES,
    BARRIER_DISCRETE_CROSS_ENGINE_CASES,
    BARRIER_HAUG_CASES,
    BARRIER_IDENTITY_CASES,
    BARRIER_LIMIT_CASES,
    BARRIER_MC_MONITORING,
    BARRIER_MC_ORDER_CASES,
    BARRIER_MC_PATHS,
    BARRIER_MC_SEED,
    BARRIER_MC_STDERR_MULTIPLE,
    BARRIER_MC_VARIANCE_REDUCTION,
    BARRIER_PDE_CONCENTRATION,
    BARRIER_PDE_DISCRETE_STDERR_MULTIPLE,
    BARRIER_PDE_GRID,
    BARRIER_PDE_N,
    BARRIER_PDE_ORDER_CASES,
    BARRIER_PDE_STRIKE_ALIGNMENT,
    BARRIER_PDE_TIME_STEPPING,
    BARRIER_PDE_TOLERANCE,
    BARRIER_TREE_ORDER_CASES,
    MC_VARIANCE_REDUCTION_CASES,
    BarrierBSCase,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.pde.pricers import PDEConfig
from qpl.engines.tree import TreeConfig, boyle_lau_steps
from qpl.instruments.options import EuropeanOption
from qpl.models.black_scholes import bs_price
from qpl.pricing import price
from qpl.validation import EvidenceClass


def _ids(cases: tuple[BarrierBSCase, ...]) -> list[str]:
    return [case.row.id for case in cases]


def _analytic(case: BarrierBSCase) -> float:
    spec = case.spec
    return price(spec.option(), spec.model(), spec.market()).value


# --------------------------------------------------------------------------
# (i) Published values.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", BARRIER_HAUG_CASES, ids=_ids(BARRIER_HAUG_CASES))
def test_published_barrier_rows(case: BarrierBSCase) -> None:
    """Evidence class: PUBLISHED_BENCHMARK, to the precision of the table.

    Four-decimal published figures, so agreement can be claimed no more
    tightly than half a unit in the last place; the measured residuals (3.23e-05,
    3.66e-05, 1.98e-05) are all inside the published figure's own rounding.
    """
    assert case.row.evidence is EvidenceClass.PUBLISHED_BENCHMARK
    assert _analytic(case) == pytest.approx(case.row.expected, abs=case.row.tolerance)


def test_the_published_rows_vary_the_strike_and_not_the_spot() -> None:
    """The mis-reading the slice arrived with, pinned so it cannot recur.

    In the published table the spot is fixed at 100 and the column heading is
    the **strike**. Read the other way round -- spot 90 against a down barrier
    at 95 -- the contract is already knocked out at inception and worth its
    rebate, 3.00, not 9.02. The gap is a factor of three, so the reading is not
    a detail.
    """
    row, k90 = BARRIER_HAUG_CASES[0].row, BARRIER_HAUG_CASES[0].spec
    assert k90.spot == 100.0 and k90.strike == 90.0
    misread = k90.__class__(**{**k90.__dict__, "spot": 90.0, "strike": 100.0})
    value = price(misread.option(), misread.model(), misread.market())
    assert value.value == misread.rebate == 3.0
    assert value.meta["touched_at_inception"] is True
    assert abs(value.value - row.expected) > 5.0


# --------------------------------------------------------------------------
# (ii) and (iii): identities and limits.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", BARRIER_IDENTITY_CASES, ids=_ids(BARRIER_IDENTITY_CASES)
)
def test_in_out_parity_rows(case: BarrierBSCase) -> None:
    """Evidence class: EXACT_IDENTITY, at a round-off budget."""
    assert case.row.evidence is EvidenceClass.EXACT_IDENTITY
    spec = case.spec
    model, market = spec.model(), spec.market()
    knock_out = price(spec.option(), model, market).value
    knock_in = price(spec.flipped_knock().option(), model, market).value
    vanilla = bs_price(
        S=spec.spot, K=spec.strike, T=spec.expiry, r=spec.rate, sigma=spec.sigma,
        q=spec.dividend, kind=spec.kind,
    )
    assert spec.rebate == 0.0
    residual = knock_in + knock_out - vanilla
    assert residual == pytest.approx(case.row.expected, abs=case.row.tolerance)


@pytest.mark.parametrize("case", BARRIER_LIMIT_CASES, ids=_ids(BARRIER_LIMIT_CASES))
def test_barrier_limit_rows(case: BarrierBSCase) -> None:
    """Evidence class: CLOSED_FORM, against the vanilla engine and the rebate."""
    assert case.row.evidence is EvidenceClass.CLOSED_FORM
    spec = case.spec
    model, market = spec.model(), spec.market()

    if case.row.id.endswith("is_the_vanilla"):
        far = spec.__class__(**{**spec.__dict__, "barrier": spec.spot * 1e-6})
        residual = price(far.option(), model, market).value - price(
            EuropeanOption(spec.kind, spec.strike, spec.expiry), model, market
        ).value
    else:
        at_spot = spec.__class__(**{**spec.__dict__, "barrier": spec.spot})
        residual = price(at_spot.option(), model, market).value - spec.rebate
    assert residual == pytest.approx(case.row.expected, abs=case.row.tolerance)


# --------------------------------------------------------------------------
# (iv) and (v): the two order studies carry their claims as data.
# --------------------------------------------------------------------------


_ORDER_CASES = (
    BARRIER_MC_ORDER_CASES + BARRIER_TREE_ORDER_CASES + BARRIER_PDE_ORDER_CASES
)


@pytest.mark.parametrize("case", _ORDER_CASES, ids=_ids(_ORDER_CASES))
def test_order_rows_state_a_measurable_claim(case: BarrierBSCase) -> None:
    """The rows are data; the measurements are in the two dedicated files.

    Re-running a five-level Monte Carlo ladder and three full sawtooth periods
    here would double this slice's runtime to recompute numbers that are
    already asserted with their standard errors and fit residuals in
    `tests/test_barrier_mc.py` and
    `tests/test_barrier_tree_convergence.py`. What this file checks is that the
    rows say something a reader can act on: an evidence class that matches the
    claim, a citation, a tolerance that is neither vacuous nor impossible, and
    notes carrying the measured value.
    """
    row = case.row
    assert row.evidence in {
        EvidenceClass.CONVERGENCE_ORDER,
        EvidenceClass.STATISTICAL,
        EvidenceClass.NEGATIVE_FINDING,
    }
    assert row.tolerance > 0.0
    assert "Measured" in row.notes or "measured" in row.notes
    assert "in-repo" in row.source or "derived" in row.source
    if row.evidence is EvidenceClass.CONVERGENCE_ORDER:
        # An order claim has to be tight enough to exclude the neighbouring
        # integer/half-integer order, or it is not a claim. The mesh row is
        # the exception the rule has to name: its `expected` is an error
        # RATIO of 15 and not an order, so a 0.25 band would be absurd.
        if "mesh" not in row.id:
            assert row.tolerance < 0.25 + 1e-12
        else:
            assert row.tolerance < 0.5 * row.expected


# --------------------------------------------------------------------------
# (vi) Three engines on one number.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", BARRIER_CROSS_ENGINE_CASES, ids=_ids(BARRIER_CROSS_ENGINE_CASES)
)
def test_cross_engine_agreement(case: BarrierBSCase) -> None:
    """Evidence class: INDEPENDENT_ENGINE, with each leg on its own budget.

    Three genuinely different routes to one number: the closed form (an
    integral against the joint law of the terminal value and the running
    extremum), a binomial lattice at a Boyle-Lau step count (a discrete
    backward induction that knocks out at nodes), and a Monte Carlo simulation
    of the **discretely** monitored contract weighted by the Brownian-bridge
    survival probability (which is what makes the third leg an estimate of the
    same, continuous, contract).

    The lattice budget is `BARRIER_BOYLE_LAU_ENVELOPE / n` rather than the
    error measured at the chosen layer, because the Boyle-Lau constant is
    erratic and reading a tolerance off one layer would be cherry-picking. The
    Monte Carlo budget is a multiple of its own standard error, which is a
    statistical statement and not an accuracy claim.
    """
    assert case.row.evidence is EvidenceClass.INDEPENDENT_ENGINE
    spec = case.spec
    model, market = spec.model(), spec.market()

    exact = price(spec.option(), model, market).value

    n_steps = boyle_lau_steps(
        spec.boyle_lau_layer, spot=spec.spot, barrier=spec.barrier,
        sigma=spec.sigma, expiry=spec.expiry,
    )
    lattice = price(
        spec.option(), model, market, method="tree", cfg=TreeConfig(n_steps=n_steps)
    )
    lattice_budget = BARRIER_BOYLE_LAU_ENVELOPE / n_steps
    assert lattice.value == pytest.approx(exact, abs=lattice_budget)

    simulation = price(
        spec.option(monitoring=BARRIER_MC_MONITORING),
        model,
        market,
        method="mc",
        cfg=MCConfig(
            n_paths=BARRIER_MC_PATHS,
            seed=BARRIER_MC_SEED,
            variance_reduction=BARRIER_MC_VARIANCE_REDUCTION,
            barrier_correction="brownian_bridge",
        ),
    )
    assert simulation.meta["estimates"] == "continuous"
    assert simulation.value == pytest.approx(
        exact, abs=BARRIER_MC_STDERR_MULTIPLE * simulation.stderr
    )

    # The row's own tolerance is the envelope constant; the realised gap is
    # recorded against the looser of the two legs, which bounds any pair.
    # Slice 13's fourth leg: a finite-difference solve on a sinh grid whose
    # domain is truncated at the barrier, so the barrier is node 0 exactly.
    # Its budget is a flat constant rather than an envelope, because a
    # boundary is not a node the scheme rounds to.
    grid = price(
        spec.option(),
        model,
        market,
        method="pde",
        cfg=PDEConfig(
            n_s=BARRIER_PDE_N,
            n_t=BARRIER_PDE_N,
            grid=BARRIER_PDE_GRID,
            concentration=BARRIER_PDE_CONCENTRATION,
            strike_alignment=BARRIER_PDE_STRIKE_ALIGNMENT,
            time_stepping=BARRIER_PDE_TIME_STEPPING,
        ),
    )
    assert grid.meta["barrier_on_node"] is True
    assert grid.meta["effective_barrier"] == spec.barrier
    assert grid.value == pytest.approx(exact, abs=BARRIER_PDE_TOLERANCE)

    # The row's own tolerance is the envelope constant; the realised gap is
    # recorded against the looser of the two legs, which bounds any pair.
    gap = max(abs(lattice.value - exact), abs(simulation.value - exact))
    assert gap < max(lattice_budget, BARRIER_MC_STDERR_MULTIPLE * simulation.stderr)
    assert exact > 0.1  # not four engines agreeing on nothing


@pytest.mark.parametrize(
    "case",
    BARRIER_DISCRETE_CROSS_ENGINE_CASES,
    ids=_ids(BARRIER_DISCRETE_CROSS_ENGINE_CASES),
)
def test_discrete_cross_engine_agreement(case: BarrierBSCase) -> None:
    """Evidence class: INDEPENDENT_ENGINE, on the *other* contract.

    The discretely monitored barrier is a different contract from the
    continuous one -- worth O(1/sqrt(m)) more for a knock-out -- and until
    Slice 13 only one engine in this package could price it. Now two can, by
    routes with nothing in common: the grid projects the value function onto
    the rebate at each monitoring date, and the simulation draws the path at
    those dates and looks.

    There is deliberately no lattice leg. `qpl.engines.tree.barrier` refuses a
    discrete schedule, because knocking out only at the time levels that
    coincide with monitoring dates needs `n` to be a multiple of `m` *and*
    barrier-aligned at once -- two conditions on one integer. The grid has no
    such conflict: its time levels are built to contain the monitoring dates,
    and its node placement is a separate axis.
    """
    assert case.row.evidence is EvidenceClass.INDEPENDENT_ENGINE
    spec = case.spec
    model, market = spec.model(), spec.market()
    option = spec.option(monitoring=BARRIER_MC_MONITORING)

    grid = price(
        option,
        model,
        market,
        method="pde",
        cfg=PDEConfig(
            n_s=BARRIER_PDE_N,
            n_t=BARRIER_PDE_N,
            grid=BARRIER_PDE_GRID,
            concentration=BARRIER_PDE_CONCENTRATION,
            strike_alignment=BARRIER_PDE_STRIKE_ALIGNMENT,
            time_stepping=BARRIER_PDE_TIME_STEPPING,
        ),
    )
    assert grid.meta["monitoring"] == "discrete"
    assert grid.meta["n_projections"] == BARRIER_MC_MONITORING

    simulation = price(
        option,
        model,
        market,
        method="mc",
        cfg=MCConfig(
            n_paths=BARRIER_MC_PATHS,
            seed=BARRIER_MC_SEED,
            variance_reduction=BARRIER_MC_VARIANCE_REDUCTION,
            barrier_correction="none",
        ),
    )
    assert simulation.meta["estimates"] == "discrete"
    assert grid.value == pytest.approx(
        simulation.value,
        abs=BARRIER_PDE_DISCRETE_STDERR_MULTIPLE * simulation.stderr,
    )

    # The lattice cannot be a third leg here, and says so rather than guessing.
    from qpl.exceptions import NotSupportedError

    with pytest.raises(NotSupportedError, match="method='mc'"):
        price(option, model, market, method="tree", cfg=TreeConfig(n_steps=101))


# --------------------------------------------------------------------------
# Metadata contract.
# --------------------------------------------------------------------------


def test_every_barrier_row_states_its_evidence_and_source() -> None:
    ids = [case.row.id for case in ALL_BARRIER_CASES]
    assert len(ids) == len(set(ids))
    for case in ALL_BARRIER_CASES:
        row = case.row
        assert row.id.startswith("barrier_")
        assert row.description
        assert isinstance(row.evidence, EvidenceClass)
        assert row.source
        assert row.tolerance >= 0.0
        assert case.specs


def test_the_seventh_id_space_is_disjoint_from_the_other_six() -> None:
    """Seven id spaces now, asserted pairwise disjoint.

    A duplicate id would make two different claims share a pytest parameter id
    and silently hide one of them in a failure report.
    """
    spaces = {
        "european": {case.row.id for case in ALL_CASES},
        "american": {case.row.id for case in ALL_AMERICAN_CASES},
        "digital": {case.row.id for case in ALL_DIGITAL_CASES},
        "mc_variance": {case.row.id for case in MC_VARIANCE_REDUCTION_CASES},
        "asian": {case.row.id for case in ALL_ASIAN_CASES},
        "sde": {case.row.id for case in ALL_SDE_CASES},
        "barrier": {case.row.id for case in ALL_BARRIER_CASES},
    }
    names = sorted(spaces)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert spaces[left].isdisjoint(spaces[right]), (left, right)


def test_specs_round_trip_through_their_domain_objects() -> None:
    """A spec builds the instrument it describes, in both monitoring modes."""
    for case in ALL_BARRIER_CASES:
        for spec in case.specs:
            continuous = spec.option()
            assert continuous.is_continuous
            assert continuous.barrier == spec.barrier
            assert continuous.barrier_type == spec.barrier_type
            assert continuous.rebate == spec.rebate
            discrete = spec.option(monitoring=8)
            assert discrete.n_monitoring == 8
            assert discrete.monitoring_times[-1] == pytest.approx(spec.expiry)
            assert spec.market().spot == spec.spot
            assert spec.model().sigma == spec.sigma
            # The flip is an involution, and it changes the knock direction
            # without touching anything else.
            flipped = spec.flipped_knock()
            assert flipped.flipped_knock() == spec
            assert flipped.barrier_type != spec.barrier_type
            assert flipped.barrier == spec.barrier
