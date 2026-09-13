"""The Asian case rows, evaluated.

`qpl.cases.asian_black_scholes` holds the claims; this file evaluates them. The
split matters here more than it does for the earlier instruments, because the
Asian rows are the first set in which one instrument carries three different
evidence classes at once -- exact closed forms for the geometric average,
statistical references for the arithmetic one, and a published benchmark that
certifies an *approximation* rather than a price. A reader who wants to know
what this package claims about Asian options reads the case module; a reader who
wants to know whether the claims hold reads this file.

Cost: the arithmetic rows each run 200 000 control-variate paths, which is what
the slice's tolerance derivation is built on. Measured 1.4 s for 33 tests.
"""

from __future__ import annotations

import math

import pytest

from qpl.cases import (
    ALL_ASIAN_CASES,
    ASIAN_APPROXIMATION_CASES,
    ASIAN_CONTINUOUS_LIMIT_CASES,
    ASIAN_IDENTITY_CASES,
    ASIAN_KNOWN_VALUE_CASES,
    ASIAN_MC_CASES,
    ASIAN_MC_PATHS,
    ASIAN_MC_SEED,
    ASIAN_MC_STDERR_MULTIPLE,
    ASIAN_MC_VARIANCE_REDUCTION,
    ASIAN_ORDER_LEVELS,
    HAUG_CONTINUOUS_SPEC,
)
from qpl.engines.analytic.asian import (
    expected_geometric_average,
    turnbull_wakeman_price,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.instruments import uniform_fixing_times
from qpl.pricing import price
from qpl.validation import EvidenceClass, fit_convergence_order


def _ids(cases):
    return [case.row.id for case in cases]


def _analytic(spec) -> float:
    return price(spec.option(), spec.model(), spec.market()).value


def _mc(spec, *, n_paths: int = ASIAN_MC_PATHS, seed: int = ASIAN_MC_SEED):
    return price(
        spec.option(),
        spec.model(),
        spec.market(),
        method="mc",
        cfg=MCConfig(
            n_paths=n_paths, seed=seed, variance_reduction=ASIAN_MC_VARIANCE_REDUCTION
        ),
    )


def _continuous_geometric(spec) -> float:
    """Kemna-Vorst continuous limit, written out here rather than imported.

    Same reason as in `tests/test_asian_analytic.py`: it is what the discrete
    formula is measured against, so it must not share code with it.
    """
    mu = spec.rate - spec.dividend
    m = math.log(spec.spot) + (mu - 0.5 * spec.sigma**2) * spec.expiry / 2.0
    v = spec.sigma**2 * spec.expiry / 3.0
    forward = math.exp(m + 0.5 * v)
    sd = math.sqrt(v)
    d1 = (math.log(forward / spec.strike) + 0.5 * v) / sd
    d2 = d1 - sd

    def ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    disc = spec.discount()
    if spec.kind == "call":
        return disc * (forward * ncdf(d1) - spec.strike * ncdf(d2))
    return disc * (spec.strike * ncdf(-d2) - forward * ncdf(-d1))


# --------------------------------------------------------------------------
# Row-set hygiene.
# --------------------------------------------------------------------------


def test_every_asian_row_states_its_evidence_and_source():
    """The metadata contract, and the fifth id space's disjointness.

    Five id spaces now: European, American, digital, variance-reduction and
    Asian. They are asserted pairwise disjoint here because a duplicate id
    across two modules would silently make one row's failure report the other's
    description.
    """
    from qpl.cases import (
        ALL_AMERICAN_CASES,
        ALL_CASES,
        ALL_DIGITAL_CASES,
        MC_VARIANCE_REDUCTION_CASES,
    )

    ids = _ids(ALL_ASIAN_CASES)
    assert len(set(ids)) == len(ids)
    others = {
        case.row.id
        for case in (
            *ALL_CASES,
            *ALL_AMERICAN_CASES,
            *ALL_DIGITAL_CASES,
            *MC_VARIANCE_REDUCTION_CASES,
        )
    }
    assert others.isdisjoint(ids)

    for case in ALL_ASIAN_CASES:
        row = case.row
        assert isinstance(row.evidence, EvidenceClass)
        assert row.description.strip()
        assert row.source.strip()
        assert row.tolerance >= 0.0
        assert case.specs
        for spec in case.specs:
            assert spec.n_fixings >= 1
            assert spec.option().n_fixings == spec.n_fixings


def test_the_evidence_classes_are_assigned_the_way_the_module_says():
    """Geometric rows are exact, arithmetic rows are statistical, and no row lies.

    This is the check that keeps the layer honest as it grows: an arithmetic
    Asian has no closed form, so a `CLOSED_FORM` row on one would be a false
    claim about what this package knows, and it is refused here rather than in
    review.
    """
    exact = {EvidenceClass.EXACT_IDENTITY, EvidenceClass.CLOSED_FORM}
    for case in ALL_ASIAN_CASES:
        if case.row.evidence in exact:
            assert all(
                spec.averaging == "geometric" or case.row.id.startswith("asian_am_gm")
                for spec in case.specs
            ), case.row.id
        if any(spec.averaging == "arithmetic" for spec in case.specs):
            assert case.row.evidence in {
                EvidenceClass.STATISTICAL,
                EvidenceClass.PUBLISHED_BENCHMARK,
                EvidenceClass.EXACT_IDENTITY,
            }, case.row.id


# --------------------------------------------------------------------------
# (i) Identities.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [c for c in ASIAN_IDENTITY_CASES if c.row.id.startswith("asian_geometric_parity")],
    ids=lambda c: c.row.id,
)
def test_geometric_asian_put_call_parity(case):
    """EXACT_IDENTITY: `C - P - e^{-rT}(E[G] - K) = 0`."""
    assert case.row.evidence is EvidenceClass.EXACT_IDENTITY
    spec = case.spec
    call = _analytic(spec)
    put = _analytic(spec.flipped())
    expected = expected_geometric_average(
        S=spec.spot,
        mu=spec.market().rate(spec.expiry) - spec.market().dividend_yield(spec.expiry),
        sigma=spec.sigma,
        fixing_times=spec.fixing_times(),
    )
    residual = (
        call - put - spec.market().df_r(spec.expiry) * (expected - spec.strike)
    )
    assert abs(residual - case.row.expected) <= case.row.tolerance, residual


@pytest.mark.parametrize(
    "case",
    [c for c in ASIAN_IDENTITY_CASES if c.row.id.startswith("asian_am_gm")],
    ids=lambda c: c.row.id,
)
def test_arithmetic_beats_geometric_for_calls_and_loses_for_puts(case):
    """EXACT_IDENTITY: the AM-GM ordering, and the fact that it **reverses**.

    A detail that is easy to get backwards and is therefore a row of its own:
    `A_arith >= A_geom` means the arithmetic *call* is worth more and the
    arithmetic *put* is worth less. Run on the same seed, so both legs see the
    same paths and the comparison is pathwise rather than statistical.
    """
    assert case.row.evidence is EvidenceClass.EXACT_IDENTITY
    spec = case.spec
    arithmetic = _mc(spec.with_averaging("arithmetic"), n_paths=20_000).value
    geometric = _mc(spec.with_averaging("geometric"), n_paths=20_000).value
    if spec.kind == "call":
        assert arithmetic > geometric, case.row.id
    else:
        assert arithmetic < geometric, case.row.id


# --------------------------------------------------------------------------
# (ii) Geometric closed forms and the published values.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", ASIAN_KNOWN_VALUE_CASES, ids=_ids(ASIAN_KNOWN_VALUE_CASES))
def test_geometric_closed_form_values(case):
    """CLOSED_FORM / PUBLISHED_BENCHMARK: the exact discrete geometric price.

    Every row goes through the dispatcher rather than through the module
    function, so the registration and the `NotSupportedError` boundary are
    exercised by the same call that checks the number.
    """
    assert case.row.evidence in {
        EvidenceClass.CLOSED_FORM,
        EvidenceClass.PUBLISHED_BENCHMARK,
    }
    value = _analytic(case.spec)
    assert abs(value - case.row.expected) <= case.row.tolerance, value


# --------------------------------------------------------------------------
# (iii) The continuous limit.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [c for c in ASIAN_CONTINUOUS_LIMIT_CASES if c.row.evidence is EvidenceClass.CONVERGENCE_ORDER],
    ids=lambda c: c.row.id,
)
def test_discrete_to_continuous_order(case):
    """CONVERGENCE_ORDER: order 1 in the number of fixings."""
    assert case.row.evidence is EvidenceClass.CONVERGENCE_ORDER
    spec = case.spec
    limit = _continuous_geometric(spec)
    errors = [
        abs(_analytic(spec.with_fixings(n)) - limit) for n in ASIAN_ORDER_LEVELS
    ]
    fit = fit_convergence_order([1.0 / n for n in ASIAN_ORDER_LEVELS], errors)
    assert abs(fit.order - case.row.expected) <= case.row.tolerance, fit.order
    assert fit.residual < 0.02, fit.residual


def test_the_haug_continuous_value_is_reached_by_a_dense_fixing_grid():
    """PUBLISHED_BENCHMARK: 4.6922, from the discrete formula at 20 000 fixings."""
    case = next(
        c
        for c in ASIAN_CONTINUOUS_LIMIT_CASES
        if c.row.id == "asian_geometric_haug_continuous_put"
    )
    assert case.row.evidence is EvidenceClass.PUBLISHED_BENCHMARK
    value = _analytic(case.spec)
    assert abs(value - case.row.expected) <= case.row.tolerance, value
    # And the convention the row documents: 90/360, not 90/365.
    on_365 = _continuous_geometric(
        type(HAUG_CONTINUOUS_SPEC)(
            spot=HAUG_CONTINUOUS_SPEC.spot,
            strike=HAUG_CONTINUOUS_SPEC.strike,
            expiry=90.0 / 365.0,
            rate=HAUG_CONTINUOUS_SPEC.rate,
            dividend=HAUG_CONTINUOUS_SPEC.dividend,
            sigma=HAUG_CONTINUOUS_SPEC.sigma,
            n_fixings=1,
            kind="put",
        )
    )
    assert abs(on_365 - case.row.expected) > 2.0e-4


# --------------------------------------------------------------------------
# (iv) The arithmetic average, by Monte Carlo.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", ASIAN_MC_CASES, ids=_ids(ASIAN_MC_CASES))
def test_arithmetic_control_variate_monte_carlo_values(case):
    """STATISTICAL: the control-variate price against an independent reference.

    The reference is a 2 000 000-path run at seed 20250913 and this run is
    200 000 paths at seed 7, so the two samples are independent and the
    comparison is a check. The tolerance is four times the root-sum-square of
    the two standard errors, which is a statement about sampling and not about
    accuracy.

    The row also asserts that the estimator *did* use the Kemna-Vorst control:
    `control_variance_factor_predicted` above 100 rules out the degenerate case
    where the control has no sample variance and the estimator silently becomes
    the plain one (pinned in `tests/test_asian_variance_ratios.py`).
    """
    assert case.row.evidence is EvidenceClass.STATISTICAL
    result = _mc(case.spec)
    assert abs(result.value - case.row.expected) <= case.row.tolerance, result.value
    meta = result.meta
    assert meta is not None
    assert meta["control_variate"] == "discounted_geometric_average_payoff"
    assert meta["control_variance_factor_predicted"] > 100.0
    # The tolerance really is the stated multiple of this run's own noise.
    assert case.row.tolerance > ASIAN_MC_STDERR_MULTIPLE * result.stderr * 0.9


# --------------------------------------------------------------------------
# (v) The approximation.
# --------------------------------------------------------------------------


def test_the_turnbull_wakeman_published_value_is_reproduced():
    """PUBLISHED_BENCHMARK **for the approximation**, not for the price."""
    case = next(
        c
        for c in ASIAN_APPROXIMATION_CASES
        if c.row.id == "asian_turnbull_wakeman_published_value"
    )
    assert case.row.evidence is EvidenceClass.PUBLISHED_BENCHMARK
    spec = case.spec
    market = spec.market()
    value = turnbull_wakeman_price(
        S=spec.spot,
        K=spec.strike,
        T=spec.expiry,
        r=market.rate(spec.expiry),
        sigma=spec.sigma,
        fixing_times=spec.fixing_times(),
        q=market.dividend_yield(spec.expiry),
        kind=spec.kind,
    )
    assert abs(value - case.row.expected) <= case.row.tolerance, value
    # ... and it is NOT the price. The reference row for the same point is the
    # Monte Carlo one, and they differ by two orders of magnitude more than the
    # residual above.
    truth_row = next(
        c for c in ASIAN_MC_CASES if c.row.id == "asian_arithmetic_mc_itm_6m_26f"
    )
    assert abs(value - truth_row.row.expected) > 100.0 * abs(value - case.row.expected)


@pytest.mark.parametrize(
    "case",
    [c for c in ASIAN_APPROXIMATION_CASES if c.row.id.startswith("asian_turnbull_wakeman_gap")],
    ids=lambda c: c.row.id,
)
def test_the_turnbull_wakeman_gap_is_what_the_row_records(case):
    """STATISTICAL: the signed gap to the simulated price, always positive.

    Reported, not asserted away. The approximation fits a lognormal to the two
    exact moments of the arithmetic average; the fitted law is more
    right-skewed than the true one, so it puts too much mass in the region that
    pays and the call is overpriced at every point measured here (+0.009% deep
    in the money to +0.89% at 40% volatility).
    """
    assert case.row.evidence is EvidenceClass.STATISTICAL
    spec = case.spec
    market = spec.market()
    approximation = turnbull_wakeman_price(
        S=spec.spot,
        K=spec.strike,
        T=spec.expiry,
        r=market.rate(spec.expiry),
        sigma=spec.sigma,
        fixing_times=spec.fixing_times(),
        q=market.dividend_yield(spec.expiry),
        kind=spec.kind,
    )
    simulated = _mc(spec)
    gap = approximation - simulated.value
    assert gap > 0.0, case.row.id
    assert abs(gap - case.row.expected) <= case.row.tolerance, gap
    # The gap is resolved, i.e. it is a bias and not this seed's noise.
    assert gap > ASIAN_MC_STDERR_MULTIPLE * simulated.stderr


def test_the_fixing_convention_is_the_one_the_case_module_documents():
    """`t_i = i T / n`, last fixing at expiry, for every spec in the layer.

    The three published values are reproduced only under this convention, so it
    is worth an assertion rather than only a docstring: a spec that built its
    schedule some other way would still price, and would still look plausible.
    """
    for case in ALL_ASIAN_CASES:
        for spec in case.specs:
            times = spec.fixing_times()
            assert times == uniform_fixing_times(spec.expiry, spec.n_fixings)
            assert times[-1] == spec.expiry
            assert len(times) == spec.n_fixings
