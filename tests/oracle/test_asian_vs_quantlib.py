"""The Asian engines against QuantLib's, analytic and Monte Carlo.

Four comparisons, and they are not all the same kind of statement.

**Discrete geometric, analytic against analytic.** QuantLib's
`AnalyticDiscreteGeometricAveragePriceAsianEngine` computes the same exactly
lognormal average this package derives in `qpl.engines.analytic.asian`, by a
different arrangement of the algebra. Agreement is therefore expected at
round-off and is measured at **2.5e-14 or better** over four points and both
kinds -- about four ulp at a price of 6 to 28. This leg checks the derivation
(and in particular the collapsed variance sum), not any discretisation.

**Continuous geometric.** QuantLib has a closed form for the *continuously*
averaged geometric Asian (Kemna-Vorst); this package does not, and reaches it
by making the fixing grid dense. So this leg is a **convergence** statement
rather than an agreement statement, and it is the only place in the repository
where the limit is supplied by an independent implementation rather than by a
second formula written here. Measured order against QuantLib's value: **1.0002**
with log-space residual 0.00024, over `n` in (20 ... 2560).

**Turnbull-Wakeman.** `ql.TurnbullWakemanAsianEngine` implements the same
two-moment lognormal fit. Agreement is measured at **3.1e-13 or better**, which
is looser than the geometric leg by an order of magnitude and for a visible
reason: the second moment is a double sum of `n^2` exponentials and the two
implementations arrange it differently, so the 73-fixing points (5329 terms)
lose about a decimal to accumulation order while the 5-fixing points agree to
the last bit.

**Arithmetic, Monte Carlo against Monte Carlo.** `ql.MCDiscreteArithmeticAPEngine`
with `controlVariate=True` uses the *same* control this package does -- the
discrete geometric average priced by its analytic engine -- which makes this
leg a check of two independent implementations of one idea rather than of two
different ideas. Nothing else is shared: different path generators, different
random number streams, different seeds. Compared at four standard errors of the
two runs combined; measured `z` at these settings: +0.72, -0.62, -0.61, +0.39.

Day count and the fixing convention
-----------------------------------
`Actual365Fixed` with 365 days makes `T = 1.0` exactly in double precision.
Fixing dates are placed at 73-day (five fixings) or 5-day (73 fixings)
intervals, both of which divide 365, so QuantLib's year fractions are exactly
`i/5` and `i/73` -- the `t_i = i T / n` convention this package uses
(`qpl.instruments.uniform_fixing_times`). **Day-count residual:** the two are
not bit-identical, because `73 i / 365` and `linspace(1/5, 1, 5)[i]` round
differently; the measured worst difference is **1.11e-16**, one ulp. With a
price sensitivity to a fixing time of order 1, that contributes well below the
1e-12 tolerance the analytic legs use, and the measured residual (2.5e-14) is
dominated by the arithmetic arrangement rather than by the schedule.

Fixing counts that do *not* divide 365 cannot be expressed exactly this way,
which is why the published Turnbull-Wakeman point (26 fixings over half a year)
is checked in `tests/test_asian_analytic.py` against its published value rather
than here against QuantLib: at `Actual/365` its fixings would fall 7.019 days
apart, and rounding them to calendar dates would compare two different
contracts.
"""

from __future__ import annotations

import math

import pytest

ql = pytest.importorskip("QuantLib")

from qpl.engines.analytic.asian import (  # noqa: E402
    discrete_geometric_price,
    turnbull_wakeman_price,
)
from qpl.engines.mc.pricers import MCConfig  # noqa: E402
from qpl.instruments.options import AsianOption, uniform_fixing_times  # noqa: E402
from qpl.market.curves import FlatDividendCurve, FlatRateCurve  # noqa: E402
from qpl.market.market import Market  # noqa: E402
from qpl.models.black_scholes import BlackScholesModel  # noqa: E402
from qpl.pricing import price  # noqa: E402
from qpl.validation import fit_convergence_order  # noqa: E402

_EVALUATION_DATE = ql.Date(15, 1, 2024)
_DAY_COUNT = ql.Actual365Fixed()
_CALENDAR = ql.NullCalendar()
_ONE_YEAR_DAYS = 365
_EXPIRY = 1.0

_ANALYTIC_TOLERANCE = 1e-12
"""Geometric leg. Worst measured residual 2.487e-14 over four points and both
kinds, so this keeps a factor of 40."""

_TURNBULL_WAKEMAN_TOLERANCE = 1e-11
"""Moment-matching leg. Worst measured residual 3.055e-13, a factor of 33. An
order of magnitude looser than the geometric leg because `E[A^2]` is a double
sum of `n^2` terms that the two implementations accumulate in different orders;
the 5-fixing points agree to 1.2e-14 and the 73-fixing ones to 3.1e-13."""

_CONTINUOUS_TOLERANCE = 3.5e-4
"""Continuous-limit leg at `ASIAN_ORACLE_DENSE_FIXINGS`. This is a
discretisation budget, not an agreement budget: the discrete price approaches
the continuous one at order 1 with a measured constant of 4.71, so 20 000
fixings leave 2.355e-04 and the tolerance keeps a factor of 1.5. Making it
tighter means more fixings, not a better formula."""

_MC_STDERR_MULTIPLE = 4.0
_MC_SAMPLES = 100_000
_MC_QL_SEED = 42
_MC_QPL_SEED = 7

ASIAN_ORACLE_DENSE_FIXINGS = 20_000
_ORDER_LEVELS = (20, 40, 80, 160, 320, 640, 1280, 2560)

# (id, spot, strike, rate, dividend, sigma, n_fixings, day step)
# `n_fixings * step == 365` in every row, so QuantLib's year fractions are
# exactly `i / n_fixings`.
_POINTS = (
    ("atm_1y_5f", 100.0, 100.0, 0.05, 0.00, 0.20, 5, 73),
    ("atm_1y_73f", 100.0, 100.0, 0.05, 0.00, 0.20, 73, 5),
    ("otm_1y_73f_div", 100.0, 110.0, 0.03, 0.01, 0.25, 73, 5),
    ("itm_1y_5f_div", 120.0, 90.0, 0.03, 0.05, 0.35, 5, 73),
)
_ARGNAMES = ("spot", "strike", "rate", "div", "sigma", "n_fixings", "step")
_IDS = [row[0] for row in _POINTS]
_ARGS = [row[1:] for row in _POINTS]


def _process(spot: float, rate: float, div: float, sigma: float):
    ql.Settings.instance().evaluationDate = _EVALUATION_DATE
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(
            ql.FlatForward(_EVALUATION_DATE, div, _DAY_COUNT)
        ),
        ql.YieldTermStructureHandle(
            ql.FlatForward(_EVALUATION_DATE, rate, _DAY_COUNT)
        ),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(_EVALUATION_DATE, _CALENDAR, sigma, _DAY_COUNT)
        ),
    )


def _fixing_dates(n_fixings: int, step: int):
    return [
        _EVALUATION_DATE + ql.Period(step * (i + 1), ql.Days) for i in range(n_fixings)
    ]


def _ql_option(
    average_type, kind: str, strike: float, n_fixings: int, step: int
):
    dates = _fixing_dates(n_fixings, step)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if kind == "call" else ql.Option.Put, strike
    )
    running = 1.0 if average_type == ql.Average.Geometric else 0.0
    return ql.DiscreteAveragingAsianOption(
        average_type, running, 0, dates, payoff, ql.EuropeanExercise(dates[-1])
    )


def _qpl_market(spot: float, rate: float, div: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(rate, allow_negative=True),
        dividend_curve=FlatDividendCurve(div, allow_negative=True),
    )


# --------------------------------------------------------------------------
# Is it the same contract, on the same schedule?
# --------------------------------------------------------------------------


@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_the_two_fixing_schedules_agree_to_one_ulp(
    spot, strike, rate, div, sigma, n_fixings, step
):
    """EXACT_IDENTITY on the schedule, before any pricing.

    QuantLib derives its fixing times from calendar dates through a day count;
    this package takes them as year fractions. The comparison only means
    something if the two describe the same monitoring dates, and `n * step =
    365` under `Actual365Fixed` is what makes that true. Measured worst
    difference: 1.11e-16, one ulp -- not zero, because `73 i / 365` and
    `linspace(1/5, 1, 5)[i]` are different computations of the same real number.
    """
    ql.Settings.instance().evaluationDate = _EVALUATION_DATE
    ql_times = [
        _DAY_COUNT.yearFraction(_EVALUATION_DATE, d)
        for d in _fixing_dates(n_fixings, step)
    ]
    ours = uniform_fixing_times(_EXPIRY, n_fixings)
    assert len(ql_times) == len(ours)
    assert max(abs(a - b) for a, b in zip(ql_times, ours, strict=True)) <= 2.3e-16
    assert ql_times[-1] == _EXPIRY == ours[-1]


# --------------------------------------------------------------------------
# Analytic against analytic.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_discrete_geometric_agrees_to_round_off(
    kind, spot, strike, rate, div, sigma, n_fixings, step
):
    """Evidence class: INDEPENDENT_ENGINE, at round-off.

    Both engines evaluate a closed form for the same exactly lognormal average,
    so the only thing that can differ is how the arithmetic is arranged. This
    package collapses the covariance double sum to
    `sum_i (2(n-i)+1) t_i` and evaluates one dot product; QuantLib accumulates
    its own weights. Measured worst residual over these eight cells:
    **2.487e-14**, i.e. about four ulp at prices from 1.45 to 28.05.

    The 73-fixing rows are the ones that would catch an index slip in the
    collapsed formula, because the weights there run from 145 down to 1 and a
    reversed ordering would be wrong by a large factor rather than by a
    round-off.
    """
    process = _process(spot, rate, div, sigma)
    option = _ql_option(ql.Average.Geometric, kind, strike, n_fixings, step)
    option.setPricingEngine(
        ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process)
    )

    market = _qpl_market(spot, rate, div)
    ours = discrete_geometric_price(
        S=spot,
        K=strike,
        T=_EXPIRY,
        r=market.rate(_EXPIRY),
        sigma=sigma,
        fixing_times=uniform_fixing_times(_EXPIRY, n_fixings),
        q=market.dividend_yield(_EXPIRY),
        kind=kind,
    )
    assert ours == pytest.approx(option.NPV(), abs=_ANALYTIC_TOLERANCE)

    # And the dispatcher route returns the same number to within one ulp:
    # `Market.rate(t)` recovers the rate as `-log(df)/t` rather than storing it.
    through_dispatcher = price(
        AsianOption(
            kind, strike, _EXPIRY, uniform_fixing_times(_EXPIRY, n_fixings), "geometric"
        ),
        BlackScholesModel(sigma=sigma),
        market,
    ).value
    assert through_dispatcher == pytest.approx(ours, abs=1e-13)


def test_the_geometric_agreement_is_not_vacuous():
    """A geometric Asian is not a vanilla, and 1e-12 is not a wide net.

    The same market and strike priced as a plain European call is 10.4506
    against this Asian's 6.4945 -- a gap of 3.96, twelve decimal orders above
    the tolerance the rows above use. And the arithmetic average of the same
    path is worth 6.70, 0.21 more than the geometric one, so the tolerance is
    also far below the difference between the two averagings.
    """
    process = _process(100.0, 0.05, 0.0, 0.20)
    maturity = _EVALUATION_DATE + ql.Period(_ONE_YEAR_DAYS, ql.Days)
    vanilla = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, 100.0), ql.EuropeanExercise(maturity)
    )
    vanilla.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    asian = _ql_option(ql.Average.Geometric, "call", 100.0, 5, 73)
    asian.setPricingEngine(
        ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process)
    )

    assert vanilla.NPV() - asian.NPV() > 3.0
    assert vanilla.NPV() == pytest.approx(10.4506, abs=1e-3)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_turnbull_wakeman_agrees_with_quantlibs_own_moment_matching(
    kind, spot, strike, rate, div, sigma, n_fixings, step
):
    """Evidence class: INDEPENDENT_ENGINE, on the **approximation**.

    Two implementations of the same two-moment lognormal fit agreeing tells you
    that both computed `E[A]` and `E[A^2]` correctly. It tells you nothing about
    whether the fitted lognormal prices the option -- that gap is measured
    against Monte Carlo in `tests/test_asian_mc.py` and recorded in
    `qpl.cases.asian_black_scholes`, and is 10 to 500 times larger than the
    tolerance here.

    Measured residuals: 1.2e-14 at the 5-fixing points and 3.1e-13 at the
    73-fixing ones. The `n^2` double sum for `E[A^2]` is 5329 terms at 73
    fixings, and the two implementations accumulate it in different orders; the
    residual growing with `n` rather than staying flat is what that looks like.
    """
    process = _process(spot, rate, div, sigma)
    option = _ql_option(ql.Average.Arithmetic, kind, strike, n_fixings, step)
    option.setPricingEngine(ql.TurnbullWakemanAsianEngine(process))

    market = _qpl_market(spot, rate, div)
    ours = turnbull_wakeman_price(
        S=spot,
        K=strike,
        T=_EXPIRY,
        r=market.rate(_EXPIRY),
        sigma=sigma,
        fixing_times=uniform_fixing_times(_EXPIRY, n_fixings),
        q=market.dividend_yield(_EXPIRY),
        kind=kind,
    )
    assert ours == pytest.approx(option.NPV(), abs=_TURNBULL_WAKEMAN_TOLERANCE)


# --------------------------------------------------------------------------
# The continuous limit, supplied by the other engine.
# --------------------------------------------------------------------------


def test_dense_fixings_converge_to_quantlibs_continuous_geometric_price():
    """INDEPENDENT_ENGINE + CONVERGENCE_ORDER: order 1 onto QuantLib's value.

    This package has no continuous-averaging engine and does not need one: the
    discrete formula converges to it. What makes this row worth having is that
    the limit comes from an independent implementation, so the measured order is
    not a statement about two formulas written in the same file.

    Measured against `ql.AnalyticContinuousGeometricAveragePriceAsianEngine` at
    `S = K = 100`, `r = 5%`, `q = 0`, `sigma = 20%`, `T = 1` (QuantLib's value:
    5.546818633789218), over `n` in (20 ... 2560):

        fitted order 1.0002, log-space residual 0.00024,
        errors 1.837e-01 down to 1.840e-03, one-signed and halving.

    That is the same 1.0002 the in-repo continuous formula produces in
    `tests/test_asian_analytic.py`, which is the point: the rate is a property
    of the right-endpoint fixing convention, not of either implementation.

    At 20 000 fixings the residual is 2.355e-04, so the constant is 4.71 and the
    tolerance below is a discretisation budget rather than an agreement one.
    """
    process = _process(100.0, 0.05, 0.0, 0.20)
    continuous = ql.ContinuousAveragingAsianOption(
        ql.Average.Geometric,
        ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
        ql.EuropeanExercise(_EVALUATION_DATE + ql.Period(_ONE_YEAR_DAYS, ql.Days)),
    )
    continuous.setPricingEngine(
        ql.AnalyticContinuousGeometricAveragePriceAsianEngine(process)
    )
    limit = continuous.NPV()

    market = _qpl_market(100.0, 0.05, 0.0)

    def discrete(n: int) -> float:
        return discrete_geometric_price(
            S=100.0,
            K=100.0,
            T=_EXPIRY,
            r=market.rate(_EXPIRY),
            sigma=0.20,
            fixing_times=uniform_fixing_times(_EXPIRY, n),
            q=market.dividend_yield(_EXPIRY),
        )

    signed = [discrete(n) - limit for n in _ORDER_LEVELS]
    fit = fit_convergence_order(
        [1.0 / n for n in _ORDER_LEVELS], [abs(e) for e in signed]
    )
    assert fit.order == pytest.approx(1.0, abs=0.05), fit.order
    assert fit.residual < 0.01, fit.residual
    assert all(e > 0 for e in signed) or all(e < 0 for e in signed), signed

    dense = discrete(ASIAN_ORACLE_DENSE_FIXINGS)
    assert abs(dense - limit) < _CONTINUOUS_TOLERANCE, dense - limit
    # The constant of the order-1 law, pinned: err ~ 4.71 / n.
    assert abs(dense - limit) * ASIAN_ORACLE_DENSE_FIXINGS == pytest.approx(
        4.71, rel=0.05
    )


# --------------------------------------------------------------------------
# Monte Carlo against Monte Carlo, with the same control variate.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_control_variate_monte_carlo_agrees_with_quantlibs(
    spot, strike, rate, div, sigma, n_fixings, step
):
    """Evidence class: INDEPENDENT_ENGINE + STATISTICAL.

    `ql.MCDiscreteArithmeticAPEngine(..., controlVariate=True)` uses the
    discrete geometric average as its control, priced by its analytic engine --
    the same construction this package uses, arrived at independently from the
    same 1990 paper. Everything else differs: path generation, the random
    number stream, the seed (42 against 7), and the standard-error formula
    (QuantLib reports the sample statistics of its controlled estimator; this
    package reports the `ddof=2` regression residual).

    Compared at four standard errors of the two runs combined, 100 000 samples
    each. Measured `z`:

        point             QuantLib          qpl                  z
        atm_1y_5f          6.705307         6.704296          +0.72
        atm_1y_73f         5.827344         5.828160          -0.62
        otm_1y_73f_div     2.627535         2.628649          -0.61
        itm_1y_5f_div     28.981187        28.979279          +0.39

    The standard errors are also worth comparing: at `atm_1y_73f` QuantLib
    reports 1.10e-03 and this package 7.05e-04 on the same sample count. That is
    not a claim to a better estimator -- the two report different statistics of
    different estimators -- but it does mean the tolerance here is set mostly by
    QuantLib's leg.
    """
    process = _process(spot, rate, div, sigma)
    theirs = _ql_option(ql.Average.Arithmetic, "call", strike, n_fixings, step)
    theirs.setPricingEngine(
        ql.MCDiscreteArithmeticAPEngine(
            process,
            "pseudorandom",
            False,  # brownianBridge
            False,  # antitheticVariate
            True,  # controlVariate
            _MC_SAMPLES,
            ql.nullDouble(),
            ql.nullInt(),
            _MC_QL_SEED,
        )
    )

    market = _qpl_market(spot, rate, div)
    ours = price(
        AsianOption(
            "call",
            strike,
            _EXPIRY,
            uniform_fixing_times(_EXPIRY, n_fixings),
            "arithmetic",
        ),
        BlackScholesModel(sigma=sigma),
        market,
        method="mc",
        cfg=MCConfig(
            n_paths=_MC_SAMPLES,
            seed=_MC_QPL_SEED,
            variance_reduction="control_variate",
        ),
    )

    combined = math.hypot(theirs.errorEstimate(), ours.stderr)
    z = (theirs.NPV() - ours.value) / combined
    assert abs(z) < _MC_STDERR_MULTIPLE, (theirs.NPV(), ours.value, z)

    # Both control variates actually engaged: a run whose control had collapsed
    # would still price, just far more noisily (see
    # `tests/test_asian_variance_ratios.py`).
    assert ours.meta is not None
    assert ours.meta["control_variance_factor_predicted"] > 50.0
    assert theirs.errorEstimate() < 0.01 * theirs.NPV()


def test_quantlib_refuses_nothing_that_this_package_prices_and_vice_versa():
    """A scope note, pinned: the two packages do not cover the same set.

    QuantLib prices *average-strike* Asians
    (`AnalyticDiscreteGeometricAverageStrikeAsianEngine`) and this package
    deliberately does not -- `AsianOption` is fixed-strike only, and a floating
    strike is a different contract rather than a flag. Conversely this package
    refuses `method="tree"` and `method="pde"` for an Asian while QuantLib ships
    `FdBlackScholesAsianEngine`, which carries the average as a second state
    variable; that is a later slice.

    Recorded as a test rather than a comment so that the day one of these
    changes, the boundary is re-examined deliberately.
    """
    assert hasattr(ql, "AnalyticDiscreteGeometricAverageStrikeAsianEngine")
    assert hasattr(ql, "FdBlackScholesAsianEngine")
    from qpl.exceptions import NotSupportedError

    option = AsianOption(
        "call", 100.0, _EXPIRY, uniform_fixing_times(_EXPIRY, 5), "arithmetic"
    )
    with pytest.raises(NotSupportedError):
        price(
            option,
            BlackScholesModel(sigma=0.2),
            _qpl_market(100.0, 0.05, 0.0),
            method="analytic",
        )
