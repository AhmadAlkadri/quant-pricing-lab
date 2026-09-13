"""The digital engines against QuantLib's, analytic and finite-difference.

Two comparisons with very different characters.

**Analytic.** QuantLib's `AnalyticEuropeanEngine` prices a
`ql.CashOrNothingPayoff` through its `BlackCalculator`, which is a different
arrangement of the same algebra this package derives in
`qpl.engines.analytic.digital`. Agreement is therefore expected at machine
precision, and measured at **2.2e-16 or better on the price and every one of
the five Greeks** -- so this leg checks the derivation, not the discretisation.

**Finite differences.** Here the two engines share almost nothing:

- `qpl` marches a grid uniform in the *spot* over `[0, 4 S]`, with
  `strike_alignment="midpoint"` putting the jump exactly on a cell face, and
  Rannacher start-up;
- QuantLib's `FdBlackScholesVanillaEngine` marches a grid uniform in
  `log S` around the forward, with nothing aligned to the strike, and
  `dampingSteps = 0` by default (pinned in `test_pde_vs_quantlib.py`).

At `tGrid = xGrid = n_s = n_t = 800` they agree to 3.514e-05 on the price,
1.056e-06 on delta and 9.922e-08 on gamma, against tolerances derived from
both engines' own errors below.

The interesting result is the *shape* of QuantLib's error rather than its
size. On `n` in (100, 200, 400, 800) its digital price error is
-9.20e-03, +7.79e-05, -2.67e-05, -1.80e-07 at the money: three sign changes,
five decimal orders of magnitude, log-space fit residual 0.85. That is the
same sawtooth the CRR lattice shows on a digital and for the same reason --
nothing places the jump consistently relative to the nodes, so which side of a
node the strike lands on changes with `n`. `qpl`'s errors on the same grids are
+1.22e-05, +3.38e-06, +8.80e-07, +2.23e-07: monotone, one-signed, residual
0.023. QuantLib is *sometimes ahead* at a given `n` and has no usable order;
this package is behind at the finest grid tested and has one. Both statements
are asserted.

Day count: `Actual365Fixed` with 365 days makes `T = 1.0` exactly in double
precision, so the two engines are compared at the same maturity with no
day-count residual.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

ql = pytest.importorskip("QuantLib")

from qpl.engines.analytic.digital import (  # noqa: E402
    digital_price,
    greeks_digital as analytic_greeks,
)
from qpl.engines.pde.pricers import PDEConfig  # noqa: E402
from qpl.instruments.options import DigitalOption  # noqa: E402
from qpl.market.curves import FlatDividendCurve, FlatRateCurve  # noqa: E402
from qpl.market.market import Market  # noqa: E402
from qpl.models.black_scholes import BlackScholesModel  # noqa: E402
from qpl.pricing import greeks, price  # noqa: E402
from qpl.validation import fit_convergence_order  # noqa: E402

_EVALUATION_DATE = ql.Date(15, 1, 2024)
_DAY_COUNT = ql.Actual365Fixed()
_CALENDAR = ql.NullCalendar()
_ONE_YEAR_DAYS = 365
_EXPIRY = 1.0

_COMPARISON_N = 800
"""`tGrid = xGrid = n_s = n_t` at which the two grids are compared."""

_REFINEMENT_LEVELS = (100, 200, 400, 800)

# Tolerances for the finite-difference leg, derived from both engines' own
# errors at `n = 800` rather than chosen. Worst over the three points and both
# kinds:
#
# | quantity | worst |qpl - BS| | worst |QL - BS| | worst gap |
# |----------|----------------|-----------------|-----------|
# | price    | 2.331e-05      | 1.184e-05       | 3.514e-05 |
# | delta    | 9.621e-07      | 2.421e-07       | 1.056e-06 |
# | gamma    | 9.955e-08      | 2.717e-09       | 9.922e-08 |
#
# Each tolerance keeps a factor of about 3.4 over the worst gap, which is
# roughly one refinement level of an order-2 sequence.
_PRICE_TOLERANCE = 1.2e-4
_DELTA_TOLERANCE = 3.5e-6
_GAMMA_TOLERANCE = 3.5e-7

_ANALYTIC_TOLERANCE = 1e-14
"""Both engines evaluate the same closed form; measured agreement is 2.2e-16."""

# (id, spot, strike, rate, dividend, sigma, cash)
_POINTS = (
    ("atm_1y", 100.0, 100.0, 0.05, 0.00, 0.20, 1.0),
    ("otm_1y_div", 100.0, 110.0, 0.03, 0.01, 0.25, 1.0),
    ("itm_1y_div", 120.0, 90.0, 0.03, 0.05, 0.35, 2.5),
)
_ARGNAMES = ("spot", "strike", "rate", "div", "sigma", "cash")
_IDS = [row[0] for row in _POINTS]
_ARGS = [row[1:] for row in _POINTS]


def _process(spot: float, rate: float, div: float, sigma: float):
    ql.Settings.instance().evaluationDate = _EVALUATION_DATE
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(_EVALUATION_DATE, div, _DAY_COUNT)),
        ql.YieldTermStructureHandle(ql.FlatForward(_EVALUATION_DATE, rate, _DAY_COUNT)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(_EVALUATION_DATE, _CALENDAR, sigma, _DAY_COUNT)
        ),
    )


def _ql_payoff(kind: str, strike: float, cash: float):
    return ql.CashOrNothingPayoff(
        ql.Option.Call if kind == "call" else ql.Option.Put, strike, cash
    )


def _ql_option(kind: str, strike: float, cash: float):
    maturity = _EVALUATION_DATE + ql.Period(_ONE_YEAR_DAYS, ql.Days)
    return ql.VanillaOption(
        _ql_payoff(kind, strike, cash), ql.EuropeanExercise(maturity)
    )


def _qpl_triple(kind: str, spot: float, strike: float, rate: float, div: float, sigma: float, cash: float):
    return (
        DigitalOption(kind=kind, strike=strike, expiry=_EXPIRY, cash=cash),
        BlackScholesModel(sigma=sigma),
        Market(
            spot=spot,
            rate_curve=FlatRateCurve(rate),
            dividend_curve=FlatDividendCurve(div),
        ),
    )


def _qpl_cfg(n: int, *, alignment: str = "midpoint", time_stepping: str = "rannacher"):
    return PDEConfig(
        n_s=n,
        n_t=n,
        theta=0.5,
        s_max_multiplier=4.0,
        strike_alignment=alignment,  # type: ignore[arg-type]
        time_stepping=time_stepping,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# Is it the same contract?
# --------------------------------------------------------------------------


def test_quantlib_uses_the_same_strict_payoff_convention() -> None:
    """Evidence class: EXACT_IDENTITY on the contract, before any pricing.

    `ql.CashOrNothingPayoff` pays `cash` when `S - K > 0` for a call and when
    `K - S > 0` for a put, so a point exactly on the strike pays nothing either
    way -- which is the convention `qpl.instruments.payoffs.digital_payoff`
    adopts, and the reason the two can be compared at all. Checked directly on
    the payoff objects rather than inferred from a price.
    """
    from qpl.instruments.payoffs import digital_payoff

    call = _ql_payoff("call", 100.0, 2.5)
    put = _ql_payoff("put", 100.0, 2.5)

    for spot in (99.0, 99.999999, 100.0, 100.000001, 101.0):
        assert call(spot) == digital_payoff(spot, 100.0, 2.5, "call")
        assert put(spot) == digital_payoff(spot, 100.0, 2.5, "put")

    assert call(100.0) == 0.0 and put(100.0) == 0.0


# --------------------------------------------------------------------------
# Analytic against analytic.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_analytic_price_and_greeks_agree_to_machine_precision(
    kind: str,
    spot: float,
    strike: float,
    rate: float,
    div: float,
    sigma: float,
    cash: float,
) -> None:
    """Evidence class: INDEPENDENT_ENGINE, at round-off.

    Both engines evaluate a closed form, so the only thing that can differ is
    the arrangement of the arithmetic. Measured worst residual over these six
    cells: **2.220e-16** on the price, 3.469e-18 on delta, 2.168e-19 on gamma,
    and at or below 1.665e-16 on vega, theta and rho.

    QuantLib's `vega()` is per unit of volatility (not per volatility point),
    and its `theta()` is per year, which is what this package reports too; no
    unit conversion is applied below, and if either convention changed the
    test would fail rather than silently rescale.
    """
    process = _process(spot, rate, div, sigma)
    option = _ql_option(kind, strike, cash)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    triple = _qpl_triple(kind, spot, strike, rate, div, sigma, cash)
    ours_price = digital_price(
        S=spot, K=strike, T=_EXPIRY, r=rate, sigma=sigma, q=div, cash=cash, kind=kind
    )
    ours = analytic_greeks(*triple)

    assert ours_price == pytest.approx(option.NPV(), abs=_ANALYTIC_TOLERANCE)
    assert ours.delta == pytest.approx(option.delta(), abs=_ANALYTIC_TOLERANCE)
    assert ours.gamma == pytest.approx(option.gamma(), abs=_ANALYTIC_TOLERANCE)
    assert ours.vega == pytest.approx(option.vega(), abs=_ANALYTIC_TOLERANCE)
    assert ours.theta == pytest.approx(option.theta(), abs=_ANALYTIC_TOLERANCE)
    assert ours.rho == pytest.approx(option.rho(), abs=_ANALYTIC_TOLERANCE)

    # The dispatcher route returns the same number as the module route, to
    # within one ulp: `Market.rate(t)` recovers the rate as `-log(df) / t`
    # rather than storing it, so the two can differ in the last bit.
    assert price(*triple).value == pytest.approx(ours_price, abs=1e-15)


def test_the_analytic_agreement_is_not_vacuous() -> None:
    """A digital is not a vanilla, and 1e-14 is not a wide net.

    The same maturity and strike priced as a plain vanilla differs by 9.92 --
    seventeen decimal orders above the tolerance the digital rows use -- so the
    machine-precision agreement above is a statement about the digital formula
    and not about two engines both returning something plausible.
    """
    process = _process(100.0, 0.05, 0.0, 0.20)
    digital = _ql_option("call", 100.0, 1.0)
    digital.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    maturity = _EVALUATION_DATE + ql.Period(_ONE_YEAR_DAYS, ql.Days)
    vanilla = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, 100.0), ql.EuropeanExercise(maturity)
    )
    vanilla.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    assert abs(vanilla.NPV() - digital.NPV()) > 9.0


# --------------------------------------------------------------------------
# Grid against grid.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_finite_difference_engines_agree_at_the_comparison_grid(
    kind: str,
    spot: float,
    strike: float,
    rate: float,
    div: float,
    sigma: float,
    cash: float,
) -> None:
    """Evidence class: INDEPENDENT_ENGINE.

    Two grids that share no construction: uniform in spot with the jump on a
    cell face and Rannacher start-up, against uniform in log-spot with nothing
    aligned and no damping. The tolerances are the module constants, derived
    from both engines' measured errors at `n = 800`; each engine's own residual
    against the closed form is asserted separately, so a failure says which one
    moved.
    """
    process = _process(spot, rate, div, sigma)
    quantlib = _ql_option(kind, strike, cash)
    quantlib.setPricingEngine(
        ql.FdBlackScholesVanillaEngine(
            process, _COMPARISON_N, _COMPARISON_N, 0, ql.FdmSchemeDesc.Douglas()
        )
    )

    triple = _qpl_triple(kind, spot, strike, rate, div, sigma, cash)
    cfg = _qpl_cfg(_COMPARISON_N)
    ours_price = price(*triple, method="pde", cfg=cfg).value
    ours_greeks = greeks(*triple, method="pde", cfg=cfg)

    closed_form = digital_price(
        S=spot, K=strike, T=_EXPIRY, r=rate, sigma=sigma, q=div, cash=cash, kind=kind
    )
    exact = analytic_greeks(*triple)

    comparisons = (
        ("price", ours_price, quantlib.NPV(), closed_form, _PRICE_TOLERANCE),
        ("delta", ours_greeks.delta, quantlib.delta(), exact.delta, _DELTA_TOLERANCE),
        ("gamma", ours_greeks.gamma, quantlib.gamma(), exact.gamma, _GAMMA_TOLERANCE),
    )
    for name, ours, theirs, reference, tolerance in comparisons:
        assert abs(ours - theirs) <= tolerance, (name, ours, theirs)
        assert abs(ours - reference) <= tolerance, (name, "qpl", ours, reference)
        assert abs(theirs - reference) <= tolerance, (name, "ql", theirs, reference)


@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_quantlib_has_no_usable_order_on_a_digital_and_this_engine_does(
    spot: float, strike: float, rate: float, div: float, sigma: float, cash: float
) -> None:
    """Evidence class: NEGATIVE_FINDING for QuantLib, CONVERGENCE_ORDER for `qpl`.

    Measured over `n` in (100, 200, 400, 800), call leg, against the closed
    form:

    | point      | QL fitted order / residual | qpl fitted order / residual |
    |------------|----------------------------|-----------------------------|
    | atm_1y     | 4.8464 / 0.8547            | 1.9248 / 0.0233             |
    | otm_1y_div | 1.6990 / 0.5822            | 2.0114 / 0.0082             |
    | itm_1y_div | 1.9551 / 0.8400            | 2.0149 / 0.0088             |

    QuantLib's residuals are 25x to 100x this package's, and its errors change
    sign inside every sequence: at the money they run -9.20e-03, +7.79e-05,
    -2.67e-05, -1.80e-07. A slope through that measures nothing -- the 4.85 in
    the first row is an artefact of one enormous coarse-grid point, not an
    order. The mechanism is the same one CRR shows on a digital: the log-spot
    mesher places no node consistently relative to the jump, so the error is a
    sawtooth in `n`.

    `qpl`'s sequences are monotone and one-signed at all three points, which is
    what "second order" is allowed to mean. Note that this is *not* a claim to
    be more accurate: at the money QuantLib's `n = 800` error (1.80e-07) is
    smaller than this engine's (2.23e-07). It is a claim about predictability
    -- refining QuantLib's grid is not reliably an improvement, and refining
    this one is.

    `dampingSteps = 2` does not change the picture (fitted 4.7897, 1.8355,
    2.1526 with residuals 0.84, 0.55, 1.17), so the erratic behaviour is the
    mesher's and not the time scheme's.
    """
    process = _process(spot, rate, div, sigma)
    triple = _qpl_triple("call", spot, strike, rate, div, sigma, cash)
    closed_form = digital_price(
        S=spot, K=strike, T=_EXPIRY, r=rate, sigma=sigma, q=div, cash=cash, kind="call"
    )

    ql_errors, qpl_errors = [], []
    for n in _REFINEMENT_LEVELS:
        option = _ql_option("call", strike, cash)
        option.setPricingEngine(
            ql.FdBlackScholesVanillaEngine(process, n, n, 0, ql.FdmSchemeDesc.Douglas())
        )
        ql_errors.append(option.NPV() - closed_form)
        qpl_errors.append(price(*triple, method="pde", cfg=_qpl_cfg(n)).value - closed_form)

    ql_fit = fit_convergence_order(
        [1.0 / n for n in _REFINEMENT_LEVELS], [abs(e) for e in ql_errors]
    )
    qpl_fit = fit_convergence_order(
        [1.0 / n for n in _REFINEMENT_LEVELS], [abs(e) for e in qpl_errors]
    )

    # `qpl`: a clean, monotone, one-signed order-2 sequence.
    assert abs(qpl_fit.order - 2.0) <= 0.2, (qpl_fit.order, qpl_errors)
    assert qpl_fit.residual < 0.05, qpl_fit.residual
    assert all(abs(a) > abs(b) for a, b in pairwise(qpl_errors)), qpl_errors
    assert all(e > 0 for e in qpl_errors) or all(e < 0 for e in qpl_errors), qpl_errors

    # QuantLib: not a power law. Either the residual is large or the sequence
    # changes sign; at these three points both are true.
    assert ql_fit.residual > 10.0 * qpl_fit.residual, (ql_fit.residual, qpl_fit.residual)
    assert int(np.sum(np.diff(np.sign(ql_errors)) != 0)) >= 1, ql_errors


def test_quantlib_is_far_worse_than_this_engine_on_a_coarse_grid() -> None:
    """Evidence class: NEGATIVE_FINDING, the practical form of the above.

    At `n = 100` and the money, QuantLib's digital price error is **-9.20e-03**
    -- 1.7% of a price of 0.5323 -- against **+1.22e-05** here, a factor of
    755. That is the cost of not placing the jump relative to the nodes, and it
    is the same coarse-grid failure the unaligned `qpl` grid shows when
    `strike_alignment="none"` (-3.76e-02 at `n = 100`).
    """
    process = _process(100.0, 0.05, 0.0, 0.20)
    closed_form = digital_price(S=100.0, K=100.0, T=_EXPIRY, r=0.05, sigma=0.20, kind="call")

    option = _ql_option("call", 100.0, 1.0)
    option.setPricingEngine(
        ql.FdBlackScholesVanillaEngine(process, 100, 100, 0, ql.FdmSchemeDesc.Douglas())
    )
    triple = _qpl_triple("call", 100.0, 100.0, 0.05, 0.0, 0.20, 1.0)

    ql_error = abs(option.NPV() - closed_form)
    aligned = abs(price(*triple, method="pde", cfg=_qpl_cfg(100)).value - closed_form)
    unaligned = abs(
        price(*triple, method="pde", cfg=_qpl_cfg(100, alignment="none", time_stepping="theta")).value
        - closed_form
    )

    assert ql_error > 100.0 * aligned, (ql_error, aligned)
    assert unaligned > 100.0 * aligned, (unaligned, aligned)
