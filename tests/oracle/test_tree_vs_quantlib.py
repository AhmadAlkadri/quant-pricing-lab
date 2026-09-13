"""The CRR tree against QuantLib, and what "CoxRossRubinstein" actually means.

The slice this file was written for expected agreement with QuantLib's
`BinomialVanillaEngine<CoxRossRubinstein>` to 1e-10 at the same `n`. That
expectation is **wrong**, and the tests below encode what is true instead.

Both engines build the same *lattice*: `u = exp(sigma sqrt(dt))`, `d = 1/u`.
They differ in the risk-neutral probability.

- `qpl` uses the probability that reprices the one-step forward exactly,
  `p = (e^{(r-q) dt} - d) / (u - d)`, which is the choice in Cox, Ross and
  Rubinstein (1979), Journal of Financial Economics 7, 229-263.
- QuantLib's `CoxRossRubinstein` class uses the log-space probability
  `p = 1/2 + (r - q - sigma^2/2) dt / (2 sigma sqrt(dt))`, i.e. it matches the
  mean of the *log* return rather than the mean of the return.

Expanding both in `x = sigma sqrt(dt)` gives `1/2 + (r-q) dt / (2x) - x/4` for
each, so they agree to `O(x**2)` and differ at `O(x**3) = O(dt**1.5)` per
step. Over `n` steps that is an `O(1/n)` difference in price -- the same order
as each engine's own discretisation error. No amount of refinement makes the
two agree to 1e-10 at a shared `n`; they agree in the limit, which is a
different statement and is tested as such.

Day count: `Actual365Fixed` makes `T = days / 365`, so `T = 1.0` (365 days)
and `T = 2.0` (730 days) are exactly representable and carry no day-count
residual at all. `T = 0.5` would need 182.5 days and is not representable, so
it is not used here.

Evidence classes: INDEPENDENT_ENGINE for the closed-form comparison, for the
identification of QuantLib's probability, and for the shared first-order limit
at the money; NEGATIVE_FINDING for the pinned `O(1/n)` discrepancy and for the
erratic off-the-money error constant, both of which exist here so that they
cannot later be mistaken for a bug in either engine.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

ql = pytest.importorskip("QuantLib")

from qpl.engines.tree import TreeConfig  # noqa: E402
from qpl.instruments.options import EuropeanOption  # noqa: E402
from qpl.market.curves import FlatDividendCurve, FlatRateCurve  # noqa: E402
from qpl.market.market import Market  # noqa: E402
from qpl.models.black_scholes import BlackScholesModel  # noqa: E402
from qpl.pricing import price  # noqa: E402
from qpl.validation import fit_convergence_order  # noqa: E402

_EVALUATION_DATE = ql.Date(15, 1, 2024)
_DAY_COUNT = ql.Actual365Fixed()
_CALENDAR = ql.NullCalendar()

# (id, spot, strike, expiry_years, expiry_days, rate, dividend, sigma)
_POINTS = (
    ("atm_1y", 100.0, 100.0, 1.0, 365, 0.05, 0.00, 0.20),
    ("otm_2y_div", 100.0, 110.0, 2.0, 730, 0.03, 0.01, 0.25),
    ("itm_1y_div", 120.0, 90.0, 1.0, 365, 0.03, 0.05, 0.35),
)
_IDS = [row[0] for row in _POINTS]
_ARGS = [row[1:] for row in _POINTS]
_ARGNAMES = ("spot", "strike", "expiry", "days", "rate", "div", "sigma")


def _quantlib_option(
    kind: str,
    spot: float,
    strike: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
) -> tuple[object, object, float]:
    """Build the QuantLib option/process pair and return the realised `T`."""
    ql.Settings.instance().evaluationDate = _EVALUATION_DATE
    maturity = _EVALUATION_DATE + ql.Period(days, ql.Days)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(_EVALUATION_DATE, div, _DAY_COUNT)),
        ql.YieldTermStructureHandle(ql.FlatForward(_EVALUATION_DATE, rate, _DAY_COUNT)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(_EVALUATION_DATE, _CALENDAR, sigma, _DAY_COUNT)
        ),
    )
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if kind == "call" else ql.Option.Put, strike
    )
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(maturity))
    return option, process, _DAY_COUNT.yearFraction(_EVALUATION_DATE, maturity)


def _qpl_market(spot: float, rate: float, div: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(rate),
        dividend_curve=FlatDividendCurve(div),
    )


@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_day_count_makes_the_expiry_exact(
    spot: float,
    strike: float,
    expiry: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """No day-count residual to account for: `T` is a whole number of years.

    `Actual365Fixed` computes `T = days / 365`, so 365 and 730 days are
    exactly 1.0 and 2.0 in double precision. Everything downstream compares
    two engines at the same `T`, not at two nearly-equal `T`s.
    """
    _, _, realised = _quantlib_option("call", spot, strike, days, rate, div, sigma)
    assert realised == expiry


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_qpl_analytic_matches_quantlib_analytic(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: INDEPENDENT_ENGINE.

    This is the leg where 1e-10 agreement is genuinely available, and it is
    what licenses using this repository's closed form as the reference that
    the tree converges to. Measured residual across these six comparisons: at
    most 1.1e-14. Tolerance 1e-12.
    """
    option, process, _ = _quantlib_option(kind, spot, strike, days, rate, div, sigma)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    ours = price(
        EuropeanOption(kind=kind, strike=strike, expiry=expiry),  # type: ignore[arg-type]
        BlackScholesModel(sigma=sigma),
        _qpl_market(spot, rate, div),
        method="analytic",
    ).value
    assert ours == pytest.approx(option.NPV(), abs=1e-12)


def _log_space_crr(
    kind: str,
    *,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
    n_steps: int,
) -> float:
    """QuantLib's `CoxRossRubinstein` parameterisation, written out here.

    Same lattice as `qpl.engines.tree` (`u = e^{sigma sqrt(dt)}`, `d = 1/u`),
    but with the log-space probability
    `p = 1/2 + (r - q - sigma^2/2) dt / (2 sigma sqrt(dt))`.
    """
    dt = expiry / n_steps
    dx = sigma * math.sqrt(dt)
    p_up = 0.5 + 0.5 * ((rate - div - 0.5 * sigma * sigma) * dt) / dx

    j = np.arange(n_steps + 1, dtype=float)
    terminal = spot * np.exp(dx * (2.0 * j - n_steps))
    values = (
        np.maximum(terminal - strike, 0.0)
        if kind == "call"
        else np.maximum(strike - terminal, 0.0)
    )
    discount = math.exp(-rate * dt)
    for _ in range(n_steps):
        values = discount * (p_up * values[1:] + (1.0 - p_up) * values[:-1])
    return float(values[0])


@pytest.mark.parametrize("n_steps", [50, 200])
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_quantlib_crr_is_the_log_space_parameterisation(
    kind: str,
    n_steps: int,
    spot: float,
    strike: float,
    expiry: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: INDEPENDENT_ENGINE -- the *mechanism* of the gap.

    Observing that two engines disagree is weak; identifying why is not. A
    twelve-line reimplementation of QuantLib's probability, on the same
    lattice, reproduces `BinomialVanillaEngine(process, "crr", n)` to a
    measured 8.3e-11 or better across these twelve comparisons. Whatever
    separates `qpl` from QuantLib here is therefore the probability, and only
    the probability.

    The residual is 1e-11-scale rather than 1e-15-scale because QuantLib rolls
    back through its own `TimeGrid` and lattice machinery, accumulating
    round-off in a different order; 1e-9 is the tolerance, with roughly a
    factor of twelve of headroom.
    """
    option, process, realised_t = _quantlib_option(
        kind, spot, strike, days, rate, div, sigma
    )
    option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", n_steps))

    mirrored = _log_space_crr(
        kind,
        spot=spot,
        strike=strike,
        expiry=realised_t,
        rate=rate,
        div=div,
        sigma=sigma,
        n_steps=n_steps,
    )
    assert mirrored == pytest.approx(option.NPV(), abs=1e-9)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_qpl_tree_and_quantlib_crr_differ_at_first_order_in_one_over_n(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: NEGATIVE_FINDING.

    Pinned so that the gap cannot silently change and cannot be mistaken for
    a bug. The two probabilities differ at `O(dt**1.5)` per step, so the price
    difference is `O(1/n)`: `n * (qpl - quantlib)` is constant to better than
    1% over `n` in {50, 200, 800}. Measured constants:

        atm_1y      call +0.0626   put -0.0357
        otm_2y_div  call +0.0307   put -0.0298
        itm_1y_div  call +0.1164   put -0.0244

    At `n = 200` that is a difference of 3.1e-04 (ATM call) -- six orders of
    magnitude above 1e-10. The right way to read it: both engines are order-1
    approximations of the same number and neither is "the" CRR answer.
    """
    market = _qpl_market(spot, rate, div)
    option = EuropeanOption(kind=kind, strike=strike, expiry=expiry)  # type: ignore[arg-type]
    model = BlackScholesModel(sigma=sigma)

    scaled = []
    for n_steps in (50, 200, 800):
        ql_option, process, _ = _quantlib_option(
            kind, spot, strike, days, rate, div, sigma
        )
        ql_option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", n_steps))
        ours = price(
            option, model, market, method="tree", cfg=TreeConfig(n_steps=n_steps)
        ).value
        difference = ours - ql_option.NPV()
        # Definitively not round-off: six orders of magnitude above 1e-10.
        assert abs(difference) > 1e-6
        scaled.append(n_steps * difference)

    reference = scaled[0]
    for value in scaled[1:]:
        assert value == pytest.approx(reference, rel=0.01)


_REFINEMENT_LEVELS = (50, 100, 200, 400, 800)


def _errors_against_quantlib_closed_form(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
) -> tuple[list[float], list[float]]:
    """Absolute errors of (qpl tree, QuantLib tree) against QuantLib's closed
    form, over `_REFINEMENT_LEVELS`."""
    exact_option, process, _ = _quantlib_option(kind, spot, strike, days, rate, div, sigma)
    exact_option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    exact = exact_option.NPV()

    market = _qpl_market(spot, rate, div)
    option = EuropeanOption(kind=kind, strike=strike, expiry=expiry)  # type: ignore[arg-type]
    model = BlackScholesModel(sigma=sigma)

    ours, theirs = [], []
    for n_steps in _REFINEMENT_LEVELS:
        ql_option, ql_process, _ = _quantlib_option(
            kind, spot, strike, days, rate, div, sigma
        )
        ql_option.setPricingEngine(ql.BinomialVanillaEngine(ql_process, "crr", n_steps))
        theirs.append(abs(ql_option.NPV() - exact))
        ours.append(
            abs(
                price(
                    option, model, market, method="tree", cfg=TreeConfig(n_steps=n_steps)
                ).value
                - exact
            )
        )
    return ours, theirs


@pytest.mark.parametrize("kind", ["call", "put"])
def test_both_trees_are_first_order_at_the_money(kind: str) -> None:
    """Evidence class: INDEPENDENT_ENGINE + CONVERGENCE_ORDER.

    What survives of the original 1e-10 expectation: the two engines disagree
    at `O(1/n)` but converge to the same value at the same rate, and that
    value is QuantLib's own closed form. At the money, on the even-`n`
    sequence {50, 100, 200, 400, 800}, both fit an order of 0.999 with a
    log-space RMS residual below 1e-03. Measured errors at `n = 800`:
    2.50e-03 (qpl) and 2.58e-03 / 2.45e-03 (QuantLib, call / put).
    """
    spot, strike, expiry, days, rate, div, sigma = _ARGS[0]
    ours, theirs = _errors_against_quantlib_closed_form(
        kind, spot, strike, expiry, days, rate, div, sigma
    )

    h = [1.0 / n for n in _REFINEMENT_LEVELS]
    for label, errs in (("qpl", ours), ("quantlib", theirs)):
        fit = fit_convergence_order(h, errs)
        assert 0.8 <= fit.order <= 1.2, (label, fit.order)
        assert fit.residual < 0.01, (label, fit.residual)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS[1:], ids=_IDS[1:])
def test_away_from_the_money_the_error_constant_is_erratic_in_both_engines(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: NEGATIVE_FINDING.

    The clean odd/even story of `tests/test_tree_convergence.py` is an *at the
    money* story. When `K != S` the strike's position among the terminal nodes
    does not follow the parity of `n`, and the order-1 constant jumps around:
    on the same even-`n` sequence the fitted order comes out at 1.21 to 1.48
    with log-space residuals of 0.37 to 1.52, which is the fit saying that a
    single power law does not describe the data. Measured errors for the ITM
    point run 2.8e-02, 1.3e-02, 2.8e-04, 5.2e-04, 2.0e-03 -- non-monotone.

    QuantLib's tree does the same thing at the same points (residuals 0.37 to
    1.52), which is the reason this test is here rather than a bug report:
    the erratic constant is a property of the CRR scheme off the money, not of
    either implementation. It is pinned so that it cannot be mistaken for one
    later.

    Both engines still get somewhere: every error at `n = 800` is below
    3e-03, roughly a factor of ten better than at `n = 50`.
    """
    ours, theirs = _errors_against_quantlib_closed_form(
        kind, spot, strike, expiry, days, rate, div, sigma
    )

    h = [1.0 / n for n in _REFINEMENT_LEVELS]
    for label, errs in (("qpl", ours), ("quantlib", theirs)):
        fit = fit_convergence_order(h, errs)
        assert fit.residual > 0.2, (label, fit.residual)
        assert errs[-1] < 3e-3, (label, errs)
        assert errs[-1] < errs[0] / 10.0, (label, errs)
