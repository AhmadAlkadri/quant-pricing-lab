"""Heston calibration against QuantLib, and what the two solvers agree about.

QuantLib's `HestonModel` + `HestonModelHelper` + `LevenbergMarquardt` is a
genuinely independent route to the same inverse problem: a different
characteristic function (Gatheral's arrangement), a different pricer
(`AnalyticHestonEngine`, adaptive Gauss-Lobatto or Gauss-Laguerre against this
package's cosine expansion), a different Levenberg-Marquardt (MINPACK's
`lmdif`, finite-difference Jacobian, against `scipy.optimize.least_squares`
driven by an analytic one), and a different codebase.

Five results.

1. **Both recover the reference set.** From the same perturbed start on a
   5-strike by 6-maturity grid, QuantLib's worst parameter error is
   **3.66e-07** (`ImpliedVolError`), **5.41e-07** (`PriceError`) and
   **3.24e-06** (`RelativePriceError`), all on `kappa` or `xi` -- the two the
   Jacobian says are the flat ones.

2. **They stop at the same point, not merely near it.** On the price
   objective the two fitted vectors differ by **1.30e-10** in `kappa` and less
   in everything else, while both sit **-5.41e-07** from the truth in `kappa`
   -- the same offset, same sign. Two solvers that share no code do not land on
   the same wrong answer by accident, so the offset is a property of the data
   and not of either of them. It is: the quotes are implied volatilities
   produced by a Brent inversion at `xtol = 1e-07`, and converting them back to
   prices moves the price objective's minimum by that tolerance times vega.
   Fed prices directly, with no round trip, this package's price objective
   recovers `kappa` to **4.45e-12**. The finding is about how a synthetic
   experiment is built, and it is why the implied-volatility objective is the
   one whose clean recovery reads 1e-12 while the price objective's reads
   1e-07.

3. **Both recover the Feller-violating set.** Worst 5.93e-07 on `kappa`,
   agreeing with this package to 5.93e-07 as well (this package is at 1e-12
   there, so the difference *is* QuantLib's error). Neither library refuses the
   parameters or regularises toward the Feller region.

4. **Both show the single-maturity degeneracy, and they disagree about where
   in the valley to stop.** From four starts on one smile, this package lands
   on `kappa` in [2.918, 7.288] and QuantLib on `kappa` in [3.930, 10.956].
   Both fit the smile; neither recovers `kappa = 4`; the two libraries do not
   even agree on the wrong answer, which is the strongest available statement
   that the flat direction is in the problem and not in a solver.

5. **The start-grid success rate is a property of the solver, not of the
   problem.** On the same twenty starts and the same exact quotes, this
   package reaches `kappa = 4` from all twenty and QuantLib from **eleven**,
   stopping elsewhere at `kappa` of 0.00000, 0.59951, 1.00750, 10.42008,
   22.11941 and so on, several of them pinned against a parameter constraint.
   This is not a ranking -- three things differ between the two solvers at once
   and this slice separates none of them -- it is the generalisation of what
   this slice already learned the hard way about its own 11/14.

QuantLib's helper conventions, recorded because they are load-bearing here.

- `HestonModelHelper(maturity, calendar, s0, strike, volatility, rTS, qTS,
  errorType)` takes an **implied volatility quote**, never a price: it builds
  its own market value from Black at that volatility. Feeding it a premium
  silently calibrates to the wrong surface.
- The maturity is a `Period` advanced from the evaluation date by the
  calendar, and the year fraction comes from the **term structure's** day
  counter. `Actual365Fixed` cannot express `T = 0.25` (91/365 = 0.24932), so
  this file uses `Actual360` and day counts that are multiples of 90, which
  makes every `T` in `(0.25, 0.5, 1, 1.5, 2, 3)` exact and removes the
  day-count residual from the comparison entirely.
- `AnalyticHestonEngine(model, relTolerance, maxEvaluations)` with
  `maxEvaluations = 1000` **raises** `RuntimeError("max number of iterations
  reached")` from `helper.modelValue()` itself, before any calibration starts,
  at the starting parameters used here. `10**6` is what works, matching
  `tests/oracle/test_heston_vs_quantlib.py`; the fixed 192-point Gauss-Laguerre
  variant reaches the same optimum to the digits reported above and is 7x
  faster.
- `CalibratedModel.calibrate` signals failure by raising, not by a return
  value, and leaves the model at its starting parameters when it does.

Evidence classes: INDEPENDENT_ENGINE for every agreement, NEGATIVE_FINDING for
the implied-volatility round trip, for the single-maturity disagreement and for
the start-grid rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

ql = pytest.importorskip("QuantLib")

from qpl.calibration import (  # noqa: E402
    HESTON_PARAMETERS,
    OptionQuote,
    calibrate_heston,
    heston_quote_values,
)
from qpl.cases import (  # noqa: E402
    HESTON_CALIBRATION_FELLER_VIOLATED,
    HESTON_CALIBRATION_MARKET,
    HESTON_CALIBRATION_REFERENCE,
    HESTON_CALIBRATION_SINGLE_MATURITY_STARTS,
    HESTON_CALIBRATION_START_GRID,
    HESTON_CALIBRATION_STRIKES,
    synthetic_quotes,
)
from qpl.models.heston import HestonModel  # noqa: E402
from qpl.validation import EvidenceClass  # noqa: E402

_EVALUATION_DATE = ql.Date(15, 1, 2024)
_DAY_COUNT = ql.Actual360()
"""`Actual360` and day counts that are multiples of 90 make every maturity in
`MATURITIES` exact. `Actual365Fixed` cannot: 91/365 = 0.2493."""

_CALENDAR = ql.NullCalendar()
_SPOT = 100.0
_RATE = 0.01
_DIVIDEND = 0.02

MATURITIES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
ONE_MATURITY = (1.0,)
REFERENCE_START = (0.08, 2.0, 0.15, 0.6, -0.2)
FELLER_VIOLATED_START = (0.08, 0.9, 0.09, 0.6, -0.6)

ENGINE_EVALUATIONS = 10**6
"""`AnalyticHestonEngine`'s `maxEvaluations`. 1000 is not enough: at the
starting parameters below `helper.modelValue()` raises before the calibration
runs at all (module docstring)."""

ENGINE_TOLERANCE = 1e-12
LM_EPSILON = 1e-08
END_CRITERIA = (1000, 100, 1e-08, 1e-08, 1e-08)


@dataclass(frozen=True)
class QuantLibFit:
    parameters: tuple[float, ...]
    worst_helper_error: float
    message: str


def _term_structures():
    ql.Settings.instance().evaluationDate = _EVALUATION_DATE
    spot = ql.QuoteHandle(ql.SimpleQuote(_SPOT))
    rates = ql.YieldTermStructureHandle(
        ql.FlatForward(_EVALUATION_DATE, _RATE, _DAY_COUNT)
    )
    dividends = ql.YieldTermStructureHandle(
        ql.FlatForward(_EVALUATION_DATE, _DIVIDEND, _DAY_COUNT)
    )
    return spot, rates, dividends


def _quantlib_calibrate(quotes, start, *, error_type, laguerre: bool = False):
    """QuantLib's own Heston calibration on the same implied-volatility quotes."""
    spot, rates, dividends = _term_structures()
    process = ql.HestonProcess(rates, dividends, spot, *start)
    model = ql.HestonModel(process)
    engine = (
        ql.AnalyticHestonEngine(model, 192)
        if laguerre
        else ql.AnalyticHestonEngine(model, ENGINE_TOLERANCE, ENGINE_EVALUATIONS)
    )
    helpers = []
    for quote in quotes:
        days = round(quote.expiry * 360)
        assert _DAY_COUNT.yearFraction(
            _EVALUATION_DATE, _EVALUATION_DATE + ql.Period(days, ql.Days)
        ) == quote.expiry
        helper = ql.HestonModelHelper(
            ql.Period(days, ql.Days),
            _CALENDAR,
            _SPOT,
            quote.strike,
            ql.QuoteHandle(ql.SimpleQuote(quote.value)),
            rates,
            dividends,
            error_type,
        )
        helper.setPricingEngine(engine)
        helpers.append(helper)

    message = ""
    try:
        model.calibrate(
            helpers,
            ql.LevenbergMarquardt(LM_EPSILON, LM_EPSILON, LM_EPSILON),
            ql.EndCriteria(*END_CRITERIA),
        )
    except RuntimeError as error:  # pragma: no cover - only on a failed solve
        message = str(error)
    try:
        worst = max(abs(h.calibrationError()) for h in helpers)
    except RuntimeError:  # the engine can fail at the parameters it stopped on
        worst = float("nan")
    return QuantLibFit(
        parameters=(
            model.v0(),
            model.kappa(),
            model.theta(),
            model.sigma(),
            model.rho(),
        ),
        worst_helper_error=worst,
        message=message,
    )


# --------------------------------------------------------------------------
# (1) Both libraries recover the reference set.
# --------------------------------------------------------------------------

RECOVERY_BUDGET = {
    "implied_vol": 4e-06,
    "price": 5e-06,
    "relative_price": 2e-05,
}
"""Worst measured QuantLib parameter error per `CalibrationErrorType`:
3.66e-07 (`ImpliedVolError`), 5.41e-07 (`PriceError`), 3.24e-06
(`RelativePriceError`), all on `kappa` or `xi`. The budgets are an order of
magnitude above each, which is the identifiability-limited tolerance this
comparison can honestly carry -- the two flat directions are exactly where the
two libraries differ, and a tighter budget would be pinning MINPACK's stopping
rule."""

_ERROR_TYPES = {
    "implied_vol": ql.BlackCalibrationHelper.ImpliedVolError,
    "price": ql.BlackCalibrationHelper.PriceError,
    "relative_price": ql.BlackCalibrationHelper.RelativePriceError,
}


@pytest.mark.parametrize("error_type", sorted(RECOVERY_BUDGET))
def test_quantlib_recovers_the_reference_set(error_type: str) -> None:
    """Evidence class: INDEPENDENT_ENGINE."""
    assert EvidenceClass.INDEPENDENT_ENGINE is EvidenceClass.INDEPENDENT_ENGINE
    quotes = synthetic_quotes(
        HestonModel(*HESTON_CALIBRATION_REFERENCE), MATURITIES, HESTON_CALIBRATION_STRIKES
    )
    fit = _quantlib_calibrate(
        quotes, REFERENCE_START, error_type=_ERROR_TYPES[error_type]
    )
    assert fit.message == ""
    errors = np.abs(
        np.array(fit.parameters) - np.array(HESTON_CALIBRATION_REFERENCE)
    )
    assert float(np.max(errors)) < RECOVERY_BUDGET[error_type], dict(
        zip(HESTON_PARAMETERS, errors, strict=True)
    )


def test_the_gauss_laguerre_engine_reaches_the_same_optimum() -> None:
    """The reference is converged: the quadrature choice does not move it.

    The fixed 192-point Gauss-Laguerre engine and the adaptive Gauss-Lobatto
    one at `relTolerance = 1e-12` land on the same parameters, so the residuals
    above are the calibration's and not the pricer's.
    """
    quotes = synthetic_quotes(
        HestonModel(*HESTON_CALIBRATION_REFERENCE), MATURITIES, HESTON_CALIBRATION_STRIKES
    )
    adaptive = _quantlib_calibrate(
        quotes, REFERENCE_START, error_type=ql.BlackCalibrationHelper.ImpliedVolError
    )
    laguerre = _quantlib_calibrate(
        quotes,
        REFERENCE_START,
        error_type=ql.BlackCalibrationHelper.ImpliedVolError,
        laguerre=True,
    )
    assert np.max(
        np.abs(np.array(adaptive.parameters) - np.array(laguerre.parameters))
    ) < 1e-09


# --------------------------------------------------------------------------
# (2) The two solvers stop at the same point, and it is not the truth.
# --------------------------------------------------------------------------

SOLVER_AGREEMENT = 1e-08
"""Worst measured difference between the two fitted parameter vectors on the
price objective: **1.30e-10**, on `kappa`. Both are 5.41e-07 from the truth,
with the same sign."""


def test_the_two_price_objective_fits_agree_far_closer_than_either_is_to_truth() -> None:
    """NEGATIVE_FINDING: the offset is in the data, not in either solver.

    This is the sharpest thing the oracle says here. Two Levenberg-Marquardt
    implementations sharing no code -- MINPACK with a finite-difference
    Jacobian against SciPy's trust-region with an analytic one, on two
    different characteristic-function arrangements and two different pricers --
    land on the same parameter vector to 1.30e-10 and are both 5.41e-07 from
    the truth in `kappa`, with the same sign. The only thing they share is the
    data, so that is where the offset must be.

    And it is: the quotes are implied volatilities produced by a Brent
    inversion at `xtol = 1e-07`, so converting them back to prices for a price
    objective displaces its minimum by that tolerance times vega. The next test
    removes the round trip and the offset with it.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    quotes = synthetic_quotes(
        HestonModel(*HESTON_CALIBRATION_REFERENCE), MATURITIES, HESTON_CALIBRATION_STRIKES
    )
    quantlib = _quantlib_calibrate(
        quotes, REFERENCE_START, error_type=ql.BlackCalibrationHelper.PriceError
    )
    mine = calibrate_heston(
        quotes, HESTON_CALIBRATION_MARKET, initial=REFERENCE_START, objective="price"
    )
    difference = np.abs(np.array(quantlib.parameters) - np.array(mine.parameters))
    to_truth = np.abs(
        np.array(mine.parameters) - np.array(HESTON_CALIBRATION_REFERENCE)
    )
    assert float(np.max(difference)) < SOLVER_AGREEMENT
    # The two agree with each other 1000x more closely than either agrees with
    # the parameters the quotes were generated from.
    assert float(np.max(difference)) < float(np.max(to_truth)) / 1e03
    # ... and they are displaced in the same direction.
    quantlib_signed = np.array(quantlib.parameters) - np.array(
        HESTON_CALIBRATION_REFERENCE
    )
    mine_signed = np.array(mine.parameters) - np.array(HESTON_CALIBRATION_REFERENCE)
    assert np.all(np.sign(quantlib_signed) == np.sign(mine_signed))


def test_price_quotes_without_the_inversion_round_trip_recover_exactly() -> None:
    """The other half of the finding, and it is this package's own quotes.

    Fed prices directly the price objective recovers `kappa` to 4.45e-12 --
    five decimal orders better than the 5.41e-07 above, on the same grid, the
    same start and the same solver. Nothing changed but how the synthetic
    quotes were written down.
    """
    skeleton = [
        OptionQuote(strike=k, expiry=t, kind="call", value=1.0, value_type="price")
        for t in MATURITIES
        for k in HESTON_CALIBRATION_STRIKES
    ]
    prices = heston_quote_values(
        HestonModel(*HESTON_CALIBRATION_REFERENCE),
        HESTON_CALIBRATION_MARKET,
        skeleton,
    )
    quotes = [
        OptionQuote(
            strike=q.strike,
            expiry=q.expiry,
            kind="call",
            value=float(p),
            value_type="price",
        )
        for q, p in zip(skeleton, prices, strict=True)
    ]
    fit = calibrate_heston(
        quotes, HESTON_CALIBRATION_MARKET, initial=REFERENCE_START, objective="price"
    )
    errors = np.abs(np.array(fit.parameters) - np.array(HESTON_CALIBRATION_REFERENCE))
    assert float(np.max(errors)) < 1e-10


# --------------------------------------------------------------------------
# (3) The Feller-violating set.
# --------------------------------------------------------------------------

FELLER_VIOLATED_BUDGET = 5e-06
"""Worst measured QuantLib error on the Feller-violating set: 5.93e-07 on
`kappa`. This package is at 1e-12 there, so the agreement between the two
*is* QuantLib's error and the budget is set from it."""


def test_quantlib_recovers_the_feller_violating_set_too() -> None:
    """INDEPENDENT_ENGINE, and neither library regularises toward Feller.

    `2 kappa theta = 0.04` against `xi^2 = 1.0` -- the condition is violated by
    a factor of 25 -- and both calibrations return the violating parameters
    rather than being pulled toward the admissible region. That matters because
    forbidding the violating side is a common and unstated prior, and this
    checks that neither side of the comparison carries it.
    """
    assert EvidenceClass.INDEPENDENT_ENGINE is EvidenceClass.INDEPENDENT_ENGINE
    quotes = synthetic_quotes(
        HestonModel(*HESTON_CALIBRATION_FELLER_VIOLATED),
        MATURITIES,
        HESTON_CALIBRATION_STRIKES,
    )
    quantlib = _quantlib_calibrate(
        quotes,
        FELLER_VIOLATED_START,
        error_type=ql.BlackCalibrationHelper.ImpliedVolError,
    )
    mine = calibrate_heston(
        quotes,
        HESTON_CALIBRATION_MARKET,
        initial=FELLER_VIOLATED_START,
        objective="implied_vol",
    )
    truth = np.array(HESTON_CALIBRATION_FELLER_VIOLATED)
    assert float(np.max(np.abs(np.array(quantlib.parameters) - truth))) < (
        FELLER_VIOLATED_BUDGET
    )
    assert float(np.max(np.abs(np.array(mine.parameters) - truth))) < 1e-06
    assert not mine.feller_satisfied
    quantlib_feller = (
        4.0
        * quantlib.parameters[1]
        * quantlib.parameters[2]
        / (quantlib.parameters[3] ** 2)
    )
    assert quantlib_feller == pytest.approx(0.08, abs=1e-04)


# --------------------------------------------------------------------------
# (4) The single-maturity degeneracy, seen from both sides.
# --------------------------------------------------------------------------

SINGLE_MATURITY_STARTS = HESTON_CALIBRATION_SINGLE_MATURITY_STARTS[::2]
"""Every other start from the published six, to keep the oracle cheap; the
spread is the same claim at half the cost."""


@pytest.mark.slow
def test_both_libraries_fail_to_identify_kappa_at_one_maturity() -> None:
    """NEGATIVE_FINDING, confirmed independently.

    This package lands on `kappa` in [2.918, 7.288] across the published
    starts; QuantLib lands in [3.930, 10.956] across the same starts. Both fit
    the smile. Neither recovers `kappa = 4`. And the two do not agree with each
    other on where in the valley to stop, which is the strongest statement
    available that the flat direction belongs to the problem -- an identical
    failure in two independent solvers is at least conceivably a shared bug; a
    *different* wrong answer from each is not.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    quotes = synthetic_quotes(
        HestonModel(*HESTON_CALIBRATION_REFERENCE),
        ONE_MATURITY,
        HESTON_CALIBRATION_STRIKES,
    )
    mine = [
        calibrate_heston(
            quotes,
            HESTON_CALIBRATION_MARKET,
            initial=start,
            objective="implied_vol",
        ).parameter("kappa")
        for start in SINGLE_MATURITY_STARTS
    ]
    theirs = [
        _quantlib_calibrate(
            quotes, start, error_type=ql.BlackCalibrationHelper.ImpliedVolError
        ).parameters[1]
        for start in SINGLE_MATURITY_STARTS
    ]
    assert max(mine) / min(mine) > 1.5
    assert max(theirs) / min(theirs) > 1.5
    # Neither brackets the truth tightly, and they do not agree with each other.
    assert max(abs(k - 4.0) for k in mine) > 1.0
    assert max(abs(k - 4.0) for k in theirs) > 1.0
    assert max(abs(a - b) for a, b in zip(mine, theirs, strict=True)) > 0.5


QUANTLIB_GLOBAL_COUNT = 11
"""Out of the same 20 starts on which this package scores 20/20.

Measured on the 5-strike by 6-maturity reference grid with exact quotes, with
"reached" meaning `|kappa - 4| < 1e-03`: QuantLib's Levenberg-Marquardt gets
there from 11, and stops at `kappa` of 0.00000, 0.00000, 0.59951, 0.69477,
1.00750, 3.71951, 10.42008, 11.38892 and 22.11941 from the other nine, several
of them pinned against a parameter constraint (`xi = 0.00000`, `xi = 0.00864`).
One start (`(0.04, 1.0, 0.04, 1.5, +0.5)`) additionally leaves the engine in a
state where `helper.calibrationError()` itself raises.

This is **not** a claim that one library's optimiser is better than the
other's. The two differ in at least three ways at once -- a bounded
trust-region region against unconstrained MINPACK on QuantLib's internal
parameter transformations, an analytic Jacobian against a finite-difference
one, and QuantLib's `ImpliedVolError` being a vega-divided price residual
rather than a real inversion -- and this slice does not separate them. What it
is is the (d) lesson generalised: a start-grid success rate measures a
*solver-and-pricer pair on a data set*, never the problem. This package's own
rate on this grid was 11/14 until a payoff-leg bug was fixed, and it is 20/20
now; quoting either number as "the Heston calibration success rate" would have
been wrong both times."""


@pytest.mark.slow
def test_the_start_grid_success_rate_is_a_property_of_the_solver() -> None:
    """NEGATIVE_FINDING: 11/20 against 20/20 on identical data.

    The same twenty starts, the same fifteen-to-thirty exact quotes, the same
    parameters to recover. This package reaches `kappa = 4` from every one;
    QuantLib from eleven. The point is not the ranking -- too many things
    differ between the two to attribute it -- but that a number quoted as "how
    often a Heston calibration converges" is a property of the pair that
    produced it.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    quotes = synthetic_quotes(
        HestonModel(*HESTON_CALIBRATION_REFERENCE),
        MATURITIES,
        HESTON_CALIBRATION_STRIKES,
    )
    starts = tuple(HESTON_CALIBRATION_START_GRID) + tuple(
        HESTON_CALIBRATION_SINGLE_MATURITY_STARTS
    )
    mine = 0
    theirs = 0
    for start in starts:
        mine += (
            abs(
                calibrate_heston(
                    quotes,
                    HESTON_CALIBRATION_MARKET,
                    initial=start,
                    objective="implied_vol",
                ).parameter("kappa")
                - 4.0
            )
            < 1e-03
        )
        theirs += (
            abs(
                _quantlib_calibrate(
                    quotes, start, error_type=ql.BlackCalibrationHelper.ImpliedVolError
                ).parameters[1]
                - 4.0
            )
            < 1e-03
        )
    assert mine == len(starts)
    assert theirs == QUANTLIB_GLOBAL_COUNT
    assert theirs < mine


def test_three_maturities_make_both_libraries_agree_on_kappa() -> None:
    """The control for the degeneracy test: it is the data, not either solver.

    Restricted to the starts QuantLib reaches the optimum from (see
    `QUANTLIB_GLOBAL_COUNT` for the ones it does not), the six-maturity grid
    makes both libraries return `kappa = 4` where the single-maturity grid made
    neither.
    """
    quotes = synthetic_quotes(
        HestonModel(*HESTON_CALIBRATION_REFERENCE),
        MATURITIES,
        HESTON_CALIBRATION_STRIKES,
    )
    for start in ((0.04, 2.0, 0.25, 0.70, -0.5), (0.04, 8.0, 0.25, 1.40, -0.5)):
        mine = calibrate_heston(
            quotes,
            HESTON_CALIBRATION_MARKET,
            initial=start,
            objective="implied_vol",
        ).parameter("kappa")
        theirs = _quantlib_calibrate(
            quotes, start, error_type=ql.BlackCalibrationHelper.ImpliedVolError
        ).parameters[1]
        assert mine == pytest.approx(4.0, abs=1e-05)
        assert theirs == pytest.approx(4.0, abs=1e-04)
