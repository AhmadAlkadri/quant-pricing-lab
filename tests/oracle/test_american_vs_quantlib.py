"""The American CRR tree against QuantLib: one agreement, one disagreement.

Slice item (f), in two halves.

**The agreement.** QuantLib's `FdBlackScholesVanillaEngine` with an
`AmericanExercise` solves the same free-boundary problem by a completely
different route -- an implicit finite-difference sweep on a log-spot grid with
a projection at each step, rather than backward induction on a binomial
lattice. It has no closed form to check against either, so the useful
statement is that two engines with nothing in common except the model converge
to the same number, and that the residual gap is the size their own measured
refinements predict.

Measured refinements at `S = K = 100, r = 5%, q = 0, sigma = 20%, T = 1`
(American put), against the best available estimate of the limit,
6.090376463020103 (this tree's average of `n = 64000` and `n = 64001`, which
brackets and so cancels the odd/even oscillation):

    QuantLib FD, tGrid = xGrid = g        qpl tree, n
    g      value        error             n       value        error
    200    6.08701549   -3.36e-03         2000    6.08998995   -3.87e-04
    400    6.08880946   -1.57e-03         8001    6.09055641   +1.80e-04
    800    6.08961524   -7.61e-04        16001    6.09046362   +8.72e-05
    1600   6.08999849   -3.78e-04        32001    6.09041722   +4.08e-05
    3200   6.09018558   -1.91e-04

Both are first order in their own refinement parameter (the FD errors halve
per doubling; the tree's odd-`n` errors halve per doubling), and they approach
from **opposite sides**: QuantLib's FD from below, this tree's odd-`n`
sequence from above. So the gap between them at the settings used below is the
sum of two same-order errors, not a cancellation:

    |tree(8001) - FD(3200, 3200)| = 1.80e-04 + 1.91e-04 = 3.71e-04.

The tolerance is 1e-03, a factor of 2.7. It is derived from that table, not
chosen: anything below about 4e-04 would be asserting that the two errors
cancel, and anything above about 3e-03 would pass even if one engine had
fallen to half-order accuracy.

Measured `|tree(8001) - FD(3200, 3200)|` at the three points below: 3.71e-04
(ATM put), 1.20e-04 (Longstaff-Schwartz row 1), 3.10e-04 (call with a 6%
yield). All inside 1e-03.

**The disagreement**, preserved from Slice 1. QuantLib's
`BinomialVanillaEngine(process, "crr", n)` uses the log-space probability
`1/2 + (r - q - sigma^2/2) dt / (2 sigma sqrt(dt))` where this package uses
the forward-matching `p = (e^{(r-q) dt} - d) / (u - d)`. The two differ at
`O(dt**1.5)` per step and therefore `O(1/n)` in price, which is the same order
as each engine's own discretisation error. Early exercise does not change
that: the probability enters the continuation value, and the Bellman maximum
is applied to it afterwards. Measured `n * (qpl - QuantLib)` at
`n` in {200, 800, 3200}:

    ATM put          -0.026367  -0.026404  -0.026413
    L&S row 1        -0.013281  -0.013237  -0.013260
    call, q = 6%     +0.007084  +0.007099  +0.007104

constant to better than 0.5% over a sixteen-fold refinement. Pinned as a
NEGATIVE_FINDING so it cannot later be mistaken for a bug in either engine.

Day count: `Actual365Fixed` makes `T = days / 365`, so 365 days is exactly
1.0 in double precision and there is no day-count residual to account for --
the same convention as `tests/oracle/test_tree_vs_quantlib.py`.

Evidence classes: INDEPENDENT_ENGINE for the finite-difference agreement and
for the shared first-order convergence; NEGATIVE_FINDING for the pinned
`O(1/n)` binomial discrepancy.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

ql = pytest.importorskip("QuantLib")

from qpl.cases import (  # noqa: E402
    AMERICAN_REFERENCE_CASES,
    AMERICAN_REFERENCE_N_STEPS,
    AMERICAN_REFERENCE_VALUE,
)
from qpl.engines.tree import TreeConfig  # noqa: E402
from qpl.instruments.options import AmericanOption  # noqa: E402
from qpl.market.curves import FlatDividendCurve, FlatRateCurve  # noqa: E402
from qpl.market.market import Market  # noqa: E402
from qpl.models.black_scholes import BlackScholesModel  # noqa: E402
from qpl.pricing import price  # noqa: E402
from qpl.validation import EvidenceClass, fit_convergence_order  # noqa: E402

_EVALUATION_DATE = ql.Date(15, 1, 2024)
_DAY_COUNT = ql.Actual365Fixed()
_CALENDAR = ql.NullCalendar()

FD_GRID = 3200
"""`tGrid` and `xGrid` for the fine finite-difference comparison (0.38 s)."""

TREE_N_STEPS = AMERICAN_REFERENCE_N_STEPS
"""Tree refinement for the comparison, the same 8001 the cases layer pins."""

FD_TOLERANCE = 1e-3
"""Agreement budget, derived in this module's docstring from both refinements."""

# (id, kind, spot, strike, expiry_years, expiry_days, rate, dividend, sigma)
_POINTS = (
    ("atm_put", "put", 100.0, 100.0, 1.0, 365, 0.05, 0.00, 0.20),
    ("ls2001_row1_put", "put", 36.0, 40.0, 1.0, 365, 0.06, 0.00, 0.20),
    ("call_with_yield", "call", 100.0, 100.0, 1.0, 365, 0.05, 0.06, 0.20),
)
_IDS = [row[0] for row in _POINTS]
_ARGS = [row[1:] for row in _POINTS]
_ARGNAMES = ("kind", "spot", "strike", "expiry", "days", "rate", "div", "sigma")


def _quantlib_american(
    kind: str,
    spot: float,
    strike: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
) -> tuple[object, object, float]:
    """Build the QuantLib American option / process pair; return the realised `T`."""
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
    option = ql.VanillaOption(
        payoff, ql.AmericanExercise(_EVALUATION_DATE, maturity)
    )
    return option, process, _DAY_COUNT.yearFraction(_EVALUATION_DATE, maturity)


def _quantlib_fd(
    kind: str,
    spot: float,
    strike: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
    *,
    grid: int,
) -> float:
    option, process, _ = _quantlib_american(kind, spot, strike, days, rate, div, sigma)
    option.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, grid, grid))
    return float(option.NPV())


def _quantlib_crr(
    kind: str,
    spot: float,
    strike: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
    *,
    n_steps: int,
) -> float:
    option, process, _ = _quantlib_american(kind, spot, strike, days, rate, div, sigma)
    option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", n_steps))
    return float(option.NPV())


def _qpl_tree(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
    *,
    n_steps: int,
) -> float:
    return price(
        AmericanOption(kind=kind, strike=strike, expiry=expiry),  # type: ignore[arg-type]
        BlackScholesModel(sigma=sigma),
        Market(
            spot=spot,
            rate_curve=FlatRateCurve(rate),
            dividend_curve=FlatDividendCurve(div),
        ),
        method="tree",
        cfg=TreeConfig(n_steps=n_steps),
    ).value


@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_day_count_makes_the_expiry_exact(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """No day-count residual: `T = 365 / 365` is exactly 1.0."""
    _, _, realised = _quantlib_american(kind, spot, strike, days, rate, div, sigma)
    assert realised == expiry


@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_tree_agrees_with_quantlib_finite_differences(
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

    Tolerance derived in the module docstring from both engines' measured
    refinements; the two approach the limit from opposite sides, so the gap is
    the *sum* of their errors. Measured gaps: 3.71e-04, 1.20e-04, 3.10e-04
    against a 1e-03 budget.
    """
    fd = _quantlib_fd(kind, spot, strike, days, rate, div, sigma, grid=FD_GRID)
    tree = _qpl_tree(kind, spot, strike, expiry, rate, div, sigma, n_steps=TREE_N_STEPS)
    assert tree == pytest.approx(fd, abs=FD_TOLERANCE)


def test_quantlib_finite_differences_confirm_the_pinned_reference_row() -> None:
    """Evaluate `atm_1y_american_put_reference_vs_quantlib_fd`.

    This is the INDEPENDENT_ENGINE leg of the in-repo reference value that
    `qpl.cases.american_black_scholes` pins. The row carries its own tolerance
    and the justification for it; nothing numeric is chosen here.
    """
    case = next(
        c
        for c in AMERICAN_REFERENCE_CASES
        if c.row.evidence is EvidenceClass.INDEPENDENT_ENGINE
    )
    spec = case.spec
    fd = _quantlib_fd(
        spec.kind, spec.spot, spec.strike, 365, spec.rate, spec.dividend, spec.sigma,
        grid=FD_GRID,
    )
    assert fd == pytest.approx(case.row.expected, abs=case.row.tolerance), case.row.notes
    # And the row's expected value really is this engine's pinned output.
    assert case.row.expected == AMERICAN_REFERENCE_VALUE


_FD_GRIDS = (200, 400, 800, 1600, 3200)


def test_quantlib_finite_differences_are_first_order_in_the_grid() -> None:
    """Evidence class: CONVERGENCE_ORDER, measured without a reference value.

    The order is fitted on *successive differences*, `|v(2g) - v(g)|`, which
    behave like `C g^{-p} (1 - 2^{-p})` and so carry the same slope as the
    error itself -- no limit value is needed, which matters because there is
    no closed form to supply one and using this tree's own answer would make
    the comparison circular.

    Measured on the ATM American put: successive differences 1.794e-03,
    8.058e-04, 3.833e-04, 1.871e-04 over `g` in {200, 400, 800, 1600}, fitted
    order 1.0856 with a log-space RMS residual of 0.0211. That the joint
    refinement of `tGrid` and `xGrid` comes out first order rather than second
    is expected for an American payoff: the free boundary is resolved only to
    the grid, and that error is first order however good the time stepping is.
    """
    kind, spot, strike, expiry, days, rate, div, sigma = _ARGS[0]
    values = [
        _quantlib_fd(kind, spot, strike, days, rate, div, sigma, grid=g) for g in _FD_GRIDS
    ]
    diffs = [abs(b - a) for a, b in pairwise(values)]
    fit = fit_convergence_order([1.0 / g for g in _FD_GRIDS[:-1]], diffs)

    assert 0.8 <= fit.order <= 1.3, fit.order
    assert fit.residual < 0.1, fit.residual
    # Monotone approach from below, which is what makes the sum-of-errors
    # argument in the module docstring valid.
    assert all(b > a for a, b in pairwise(values)), values


_CRR_LEVELS = (200, 800, 3200)


@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_quantlib_crr_american_differs_from_ours_at_first_order_in_one_over_n(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: NEGATIVE_FINDING -- the Slice 1 result, preserved.

    QuantLib's `CoxRossRubinstein` is the log-space parameterisation, not the
    forward-matching one, so `qpl` and QuantLib disagree at `O(1/n)` on the
    same lattice. Slice 1 established this for European payoffs and identified
    the mechanism by reimplementing QuantLib's probability. Early exercise
    leaves it untouched, because the probability enters the continuation value
    and the Bellman maximum is applied afterwards.

    `n * (qpl - QuantLib)` measured at `n` in {200, 800, 3200}:

        ATM put          -0.026367  -0.026404  -0.026413
        L&S row 1        -0.013281  -0.013237  -0.013260
        call, q = 6%     +0.007084  +0.007099  +0.007104

    constant to better than 0.5% over a sixteen-fold refinement, and at
    `n = 200` six orders of magnitude above the 1e-10 that the Slice 1 brief
    originally expected. Neither engine is "the" CRR American answer; both are
    order-1 approximations of the same number.
    """
    scaled = []
    for n_steps in _CRR_LEVELS:
        theirs = _quantlib_crr(kind, spot, strike, days, rate, div, sigma, n_steps=n_steps)
        ours = _qpl_tree(kind, spot, strike, expiry, rate, div, sigma, n_steps=n_steps)
        difference = ours - theirs
        # Definitively not round-off.
        assert abs(difference) > 1e-6
        scaled.append(n_steps * difference)

    reference = scaled[0]
    for value in scaled[1:]:
        assert value == pytest.approx(reference, rel=0.005), scaled
