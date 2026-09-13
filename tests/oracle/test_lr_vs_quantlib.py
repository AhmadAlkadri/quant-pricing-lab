"""The Leisen-Reimer tree against QuantLib: one agreement and two quirks.

Slice item (e). Unlike the CRR comparison in
`tests/oracle/test_tree_vs_quantlib.py`, where the expected 1e-10 agreement
turned out to be impossible because the two engines use different
probabilities, this one *does* agree to round-off -- both engines implement
the same construction from the same paper, including the same Peizer-Pratt
method-2 inversion.

**The agreement.** European call and put, `n` in {51, 201, 801}, at the money
and at one off-money strike: the worst measured residual against
`BinomialVanillaEngine(process, "leisenreimer", n)` is **2.71e-11**, on prices
of 4 to 22. The tolerance is 1e-9, a factor of 37. The residual is
1e-11-scale rather than 1e-15-scale for the same reason it was in the CRR
mirror test: QuantLib rolls back through its own `TimeGrid` and lattice
machinery and accumulates round-off in a different order.

For the American put the two agree just as closely -- 7.3e-12 or better at
`n` in {201, 801, 2001} -- and this package's value sits 1.52e-04 from
QuantLib's `FdBlackScholesVanillaEngine` on a 3200x3200 grid, which is what
the two engines' own measured refinements predict.

**Quirk 1: QuantLib's even `n` is not a Leisen-Reimer price.** This package
rejects even `n`; QuantLib's `LeisenReimer` tree rounds the step count up to
the next odd number. That is not the harmless convenience it looks like,
because the *engine's* time grid keeps the even count the caller asked for.
The measured European ATM call error against Black-Scholes:

    n      QuantLib error      n      QuantLib error
    50     -2.072e-01          51     -1.323e-04
    100    -1.049e-01          101    -3.424e-05
    200    -5.286e-02          201    -8.712e-06
    400    -2.654e-02          401    -2.198e-06
    800    -1.330e-02          801    -5.518e-07

The odd column halves twice per doubling (order 2, and it is this package's
own sequence to 1e-11). The even column halves once per doubling: order 1,
and 24_000 times less accurate at `n = 800` than at `n = 801`. Rolling a
20-line mirror that builds the parameters for `n + 1` steps and rolls back
over `n` reproduces about 95% of that error (10.4379 against QuantLib's
10.4373 at `n = 800`, where Black-Scholes is 10.4506), which identifies the
mechanism: the terminal distribution becomes an `n`-fold convolution of an
`(n+1)`-step parameterisation, so the strike is no longer where the
construction put it.

**Quirk 2: QuantLib's American Leisen-Reimer jumps off its own curve at
isolated `n`.** At `n` in {501, 1601, 8001} its value leaves the smooth
convergence sequence it follows everywhere else, by 9.5e-03, 2.8e-03 and
5.6e-04 respectively, always *away* from the limit; this package's value stays
on the curve at those same `n` and its neighbours. Measured signed errors
against the bracketed limit 6.090376463020103:

    n       qpl          QuantLib
    451     -6.606e-04   -6.606e-04
    501     -5.918e-04   -9.524e-03     <-- QuantLib jumps
    551     -5.308e-04   -5.308e-04
    1501    -1.916e-04   -1.916e-04
    1601    -1.795e-04   -2.975e-03     <-- QuantLib jumps
    1701    -1.694e-04   -1.694e-04

Both quirks are pinned as NEGATIVE_FINDINGs so that they cannot be mistaken
for a bug here, and so that they fail loudly if a future QuantLib fixes them
(at which point the pins should be deleted, not loosened). The QuantLib pinned
here is the one in the `[oracle]` extra; these are observations about that
version, not claims about the library in general.

Day count: `Actual365Fixed` makes `T = days / 365`, so 365 days is exactly
1.0 in double precision and there is no day-count residual -- the same
convention the other two oracle files use.

Evidence classes: INDEPENDENT_ENGINE for the European and American agreement
and for the finite-difference cross-check; NEGATIVE_FINDING for the two
quirks.
"""

from __future__ import annotations

import pytest

ql = pytest.importorskip("QuantLib")

from qpl.cases import AMERICAN_BRACKETED_LIMIT  # noqa: E402
from qpl.engines.tree import TreeConfig  # noqa: E402
from qpl.instruments.options import AmericanOption, EuropeanOption  # noqa: E402
from qpl.market.curves import FlatDividendCurve, FlatRateCurve  # noqa: E402
from qpl.market.market import Market  # noqa: E402
from qpl.models.black_scholes import BlackScholesModel  # noqa: E402
from qpl.pricing import price  # noqa: E402

_EVALUATION_DATE = ql.Date(15, 1, 2024)
_DAY_COUNT = ql.Actual365Fixed()
_CALENDAR = ql.NullCalendar()

LR = "leisen-reimer"

ODD_STEPS = (51, 201, 801)
"""European refinement levels. Odd, which this package requires and which is
also the only place QuantLib's Leisen-Reimer means what it says."""

EUROPEAN_TOLERANCE = 1e-9
"""Agreement budget for the European comparison.

Derived from the measurement: the worst residual over the twelve comparisons
below is 2.71e-11, so this keeps a factor of 37. It is the same budget the
CRR mirror test uses, and for the same reason -- QuantLib's `TimeGrid`
rollback accumulates round-off in a different order, which puts the floor at
1e-11 rather than at 1e-15.
"""

AMERICAN_STEPS = (201, 801, 2001)
"""American refinement levels. Chosen to avoid the isolated `n` at which
QuantLib's American Leisen-Reimer leaves its own convergence curve; those are
pinned separately in `test_quantlib_american_lr_jumps_off_its_curve`."""

AMERICAN_TOLERANCE = 1e-9
"""Same budget as the European leg; worst measured residual 7.3e-12."""

FD_GRID = 3200
"""`tGrid` and `xGrid` for the finite-difference cross-check (0.15 s)."""

FD_TREE_N_STEPS = 8001

FD_TOLERANCE = 5e-4
"""Budget for `|LR tree(8001) - QuantLib FD(3200, 3200)|`.

Derived, not chosen. Against the bracketed limit the Leisen-Reimer tree at
`n = 8001` sits at -3.888e-05 and QuantLib's FD at 3200 sits at -1.909e-04,
both below, so the gap is a difference of two same-signed errors: 1.52e-04.
5e-4 keeps a factor of 3.3 -- enough to absorb a platform's round-off and the
FD engine's own interpolation, not enough to pass if either engine had fallen
to half-order accuracy.
"""

# (id, spot, strike, expiry_years, expiry_days, rate, dividend, sigma)
_POINTS = (
    ("atm_1y", 100.0, 100.0, 1.0, 365, 0.05, 0.00, 0.20),
    ("otm_1y_div", 100.0, 120.0, 1.0, 365, 0.03, 0.01, 0.25),
)
_IDS = [row[0] for row in _POINTS]
_ARGS = [row[1:] for row in _POINTS]
_ARGNAMES = ("spot", "strike", "expiry", "days", "rate", "div", "sigma")

_AMERICAN_SPEC = (100.0, 100.0, 1.0, 365, 0.05, 0.00, 0.20)


def _process(spot: float, days: int, rate: float, div: float, sigma: float):
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
    return process, maturity


def _quantlib(
    kind: str,
    spot: float,
    strike: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
    *,
    american: bool,
):
    process, maturity = _process(spot, days, rate, div, sigma)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if kind == "call" else ql.Option.Put, strike
    )
    exercise = (
        ql.AmericanExercise(_EVALUATION_DATE, maturity)
        if american
        else ql.EuropeanExercise(maturity)
    )
    option = ql.VanillaOption(payoff, exercise)
    return option, process, _DAY_COUNT.yearFraction(_EVALUATION_DATE, maturity)


def _quantlib_lr(
    kind: str,
    spot: float,
    strike: float,
    days: int,
    rate: float,
    div: float,
    sigma: float,
    *,
    n_steps: int,
    american: bool = False,
) -> float:
    option, process, _ = _quantlib(
        kind, spot, strike, days, rate, div, sigma, american=american
    )
    option.setPricingEngine(
        ql.BinomialVanillaEngine(process, "leisenreimer", n_steps)
    )
    return float(option.NPV())


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
    option, process, _ = _quantlib(
        kind, spot, strike, days, rate, div, sigma, american=True
    )
    option.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, grid, grid))
    return float(option.NPV())


def _qpl_lr(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
    *,
    n_steps: int,
    american: bool = False,
) -> float:
    instrument = (
        AmericanOption(kind=kind, strike=strike, expiry=expiry)  # type: ignore[arg-type]
        if american
        else EuropeanOption(kind=kind, strike=strike, expiry=expiry)  # type: ignore[arg-type]
    )
    return price(
        instrument,
        BlackScholesModel(sigma=sigma),
        Market(
            spot=spot,
            rate_curve=FlatRateCurve(rate),
            dividend_curve=FlatDividendCurve(div),
        ),
        method="tree",
        cfg=TreeConfig(n_steps=n_steps, scheme=LR),
    ).value


# --------------------------------------------------------------------------
# The agreement
# --------------------------------------------------------------------------


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
    """`T = 365 / 365 = 1.0` exactly, so nothing below compares two engines at
    two nearly-equal maturities."""
    _, _, realised = _quantlib(
        "call", spot, strike, days, rate, div, sigma, american=False
    )
    assert realised == expiry


@pytest.mark.parametrize("n_steps", ODD_STEPS)
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_european_lr_matches_quantlib_to_round_off(
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
    """Evidence class: INDEPENDENT_ENGINE.

    The expectation this slice was written with -- "near round-off if the
    construction is the same" -- holds, which is worth stating because the
    equivalent CRR expectation did not. Both engines invert the same
    Peizer-Pratt method-2 formula at the same `d1` and `d2` and then build the
    same multipliers, so nothing is left to differ except the order of the
    floating-point operations.

    Measured residuals over these twelve comparisons: at most 2.71e-11, on
    prices from 4.16 to 21.61. See `EUROPEAN_TOLERANCE` for the budget.
    """
    ours = _qpl_lr(kind, spot, strike, expiry, rate, div, sigma, n_steps=n_steps)
    theirs = _quantlib_lr(kind, spot, strike, days, rate, div, sigma, n_steps=n_steps)
    assert ours == pytest.approx(theirs, abs=EUROPEAN_TOLERANCE)


@pytest.mark.parametrize("n_steps", AMERICAN_STEPS)
def test_american_put_lr_matches_quantlib_to_round_off(n_steps: int) -> None:
    """Evidence class: INDEPENDENT_ENGINE.

    Early exercise does not disturb the agreement: the Bellman maximum is
    applied to a continuation value both engines compute identically, so the
    two American sequences coincide to 7.3e-12 or better at every `n` here.

    That is a stronger statement than it was for CRR, where the two engines'
    probabilities differed at `O(1/n)` and the American comparison could only
    pin the discrepancy. Here there is no discrepancy to pin -- except at the
    isolated `n` of the next test.
    """
    spot, strike, expiry, days, rate, div, sigma = _AMERICAN_SPEC
    ours = _qpl_lr(
        "put", spot, strike, expiry, rate, div, sigma, n_steps=n_steps, american=True
    )
    theirs = _quantlib_lr(
        "put", spot, strike, days, rate, div, sigma, n_steps=n_steps, american=True
    )
    assert ours == pytest.approx(theirs, abs=AMERICAN_TOLERANCE)


def test_american_put_lr_agrees_with_quantlib_finite_differences() -> None:
    """Evidence class: INDEPENDENT_ENGINE.

    Two engines with nothing in common except the model: a Leisen-Reimer
    binomial rollback and an implicit finite-difference sweep on a log-spot
    grid with a projection at each step. Both approach the limit from below at
    these settings -- -3.888e-05 (tree, n = 8001) and -1.909e-04 (FD,
    3200x3200) -- so the gap is a sum of same-signed errors, 1.52e-04, and the
    budget in `FD_TOLERANCE` is derived from that rather than chosen.

    Slice 2 ran the same comparison for the CRR tree at `n = 8001` and
    measured a gap of 3.71e-04; the Leisen-Reimer leg more than halves it, at
    the same lattice size, which is the American benefit of the scheme showing
    up against an independent engine rather than against an in-repo reference.
    """
    spot, strike, expiry, days, rate, div, sigma = _AMERICAN_SPEC
    ours = _qpl_lr(
        "put",
        spot,
        strike,
        expiry,
        rate,
        div,
        sigma,
        n_steps=FD_TREE_N_STEPS,
        american=True,
    )
    fd = _quantlib_fd(
        "put", spot, strike, days, rate, div, sigma, grid=FD_GRID
    )

    assert ours == pytest.approx(fd, abs=FD_TOLERANCE)
    # Both below the limit, as the docstring table says.
    assert ours < AMERICAN_BRACKETED_LIMIT
    assert fd < AMERICAN_BRACKETED_LIMIT
    assert abs(ours - AMERICAN_BRACKETED_LIMIT) < abs(fd - AMERICAN_BRACKETED_LIMIT)


# --------------------------------------------------------------------------
# Quirk 1: QuantLib's even n
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("even_n", "odd_n"), [(50, 51), (200, 201), (800, 801)])
def test_quantlib_even_n_leisen_reimer_is_an_order_one_scheme(
    even_n: int, odd_n: int
) -> None:
    """Evidence class: NEGATIVE_FINDING.

    This is the measurement behind this package's decision to reject even `n`
    rather than round it up. QuantLib rounds up inside the tree but not inside
    the engine's time grid, and the result is not a Leisen-Reimer price: at
    the money the even-`n` error against Black-Scholes is -2.07e-01, -5.29e-02
    and -1.33e-02 at `n` = 50, 200, 800 -- halving once per doubling, i.e.
    order 1 -- while the odd-`n` error at `n + 1` is -1.32e-04, -8.71e-06 and
    -5.52e-07, halving twice per doubling.

    At `n = 800` that is a factor of 24_000 between asking for 800 steps and
    asking for 801. A caller who assumed "rounds up" would read the even
    number as the `n + 1` answer and be wrong by four orders of magnitude.
    Refusing costs that caller one character.
    """
    spot, strike, _expiry, days, rate, div, sigma = _POINTS[0][1:]
    exact = price(
        EuropeanOption(kind="call", strike=strike, expiry=1.0),
        BlackScholesModel(sigma=sigma),
        Market(
            spot=spot,
            rate_curve=FlatRateCurve(rate),
            dividend_curve=FlatDividendCurve(div),
        ),
        method="analytic",
    ).value

    even_err = abs(
        _quantlib_lr("call", spot, strike, days, rate, div, sigma, n_steps=even_n)
        - exact
    )
    odd_err = abs(
        _quantlib_lr("call", spot, strike, days, rate, div, sigma, n_steps=odd_n)
        - exact
    )

    # Not "rounded up to n+1": the even answer is nothing like the odd one.
    assert even_err > 1_000.0 * odd_err, (even_err, odd_err)
    # And it is order 1: at n = 800 an order-2 scheme would be near 1e-06.
    assert even_err > 1e-2 / (even_n / 50.0), (even_n, even_err)


# --------------------------------------------------------------------------
# Quirk 2: QuantLib's American LR at isolated n
# --------------------------------------------------------------------------

_JUMP_STEPS = (501, 1601)
"""Step counts at which QuantLib's American Leisen-Reimer leaves its curve.

Found by scanning `n` in 101..3000 on the odd grid; `n = 8001` does it too.
Pinned as data because the finding is about specific step counts, and it is
better for this list to fail loudly if a future QuantLib changes than for the
observation to rot silently in a docstring.
"""


@pytest.mark.parametrize("n_steps", _JUMP_STEPS)
def test_quantlib_american_lr_jumps_off_its_curve_at_isolated_n(n_steps: int) -> None:
    """Evidence class: NEGATIVE_FINDING.

    Everywhere else the two American sequences agree to 1e-11 (the test above
    pins three such `n`). At `n` = 501 and 1601 -- and at 8001 -- QuantLib's
    value moves 9.5e-03 and 2.8e-03 *away* from the limit while this package's
    stays on the smooth sequence its neighbours at `n - 50` and `n + 50` sit
    on.

    The direction of the discrepancy is what identifies whose it is: an error
    here would have to move this package's value, and it does not move. The
    check below asserts three things at once -- that the gap is real and far
    above round-off, that QuantLib is the side that is further from the limit,
    and that this package's error stays sandwiched between its own neighbours,
    which is what "still on the curve" means.

    Pinned so that it cannot be mistaken for a bug here. If a future QuantLib
    fixes it this test fails, and the right response is to delete the test,
    not to loosen it.
    """
    spot, strike, expiry, days, rate, div, sigma = _AMERICAN_SPEC

    def ours(n: int) -> float:
        return _qpl_lr(
            "put", spot, strike, expiry, rate, div, sigma, n_steps=n, american=True
        )

    theirs = _quantlib_lr(
        "put", spot, strike, days, rate, div, sigma, n_steps=n_steps, american=True
    )
    mine = ours(n_steps)

    assert abs(mine - theirs) > 1e-4, (mine, theirs)
    assert abs(theirs - AMERICAN_BRACKETED_LIMIT) > 10.0 * abs(
        mine - AMERICAN_BRACKETED_LIMIT
    ), (mine, theirs)

    below = abs(ours(n_steps - 50) - AMERICAN_BRACKETED_LIMIT)
    above = abs(ours(n_steps + 50) - AMERICAN_BRACKETED_LIMIT)
    here = abs(mine - AMERICAN_BRACKETED_LIMIT)
    assert above < here < below, (below, here, above)
