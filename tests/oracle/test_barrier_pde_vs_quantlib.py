"""The finite-difference barrier against QuantLib's `FdBlackScholesBarrierEngine`.

Both engines solve the same one-factor Black-Scholes PDE for the same
continuously monitored contract, so this is an independent-implementation check
of the *discretisation*, not of the formula -- the formula leg is
`tests/oracle/test_barrier_vs_quantlib.py`, where the closed forms agree to
2.886e-14.

Day count and maturity are that file's: `Actual365Fixed` over 365 days, so
`T = 1.0` exactly in double precision and there is no day-count residual to
confuse a discretisation difference with.

The result worth reading
------------------------
QuantLib's engine is **order one** on exactly the contracts whose terminal
payoff is discontinuous *across the barrier* -- an up-and-out call with
`K = 90 < H = 105`, say, whose payoff at the barrier is 15 while the rebate is
0 -- and order two on the rest. This engine is order two on both, and its
errors on those cells are three decimal orders smaller.

The mechanism is the domain, and it is the whole point of the slice. A grid
that carries the dead region has to represent a function that jumps at `H` at
`t = T`, and a jump sampled at nodes costs a full order (the Slice 6 digital
result). A grid **truncated at the barrier** never has that jump in its
interior: the payoff-versus-boundary incompatibility survives only at the
corner `(S = H, t = T)`, where four fully implicit Rannacher half steps damp
it. The attribution to a carried dead region is an inference from the measured
orders, not a claim about QuantLib's source.

`dampingSteps` does not rescue it, and costs accuracy where it is not needed --
the same shape of finding Slice 4 recorded for `FdBlackScholesVanillaEngine`.
"""

from __future__ import annotations

import pytest

ql = pytest.importorskip("QuantLib")

from qpl.engines.analytic.barrier import barrier_price  # noqa: E402
from qpl.engines.pde.pricers import PDEConfig  # noqa: E402
from qpl.instruments.options import BarrierOption  # noqa: E402
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

_SPOT, _RATE, _DIV, _SIGMA = 100.0, 0.08, 0.04, 0.25
_DOWN_BARRIER, _UP_BARRIER = 95.0, 105.0
_STRIKES = (90.0, 100.0, 110.0)

_QL_BARRIER = {
    "down-and-out": ql.Barrier.DownOut,
    "down-and-in": ql.Barrier.DownIn,
    "up-and-out": ql.Barrier.UpOut,
    "up-and-in": ql.Barrier.UpIn,
}

_N = 400
_CONCENTRATION = 0.05
_QL_DAMPING = 20

_AGREEMENT_TOLERANCE = 5.0e-3
"""Budget for the two engines' difference, derived from QuantLib's own error.

The difference is bounded by the sum of the two discretisation errors. Measured
against the closed form over the 24 cells below, at `n = 400` for both:
QuantLib's worst is **2.663e-03** (`tGrid = xGrid = 400`, 20 damping steps) and
this engine's is **2.958e-05** on the `sinh` grid -- a factor of **90**. So the
budget is QuantLib's error with about 1.9x of headroom, and the measured worst
difference is 2.663e-03, i.e. QuantLib's error almost exactly. Reading this as
an accuracy claim for *either* engine would be wrong; what it establishes is
that two independent finite-difference implementations of the same contract
land on the same number to the accuracy of the coarser one.
"""


def _process():
    ql.Settings.instance().evaluationDate = _EVALUATION_DATE
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(_SPOT)),
        ql.YieldTermStructureHandle(ql.FlatForward(_EVALUATION_DATE, _DIV, _DAY_COUNT)),
        ql.YieldTermStructureHandle(ql.FlatForward(_EVALUATION_DATE, _RATE, _DAY_COUNT)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(_EVALUATION_DATE, _CALENDAR, _SIGMA, _DAY_COUNT)
        ),
    )


def _ql_option(barrier_type: str, barrier: float, kind: str, strike: float):
    maturity = _EVALUATION_DATE + ql.Period(_ONE_YEAR_DAYS, ql.Days)
    return ql.BarrierOption(
        _QL_BARRIER[barrier_type],
        barrier,
        0.0,
        ql.PlainVanillaPayoff(
            ql.Option.Call if kind == "call" else ql.Option.Put, strike
        ),
        ql.EuropeanExercise(maturity),
    )


def _ql_fd_price(
    barrier_type: str,
    barrier: float,
    kind: str,
    strike: float,
    *,
    n: int = _N,
    damping: int = _QL_DAMPING,
) -> float:
    option = _ql_option(barrier_type, barrier, kind, strike)
    option.setPricingEngine(
        ql.FdBlackScholesBarrierEngine(_process(), n, n, damping)
    )
    return float(option.NPV())


def _cfg(n: int, **kwargs: object) -> PDEConfig:
    base: dict[str, object] = {
        "n_s": n,
        "n_t": n,
        "strike_alignment": "midpoint",
        "time_stepping": "rannacher",
        "grid": "sinh",
        "concentration": _CONCENTRATION,
    }
    base.update(kwargs)
    return PDEConfig(**base)  # type: ignore[arg-type]


def _qpl_price(
    barrier_type: str, barrier: float, kind: str, strike: float, *, n: int = _N
) -> float:
    option = BarrierOption(
        kind=kind,  # type: ignore[arg-type]
        strike=strike,
        expiry=_EXPIRY,
        barrier=barrier,
        barrier_type=barrier_type,  # type: ignore[arg-type]
    )
    market = Market(
        spot=_SPOT,
        rate_curve=FlatRateCurve(_RATE),
        dividend_curve=FlatDividendCurve(_DIV),
    )
    return float(
        price(
            option,
            BlackScholesModel(sigma=_SIGMA),
            market,
            method="pde",
            cfg=_cfg(n),
        ).value
    )


def _closed_form(barrier_type: str, barrier: float, kind: str, strike: float) -> float:
    return barrier_price(
        S=_SPOT,
        K=strike,
        T=_EXPIRY,
        r=_RATE,
        sigma=_SIGMA,
        q=_DIV,
        H=barrier,
        barrier_type=barrier_type,
        kind=kind,
    )


_CELLS = [
    (barrier_type, barrier, kind, strike)
    for barrier_type, barrier in (
        ("down-and-out", _DOWN_BARRIER),
        ("down-and-in", _DOWN_BARRIER),
        ("up-and-out", _UP_BARRIER),
        ("up-and-in", _UP_BARRIER),
    )
    for kind in ("call", "put")
    for strike in _STRIKES
]
_CELL_IDS = [f"{b}-{k}-K{int(s)}" for b, _, k, s in _CELLS]


@pytest.mark.parametrize(
    ("barrier_type", "barrier", "kind", "strike"), _CELLS, ids=_CELL_IDS
)
def test_fd_barrier_agrees_with_quantlib_on_all_eight_types(
    barrier_type: str, barrier: float, kind: str, strike: float
) -> None:
    """Evidence class: INDEPENDENT_ENGINE, at the coarser engine's accuracy.

    All eight single-barrier types, both kinds, three strikes, zero rebate, at
    `T = 1.0` exactly. Each engine is also checked against the closed form it
    is approximating, so a cell where they agreed with each other and both
    disagreed with the formula would still fail.
    """
    exact = _closed_form(barrier_type, barrier, kind, strike)
    ours = _qpl_price(barrier_type, barrier, kind, strike)
    theirs = _ql_fd_price(barrier_type, barrier, kind, strike)

    assert ours == pytest.approx(theirs, abs=_AGREEMENT_TOLERANCE)
    assert ours == pytest.approx(exact, abs=1e-3)
    assert theirs == pytest.approx(exact, abs=_AGREEMENT_TOLERANCE)


def test_this_engine_is_an_order_of_magnitude_closer_than_quantlibs() -> None:
    """Evidence class: INDEPENDENT_ENGINE, stated as a comparison and not a boast.

    Over the 24 cells at `n = 400` for both: worst error against the closed
    form 2.958e-05 here and 2.663e-03 in QuantLib, a factor of about 90. The
    next test says where that factor comes from; it is a property of the
    domain, not of the arithmetic.
    """
    ours, theirs = 0.0, 0.0
    for barrier_type, barrier, kind, strike in _CELLS:
        exact = _closed_form(barrier_type, barrier, kind, strike)
        ours = max(ours, abs(_qpl_price(barrier_type, barrier, kind, strike) - exact))
        theirs = max(
            theirs, abs(_ql_fd_price(barrier_type, barrier, kind, strike) - exact)
        )
    assert ours < 1e-4
    assert theirs > 1e-3
    assert theirs / ours > 20.0


_ORDER_LEVELS = (100, 200, 400, 800)
_ORDER_H = tuple(1.0 / n for n in _ORDER_LEVELS)

_SMOOTH_CELL = ("down-and-out", _DOWN_BARRIER, "call", 100.0)
"""`K = 100 > H = 95`: the payoff is zero on both sides of the barrier at
expiry, so the terminal condition is continuous across it."""

_JUMPING_CELL = ("down-and-out", _DOWN_BARRIER, "put", 110.0)
"""`K = 110 > H = 95` for a *put*: the payoff at the barrier is `K - H = 15`
while the knock-out pays the rebate `0`, so the terminal condition jumps by 15
across the barrier."""


def test_quantlib_fd_barrier_is_first_order_when_the_payoff_jumps_at_the_barrier() -> None:
    """Evidence class: NEGATIVE_FINDING, about the oracle rather than about us.

    Measured fitted orders over `n = tGrid = xGrid = 100, 200, 400, 800`
    against the closed form, with 20 damping steps:

        cell                         QuantLib        this engine
        down-and-out call K=100     1.8767           1.9983
        down-and-out put  K=110     0.9502           1.9810

    and QuantLib's errors on the second cell are 9.49e-03 down to 1.36e-03
    against this engine's 3.13e-05 down to 2.05e-06 -- three decimal orders, and
    the gap *widens* under refinement because the orders differ.

    The distinguishing feature is the terminal condition **across** the
    barrier: continuous in the first cell (both sides zero), a jump of
    `K - H = 15` in the second. A grid that carries the dead region must
    represent that jump at `t = T`, and Slice 6 measured what a jump sampled at
    nodes costs a finite-difference scheme: one full order. A grid truncated at
    the barrier never has the jump in its interior.

    `dampingSteps=0` is *better* here than `dampingSteps=20` (1.9855 against
    1.8767 on the smooth cell, and 0.9945 against 0.9502 on the jumping one),
    so damping is not the missing ingredient and turning it up is not the fix.
    That mirrors the Slice 4 finding for `FdBlackScholesVanillaEngine`.
    """
    fits = {}
    for label, cell in (("smooth", _SMOOTH_CELL), ("jumping", _JUMPING_CELL)):
        barrier_type, barrier, kind, strike = cell
        exact = _closed_form(barrier_type, barrier, kind, strike)
        ql_errors = [
            abs(_ql_fd_price(barrier_type, barrier, kind, strike, n=n) - exact)
            for n in _ORDER_LEVELS
        ]
        our_errors = [
            abs(_qpl_price(barrier_type, barrier, kind, strike, n=n) - exact)
            for n in _ORDER_LEVELS
        ]
        fits[label] = (
            fit_convergence_order(_ORDER_H, ql_errors),
            fit_convergence_order(_ORDER_H, our_errors),
            ql_errors,
            our_errors,
        )

    ql_smooth, our_smooth, _, _ = fits["smooth"]
    ql_jump, our_jump, ql_jump_errors, our_jump_errors = fits["jumping"]

    # QuantLib: order two where the payoff is continuous across the barrier.
    assert ql_smooth.order == pytest.approx(2.0, abs=0.2), ql_smooth.order
    # QuantLib: order ONE where it jumps. This is the finding.
    assert ql_jump.order == pytest.approx(1.0, abs=0.15), ql_jump.order

    # This engine: order two on both.
    assert our_smooth.order == pytest.approx(2.0, abs=0.2), our_smooth.order
    assert our_jump.order == pytest.approx(2.0, abs=0.2), our_jump.order

    # And the gap widens, which is what a difference of orders means.
    ratios = [
        q / o for q, o in zip(ql_jump_errors, our_jump_errors, strict=True)
    ]
    assert ratios[-1] > ratios[0] > 100.0


def test_damping_steps_do_not_rescue_quantlibs_barrier_engine() -> None:
    """Evidence class: NEGATIVE_FINDING -- the obvious remedy, ruled out.

    Zero damping steps beat twenty on both cells, so the first-order behaviour
    above is not undamped Crank-Nicolson oscillation. Measured at `n = 400`:
    8.76e-06 against 6.00e-05 on the smooth cell and 1.37e-03 against 2.66e-03
    on the jumping one.
    """
    for cell in (_SMOOTH_CELL, _JUMPING_CELL):
        barrier_type, barrier, kind, strike = cell
        exact = _closed_form(barrier_type, barrier, kind, strike)
        undamped = abs(
            _ql_fd_price(barrier_type, barrier, kind, strike, damping=0) - exact
        )
        damped = abs(
            _ql_fd_price(barrier_type, barrier, kind, strike, damping=20) - exact
        )
        assert undamped < damped


def test_quantlib_has_no_finite_difference_engine_for_a_discrete_schedule() -> None:
    """The half of this slice the oracle cannot check, said out loud.

    `FdBlackScholesBarrierEngine` prices the continuously monitored contract;
    QuantLib's discretely monitored barrier is a *Monte Carlo* engine
    (`MCBarrierEngine(..., isBiased=True)`, checked in
    `tests/oracle/test_barrier_vs_quantlib.py`). So the projection-at-monitoring
    dates half of `qpl.engines.pde.barrier` has no finite-difference oracle
    here, and its cross-checks are this repository's own Monte Carlo
    (`tests/test_pde_barrier.py`) and the Broadie-Glasserman-Kou shift. This
    test records that, and fails if a future QuantLib grows an FD engine that
    accepts a monitoring schedule -- at which point this file should gain a leg.
    """
    assert not hasattr(ql, "FdDiscreteBarrierEngine")
    signature = ql.FdBlackScholesBarrierEngine.__init__.__doc__ or ""
    assert "monitoring" not in signature.lower()
