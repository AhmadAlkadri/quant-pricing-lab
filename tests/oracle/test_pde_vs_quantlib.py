"""The finite-difference engine against QuantLib's, and what "default" means.

Two engines solve the same Black-Scholes PDE here, and almost nothing about
*how* is shared:

- `qpl` marches a grid that is **uniform in the spot** over `[0, 4 S]`, with
  `strike_alignment="midpoint"` nudging the spacing so the payoff kink sits
  exactly between two nodes, and Dirichlet boundaries written from the
  discounted forward.
- QuantLib's `FdBlackScholesVanillaEngine` marches a grid that is **uniform in
  `log S`** (`FdmBlackScholesMesher`), spanning a fixed number of standard
  deviations of the log-spot around the forward, with nothing aligned to the
  strike -- at `S = K = 100, sigma = 20%, T = 1` and 11 nodes the node nearest
  the strike is at 102.53.

So the agreement measured below is between two genuinely different
discretisations, which is what makes it worth measuring. Evidence class
INDEPENDENT_ENGINE throughout, except where marked.

**The slice that wrote this file expected QuantLib's FD engine to use
Crank-Nicolson with Rannacher damping by default. It does not.** The default
`schemeDesc` is `FdmSchemeDesc.Douglas()` and the default `dampingSteps` is
**0** -- undamped Crank-Nicolson. That is pinned below, bit-for-bit, together
with the consequence: on a grid where the time step is large relative to
`ds**2`, QuantLib's default gamma is wrong by a factor of up to **1353**,
which is two orders of magnitude worse than this package's plain-Crank-Nicolson
gamma on the same problem, because its log-spot mesher packs far more nodes
into the neighbourhood of the strike and so is far stiffer there. Its own
remedy -- `dampingSteps = 2`, i.e. two fully implicit start-up steps -- brings
it back to 1e-3, exactly as `time_stepping="rannacher"` does here.

Day count: `Actual365Fixed` makes `T = days / 365`, so 365 days is exactly
`T = 1.0` in double precision and the two engines are compared at the same `T`
with no day-count residual. The short-maturity study uses 18 days and passes
the realised `18 / 365` to `qpl` rather than a rounded 0.05.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

ql = pytest.importorskip("QuantLib")

from qpl.engines.pde.pricers import PDEConfig  # noqa: E402
from qpl.instruments.options import EuropeanOption  # noqa: E402
from qpl.market.curves import FlatDividendCurve, FlatRateCurve  # noqa: E402
from qpl.market.market import Market  # noqa: E402
from qpl.models.black_scholes import BlackScholesModel  # noqa: E402
from qpl.pricing import greeks, price  # noqa: E402
from qpl.validation import fit_convergence_order  # noqa: E402

_EVALUATION_DATE = ql.Date(15, 1, 2024)
_DAY_COUNT = ql.Actual365Fixed()
_CALENDAR = ql.NullCalendar()

_COMPARISON_N = 800
"""`tGrid = xGrid = n_s = n_t` at which the two engines are compared."""

_REFINEMENT_LEVELS = (100, 200, 400, 800)
"""The refinement that justifies the tolerances below."""

# Tolerances, derived rather than chosen. Both engines converge at order two
# (fits below), so at n = 800 each carries a known error against the closed
# form and the gap between them is bounded by the sum. Worst measured values
# over the three points and both kinds at n = 800:
#
# | quantity | worst |qpl - BS| | worst |QL - BS| | worst gap |
# |----------|----------------|-----------------|-----------|
# | price    | 2.00e-04       | 4.00e-04        | 2.00e-04  |
# | delta    | 3.59e-05       | 1.22e-05        | 4.28e-05  |
# | gamma    | 7.70e-07       | 2.71e-07        | 1.31e-06  |
#
# Each tolerance keeps a factor of about three over the worst gap, which is
# well inside one refinement level of an order-2 sequence. Tightening any of
# them to the measured gap would make the test a pin on two engines' error
# constants rather than a statement that they agree.
_PRICE_TOLERANCE = 6.0e-4
_DELTA_TOLERANCE = 1.5e-4
_GAMMA_TOLERANCE = 4.0e-6

# (id, spot, strike, rate, dividend, sigma); every point is 365 days.
_POINTS = (
    ("atm_1y", 100.0, 100.0, 0.05, 0.00, 0.20),
    ("otm_1y_div", 100.0, 110.0, 0.03, 0.01, 0.25),
    ("itm_1y_div", 120.0, 90.0, 0.03, 0.05, 0.35),
)
_ONE_YEAR_DAYS = 365
_ARGNAMES = ("spot", "strike", "rate", "div", "sigma")
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


def _ql_option(kind: str, strike: float, days: int):
    maturity = _EVALUATION_DATE + ql.Period(days, ql.Days)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if kind == "call" else ql.Option.Put, strike
    )
    return ql.VanillaOption(payoff, ql.EuropeanExercise(maturity))


def _qpl_market(spot: float, rate: float, div: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(rate),
        dividend_curve=FlatDividendCurve(div),
    )


def _qpl_cfg(n: int, *, time_stepping: str = "rannacher", alignment: str = "midpoint"):
    return PDEConfig(
        n_s=n,
        n_t=n,
        theta=0.5,
        s_max_multiplier=4.0,
        strike_alignment=alignment,  # type: ignore[arg-type]
        time_stepping=time_stepping,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# What QuantLib's engine actually does.
# --------------------------------------------------------------------------


def test_quantlib_fd_default_is_undamped_crank_nicolson() -> None:
    """The default is `Douglas()` with `dampingSteps = 0`. No Rannacher.

    Evidence class: NEGATIVE_FINDING, in the sense that it contradicts a
    written expectation and is pinned so it cannot silently change.

    Three things are asserted, all bit-for-bit:

    - `FdBlackScholesVanillaEngine(process, n, n)` is *identical* to
      `(process, n, n, 0, FdmSchemeDesc.Douglas())` -- so the default carries
      no damping steps at all;
    - `dampingSteps = 2` gives a different answer, so the parameter is real and
      the equality above is not vacuous;
    - `FdmSchemeDesc.CrankNicolson()` and `FdmSchemeDesc.Douglas()` differ only
      at round-off (1.8e-15 on the price, 2.3e-14 on gamma). They are distinct
      scheme *types* in QuantLib (8 and 1) and both report `theta = 0.5`;
      Douglas splitting in one dimension is the theta scheme, so there is
      nothing for the splitting to do and the two coincide. Choosing between
      them for a one-factor Black-Scholes problem is therefore not a choice.
    """
    process = _process(100.0, 0.05, 0.0, 0.2)

    def _values(*args) -> tuple[float, float, float]:
        option = _ql_option("call", 100.0, _ONE_YEAR_DAYS)
        option.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, *args))
        return option.NPV(), option.delta(), option.gamma()

    default = _values(400, 400)
    douglas_undamped = _values(400, 400, 0, ql.FdmSchemeDesc.Douglas())
    douglas_damped = _values(400, 400, 2, ql.FdmSchemeDesc.Douglas())
    crank_nicolson = _values(400, 400, 0, ql.FdmSchemeDesc.CrankNicolson())

    assert default == douglas_undamped
    assert default != douglas_damped

    assert ql.FdmSchemeDesc.Douglas().theta == 0.5
    assert ql.FdmSchemeDesc.CrankNicolson().theta == 0.5
    assert ql.FdmSchemeDesc.Douglas().type != ql.FdmSchemeDesc.CrankNicolson().type
    for a, b in zip(douglas_undamped, crank_nicolson, strict=True):
        assert abs(a - b) < 1e-12


def test_quantlib_mesher_is_uniform_in_log_spot_not_in_spot() -> None:
    """The two engines do not share a grid, which is the point of the oracle.

    Evidence class: INDEPENDENT_ENGINE. `FdmBlackScholesMesher` places nodes at
    equal spacing in `log S`, so in spot they are geometrically spaced and the
    strike is not on, or symmetric about, any node. `qpl`'s grid is uniform in
    spot with the strike deliberately placed at a half-integer node position.
    Agreement between them is therefore not two implementations of the same
    discretisation agreeing with each other.
    """
    process = _process(100.0, 0.05, 0.0, 0.2)
    mesher = ql.FdmBlackScholesMesher(11, process, 1.0, 100.0)
    log_locations = np.array(list(mesher.locations()))
    spacings = np.diff(log_locations)

    assert np.allclose(spacings, spacings[0], rtol=0.0, atol=1e-12)
    spots = np.exp(log_locations)
    # Geometric in spot, so the spacings in S are not equal at all.
    spot_spacings = np.diff(spots)
    assert spot_spacings[-1] > 3.0 * spot_spacings[0]
    # And the strike lands nowhere in particular.
    assert np.min(np.abs(spots - 100.0)) > 2.0


# --------------------------------------------------------------------------
# Agreement on price, delta and gamma.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_price_delta_gamma_agree_with_quantlib(
    kind: str, spot: float, strike: float, rate: float, div: float, sigma: float
) -> None:
    """Two different grids, two different meshes, three quantities.

    Evidence class: INDEPENDENT_ENGINE. The tolerances are the module
    constants, derived from both engines' measured errors at `n = 800` (see
    the comment beside them). The test also asserts each engine's own error
    against the closed form, so a failure says which of the two moved rather
    than only that they disagree.
    """
    process = _process(spot, rate, div, sigma)

    analytic = _ql_option(kind, strike, _ONE_YEAR_DAYS)
    analytic.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    bs_price, bs_delta, bs_gamma = analytic.NPV(), analytic.delta(), analytic.gamma()

    quantlib = _ql_option(kind, strike, _ONE_YEAR_DAYS)
    quantlib.setPricingEngine(
        ql.FdBlackScholesVanillaEngine(
            process, _COMPARISON_N, _COMPARISON_N, 0, ql.FdmSchemeDesc.Douglas()
        )
    )

    option = EuropeanOption(kind=kind, strike=strike, expiry=1.0)
    model = BlackScholesModel(sigma=sigma)
    market = _qpl_market(spot, rate, div)
    cfg = _qpl_cfg(_COMPARISON_N)
    qpl_price = price(option, model, market, method="pde", cfg=cfg).value
    qpl_greeks = greeks(option, model, market, method="pde", cfg=cfg)

    pairs = (
        ("price", qpl_price, quantlib.NPV(), bs_price, _PRICE_TOLERANCE),
        ("delta", qpl_greeks.delta, quantlib.delta(), bs_delta, _DELTA_TOLERANCE),
        ("gamma", qpl_greeks.gamma, quantlib.gamma(), bs_gamma, _GAMMA_TOLERANCE),
    )
    for name, ours, theirs, closed_form, tolerance in pairs:
        assert abs(ours - theirs) <= tolerance, (name, ours, theirs)
        # Each engine is separately within the same budget of the closed form,
        # which is what makes the gap tolerance a derivation and not a fudge.
        assert abs(ours - closed_form) <= tolerance, (name, "qpl", ours, closed_form)
        assert abs(theirs - closed_form) <= tolerance, (name, "ql", theirs, closed_form)


@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_both_engines_are_second_order_on_price_and_delta(
    spot: float, strike: float, rate: float, div: float, sigma: float
) -> None:
    """The rate, not the level, is what justifies the tolerances above.

    Evidence class: CONVERGENCE_ORDER for both legs.

    Measured over `n` in (100, 200, 400, 800), call, against the closed form:

    | point      | price: QL / qpl | delta: QL / qpl |
    |------------|-----------------|-----------------|
    | atm_1y     | 2.008 / 2.013   | 2.009 / 2.016   |
    | otm_1y_div | 2.009 / 2.014   | 2.008 / 1.832   |
    | itm_1y_div | 2.009 / 2.014   | 2.009 / 1.915   |

    QuantLib's fits are the tighter ones (log-space residuals 0.002 across the
    board) because its log-spot mesher has no strike-alignment effect to
    wobble. `qpl`'s delta fit at the two off-money points is looser (residuals
    0.108 and 0.299) because the spot is a node there or nearly so, and which
    side of a node it lands on changes with `n`.

    Gamma is deliberately not fitted here. At the off-money points this
    package's gamma error at `n = 800` is already down at 2.5e-08 and 5.7e-09
    -- round-off-scale, with sign changes -- and a slope fitted through that is
    measuring nothing (measured 0.44 with residual 1.00 at `otm_1y_div`). The
    gamma claim is the tolerance in the test above, not a rate.
    """
    process = _process(spot, rate, div, sigma)
    analytic = _ql_option("call", strike, _ONE_YEAR_DAYS)
    analytic.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    bs_price, bs_delta = analytic.NPV(), analytic.delta()

    option = EuropeanOption(kind="call", strike=strike, expiry=1.0)
    model = BlackScholesModel(sigma=sigma)
    market = _qpl_market(spot, rate, div)

    ql_price_err, ql_delta_err = [], []
    qpl_price_err, qpl_delta_err = [], []
    for n in _REFINEMENT_LEVELS:
        fd = _ql_option("call", strike, _ONE_YEAR_DAYS)
        fd.setPricingEngine(
            ql.FdBlackScholesVanillaEngine(process, n, n, 0, ql.FdmSchemeDesc.Douglas())
        )
        ql_price_err.append(abs(fd.NPV() - bs_price))
        ql_delta_err.append(abs(fd.delta() - bs_delta))

        cfg = _qpl_cfg(n)
        qpl_price_err.append(
            abs(price(option, model, market, method="pde", cfg=cfg).value - bs_price)
        )
        qpl_delta_err.append(
            abs(greeks(option, model, market, method="pde", cfg=cfg).delta - bs_delta)
        )

    h = [1.0 / n for n in _REFINEMENT_LEVELS]
    for label, errs, band in (
        ("ql price", ql_price_err, 0.15),
        ("ql delta", ql_delta_err, 0.15),
        ("qpl price", qpl_price_err, 0.15),
        ("qpl delta", qpl_delta_err, 0.25),
    ):
        fit = fit_convergence_order(h, errs)
        assert abs(fit.order - 2.0) <= band, (label, fit.order, fit.residual)


# --------------------------------------------------------------------------
# The start-up pathology, confirmed in the other engine.
# --------------------------------------------------------------------------


_SHORT_DAYS = 18
"""18 / 365 = 0.049315..., exactly representable and close to the T = 0.05 used
by `tests/test_pde_greeks.py`. A rounded 0.05 is not a whole number of days, so
it would put the two engines at slightly different maturities."""


def test_the_gamma_pathology_is_in_the_scheme_not_in_this_package() -> None:
    """QuantLib's undamped Crank-Nicolson fails the same way, harder.

    Evidence class: INDEPENDENT_ENGINE for the mechanism, NEGATIVE_FINDING for
    QuantLib's default. This is the cross-check that matters most: if the
    gamma blow-up pinned in `tests/test_pde_greeks.py` were a bug in this
    package's stencils or grid, a completely different implementation on a
    completely different mesh would not reproduce it.

    T = 18/365, S = K = 100, r = 5%, q = 0, sigma = 20%, closed-form gamma
    0.08955265. Grids run `xGrid = 80 * tGrid` (i.e. a time step large relative
    to `ds**2` at the strike, which is the regime the ripple needs). Relative
    gamma error `(gamma - gamma_BS) / gamma_BS`:

    | tGrid | xGrid | QL damping 0 | QL damping 2 | qpl theta  | qpl rannacher |
    |-------|-------|--------------|--------------|------------|---------------|
    | 5     | 400   | +1.799e+02   | +1.733e-02   | -2.462e-01 | +1.313e-02    |
    | 10    | 800   | -2.361e+02   | -2.970e-04   | +5.311e-01 | +4.181e-03    |
    | 20    | 1600  | -6.217e+02   | -1.110e-03   | +1.081e+00 | +9.945e-04    |
    | 40    | 3200  | -1.353e+03   | -1.065e-03   | +2.172e+00 | +2.429e-04    |

    Both undamped columns get worse as the grid is refined. QuantLib's is two
    to three orders of magnitude worse than this package's, and for a reason
    that is visible in the mesher: its nodes are equally spaced in `log S`
    across a fixed number of standard deviations, so at `xGrid = 3200` the
    spacing near the strike is far smaller than this package's uniform-in-spot
    `4 S / n_s`, and `lambda dt` -- which is what drives the amplification
    factor towards -1 -- is correspondingly larger. Finer is stiffer; the
    Crank-Nicolson ripple rewards that with a bigger error, not a smaller one.

    Both damped columns converge. Two implicit start-up steps (QuantLib) and
    four implicit half steps (here) are the same remedy at slightly different
    strengths.
    """
    expiry = _SHORT_DAYS / 365.0
    process = _process(100.0, 0.05, 0.0, 0.2)
    analytic = _ql_option("call", 100.0, _SHORT_DAYS)
    analytic.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    bs_gamma = analytic.gamma()

    option = EuropeanOption(kind="call", strike=100.0, expiry=expiry)
    model = BlackScholesModel(sigma=0.2)
    market = _qpl_market(100.0, 0.05, 0.0)

    time_levels = (5, 10, 20, 40)
    ql_undamped, ql_damped, qpl_plain, qpl_damped = [], [], [], []
    for t_grid in time_levels:
        x_grid = 80 * t_grid
        for damping, sink in ((0, ql_undamped), (2, ql_damped)):
            fd = _ql_option("call", 100.0, _SHORT_DAYS)
            fd.setPricingEngine(
                ql.FdBlackScholesVanillaEngine(
                    process, t_grid, x_grid, damping, ql.FdmSchemeDesc.Douglas()
                )
            )
            sink.append(abs(fd.gamma() - bs_gamma) / bs_gamma)
        for time_stepping, sink in (("theta", qpl_plain), ("rannacher", qpl_damped)):
            cfg = PDEConfig(
                n_s=x_grid,
                n_t=t_grid,
                theta=0.5,
                s_max_multiplier=4.0,
                strike_alignment="none",
                time_stepping=time_stepping,  # type: ignore[arg-type]
            )
            res = greeks(option, model, market, method="pde", cfg=cfg)
            sink.append(abs(res.gamma - bs_gamma) / bs_gamma)

    # Both undamped engines get worse under refinement.
    assert ql_undamped[-1] > ql_undamped[0]
    assert qpl_plain[-1] > qpl_plain[0]
    # QuantLib's is the worse of the two, by orders of magnitude.
    assert ql_undamped[-1] > 100.0 * qpl_plain[-1]
    # Both damped engines converge, monotonically for ours.
    assert ql_damped[-1] < 1e-2
    assert qpl_damped[-1] < 1e-3
    assert all(b < a for a, b in pairwise(qpl_damped)), qpl_damped
    # And the damping buys three orders of magnitude at the finest grid.
    assert ql_undamped[-1] > 1000.0 * ql_damped[-1]
    assert qpl_plain[-1] > 1000.0 * qpl_damped[-1]


def test_day_count_makes_the_expiry_exact() -> None:
    """`Actual365Fixed` makes 365 days exactly `T = 1.0`, and 18 days exactly
    `18 / 365`, so the two engines are always compared at the same maturity."""
    ql.Settings.instance().evaluationDate = _EVALUATION_DATE
    for days, expected in ((_ONE_YEAR_DAYS, 1.0), (_SHORT_DAYS, _SHORT_DAYS / 365.0)):
        maturity = _EVALUATION_DATE + ql.Period(days, ql.Days)
        assert _DAY_COUNT.yearFraction(_EVALUATION_DATE, maturity) == expected
