"""The transform engines against QuantLib, and which QuantLib engine can judge them.

The slice statement said: "QuantLib's `AnalyticEuropeanEngine` is the 1e-14
oracle for prices and, if 1.43 exposes it, `COSHestonEngine`... no: QuantLib's
COS engine is Heston-specific; for BS, the oracle is the analytic engine only
(say so)." Half of that is right and the interesting half is not, so this file
says something better than "say so".

**QuantLib does have a Fourier engine that prices a Black-Scholes European.**
`AnalyticHestonEngine` -- the Lewis/Gatheral integral -- priced on a Heston
model whose volatility of volatility is driven to zero reproduces
`AnalyticEuropeanEngine` to 0.0 at every point tested, exactly. That gives this
package a genuine INDEPENDENT_ENGINE oracle for the transform *method* and not
only for the answer: two different contour integrals of two differently written
characteristic functions, agreeing.

**`COSHestonEngine` cannot be used that way, and fails in an instructive
direction.** Driven toward the same Black-Scholes limit it gets *worse*, not
better -- 3.8e-04 at vol-of-vol 1e-02, 5.0e-08 at 1e-04, 7.1e-05 at 1e-06,
8.9e-02 at 1e-08 -- and no term count rescues it: the price is unchanged to
1e-15 between 200 and 4000 cosine terms. The failure is in the *truncation
range*, not the series: Heston's cumulant formulas carry the vol-of-vol in
denominators, so as it vanishes the computed `[a, b]` stops describing the law
and no number of terms can fix a wrong interval. This package's COS method on
the same problem, using the exact Black-Scholes cumulants, is at 2.9e-12. That
is what it costs to derive a truncation range from the model that is actually
being priced, and it is the single most useful thing this oracle file learned
for the Heston slice that comes next.

Day count: `Actual365Fixed` makes `T = days / 365`, so the maturities below are
given in days and the realised `days / 365` is passed to `qpl`. No day-count
residual enters any comparison.

Evidence classes: INDEPENDENT_ENGINE for every price and Greek comparison
against QuantLib; NEGATIVE_FINDING for the `COSHestonEngine` rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

ql = pytest.importorskip("QuantLib")

from qpl.engines.fourier import FourierConfig  # noqa: E402
from qpl.instruments.options import DigitalOption, EuropeanOption  # noqa: E402
from qpl.market.curves import FlatDividendCurve, FlatRateCurve  # noqa: E402
from qpl.market.market import Market  # noqa: E402
from qpl.models.black_scholes import BlackScholesModel  # noqa: E402
from qpl.pricing import greeks, price  # noqa: E402

_EVALUATION_DATE = ql.Date(15, 1, 2024)
_DAY_COUNT = ql.Actual365Fixed()
_CALENDAR = ql.NullCalendar()


@dataclass(frozen=True)
class OraclePoint:
    """One specification, in the units each side wants it."""

    name: str
    spot: float
    strike: float
    days: int
    rate: float
    dividend: float
    sigma: float

    @property
    def expiry(self) -> float:
        return self.days / 365.0

    def qpl_market(self) -> Market:
        return Market(
            spot=self.spot,
            rate_curve=FlatRateCurve(self.rate),
            dividend_curve=FlatDividendCurve(self.dividend),
        )

    def process(self):
        ql.Settings.instance().evaluationDate = _EVALUATION_DATE
        spot = ql.QuoteHandle(ql.SimpleQuote(self.spot))
        rates = ql.YieldTermStructureHandle(
            ql.FlatForward(_EVALUATION_DATE, self.rate, _DAY_COUNT)
        )
        dividends = ql.YieldTermStructureHandle(
            ql.FlatForward(_EVALUATION_DATE, self.dividend, _DAY_COUNT)
        )
        vol = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(_EVALUATION_DATE, _CALENDAR, self.sigma, _DAY_COUNT)
        )
        return ql.BlackScholesMertonProcess(spot, dividends, rates, vol), rates, dividends, spot

    def exercise(self):
        return ql.EuropeanExercise(_EVALUATION_DATE + self.days)


POINTS: tuple[OraclePoint, ...] = (
    OraclePoint("atm_1y", 100.0, 100.0, 365, 0.05, 0.00, 0.20),
    OraclePoint("otm_9m_div", 100.0, 110.0, 274, 0.03, 0.01, 0.25),
    OraclePoint("itm_1y_div", 120.0, 90.0, 365, 0.03, 0.05, 0.35),
    OraclePoint("short_atm", 100.0, 100.0, 18, 0.01, 0.00, 0.30),
    OraclePoint("long_high_vol", 80.0, 120.0, 730, 0.02, 0.03, 0.40),
)

_QL_KINDS = {"call": ql.Option.Call, "put": ql.Option.Put}

METHOD_TOLERANCES = {
    "cos": 1e-11,
    "carr_madan": 1e-09,
    "lewis": 1e-12,
    "gil_pelaez": 1e-12,
}
"""Worst measured against QuantLib's analytic engine over the ten vanilla
cells: 2.912e-12 (COS), 2.319e-10 (Carr-Madan by direct quadrature), 3.952e-14
(Lewis), 1.421e-14 (Gil-Pelaez). Each tolerance keeps a factor of three to
seventy. The Carr-Madan entry is the direct-quadrature variant; the FFT variant
interpolates and is 6.3e-04 off, which is measured in
`tests/test_fourier_carr_madan.py` and is not an oracle question."""

DIGITAL_TOLERANCE = 1e-14
"""Worst measured 6.661e-16 (COS) and 5.551e-16 (Gil-Pelaez) against
`CashOrNothingPayoff` priced by the analytic engine."""

_DEGENERATE_VOL_OF_VOL = 1e-08
"""How far the Heston model is pushed toward Black-Scholes for the transform
oracle. Not arbitrary: the scan in this module's docstring shows
`AnalyticHestonEngine` improving monotonically toward it (3.8e-08 at 1e-04,
3.8e-12 at 1e-06, 0.0 at 1e-08) while `COSHestonEngine` gets worse."""


def _analytic_option(point: OraclePoint, kind: str, payoff: str = "vanilla"):
    process, *_ = point.process()
    quantlib_kind = _QL_KINDS[kind]
    contract = (
        ql.PlainVanillaPayoff(quantlib_kind, point.strike)
        if payoff == "vanilla"
        else ql.CashOrNothingPayoff(quantlib_kind, point.strike, 1.0)
    )
    option = ql.VanillaOption(contract, point.exercise())
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option


def _degenerate_heston_model(point: OraclePoint, vol_of_vol: float):
    """A Heston model that *is* Black-Scholes: `v0 = theta = sigma^2`, `xi -> 0`.

    With no volatility of volatility and no correlation the variance process
    cannot move off its initial level, which is its long-run mean, so the
    variance is the constant `sigma^2` and the log spot is the Black-Scholes
    one. `kappa` is then irrelevant and is set to 1.
    """
    _, rates, dividends, spot = point.process()
    process = ql.HestonProcess(
        rates,
        dividends,
        spot,
        point.sigma**2,
        1.0,
        point.sigma**2,
        vol_of_vol,
        0.0,
    )
    return ql.HestonModel(process)


def _qpl_price(point: OraclePoint, kind: str, method: str, payoff: str = "vanilla") -> float:
    instrument = (
        EuropeanOption(kind=kind, strike=point.strike, expiry=point.expiry)
        if payoff == "vanilla"
        else DigitalOption(kind=kind, strike=point.strike, expiry=point.expiry, cash=1.0)
    )
    extra = {"carr_madan_transform": "quadrature"} if method == "carr_madan" else {}
    return price(
        instrument,
        BlackScholesModel(sigma=point.sigma),
        point.qpl_market(),
        method="fourier",
        cfg=FourierConfig(method=method, **extra),
    ).value


# --------------------------------------------------------------------------
# Prices and Greeks against the analytic engine.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("kind", ("call", "put"))
@pytest.mark.parametrize("method", tuple(METHOD_TOLERANCES))
def test_every_transform_method_against_the_analytic_engine(
    point: OraclePoint, kind: str, method: str
) -> None:
    """Evidence class: INDEPENDENT_ENGINE (a different library's closed form)."""
    reference = _analytic_option(point, kind).NPV()
    assert _qpl_price(point, kind, method) == pytest.approx(
        reference, abs=METHOD_TOLERANCES[method]
    )


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("kind", ("call", "put"))
@pytest.mark.parametrize("method", ("cos", "gil_pelaez"))
def test_the_digital_against_quantlibs_cash_or_nothing_payoff(
    point: OraclePoint, kind: str, method: str
) -> None:
    """INDEPENDENT_ENGINE. QuantLib prices the same contract by Reiner-Rubinstein."""
    reference = _analytic_option(point, kind, "digital").NPV()
    assert _qpl_price(point, kind, method, "digital") == pytest.approx(
        reference, abs=DIGITAL_TOLERANCE
    )


GREEK_TOLERANCES = {
    "delta": 1e-13,
    "gamma": 1e-16,
    "vega": 5e-07,
    "rho": 5e-07,
    "theta": 5e-07,
}
"""Worst measured over the ten cells: 1.743e-14, 1.388e-17, 5.605e-08,
1.183e-07, 1.384e-07. The split is the split between the two ways this engine
produces a Greek: delta and gamma are exact derivatives of the cosine sum and
land at the price's own accuracy; vega, rho and theta are central differences
and land at their bump's `h^2` bias. QuantLib's `vega`, `rho` and `theta` are
per unit volatility, per unit rate and per year, matching this package's units
with no rescaling."""


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("kind", ("call", "put"))
def test_cos_greeks_against_quantlib(point: OraclePoint, kind: str) -> None:
    """INDEPENDENT_ENGINE for all five Greeks at once."""
    reference = _analytic_option(point, kind)
    computed = greeks(
        EuropeanOption(kind=kind, strike=point.strike, expiry=point.expiry),
        BlackScholesModel(sigma=point.sigma),
        point.qpl_market(),
        method="fourier",
        cfg=FourierConfig(),
    )
    for name, value in (
        ("delta", reference.delta()),
        ("gamma", reference.gamma()),
        ("vega", reference.vega()),
        ("rho", reference.rho()),
        ("theta", reference.theta()),
    ):
        assert getattr(computed, name) == pytest.approx(
            value, abs=GREEK_TOLERANCES[name]
        ), name


# --------------------------------------------------------------------------
# A transform oracle, and the one that cannot be.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
def test_quantlibs_heston_integral_reduces_to_black_scholes_exactly(
    point: OraclePoint,
) -> None:
    """INDEPENDENT_ENGINE, and this is the oracle the slice said did not exist.

    `AnalyticHestonEngine` is a Fourier engine: it integrates a characteristic
    function along a contour, exactly as `qpl.engines.fourier.lewis` does. Given
    a Heston model with `v0 = theta = sigma^2` and vol-of-vol 1e-08 it has
    nothing left to integrate but the Black-Scholes law, and it reproduces
    `AnalyticEuropeanEngine` to **0.0** at every point tested -- a residual
    asserted as a bound rather than as equality, because bit-for-bit agreement
    between two of another library's engines is not this repository's to pin.
    """
    reference = _analytic_option(point, "call").NPV()
    model = _degenerate_heston_model(point, _DEGENERATE_VOL_OF_VOL)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, point.strike), point.exercise()
    )
    option.setPricingEngine(ql.AnalyticHestonEngine(model, 192))
    assert abs(option.NPV() - reference) < 1e-12


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("method", tuple(METHOD_TOLERANCES))
def test_transform_against_transform(point: OraclePoint, method: str) -> None:
    """INDEPENDENT_ENGINE between two *transform* engines, not two closed forms.

    This is what the analytic comparison above cannot give: QuantLib's engine
    reaches the number by a contour integral of its own characteristic function,
    so agreement is evidence that the payoff transforms in
    `qpl.engines.fourier` are right, and not merely that some closed form is.
    """
    model = _degenerate_heston_model(point, _DEGENERATE_VOL_OF_VOL)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, point.strike), point.exercise()
    )
    option.setPricingEngine(ql.AnalyticHestonEngine(model, 192))
    assert _qpl_price(point, "call", method) == pytest.approx(
        option.NPV(), abs=METHOD_TOLERANCES[method]
    )


COS_HESTON_VOL_OF_VOL = (1e-02, 1e-04, 1e-06, 1e-08)
COS_HESTON_TERMS = (200, 1000, 4000)


def test_quantlibs_cos_engine_fails_in_the_black_scholes_limit() -> None:
    """NEGATIVE_FINDING, and the one worth carrying into the Heston slice.

    `COSHestonEngine` implements the same method as `qpl.engines.fourier.cos`.
    Pushed toward Black-Scholes it does not converge to it -- it diverges from
    it. Measured error against `AnalyticEuropeanEngine` at `atm_1y`, `L = 25`:

        vol-of-vol   1e-02      1e-04      1e-06      1e-08
        error       -3.8e-04   -5.0e-08   +7.1e-05   +8.9e-02

    and the term count is irrelevant: 200, 1000 and 4000 cosine terms give the
    same price to 1e-15. A COS error that does not respond to `N` is a *range*
    error, not a series error, and Heston's cumulant formulas -- which is where
    a COS engine gets its range -- carry the vol-of-vol in denominators. This
    package's COS method, deriving its range from the cumulants of the law it is
    actually pricing, is at 2.9e-12 on the same problem.

    The consequence for the next slice is concrete: a Heston COS implementation
    must be tested at small vol-of-vol *specifically*, because that is the
    corner where the range rule, not the expansion, decides the answer.
    """
    point = POINTS[0]
    reference = _analytic_option(point, "call").NPV()

    def cos_heston(vol_of_vol: float, terms: int) -> float:
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, point.strike), point.exercise()
        )
        option.setPricingEngine(
            ql.COSHestonEngine(_degenerate_heston_model(point, vol_of_vol), 25, terms)
        )
        return option.NPV()

    errors = {xi: abs(cos_heston(xi, 1000) - reference) for xi in COS_HESTON_VOL_OF_VOL}
    # It is worse at the deepest degeneracy than at the shallowest, which is the
    # opposite of what "converging to the Black-Scholes limit" would look like.
    assert errors[1e-08] > errors[1e-02] > 1e-05
    assert errors[1e-08] > 1e-02

    # And no number of terms helps, which is what identifies it as a range error.
    at_1e08 = [cos_heston(1e-08, terms) for terms in COS_HESTON_TERMS]
    assert max(at_1e08) - min(at_1e08) < 1e-12

    # This package's COS, same problem, same method, its own range rule.
    assert abs(_qpl_price(point, "call", "cos") - reference) < 1e-11


def test_quantlib_has_no_black_scholes_cos_engine_to_compare_against() -> None:
    """The structural half of the finding, pinned so it cannot rot.

    `COSHestonEngine` takes a `HestonModel`, not a process or a generic
    characteristic function, so there is no supported way to hand it a
    Black-Scholes law. The degenerate model above is the only route and the test
    before this one shows where it leads. If a future QuantLib grows a
    model-agnostic COS engine, this test is the one that should fail first.
    """
    point = POINTS[0]
    process, *_ = point.process()
    with pytest.raises(TypeError):
        ql.COSHestonEngine(process, 25, 1000)
