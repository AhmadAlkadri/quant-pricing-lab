"""Heston against QuantLib, and which of this package's four methods survives it.

QuantLib's `AnalyticHestonEngine` is a genuinely independent route to the same
number: a different characteristic function (Gatheral's arrangement), a
different contour, a different quadrature, a different codebase. Driven with
adaptive Gauss-Lobatto at a relative tolerance of 1e-13 it is the reference
here, on two parameter sets and two maturities that are exact under
`Actual365Fixed` (365 and 3650 days, so `T = 1.0` and `T = 10.0` with no
day-count residual anywhere).

What the comparison says, in four lines.

1. **Lewis and Gil-Pelaez are right everywhere.** Over all 24 cells (two
   parameter sets x two maturities x three strikes x calls and puts) the worst
   residual is **2.38e-10** for Lewis and **1.24e-11** for Gil-Pelaez, both
   attained at the Feller-violating set at `T = 1`; on the Feller-satisfying
   set every cell is at 1e-13 or better. The slice statement hoped for 1e-08.

2. **Carr-Madan is right everywhere too, once its reach is set from the law.**
   At the package default (`u_max = 12 / sqrt(c2)`, `n_quad = 512`) it is
   4.9e-02 off on the Feller-violating set -- the transform of a fat-tailed law
   does not decay on the scale `1 / sqrt(c2)`. Given `u_max = 1000` and 16384
   nodes it is the most accurate method in the table, worst **1.00e-13**.

3. **The COS *call* is the one method with no setting that works.** Its payoff
   coefficient carries `e^b` with `b ~ L sqrt(c2)`, so widening the range to
   cover a fat tail amplifies round-off exponentially: on the Feller-violating
   set at `T = 10` the direct call error runs 7.2e+00 / 1.3e+01 / 4.9e+01 /
   4.1e+02 / 1.6e+05 / 9.4e+08 / 4.4e+15 as `L` goes 8 / 10 / 14 / 20 / 28 /
   36 / 50 -- monotone *upward*, with no interior optimum below 1e+00.

4. **The COS *put* is uniformly well behaved, and that is the repair.** The
   put's coefficient integrates `e^z` over `[a, z*]`, so it carries `e^{z*} =
   K / S_0` and nothing else; its error falls monotonically with `L` and
   saturates. Pricing the put by COS and taking the call by parity gives a
   worst residual of **7.66e-07** over the 24 cells at `L = 28, N = 2048`
   (2.1e-14 on the Feller-satisfying set), against 4.4e+15 for the direct call
   at the same range. This slice does **not** change the engine to do that
   automatically -- it would destroy the one place where put-call parity is a
   real check on the COS coefficients rather than an identity of the
   implementation (Slice 14) -- so it is recorded as a recipe instead.

Two findings about QuantLib itself.

- **Its `Gatheral` and `BranchCorrection` formulations agree to round-off**
  (worst 4.2e-14 at matched Gauss-Legendre order), so its default is already
  branch-safe on these sets. The 4.9e-09 gap visible at unmatched quadrature is
  the quadrature, not the branch, and is reported as such rather than as a
  little-Heston-trap sighting.
- **Its `COSHestonEngine` has exactly the failure this package's COS has**, and
  for exactly the reason Slice 14 predicted: on the Feller-violating set at
  `T = 1` its error is flat in the term count at -1.10e-02 for `L = 10`
  (-9.92e-03 / -1.096e-02 / -1.096e-02 at 200 / 800 / 3200 terms) and falls to
  -3.0e-09 only at `L = 32`. A COS error that does not respond to `N` is a
  range error, in QuantLib's engine as in this one.

Evidence classes: INDEPENDENT_ENGINE for every residual against
`AnalyticHestonEngine`; NEGATIVE_FINDING for the COS call, for the default
Carr-Madan reach, and for the `COSHestonEngine` rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise

import pytest

ql = pytest.importorskip("QuantLib")

from qpl.engines.fourier import FourierConfig  # noqa: E402
from qpl.instruments.options import EuropeanOption  # noqa: E402
from qpl.market.curves import FlatDividendCurve, FlatRateCurve  # noqa: E402
from qpl.market.market import Market  # noqa: E402
from qpl.models.heston import HestonModel  # noqa: E402
from qpl.pricing import price  # noqa: E402

_EVALUATION_DATE = ql.Date(15, 1, 2024)
_DAY_COUNT = ql.Actual365Fixed()
_ENGINE = ql.AnalyticHestonEngine
_INTEGRATION = ql.AnalyticHestonEngine_Integration

REFERENCE_TOLERANCE = 1e-13
REFERENCE_EVALUATIONS = 10**6
"""Adaptive Gauss-Lobatto settings for the reference engine.

Tight on purpose: QuantLib's default `AnalyticHestonEngine(model, 144)` is a
fixed 144-point Gauss-Laguerre rule and is 2.9e-08 off on the Feller-violating
set at `T = 1` -- four decimal orders outside the tolerances this file
asserts. The adaptive rule at 1e-13 agrees with a 1024-point Gauss-Legendre
rule to 1.05e-12 on every cell, which is the check that the reference is
converged."""


@dataclass(frozen=True)
class OraclePoint:
    """One Heston specification, in the units each side wants it."""

    name: str
    spot: float
    rate: float
    dividend: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    @property
    def feller_number(self) -> float:
        return 4.0 * self.kappa * self.theta / (self.xi * self.xi)

    def qpl_model(self) -> HestonModel:
        return HestonModel(
            v0=self.v0, kappa=self.kappa, theta=self.theta, xi=self.xi, rho=self.rho
        )

    def qpl_market(self) -> Market:
        return Market(
            spot=self.spot,
            rate_curve=FlatRateCurve(self.rate),
            dividend_curve=FlatDividendCurve(self.dividend),
        )

    def ql_model(self):
        ql.Settings.instance().evaluationDate = _EVALUATION_DATE
        spot = ql.QuoteHandle(ql.SimpleQuote(self.spot))
        rates = ql.YieldTermStructureHandle(
            ql.FlatForward(_EVALUATION_DATE, self.rate, _DAY_COUNT)
        )
        dividends = ql.YieldTermStructureHandle(
            ql.FlatForward(_EVALUATION_DATE, self.dividend, _DAY_COUNT)
        )
        return ql.HestonModel(
            ql.HestonProcess(
                rates, dividends, spot, self.v0, self.kappa, self.theta, self.xi, self.rho
            )
        )


LEWIS = OraclePoint(
    name="lewis", spot=100.0, rate=0.01, dividend=0.02,
    v0=0.04, kappa=4.0, theta=0.25, xi=1.0, rho=-0.5,
)
"""The published reference set. Feller number 4.0: the condition **holds**."""

FELLER_VIOLATED = OraclePoint(
    name="feller_violated", spot=100.0, rate=0.01, dividend=0.02,
    v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9,
)
"""The variance parameters of `qpl.cases.sde_discretization.CIR_FELLER_VIOLATED`
with `rho = -0.9`. Feller number 0.08, so zero variance is attainable and the
log-return density has the fat left tail every range rule in this file trips
over."""

POINTS = (LEWIS, FELLER_VIOLATED)
DAY_COUNTS = ((365, 1.0), (3650, 10.0))
"""`(days, years)`. Both are exact under `Actual365Fixed`."""
STRIKES = (80.0, 100.0, 120.0)
KINDS = ("call", "put")

_QL_KINDS = {"call": ql.Option.Call, "put": ql.Option.Put}


@lru_cache(maxsize=None)
def _reference(point: OraclePoint, strike: float, kind: str, days: int) -> float:
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(_QL_KINDS[kind], strike),
        ql.EuropeanExercise(_EVALUATION_DATE + days),
    )
    option.setPricingEngine(
        _ENGINE(
            point.ql_model(),
            _ENGINE.Gatheral,
            _INTEGRATION.gaussLobatto(
                REFERENCE_TOLERANCE, REFERENCE_TOLERANCE, REFERENCE_EVALUATIONS
            ),
        )
    )
    return option.NPV()


def _qpl(
    point: OraclePoint, strike: float, kind: str, expiry: float, cfg: FourierConfig
) -> float:
    return price(
        EuropeanOption(kind=kind, strike=strike, expiry=expiry),
        point.qpl_model(),
        point.qpl_market(),
        method="fourier",
        cfg=cfg,
    ).value


# --------------------------------------------------------------------------
# The two range-free methods, at the package defaults.
# --------------------------------------------------------------------------

RANGE_FREE_TOLERANCES = {"lewis": 1e-09, "gil_pelaez": 1e-10}
"""Worst measured over all 24 cells: Lewis 2.384e-10, Gil-Pelaez 1.242e-11,
both at the Feller-violating set at `T = 1` (on the Feller-satisfying set they
are 2.3e-14 and 2.8e-14). Each tolerance keeps a factor of four to eight.

These two are the methods with no truncation range and no damping parameter:
both integrate an analytic function along a contour with adaptive QUADPACK at
`quad_tolerance = 1e-10`. That is why they are the only two here that need no
per-parameter tuning, and it is the practical conclusion of this file."""


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("days_years", DAY_COUNTS, ids=["T1", "T10"])
@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("method", tuple(RANGE_FREE_TOLERANCES))
def test_the_contour_methods_against_quantlib(
    point: OraclePoint, days_years, strike: float, kind: str, method: str
) -> None:
    """Evidence class: INDEPENDENT_ENGINE."""
    days, expiry = days_years
    reference = _reference(point, strike, kind, days)
    computed = _qpl(point, strike, kind, expiry, FourierConfig(method=method))
    assert computed == pytest.approx(
        reference, abs=RANGE_FREE_TOLERANCES[method]
    )


# --------------------------------------------------------------------------
# Carr-Madan: the default reach fails, and why.
# --------------------------------------------------------------------------

CARR_MADAN_REACH = FourierConfig(
    method="carr_madan", carr_madan_transform="quadrature", u_max=1000.0, n_quad=16384
)
"""A reach set from the law's tails rather than from its variance.

The package default derives `u_max = 12 / sqrt(c2)` (Slice 14), which is right
for a Gaussian law and wrong for a fat-tailed one: the Heston transform decays
like `exp(-const |u|)` with a constant that has nothing to do with `sqrt(c2)`
when the Feller condition fails. Measured worst over the six cells of the
Feller-violating set at `T = 1`: **4.882e-02** at the default, 8.307e-05 at
`u_max = 200`, 3.603e-09 at 500 and 1.004e-13 at 1000."""

CARR_MADAN_TOLERANCE = 1e-12
"""Worst measured 1.004e-13 over all 24 cells with `CARR_MADAN_REACH` -- the
most accurate method in this file once its one knob is set."""


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("days_years", DAY_COUNTS, ids=["T1", "T10"])
@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("kind", KINDS)
def test_carr_madan_with_a_reach_set_from_the_tails(
    point: OraclePoint, days_years, strike: float, kind: str
) -> None:
    """INDEPENDENT_ENGINE."""
    days, expiry = days_years
    reference = _reference(point, strike, kind, days)
    assert _qpl(point, strike, kind, expiry, CARR_MADAN_REACH) == pytest.approx(
        reference, abs=CARR_MADAN_TOLERANCE
    )


def test_the_default_carr_madan_reach_is_wrong_for_a_fat_tailed_law() -> None:
    """NEGATIVE_FINDING, pinned with the number and the repair.

    `default_u_max` is `12 / sqrt(c2)`, derived in Slice 14 from the fact that
    a law with variance `c2` has a transform decaying on the scale
    `1 / sqrt(c2)`. That is a statement about a *Gaussian* law. Under a
    Feller-violating Heston the variance understates the tails badly and the
    derived reach is 50 against the 500 the integrand needs.
    """
    reference = _reference(FELLER_VIOLATED, 100.0, "call", 365)
    default = _qpl(
        FELLER_VIOLATED,
        100.0,
        "call",
        1.0,
        FourierConfig(method="carr_madan", carr_madan_transform="quadrature"),
    )
    assert abs(default - reference) > 1e-03
    assert abs(
        _qpl(FELLER_VIOLATED, 100.0, "call", 1.0, CARR_MADAN_REACH) - reference
    ) < CARR_MADAN_TOLERANCE


# --------------------------------------------------------------------------
# COS: the call has no usable range, the put does.
# --------------------------------------------------------------------------

COS_PUT_CFG = FourierConfig(truncation_l=28.0, n_terms=2048)
COS_PUT_TOLERANCES = {"lewis": 1e-13, "feller_violated": 1e-06}
"""Worst measured with `COS_PUT_CFG`: 2.132e-14 on the Feller-satisfying set
(both maturities) and 7.656e-07 on the Feller-violating one at `T = 10`
(5.705e-08 at `T = 1`). The second number is what a fat-tailed law costs a
density expansion even when the range is generous and the coefficient is
bounded; it is six decimal orders better than the direct call at the same
settings and nine better than the direct call at any settings."""


def _cos_call_via_parity(
    point: OraclePoint, strike: float, expiry: float, cfg: FourierConfig
) -> float:
    put = _qpl(point, strike, "put", expiry, cfg)
    return (
        put
        + point.spot * math.exp(-point.dividend * expiry)
        - strike * math.exp(-point.rate * expiry)
    )


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("days_years", DAY_COUNTS, ids=["T1", "T10"])
@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("kind", KINDS)
def test_the_cos_put_and_the_call_it_implies(
    point: OraclePoint, days_years, strike: float, kind: str
) -> None:
    """INDEPENDENT_ENGINE, on the only COS route that works at every cell."""
    days, expiry = days_years
    reference = _reference(point, strike, kind, days)
    computed = (
        _qpl(point, strike, "put", expiry, COS_PUT_CFG)
        if kind == "put"
        else _cos_call_via_parity(point, strike, expiry, COS_PUT_CFG)
    )
    assert computed == pytest.approx(
        reference, abs=COS_PUT_TOLERANCES[point.name]
    )


COS_CALL_RANGE_LADDER = (8.0, 10.0, 14.0, 20.0, 28.0, 36.0, 50.0)


def test_the_cos_call_has_no_usable_range_on_a_fat_tailed_long_dated_law() -> None:
    """NEGATIVE_FINDING, and the clearest statement of the COS trade-off here.

    Two range errors pull in opposite directions. Too narrow and the expansion
    misses tail mass; too wide and the call's payoff coefficient
    `chi_k(z*, b) ~ e^b` with `b ~ c1 + L sqrt(c2)` amplifies round-off by
    `exp(L sqrt(c2))`. There is normally a window between them -- on the
    Feller-satisfying set at `T = 1` it is three decimal orders wide -- and at
    `v0 = 0.04, kappa = 0.5, theta = 0.04, xi = 1, rho = -0.9, T = 10` it is
    **empty**:

        L         8        10        14        20        28        36        50
        call    7.2e+00  1.3e+01  4.9e+01  4.1e+02  1.6e+05  9.4e+08  4.4e+15
        put     3.1e-02  1.0e-02  1.2e-03  4.6e-05  1.5e-06  2.1e-06  2.1e-06

    The call is monotone *increasing* in `L` with no cell below 1e+00. The put,
    whose coefficient integrates `e^z` over `[a, z*]` and therefore carries
    only `e^{z*} = K / S_0`, falls monotonically and saturates at 2.1e-06.

    The engine is deliberately **not** changed to route calls through the put:
    COS computing its put from its own coefficients is what makes put-call
    parity a real check on those coefficients rather than an identity of the
    implementation, and that is the one piece of evidence the change would
    destroy (Slice 14). The recipe is recorded here and in
    `docs/notes/heston_characteristic_function.md` instead.
    """
    reference = _reference(FELLER_VIOLATED, 100.0, "call", 3650)
    call_errors = []
    put_errors = []
    for truncation_l in COS_CALL_RANGE_LADDER:
        cfg = FourierConfig(truncation_l=truncation_l, n_terms=2048)
        call_errors.append(abs(_qpl(FELLER_VIOLATED, 100.0, "call", 10.0, cfg) - reference))
        put_errors.append(
            abs(_cos_call_via_parity(FELLER_VIOLATED, 100.0, 10.0, cfg) - reference)
        )
    assert min(call_errors) > 1.0
    assert call_errors[-1] > 1e10
    assert all(
        later >= earlier
        for earlier, later in pairwise(call_errors)
    )
    assert min(put_errors) < 1e-05
    assert put_errors[0] > put_errors[-1]


# --------------------------------------------------------------------------
# Two findings about QuantLib itself.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("days_years", DAY_COUNTS, ids=["T1", "T10"])
def test_quantlibs_gatheral_and_branch_correction_agree_at_matched_quadrature(
    point: OraclePoint, days_years
) -> None:
    """A negative result, reported rather than dressed up as a trap sighting.

    QuantLib offers `ComplexLogFormula.Gatheral` and
    `ComplexLogFormula.BranchCorrection`, and it is tempting to read a
    difference between them as the little Heston trap. There is none:
    at a matched 512- and 1024-point Gauss-Legendre rule the two agree to
    **4.2e-14** on every cell here. QuantLib's default arrangement is already
    branch-safe on these parameter sets, so the trap has to be demonstrated
    against a deliberately unstable implementation -- which is what
    `qpl.models.heston.heston_characteristic_function_original_branch` is for.
    """
    days, _ = days_years
    model = point.ql_model()
    for order in (512, 1024):
        rule = _INTEGRATION.gaussLegendre(order)
        prices = []
        for formula in (_ENGINE.Gatheral, _ENGINE.BranchCorrection):
            option = ql.VanillaOption(
                ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
                ql.EuropeanExercise(_EVALUATION_DATE + days),
            )
            option.setPricingEngine(_ENGINE(model, formula, rule))
            prices.append(option.NPV())
        assert abs(prices[0] - prices[1]) < 1e-12


QL_COS_TERMS = (200, 800, 3200)


def test_quantlibs_cos_engine_has_the_same_range_failure_this_one_does() -> None:
    """NEGATIVE_FINDING, and the Slice 14 hand-off closed from the other side.

    Slice 14 measured `COSHestonEngine` diverging in the Black-Scholes limit
    and diagnosed it as a range error rather than a series error, because no
    term count helped. The same engine on a genuinely Feller-violating Heston
    shows the same signature: at `L = 10` its error is -9.92e-03 / -1.096e-02 /
    -1.096e-02 at 200 / 800 / 3200 terms -- flat once the series has converged
    -- and only widening the range helps, to -3.0e-09 at `L = 32`.

    So the failure this package's COS method has on this parameter set is not a
    defect of this implementation: it is a property of the Fang-Oosterlee range
    rule when `c4` is reported as zero, and the reference implementation has it
    too.
    """
    model = FELLER_VIOLATED.ql_model()
    reference = _reference(FELLER_VIOLATED, 100.0, "call", 365)

    def cos_heston(truncation_l: int, terms: int) -> float:
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
            ql.EuropeanExercise(_EVALUATION_DATE + 365),
        )
        option.setPricingEngine(ql.COSHestonEngine(model, truncation_l, terms))
        return option.NPV()

    at_ten = [cos_heston(10, terms) for terms in QL_COS_TERMS]
    assert abs(at_ten[-1] - reference) > 1e-03
    assert abs(at_ten[-1] - at_ten[-2]) < 1e-06  # flat in N: a range error
    assert abs(cos_heston(32, 3200) - reference) < 1e-07


def test_the_reference_engine_is_converged() -> None:
    """What makes the adaptive rule usable as a 1e-13 oracle.

    A fixed 1024-point Gauss-Legendre rule reproduces it to 1.05e-12 on every
    cell. QuantLib's own default `AnalyticHestonEngine(model, 144)` -- a fixed
    144-point Gauss-Laguerre rule -- is at 2.5e-14 and 7.2e-13 on the
    Feller-satisfying set but **2.9e-08** on the Feller-violating one at
    `T = 1`, four decimal orders outside the tolerances asserted above, which
    is why this file never uses it.
    """
    for point in POINTS:
        for days, _ in DAY_COUNTS:
            reference = _reference(point, 100.0, "call", days)
            option = ql.VanillaOption(
                ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
                ql.EuropeanExercise(_EVALUATION_DATE + days),
            )
            option.setPricingEngine(
                _ENGINE(
                    point.ql_model(),
                    _ENGINE.Gatheral,
                    _INTEGRATION.gaussLegendre(1024),
                )
            )
            assert abs(option.NPV() - reference) < 1e-11


def test_the_two_parameter_sets_are_on_opposite_sides_of_feller() -> None:
    assert LEWIS.feller_number == pytest.approx(4.0)
    assert FELLER_VIOLATED.feller_number == pytest.approx(0.08)
    assert LEWIS.qpl_model().feller_satisfied
    assert not FELLER_VIOLATED.qpl_model().feller_satisfied
