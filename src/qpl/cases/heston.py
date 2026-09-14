"""Benchmark cases for European options under the Heston model.

Every case pairs a specification with one
:class:`~qpl.validation.BenchmarkRow` stating what is expected, how tight the
comparison is, which :class:`~qpl.validation.EvidenceClass` justifies it, and
where the expectation comes from. Tests parametrise over these lists; the lists
themselves contain no assertions and no pytest dependency.

Four families of row:

1. **Published values.** Six prices at one parameter set, quoted to four
   decimals in a cited source and used as fixtures with that citation. This is
   the first `PUBLISHED_BENCHMARK` family in the repository whose *model* has
   no closed form at all -- every Black-Scholes case could fall back on a
   formula, and this one cannot.
2. **The Black-Scholes limit.** As `xi -> 0` with `v0 = theta` the Heston price
   must become the Black-Scholes price at `sigma = sqrt(theta)`. The expected
   values are recomputed here from the Black-Scholes formula rather than read
   back from the transform, and the rows carry the measured **order in `xi`**,
   which is 1 when `rho != 0` and 2 when `rho = 0`.
3. **Put-call parity.** Exact in any model with a forward, so the residual is a
   round-off budget -- except that under Heston at the package defaults it is
   *not* round-off, it is the COS range's truncated left tail, and the rows say
   so and pin its size.
4. **Smile shape.** Sign of the skew against the sign of `rho`, the ordering of
   `|skew|` across maturities, and the term structure of at-the-forward implied
   variance between `v0` and `theta`. These are boolean-valued claims, so each
   row's ``expected`` is 1.0 with a tolerance of 0 and the test converts the
   shape statement to that indicator -- which keeps the claim, its evidence
   class and its citation in the same data structure as every numeric row
   rather than hiding it in a test body.

Provenance: the model, its characteristic function and its cumulants are
derived in `qpl.models.heston` from the sources cited there. The six published
values below are the only numbers in this module taken from outside; everything
else is either recomputed from the Black-Scholes formula in this file or
measured in this repository, and the ``source`` field says which.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from ..engines.fourier.pricers import FourierConfig
from ..instruments.options import DigitalOption, EuropeanOption
from ..market.curves import FlatDividendCurve, FlatRateCurve
from ..market.market import Market
from ..models.black_scholes import BlackScholesModel, bs_price
from ..models.heston import HestonModel
from ..validation import BenchmarkRow, EvidenceClass

__all__ = [
    "ALL_HESTON_CASES",
    "HESTON_BS_LIMIT_CASES",
    "HESTON_BS_LIMIT_ORDER_CASES",
    "HESTON_BS_LIMIT_TOLERANCE",
    "HESTON_BS_LIMIT_XI",
    "HESTON_CALL_TOLERANCE",
    "HESTON_CROSS_METHOD_CASES",
    "HESTON_CROSS_METHOD_TOLERANCE",
    "HESTON_LEWIS_SPEC",
    "HESTON_PARITY_CASES",
    "HESTON_PARITY_TOLERANCE",
    "HESTON_PUBLISHED_CASES",
    "HESTON_PUBLISHED_TOLERANCE",
    "HESTON_PUT_TOLERANCE",
    "HESTON_SMILE_CASES",
    "HESTON_SMILE_MATURITIES",
    "HESTON_SMILE_N_TERMS",
    "HESTON_SMILE_STRIKES",
    "HESTON_SMILE_TRUNCATION_L",
    "HESTON_TERM_STRUCTURE_MATURITIES",
    "HESTON_TRANSFORM_METHODS",
    "SHAPE_HOLDS",
    "HestonCase",
    "HestonSpec",
    "black_scholes_limit_model",
    "heston_parity_residual",
    "method_config",
]


@dataclass(frozen=True)
class HestonSpec:
    """A single Heston pricing point, as scalars."""

    spot: float
    strike: float
    expiry: float
    rate: float
    dividend: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    kind: Literal["call", "put"] = "call"

    def option(self) -> EuropeanOption:
        return EuropeanOption(kind=self.kind, strike=self.strike, expiry=self.expiry)

    def digital(self, cash: float = 1.0) -> DigitalOption:
        return DigitalOption(
            kind=self.kind, strike=self.strike, expiry=self.expiry, cash=cash
        )

    def model(self) -> HestonModel:
        return HestonModel(
            v0=self.v0, kappa=self.kappa, theta=self.theta, xi=self.xi, rho=self.rho
        )

    def market(self) -> Market:
        return Market(
            spot=self.spot,
            rate_curve=FlatRateCurve(self.rate),
            dividend_curve=FlatDividendCurve(self.dividend),
        )

    def at(self, **changes) -> HestonSpec:
        return replace(self, **changes)

    @property
    def feller_number(self) -> float:
        """`4 kappa theta / xi^2`; the condition holds at `>= 2`."""
        return 4.0 * self.kappa * self.theta / (self.xi * self.xi)


@dataclass(frozen=True)
class HestonCase:
    """One benchmark row plus the point(s) it is evaluated at."""

    row: BenchmarkRow
    specs: tuple[HestonSpec, ...]

    @property
    def spec(self) -> HestonSpec:
        if len(self.specs) != 1:
            raise ValueError(f"case {self.row.id} holds {len(self.specs)} specs, not 1")
        return self.specs[0]


HESTON_TRANSFORM_METHODS: tuple[str, ...] = ("cos", "lewis", "gil_pelaez", "carr_madan")
"""The four methods every published row is checked with.

`carr_madan` is used in its direct-quadrature variant throughout: the FFT
variant interpolates between log-strike nodes and its 6.3e-04 error is a
statement about interpolation, measured in `tests/test_fourier_carr_madan.py`,
not about Heston."""


def method_config(method: str) -> FourierConfig:
    """The `FourierConfig` a case row means by a method name."""
    if method == "carr_madan":
        return FourierConfig(method=method, carr_madan_transform="quadrature")
    return FourierConfig(method=method)


# --------------------------------------------------------------------------
# (1) The published reference values.
# --------------------------------------------------------------------------

HESTON_LEWIS_SPEC = HestonSpec(
    spot=100.0,
    strike=100.0,
    expiry=1.0,
    rate=0.01,
    dividend=0.02,
    v0=0.04,
    kappa=4.0,
    theta=0.25,
    xi=1.0,
    rho=-0.5,
)
"""The parameter set behind the six published values.

Its Feller number is `4 kappa theta / xi^2 = 4.0`, so the Feller condition
**is** satisfied (`2 kappa theta = 2.0` against `xi^2 = 1.0`). Worth recording
because it is easy to assume the canonical stress-test set violates it; the
Feller-violating measurements in this slice use a different set, in
`tests/heston_points.py`.

A second property is load-bearing for the branch-cut study and is pure
coincidence: `2 kappa theta / xi^2 = 2` is an **integer**, which makes the
little Heston trap's branch jump multiply the transform by
`exp(-4 pi i kappa theta / xi^2) = 1`. The trap cannot be demonstrated on this
set at any maturity (`tests/test_heston_fourier.py`)."""

_PUBLISHED_SOURCE = (
    "QuantLib test suite (Heston process cases), which attributes the values to "
    "Alan Lewis' posting on the Wilmott forums; used as fixtures with this "
    "citation. The pricing code here is derived independently in "
    "qpl.models.heston and qpl.engines.fourier."
)

_PUBLISHED_VALUES: tuple[tuple[float, str, float], ...] = (
    (80.0, "put", 7.9589),
    (80.0, "call", 26.7748),
    (100.0, "put", 17.0553),
    (100.0, "call", 16.0702),
    (120.0, "put", 29.8110),
    (120.0, "call", 9.0249),
)

HESTON_PUBLISHED_TOLERANCE = 1e-04
"""The resolution of the source, not a statement about the methods.

The values are quoted to four decimals, so half a unit in the last place is
5e-05 and the worst measured residual is 4.51e-05. What the methods are worth
is `HESTON_CROSS_METHOD_TOLERANCE`, three decimal orders below this."""

HESTON_PUBLISHED_CASES: tuple[HestonCase, ...] = tuple(
    HestonCase(
        row=BenchmarkRow(
            id=f"heston_lewis_K{int(strike)}_{kind}",
            description=(
                f"Heston European {kind}, S=100, K={strike:g}, T=1, r=1%, q=2%, "
                "v0=0.04, kappa=4, theta=0.25, xi=1, rho=-0.5"
            ),
            expected=value,
            tolerance=HESTON_PUBLISHED_TOLERANCE,
            evidence=EvidenceClass.PUBLISHED_BENCHMARK,
            source=_PUBLISHED_SOURCE,
            notes=(
                "Tolerance is the published table's four-decimal resolution. The "
                "three contour methods agree with each other to 1.0e-09, which is "
                "what identifies the 1.3e-05 to 4.5e-05 residuals as the table's "
                "rounding rather than method error."
            ),
        ),
        specs=(HESTON_LEWIS_SPEC.at(strike=strike, kind=kind),),
    )
    for strike, kind, value in _PUBLISHED_VALUES
)

HESTON_CROSS_METHOD_TOLERANCE = 5e-09
"""Spread allowed across Lewis, Gil-Pelaez and Carr-Madan on a published row.

Worst measured 1.04e-09. COS is excluded from this comparison and carries its
own two tolerances below, because its put is four decimal orders worse than its
call and averaging that away would hide the slice's most specific finding."""

HESTON_CALL_TOLERANCE = 1e-11
HESTON_PUT_TOLERANCE = 1e-07
"""The COS call and the COS put, against a Lewis integral of the same transform.

Measured at the package defaults (`L = 10`, `N = 256`) over the three published
strikes: call 9.97e-13 to 1.41e-12, put **2.70e-08 at every strike**. The two
are the same cosine sum against the same coefficients and differ only in which
end of `[a, b]` the payoff coefficient integrates from; with `rho = -0.5` the
density is left-skewed and the range misses mass at the lower end, which is the
put's end. The put error being strike-independent is what identifies it as
missing tail mass."""

HESTON_CROSS_METHOD_CASES: tuple[HestonCase, ...] = tuple(
    HestonCase(
        row=BenchmarkRow(
            id=f"heston_cross_method_K{int(strike)}_{kind}",
            description=(
                f"Lewis, Gil-Pelaez and Carr-Madan agree on the {kind} at "
                f"K={strike:g} far inside the published resolution"
            ),
            expected=0.0,
            tolerance=HESTON_CROSS_METHOD_TOLERANCE,
            evidence=EvidenceClass.CLOSED_FORM,
            source=(
                "derived in-repo: three payoff transforms of the same "
                "characteristic function (qpl.engines.fourier). Deliberately NOT "
                "labelled INDEPENDENT_ENGINE -- the three share the model's "
                "transform, so agreement is evidence about the payoff transforms "
                "and the quadratures, not about the model."
            ),
            notes="Expected value is the spread max - min over the three methods.",
        ),
        specs=(HESTON_LEWIS_SPEC.at(strike=strike, kind=kind),),
    )
    for strike, kind, _ in _PUBLISHED_VALUES
)


# --------------------------------------------------------------------------
# (2) The Black-Scholes limit.
# --------------------------------------------------------------------------

HESTON_BS_LIMIT_XI: tuple[float, ...] = (0.1, 0.05, 0.025, 0.0125, 0.00625)
"""Five halvings. Short enough to stay far above the transform's own floor and
long enough that a fitted order has five points."""

_BS_LIMIT_BASE = HestonSpec(
    spot=100.0,
    strike=100.0,
    expiry=1.0,
    rate=0.05,
    dividend=0.01,
    v0=0.04,
    kappa=1.0,
    theta=0.04,
    xi=0.1,
    rho=0.0,
)
"""`v0 = theta` so that the limit is a *constant*-variance Black-Scholes model
at `sigma = sqrt(theta) = 0.2`, with no integrated-variance correction."""


def _black_scholes_limit_value(spec: HestonSpec) -> float:
    """The Black-Scholes price the `xi -> 0` limit must equal."""
    return float(
        bs_price(
            S=spec.spot,
            K=spec.strike,
            T=spec.expiry,
            r=spec.rate,
            sigma=math.sqrt(spec.theta),
            q=spec.dividend,
            kind=spec.kind,
        )
    )


_BS_LIMIT_SOURCE = (
    "derived in this module: at xi = 0 with v0 = theta the variance is the "
    "constant theta, so the law is Black-Scholes with sigma = sqrt(theta); the "
    "expected value is that formula evaluated here, not read back from the "
    "transform"
)

HESTON_BS_LIMIT_TOLERANCE = 5e-04
"""Slack for the smallest `xi` on the ladder, at `rho = -0.5`.

Worst measured |Heston - Black-Scholes| at `xi = 0.00625`: 5.96e-05 at
`rho = 0` and 2.02e-05 to 1.54e-02 at `rho = -0.5` depending on strike. The
rows below use `rho = 0`, where the limit is second order and the measured
error is at most 1.5e-04; the correlated case is carried by the *order* rows
instead, because a tolerance on a first-order limit says nothing the order does
not say better."""

HESTON_BS_LIMIT_CASES: tuple[HestonCase, ...] = tuple(
    HestonCase(
        row=BenchmarkRow(
            id=f"heston_bs_limit_{kind}_K{int(strike)}",
            description=(
                f"xi -> 0 with v0 = theta: Heston {kind} at K={strike:g} equals the "
                "Black-Scholes price at sigma = sqrt(theta)"
            ),
            expected=_black_scholes_limit_value(
                _BS_LIMIT_BASE.at(strike=strike, kind=kind)
            ),
            tolerance=HESTON_BS_LIMIT_TOLERANCE,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_BS_LIMIT_SOURCE,
            notes=(
                "Evaluated at the smallest xi on HESTON_BS_LIMIT_XI with rho = 0, "
                "where the leading correction is O(xi^2)."
            ),
        ),
        specs=(
            _BS_LIMIT_BASE.at(strike=strike, kind=kind, xi=HESTON_BS_LIMIT_XI[-1]),
        ),
    )
    for strike in (80.0, 100.0, 120.0)
    for kind in ("call", "put")
)

_BS_ORDER_SOURCE = (
    "derived in qpl.models.heston: the leading correction to the Black-Scholes "
    "transform is the covariance term rho xi, which enters d through "
    "beta = kappa - rho xi i u and is therefore O(xi); at rho = 0 that term is "
    "absent and the next one is the xi^2 inside the discriminant"
)

HESTON_BS_LIMIT_ORDER_CASES: tuple[HestonCase, ...] = (
    HestonCase(
        row=BenchmarkRow(
            id="heston_bs_limit_order_uncorrelated",
            description="order in xi of the Black-Scholes limit at rho = 0",
            expected=2.0,
            tolerance=0.05,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_BS_ORDER_SOURCE,
            notes=(
                "Measured 1.996 / 1.997 / 1.991 at K = 80 / 100 / 120, every "
                "log-space residual below 0.006."
            ),
        ),
        specs=tuple(
            _BS_LIMIT_BASE.at(strike=100.0, xi=xi) for xi in HESTON_BS_LIMIT_XI
        ),
    ),
    HestonCase(
        row=BenchmarkRow(
            id="heston_bs_limit_order_correlated",
            description="order in xi of the Black-Scholes limit at rho = -0.5",
            expected=1.0,
            tolerance=0.06,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_BS_ORDER_SOURCE,
            notes=(
                "Measured 0.999 / 0.951 / 1.030 at K = 80 / 90 / 120. NOT "
                "measurable at the money: the rho xi term is odd in "
                "log-moneyness, changes sign across the strike and vanishes near "
                "the forward, where the fit reports order 0.32 with a log-space "
                "residual of 0.33. The slice statement expected order 2 for every "
                "rho; that holds only at rho = 0."
            ),
        ),
        specs=tuple(
            _BS_LIMIT_BASE.at(strike=120.0, xi=xi, rho=-0.5)
            for xi in HESTON_BS_LIMIT_XI
        ),
    ),
)


# --------------------------------------------------------------------------
# (3) Put-call parity.
# --------------------------------------------------------------------------


def heston_parity_residual(
    call_price: float, put_price: float, spec: HestonSpec
) -> float:
    """`C - P - (S e^{-qT} - K e^{-rT})`, which must vanish in any model."""
    forward_leg = spec.spot * math.exp(-spec.dividend * spec.expiry)
    strike_leg = spec.strike * math.exp(-spec.rate * spec.expiry)
    return (call_price - put_price) - (forward_leg - strike_leg)


_PARITY_SOURCE = (
    "derived: static replication of C - P by a forward contract, which holds in "
    "any model with a traded forward; standard result, e.g. Hull, 'Options, "
    "Futures, and Other Derivatives', chapter on properties of stock options"
)

HESTON_PARITY_TOLERANCE = 1e-07
"""**Not** a round-off budget, unlike every other parity row in this repository.

Under Black-Scholes the COS parity residual is 2.4e-14 (Slice 14). Under Heston
at the same settings it is 2.70e-08, and the excess is entirely the COS put's
truncated left tail: the residual is identical at all three strikes and falls
2.28e-06 -> 2.70e-08 -> 3.13e-10 -> 1.34e-11 as `L` goes 8 -> 10 -> 12 -> 14,
which is a tail-mass term and not a series term. The rows below are labelled
EXACT_IDENTITY for the relation and carry that measurement in their notes."""

HESTON_PARITY_CASES: tuple[HestonCase, ...] = tuple(
    HestonCase(
        row=BenchmarkRow(
            id=f"heston_parity_K{int(strike)}",
            description=(
                f"put-call parity residual under Heston at K={strike:g}, COS at the "
                "package defaults"
            ),
            expected=0.0,
            tolerance=HESTON_PARITY_TOLERANCE,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=_PARITY_SOURCE,
            notes=(
                "The identity is exact in the model; the tolerance is a "
                "truncation budget, not a round-off budget, because the COS put "
                "carries 2.70e-08 of missing left-tail mass at L = 10 while the "
                "COS call carries 1.4e-12. Parity is a real check for COS alone: "
                "Lewis, Gil-Pelaez and Carr-Madan build the put from the call by "
                "parity, so for them it is an identity of the implementation."
            ),
        ),
        specs=(HESTON_LEWIS_SPEC.at(strike=strike),),
    )
    for strike in (80.0, 100.0, 120.0)
)


# --------------------------------------------------------------------------
# (4) Smile shape, as boolean-valued rows.
# --------------------------------------------------------------------------

SHAPE_HOLDS = 1.0
"""``expected`` for a shape row: the indicator that the statement holds.

`BenchmarkRow` carries a float, and a shape claim is a predicate. Encoding the
predicate as ``1.0`` with tolerance ``0.0`` keeps the claim, its evidence class
and its citation in the same data structure as every numeric row; the test
evaluates the shape and passes the indicator in. The alternative -- leaving
shape claims in test bodies -- was rejected because it would make the cases
layer silent about the three statements this slice exists to check."""

HESTON_SMILE_STRIKES: tuple[float, ...] = (70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0)
HESTON_SMILE_MATURITIES: tuple[float, ...] = (0.25, 1.0, 5.0)
HESTON_TERM_STRUCTURE_MATURITIES: tuple[float, ...] = (
    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0
)

HESTON_SMILE_TRUNCATION_L = 6.0
HESTON_SMILE_N_TERMS = 512
"""COS settings for the smile rows, and they are *narrower* than the default.

The COS call's payoff coefficient carries `e^b` with `b ~ L sqrt(theta T)`, so
a long maturity costs precision the same way a large `L` does. Measured at the
forward strike, `T = 30`: -9.5e-01 at `L = 4`, -7.2e-05 at 6, +4.2e-07 at 8 and
+8.4e-05 at 10. At `T = 100` the default `L = 10` is off by 1.2e+02 on a price
of 13.3. `L = 6` is chosen because the term-structure ladder runs to `T = 30`
and it is the value that is accurate across the whole of it."""

_GATHERAL = (
    "Gatheral (2006), 'The Volatility Surface', chapter 2, for the qualitative "
    "statement; the sign and the magnitudes are measured in this repository "
    "(tests/test_heston_smile.py)"
)

HESTON_SMILE_CASES: tuple[HestonCase, ...] = (
    HestonCase(
        row=BenchmarkRow(
            id="heston_smile_negative_rho_is_decreasing_in_strike",
            description=(
                "rho < 0: implied volatility strictly decreasing in strike over "
                "K = 70 ... 140 at T = 0.25, 1 and 5"
            ),
            expected=SHAPE_HOLDS,
            tolerance=0.0,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_GATHERAL,
            notes=(
                "At-the-forward skew measured -0.2235 / -0.0943 / -0.0231 at "
                "T = 0.25 / 1 / 5."
            ),
        ),
        specs=(HESTON_LEWIS_SPEC,),
    ),
    HestonCase(
        row=BenchmarkRow(
            id="heston_smile_positive_rho_flips_the_skew",
            description="rho > 0: the at-the-forward skew is positive at every maturity",
            expected=SHAPE_HOLDS,
            tolerance=0.0,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_GATHERAL,
            notes=(
                "Measured +0.2248 / +0.0972 / +0.0245 against -0.2235 / -0.0943 / "
                "-0.0231 at rho = -0.5: the same magnitudes within 6%, sign "
                "reversed. At rho = 0 the skew is exactly 0.0 and the smile is "
                "symmetric but not flat."
            ),
        ),
        specs=(HESTON_LEWIS_SPEC.at(rho=0.5),),
    ),
    HestonCase(
        row=BenchmarkRow(
            id="heston_smile_flattens_with_maturity",
            description="|skew| strictly decreasing across T = 0.25, 1, 5",
            expected=SHAPE_HOLDS,
            tolerance=0.0,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_GATHERAL,
            notes=(
                "Measured 0.2235 > 0.0943 > 0.0231, fitted decay exponent 0.804 "
                "in T with a log-space residual of 0.097 -- reported as a summary "
                "and not as a theorem, and measurably not the 1/T of a "
                "large-maturity expansion."
            ),
        ),
        specs=(HESTON_LEWIS_SPEC,),
    ),
    HestonCase(
        row=BenchmarkRow(
            id="heston_atm_variance_rises_from_v0_toward_theta",
            description=(
                "v0 < theta: at-the-forward implied variance monotone increasing "
                "from v0 toward theta over T = 0.01 ... 30"
            ),
            expected=SHAPE_HOLDS,
            tolerance=0.0,
            evidence=EvidenceClass.CLOSED_FORM,
            source=(
                "derived: sigma_imp(F,T)^2 T is an average of the instantaneous "
                "variance to leading order, and E[v_t] = theta + (v0 - theta) "
                "e^{-kappa t} runs from v0 to theta"
            ),
            notes=(
                "Measured 0.0434 (T = 0.01) ... 0.2326 (T = 30) with v0 = 0.04, "
                "theta = 0.25. It UNDERSHOOTS theta, by 0.0174, and does so from "
                "both directions of approach (see the row below): the gap is the "
                "smile's own curvature, not a convergence failure."
            ),
        ),
        specs=(HESTON_LEWIS_SPEC,),
    ),
    HestonCase(
        row=BenchmarkRow(
            id="heston_atm_variance_falls_from_v0_toward_theta",
            description=(
                "v0 > theta: at-the-forward implied variance monotone decreasing "
                "from v0 toward theta over T = 0.01 ... 30"
            ),
            expected=SHAPE_HOLDS,
            tolerance=0.0,
            evidence=EvidenceClass.CLOSED_FORM,
            source="derived: same statement as the row above with v0 and theta swapped",
            notes=(
                "Measured 0.2448 (T = 0.01) ... 0.0387 (T = 30) with v0 = 0.25, "
                "theta = 0.04 -- again below theta at the long end, which is what "
                "makes the undershoot a curvature effect rather than a sign of "
                "incomplete convergence."
            ),
        ),
        specs=(HESTON_LEWIS_SPEC.at(v0=0.25, theta=0.04),),
    ),
)


ALL_HESTON_CASES: tuple[HestonCase, ...] = (
    HESTON_PUBLISHED_CASES
    + HESTON_CROSS_METHOD_CASES
    + HESTON_BS_LIMIT_CASES
    + HESTON_BS_LIMIT_ORDER_CASES
    + HESTON_PARITY_CASES
    + HESTON_SMILE_CASES
)
"""Every Heston row, for the "ids are unique and evidence is stated" meta-test."""


def black_scholes_limit_model(spec: HestonSpec) -> BlackScholesModel:
    """The Black-Scholes model a `v0 = theta` spec tends to as `xi -> 0`."""
    return BlackScholesModel(sigma=math.sqrt(spec.theta))
