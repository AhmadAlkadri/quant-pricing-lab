"""Benchmark cases for fixed-strike Asian options under Black-Scholes.

A **fifth** id space, alongside the European, American, digital and
variance-reduction ones, and the first in which the two halves of the
instrument are backed by different *kinds* of evidence:

- the **geometric**-average rows are `CLOSED_FORM` or `PUBLISHED_BENCHMARK`,
  because the geometric average of lognormals is exactly lognormal and the
  price is Black (1976) in its moments;
- the **arithmetic**-average rows are `STATISTICAL`, because there is no closed
  form and the reference is a control-variate Monte Carlo run carrying its own
  standard error;
- the **Turnbull-Wakeman** rows are `PUBLISHED_BENCHMARK` *for the
  approximation* -- which is a weaker thing than a benchmark for the price, and
  the module goes out of its way to say so. Each one is paired with the
  measured gap to the simulated price, signed.

Fixing-time convention, everywhere in this package: `t_i = i T / n` for
`i = 1..n`, so the last fixing coincides with expiry, built by
`qpl.instruments.uniform_fixing_times` (which uses `linspace`, so `t_n == T`
exactly rather than one ulp above it). All three published values below are
reproduced under this convention, which is how it was chosen.

Sources. Kemna, A.G.Z. and Vorst, A.C.F. (1990), "A pricing method for options
based on average asset values", *Journal of Banking and Finance* 14, 113-129:
the continuous geometric-average closed form and the proposal to use the
geometric average as a control variate for the arithmetic one. Turnbull, S.M.
and Wakeman, L.M. (1991), *Journal of Financial and Quantitative Analysis*
26(3), 377-389, and Levy, E. (1992), *Journal of International Money and
Finance* 11, 474-491: the two-moment lognormal approximation. Glasserman (2003),
*Monte Carlo Methods in Financial Engineering*, sections 3.2 and 4.1. The three
published reference *values* are Clewlow & Strickland, *Implementing Derivatives
Models*, and Haug, *The Complete Guide to Option Pricing Formulas*, both as
carried in the QuantLib test suite (`asianoptions.cpp`), used here as fixtures
with this citation. Every formula was re-derived in
`qpl.engines.analytic.asian`, and every other number below was measured in this
repository.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from ..instruments.options import AsianOption, uniform_fixing_times
from ..market.curves import FlatDividendCurve, FlatRateCurve
from ..market.market import Market
from ..models.black_scholes import BlackScholesModel
from ..validation import BenchmarkRow, EvidenceClass

__all__ = [
    "ALL_ASIAN_CASES",
    "ASIAN_APPROXIMATION_CASES",
    "ASIAN_CONTINUOUS_LIMIT_CASES",
    "ASIAN_DENSE_FIXINGS",
    "ASIAN_IDENTITY_CASES",
    "ASIAN_KNOWN_VALUE_CASES",
    "ASIAN_MC_CASES",
    "ASIAN_MC_PATHS",
    "ASIAN_MC_SEED",
    "ASIAN_MC_STDERR_MULTIPLE",
    "ASIAN_MC_VARIANCE_REDUCTION",
    "ASIAN_ORDER_LEVELS",
    "ASIAN_REFERENCE_ATM_10F",
    "ASIAN_REFERENCE_ATM_52F",
    "CLEWLOW_STRICKLAND_SPEC",
    "CLEWLOW_STRICKLAND_VALUE",
    "HAUG_CONTINUOUS_PUT_VALUE",
    "HAUG_CONTINUOUS_SPEC",
    "TURNBULL_WAKEMAN_SPEC",
    "TURNBULL_WAKEMAN_VALUE",
    "AsianBSCase",
    "AsianBSSpec",
]


@dataclass(frozen=True)
class AsianBSSpec:
    """A single Asian pricing point, as scalars.

    `n_fixings` rather than an explicit schedule: every case here uses the
    uniform `t_i = i T / n` convention, and carrying the derived tuple in the
    dataclass would make two specs that describe the same contract compare
    unequal because their floats were built two different ways.
    """

    spot: float
    strike: float
    expiry: float
    rate: float
    dividend: float
    sigma: float
    n_fixings: int
    kind: Literal["call", "put"] = "call"
    averaging: Literal["arithmetic", "geometric"] = "geometric"

    def fixing_times(self) -> tuple[float, ...]:
        return uniform_fixing_times(self.expiry, self.n_fixings)

    def option(self) -> AsianOption:
        return AsianOption(
            kind=self.kind,
            strike=self.strike,
            expiry=self.expiry,
            fixing_times=self.fixing_times(),
            averaging=self.averaging,
        )

    def model(self) -> BlackScholesModel:
        return BlackScholesModel(sigma=self.sigma)

    def market(self) -> Market:
        return Market(
            spot=self.spot,
            rate_curve=FlatRateCurve(self.rate, allow_negative=True),
            dividend_curve=FlatDividendCurve(self.dividend, allow_negative=True),
        )

    def flipped(self) -> AsianBSSpec:
        """The same point with call and put exchanged."""
        return replace(self, kind="put" if self.kind == "call" else "call")

    def with_averaging(self, averaging: str) -> AsianBSSpec:
        return replace(self, averaging=averaging)  # type: ignore[arg-type]

    def with_fixings(self, n_fixings: int) -> AsianBSSpec:
        return replace(self, n_fixings=n_fixings)

    def discount(self) -> float:
        return math.exp(-self.rate * self.expiry)


@dataclass(frozen=True)
class AsianBSCase:
    """One benchmark row plus the point (or points) it is evaluated at."""

    row: BenchmarkRow
    specs: tuple[AsianBSSpec, ...]

    @property
    def spec(self) -> AsianBSSpec:
        if len(self.specs) != 1:
            raise ValueError(f"case {self.row.id} holds {len(self.specs)} specs, not 1")
        return self.specs[0]


# --------------------------------------------------------------------------
# The points.
# --------------------------------------------------------------------------

ASIAN_REFERENCE_ATM_10F = AsianBSSpec(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, 10, "call")
"""`S = K = 100, r = 5%, q = 0, sigma = 20%, T = 1`, ten fixings, geometric.
Deliberately the same market as the European and digital reference rows, so the
three prices can be read side by side: the vanilla call is 10.450584, the
cash-or-nothing call 0.532325, this geometric Asian 6.019116 and its arithmetic
twin 6.2342."""

ASIAN_REFERENCE_ATM_52F = ASIAN_REFERENCE_ATM_10F.with_fixings(52)
"""The same contract monitored weekly. The price falls (5.637432 against
6.019116) because more fixings mean a better-diversified average and less
variance in it, not because anything about the model changed."""

_ATM_VOL40_52F = AsianBSSpec(100.0, 100.0, 1.0, 0.05, 0.0, 0.40, 52, "call")
"""Volatility 40%, which quadruples `sigma^2 T`. This is the point where the
Turnbull-Wakeman approximation is worst (+0.89%) and where the Kemna-Vorst
control variate is weakest at the money (predicted factor 309 against 1262)."""

CLEWLOW_STRICKLAND_SPEC = AsianBSSpec(100.0, 100.0, 1.0, 0.06, 0.03, 0.20, 10, "call")
"""The published discrete geometric reference: ten fixings over a year on an
Actual/360-style schedule, which under the `t_i = i T / n` convention is `T = 1`
with `t_i = i/10`."""

CLEWLOW_STRICKLAND_VALUE = 5.3425606635
"""Clewlow & Strickland, as carried in QuantLib's `asianoptions.cpp`. Reproduced
by `qpl.engines.analytic.asian.discrete_geometric_price` to **5.8e-11**, i.e. to
the last published digit, and by QuantLib's own analytic engine to one ulp
(`tests/oracle/test_asian_vs_quantlib.py`)."""

HAUG_CONTINUOUS_SPEC = AsianBSSpec(80.0, 85.0, 0.25, 0.05, -0.03, 0.20, 1, "put")
"""The published *continuous*-averaging geometric reference, reached here by
making the fixing grid dense (`ASIAN_DENSE_FIXINGS`). `n_fixings = 1` in the
spec is a placeholder; every case built on this point overrides it.

Two corrections to how this point is usually quoted, both measured rather than
assumed (`tests/test_asian_analytic.py`):

- the published 4.6922 is the **put**; the call at the same point is 0.4714;
- it sits on a **90/360 = 0.25** year fraction. At `T = 90/365` the exact value
  is 4.6924339, which misses the published four decimals by 2.34e-04.
"""

HAUG_CONTINUOUS_PUT_VALUE = 4.6922
"""Haug, *The Complete Guide to Option Pricing Formulas*, as carried in
QuantLib's `asianoptions.cpp`."""

TURNBULL_WAKEMAN_SPEC = AsianBSSpec(
    100.0, 80.0, 0.5, 0.05, 0.05, 0.20, 26, "call", "arithmetic"
)
"""The published moment-matching reference: 26 fixings over half a year with
`q = r = 5%`, so every forward equals the spot and `E[A] = S` exactly. The
dividend yield is not usually quoted with this point and was recovered by
matching -- at `q = 0` the same construction gives 20.7865."""

TURNBULL_WAKEMAN_VALUE = 19.5152
"""The published Turnbull-Wakeman value. Reproduced to 9.8e-06 by
`turnbull_wakeman_price`, and **1.8e-03 away from the true price** (measured
below). A benchmark for an approximation is not a benchmark for what it
approximates, and this pair of numbers is the cheapest demonstration of that
distinction in the repository."""

ASIAN_DENSE_FIXINGS = 20_000
"""Fixings used to approach the continuous-averaging limit. The discrete error
at the Haug point is `0.30 / n`, so 20 000 fixings leave 1.5e-05 against the
2.1e-05 by which the published four-decimal quote differs from the exact
continuous value."""

ASIAN_ORDER_LEVELS: tuple[int, ...] = (20, 40, 80, 160, 320, 640, 1280, 2560)
"""Fixing counts for the discrete-to-continuous convergence fits; `h = 1/n`."""

ASIAN_MC_VARIANCE_REDUCTION = "control_variate"
ASIAN_MC_PATHS = 200_000
ASIAN_MC_SEED = 7
ASIAN_MC_STDERR_MULTIPLE = 4.0
"""Settings for the arithmetic legs. The reference values the rows carry were
produced at **2 000 000 paths and seed 20250913**, so the row's own run (200 000
paths, seed 7) is an independent sample and the comparison is a check rather
than a restatement; measured `z` at these settings is +0.68, +0.50, -0.43 and
+0.62 over the four points. Each tolerance is four times the root-sum-square of
the two runs' standard errors, which is what makes it a statistical claim and
not an accuracy claim."""


# --------------------------------------------------------------------------
# (i) Identities.
# --------------------------------------------------------------------------

_PARITY_SOURCE = (
    "derived: max(G - K, 0) - max(K - G, 0) = G - K identically, so "
    "C - P = e^{-rT}(E[G] - K) with E[G] = exp(m + v/2) from the exactly "
    "normal log G; see the module docstring of qpl.engines.analytic.asian. "
    "The continuous-averaging form of the same closed form is Kemna & Vorst "
    "(1990), Journal of Banking and Finance 14, 113-129"
)

_AM_GM_SOURCE = (
    "derived: the arithmetic-geometric mean inequality on the n positive "
    "fixings, which holds path by path and owes nothing to the model; "
    "max(. - K, 0) is non-decreasing, so the call ordering follows, and "
    "max(K - ., 0) is non-increasing, so the put ordering is reversed"
)

_IDENTITY_POINTS: tuple[tuple[str, AsianBSSpec], ...] = (
    ("atm_1y_10f", ASIAN_REFERENCE_ATM_10F),
    ("atm_1y_52f", ASIAN_REFERENCE_ATM_52F),
    ("clewlow_1y_10f", CLEWLOW_STRICKLAND_SPEC),
    ("atm_1y_52f_vol40", _ATM_VOL40_52F),
)

ASIAN_IDENTITY_CASES: tuple[AsianBSCase, ...] = tuple(
    AsianBSCase(
        row=BenchmarkRow(
            id=f"asian_geometric_parity_{name}",
            description=(
                "geometric Asian call - put - e^-rT (E[G] - K) at "
                f"S={spec.spot}, K={spec.strike}, T={spec.expiry}, "
                f"r={spec.rate}, q={spec.dividend}, sigma={spec.sigma}, "
                f"n={spec.n_fixings}"
            ),
            expected=0.0,
            tolerance=1e-12,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=_PARITY_SOURCE,
            notes=(
                "Volatility-free on the right-hand side, like the vanilla "
                "identity and unlike the price itself. Worst measured residual "
                "over these four points: 1.8e-15. The tolerance is a round-off "
                "budget for the difference of two numbers of size ~6."
            ),
        ),
        specs=(spec,),
    )
    for name, spec in _IDENTITY_POINTS
) + tuple(
    AsianBSCase(
        row=BenchmarkRow(
            id=f"asian_am_gm_{kind}_{name}",
            description=(
                f"arithmetic-average {kind} minus geometric-average {kind} at "
                f"{name}: {'positive' if kind == 'call' else 'negative'}"
            ),
            expected=0.0,
            tolerance=0.0,
            evidence=EvidenceClass.EXACT_IDENTITY,
            source=_AM_GM_SOURCE,
            notes=(
                "An ordering, not a level: `expected`/`tolerance` are 0 and the "
                "test asserts the sign. It holds **per path**, so under common "
                "random numbers it holds for every seed rather than within "
                "noise. Measured gaps at 2e6 paths: call +0.2155 (10 fixings) "
                "and +0.2167 (52); the put gap has the opposite sign, "
                "-0.1168 at ten fixings, because max(K - ., 0) is "
                "non-increasing. The ratio of the averages themselves is only "
                "1.00337 -- the payoff's kink is what amplifies it to 3.5% of "
                "the price."
            ),
        ),
        specs=(spec,),
    )
    for name, spec in _IDENTITY_POINTS
    for kind in ("call", "put")
)


# --------------------------------------------------------------------------
# (ii) Geometric closed-form values.
# --------------------------------------------------------------------------

_CLOSED_FORM_SOURCE = (
    "derived in-repo: qpl.engines.analytic.asian.discrete_geometric_price, "
    "which evaluates Black (1976) in F = E[G] with the exact total variance "
    "v = (sigma^2/n^2) sum_i (2(n-i)+1) t_i; not a published table"
)

_PUBLISHED_GEOMETRIC_SOURCE = (
    "Clewlow, L. and Strickland, C., 'Implementing Derivatives Models', "
    "discrete geometric-average reference value, as carried in the QuantLib "
    "test suite (asianoptions.cpp). Used as a fixture with this citation; the "
    "formula that reproduces it was derived independently in "
    "qpl.engines.analytic.asian"
)

ASIAN_KNOWN_VALUE_CASES: tuple[AsianBSCase, ...] = (
    AsianBSCase(
        row=BenchmarkRow(
            id="asian_geometric_clewlow_strickland",
            description=(
                "S=K=100, r=6%, q=3%, sigma=20%, T=1, ten fixings, discrete "
                "geometric-average call"
            ),
            expected=CLEWLOW_STRICKLAND_VALUE,
            tolerance=1e-8,
            evidence=EvidenceClass.PUBLISHED_BENCHMARK,
            source=_PUBLISHED_GEOMETRIC_SOURCE,
            notes=(
                "Measured residual 5.8e-11 -- the published quote has ten "
                "decimals and this formula matches all of them. The tolerance "
                "is 1e-8 because that is what a ten-decimal quote deserves; "
                "the agreement is two decades better. QuantLib's own "
                "AnalyticDiscreteGeometricAveragePriceAsianEngine agrees with "
                "this implementation to one ulp "
                "(tests/oracle/test_asian_vs_quantlib.py)."
            ),
        ),
        specs=(CLEWLOW_STRICKLAND_SPEC,),
    ),
    AsianBSCase(
        row=BenchmarkRow(
            id="asian_geometric_atm_1y_10f_call",
            description="S=K=100, r=5%, q=0, sigma=20%, T=1, ten fixings, geometric call",
            expected=6.01911607933648,
            tolerance=1e-13,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_CLOSED_FORM_SOURCE,
            notes=(
                "E[G] = 102.449520 and the total variance of log G is 0.015400, "
                "against sigma^2 T = 0.040000 for the vanilla at the same point "
                "-- the averaging removes 61.5% of the variance, which is why "
                "the Asian (6.019) is so much cheaper than the vanilla "
                "(10.451)."
            ),
        ),
        specs=(ASIAN_REFERENCE_ATM_10F,),
    ),
    AsianBSCase(
        row=BenchmarkRow(
            id="asian_geometric_atm_1y_10f_put",
            description="the same point, geometric put",
            expected=3.6890609179406733,
            tolerance=1e-13,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_CLOSED_FORM_SOURCE,
            notes="Sums with the call row through parity: 6.019116 - 3.689061 = e^-0.05 (102.449520 - 100).",
        ),
        specs=(ASIAN_REFERENCE_ATM_10F.flipped(),),
    ),
    AsianBSCase(
        row=BenchmarkRow(
            id="asian_geometric_atm_1y_52f_call",
            description="the same contract monitored weekly (52 fixings)",
            expected=5.637431620369425,
            tolerance=1e-13,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_CLOSED_FORM_SOURCE,
            notes=(
                "Below the ten-fixing value by 0.381684. More fixings average "
                "away more of the path's variance; the limit as n -> infinity "
                "is the Kemna-Vorst continuous price, approached at order 1."
            ),
        ),
        specs=(ASIAN_REFERENCE_ATM_52F,),
    ),
    AsianBSCase(
        row=BenchmarkRow(
            id="asian_geometric_atm_1y_52f_vol40_call",
            description="S=K=100, r=5%, q=0, sigma=40%, T=1, 52 fixings, geometric call",
            expected=9.517542988491071,
            tolerance=1e-13,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_CLOSED_FORM_SOURCE,
            notes="The stress point for both the approximation and the control variate.",
        ),
        specs=(_ATM_VOL40_52F,),
    ),
    AsianBSCase(
        row=BenchmarkRow(
            id="asian_geometric_itm_6m_26f_call",
            description="S=100, K=80, r=q=5%, sigma=20%, T=0.5, 26 fixings, geometric call",
            expected=19.353547926014723,
            tolerance=1e-13,
            evidence=EvidenceClass.CLOSED_FORM,
            source=_CLOSED_FORM_SOURCE,
            notes=(
                "The geometric twin of the Turnbull-Wakeman benchmark point, "
                "carried so the two can be read together: 19.353548 geometric "
                "against 19.513387 arithmetic, a gap of 0.16."
            ),
        ),
        specs=(TURNBULL_WAKEMAN_SPEC.with_averaging("geometric"),),
    ),
)


# --------------------------------------------------------------------------
# (iii) The continuous-averaging limit.
# --------------------------------------------------------------------------

_CONTINUOUS_SOURCE = (
    "Kemna, A.G.Z. and Vorst, A.C.F. (1990), 'A pricing method for options "
    "based on average asset values', Journal of Banking and Finance 14, "
    "113-129, for the continuous geometric-average closed form; the "
    "discrete-to-continuous rate and every number in `notes` were derived and "
    "measured in-repo (tests/test_asian_analytic.py)"
)

_HAUG_SOURCE = (
    "Haug, E.G., 'The Complete Guide to Option Pricing Formulas', continuous "
    "geometric-average reference value, as carried in the QuantLib test suite "
    "(asianoptions.cpp). Used as a fixture with this citation"
)

ASIAN_CONTINUOUS_LIMIT_CASES: tuple[AsianBSCase, ...] = (
    AsianBSCase(
        row=BenchmarkRow(
            id="asian_discrete_to_continuous_order_one_call",
            description=(
                "the discrete geometric price approaches the Kemna-Vorst "
                "continuous one at order 1 in the number of fixings"
            ),
            expected=1.0,
            tolerance=0.10,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_CONTINUOUS_SOURCE,
            notes=(
                "Measured 1.0002, log-space residual 0.00017, over n in "
                "(20 ... 2560) at the Clewlow-Strickland point; errors "
                "2.030e-01 down to 1.584e-03, one-signed and halving. Derived "
                "before it was measured: with t_i = iT/n the two moments are "
                "tbar = (T/2)(1 + 1/n) and "
                "v = (sigma^2 T/3)(1 + 3/(2n) + 1/(2n^2)), each carrying an "
                "O(1/n) term with no cancellation between them. A midpoint "
                "fixing convention would be order 2; the convention that puts "
                "the last fixing at expiry is a right-endpoint rule and is "
                "order 1. This is a modelling fact, not a defect -- a "
                "monthly-fixing Asian genuinely is not a continuously averaged "
                "one."
            ),
        ),
        specs=(CLEWLOW_STRICKLAND_SPEC,),
    ),
    AsianBSCase(
        row=BenchmarkRow(
            id="asian_discrete_to_continuous_order_one_put",
            description="the same rate at the Haug point, on the put",
            expected=1.0,
            tolerance=0.10,
            evidence=EvidenceClass.CONVERGENCE_ORDER,
            source=_CONTINUOUS_SOURCE,
            notes=(
                "Measured 1.0096, residual 0.00917; errors 1.622e-02 down to "
                "1.200e-04. The larger residual is the 1/n^2 term in v showing "
                "against a leading constant an order of magnitude smaller than "
                "the call point's."
            ),
        ),
        specs=(HAUG_CONTINUOUS_SPEC,),
    ),
    AsianBSCase(
        row=BenchmarkRow(
            id="asian_geometric_haug_continuous_put",
            description=(
                "S=80, K=85, r=5%, q=-3%, T=90/360, continuous geometric "
                "average put, reached with 20 000 fixings"
            ),
            expected=HAUG_CONTINUOUS_PUT_VALUE,
            tolerance=1e-4,
            evidence=EvidenceClass.PUBLISHED_BENCHMARK,
            source=_HAUG_SOURCE,
            notes=(
                "Measured gap 3.67e-05 at 20 000 fixings, of which 2.13e-05 is "
                "the rounding of the published four-decimal quote and 1.54e-05 "
                "is the remaining discretisation. Two corrections to how this "
                "point is usually stated, both pinned by their own tests: the "
                "4.6922 is the PUT (the call is 0.4714), and it is quoted on a "
                "90/360 year fraction -- at 90/365 the exact value is 4.6924339 "
                "and misses the published figure by 2.34e-04."
            ),
        ),
        specs=(HAUG_CONTINUOUS_SPEC.with_fixings(ASIAN_DENSE_FIXINGS),),
    ),
)


# --------------------------------------------------------------------------
# (iv) The arithmetic average: Monte Carlo, with its standard error.
# --------------------------------------------------------------------------

_MC_SOURCE = (
    "derived in-repo: qpl.engines.mc.asian with the Kemna-Vorst control "
    "variate (the geometric-average discounted payoff, whose mean is the exact "
    "closed form), 2 000 000 paths at seed 20250913. The control variate is "
    "Kemna & Vorst (1990), Journal of Banking and Finance 14, 113-129; the "
    "estimator layer is Glasserman (2003), section 4.1. STATISTICAL: the "
    "tolerance is four times the root-sum-square of the reference run's and "
    "the test run's standard errors, and is not an accuracy claim"
)

# (id, spec, reference value, reference stderr, test-run stderr, note)
_MC_ROWS: tuple[tuple[str, AsianBSSpec, float, float, float, str], ...] = (
    (
        "atm_1y_10f",
        ASIAN_REFERENCE_ATM_10F.with_averaging("arithmetic"),
        6.234153779990289,
        1.6951e-04,
        5.3735e-04,
        "The headline arithmetic price: 6.2342 against the geometric 6.0191 "
        "and the vanilla 10.4506. Measured z between the reference run and "
        "the row's own run: +0.68.",
    ),
    (
        "atm_1y_52f",
        ASIAN_REFERENCE_ATM_52F.with_averaging("arithmetic"),
        5.853913951404087,
        1.5785e-04,
        5.0420e-04,
        "Weekly monitoring. The arithmetic-minus-geometric gap is 0.216482 "
        "here against 0.215038 at ten fixings -- almost unchanged, because it "
        "is set by the dispersion of the log-spot over the averaging window "
        "and not by how often that window is sampled. Measured z: +0.50.",
    ),
    (
        "itm_6m_26f",
        TURNBULL_WAKEMAN_SPEC,
        19.51338719411278,
        1.0137e-04,
        3.2018e-04,
        "The true price at the Turnbull-Wakeman benchmark point, which the "
        "published 19.5152 is NOT: the approximation is 1.8e-03 above this, "
        "eighteen times the reference run's standard error. Measured z: -0.43.",
    ),
    (
        "atm_1y_52f_vol40",
        _ATM_VOL40_52F.with_averaging("arithmetic"),
        10.289898905643582,
        6.4675e-04,
        2.0709e-03,
        "Volatility 40%. The control variate's predicted factor falls from "
        "1262 to 309 here and the standard error rises accordingly, which is "
        "the same mechanism that makes the approximation worst at this point. "
        "Measured z: +0.62.",
    ),
)


def _combined_tolerance(reference_stderr: float, run_stderr: float) -> float:
    return ASIAN_MC_STDERR_MULTIPLE * math.hypot(reference_stderr, run_stderr)


ASIAN_MC_CASES: tuple[AsianBSCase, ...] = tuple(
    AsianBSCase(
        row=BenchmarkRow(
            id=f"asian_arithmetic_mc_{name}",
            description=(
                f"control-variate Monte Carlo price of the arithmetic-average "
                f"call at {name} ({ASIAN_MC_PATHS} paths, seed "
                f"{ASIAN_MC_SEED})"
            ),
            expected=reference,
            tolerance=_combined_tolerance(reference_stderr, run_stderr),
            evidence=EvidenceClass.STATISTICAL,
            source=_MC_SOURCE,
            notes=(
                f"Reference standard error {reference_stderr:.3e}, row-run "
                f"standard error {run_stderr:.3e}, tolerance "
                f"{_combined_tolerance(reference_stderr, run_stderr):.3e} "
                f"= {ASIAN_MC_STDERR_MULTIPLE:g} x their root-sum-square. "
                + note
            ),
        ),
        specs=(spec,),
    )
    for name, spec, reference, reference_stderr, run_stderr, note in _MC_ROWS
)
"""Four `STATISTICAL` rows. There is no closed form for the arithmetic average
and this layer does not pretend otherwise: the reference is a simulation with a
standard error, and the tolerance says so."""


# --------------------------------------------------------------------------
# (v) The approximation, labelled as one.
# --------------------------------------------------------------------------

_TW_PUBLISHED_SOURCE = (
    "Haug, E.G., 'The Complete Guide to Option Pricing Formulas', "
    "Turnbull-Wakeman reference value, as carried in the QuantLib test suite "
    "(asianoptions.cpp). PUBLISHED_BENCHMARK **for the approximation**: it "
    "certifies that this implementation of Turnbull & Wakeman (1991), Journal "
    "of Financial and Quantitative Analysis 26(3), 377-389, agrees with the "
    "published Turnbull-Wakeman number, and says nothing about whether either "
    "is the price"
)

_TW_GAP_SOURCE = (
    "derived in-repo: qpl.engines.analytic.asian.turnbull_wakeman_price minus "
    "the control-variate Monte Carlo reference (2 000 000 paths, seed "
    "20250913). The approximation is Turnbull & Wakeman (1991) and Levy "
    "(1992), Journal of International Money and Finance 11, 474-491; the gap "
    "is a measurement, reported with its sign and NOT asserted to be zero"
)

# (id, spec, measured gap, tolerance, note)
_TW_GAP_ROWS: tuple[tuple[str, AsianBSSpec, float, float, str], ...] = (
    (
        "atm_1y_10f",
        ASIAN_REFERENCE_ATM_10F.with_averaging("arithmetic"),
        +0.018162,
        2.3e-3,
        "+0.29% of the price. sigma^2 T = 0.040.",
    ),
    (
        "atm_1y_52f",
        ASIAN_REFERENCE_ATM_52F.with_averaging("arithmetic"),
        +0.019331,
        2.2e-3,
        "+0.33%. Monitoring frequency barely moves the approximation error.",
    ),
    (
        "itm_6m_26f",
        TURNBULL_WAKEMAN_SPEC,
        +0.001823,
        1.4e-3,
        "+0.009%, the smallest gap here: deep in the money the option is "
        "nearly a forward on the average, and the first moment -- which the "
        "two-moment fit matches EXACTLY -- determines the price.",
    ),
    (
        "atm_1y_52f_vol40",
        _ATM_VOL40_52F.with_averaging("arithmetic"),
        +0.092724,
        8.7e-3,
        "+0.89%, the worst here. Quadrupling sigma^2 T multiplies the relative "
        "error by about three.",
    ),
)

ASIAN_APPROXIMATION_CASES: tuple[AsianBSCase, ...] = (
    AsianBSCase(
        row=BenchmarkRow(
            id="asian_turnbull_wakeman_published_value",
            description=(
                "S=100, K=80, r=q=5%, sigma=20%, T=0.5, 26 fixings: this "
                "implementation of the Turnbull-Wakeman approximation against "
                "its published value"
            ),
            expected=TURNBULL_WAKEMAN_VALUE,
            tolerance=1e-4,
            evidence=EvidenceClass.PUBLISHED_BENCHMARK,
            source=_TW_PUBLISHED_SOURCE,
            notes=(
                "Measured residual 9.8e-06. Read this row next to "
                "`asian_arithmetic_mc_itm_6m_26f`, whose reference is "
                "19.513387: the approximation reproduces its own published "
                "value to 1e-05 and is 1.8e-03 from the truth, two orders of "
                "magnitude larger. The dividend yield q = r = 5% was recovered "
                "by matching; at q = 0 the same construction gives 20.7865."
            ),
        ),
        specs=(TURNBULL_WAKEMAN_SPEC,),
    ),
) + tuple(
    AsianBSCase(
        row=BenchmarkRow(
            id=f"asian_turnbull_wakeman_gap_{name}",
            description=(
                f"Turnbull-Wakeman minus the control-variate Monte Carlo price "
                f"at {name}"
            ),
            expected=gap,
            tolerance=tolerance,
            evidence=EvidenceClass.STATISTICAL,
            source=_TW_GAP_SOURCE,
            notes=(
                "The gap is POSITIVE at every point measured: the fitted "
                "lognormal is more right-skewed than the true law of the "
                "arithmetic average, so it puts too much mass where the call "
                "pays. " + note
            ),
        ),
        specs=(spec,),
    )
    for name, spec, gap, tolerance, note in _TW_GAP_ROWS
)


ALL_ASIAN_CASES: tuple[AsianBSCase, ...] = (
    ASIAN_IDENTITY_CASES
    + ASIAN_KNOWN_VALUE_CASES
    + ASIAN_CONTINUOUS_LIMIT_CASES
    + ASIAN_MC_CASES
    + ASIAN_APPROXIMATION_CASES
)
