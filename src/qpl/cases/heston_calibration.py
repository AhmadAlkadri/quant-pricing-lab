"""Benchmark cases for calibrating Heston to European quotes.

The eighth id space, and the first keyed by an **inverse** problem rather than
by a pricing one. A row here is not "this option is worth 16.07"; it is "on
this grid, from this start, the fitted `kappa` is this far from the truth, and
this is what the Jacobian said it would be". That difference is the point of
the slice: a calibration that finds parameters is not evidence that the
parameters are identified, so every recovery row is paired with a conditioning
row that says what the recovery is worth.

Six families:

1. **Clean recovery** (`CLOSED_FORM`). Quotes generated from a known model on
   a 5-strike by 6-maturity grid, recovered from a heavily perturbed start.
   One row per parameter per study set: the expected value is 0 and the
   tolerance is derived from the Jacobian's smallest singular value, not
   chosen.
2. **Conditioning** (`CLOSED_FORM`). The condition number of the residual
   Jacobian at the *true* parameters as maturities are added, under each
   objective. Deterministic linear algebra, so the tolerance is a round-off
   budget.
3. **Single-maturity degeneracy** (`NEGATIVE_FINDING`). Six starts fit one
   smile to 2.88e-06 in implied volatility while landing on `kappa` values
   spanning a factor of 2.50. This is the slice's central claim and it is
   carried as four rows -- the smile fit, the `kappa` spread, the `xi` spread
   and the `rho` recovery -- because "not identified" is not one number.
4. **Noise and the error bars** (`STATISTICAL`). Over 20 noise draws, the
   empirical spread of each fitted parameter against the standard error the
   Jacobian predicts inside a single draw. The row's `expected` is the ratio
   1.0 and its tolerance is the measured band.
5. **Objective choice** (`STATISTICAL` and `NEGATIVE_FINDING`). The
   implied-volatility and price RMSE of the three objectives on the same noisy
   surface, plus the finding that the parameters are *not* separated.
6. **Initialisation** (`NEGATIVE_FINDING`). The measured fraction of starts
   that reach the global optimum, and the multi-start recovery from one that
   does not.

Every number below is measured in this repository by
`tests/test_heston_calibration.py` and `tests/cases/test_heston_calibration_cases.py`;
none is taken from a source. The sources cited in the `source` fields are for
the *claims' provenance* -- which effect to look for and why -- not for the
values.

Provenance: Gatheral (2006), "The Volatility Surface", chapter 3 (calibration
in practice, the objective choice, the `kappa`/`xi` flat direction); Cui, del
Bano Rollin and Germano (2017), EJOR 263(2), 625-638, section 3 (the analytic
gradient and the ill-conditioning along `kappa`-`xi`); Mikhailov and Nogel
(2003), Wilmott Magazine (July), 74-79 (Heston calibration as least squares and
the role of the initial guess); Nocedal and Wright (2006), "Numerical
Optimization" 2nd ed., chapter 10 (Gauss-Newton covariance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..calibration.heston import (
    CosSettings,
    OptionQuote,
    default_cos_settings,
    heston_quote_values,
)
from ..engines.analytic.black_scholes import implied_volatility
from ..instruments.options import EuropeanOption
from ..market.curves import FlatDividendCurve, FlatRateCurve
from ..market.market import Market
from ..models.heston import HestonModel
from ..validation import BenchmarkRow, EvidenceClass

__all__ = [
    "ALL_HESTON_CALIBRATION_CASES",
    "HESTON_CALIBRATION_CONDITION_CASES",
    "HESTON_CALIBRATION_CONDITION_TOLERANCE",
    "HESTON_CALIBRATION_FELLER_VIOLATED",
    "HESTON_CALIBRATION_GLOBAL_COUNT",
    "HESTON_CALIBRATION_GLOBAL_TOLERANCE",
    "HESTON_CALIBRATION_INITIALISATION_CASES",
    "HESTON_CALIBRATION_MARKET",
    "HESTON_CALIBRATION_MULTISTART_COUNT",
    "HESTON_CALIBRATION_NOISE_CASES",
    "HESTON_CALIBRATION_NOISE_LEVELS_BP",
    "HESTON_CALIBRATION_NOISE_SEEDS",
    "HESTON_CALIBRATION_OBJECTIVE_CASES",
    "HESTON_CALIBRATION_OBJECTIVE_SEEDS",
    "HESTON_CALIBRATION_OBJECTIVE_TOLERANCE",
    "HESTON_CALIBRATION_ONE_MATURITY",
    "HESTON_CALIBRATION_RECOVERY_CASES",
    "HESTON_CALIBRATION_REFERENCE",
    "HESTON_CALIBRATION_SAFETY_FACTOR",
    "HESTON_CALIBRATION_SINGLE_MATURITY_CASES",
    "HESTON_CALIBRATION_SINGLE_MATURITY_STARTS",
    "HESTON_CALIBRATION_SIX_MATURITIES",
    "HESTON_CALIBRATION_START_GRID",
    "HESTON_CALIBRATION_STDERR_BAND",
    "HESTON_CALIBRATION_STRIKES",
    "HESTON_CALIBRATION_THREE_MATURITIES",
    "CalibrationCase",
    "CalibrationSpec",
    "noisy_quotes",
    "synthetic_quotes",
]


# --------------------------------------------------------------------------
# The experiment.
# --------------------------------------------------------------------------

HESTON_CALIBRATION_MARKET = Market(
    spot=100.0,
    rate_curve=FlatRateCurve(0.01),
    dividend_curve=FlatDividendCurve(0.02),
)
"""The Alan Lewis reference market used by every other Heston case here, so a
calibration number is comparable with a pricing one."""

HESTON_CALIBRATION_REFERENCE = (0.04, 4.0, 0.25, 1.0, -0.5)
"""`(v0, kappa, theta, xi, rho)` of `qpl.cases.heston.HESTON_LEWIS_SPEC`.
Feller number 4.0."""

HESTON_CALIBRATION_FELLER_VIOLATED = (0.04, 0.5, 0.04, 1.0, -0.9)
"""This repository's own Feller-violating set (number 0.08), shared with the
Slice 9 CIR study, the Slice 15 transform study and the Slice 16 QE study."""

HESTON_CALIBRATION_STRIKES = (80.0, 90.0, 100.0, 110.0, 120.0)
HESTON_CALIBRATION_ONE_MATURITY = (1.0,)
HESTON_CALIBRATION_THREE_MATURITIES = (0.25, 1.0, 2.0)
HESTON_CALIBRATION_SIX_MATURITIES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)

HESTON_CALIBRATION_NOISE_SEEDS = 20
HESTON_CALIBRATION_NOISE_LEVELS_BP = (5.0, 20.0)
HESTON_CALIBRATION_OBJECTIVE_SEEDS = 10
HESTON_CALIBRATION_MULTISTART_COUNT = 6

HESTON_CALIBRATION_SAFETY_FACTOR = 10.0
"""Slack over the `||r|| / s_min` bound that the clean-recovery rows assert.

A residual perturbation `dr` moves the Gauss-Newton solution by at most
`||dr|| / s_min`, so a converged fit's parameter error is at most the residual
it stopped at divided by the Jacobian's smallest singular value there. On the
reference set with the implied-volatility objective that bound is 2.67e-10
against a worst measured error of 1.48e-10 -- tight to a factor of 1.8. The
factor of 10 covers the two places the linearisation is not exact (the model is
nonlinear in the parameters, and the solver's stopping point is not the exact
minimiser). It is not headroom over a guess."""

HESTON_CALIBRATION_CONDITION_TOLERANCE = 0.05
"""Relative budget on the pinned condition numbers: they are deterministic
linear algebra, so the only drift is BLAS round-off."""

HESTON_CALIBRATION_START_GRID = (
    (0.04, 4.0, 0.25, 1.0, -0.5),
    (0.08, 2.0, 0.15, 0.6, -0.2),
    (0.01, 0.5, 0.05, 0.3, -0.9),
    (0.20, 10.0, 0.50, 2.0, -0.1),
    (0.04, 1.0, 0.04, 1.5, 0.5),
    (0.10, 0.3, 0.10, 1.8, 0.8),
    (0.50, 0.1, 0.80, 3.0, -0.99),
    (0.02, 15.0, 0.02, 0.05, -0.05),
    (0.30, 6.0, 0.30, 0.2, 0.9),
    (0.005, 0.01, 0.005, 0.01, 0.0),
    (0.9, 19.0, 0.9, 4.5, 0.95),
    (0.06, 3.0, 0.90, 2.5, -0.75),
    (0.04, 0.2, 0.60, 4.0, -0.3),
    (0.15, 8.0, 0.02, 1.2, 0.3),
)
"""Fourteen starts: the truth, a mild perturbation, four corners of the
parameter box, five with the **wrong sign** of `rho`, and six that are
Feller-violating. Spread deliberately: a start grid clustered near the answer
measures the solver's last iteration and reports a success rate of 1."""


@dataclass(frozen=True)
class CalibrationSpec:
    """One calibration experiment, as data.

    Carries the model the quotes come from, the grid they sit on, the objective
    and the starting guess -- everything needed to reproduce a row, with no
    solver settings hidden in a test body.
    """

    parameters: tuple[float, float, float, float, float]
    strikes: tuple[float, ...]
    maturities: tuple[float, ...]
    objective: str = "implied_vol"
    start: tuple[float, float, float, float, float] = (0.08, 2.0, 0.15, 0.6, -0.2)
    noise_bp: float = 0.0
    seed: int = 0

    def model(self) -> HestonModel:
        return HestonModel(*self.parameters)

    def market(self) -> Market:
        return HESTON_CALIBRATION_MARKET

    def cos_settings(self) -> CosSettings:
        return default_cos_settings(self.parameters)

    def at(self, **changes) -> CalibrationSpec:
        return replace(self, **changes)

    @property
    def feller_number(self) -> float:
        _v0, kappa, theta, xi, _rho = self.parameters
        return 4.0 * kappa * theta / (xi * xi)

    @property
    def n_quotes(self) -> int:
        return len(self.strikes) * len(self.maturities)

    def quotes(self) -> list[OptionQuote]:
        if self.noise_bp == 0.0:
            return synthetic_quotes(self.model(), self.maturities, self.strikes)
        return noisy_quotes(
            self.model(),
            self.maturities,
            self.strikes,
            noise_bp=self.noise_bp,
            seed=self.seed,
        )


@dataclass(frozen=True)
class CalibrationCase:
    """One benchmark row plus the experiment it is measured on."""

    row: BenchmarkRow
    spec: CalibrationSpec
    parameter: str | None = None
    """Which Heston parameter the row is about, where it is about one."""


# --------------------------------------------------------------------------
# Quote generation (shared by the cases tests and the example).
# --------------------------------------------------------------------------


def synthetic_quotes(model, maturities, strikes=HESTON_CALIBRATION_STRIKES):
    """Exact implied-volatility quotes from `model` on the `(T, K)` grid.

    Generated by the calibrator's own pricer on purpose: that makes a recovery
    row a statement about the **inverse problem** and not about the transform,
    which Slice 15 already checked against six published values and against
    QuantLib. A synthetic-recovery study whose forward model differs from the
    calibrator's measures the gap between two pricers instead.
    """
    market = HESTON_CALIBRATION_MARKET
    skeleton = [
        OptionQuote(strike=k, expiry=t, kind="call", value=1.0, value_type="price")
        for t in maturities
        for k in strikes
    ]
    prices = heston_quote_values(model, market, skeleton)
    quotes = []
    for quote, price in zip(skeleton, prices, strict=True):
        option = EuropeanOption(kind="call", strike=quote.strike, expiry=quote.expiry)
        quotes.append(
            OptionQuote(
                strike=quote.strike,
                expiry=quote.expiry,
                kind="call",
                value=float(implied_volatility(float(price), option, market)),
                value_type="implied_vol",
            )
        )
    return quotes


def noisy_quotes(
    model,
    maturities,
    strikes=HESTON_CALIBRATION_STRIKES,
    *,
    noise_bp: float,
    seed: int,
):
    """`synthetic_quotes` plus independent Gaussian implied-volatility noise.

    The noise is added in **volatility** units because that is how a bid/ask is
    quoted, and because it makes the price and implied-volatility objectives
    comparable on one data set rather than on two.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    return [
        OptionQuote(
            strike=quote.strike,
            expiry=quote.expiry,
            kind="call",
            value=float(quote.value + rng.normal(0.0, noise_bp * 1e-04)),
            value_type="implied_vol",
        )
        for quote in synthetic_quotes(model, maturities, strikes)
    ]


# --------------------------------------------------------------------------
# (1) Clean synthetic recovery.
# --------------------------------------------------------------------------

_RECOVERY_MEASURED = {
    ("reference", "price"): (2.95e-09, 5.42e-07, 3.28e-09, 4.08e-07, 1.29e-07),
    ("reference", "implied_vol"): (2.19e-12, 1.48e-10, 2.06e-12, 9.68e-12, 2.85e-12),
    ("feller_violated", "price"): (2.51e-10, 8.23e-08, 5.75e-09, 1.27e-09, 5.76e-09),
    ("feller_violated", "implied_vol"): (
        2.20e-10,
        9.07e-09,
        1.61e-09,
        5.27e-09,
        1.01e-09,
    ),
}
"""Measured absolute recovery errors, in `(v0, kappa, theta, xi, rho)` order.

Recorded in the module rather than only in the rows' notes so that a reader can
see the whole table at once. The rows below do not assert *these* numbers -- a
per-parameter tolerance read off a single run would be cherry-picking -- they
assert the Jacobian-derived bound and carry these as the notes."""

_RECOVERY_SPECS = {
    ("reference", "price"): CalibrationSpec(
        parameters=HESTON_CALIBRATION_REFERENCE,
        strikes=HESTON_CALIBRATION_STRIKES,
        maturities=HESTON_CALIBRATION_SIX_MATURITIES,
        objective="price",
        start=(0.08, 2.0, 0.15, 0.6, -0.2),
    ),
    ("reference", "implied_vol"): CalibrationSpec(
        parameters=HESTON_CALIBRATION_REFERENCE,
        strikes=HESTON_CALIBRATION_STRIKES,
        maturities=HESTON_CALIBRATION_SIX_MATURITIES,
        objective="implied_vol",
        start=(0.08, 2.0, 0.15, 0.6, -0.2),
    ),
    ("feller_violated", "price"): CalibrationSpec(
        parameters=HESTON_CALIBRATION_FELLER_VIOLATED,
        strikes=HESTON_CALIBRATION_STRIKES,
        maturities=HESTON_CALIBRATION_SIX_MATURITIES,
        objective="price",
        start=(0.08, 0.9, 0.09, 0.6, -0.6),
    ),
    ("feller_violated", "implied_vol"): CalibrationSpec(
        parameters=HESTON_CALIBRATION_FELLER_VIOLATED,
        strikes=HESTON_CALIBRATION_STRIKES,
        maturities=HESTON_CALIBRATION_SIX_MATURITIES,
        objective="implied_vol",
        start=(0.08, 0.9, 0.09, 0.6, -0.6),
    ),
}

_PARAMETER_NAMES = ("v0", "kappa", "theta", "xi", "rho")

HESTON_CALIBRATION_RECOVERY_CASES: tuple[CalibrationCase, ...] = tuple(
    CalibrationCase(
        row=BenchmarkRow(
            id=f"heston_cal_recovery_{label}_{objective}_{name}",
            description=(
                f"{name} recovered from a perturbed start on a "
                f"{len(HESTON_CALIBRATION_STRIKES)}x"
                f"{len(HESTON_CALIBRATION_SIX_MATURITIES)} grid "
                f"({label} set, {objective} objective)"
            ),
            expected=0.0,
            tolerance=math.inf,
            evidence=EvidenceClass.CLOSED_FORM,
            source="derived in-repo: tests/test_heston_calibration.py",
            notes=(
                f"Measured absolute error {_RECOVERY_MEASURED[(label, objective)][index]:.2e}. "
                "The asserted tolerance is not this number: it is "
                "HESTON_CALIBRATION_SAFETY_FACTOR * ||r|| / s_min from the fit's own "
                "Jacobian, which is why `tolerance` here is infinite -- the row cannot "
                "carry a bound that has to be computed from the result. Recovery this "
                "close is NOT evidence of identifiability: the quotes are exact to "
                "1e-12 and the errors are the solver's tolerances. See the "
                "single-maturity rows for what identifiability costs."
            ),
        ),
        spec=spec,
        parameter=name,
    )
    for (label, objective), spec in _RECOVERY_SPECS.items()
    for index, name in enumerate(_PARAMETER_NAMES)
)


# --------------------------------------------------------------------------
# (2) Conditioning.
# --------------------------------------------------------------------------

_CONDITION_MEASURED = {
    ("implied_vol", 1): 6.7137e07,
    ("implied_vol", 3): 5.6614e02,
    ("implied_vol", 6): 4.7754e02,
    ("price", 1): 6.4804e07,
    ("price", 3): 7.8250e02,
    ("price", 6): 9.5237e02,
}
"""Condition number of the residual Jacobian **at the true parameters**, on
five strikes, as maturities are added.

Read the two columns against each other. Under the implied-volatility objective
the number falls monotonically, which is the expected result: a term structure
is what separates `kappa` from `xi`. Under the price objective it falls by five
decimal orders from one maturity to three and then rises 22% going to six --
the slice statement asserted the ordering without saying which objective, and
for prices it is false. The reason is an identity: a vega-weighted price
Jacobian *is* the implied-volatility Jacobian (agreement 2.2e-16), so an
unweighted price objective over-weights the long-dated at-the-money cells,
which inflates the largest singular value faster than the smallest."""

_MATURITY_SETS = {
    1: HESTON_CALIBRATION_ONE_MATURITY,
    3: HESTON_CALIBRATION_THREE_MATURITIES,
    6: HESTON_CALIBRATION_SIX_MATURITIES,
}

HESTON_CALIBRATION_CONDITION_CASES: tuple[CalibrationCase, ...] = tuple(
    CalibrationCase(
        row=BenchmarkRow(
            id=f"heston_cal_condition_{objective}_{count}mat",
            description=(
                f"condition number of the {objective} Jacobian at the true "
                f"parameters, {count} maturit{'y' if count == 1 else 'ies'}"
            ),
            expected=value,
            tolerance=HESTON_CALIBRATION_CONDITION_TOLERANCE * value,
            evidence=EvidenceClass.CLOSED_FORM,
            source="derived in-repo: tests/test_heston_calibration.py",
            notes=(
                "Deterministic linear algebra on a Jacobian evaluated at the known "
                "parameters, so the tolerance is a round-off budget and not a "
                "statement about a fit. Measured at the TRUE parameters on purpose: "
                "a condition number read at a fitted point also carries the fit's "
                "own error, and this is a claim about the experiment design."
            ),
        ),
        spec=CalibrationSpec(
            parameters=HESTON_CALIBRATION_REFERENCE,
            strikes=HESTON_CALIBRATION_STRIKES,
            maturities=_MATURITY_SETS[count],
            objective=objective,
        ),
    )
    for (objective, count), value in _CONDITION_MEASURED.items()
)


# --------------------------------------------------------------------------
# (3) The single-maturity degeneracy.
# --------------------------------------------------------------------------

HESTON_CALIBRATION_SINGLE_MATURITY_STARTS = (
    (0.04, 0.5, 0.25, 0.30, -0.5),
    (0.04, 1.0, 0.25, 0.45, -0.5),
    (0.04, 2.0, 0.25, 0.70, -0.5),
    (0.04, 8.0, 0.25, 1.40, -0.5),
    (0.04, 12.0, 0.25, 1.80, -0.5),
    (0.05, 6.0, 0.10, 1.20, -0.4),
)
"""Six starts that differ only in where they enter the `kappa`-`xi` valley."""

_SINGLE_MATURITY_SPEC = CalibrationSpec(
    parameters=HESTON_CALIBRATION_REFERENCE,
    strikes=HESTON_CALIBRATION_STRIKES,
    maturities=HESTON_CALIBRATION_ONE_MATURITY,
    objective="implied_vol",
)

HESTON_CALIBRATION_SINGLE_MATURITY_CASES: tuple[CalibrationCase, ...] = (
    CalibrationCase(
        row=BenchmarkRow(
            id="heston_cal_one_maturity_smile_is_recovered",
            description=(
                "worst implied-volatility RMSE over six starts on a single-maturity "
                "smile"
            ),
            expected=0.0,
            tolerance=1e-05,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source="derived in-repo: tests/test_heston_calibration.py",
            notes=(
                "Measured 2.88e-06 -- three hundredths of a basis point. Every one of "
                "the six fits reproduces the smile. This row exists to be read next "
                "to the two below: the fit is perfect and the parameters are not."
            ),
        ),
        spec=_SINGLE_MATURITY_SPEC,
    ),
    CalibrationCase(
        row=BenchmarkRow(
            id="heston_cal_one_maturity_kappa_is_not_identified",
            description="ratio of largest to smallest fitted kappa over the same six starts",
            expected=2.50,
            tolerance=0.30,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source="derived in-repo: tests/test_heston_calibration.py",
            notes=(
                "Fitted kappa ranges 2.918 to 7.288 against a true 4.0, and two of the "
                "six drive v0 to its lower bound. The tolerance is wide because the "
                "ratio is a property of a nearly-flat valley and the solver's stopping "
                "point inside it moves with the arithmetic; the claim is the factor, "
                "not its third digit. Same objective value to within one decimal order "
                "(1.0e-12 to 2.1e-11) across all six."
            ),
        ),
        spec=_SINGLE_MATURITY_SPEC,
        parameter="kappa",
    ),
    CalibrationCase(
        row=BenchmarkRow(
            id="heston_cal_one_maturity_xi_is_not_identified",
            description="ratio of largest to smallest fitted xi over the same six starts",
            expected=1.95,
            tolerance=0.30,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source="derived in-repo: tests/test_heston_calibration.py",
            notes=(
                "Fitted xi ranges 0.806 to 1.573 against a true 1.0. kappa and xi move "
                "together: the flat right singular vector at one maturity is "
                "kappa -0.926, v0 +0.276, xi -0.247, theta -0.075, rho +0.001."
            ),
        ),
        spec=_SINGLE_MATURITY_SPEC,
        parameter="xi",
    ),
    CalibrationCase(
        row=BenchmarkRow(
            id="heston_cal_one_maturity_rho_is_identified",
            description="worst absolute rho error over the same six starts",
            expected=0.0,
            tolerance=5e-03,
            evidence=EvidenceClass.CLOSED_FORM,
            source="derived in-repo: tests/test_heston_calibration.py",
            notes=(
                "rho lands in [-0.5012, -0.4961] against a true -0.5 from every start. "
                "It is the smile's slope and one smile determines it, which is what "
                "makes the kappa/xi rows a statement about the term structure rather "
                "than about calibration in general. theta is in between: [0.2302, "
                "0.2494] against 0.25."
            ),
        ),
        spec=_SINGLE_MATURITY_SPEC,
        parameter="rho",
    ),
)


# --------------------------------------------------------------------------
# (4) Noise and the Jacobian error bars.
# --------------------------------------------------------------------------

_STDERR_RATIO_MEASURED = {
    5.0: (1.07, 1.08, 0.99, 1.14, 1.16),
    20.0: (1.09, 1.10, 1.00, 1.15, 1.23),
}
_STDERR_AT_20BP = (3.791e-03, 1.896e-01, 1.622e-03, 8.562e-02, 3.295e-02)
"""Mean Jacobian standard errors at 20 bp of implied-volatility noise, in
parameter order. On true values `(0.04, 4.0, 0.25, 1.0, -0.5)` that is 9.5% on
`v0`, 4.7% on `kappa`, 0.65% on `theta`, 8.6% on `xi` and 6.6% on `rho`."""

HESTON_CALIBRATION_STDERR_BAND = 0.30
"""How far the empirical-to-predicted ratio may sit from 1.

Measured worst 1.23 (rho at 20 bp) and best 0.99 (theta at 5 bp). The
predicted errors are consistently on the **small** side, which is the expected
direction: the Gauss-Newton covariance drops the second-order term and the
model is not linear in `kappa`."""

HESTON_CALIBRATION_NOISE_CASES: tuple[CalibrationCase, ...] = tuple(
    CalibrationCase(
        row=BenchmarkRow(
            id=f"heston_cal_stderr_ratio_{int(noise_bp)}bp_{name}",
            description=(
                f"empirical spread of fitted {name} over "
                f"{HESTON_CALIBRATION_NOISE_SEEDS} noise draws, divided by the mean "
                f"Jacobian standard error, at {noise_bp:.0f} bp"
            ),
            expected=1.0,
            tolerance=HESTON_CALIBRATION_STDERR_BAND,
            evidence=EvidenceClass.STATISTICAL,
            source="derived in-repo: tests/test_heston_calibration.py",
            notes=(
                f"Measured {ratios[index]:.2f}. Two independent computations of the "
                "same quantity: the spread ACROSS draws against the error bar computed "
                "WITHIN one draw. Their agreement is what licenses quoting "
                "CalibrationResult.standard_errors at all. At 20 bp the standard error "
                f"on {name} is {_STDERR_AT_20BP[index]:.3e}."
            ),
        ),
        spec=CalibrationSpec(
            parameters=HESTON_CALIBRATION_REFERENCE,
            strikes=HESTON_CALIBRATION_STRIKES,
            maturities=HESTON_CALIBRATION_SIX_MATURITIES,
            objective="implied_vol",
            noise_bp=noise_bp,
        ),
        parameter=name,
    )
    for noise_bp, ratios in _STDERR_RATIO_MEASURED.items()
    for index, name in enumerate(_PARAMETER_NAMES)
)


# --------------------------------------------------------------------------
# (5) The objective choice.
# --------------------------------------------------------------------------

_OBJECTIVE_MEASURED = {
    "price": (0.002029, 0.079817),
    "vega_price": (0.001908, 0.081344),
    "implied_vol": (0.001908, 0.081348),
}
"""`(implied-volatility RMSE, price RMSE)` of each fit, averaged over
`HESTON_CALIBRATION_OBJECTIVE_SEEDS` draws at 20 bp on the 5-by-6 grid."""

HESTON_CALIBRATION_OBJECTIVE_TOLERANCE = 0.10
"""Relative budget on the RMSE rows. The seed-to-seed drift of these means over
10, 12 and 16 draws is about 4%, so a 10% budget pins the ordering (which is
stable) without pinning a mean that is not."""

HESTON_CALIBRATION_OBJECTIVE_CASES: tuple[CalibrationCase, ...] = tuple(
    CalibrationCase(
        row=BenchmarkRow(
            id=f"heston_cal_objective_{name}_{metric}_rmse",
            description=(
                f"{metric} RMSE of the fit obtained by minimising the {name} objective"
            ),
            expected=value,
            tolerance=HESTON_CALIBRATION_OBJECTIVE_TOLERANCE * value,
            evidence=EvidenceClass.STATISTICAL,
            source=(
                "derived in-repo: tests/test_heston_calibration.py; claim provenance "
                "Gatheral (2006), The Volatility Surface, ch. 3"
            ),
            notes=(
                "Each objective wins on its own metric: the price fit is 1.9% better "
                "in price RMSE and 6.3% worse in implied-volatility RMSE, and that "
                "ordering holds on every seed set and grid tried. Vega-weighted prices "
                "and implied volatilities agree to four significant figures on both "
                "metrics, which is the exact Jacobian identity surviving the noise."
            ),
        ),
        spec=CalibrationSpec(
            parameters=HESTON_CALIBRATION_REFERENCE,
            strikes=HESTON_CALIBRATION_STRIKES,
            maturities=HESTON_CALIBRATION_SIX_MATURITIES,
            objective="price" if name != "implied_vol" else "implied_vol",
            noise_bp=20.0,
        ),
    )
    for name, values in _OBJECTIVE_MEASURED.items()
    for metric, value in zip(("implied_vol", "price"), values, strict=True)
) + (
    CalibrationCase(
        row=BenchmarkRow(
            id="heston_cal_objective_does_not_separate_the_parameters",
            description=(
                "spread of the price objective's mean parameter error relative to the "
                "implied-volatility objective's, across the five parameters"
            ),
            expected=1.0,
            tolerance=0.80,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source="derived in-repo: tests/test_heston_calibration.py",
            notes=(
                "The written plan for this slice presumed the fit metric decides the "
                "objective. It does not decide the parameters. Measured ratios of mean "
                "|error|, price over implied_vol: 1.28/1.50/1.07/0.87/0.98 over 10 "
                "draws, 1.16/1.32/1.06/1.06/1.39 over 12, 1.14/1.22/0.99/0.94/1.22 "
                "over 16 -- straddling 1 with no consistent sign. At 20 bp on this "
                "grid the objective choice shows up in the CONDITIONING (rows above) "
                "and not in the answer. The wide tolerance is the claim: every ratio "
                "sits inside [0.87, 1.63] rather than on one side of 1."
            ),
        ),
        spec=CalibrationSpec(
            parameters=HESTON_CALIBRATION_REFERENCE,
            strikes=HESTON_CALIBRATION_STRIKES,
            maturities=HESTON_CALIBRATION_SIX_MATURITIES,
            objective="implied_vol",
            noise_bp=20.0,
        ),
    ),
)


# --------------------------------------------------------------------------
# (6) Initialisation.
# --------------------------------------------------------------------------

HESTON_CALIBRATION_GLOBAL_COUNT = 11
HESTON_CALIBRATION_GLOBAL_TOLERANCE = 1e-02
"""Relative slack on the best objective for a start to count as having reached
it. The eleven winners agree to 1e-06 relative and the three failures are four
to seven decimal orders away, so any tolerance between 1e-06 and 1e+02 gives
the same count."""

_INITIALISATION_SPEC = CalibrationSpec(
    parameters=HESTON_CALIBRATION_REFERENCE,
    strikes=HESTON_CALIBRATION_STRIKES,
    maturities=HESTON_CALIBRATION_SIX_MATURITIES,
    objective="implied_vol",
    noise_bp=20.0,
    seed=3000,
)

HESTON_CALIBRATION_INITIALISATION_CASES: tuple[CalibrationCase, ...] = (
    CalibrationCase(
        row=BenchmarkRow(
            id="heston_cal_global_basin_fraction",
            description=(
                f"starts out of {len(HESTON_CALIBRATION_START_GRID)} that reach the "
                "best objective found"
            ),
            expected=float(HESTON_CALIBRATION_GLOBAL_COUNT),
            tolerance=0.0,
            evidence=EvidenceClass.NEGATIVE_FINDING,
            source="derived in-repo: tests/test_heston_calibration.py",
            notes=(
                "11/14, so the failure rate is 21% from a deliberately spread grid. "
                "The three failures -- (0.10,0.3,0.10,1.8,+0.8), "
                "(0.50,0.1,0.80,3.0,-0.99) and (0.04,0.2,0.60,4.0,-0.3) -- stop within "
                "1e-02 of where they started, and the cause is measured rather than "
                "assumed: all three have a huge theta or xi, which is where the "
                "COS_LOG_RANGE_CAP clip binds hard, so the pricer is wrong by three "
                "decimal orders and the residual is flat. They are not local minima of "
                "the true objective. Before the clip existed the same grid scored 5/14 "
                "and the failures returned their starting vector exactly. Tolerance 0: "
                "this is a count, and a change in it is a change in the finding."
            ),
        ),
        spec=_INITIALISATION_SPEC,
    ),
    CalibrationCase(
        row=BenchmarkRow(
            id="heston_cal_multistart_recovers_from_a_dead_start",
            description=(
                f"relative objective gap between an {HESTON_CALIBRATION_MULTISTART_COUNT}"
                "-start run from a dead start and the best single-start run"
            ),
            expected=0.0,
            tolerance=1e-03,
            evidence=EvidenceClass.CLOSED_FORM,
            source="derived in-repo: tests/test_heston_calibration.py",
            notes=(
                "Started at (0.50, 0.1, 0.80, 3.0, -0.99), one of the three stalls, "
                "n_starts=6 reaches 3.116984e-05, the same objective the eleven good "
                "starts reach. n_starts of 4, 8, 12 and 16 reach the same six digits at "
                "1.5, 3.4, 5.4 and 7.2 seconds, so 6 is a run-time choice and not a "
                "tuned one. The draw is reproducible at a fixed seed."
            ),
        ),
        spec=_INITIALISATION_SPEC,
    ),
)


ALL_HESTON_CALIBRATION_CASES: tuple[CalibrationCase, ...] = (
    HESTON_CALIBRATION_RECOVERY_CASES
    + HESTON_CALIBRATION_CONDITION_CASES
    + HESTON_CALIBRATION_SINGLE_MATURITY_CASES
    + HESTON_CALIBRATION_NOISE_CASES
    + HESTON_CALIBRATION_OBJECTIVE_CASES
    + HESTON_CALIBRATION_INITIALISATION_CASES
)
"""All 49 rows, in family order: 20 recovery, 6 conditioning, 4
single-maturity, 10 noise, 7 objective and 2 initialisation."""
