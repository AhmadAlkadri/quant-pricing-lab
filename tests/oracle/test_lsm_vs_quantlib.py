"""This package's Longstaff-Schwartz engine against QuantLib's.

Slice 11 item (h). Unlike the tree and finite-difference oracles, this one
compares **the same method** implemented twice, so the comparison is not
"two discretisations agree" but "two implementations of one estimator agree
within the noise they both report". That is a weaker claim about the model and
a stronger one about the code: a sign error in the discounting, an
in-the-money filter applied to the wrong array, or a policy accidentally
applied to its own training sample would all survive a lattice cross-check at
this precision and none of them survives this one.

What QuantLib's engine is, in its own settings
----------------------------------------------
`ql.MCAmericanEngine(process, "pseudorandom", timeSteps=m,
antitheticVariate=True, requiredSamples=N, seed=s, polynomOrder=p,
polynomType=t, nCalibrationSamples=c)`:

- `timeSteps=m` is the exercise grid: `m` equally spaced dates on `(0, T]`,
  the same convention `qpl.engines.mc.american` uses.
- `polynomType` selects the basis family from `ql.LsmBasisSystem`. Two are
  used below: `Monomial` (`1, x, ..., x^p` in the **raw** underlying value) and
  `Laguerre`. `polynomOrder=p` gives `p + 1` functions including the constant,
  which is the same counting as `MCConfig.lsm_degree`.
- `nCalibrationSamples=c` is a separate calibration pass: the policy is fitted
  on `c` samples and valued on `requiredSamples`, so QuantLib's default
  estimator is the **out-of-sample**, low-biased one -- the same choice
  `MCConfig.lsm_in_sample=False` makes.
- With `antitheticVariate=True`, `requiredSamples` counts antithetic *pairs*,
  so `requiredSamples=50 000` is 100 000 paths. That is why the settings below
  put 50 000 there against 100 000 in `MCConfig.n_paths`, and why the two
  engines report standard errors within 1% of each other.

Two differences that are deliberate and not defects in either. QuantLib's
monomials are in the raw underlying, where this package scales by the strike
(`qpl.engines.mc.american.basis_matrix`); at `S ~ 36, K = 40` that is a
conditioning difference and not a modelling one, and it is why this package's
Laguerre basis needs the scaling and QuantLib's does not use the same weight.
And the two draw different pseudo-random streams, so nothing here can agree
better than the standard errors allow -- which is the point.

Measured
--------
Longstaff-Schwartz row 1 (`S = 36, K = 40, T = 1, r = 6%, q = 0,
sigma = 20%`), 50 exercise dates, order 3, 100 000 paths each, out of sample,
against the 50-date lattice Bermudan 4.477922:

    engine                     value      stderr     bias
    QuantLib, Monomial       4.473001   6.091e-03  -4.92e-03
    QuantLib, Laguerre       4.470848   6.051e-03  -7.07e-03
    qpl, laguerre degree 3   4.482564   6.052e-03  +4.64e-03

`z = (QuantLib - qpl) / sqrt(se_ql^2 + se_qpl^2)` is **-1.11** for the Monomial
run. QuantLib's own two basis families differ by 2.15e-03, or 0.36 of a
standard error, which is the same "the family does not matter, the count does"
result `tests/test_lsm_american_bias.py` measures inside this package.

Both engines also reproduce the published 4.472 comfortably: QuantLib's
Monomial run is 1.0e-03 away, 0.16 standard errors.

Evidence classes: INDEPENDENT_ENGINE for the agreement; STATISTICAL for the
shared low bias below.

Day count: `Actual365Fixed` over 365 days makes `T = 1.0` exactly, the same
convention as the other oracle modules, so there is no day-count residual.
"""

from __future__ import annotations

import math

import pytest

ql = pytest.importorskip("QuantLib")

from qpl.cases import (  # noqa: E402
    AMERICAN_REFERENCE_SPEC,
    LS2001_BERMUDAN_EXERCISES_PER_YEAR,
    LS2001_LSM_PATHS,
    LS2001_PUBLISHED_LSM_STDERR,
    LS2001_PUBLISHED_LSM_VALUE,
    LS2001_ROW1,
    LSM_DEGREE,
    LSM_LATTICE_N_STEPS,
    LSM_STDERR_MULTIPLE,
    AmericanBSSpec,
    bermudan_value_on_lattice,
)
from qpl.engines.mc.pricers import MCConfig  # noqa: E402
from qpl.pricing import price  # noqa: E402

_EVALUATION_DATE = ql.Date(15, 1, 2024)
_DAY_COUNT = ql.Actual365Fixed()
_CALENDAR = ql.NullCalendar()
_DAYS = 365

QL_SEED = 42
QL_SAMPLES = LS2001_LSM_PATHS // 2
"""`requiredSamples` counts antithetic pairs, so this is `LS2001_LSM_PATHS`
paths -- the same sample size this package's `n_paths` asks for."""

QPL_SEED = 7

ATM_DATES = 50
"""Exercise frequency for the shared-low-bias test.

Fifty rather than the 250 the cases layer uses for the three-engine row:
QuantLib's engine costs 8.5 s at 250 dates against 1.7 s at 50, and the effect
being measured -- that both engines sit below the lattice Bermudan -- is
present at both frequencies (measured bias at 250 dates: QuantLib
-9.01e-03 +- 3.50e-03 over five seeds, this package -1.88e-02 +- 3.96e-03 over
ten)."""


def _quantlib_american(spec: AmericanBSSpec):
    ql.Settings.instance().evaluationDate = _EVALUATION_DATE
    maturity = _EVALUATION_DATE + ql.Period(_DAYS, ql.Days)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spec.spot)),
        ql.YieldTermStructureHandle(
            ql.FlatForward(_EVALUATION_DATE, spec.dividend, _DAY_COUNT)
        ),
        ql.YieldTermStructureHandle(
            ql.FlatForward(_EVALUATION_DATE, spec.rate, _DAY_COUNT)
        ),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(_EVALUATION_DATE, _CALENDAR, spec.sigma, _DAY_COUNT)
        ),
    )
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if spec.kind == "call" else ql.Option.Put, spec.strike
    )
    option = ql.VanillaOption(
        payoff, ql.AmericanExercise(_EVALUATION_DATE, maturity)
    )
    return option, process


def _quantlib_lsm(
    spec: AmericanBSSpec, *, n_exercise: int, polynom_type, seed: int = QL_SEED
) -> tuple[float, float]:
    """`(value, stderr)` from `MCAmericanEngine` at the settings above."""
    option, process = _quantlib_american(spec)
    option.setPricingEngine(
        ql.MCAmericanEngine(
            process,
            "pseudorandom",
            timeSteps=n_exercise,
            antitheticVariate=True,
            requiredSamples=QL_SAMPLES,
            seed=seed,
            polynomOrder=LSM_DEGREE,
            polynomType=polynom_type,
            nCalibrationSamples=QL_SAMPLES,
        )
    )
    return float(option.NPV()), float(option.errorEstimate())


def _qpl_lsm(spec: AmericanBSSpec, *, n_exercise: int) -> tuple[float, float]:
    result = price(
        spec.option(),
        spec.model(),
        spec.market(),
        method="mc",
        cfg=MCConfig(
            n_paths=LS2001_LSM_PATHS,
            seed=QPL_SEED,
            variance_reduction="antithetic",
            exercise_dates=n_exercise,
            lsm_degree=LSM_DEGREE,
        ),
    )
    return result.value, result.stderr


@pytest.fixture(scope="module")
def ls_row_prices() -> dict[str, tuple[float, float]]:
    """The Longstaff-Schwartz row from three engines. About 4.6 s."""
    return {
        "ql_monomial": _quantlib_lsm(
            LS2001_ROW1,
            n_exercise=LS2001_BERMUDAN_EXERCISES_PER_YEAR,
            polynom_type=ql.LsmBasisSystem.Monomial,
        ),
        "ql_laguerre": _quantlib_lsm(
            LS2001_ROW1,
            n_exercise=LS2001_BERMUDAN_EXERCISES_PER_YEAR,
            polynom_type=ql.LsmBasisSystem.Laguerre,
        ),
        "qpl": _qpl_lsm(
            LS2001_ROW1, n_exercise=LS2001_BERMUDAN_EXERCISES_PER_YEAR
        ),
    }


def _z_score(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) / math.hypot(a[1], b[1])


def test_day_count_makes_the_expiry_exact() -> None:
    """`T = 365 / 365` is exactly 1.0, so there is no day-count residual."""
    ql.Settings.instance().evaluationDate = _EVALUATION_DATE
    maturity = _EVALUATION_DATE + ql.Period(_DAYS, ql.Days)
    assert _DAY_COUNT.yearFraction(_EVALUATION_DATE, maturity) == LS2001_ROW1.expiry


@pytest.mark.slow
def test_quantlib_lsm_agrees_on_the_longstaff_schwartz_row(
    ls_row_prices: dict[str, tuple[float, float]],
) -> None:
    """Evidence class: INDEPENDENT_ENGINE.

    Two independent implementations of the same estimator, at the same sample
    size, exercise frequency and number of basis functions, on different
    pseudo-random streams. The budget is `LSM_STDERR_MULTIPLE` times the
    root-sum-square of the two reported standard errors -- which is the only
    tolerance that means anything here, since neither number is "the" answer.

    Measured `z = (QuantLib - qpl) / sqrt(se_ql^2 + se_qpl^2)`: **-1.11** for
    QuantLib's Monomial basis, and the two standard errors agree to 0.6%
    (6.091e-03 against 6.052e-03), which is the check that `requiredSamples`
    really does count antithetic pairs and that the two runs are comparing
    equal amounts of information.

    See the module docstring for QuantLib's settings; nothing is defaulted
    silently.
    """
    quantlib = ls_row_prices["ql_monomial"]
    ours = ls_row_prices["qpl"]

    assert abs(_z_score(quantlib, ours)) < LSM_STDERR_MULTIPLE, (quantlib, ours)
    assert quantlib[1] == pytest.approx(ours[1], rel=0.05), (quantlib[1], ours[1])


def test_quantlib_lsm_also_reproduces_the_published_value(
    ls_row_prices: dict[str, tuple[float, float]],
) -> None:
    """Evidence class: PUBLISHED_BENCHMARK, through someone else's code.

    The point of this one is triangulation rather than accuracy: the published
    4.472 is checked here by an implementation that is neither the paper's nor
    this repository's. Measured 4.473001 with standard error 6.091e-03, which
    is 1.0e-03 away -- 0.16 standard errors, and inside the 0.010 the table
    itself prints.

    Note the estimator mismatch, which is why the budget is three standard
    errors and not one: the table's value is the paper's *in-sample* estimate
    (biased high) while QuantLib's is out of sample (biased low), so the two
    are estimating the same Bermudan value with opposite-signed biases. At
    100 000 paths both biases are small enough that the difference is noise,
    which is itself the useful statement.
    """
    value, stderr = ls_row_prices["ql_monomial"]
    assert abs(value - LS2001_PUBLISHED_LSM_VALUE) < LSM_STDERR_MULTIPLE * stderr
    assert stderr <= LS2001_PUBLISHED_LSM_STDERR


def test_quantlibs_own_basis_choice_moves_it_less_than_its_noise(
    ls_row_prices: dict[str, tuple[float, float]],
) -> None:
    """Evidence class: NEGATIVE_FINDING, confirmed outside this repository.

    `tests/test_lsm_american_bias.py` measures, inside this package, that
    swapping the basis *family* at fixed size is not resolvable while changing
    the *number* of functions is. QuantLib says the same thing on the same
    point with its own two families: Monomial 4.473001, Laguerre 4.470848, a
    difference of 2.15e-03 against a standard error of 6.07e-03 -- **0.36** of
    one, on runs that share a seed and therefore share much of their noise.

    Pinned because it is the kind of knob that invites tuning. If a future
    change makes the family matter here, something else has changed with it.
    """
    monomial = ls_row_prices["ql_monomial"]
    laguerre = ls_row_prices["ql_laguerre"]
    assert abs(monomial[0] - laguerre[0]) < monomial[1], (monomial, laguerre)


def test_both_engines_sit_below_the_lattice_bermudan_at_the_money() -> None:
    """Evidence class: STATISTICAL -- the low bias is the method's, not ours.

    The three-engine row in `qpl.cases` carries a `finite_sample_low_bias`
    term of 1.89e-02 at the ATM point, which is larger than both the Bermudan
    gap and the standard error and is the term the slice statement did not
    anticipate. If it were an artefact of this implementation, an independent
    one would not show it.

    It does. At 50 exercise dates and 100 000 paths, against the lattice
    Bermudan 6.078525: QuantLib -6.22e-03, this package -1.05e-02 +- 2.55e-03
    (mean over ten seeds; the single-seed run asserted below draws -1.28e-02).
    At 250
    dates, where the effect is larger and the runs cost 8.5 s each, the
    measured means are QuantLib **-9.01e-03 +- 3.50e-03** over five seeds and
    this package **-1.88e-02 +- 3.96e-03** over ten.

    What is asserted: both are below the lattice value, and the gap between the
    two engines is inside the noise. What is deliberately **not** asserted:
    that the two biases are equal. They differ by 9.8e-03 +- 5.3e-03 at 250
    dates, 1.9 standard errors, so this comparison confirms the sign and the
    order of magnitude and does not resolve the size. Matching them would need
    equal calibration samples path for path and many more seeds than this file
    can afford, and the answer would be a statement about two calibration
    schemes rather than about the method.
    """
    spec = AMERICAN_REFERENCE_SPEC
    lattice = bermudan_value_on_lattice(
        spec, n_steps=LSM_LATTICE_N_STEPS, n_exercise=ATM_DATES
    )
    quantlib = _quantlib_lsm(
        spec, n_exercise=ATM_DATES, polynom_type=ql.LsmBasisSystem.Monomial
    )
    ours = _qpl_lsm(spec, n_exercise=ATM_DATES)

    assert quantlib[0] < lattice, (quantlib, lattice)
    assert ours[0] < lattice, (ours, lattice)
    assert abs(_z_score(quantlib, ours)) < LSM_STDERR_MULTIPLE, (quantlib, ours)
