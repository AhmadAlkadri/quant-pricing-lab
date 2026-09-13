"""What the Kemna-Vorst control variate is actually worth, measured.

`tests/test_asian_mc.py` establishes that the Asian estimators are unbiased and
report calibrated standard errors. This file answers the other question: how
much variance does the geometric control remove, and does it remove what the
theory says it should?

The cost axis is **normal draws**, not paths -- `n_paths * n_fixings` here,
since an Asian path consumes one normal per fixing. `ASIAN_VR_DRAWS = 41 600`
is divisible by both 10 and 52, so the ten-fixing and fifty-two-fixing columns
are run at exactly the same cost and can be compared. Antithetic sampling
evaluates the payoff twice per draw and therefore runs twice the paths.

The variance compared is the variance of the *estimator*: fifty independent
seeds, one price each, sample variance of those fifty numbers. Each such
variance is a 49-degree-of-freedom estimate with 20% relative standard
deviation, so the assertion band is a factor of 1.7 either way -- the same band
and the same derivation as Slice 7 (`qpl.cases.mc_variance_reduction`).

Predictions are computed from correlations measured on an **independent pilot**
(40 000 paths, seeds 7 and 11), not hard-coded, so each row compares a measured
ratio against a measured prediction of it rather than restating this file.

Reference: Kemna & Vorst (1990), *Journal of Banking and Finance* 14, 113-129,
who propose the geometric average as a control for the arithmetic one;
Glasserman (2003), sections 4.1 and 4.2, for the two equal-cost variance
ratios. Every number below was measured in this repository.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.engines.analytic.asian import discrete_geometric_price
from qpl.engines.mc.asian import asian_terminal_sample
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.mc.processes import gbm_paths_from_normals
from qpl.instruments import (
    AsianOption,
    arithmetic_average,
    asian_payoff,
    uniform_fixing_times,
)
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price
from qpl.validation import EvidenceClass

SPOT, STRIKE, EXPIRY, SIGMA = 100.0, 100.0, 1.0, 0.20
MARKET = Market(
    spot=SPOT, rate_curve=FlatRateCurve(0.05), dividend_curve=FlatDividendCurve(0.0)
)
MODEL = BlackScholesModel(sigma=SIGMA)
MU = MARKET.rate(EXPIRY) - MARKET.dividend_yield(EXPIRY)
DISCOUNT = MARKET.df_r(EXPIRY)

ASIAN_VR_SEEDS = tuple(range(101, 151))
ASIAN_VR_DRAWS = 41_600
"""`4160 * 10 == 800 * 52`: both fixing counts spend the same number of normals,
which is the only way the two columns of the table below mean the same thing."""

ASIAN_VR_BAND = 1.7
"""Multiplicative tolerance, derived rather than chosen: a ratio of two
independent 49-degree-of-freedom variance estimates has a 99% band of
[0.47, 2.11], and the measured-over-predicted ratios below run 0.89 to 1.52, so
1.7 sits inside the sampling band and about 3x outside the observed spread."""

PILOT_PATHS = 40_000
PILOT_SEED = 7
ANTITHETIC_PILOT_SEED = 11

FIXING_COUNTS = (10, 52)


def _option(n_fixings: int, strike: float = STRIKE) -> AsianOption:
    return AsianOption(
        "call", strike, EXPIRY, uniform_fixing_times(EXPIRY, n_fixings), "arithmetic"
    )


def _paths_for(vr, n_fixings: int) -> int:
    base = ASIAN_VR_DRAWS // n_fixings
    return 2 * base if "antithetic" in vr else base


def _estimator_variance(n_fixings: int, vr) -> float:
    option = _option(n_fixings)
    values = [
        price(
            option,
            MODEL,
            MARKET,
            method="mc",
            cfg=MCConfig(
                n_paths=_paths_for(vr, n_fixings), seed=seed, variance_reduction=vr
            ),
        ).value
        for seed in ASIAN_VR_SEEDS
    ]
    return float(np.var(values, ddof=1))


def _pilot_control_correlation(
    n_fixings: int, *, model: BlackScholesModel = MODEL, strike: float = STRIKE
) -> float:
    """`corr(Y, X)` between the arithmetic and geometric discounted payoffs.

    Independent of the fifty measurement seeds, so the prediction it feeds is a
    genuine prediction. Computed on the *engine's own* sample object, because
    the correlation that matters is the one between the exact pair the estimator
    regresses -- rebuilding the paths here would measure a different sample.
    """
    sample = asian_terminal_sample(
        _option(n_fixings, strike),
        model,
        MARKET,
        cfg=MCConfig(
            n_paths=PILOT_PATHS, seed=PILOT_SEED, variance_reduction="control_variate"
        ),
        methods=("control_variate",),
    )
    return float(np.corrcoef(sample.y, sample.x)[0, 1])


def _pilot_pair_control_correlation(n_fixings: int) -> float:
    """The same correlation on **pair-averaged** units, which is what the
    combined estimator actually regresses."""
    sample = asian_terminal_sample(
        _option(n_fixings),
        MODEL,
        MARKET,
        cfg=MCConfig(
            n_paths=2 * PILOT_PATHS,
            seed=PILOT_SEED,
            variance_reduction=("antithetic", "control_variate"),
        ),
        methods=("antithetic", "control_variate"),
    )
    return float(np.corrcoef(sample.y, sample.x)[0, 1])


def _pilot_antithetic_correlation(n_fixings: int) -> float:
    """`corr(Y(Z), Y(-Z))` for the arithmetic-average call payoff."""
    times = np.asarray(uniform_fixing_times(EXPIRY, n_fixings))
    rng = np.random.default_rng(ANTITHETIC_PILOT_SEED)
    z = rng.normal(size=(PILOT_PATHS, n_fixings))

    def payoff(normals: np.ndarray) -> np.ndarray:
        fixings = gbm_paths_from_normals(
            normals, s0=SPOT, mu=MU, sigma=SIGMA, times=times
        )
        return DISCOUNT * np.asarray(
            asian_payoff(arithmetic_average(fixings), STRIKE, "call"), dtype=float
        )

    return float(np.corrcoef(payoff(z), payoff(-z))[0, 1])


# --------------------------------------------------------------------------
# (b) The table.
# --------------------------------------------------------------------------

_CONTROL_ROWS = ((10, 0.999608, 1276.7, 1433.0), (52, 0.999604, 1262.2, 1196.2))


@pytest.mark.parametrize(
    ("n_fixings", "pilot_rho", "predicted", "measured"),
    _CONTROL_ROWS,
    ids=["10_fixings", "52_fixings"],
)
def test_control_variate_factor_matches_one_over_one_minus_rho_squared(
    n_fixings, pilot_rho, predicted, measured
):
    """STATISTICAL: `Var(plain)/Var(control) = 1/(1 - rho^2)`, and rho is ~0.9996.

    This is the row the slice exists for. Kemna & Vorst's observation is that
    the geometric average of a path and its arithmetic average are almost the
    same random variable -- they differ by a term of order `Var(log A)`, which
    is small for the volatilities and maturities Asian options are written at --
    so a control variate built from the geometric payoff, whose mean is known
    exactly, removes almost all of the variance.

    Measured over 50 seeds at 41 600 normal draws, with `rho` from an
    independent 40 000-path pilot (seed 7):

        fixings    rho        1/(1-rho^2)   measured   measured/predicted
        10        0.999608      1276.7       1433.0          1.12
        52        0.999604      1262.2       1196.2          0.95

    Three things worth noticing.

    - **The factor is three orders of magnitude**, against the 7.6 that the
      discounted terminal spot buys on a vanilla ATM call in Slice 7. The
      difference is entirely `rho`: 0.9996 against 0.9246, and `1/(1-rho^2)` is
      brutally sensitive up there.
    - **`rho` barely moves between 10 and 52 fixings** (0.999608 against
      0.999604, a difference well inside the pilot's own noise). The control's
      quality is a property of the *contract*, not of the monitoring frequency:
      both averages see the same path and the gap between them is governed by
      the dispersion of the log-spot over the averaging window, which the fixing
      count changes only through an `O(1/n)` correction.
    - The 52-fixing column is run at 800 paths and still resolves the price to
      four decimals, which is the practical content of a factor of 1200: the
      control variate is worth about `sqrt(1200) = 35x` in *accuracy* at fixed
      cost, or 1200x in cost at fixed accuracy.
    """
    assert EvidenceClass.STATISTICAL
    rho = _pilot_control_correlation(n_fixings)
    assert rho == pytest.approx(pilot_rho, abs=5e-5)
    prediction = 1.0 / (1.0 - rho * rho)
    assert prediction == pytest.approx(predicted, rel=0.05)

    factor = _estimator_variance(n_fixings, "none") / _estimator_variance(
        n_fixings, "control_variate"
    )
    assert factor == pytest.approx(measured, rel=0.05), factor
    assert prediction / ASIAN_VR_BAND < factor < prediction * ASIAN_VR_BAND, (
        f"{n_fixings} fixings: measured {factor:.1f} against predicted "
        f"{prediction:.1f}"
    )


_ANTITHETIC_ROWS = ((10, -0.5202, 4.17, 5.54), (52, -0.5172, 4.14, 6.31))


@pytest.mark.parametrize(
    ("n_fixings", "pilot_rho_a", "predicted", "measured"),
    _ANTITHETIC_ROWS,
    ids=["10_fixings", "52_fixings"],
)
def test_antithetic_factor_matches_two_over_one_plus_rho(
    n_fixings, pilot_rho_a, predicted, measured
):
    """STATISTICAL: the antithetic gain, at equal normal draws.

    Glasserman 4.2: `2 / (1 + rho_a)` with `rho_a = corr(Y(Z), Y(-Z))`.
    Reflecting the whole vector of `n` normals reflects the whole path, and the
    arithmetic average is monotone in each of them, so `rho_a < 0` for the same
    reason it is for a vanilla.

    Pilot at 40 000 paths, seed 11:

        fixings    rho_a     2/(1+rho_a)   measured   measured/predicted
        10        -0.5202       4.17         5.54           1.33
        52        -0.5172       4.14         6.31           1.52

    `rho_a` is essentially the vanilla ATM call's (-0.5009 in Slice 7) and for
    the same reason -- an Asian ATM call is a monotone, nearly-affine-above-the-
    strike function of a single Gaussian factor once the average is formed. Both
    measured ratios sit above the prediction, as three of Slice 7's six
    antithetic cells did; the plain estimator's realised variance over these
    fifty seeds is above what its own standard errors predict, which inflates
    every ratio carrying it in the numerator.

    Against the control variate's 1200x this is almost nothing, and that is the
    finding: for an Asian, antithetic sampling is not the method to reach for.
    """
    assert EvidenceClass.STATISTICAL
    rho_a = _pilot_antithetic_correlation(n_fixings)
    assert rho_a < 0.0, "a monotone payoff must have a negative antithetic correlation"
    assert rho_a == pytest.approx(pilot_rho_a, abs=1e-3)
    prediction = 2.0 / (1.0 + rho_a)
    assert prediction == pytest.approx(predicted, rel=0.02)

    factor = _estimator_variance(n_fixings, "none") / _estimator_variance(
        n_fixings, "antithetic"
    )
    assert factor == pytest.approx(measured, rel=0.05), factor
    assert prediction / ASIAN_VR_BAND < factor < prediction * ASIAN_VR_BAND, (
        f"{n_fixings} fixings: measured {factor:.2f} against predicted "
        f"{prediction:.2f}"
    )


_COMBINATION_ROWS = ((10, 0.999127, 2387.7, 2113.8, 7933.0), (52, 0.999122, 2360.8, 2584.0, 7543.0))


@pytest.mark.parametrize(
    ("n_fixings", "pilot_rho_pair", "composite", "measured", "naive_product"),
    _COMBINATION_ROWS,
    ids=["10_fixings", "52_fixings"],
)
def test_the_combination_composes_on_the_units_the_estimator_actually_uses(
    n_fixings, pilot_rho_pair, composite, measured, naive_product
):
    """STATISTICAL: it composes -- but only with the *right* second factor.

    Slice 7 recorded that its combinations "do not compose multiplicatively",
    and by the same arithmetic that is true here too: the product of the two
    marginal factors (each measured against the plain estimator) is 7933 at ten
    fixings and 7543 at fifty-two, against measured 2114 and 2584. Off by 3-4x.

    But the marginal product is the wrong prediction, and this row says what the
    right one is. The combined estimator regresses the control on the
    **pair-averaged** units, so the gain it adds is `1/(1 - rho_pair^2)` with
    `rho_pair` measured on those units, and the composite prediction is

        factor(antithetic)  x  1/(1 - rho_pair^2).

    Pilot at 40 000 pairs, seed 7:

        fixings   rho_pair    2/(1+rho_a)   1/(1-rho_pair^2)   composite   measured   ratio
        10        0.999127       4.17            572.8           2387.7     2113.8    0.89
        52        0.999122       4.14            569.9           2360.8     2584.0    1.09

    Within 11% at both fixing counts -- so the failure to compose in Slice 7 was
    a statement about *which correlation was measured*, not about an interaction
    between the methods. Note the direction of the correction:
    `rho_pair < rho` (0.99913 against 0.99961), i.e. the control is a *worse*
    fit after pair-averaging, because pair-averaging has already removed the
    part of the payoff that the geometric control explains best.
    """
    assert EvidenceClass.STATISTICAL
    rho_a = _pilot_antithetic_correlation(n_fixings)
    rho_pair = _pilot_pair_control_correlation(n_fixings)
    assert rho_pair == pytest.approx(pilot_rho_pair, abs=2e-4)
    assert rho_pair < _pilot_control_correlation(n_fixings)

    prediction = (2.0 / (1.0 + rho_a)) * (1.0 / (1.0 - rho_pair * rho_pair))
    assert prediction == pytest.approx(composite, rel=0.10)

    plain = _estimator_variance(n_fixings, "none")
    factor = plain / _estimator_variance(n_fixings, ("antithetic", "control_variate"))
    assert factor == pytest.approx(measured, rel=0.05), factor
    assert prediction / ASIAN_VR_BAND < factor < prediction * ASIAN_VR_BAND

    # And the naive marginal product is wrong, by the recorded amount.
    marginal = (plain / _estimator_variance(n_fixings, "antithetic")) * (
        plain / _estimator_variance(n_fixings, "control_variate")
    )
    assert marginal == pytest.approx(naive_product, rel=0.05)
    assert marginal > 2.5 * factor


@pytest.mark.parametrize("n_fixings", FIXING_COUNTS)
def test_plain_and_control_variate_agree_within_noise(n_fixings):
    """STATISTICAL: the control variate moves the variance, not the mean.

    The estimator is `mean(Y) - b (mean(X) - E[X])` with `b` fitted on the same
    sample, so it is biased at `O(1/N)` -- but the bias is far below the plain
    estimator's own standard error at any usable `N`, which is what this row
    measures. Twenty seeds at 41 600 draws, `z` being the difference divided by
    the root-sum-square of the two reported standard errors (which is
    conservative: the two estimators share the sample and are positively
    correlated, so the true standard deviation of the difference is smaller than
    that, and `|z|` is understated):

        fixings   mean z   sd z    max |z|
        10        +0.119   1.166    2.942
        52        -0.165   1.179    2.327

    Centred on zero with roughly unit spread. If the control's known mean were
    wrong -- the single most likely bug in this engine, since it is a closed
    form evaluated at parameters read off a curve -- every `z` here would be
    displaced by the same amount and the mean would not be near zero.
    """
    assert EvidenceClass.STATISTICAL
    option = _option(n_fixings)
    n_paths = ASIAN_VR_DRAWS // n_fixings
    zs = []
    for seed in range(1, 21):
        plain = price(
            option, MODEL, MARKET, method="mc", cfg=MCConfig(n_paths=n_paths, seed=seed)
        )
        reduced = price(
            option,
            MODEL,
            MARKET,
            method="mc",
            cfg=MCConfig(
                n_paths=n_paths, seed=seed, variance_reduction="control_variate"
            ),
        )
        zs.append(
            (plain.value - reduced.value) / math.hypot(plain.stderr, reduced.stderr)
        )
    zs = np.asarray(zs)
    assert abs(float(zs.mean())) < 0.6, float(zs.mean())
    assert float(np.abs(zs).max()) < 4.0, float(np.abs(zs).max())


def test_the_control_correlation_is_insensitive_to_the_fixing_count():
    """STATISTICAL: `rho` at 10 and 52 fixings differ by 4e-06.

    Stated on its own because it is the practically useful half of the result:
    the Kemna-Vorst control does not have to be retuned when the monitoring
    schedule changes. A daily-fixing Asian and a monthly one get the same three
    orders of magnitude.
    """
    assert EvidenceClass.STATISTICAL
    rho_10 = _pilot_control_correlation(10)
    rho_52 = _pilot_control_correlation(52)
    assert abs(rho_10 - rho_52) < 5e-5, (rho_10, rho_52)
    assert min(rho_10, rho_52) > 0.9995


_DEGRADATION_ROWS = (
    ("vol10_atm", 0.10, 100.0, 0.999891, 4575.3),
    ("vol20_atm", 0.20, 100.0, 0.999604, 1262.2),
    ("vol40_atm", 0.40, 100.0, 0.998382, 309.3),
    ("vol80_atm", 0.80, 100.0, 0.992153, 64.0),
    ("vol20_otm", 0.20, 130.0, 0.992562, 67.5),
    ("vol40_otm", 0.40, 130.0, 0.995199, 104.4),
    ("vol80_otm", 0.80, 130.0, 0.989342, 47.2),
)


@pytest.mark.parametrize(
    ("label", "sigma", "strike", "rho", "predicted"),
    _DEGRADATION_ROWS,
    ids=[row[0] for row in _DEGRADATION_ROWS],
)
def test_where_the_geometric_control_degrades(label, sigma, strike, rho, predicted):
    """STATISTICAL: `rho` falls with total variance, and NOT monotonically in moneyness.

    The control is good because `log A - log G` is small, and its size is
    governed by the dispersion of the log-spot over the averaging window, i.e.
    by `sigma^2 T`. Measured at 52 fixings on a 40 000-path pilot (seed 7):

        sigma   strike    rho        1/(1-rho^2)
        10%      100    0.999891      4575.3
        20%      100    0.999604      1262.2
        40%      100    0.998382       309.3
        80%      100    0.992153        64.0
        20%      130    0.992562        67.5
        40%      130    0.995199       104.4
        80%      130    0.989342        47.2

    At the money the ordering is exactly the prediction: quadrupling the total
    variance divides the factor by about four, over a range of 70x.

    Out of the money it is **not** monotone, and that was not expected: the
    factor is *lowest* at `sigma = 20%` (67.5), rises to 104.4 at 40%, and falls
    again at 80%. The mechanism is that two things move at once. Raising the
    volatility widens `log A - log G`, which hurts; but at `K = 130` it also
    moves the option from "almost never pays" toward "pays often", and a payoff
    that is zero on most of the sample is one the control cannot explain,
    because the control is zero there too. Below about `sigma = 20%` the second
    effect dominates.

    The extreme of that is pinned separately: at `sigma = 10%` and `K = 130` the
    geometric payoff is zero on **every** path of a 40 000-path sample, so the
    control has no sample variance at all.
    """
    assert EvidenceClass.STATISTICAL
    measured = _pilot_control_correlation(
        52, model=BlackScholesModel(sigma=sigma), strike=strike
    )
    assert measured == pytest.approx(rho, abs=2e-4)
    assert 1.0 / (1.0 - measured * measured) == pytest.approx(predicted, rel=0.10)


def test_a_control_with_no_sample_variance_degenerates_to_the_plain_estimator():
    """NEGATIVE_FINDING: the control cannot help where nothing is in the money.

    At `sigma = 10%`, `K = 130`, 52 fixings and 40 000 paths, not one path's
    geometric average finishes above the strike, so the control variable is
    identically zero. `control_variate_coefficient` returns `0.0` rather than
    dividing by a vanishing sum of squares, and the estimator quietly becomes
    the plain one: `beta = 0`, `rho = 0`, predicted factor `1.0`, no NaN
    anywhere.

    What makes this worth pinning is the number it produces, and it is worse
    than "noisy". Exactly one *arithmetic* path pays, so the estimate is
    5.275e-06 with a standard error of 5.275e-06 -- 100% relative -- while the
    geometric closed form at the same point is 3.533e-05. By AM-GM the true
    arithmetic price is **at least** the geometric one, so the answer is out by
    a factor of about seven, and the reported interval does not reach the truth:
    `value + 4 * stderr = 2.64e-05 < 3.53e-05`. The confidence interval of a
    sample proportion built on one success does not cover, and no amount of
    reading it carefully repairs that.

    Three things recorded rather than fixed here.

    - The control variate's value collapses exactly where plain Monte Carlo is
      worst (deep out of the money at low volatility), so the two failure modes
      coincide instead of covering for each other.
    - `control_variance_factor_predicted == 1.0` in `meta` is the
      machine-readable form of "this control did nothing", and is the field to
      check before trusting a reduced estimator's interval.
    - The exact closed form for the geometric average is available at the same
      point and *is* a valid lower bound for the arithmetic price. A future
      slice could report `max(estimate, geometric_price)` or refuse; this one
      reports what the estimator computed and pins the failure.
    """
    assert EvidenceClass.NEGATIVE_FINDING
    option = _option(52, strike=130.0)
    model = BlackScholesModel(sigma=0.10)
    result = price(
        option,
        model,
        MARKET,
        method="mc",
        cfg=MCConfig(
            n_paths=PILOT_PATHS, seed=PILOT_SEED, variance_reduction="control_variate"
        ),
    )
    meta = result.meta
    assert meta is not None
    assert meta["control_beta"] == 0.0
    assert meta["control_correlation"] == 0.0
    assert meta["control_variance_factor_predicted"] == 1.0
    assert math.isfinite(result.value) and math.isfinite(result.stderr)
    assert result.stderr == pytest.approx(result.value, rel=1e-3)

    geometric = discrete_geometric_price(
        S=SPOT,
        K=130.0,
        T=EXPIRY,
        r=MARKET.rate(EXPIRY),
        sigma=0.10,
        fixing_times=option.fixing_times,
        q=MARKET.dividend_yield(EXPIRY),
    )
    assert geometric == pytest.approx(3.5325e-05, rel=1e-3)
    # The estimate is below a value the true price cannot be below (AM-GM), and
    # the reported interval does not reach it either: this is a cell where the
    # confidence statement itself fails, not merely a noisy one.
    assert result.value < geometric
    assert result.value + 4.0 * result.stderr < geometric


def test_the_engine_reports_the_prediction_it_can_compute():
    """`meta` carries rho and `1/(1-rho^2)` from the sample it priced with.

    That is not the same number as the pilot-based prediction the rows above
    assert against -- it is computed on the *same* sample, so it cannot be used
    to validate that sample -- but it is what a caller needs in order to see
    whether the control did anything on their problem.
    """
    result = price(
        _option(52),
        MODEL,
        MARKET,
        method="mc",
        cfg=MCConfig(n_paths=800, seed=101, variance_reduction="control_variate"),
    )
    meta = result.meta
    assert meta is not None
    assert meta["control_variate"] == "discounted_geometric_average_payoff"
    assert meta["control_correlation"] > 0.999
    assert meta["control_variance_factor_predicted"] > 500.0
    assert meta["control_mean"] == pytest.approx(
        discrete_geometric_price(
            S=SPOT,
            K=STRIKE,
            T=EXPIRY,
            r=MARKET.rate(EXPIRY),
            sigma=SIGMA,
            fixing_times=uniform_fixing_times(EXPIRY, 52),
            q=MARKET.dividend_yield(EXPIRY),
        ),
        abs=1e-14,
    )
