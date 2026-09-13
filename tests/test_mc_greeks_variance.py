"""What each Greek estimator is *worth*, measured over seeds at equal cost.

`tests/test_mc_greeks.py` establishes that the estimators are unbiased where
theory says they are. That is necessary and it settles nothing about which one
to use: three unbiased estimators of the same number can differ by three
decimal orders in variance, and on this page two of them do.

The cost axis is **normal draws**, the same axis Slice 7 measures variance
reduction on (`tests/test_mc_variance_ratios.py`). Every comparison below spends
20 000 of them. That is deliberately generous to the bump, which draws the
sample once and evaluates the payoff eight times to produce five Greeks, where
the pathwise and likelihood-ratio estimators evaluate it once and produce all
five from the same sample; on a payoff-evaluation axis the bump would look
eight times worse than it does here. Where the distinction changes a
conclusion, the docstring says so.

The variance being compared is the variance of the *estimator*: 40 independent
seeds, each producing one estimate, and the sample variance of those 40 numbers.
It is measured rather than read off the reported standard errors, which would be
circular.

Precision of the ratios: each variance is a 39-degree-of-freedom estimate, so
the 99% band for a ratio of two independent ones is roughly [0.46, 2.17]. The
assertions below are stated with margins wider than that wherever the claim is
"A beats B by a factor", and the measured numbers are recorded in the docstrings
so that drift is visible in review.

Reference: Glasserman (2003), *Monte Carlo Methods in Financial Engineering*,
sections 7.1-7.4 -- in particular the rule of thumb that the pathwise method is
preferred when the payoff is smooth enough to admit it and the likelihood-ratio
method when it is not, and the `1/T` behaviour of the likelihood-ratio delta
noted in 7.3. Every number below was measured in this repository.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpl.engines.analytic.black_scholes import greeks_european as greeks_analytic
from qpl.engines.analytic.digital import (
    digital_price,
    greeks_digital as greeks_analytic_digital,
)
from qpl.engines.mc.greeks import CONTROL_VARIATE_GREEKS
from qpl.engines.mc.pricers import MCConfig
from qpl.instruments.options import DigitalOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks
from qpl.validation import EvidenceClass, fit_convergence_order

SEEDS = tuple(range(2000, 2040))
BUMP_SEEDS = SEEDS[:24]
"""The bump-in-`h` study repeats the whole estimator six times, so it uses a
shorter seed list; 24 seeds still gives the standard deviations a 23-degree-of-
freedom estimate, which is enough for a fitted order over six levels."""

DRAWS = 20_000
"""Normal draws every estimator is allowed. Antithetic sampling spends one draw
per *pair*, so an antithetic run is given twice the paths -- the Slice 7
convention, restated here because a ratio quoted at equal paths would silently
credit antithetic with free normals."""

N_STRATA = 64

MARKET = Market(
    spot=100.0,
    rate_curve=FlatRateCurve(0.05),
    dividend_curve=FlatDividendCurve(0.01),
)
MODEL = BlackScholesModel(sigma=0.2)
CALL = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
DIGITAL = DigitalOption(kind="call", strike=100.0, expiry=1.0, cash=1.0)

BUMP_LEVELS = (10.0, 3.0, 1.0, 0.3, 0.1, 0.03)
"""Spot bump sizes for the digital bias/variance study, three decimal orders
apart end to end. They are large: the package default is `0.01`, which on this
point is already past the useful range."""


def _paths_for_draws(variance_reduction: str) -> int:
    if variance_reduction == "antithetic":
        return 2 * DRAWS
    if variance_reduction == "stratified":
        # A multiple of the strata count, as close to `DRAWS` as it can be.
        return (DRAWS // N_STRATA) * N_STRATA
    return DRAWS


def _estimates(
    option,
    *,
    greek: str,
    estimator: str,
    variance_reduction: str = "none",
    seeds=SEEDS,
    bumps: dict[str, float] | None = None,
) -> np.ndarray:
    """One estimate per seed, at equal normal draws."""
    kwargs = {"bumps": bumps} if bumps is not None else {}
    return np.array(
        [
            getattr(
                greeks(
                    option,
                    MODEL,
                    MARKET,
                    method="mc",
                    cfg=MCConfig(
                        n_paths=_paths_for_draws(variance_reduction),
                        n_steps=1,
                        seed=seed,
                        greeks_estimator=estimator,
                        variance_reduction=variance_reduction,
                        n_strata=N_STRATA,
                    ),
                    **kwargs,
                ),
                greek,
            )
            for seed in seeds
        ]
    )


def _spread(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1))


# --------------------------------------------------------------------------
# (b) Pathwise against likelihood ratio on a smooth payoff.
# --------------------------------------------------------------------------


def test_pathwise_delta_beats_likelihood_ratio_on_a_smooth_payoff() -> None:
    """Glasserman's rule, measured in the direction it predicts.

    Evidence class: STATISTICAL. 40 seeds, 20 000 normal draws each, ATM call.
    Standard deviations of the delta estimate:

        pathwise          4.2281e-03
        bump              4.2261e-03      variance ratio to pathwise 1.00
        likelihood ratio  9.4740e-03      variance ratio to pathwise 5.02

    The likelihood-ratio estimator is five times noisier in variance for the
    same draws, and the reason is structural rather than incidental: it keeps
    the *payoff* -- an `O(10)` quantity here -- and multiplies it by a mean-zero
    score, where the pathwise estimator replaces the payoff by its derivative,
    an indicator bounded by 1. Multiplying by a mean-zero weight cannot reduce
    variance.

    The bump matching the pathwise estimator to two decimal places is not a
    coincidence either: with common random numbers, a central difference of a
    Lipschitz payoff *is* the pathwise derivative plus `O(h**2)`, and at
    `h = 0.01` that remainder is far below the sampling noise. What separates
    them is gamma, and cost -- the bump spent eight payoff evaluations to get
    here and the pathwise estimator spent one.
    """
    spreads = {
        estimator: _spread(_estimates(CALL, greek="delta", estimator=estimator))
        for estimator in ("pathwise", "likelihood_ratio", "bump")
    }
    ratio = (spreads["likelihood_ratio"] / spreads["pathwise"]) ** 2
    assert ratio > 2.5, spreads
    assert 0.5 < (spreads["bump"] / spreads["pathwise"]) ** 2 < 2.0, spreads

    # Both are unbiased, so the comparison is about variance and nothing else.
    exact = greeks_analytic(CALL, MODEL, MARKET).delta
    for estimator in ("pathwise", "likelihood_ratio"):
        values = _estimates(CALL, greek="delta", estimator=estimator)
        z = (values.mean() - exact) / (_spread(values) / np.sqrt(values.size))
        assert abs(z) < 4.0, (estimator, z)


# --------------------------------------------------------------------------
# (c) The mixed gamma.
# --------------------------------------------------------------------------


def test_the_mixed_gamma_beats_pure_lr_and_annihilates_the_bump() -> None:
    """Requirement (c): pathwise delta + LR gamma is the estimator to use.

    Evidence class: STATISTICAL against a CLOSED_FORM reference. A vanilla call
    has no pathwise gamma -- `f''` is a Dirac mass -- so the three candidates
    are the pure likelihood-ratio gamma, the mixed LR-PW gamma of equation (7)
    in `qpl.engines.mc.greeks` (differentiate the payoff once, the density
    once), and a second-order bump. 40 seeds, 20 000 draws, analytic gamma
    0.018880:

        estimator            mean      sd           variance ratio to mixed
        mixed (pathwise)     0.018850  2.5862e-04        1
        pure LR              0.018844  8.6349e-04       11.1
        second-order bump    0.018110  7.3400e-03      805.5

    All three are centred; the difference is entirely variance, and it is three
    decimal orders end to end. The bump's `O(1/(N h**4))` variance is what a
    *second* difference costs: it divides by `h**2` rather than `h`, and at the
    default `h = 0.01` that is a factor of 10 000 on a quantity whose signal is
    0.019.

    A second finding, and one that was not expected: the pure LR gamma and the
    pure LR vega have **identical** variance ratios (11.1 in both rows of this
    file), because their weights are proportional path by path -- the
    Black-Scholes identity `vega = S_0**2 sigma T gamma` is reproduced by the
    estimator sample, not merely in the mean. `tests/test_mc_greeks.py` asserts
    that identity directly.
    """
    exact = greeks_analytic(CALL, MODEL, MARKET).gamma
    mixed = _estimates(CALL, greek="gamma", estimator="pathwise")
    pure_lr = _estimates(CALL, greek="gamma", estimator="likelihood_ratio")
    bumped = _estimates(CALL, greek="gamma", estimator="bump")

    for values in (mixed, pure_lr, bumped):
        z = (values.mean() - exact) / (_spread(values) / np.sqrt(values.size))
        assert abs(z) < 4.0, z

    assert (_spread(pure_lr) / _spread(mixed)) ** 2 > 4.0
    assert (_spread(bumped) / _spread(mixed)) ** 2 > 100.0

    # And the vega ratio equals the gamma ratio, because the weights are
    # proportional. Measured 11.148 for both.
    lr_vega = _spread(_estimates(CALL, greek="vega", estimator="likelihood_ratio"))
    pw_vega = _spread(_estimates(CALL, greek="vega", estimator="pathwise"))
    gamma_ratio = (_spread(pure_lr) / _spread(mixed)) ** 2
    vega_ratio = (lr_vega / pw_vega) ** 2
    assert vega_ratio == pytest.approx(gamma_ratio, rel=1e-9)


# --------------------------------------------------------------------------
# (b) The digital: the ordering reverses.
# --------------------------------------------------------------------------


def test_on_a_digital_the_likelihood_ratio_wins_by_three_decimal_orders() -> None:
    """Glasserman's rule in the other direction.

    Evidence class: STATISTICAL. Same seeds, same 20 000 draws, ATM digital,
    analytic delta 0.018880:

        estimator          sd           variance ratio to LR
        likelihood ratio   2.0176e-04        1
        bump (h = 0.01)    6.3260e-03       983

    A factor of a thousand, the opposite way round from the vanilla, and the
    pathwise estimator is not in the table at all because it does not exist
    here (it would return exactly 0.0; see `tests/test_digital_mc.py`). One
    payoff, two rules, both of them Glasserman's: differentiate the payoff when
    it is smooth, the density when it is not.
    """
    lr = _estimates(DIGITAL, greek="delta", estimator="likelihood_ratio")
    bumped = _estimates(DIGITAL, greek="delta", estimator="bump")
    assert (_spread(bumped) / _spread(lr)) ** 2 > 100.0, (
        _spread(lr),
        _spread(bumped),
    )


def test_the_digital_lr_delta_noise_grows_as_maturity_shortens() -> None:
    """The `1/T` blow-up of section 7.3, measured -- and it is not what it looks like.

    Evidence class: STATISTICAL, with a contradicted expectation attached.
    The likelihood-ratio delta weight carries `1 / (S_0 sigma sqrt(T))`, so its
    *standard deviation* should grow like `T**-1/2`. At the money, 40 seeds,
    20 000 draws:

        K      T       analytic delta   sd          sd / delta
        100    1.00    0.018880         2.0176e-04  0.0107
        100    0.05    0.088961         9.4218e-04  0.0106

    The absolute standard deviation grows by 4.67, against the 4.47 that
    `sqrt(1/0.05)` predicts -- the blow-up is real. **But the relative precision
    does not move at all** (0.0107 against 0.0106), because an at-the-money
    digital's delta carries the same `1 / (S_0 sigma sqrt(T))` factor and grows
    by 4.71 over the same interval. Reporting "the variance grows 22-fold as
    maturity shortens" without that second column would be true and misleading.

    Where the blow-up actually bites is *off* the money, where the Greek shrinks
    while the noise grows:

        K      T       analytic delta   sd          sd / delta
        110    1.00    0.017676         2.0207e-04  0.0114
        110    0.05    0.009630         4.7264e-04  0.0491

    a 4.3-fold loss of relative precision over the same maturity change. So the
    estimator does degrade as maturity shortens, but the statement has to be
    made about the ratio of the noise to the Greek, and it is a statement about
    moneyness as much as about `T`.
    """
    results: dict[tuple[float, float], tuple[float, float]] = {}
    for strike in (100.0, 110.0):
        for expiry in (1.0, 0.05):
            option = DigitalOption(
                kind="call", strike=strike, expiry=expiry, cash=1.0
            )
            values = _estimates(
                option, greek="delta", estimator="likelihood_ratio"
            )
            exact = greeks_analytic_digital(option, MODEL, MARKET).delta
            results[(strike, expiry)] = (_spread(values), exact)

    atm_long, atm_short = results[(100.0, 1.0)], results[(100.0, 0.05)]
    otm_long, otm_short = results[(110.0, 1.0)], results[(110.0, 0.05)]

    # The absolute blow-up: sqrt(1/0.05) = 4.47 predicted, 4.67 measured.
    growth = atm_short[0] / atm_long[0]
    assert 3.5 < growth < 6.0, growth
    # The contradiction: at the money the relative noise is unchanged.
    atm_relative = (atm_long[0] / atm_long[1], atm_short[0] / atm_short[1])
    assert atm_relative[1] == pytest.approx(atm_relative[0], rel=0.25), atm_relative
    # Off the money it is not: the Greek shrinks while the noise grows.
    otm_relative = (otm_long[0] / otm_long[1], otm_short[0] / otm_short[1])
    assert otm_relative[1] > 3.0 * otm_relative[0], otm_relative


def test_the_digital_bump_delta_trades_bias_against_variance_in_h() -> None:
    """Requirement (b): `sd ~ h**-1/2`, `bias ~ h**2`, and an optimal `h` visible.

    Evidence classes: CONVERGENCE_ORDER for the two fitted orders, STATISTICAL
    for the standard deviations. ATM digital, 24 seeds, 20 000 paths per run,
    `h` from 10 down to 0.03:

        h       deterministic bias   sd (24 seeds)   rmse
        10      -6.5804e-04          1.6796e-04      6.7914e-04
        3       -6.0096e-05          3.6152e-04      3.6649e-04
        1       -6.6855e-06          5.4845e-04      5.4849e-04
        0.3     -6.0178e-07          9.6709e-04      9.6709e-04
        0.1     -6.6865e-08          1.3789e-03      1.3789e-03
        0.03    -6.0179e-09          2.1100e-03      2.1100e-03

    Fitted orders: **-0.4263** for the standard deviation (log-space residual
    0.0968; `-0.5` predicted, and the shortfall is the `O(h**2)` correction to
    the crossing probability, which is still visible at `h = 10`) and
    **+1.9979** for the bias (residual 0.0042, and exactly 2.0000 with residual
    0.0000 over `h <= 1`).

    The bias column is computed from the closed form rather than from the runs,
    and that is the honest way to do it: the Monte Carlo estimator is *exactly
    unbiased for the finite difference*, so its bias **is** the deterministic
    truncation error of that difference, and measuring it from the sample
    instead would be measuring `O(h**2)` through a noise floor 100 times larger
    at the useful end of the range.

    The RMSE column has a minimum at `h = 3`, between a bias that is falling at
    order 2 and a standard deviation that is *rising* at order 1/2. That is the
    optimal bump for this point and this path count, it is three hundred times
    the package default of 0.01, and it moves with `N`: the variance branch
    carries `1/sqrt(N h)` and the bias branch does not, so the optimum scales
    like `N**-1/5`. A default bump size cannot be right for a digital.
    """
    exact = greeks_analytic_digital(DIGITAL, MODEL, MARKET).delta
    price_kwargs = {
        "K": DIGITAL.strike,
        "T": DIGITAL.expiry,
        "r": MARKET.rate(DIGITAL.expiry),
        "sigma": MODEL.sigma,
        "q": MARKET.dividend_yield(DIGITAL.expiry),
        "cash": DIGITAL.cash,
        "kind": DIGITAL.kind,
    }

    spreads, biases, rmse = [], [], []
    for h in BUMP_LEVELS:
        values = _estimates(
            DIGITAL,
            greek="delta",
            estimator="bump",
            seeds=BUMP_SEEDS,
            bumps={"spot": h},
        )
        spreads.append(_spread(values))
        bias = (
            digital_price(S=MARKET.spot + h, **price_kwargs)
            - digital_price(S=MARKET.spot - h, **price_kwargs)
        ) / (2.0 * h) - exact
        biases.append(abs(bias))
        rmse.append(float(np.hypot(bias, spreads[-1])))

    sd_fit = fit_convergence_order(h=list(BUMP_LEVELS), err=spreads)
    assert sd_fit.order == pytest.approx(-0.5, abs=0.15), sd_fit.order
    assert sd_fit.residual < 0.2, sd_fit.residual

    bias_fit = fit_convergence_order(h=list(BUMP_LEVELS), err=biases)
    assert bias_fit.order == pytest.approx(2.0, abs=0.05), bias_fit.order
    assert bias_fit.residual < 0.05, bias_fit.residual

    # The trade-off has an interior optimum: neither end of the range wins.
    best = int(np.argmin(rmse))
    assert 0 < best < len(BUMP_LEVELS) - 1, (best, rmse)
    assert BUMP_LEVELS[best] > 1.0, (BUMP_LEVELS[best], rmse)


# --------------------------------------------------------------------------
# (d) Composition with the Slice 7 variance reduction.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["pathwise", "likelihood_ratio", "bump"])
def test_every_estimator_composes_with_every_sampler(estimator: str) -> None:
    """Requirement (d), measured rather than asserted to work.

    Evidence class: STATISTICAL. Delta of the ATM call, 40 seeds, 20 000 normal
    draws in every cell (so the antithetic runs use 40 000 paths), variance
    factors against the same estimator with no reduction:

        estimator          antithetic   control_variate   stratified
        pathwise               20.43          3.92            84.42
        likelihood ratio        3.35          4.09            15.24
        bump                   20.20          3.88            86.34

    Three things in that table.

    - The reductions are reductions of the **sample**, not of the price, so they
      apply to a Greek sample unchanged. Nothing in the antithetic pair average
      or the strata weighting knows whether the quantity being averaged is a
      payoff or a derivative of one.
    - Antithetic is worth six times more to the pathwise estimator than to the
      likelihood-ratio one (20.4 against 3.3), and that is the reflection
      argument doing its work: the pathwise delta sample `e^{-rT} 1{S_T>K} S_T/S_0`
      is monotone in `Z`, so `Z -> -Z` anticorrelates it strongly, while the LR
      sample carries an explicit odd factor `Z` that the reflection leaves
      much less correlated.
    - The control variate gives a flat ~4 to all three, because it is applied to
      **delta only** and is the same control in each case (the estimator applied
      to `g(x) = x`, whose derivative of `E[X]` is `e^{-rT} e^{mu T}`, known
      exactly).
    """
    plain = _spread(_estimates(CALL, greek="delta", estimator=estimator))
    factors = {
        method: (
            plain
            / _spread(
                _estimates(
                    CALL, greek="delta", estimator=estimator, variance_reduction=method
                )
            )
        )
        ** 2
        for method in ("antithetic", "control_variate", "stratified")
    }
    for method, factor in factors.items():
        assert factor > 1.5, (estimator, method, factors)
    assert factors["stratified"] > factors["antithetic"], factors

    # And the reduced estimate is still centred on the closed form.
    exact = greeks_analytic(CALL, MODEL, MARKET).delta
    values = _estimates(
        CALL, greek="delta", estimator=estimator, variance_reduction="stratified"
    )
    z = (values.mean() - exact) / (_spread(values) / np.sqrt(values.size))
    assert abs(z) < 4.0, z


def test_the_control_variate_is_applied_to_delta_only_and_the_result_says_so() -> None:
    """Evidence class: NEGATIVE_FINDING on the scope of the control.

    Controlling a Greek needs the estimator applied to the control **and** the
    exact value of the matching derivative of `E[X]`. For delta with the
    Slice 7 control `X = e^{-rT} S_T` that derivative is `e^{-rT} e^{mu T}` and
    the algebra is one line. For vega, rho and gamma the derivative of
    `E[X] = S_0 e^{-qT}` is identically zero, so the "control" is a mean-zero
    variate worth only whatever correlation it happens to have, and for theta it
    depends on how the curve is read at a shifted maturity. Rather than ship
    four controls of unstated value, the sample-level estimators control delta
    and leave the rest alone -- and `meta["control_variate_greeks"]` says which,
    so a caller reading an unchanged vega is not left guessing.

    Measured: at 40 seeds the pathwise vega's standard deviation is 5.1724e-01
    with the control variate selected and 5.1724e-01 without it -- the same
    number, because the control was not applied to it -- while delta improves by
    a factor of 3.92.
    """
    cfg = MCConfig(
        n_paths=DRAWS,
        n_steps=1,
        seed=1,
        greeks_estimator="pathwise",
        variance_reduction="control_variate",
    )
    meta = greeks(CALL, MODEL, MARKET, method="mc", cfg=cfg).meta
    assert meta is not None
    assert meta["control_variate_greeks"] == CONTROL_VARIATE_GREEKS == ("delta",)

    plain_vega = _spread(_estimates(CALL, greek="vega", estimator="pathwise"))
    controlled_vega = _spread(
        _estimates(
            CALL,
            greek="vega",
            estimator="pathwise",
            variance_reduction="control_variate",
        )
    )
    assert controlled_vega == pytest.approx(plain_vega, rel=0.05)


def test_the_evidence_classes_this_file_uses_exist() -> None:
    """Guards the docstrings above against an `EvidenceClass` rename."""
    assert {
        EvidenceClass.STATISTICAL,
        EvidenceClass.CONVERGENCE_ORDER,
        EvidenceClass.CLOSED_FORM,
        EvidenceClass.NEGATIVE_FINDING,
    } <= set(EvidenceClass)
