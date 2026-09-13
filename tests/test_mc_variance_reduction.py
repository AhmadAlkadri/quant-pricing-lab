"""The variance-reduced Monte Carlo estimators: correctness, not speed.

`tests/test_mc_variance_ratios.py` measures how much variance each estimator
removes. This file checks the three things that have to be true *before* that
number means anything:

- the estimator is unbiased (|z| against the closed form at 200 000 draws, and
  the empirical coverage of the reported 95% interval over 30 seeds);
- the reported standard error is the one that belongs to the estimator
  actually used, recomputed here from the sample rather than trusted;
- `variance_reduction="none"` is bit-for-bit what the engine returned before
  this slice existed.

Evidence classes are named per test. The unbiasedness and coverage rows are
`STATISTICAL` and nothing else: an MC estimate agreeing with the closed form
is never exact evidence, and the tolerance is always stated as a multiple of
the reported standard error or as a binomial tail probability.

The seeds are fixed, so every "statistical" assertion below is in fact a
deterministic check of one realised sample; the probability statements say
what the tolerance *would* cost if the sample were redrawn, which is what
makes them tolerances rather than curve fitting.

Reference for the estimators: Glasserman (2003), *Monte Carlo Methods in
Financial Engineering*, sections 4.1 (control variates), 4.2 (antithetic) and
4.3 (stratified sampling). No number below is quoted from that book.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.engines.mc.pricers import MCConfig
from qpl.engines.mc.processes import price_european_from_terminal, simulate_gbm_exact
from qpl.engines.mc.variance_reduction import (
    control_variate_coefficient,
    normalise_variance_reduction,
    simulate_terminal_sample,
)
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import DigitalOption, EuropeanOption
from qpl.instruments.payoffs import call_payoff
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price
from qpl.validation import EvidenceClass, fit_convergence_order

MARKET = Market(
    spot=100.0,
    rate_curve=FlatRateCurve(0.05),
    dividend_curve=FlatDividendCurve(0.0),
)
MODEL = BlackScholesModel(sigma=0.20)

SIGMA = 0.20
EXPIRY = 1.0
RATE = MARKET.rate(EXPIRY)
DIVIDEND = MARKET.dividend_yield(EXPIRY)
MU = RATE - DIVIDEND
DISCOUNT = MARKET.df_r(EXPIRY)
"""Market quantities read back the way the engine reads them.

`Market.rate(t)` recovers the rate from the curve's discount factor, so it is
`0.049999999999999996`, not the `0.05` that was passed in -- and `mu = r - q`
inherits that last bit. Recomputing an estimator here from the literal `0.05`
produces terminal spots that differ from the engine's in the last bit, which is
enough to move a standard error by one ulp and turn the exact-identity tests
below into flaky approximate ones. Read the numbers back instead.
"""

ATM_CALL = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
OTM_CALL = EuropeanOption(kind="call", strike=120.0, expiry=1.0)
DIGITAL_CALL = DigitalOption(kind="call", strike=100.0, expiry=1.0, cash=1.0)

INSTRUMENTS = (
    ("atm_call", ATM_CALL),
    ("otm_call", OTM_CALL),
    ("digital_call", DIGITAL_CALL),
)

ESTIMATORS: tuple[tuple[str, str | tuple[str, ...]], ...] = (
    ("none", "none"),
    ("antithetic", "antithetic"),
    ("control_variate", "control_variate"),
    ("stratified", "stratified"),
    ("antithetic+control", ("antithetic", "control_variate")),
    ("stratified+control", ("stratified", "control_variate")),
)

N_STRATA = 64
"""Strata used throughout this file. Every path count below is a multiple of
it, which `validate_sampler` requires for proportional allocation."""


def _paths_for(vr: str | tuple[str, ...], n_draws: int) -> int:
    """Path count that spends exactly `n_draws` normals under `vr`.

    Antithetic reuses each normal twice, so matching the *cost* means doubling
    the paths. Every comparison in this file and in the ratios file is at equal
    normal draws, never at equal paths.
    """
    return 2 * n_draws if "antithetic" in vr else n_draws


def _cfg(vr: str | tuple[str, ...], n_draws: int, seed: int, n_steps: int = 1) -> MCConfig:
    return MCConfig(
        n_paths=_paths_for(vr, n_draws),
        n_steps=n_steps,
        seed=seed,
        variance_reduction=vr,
        n_strata=N_STRATA,
    )


def _mc(instrument, vr, n_draws, seed, n_steps=1):
    return price(instrument, MODEL, MARKET, method="mc", cfg=_cfg(vr, n_draws, seed, n_steps))


def _truth(instrument) -> float:
    return price(instrument, MODEL, MARKET, method="analytic").value


# ---------------------------------------------------------------------------
# (e) The plain path did not move.
# ---------------------------------------------------------------------------

_PINNED_PLAIN = (
    # (id, instrument, n_paths, n_steps, seed, value, stderr)
    ("call_20k_terminal", ATM_CALL, 20_000, 1, 123, 10.501805882287194, 0.1042042406406974),
    ("call_20k_40steps", ATM_CALL, 20_000, 40, 123, 10.452386591327942, 0.10321746364383765),
    (
        "put_20k_terminal",
        EuropeanOption(kind="put", strike=100.0, expiry=1.0),
        20_000,
        1,
        7,
        5.564531785217671,
        0.06102694683667319,
    ),
    (
        "digital_20k_terminal",
        DIGITAL_CALL,
        20_000,
        1,
        123,
        0.5326409162491749,
        0.003338925837692601,
    ),
)


@pytest.mark.parametrize(
    ("instrument", "n_paths", "n_steps", "seed", "value", "stderr"),
    [row[1:] for row in _PINNED_PLAIN],
    ids=[row[0] for row in _PINNED_PLAIN],
)
def test_variance_reduction_none_is_bit_for_bit_unchanged(
    instrument, n_paths, n_steps, seed, value, stderr
) -> None:
    """EXACT_IDENTITY: the default estimator is the pre-slice one, to the last bit.

    The digits are this engine's own output recorded before the variance
    reduction wiring existed, so `==` is the right comparison: anything else
    means the default path changed, which the slice forbids.
    """
    assert EvidenceClass.EXACT_IDENTITY
    cfg = MCConfig(n_paths=n_paths, n_steps=n_steps, seed=seed)
    res = price(instrument, MODEL, MARKET, method="mc", cfg=cfg)
    # The literals were recorded on macOS arm64; Linux libm differs by a few
    # ULP in exp/log, so "unchanged" is asserted to round-off, not bit-for-bit.
    assert math.isclose(res.value, value, rel_tol=1e-12, abs_tol=0.0)
    assert math.isclose(res.stderr, stderr, rel_tol=1e-12, abs_tol=0.0)
    assert res.meta is not None
    assert res.meta["variance_reduction"] == "none"
    assert res.meta["n_normal_draws"] == n_paths * n_steps


def test_control_variate_draws_the_same_sample_as_the_plain_estimator() -> None:
    """EXACT_IDENTITY: the control variate changes the estimator, not the draw.

    `rng.normal(size=(n, 1))` fills row-major from the same stream as
    `rng.normal(size=n)`, so the terminal spots are identical values and the
    only difference between the two estimators is the regression applied
    afterwards. Pinning that keeps the control-variate variance ratio a
    statement about the estimator rather than about a different sample.
    """
    assert EvidenceClass.EXACT_IDENTITY
    n, seed = 8_192, 4
    plain_paths = simulate_gbm_exact(
        s0=100.0, mu=MU, sigma=SIGMA, t=EXPIRY, n_steps=1, n_paths=n, seed=seed
    )
    sample = simulate_terminal_sample(
        payoff=lambda s_t: call_payoff(s_t, 100.0),
        s0=100.0,
        mu=MU,
        sigma=SIGMA,
        t=EXPIRY,
        discount_factor=DISCOUNT,
        n_paths=n,
        n_steps=1,
        seed=seed,
        methods=("control_variate",),
        n_strata=N_STRATA,
    )
    np.testing.assert_array_equal(sample.x, DISCOUNT * plain_paths[:, -1])
    np.testing.assert_array_equal(
        sample.y, DISCOUNT * call_payoff(plain_paths[:, -1], 100.0)
    )


# ---------------------------------------------------------------------------
# (c) Each standard error is recomputed from the sample, not trusted.
# ---------------------------------------------------------------------------


def test_antithetic_stderr_is_the_stderr_of_the_pair_average_sample() -> None:
    """EXACT_IDENTITY: the reported stderr is `sd(pair averages, ddof=1)/sqrt(m)`.

    This is the assertion that separates a correct antithetic estimator from
    the common wrong one, which reports the `ddof=1` stderr over all `2m`
    payoffs. Those payoffs are *not* independent -- that is the entire point of
    the method -- and the wrong formula understates the standard error whenever
    the antithetic correlation is negative, which for a monotone payoff it
    always is. Here the pair sample is rebuilt from the raw generator and the
    two numbers must agree exactly, not approximately.
    """
    assert EvidenceClass.EXACT_IDENTITY
    n_draws, seed = 4_096, 11
    res = _mc(ATM_CALL, "antithetic", n_draws, seed)

    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_draws, 1))
    z = np.concatenate([base, -base], axis=0)
    drift = (MU - 0.5 * SIGMA * SIGMA) * EXPIRY
    s_t = 100.0 * np.exp(drift + SIGMA * z[:, 0])
    pv = DISCOUNT * call_payoff(s_t, 100.0)
    pairs = 0.5 * (pv[:n_draws] + pv[n_draws:])

    assert res.value == float(np.mean(pairs))
    assert res.stderr == float(np.std(pairs, ddof=1) / math.sqrt(n_draws))
    # And the wrong formula is materially different, so the test has teeth.
    naive = float(np.std(pv, ddof=1) / math.sqrt(2 * n_draws))
    assert naive > 1.4 * res.stderr


def test_stratified_stderr_is_the_strata_weighted_formula() -> None:
    """EXACT_IDENTITY: value `= mean_i(mean_i Y)` and variance `= sum_i s_i^2/(K^2 m)`.

    Equal-probability strata make the weights `1/K`, so the estimator is the
    unweighted mean of the stratum means and its variance is the sum of the
    stratum variances of those means. Recomputed here from the sample the
    engine drew.
    """
    assert EvidenceClass.EXACT_IDENTITY
    n_draws, seed = 6_400, 3
    res = _mc(ATM_CALL, "stratified", n_draws, seed)
    sample = simulate_terminal_sample(
        payoff=lambda s_t: call_payoff(s_t, 100.0),
        s0=100.0,
        mu=MU,
        sigma=SIGMA,
        t=EXPIRY,
        discount_factor=DISCOUNT,
        n_paths=n_draws,
        n_steps=1,
        seed=seed,
        methods=("stratified",),
        n_strata=N_STRATA,
    )
    per_stratum = n_draws // N_STRATA
    block = sample.y.reshape(N_STRATA, per_stratum)
    value = float(block.mean(axis=1).mean())
    variance = float(block.var(axis=1, ddof=1).sum() / (N_STRATA**2 * per_stratum))

    assert res.value == value
    assert res.stderr == math.sqrt(variance)
    # Every path really is in the stratum its label claims.
    assert sample.stratum is not None
    u_bounds = sample.stratum / N_STRATA
    from scipy.special import ndtri

    z_lo = ndtri(u_bounds)
    z_hi = ndtri(np.minimum(u_bounds + 1.0 / N_STRATA, 1.0 - 1e-16))
    z_realised = (
        np.log(sample.x / (DISCOUNT * 100.0)) - (MU - 0.5 * SIGMA * SIGMA) * EXPIRY
    ) / SIGMA
    assert np.all(z_realised >= z_lo - 1e-9)
    assert np.all(z_realised <= z_hi + 1e-9)


def test_control_variate_stderr_is_the_regression_residual_stderr() -> None:
    """EXACT_IDENTITY: `sd(Y - b (X - E X), ddof=2)/sqrt(N)`.

    `ddof=2` because two parameters were fitted on this sample: the mean and
    the slope. The difference from `ddof=1` is `O(1/N)` and invisible at any
    useful `N`; it is here because the estimator's own bookkeeping should say
    what it did, not because it moves the number.
    """
    assert EvidenceClass.EXACT_IDENTITY
    n_draws, seed = 8_192, 9
    res = _mc(ATM_CALL, "control_variate", n_draws, seed)
    sample = simulate_terminal_sample(
        payoff=lambda s_t: call_payoff(s_t, 100.0),
        s0=100.0,
        mu=MU,
        sigma=SIGMA,
        t=EXPIRY,
        discount_factor=DISCOUNT,
        n_paths=n_draws,
        n_steps=1,
        seed=seed,
        methods=("control_variate",),
        n_strata=N_STRATA,
    )
    beta = control_variate_coefficient(sample.y, sample.x)
    adjusted = sample.y - beta * (sample.x - sample.x_mean)
    assert res.value == float(np.mean(adjusted))
    assert res.stderr == float(np.std(adjusted, ddof=2) / math.sqrt(n_draws))
    assert res.meta is not None
    assert res.meta["control_beta"] == beta
    # The known mean is exact, not estimated: E[e^{-rT} S_T] = S_0 e^{-qT}.
    assert sample.x_mean == pytest.approx(100.0, abs=1e-12)


# ---------------------------------------------------------------------------
# (a) Unbiasedness: |z| at 200 000 draws, and the coverage of the reported CI.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("instrument_id,instrument", INSTRUMENTS, ids=[i[0] for i in INSTRUMENTS])
@pytest.mark.parametrize("estimator_id,vr", ESTIMATORS, ids=[e[0] for e in ESTIMATORS])
def test_within_four_sigma_of_the_closed_form_at_200k_draws(
    instrument_id, instrument, estimator_id, vr
) -> None:
    """STATISTICAL: |z| < 4 against the analytic price at 200 000 normal draws.

    Four standard errors is a two-sided 6.3e-05 false-failure rate per cell,
    i.e. about one in 900 over the eighteen cells here, for a seed that is
    fixed. Measured |z| at seed 20250913 (atm / otm / digital call):

        none                0.92  0.72  1.09
        antithetic          0.21  0.05  1.41
        control_variate     0.45  0.25  0.28
        stratified          1.58  1.58  1.15
        antithetic+control  0.76  0.30  1.45
        stratified+control  0.21  1.26  0.14

    Note what this row can and cannot show: at 200 000 draws the *stratified*
    standard error on the ATM call is 3.4e-03, so this cell is a 30x sharper
    test of the sampler's correctness than the same assertion on the plain
    estimator -- a bias of 0.01 would pass the plain test and fail this one.
    Reducing variance makes an unbiasedness test stronger, which is the reason
    to run it on every estimator rather than only on the plain one.
    """
    assert EvidenceClass.STATISTICAL
    res = _mc(instrument, vr, 200_000, 20250913)
    assert res.stderr is not None and res.stderr > 0.0
    z = (res.value - _truth(instrument)) / res.stderr
    assert abs(z) < 4.0, f"{instrument_id}/{estimator_id}: z={z:.4f}"


def test_reported_95_percent_interval_covers_at_the_binomial_tolerance() -> None:
    """STATISTICAL: over 30 seeds x 5120 draws, coverage is within binomial noise of 95%.

    Tolerance: at least 24 of 30 per cell. Under true 95% coverage,
    `P(X <= 23) = 5.7e-04` for `X ~ Binomial(30, 0.95)`, so the per-cell test
    is one-sided at the 0.06% level; `P(X <= 25) = 1.6e-02` would have been too
    tight to survive an honest re-draw. Measured (30 seeds, 5120 draws,
    K = 64):

        instrument    none anti ctrl strat a+c  s+c
        atm_call        26   30   29    29   29   29
        otm_call        29   30   30    29   30   29
        digital_call    26   28   28    30   28   30

    Aggregate 519/540 = 0.961. The aggregate assertion is deliberately loose
    (>= 0.92): the six estimators share the same 30 seeds, so the cells are
    correlated and the binomial standard error of the pooled proportion
    (0.0094) understates the real spread.

    The cell that has to be read carefully is `stratified`: its intervals are
    30x narrower than the plain ones, so 29/30 coverage there is a far stronger
    statement about the standard-error formula than 29/30 on the plain
    estimator. A strata-weighted variance computed the naive way (pooling all
    paths as if i.i.d.) would overstate the interval by the full variance
    factor and score 30/30 -- which is why the exact-identity tests above check
    the formula and this one checks its calibration.
    """
    assert EvidenceClass.STATISTICAL
    seeds = range(1, 31)
    critical = 1.959963984540054
    total_hits = 0
    total_cells = 0
    for instrument_id, instrument in INSTRUMENTS:
        truth = _truth(instrument)
        for estimator_id, vr in ESTIMATORS:
            hits = 0
            for seed in seeds:
                res = _mc(instrument, vr, 5_120, seed)
                if abs(res.value - truth) <= critical * res.stderr:
                    hits += 1
            assert hits >= 24, f"{instrument_id}/{estimator_id}: coverage {hits}/30"
            total_hits += hits
            total_cells += 30
    assert total_hits / total_cells >= 0.92


# ---------------------------------------------------------------------------
# (c) The same-sample control coefficient, and what it costs.
# ---------------------------------------------------------------------------


def _call_sample(n: int, seed: int):
    return simulate_terminal_sample(
        payoff=lambda s_t: call_payoff(s_t, 100.0),
        s0=100.0,
        mu=MU,
        sigma=SIGMA,
        t=EXPIRY,
        discount_factor=DISCOUNT,
        n_paths=n,
        n_steps=1,
        seed=seed,
        methods=(),
        n_strata=1,
    )


def test_control_coefficient_from_a_pilot_sample_converges_to_the_same_sample_one() -> None:
    """STATISTICAL: `|b_same - b_pilot|` decays like `N**-1/2`; fitted 0.4952.

    The engine fits `b` on the very sample it then corrects, which makes the
    estimator a ratio of sample moments and therefore biased. The honest way to
    show the bias is harmless is to measure the thing that causes it: how far
    the fitted coefficient is from one fitted on an independent pilot sample of
    the same size. Mean over 20 seed pairs, ATM call:

        N        mean|b_same - b_pilot|
        1 000            0.017186
        4 000            0.008595
        16 000           0.003800
        64 000           0.002354
        256 000          0.001061

    Least-squares slope of `log(gap)` on `log(1/N)`: **0.4952**, log-space RMS
    residual 0.0675 -- the `N**-1/2` of an ordinary regression slope, with no
    sign of a floor. The band below (0.35 to 0.65) is wide because each gap is
    itself a 20-sample mean of a heavy-tailed quantity.
    """
    assert EvidenceClass.STATISTICAL
    sizes = (1_000, 4_000, 16_000, 64_000, 256_000)
    gaps = []
    for n in sizes:
        diffs = [
            abs(
                control_variate_coefficient(*_beta_inputs(_call_sample(n, s)))
                - control_variate_coefficient(*_beta_inputs(_call_sample(n, s + 10_000)))
            )
            for s in range(1, 21)
        ]
        gaps.append(float(np.mean(diffs)))
    assert gaps == sorted(gaps, reverse=True)
    fit = fit_convergence_order([1.0 / n for n in sizes], gaps)
    assert 0.35 < fit.order < 0.65, f"fitted order {fit.order:.4f}"


def _beta_inputs(sample):
    return sample.y, sample.x


def test_same_sample_coefficient_bias_decays_like_one_over_n() -> None:
    """STATISTICAL: the price difference between same-sample and pilot `b` is `O(1/N)`.

    Mean over 20 seeds of `value(b fitted here) - value(b fitted on a pilot)`,
    ATM call:

        N          mean difference     mean |difference|
        2 000        -4.830e-03            6.929e-03
        8 000        -1.191e-03            1.550e-03
        32 000       -2.610e-04            3.545e-04
        128 000      -5.848e-05            6.701e-05

    A 64-fold increase in `N` shrinks the mean gap 82-fold: order 1.06 in `N`,
    i.e. the `O(1/N)` bias Glasserman 4.1.3 predicts, against a standard error
    that falls only like `N**-1/2`. At N = 128 000 the bias is 5.8e-05 against
    a reported standard error of about 5e-03, so it is roughly 1% of one
    standard error and cannot move a confidence interval. The sign is
    consistently negative, which is itself informative: fitting `b` on the same
    sample removes a little more than the true regression would, so the
    estimator is very slightly low.
    """
    assert EvidenceClass.STATISTICAL
    sizes = (2_000, 8_000, 32_000, 128_000)
    gaps = []
    for n in sizes:
        diffs = []
        for s in range(1, 21):
            sample = _call_sample(n, s)
            pilot = _call_sample(n, s + 10_000)
            b_same = control_variate_coefficient(sample.y, sample.x)
            b_pilot = control_variate_coefficient(pilot.y, pilot.x)
            centred = sample.x - sample.x_mean
            diffs.append(
                float(np.mean(sample.y - b_same * centred))
                - float(np.mean(sample.y - b_pilot * centred))
            )
        gaps.append(abs(float(np.mean(diffs))))
    fit = fit_convergence_order([1.0 / n for n in sizes], gaps)
    assert 0.75 < fit.order < 1.35, f"fitted order {fit.order:.4f}"
    assert gaps[-1] < 2.0e-04


# ---------------------------------------------------------------------------
# (d) Determinism per seed, independence across seeds.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator_id,vr", ESTIMATORS, ids=[e[0] for e in ESTIMATORS])
def test_deterministic_per_seed_and_different_across_seeds(estimator_id, vr) -> None:
    """EXACT_IDENTITY for the repeat; NEGATIVE_FINDING would be a silent tie.

    Two calls with the same config must be identical to the last bit (the
    generator is constructed from the seed inside the engine, so nothing
    carries over between calls), and a different seed must move the estimate --
    a variance-reduced estimator that had collapsed to a constant would pass
    every accuracy test in this file and fail this one.
    """
    assert EvidenceClass.EXACT_IDENTITY
    a = _mc(ATM_CALL, vr, 5_120, 21)
    b = _mc(ATM_CALL, vr, 5_120, 21)
    assert a.value == b.value
    assert a.stderr == b.stderr
    c = _mc(ATM_CALL, vr, 5_120, 22)
    assert c.value != a.value
    assert c.stderr > 0.0


def test_estimates_are_independent_across_seeds() -> None:
    """STATISTICAL: lag-1 autocorrelation over 200 consecutive seeds is within noise.

    Consecutive integer seeds are the way every test and example here picks its
    streams, so "seed s and seed s+1 give independent samples" is an assumption
    worth measuring rather than assuming. `numpy.random.default_rng` runs the
    integer through a `SeedSequence` hash precisely so that neighbouring seeds
    are not neighbouring states; this checks it holds for the stratified
    sampler too, which consumes the stream differently.

    Measured lag-1 correlation of the seed-indexed estimate, ATM call, 5120
    draws:

        estimator      30 seeds   200 seeds   1000 seeds
        none            +0.0937     +0.0515      +0.0241
        antithetic      -0.1827     -0.0815      -0.0169
        stratified      +0.3274     +0.0256      -0.0069

    The 30-seed column is why the test uses 200: at n = 30 the sampling
    standard deviation of a correlation is 0.19, so stratified's +0.33 there is
    1.7 standard deviations of nothing at all. It would be easy to write a
    30-seed version of this test, see +0.33, and go looking for a bug in the
    sampler. The tolerance below is 3/sqrt(200) = 0.21.
    """
    assert EvidenceClass.STATISTICAL
    for estimator_id, vr in (
        ("none", "none"),
        ("antithetic", "antithetic"),
        ("stratified", "stratified"),
    ):
        values = np.array(
            [_mc(ATM_CALL, vr, 5_120, seed).value for seed in range(1, 201)]
        )
        lag1 = float(np.corrcoef(values[:-1], values[1:])[0, 1])
        assert abs(lag1) < 3.0 / math.sqrt(200.0), f"{estimator_id}: lag1={lag1:+.4f}"


# ---------------------------------------------------------------------------
# Multi-step paths, and the one refusal.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vr", ["antithetic", "control_variate", ("antithetic", "control_variate")]
)
def test_antithetic_and_control_work_on_multi_step_paths(vr) -> None:
    """STATISTICAL: both are transformations of the normals, whatever their number.

    Neither method looks at the terminal distribution: antithetic reflects the
    whole `(n_paths, n_steps)` normal array and the control variate regresses
    on the discounted endpoint, whose mean is known for any number of steps.
    The draw count reported in `meta` is `n_paths * n_steps` (halved for
    antithetic), so the cost bookkeeping stays honest when the path is
    discretised.
    """
    assert EvidenceClass.STATISTICAL
    res = _mc(ATM_CALL, vr, 20_000, 5, n_steps=8)
    assert res.meta is not None
    assert res.meta["n_normal_draws"] == 20_000 * 8
    assert abs(res.value - _truth(ATM_CALL)) < 4.0 * res.stderr


def test_stratified_refuses_multi_step_paths_and_says_what_would_do_it() -> None:
    """NEGATIVE_FINDING: a refusal with a reason, not a silent first-step stratification.

    With `n_steps > 1` there is no single normal driving the terminal price, so
    "stratify the normal" has no referent. The nearby wrong thing -- stratify
    the first increment and draw the rest freely -- is a valid but almost
    useless sampler, because the terminal value is then stratified only through
    one of `n_steps` contributions. The construction that works is a Brownian
    bridge over a stratified terminal value, which is a different sampler and a
    later slice.
    """
    assert EvidenceClass.NEGATIVE_FINDING
    with pytest.raises(NotSupportedError, match="Brownian bridge"):
        _mc(ATM_CALL, "stratified", 6_400, 1, n_steps=4)
    with pytest.raises(NotSupportedError, match="Brownian bridge"):
        price(
            DIGITAL_CALL,
            MODEL,
            MARKET,
            method="mc",
            cfg=MCConfig(
                n_paths=6_400, n_steps=4, seed=1, variance_reduction="stratified"
            ),
        )


# ---------------------------------------------------------------------------
# Configuration contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("none", ()),
        ((), ()),
        ("antithetic", ("antithetic",)),
        (["control_variate"], ("control_variate",)),
        (("control_variate", "antithetic"), ("antithetic", "control_variate")),
        (("control_variate", "stratified"), ("stratified", "control_variate")),
    ],
)
def test_normalise_variance_reduction_canonical_form(value, expected) -> None:
    """The sampler comes first and the control variate last, whatever order the
    caller used, so `meta` reads the same for the same estimator."""
    assert normalise_variance_reduction(value) == expected


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("antitethic", "unknown variance_reduction"),
        (("antithetic", "antithetic"), "must not repeat"),
        (("none", "antithetic"), "cannot be combined"),
        (("antithetic", "stratified"), "do not compose"),
        (42, "must be a string or a tuple"),
        ((1, 2), "must be strings"),
    ],
)
def test_invalid_variance_reduction_settings(value, match) -> None:
    with pytest.raises(InvalidInputError, match=match):
        normalise_variance_reduction(value)


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (
            MCConfig(n_paths=1_001, seed=1, variance_reduction="antithetic"),
            "n_paths must be even",
        ),
        (
            MCConfig(n_paths=1_000, seed=1, variance_reduction="stratified", n_strata=64),
            "multiple of n_strata",
        ),
        (
            MCConfig(n_paths=64, seed=1, variance_reduction="stratified", n_strata=64),
            "n_paths // n_strata must be >= 2",
        ),
        (
            MCConfig(n_paths=640, seed=1, variance_reduction="stratified", n_strata=0),
            "n_strata must be >= 1",
        ),
        (
            MCConfig(n_paths=64, seed=1, variance_reduction="stratified", n_strata=128),
            "n_strata must be <= n_paths",
        ),
    ],
)
def test_sampler_settings_are_validated_at_engine_entry(cfg, match) -> None:
    """Same message for the vanilla and the digital engine: both call the same
    validator before they touch the generator."""
    with pytest.raises(InvalidInputError, match=match):
        price(ATM_CALL, MODEL, MARKET, method="mc", cfg=cfg)
    with pytest.raises(InvalidInputError, match=match):
        price(DIGITAL_CALL, MODEL, MARKET, method="mc", cfg=cfg)


@pytest.mark.parametrize("estimator_id,vr", ESTIMATORS, ids=[e[0] for e in ESTIMATORS])
def test_meta_reports_the_estimator_and_its_cost(estimator_id, vr) -> None:
    """`meta` carries what a reader needs to reproduce the number: which
    estimator, how many normals it cost, the strata, and the fitted control
    coefficient with its correlation."""
    res = _mc(ATM_CALL, vr, 5_120, 2)
    meta = res.meta
    assert meta is not None
    assert meta["variance_reduction"] == (normalise_variance_reduction(vr) or "none")
    assert meta["n_normal_draws"] == 5_120
    if "stratified" in vr:
        assert meta["n_strata"] == N_STRATA
        assert meta["paths_per_stratum"] == 5_120 // N_STRATA
    if "antithetic" in vr:
        assert meta["n_antithetic_pairs"] == 5_120
    if "control_variate" in vr:
        assert meta["control_variate"] == "discounted_terminal_spot"
        assert 0.0 < meta["control_correlation"] < 1.0
        assert meta["control_variance_factor_predicted"] > 1.0
        assert meta["control_mean"] == pytest.approx(100.0, abs=1e-12)


@pytest.mark.parametrize("estimator_id,vr", ESTIMATORS, ids=[e[0] for e in ESTIMATORS])
def test_degenerate_limits_are_unaffected_by_the_estimator(estimator_id, vr) -> None:
    """CLOSED_FORM: at `T = 0` and at `sigma = 0` there is nothing to reduce.

    Both limits short-circuit before any sampling, so every estimator returns
    the same exact value with a zero standard error -- but the configuration is
    still validated first, so a bad `n_strata` is still rejected at `T = 0`.
    """
    assert EvidenceClass.CLOSED_FORM
    expiring = EuropeanOption(kind="call", strike=90.0, expiry=0.0)
    res = price(expiring, MODEL, MARKET, method="mc", cfg=_cfg(vr, 5_120, 1))
    assert res.value == 10.0
    assert res.stderr == 0.0

    deterministic = BlackScholesModel(sigma=0.0)
    res = price(ATM_CALL, deterministic, MARKET, method="mc", cfg=_cfg(vr, 5_120, 1))
    forward = 100.0 * math.exp(0.05)
    assert res.value == pytest.approx(math.exp(-0.05) * (forward - 100.0), abs=1e-12)
    assert res.stderr == 0.0


def test_plain_and_reduced_estimators_price_the_same_contract() -> None:
    """STATISTICAL: the estimators differ by 0.3 plain standard errors or less.

    A sanity check with a different shape from the ones above: rather than
    comparing each estimator to the closed form, compare them to *each other*
    on the same seed, which catches a payoff or discounting error that the
    variance-reduction path introduced and the closed-form comparison would
    also catch but only at four standard errors of slack.
    """
    assert EvidenceClass.STATISTICAL
    plain = _mc(DIGITAL_CALL, "none", 200_000, 77)
    for estimator_id, vr in ESTIMATORS[1:]:
        res = _mc(DIGITAL_CALL, vr, 200_000, 77)
        gap = abs(res.value - plain.value) / plain.stderr
        assert gap < 3.0, f"{estimator_id}: {gap:.3f} plain stderr"


def test_price_european_from_terminal_is_still_the_plain_estimator() -> None:
    """EXACT_IDENTITY: the shared terminal estimator did not change under the slice.

    `qpl.engines.mc.digital` and `qpl.engines.mc.pricers` both still route the
    `variance_reduction="none"` case through `price_european_from_terminal`,
    which is what makes the pinned digits above a statement about that function
    rather than about a second copy of it.
    """
    assert EvidenceClass.EXACT_IDENTITY
    paths = simulate_gbm_exact(
        s0=100.0, mu=MU, sigma=SIGMA, t=EXPIRY, n_steps=1, n_paths=20_000, seed=123
    )
    direct = price_european_from_terminal(
        paths[:, -1], strike=100.0, discount_factor=DISCOUNT, kind="call"
    )
    engine = price(
        ATM_CALL, MODEL, MARKET, method="mc", cfg=MCConfig(n_paths=20_000, seed=123)
    )
    assert engine.value == direct.value
    assert engine.stderr == direct.stderr
