"""Monte Carlo Greeks: is each estimator unbiased where theory says it is?

This file answers "is it right"; `tests/test_mc_greeks_variance.py` answers
"what is it worth". The split matters because the two questions have different
evidence classes: unbiasedness is a STATISTICAL claim about coverage against a
CLOSED_FORM reference, and a variance comparison is a measurement whose
conclusion is a ratio, not a number.

How unbiasedness is checked
---------------------------
Not by "the estimate is close to the analytic value" -- close on what scale? --
but by **confidence-interval coverage over independent seeds**. Each of 40
seeds produces an estimate and its own reported standard error; the nominal 95%
interval either covers the analytic Greek or it does not. Under the null (the
estimator is unbiased *and* its reported standard error is calibrated) the
count is Binomial(40, 0.95), mean 38 and standard deviation 1.38. A biased
estimator or an understated standard error both show up here, and the test
cannot tell them apart -- which is the honest state of affairs, since a
confidence interval is a joint claim about both.

The thresholds below are binomial tail bounds, not the measured counts:
`>= 34` of 40 has a one-sided probability near 6e-03 under the null, so a
passing run is weak evidence per Greek and strong evidence over the 15 Greek x
estimator cells the file checks together. The measured counts are recorded in
each docstring so that a drift shows up in review even when the test still
passes.

Reference: Glasserman (2003), *Monte Carlo Methods in Financial Engineering*,
sections 7.1-7.4; Broadie & Glasserman (1996), *Management Science* 42(2).
Derivations are in `qpl.engines.mc.greeks`; nothing here is quoted from either
source.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from qpl.engines.analytic.black_scholes import greeks_european as greeks_analytic
from qpl.engines.mc.greeks import (
    GREEK_NAMES,
    LIKELIHOOD_RATIO,
    MIXED_PATHWISE_LR,
    PATHWISE,
    draw_path_sample,
    estimate_greek,
    likelihood_ratio_terminal_greeks,
    pathwise_terminal_greeks,
)
from qpl.engines.mc.pricers import MCConfig, price_european
from qpl.exceptions import InvalidInputError
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks
from qpl.validation import EvidenceClass

COVERAGE_SEEDS = tuple(range(1000, 1040))
"""40 seeds. Fixed, so the counts below reproduce exactly."""

COVERAGE_PATHS = 20_000
COVERAGE_MIN = 34
"""Binomial(40, 0.95) tail bound: P(X <= 34) ~ 6e-03 one-sided."""

Z_95 = 1.959963984540054
"""Two-sided 95% normal quantile, written out rather than imported so the
interval this file calls "95%" is visible in the file that uses it."""

CALL = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
PUT = EuropeanOption(kind="put", strike=110.0, expiry=0.75)
MODEL = BlackScholesModel(sigma=0.2)
MARKET = Market(
    spot=100.0,
    rate_curve=FlatRateCurve(0.05),
    dividend_curve=FlatDividendCurve(0.01),
)


def _coverage(option, estimator: str) -> dict[str, int]:
    """How often the reported 95% interval covers the analytic Greek."""
    exact = greeks_analytic(option, MODEL, MARKET)
    hits = dict.fromkeys(GREEK_NAMES, 0)
    for seed in COVERAGE_SEEDS:
        cfg = MCConfig(
            n_paths=COVERAGE_PATHS, n_steps=1, seed=seed, greeks_estimator=estimator
        )
        result = greeks(option, MODEL, MARKET, method="mc", cfg=cfg)
        assert result.meta is not None
        for name in GREEK_NAMES:
            stderr = result.meta["stderr"][name]
            if abs(getattr(result, name) - getattr(exact, name)) <= Z_95 * stderr:
                hits[name] += 1
    return hits


@pytest.mark.parametrize("estimator", ["bump", PATHWISE, LIKELIHOOD_RATIO])
def test_every_estimator_covers_the_analytic_greek_over_40_seeds(estimator: str) -> None:
    """Evidence class: STATISTICAL against a CLOSED_FORM reference.

    Measured hit counts out of 40, at 20 000 paths on the ATM call:

        Greek    bump   pathwise   likelihood_ratio
        delta      35       35            38
        gamma      37       38            38
        vega       38       38            38
        theta      38       38            38
        rho        34       34            38

    Three things are worth reading off that table rather than glossing.

    1. The likelihood-ratio column is 38 everywhere, which is what an
       *exactly* unbiased estimator with a calibrated standard error looks
       like: 38 is the mean of Binomial(40, 0.95).
    2. The pathwise column matches the bump column almost cell for cell,
       because on a Lipschitz payoff with common random numbers the central
       difference *is* the pathwise estimator up to `O(h**2)`. The one cell
       where they separate is gamma, and it separates by a factor of 29 in
       standard deviation (see `tests/test_mc_greeks_variance.py`).
    3. `rho` at 34 is the weakest cell in both non-LR columns and it is not a
       bias: the same seeds give a mean 0.48 standard errors below the closed
       form, well inside noise. Coverage below nominal at a fixed set of seeds
       is what a 40-draw binomial does about 6 times in a thousand, and the
       threshold is set to let it.

    `bump` is included in the parametrisation even though its central
    difference is biased at `O(h**2)`: at the default bump sizes that bias is
    far below the standard error, which is itself the finding -- the bump is
    *not* detectably biased here, and the digital is where it falls apart.
    """
    hits = _coverage(CALL, estimator)
    for name in GREEK_NAMES:
        assert hits[name] >= COVERAGE_MIN, (estimator, name, hits)


def test_the_put_is_covered_too_so_the_payoff_derivative_sign_is_checked() -> None:
    """A dropped minus sign in `f'` for a put would pass every call test.

    Evidence class: STATISTICAL. Measured hit counts on the 0.75-year 110-strike
    put at 20 000 paths, pathwise: delta 37, gamma 37, vega 37, theta 38,
    rho 38.
    """
    hits = _coverage(PUT, PATHWISE)
    for name in GREEK_NAMES:
        assert hits[name] >= COVERAGE_MIN, (name, hits)


def _old_price_level_bump(option, model, market, cfg, bumps=None):
    """The pre-Slice-10 estimator, rebuilt here from `price_european` alone.

    This is the reference for the reproduction claim and it is deliberately
    written out rather than imported: the point of the claim is that the new
    sample-level implementation returns what a caller who differences *prices*
    would have got, so the reference has to difference prices.
    """
    s0 = market.spot
    t = option.expiry
    sigma = model.sigma

    def _bump(name: str, default: float) -> float:
        if bumps is None or name not in bumps:
            return default
        return float(bumps[name])

    d_s = _bump("spot", max(abs(s0) * 1e-4, 1e-6))
    if s0 <= d_s:
        d_s = 0.5 * s0
    d_sigma = _bump("sigma", 1e-4)
    if sigma > 0.0 and sigma <= d_sigma:
        d_sigma = 0.5 * sigma
    d_r = _bump("r", 1e-5)

    def _price(mkt, mdl):
        return price_european(option, mdl, mkt, cfg=cfg).value

    base = _price(market, model)

    def _at_spot(spot: float) -> float:
        return _price(
            Market(
                spot=spot,
                rate_curve=market.rate_curve,
                dividend_curve=market.dividend_curve,
            ),
            model,
        )

    up, down = _at_spot(s0 + d_s), _at_spot(s0 - d_s)
    delta = (up - down) / (2.0 * d_s)
    gamma = (up - 2.0 * base + down) / (d_s * d_s)

    if sigma == 0.0:
        vega = 0.0
    else:
        vega = (
            _price(market, BlackScholesModel(sigma=sigma + d_sigma))
            - _price(market, BlackScholesModel(sigma=sigma - d_sigma))
        ) / (2.0 * d_sigma)

    r = market.rate(t)
    q = market.dividend_yield(t)

    def _at_rate(rate: float) -> float:
        return _price(
            Market(
                spot=s0,
                rate_curve=FlatRateCurve(rate, allow_negative=True),
                dividend_curve=FlatDividendCurve(q, allow_negative=True),
            ),
            model,
        )

    rho = (_at_rate(r + d_r) - _at_rate(r - d_r)) / (2.0 * d_r)

    d_t = _bump("time", min(1e-4, t / 2.0))
    if t <= d_t:
        d_t = t * 0.5
    shifted = price_european(
        replace(option, expiry=t - d_t), model, market, cfg=cfg
    ).value
    theta = (shifted - base) / d_t
    return delta, gamma, vega, theta, rho


_REPRODUCTION_CASES = [
    ("plain", MCConfig(n_paths=20_000, n_steps=1, seed=42), None, MODEL),
    (
        "explicit_bumps",
        MCConfig(n_paths=20_000, n_steps=1, seed=7),
        {"spot": 1e-2, "sigma": 1e-4, "r": 1e-5, "time": 1e-3},
        MODEL,
    ),
    ("multi_step", MCConfig(n_paths=20_000, n_steps=4, seed=3), None, MODEL),
    (
        "antithetic",
        MCConfig(n_paths=20_000, n_steps=1, seed=3, variance_reduction="antithetic"),
        None,
        MODEL,
    ),
    (
        "control_variate",
        MCConfig(
            n_paths=20_000, n_steps=1, seed=3, variance_reduction="control_variate"
        ),
        None,
        MODEL,
    ),
    (
        "stratified",
        MCConfig(n_paths=20_480, n_steps=1, seed=3, variance_reduction="stratified"),
        None,
        MODEL,
    ),
    ("zero_sigma", MCConfig(n_paths=200, n_steps=1, seed=3), None, BlackScholesModel(0.0)),
]


@pytest.mark.parametrize(
    ("name", "cfg", "bumps", "model"),
    _REPRODUCTION_CASES,
    ids=[case[0] for case in _REPRODUCTION_CASES],
)
def test_bump_reproduces_the_pre_slice_price_level_crn_greeks(
    name: str, cfg: MCConfig, bumps: dict[str, float] | None, model: BlackScholesModel
) -> None:
    """Evidence class: EXACT_IDENTITY, and it really is exact.

    Slice 10 moved the bump estimator from "call `price_european` eight times"
    to "draw the normals once and reprice the sample eight ways", so that the
    paired per-path differences exist and a standard error can be reported at
    all. The claim here is that this changed **nothing** about the numbers:
    equality is asserted with `==`, not with a tolerance, over seven
    configurations including `n_steps = 4`, all three variance-reduction
    samplers and `sigma = 0`.

    What makes that possible rather than lucky is that the new sampler
    reproduces the generator's *consumption pattern*: `n_paths` normals per
    step in a loop for the plain path, the `(n_pairs, n_steps)` block for
    antithetic, the stratified uniforms for stratified -- and then the same
    exact-lognormal recursion, the same `mean`, the same scenario values
    differenced the same way. A block draw would have been an equally valid
    sample and a different one, and this assertion would have had to become a
    tolerance. See `qpl.engines.mc.greeks.draw_path_sample`.
    """
    expected = _old_price_level_bump(CALL, model, MARKET, cfg, bumps)
    kwargs = {"bumps": bumps} if bumps is not None else {}
    result = greeks(CALL, model, MARKET, method="mc", cfg=cfg, **kwargs)
    actual = (result.delta, result.gamma, result.vega, result.theta, result.rho)
    assert actual == expected, (name, actual, expected)


def test_reported_stderr_is_the_sample_stderr_of_the_estimator() -> None:
    """The standard error is recomputed from the realised sample and compared with `==`.

    Evidence class: EXACT_IDENTITY. Not a claim that the number is *right* --
    that is what the coverage test above is for -- but that the number reported
    is the one this file can rebuild from the public sampler and estimator
    functions, so a future change to either shows up here rather than as a
    coverage drift.
    """
    cfg = MCConfig(n_paths=20_000, n_steps=1, seed=11, greeks_estimator=PATHWISE)
    result = greeks(CALL, MODEL, MARKET, method="mc", cfg=cfg)
    assert result.meta is not None

    t = CALL.expiry
    r, q, df = MARKET.rate(t), MARKET.dividend_yield(t), MARKET.df_r(t)
    sample = draw_path_sample(
        s0=MARKET.spot,
        mu=r - q,
        sigma=MODEL.sigma,
        t=t,
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        seed=cfg.seed,
        methods=(),
        n_strata=cfg.n_strata,
    )
    payoff = np.maximum(sample.spots - CALL.strike, 0.0)
    values = pathwise_terminal_greeks(
        sample,
        payoff=payoff,
        payoff_derivative=np.asarray(sample.spots > CALL.strike, dtype=float),
        s0=MARKET.spot,
        mu=r - q,
        sigma=MODEL.sigma,
        t=t,
        r=r,
        discount_factor=df,
    )
    for name in GREEK_NAMES:
        rebuilt = estimate_greek(
            values[name], sample=sample, methods=(), estimator=PATHWISE
        )
        assert rebuilt.value == getattr(result, name), name
        assert rebuilt.stderr == result.meta["stderr"][name], name


def test_pathwise_rho_collapses_to_the_discounted_strike_indicator() -> None:
    """`rho_pw = T e^{-rT} K 1{S_T > K}` for a call, pathwise, per path.

    Evidence class: EXACT_IDENTITY. Equation (6) of `qpl.engines.mc.greeks` is
    `T e^{-rT} (f'(S_T) S_T - f(S_T))`, and for a call the bracket is
    `S_T - (S_T - K) = K` in the money and `0` out of it. Its expectation is
    `K T e^{-rT} N(d2)`, the closed-form rho -- so this identity is what turns
    "rho includes the discount factor's own term" from a remark into a check:
    drop the `- f(S_T)` and the per-path sample stops being a constant times an
    indicator, which this assertion sees immediately.
    """
    t = CALL.expiry
    r, q, df = MARKET.rate(t), MARKET.dividend_yield(t), MARKET.df_r(t)
    sample = draw_path_sample(
        s0=MARKET.spot,
        mu=r - q,
        sigma=MODEL.sigma,
        t=t,
        n_paths=5_000,
        n_steps=1,
        seed=5,
        methods=(),
        n_strata=64,
    )
    in_money = sample.spots > CALL.strike
    values = pathwise_terminal_greeks(
        sample,
        payoff=np.maximum(sample.spots - CALL.strike, 0.0),
        payoff_derivative=np.asarray(in_money, dtype=float),
        s0=MARKET.spot,
        mu=r - q,
        sigma=MODEL.sigma,
        t=t,
        r=r,
        discount_factor=df,
    )
    expected = t * df * CALL.strike * np.asarray(in_money, dtype=float)
    assert np.max(np.abs(values["rho"] - expected)) < 1e-11
    # And it is not trivially zero: some paths finish in the money.
    assert 0 < int(in_money.sum()) < in_money.size


def test_lr_vega_is_the_black_scholes_multiple_of_lr_gamma_path_by_path() -> None:
    """`vega = S_0**2 sigma T gamma` holds sample by sample, not just in the mean.

    Evidence class: EXACT_IDENTITY, and a contradicted expectation: the two
    likelihood-ratio weights were derived independently -- (10) from the
    `sigma` score and (9) from the second-order `S_0` score -- and they turn out
    to be proportional,

        (z**2 - 1)/sigma - z sqrt(T)  =  S_0**2 sigma T * (z**2 - z sigma sqrt(T) - 1) / (S_0**2 sigma**2 T),

    which is an algebraic identity in `z`. So the closed-form Black-Scholes
    relation `vega = S_0**2 sigma T gamma` is reproduced by the estimator
    *pathwise*, which means the LR vega and the LR gamma carry exactly the same
    noise up to that constant and are not two independent pieces of evidence
    about the same sample. The coverage table above shows it: their hit counts
    are equal seed for seed.
    """
    t = CALL.expiry
    r, q, df = MARKET.rate(t), MARKET.dividend_yield(t), MARKET.df_r(t)
    sample = draw_path_sample(
        s0=MARKET.spot,
        mu=r - q,
        sigma=MODEL.sigma,
        t=t,
        n_paths=5_000,
        n_steps=1,
        seed=5,
        methods=(),
        n_strata=64,
    )
    values = likelihood_ratio_terminal_greeks(
        sample,
        payoff=np.maximum(sample.spots - CALL.strike, 0.0),
        s0=MARKET.spot,
        mu=r - q,
        sigma=MODEL.sigma,
        t=t,
        r=r,
        discount_factor=df,
    )
    factor = MARKET.spot**2 * MODEL.sigma * t
    scale = float(np.max(np.abs(values["vega"])))
    assert np.max(np.abs(values["vega"] - factor * values["gamma"])) < 1e-10 * scale


def test_gamma_is_the_mixed_estimator_under_pathwise_and_pure_lr_under_lr() -> None:
    """`meta["estimator"]` is per Greek, because one result can mix families.

    Evidence class: EXACT_IDENTITY on the contract. A pathwise result cannot
    carry a pathwise gamma -- the payoff has no second derivative -- so it
    carries the LR-PW mixed estimator of equation (7) and says so rather than
    labelling the whole result "pathwise".
    """
    cfg = MCConfig(n_paths=5_000, n_steps=1, seed=1, greeks_estimator=PATHWISE)
    meta = greeks(CALL, MODEL, MARKET, method="mc", cfg=cfg).meta
    assert meta is not None
    assert meta["estimator"]["gamma"] == MIXED_PATHWISE_LR
    assert meta["estimator"]["delta"] == PATHWISE
    assert set(meta["stderr"]) == set(GREEK_NAMES)

    cfg = MCConfig(n_paths=5_000, n_steps=1, seed=1, greeks_estimator=LIKELIHOOD_RATIO)
    meta = greeks(CALL, MODEL, MARKET, method="mc", cfg=cfg).meta
    assert meta is not None
    assert set(meta["estimator"].values()) == {LIKELIHOOD_RATIO}


def test_bump_result_still_reports_a_stderr_and_names_the_backward_theta() -> None:
    """A contradicted expectation, pinned: the bump theta is **not** central.

    The slice statement described the pre-slice Greeks as "central-difference
    CRN bumps". Four of them are; theta is a *backward* difference
    `(V(T - dt) - V(T)) / dt`, which is first order in `dt` where the others are
    second order. Slice 10 reproduces it rather than silently upgrading it --
    requirement (e) is that the bump numbers do not move -- and records the
    discrepancy in `meta["fd_by_greek"]` so it is discoverable from the result
    instead of from the source.
    """
    cfg = MCConfig(n_paths=5_000, n_steps=1, seed=1)
    result = greeks(CALL, MODEL, MARKET, method="mc", cfg=cfg)
    assert result.meta is not None
    assert result.meta["fd_by_greek"]["theta"] == "backward"
    assert set(result.meta["fd_by_greek"].values()) == {"central", "backward"}
    assert all(
        math.isfinite(result.meta["stderr"][name]) for name in GREEK_NAMES
    ), result.meta["stderr"]


def test_unknown_estimator_and_misplaced_bumps_raise() -> None:
    """Evidence class: NEGATIVE_FINDING on the contract, not on the numerics."""
    with pytest.raises(InvalidInputError, match="unknown greeks_estimator"):
        greeks(
            CALL,
            MODEL,
            MARKET,
            method="mc",
            cfg=MCConfig(n_paths=100, greeks_estimator="malliavin"),  # type: ignore[arg-type]
        )
    with pytest.raises(InvalidInputError, match="bumps are meaningless"):
        greeks(
            CALL,
            MODEL,
            MARKET,
            method="mc",
            cfg=MCConfig(n_paths=100, greeks_estimator=PATHWISE),
            bumps={"spot": 1e-2},
        )


@pytest.mark.parametrize("estimator", [PATHWISE, LIKELIHOOD_RATIO])
def test_sample_estimators_refuse_the_two_degenerate_limits(estimator: str) -> None:
    """`T = 0` and `sigma = 0` make the terminal law a point mass.

    Evidence class: NEGATIVE_FINDING. The score of a point mass does not exist
    and the pathwise derivative is not integrable against it, so both limits
    raise. The analytic engine refuses the same two limits, one derivative
    lower; the bump estimator does not, and returns zeros at `T = 0`, which is
    the pre-slice behaviour and is kept.
    """
    cfg = MCConfig(n_paths=1_000, n_steps=1, seed=1, greeks_estimator=estimator)
    with pytest.raises(InvalidInputError, match="needs T > 0 and sigma > 0"):
        greeks(replace(CALL, expiry=0.0), MODEL, MARKET, method="mc", cfg=cfg)
    with pytest.raises(InvalidInputError, match="needs T > 0 and sigma > 0"):
        greeks(CALL, BlackScholesModel(sigma=0.0), MARKET, method="mc", cfg=cfg)


def test_the_evidence_classes_this_file_uses_exist() -> None:
    """Guards the docstrings above against an `EvidenceClass` rename."""
    assert {
        EvidenceClass.STATISTICAL,
        EvidenceClass.CLOSED_FORM,
        EvidenceClass.EXACT_IDENTITY,
        EvidenceClass.NEGATIVE_FINDING,
    } <= set(EvidenceClass)
