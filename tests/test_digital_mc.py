"""Monte Carlo on a digital: the one method the discontinuity does not hurt.

The estimator is a sample proportion -- count the paths that finish in the
money, discount -- so it is unbiased at every `N` and its standard error is
`cash * e^{-rT} sqrt(p (1 - p) / N)` exactly. Nothing about the jump enters:
the payoff is bounded and square-integrable, which is all the central limit
theorem asks. That is the contrast with the tree (which must decide which side
of the strike each node is on) and the grid (which must represent a step
function in a space of grid functions).

Measured at the reference point `S = K = 100, r = 5%, q = 0, sigma = 20%,
T = 1`, call, seed 123, terminal sampling:

| N       | price    | stderr   | (price - closed form) / stderr |
|---------|----------|----------|--------------------------------|
| 5_000   | 0.534971 | 6.674e-3 | +0.397                         |
| 20_000  | 0.532641 | 3.339e-3 | +0.095                         |
| 80_000  | 0.532522 | 1.669e-3 | +0.118                         |
| 200_000 | 0.531038 | 1.056e-3 | -1.218                         |
| 320_000 | 0.532778 | 8.347e-4 | +0.543                         |

Fitted standard-error order in `N`: **0.49990**, log-space residual
**1.59e-04**.

Greeks (Slice 10). The likelihood-ratio estimator is unbiased here and covers
the closed form 37-38 times in 40 seeds; the bump is available and is measured
below to be the wrong tool, with a theta that covers **0** times in 40; the
pathwise estimator is refused and the refusal is checked by computing the
exactly-zero sample it would return. Derivations: `qpl.engines.mc.greeks`.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from qpl.engines.analytic.digital import (
    digital_price,
    greeks_digital as _analytic_greeks,
)
from qpl.engines.mc.digital import digital_payoff_derivative
from qpl.engines.mc.greeks import draw_path_sample
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.mc.processes import price_european_from_terminal
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import DigitalOption, EuropeanOption
from qpl.instruments.payoffs import call_payoff, digital_payoff
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass, fit_convergence_order

_SPOT = 100.0
_STRIKE = 100.0
_EXPIRY = 1.0
_RATE = 0.05
_DIV = 0.0
_SIGMA = 0.20
_SEED = 123

PATH_LEVELS = (5_000, 20_000, 80_000, 320_000)
"""The `N` the standard-error order is fitted on; `h = 1 / N`."""

AGREEMENT_PATHS = 200_000
"""Where the price is checked against the closed form, at 4 standard errors."""


def _triple(kind: str = "call", cash: float = 1.0, sigma: float = _SIGMA, expiry: float = _EXPIRY):
    return (
        DigitalOption(kind=kind, strike=_STRIKE, expiry=expiry, cash=cash),
        BlackScholesModel(sigma=sigma),
        Market(
            spot=_SPOT,
            rate_curve=FlatRateCurve(_RATE),
            dividend_curve=FlatDividendCurve(_DIV),
        ),
    )


def _closed_form(kind: str = "call", cash: float = 1.0) -> float:
    return digital_price(
        S=_SPOT,
        K=_STRIKE,
        T=_EXPIRY,
        r=_RATE,
        sigma=_SIGMA,
        q=_DIV,
        cash=cash,
        kind=kind,
    )


@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_agrees_with_the_closed_form_within_four_standard_errors(kind: str) -> None:
    """Evidence class: STATISTICAL.

    The tolerance is a multiple of the *reported* standard error, not an
    absolute accuracy claim: the estimator is unbiased, so the only question is
    sampling noise. Four standard errors is a two-sided false-failure rate of
    about 6e-05 for a fixed seed, and this seed is already known to land at
    -1.218 (call) and +1.218 (put) at `N = 200_000` -- the two are exact
    mirrors, since every path is in the money for exactly one of them.
    """
    assert EvidenceClass.STATISTICAL.value == "statistical"

    result = price(
        *_triple(kind),
        method="mc",
        cfg=MCConfig(n_paths=AGREEMENT_PATHS, n_steps=1, seed=_SEED),
    )
    assert result.stderr is not None and result.stderr > 0.0
    z = (result.value - _closed_form(kind)) / result.stderr
    assert abs(z) <= 4.0, (kind, result.value, result.stderr, z)
    # Not a vacuous test: the standard error is small enough that 4 of them is
    # still a tight band on the price.
    assert 4.0 * result.stderr < 0.01


def test_standard_error_falls_at_order_one_half_in_the_path_count() -> None:
    """Evidence class: CONVERGENCE_ORDER, on the standard error rather than an error.

    Fitting `log(stderr)` against `log(1/N)` over `N` in
    (5_000, 20_000, 80_000, 320_000) gives order **0.49990** with log-space
    residual **1.59e-04**. That is far cleaner than any discretisation fit in
    this repository, and it should be: the standard error is not a
    discretisation error at all but `sigma_payoff / sqrt(N)` with a
    `sigma_payoff` that barely moves between levels, so the only departure from
    an exact power law is the sampling variation of the sample standard
    deviation itself.
    """
    stderrs = []
    for n_paths in PATH_LEVELS:
        result = price(
            *_triple(),
            method="mc",
            cfg=MCConfig(n_paths=n_paths, n_steps=1, seed=_SEED),
        )
        assert result.stderr is not None
        stderrs.append(result.stderr)

    fit = fit_convergence_order([1.0 / n for n in PATH_LEVELS], stderrs)
    assert abs(fit.order - 0.5) <= 0.02, (fit.order, stderrs)
    assert fit.residual < 0.01, fit.residual
    assert all(a > b for a, b in pairwise(stderrs)), stderrs


def test_the_reported_standard_error_is_the_bernoulli_one() -> None:
    """Evidence class: CLOSED_FORM.

    A discounted indicator is `cash * e^{-rT}` times a Bernoulli variable, so
    its population standard error is `cash e^{-rT} sqrt(p (1-p) / N)` with
    `p = N(d2)` the closed-form in-the-money probability. The engine computes
    the *sample* standard deviation with `ddof=1` and knows nothing about that
    formula, so agreement is a check on the estimator rather than a
    restatement of it. At `N = 320_000` the two agree to 0.06%.
    """
    result = price(
        *_triple(), method="mc", cfg=MCConfig(n_paths=320_000, n_steps=1, seed=_SEED)
    )
    assert result.stderr is not None

    discount = math.exp(-_RATE * _EXPIRY)
    probability = _closed_form() / discount
    predicted = discount * math.sqrt(probability * (1.0 - probability) / 320_000)
    assert result.stderr == pytest.approx(predicted, rel=5e-3)


def test_call_plus_put_is_exact_path_by_path() -> None:
    """Evidence class: EXACT_IDENTITY, and it holds sample by sample.

    Every simulated path finishes strictly on one side of the strike (landing
    exactly on it has probability zero and, on a continuous distribution, never
    happened in any of these runs), so the two estimators' payoffs sum to
    `cash` on every path and their prices sum to `cash e^{-rT}` with **no**
    sampling error at all -- the identity survives the estimator, not just the
    model.
    """
    cash = 2.5
    cfg = MCConfig(n_paths=AGREEMENT_PATHS, n_steps=1, seed=_SEED)
    call = price(*_triple("call", cash), method="mc", cfg=cfg).value
    put = price(*_triple("put", cash), method="mc", cfg=cfg).value
    assert call + put == pytest.approx(cash * math.exp(-_RATE * _EXPIRY), abs=1e-14)


GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")
GREEKS_SEEDS = tuple(range(1000, 1040))
GREEKS_PATHS = 20_000
GREEKS_MIN_COVERAGE = 34
"""Binomial(40, 0.95) tail bound; see `tests/test_mc_greeks.py`."""

Z_95 = 1.959963984540054


def _greek_coverage(estimator: str) -> dict[str, int]:
    triple = _triple()
    exact = _analytic_greeks(*triple)
    hits = dict.fromkeys(GREEK_NAMES, 0)
    for seed in GREEKS_SEEDS:
        cfg = MCConfig(
            n_paths=GREEKS_PATHS, n_steps=1, seed=seed, greeks_estimator=estimator
        )
        result = greeks(*triple, method="mc", cfg=cfg)
        assert result.meta is not None
        for name in GREEK_NAMES:
            if abs(getattr(result, name) - getattr(exact, name)) <= (
                Z_95 * result.meta["stderr"][name]
            ):
                hits[name] += 1
    return hits


def test_the_pathwise_delta_of_a_digital_is_exactly_zero_and_is_refused() -> None:
    """Evidence class: NEGATIVE_FINDING, and the finding is a *number*.

    A cash-or-nothing payoff is locally constant everywhere except at the
    strike, so the almost-everywhere derivative a program can evaluate is
    **identically zero** -- not noisy, not inaccurate, `0.0` at every path
    count and every seed. The pathwise interchange of derivative and
    expectation is therefore invalid here in the strongest possible way: it
    converges, and it converges to the wrong number.

    This test computes the zero rather than taking the refusal on trust: it
    builds the estimator by hand from the public pieces -- the same sampler the
    engine uses, the digital payoff's a.e. derivative, and the pathwise delta
    weight `e^{-rT} f'(S_T) S_T / S_0` -- and asserts the whole sample is zero
    while the true delta is not. Then it asserts the engine refuses, so that
    the zero can never be returned as a Greek.
    """
    option, model, market = _triple()
    t = option.expiry
    r, q = market.rate(t), market.dividend_yield(t)
    sample = draw_path_sample(
        s0=market.spot,
        mu=r - q,
        sigma=model.sigma,
        t=t,
        n_paths=20_000,
        n_steps=1,
        seed=_SEED,
        methods=(),
        n_strata=64,
    )
    derivative = digital_payoff_derivative(
        sample.spots, strike=option.strike, cash=option.cash, kind=option.kind
    )
    assert np.count_nonzero(derivative) == 0
    pathwise_delta = float(
        np.mean(market.df_r(t) * derivative * sample.spots / market.spot)
    )
    assert pathwise_delta == 0.0

    true_delta = _analytic_greeks(option, model, market).delta
    assert true_delta > 0.01, true_delta

    with pytest.raises(NotSupportedError, match="biased to exactly 0.0"):
        greeks(
            option,
            model,
            market,
            method="mc",
            cfg=MCConfig(n_paths=10_000, seed=_SEED, greeks_estimator="pathwise"),
        )


def test_likelihood_ratio_greeks_cover_the_closed_form_over_40_seeds() -> None:
    """Evidence class: STATISTICAL against a CLOSED_FORM reference.

    The estimator the Slice 6 refusal message pointed at, delivered. It
    differentiates the lognormal density instead of the payoff, so the jump is
    irrelevant: all five Greeks come from one sample and all five are unbiased.

    Measured hit counts out of 40 at 20 000 paths, ATM call:
    delta 38, gamma 37, vega 37, theta 38, rho 38 -- against a
    Binomial(40, 0.95) mean of 38.
    """
    hits = _greek_coverage("likelihood_ratio")
    for name in GREEK_NAMES:
        assert hits[name] >= GREEKS_MIN_COVERAGE, (name, hits)


def test_the_bump_is_available_and_measurably_the_wrong_tool() -> None:
    """Evidence class: NEGATIVE_FINDING, measured against the LR column.

    Slice 10 makes `greeks_estimator="bump"` work on a digital -- it is the
    package-wide default and refusing it per instrument would be a surprise --
    and this test records what it is worth. Same 40 seeds, same 20 000 paths,
    same point as the likelihood-ratio test above:

        Greek   bump coverage   LR coverage   bump sd / |analytic Greek|
        delta       36/40          38/40           0.34
        gamma       39/40          37/40        4665.55
        vega        36/40          37/40           0.67
        rho         26/40          38/40           1.35
        theta        0/40          38/40           0.03

    Three separate failures are visible in that table and they are different
    failures.

    - **theta at 0/40** is the sharpest. The estimator is not biased --
      `E[(Y(T - dt) - Y(T))/dt]` is the exact difference quotient -- but with
      `dt = 1e-4` the probability that a given path crosses the strike when the
      maturity moves is about `2e-06`, so in a 20 000-path run *no* path
      crosses, the sample is the discount factor's smooth `r V` part alone, and
      both the estimate and its standard error describe that part. The missing
      term is the whole density contribution. The estimator's true standard
      deviation is dominated by an event that does not occur in 40 runs, and its
      reported standard error is not an error bar for it. `rho` at 26/40 is the
      same mechanism one step less extreme (`dr = 1e-5`).
    - **gamma at 39/40** is the opposite failure and passes for the wrong
      reason: the second difference has standard deviation 4665 times the
      Greek, so the nominal interval is so wide that covering is trivial. A
      coverage test alone cannot see this; the ratio column is what sees it.
    - **delta at 36/40** is the honest case. With `h = 0.01` enough paths cross
      that the estimator behaves, and it is merely 34% noise.

    The engine records all of this in `meta["estimator_caveat"]`, so it is
    discoverable from a result rather than only from this file.
    """
    bump = _greek_coverage("bump")
    lr = _greek_coverage("likelihood_ratio")
    assert bump["theta"] <= 5, bump
    assert lr["theta"] >= GREEKS_MIN_COVERAGE, lr
    assert bump["rho"] < lr["rho"], (bump, lr)

    exact = _analytic_greeks(*_triple())
    spreads = {}
    for name in GREEK_NAMES:
        values = [
            getattr(
                greeks(
                    *_triple(),
                    method="mc",
                    cfg=MCConfig(
                        n_paths=GREEKS_PATHS,
                        n_steps=1,
                        seed=seed,
                        greeks_estimator="bump",
                    ),
                ),
                name,
            )
            for seed in GREEKS_SEEDS
        ]
        spreads[name] = float(np.std(values, ddof=1)) / abs(getattr(exact, name))
    # The bumped gamma's spread is three decimal orders larger than the Greek.
    assert spreads["gamma"] > 1_000.0, spreads
    # And delta, the one that behaves, is inside 50% relative noise.
    assert spreads["delta"] < 0.5, spreads

    meta = greeks(
        *_triple(),
        method="mc",
        cfg=MCConfig(n_paths=1_000, seed=_SEED, greeks_estimator="bump"),
    ).meta
    assert meta is not None
    assert "h**-1/2" in meta["estimator_caveat"]


def test_lr_beats_the_bump_on_delta_at_equal_cost() -> None:
    """Evidence class: STATISTICAL. The comparison the refusal message implied.

    At equal normal draws (20 000 paths, one normal each, both estimators) the
    standard deviation of the delta estimate over 40 seeds is 2.098e-04 for the
    likelihood ratio and 6.383e-03 for the bump -- a factor of **30.4** in
    standard deviation and **925** in variance, in favour of the estimator that
    does not touch the payoff. On a vanilla call the ordering is the other way
    round (`tests/test_mc_greeks_variance.py`), which is Glasserman's rule in
    both directions: pathwise/bump when the payoff is smooth, likelihood ratio
    when it is not.
    """
    triple = _triple()

    def _spread(estimator: str) -> float:
        values = [
            greeks(
                *triple,
                method="mc",
                cfg=MCConfig(
                    n_paths=GREEKS_PATHS,
                    n_steps=1,
                    seed=seed,
                    greeks_estimator=estimator,
                ),
            ).delta
            for seed in GREEKS_SEEDS
        ]
        return float(np.std(values, ddof=1))

    lr_sd = _spread("likelihood_ratio")
    bump_sd = _spread("bump")
    assert bump_sd > 10.0 * lr_sd, (lr_sd, bump_sd)


def test_digital_greeks_refuse_the_two_degenerate_limits() -> None:
    """Evidence class: NEGATIVE_FINDING. Both limits make the price a step."""
    for expiry, sigma in ((0.0, _SIGMA), (_EXPIRY, 0.0)):
        option = DigitalOption(kind="call", strike=_STRIKE, expiry=expiry)
        with pytest.raises(InvalidInputError, match="need T > 0 and sigma > 0"):
            greeks(
                option,
                BlackScholesModel(sigma=sigma),
                Market(
                    spot=_SPOT,
                    rate_curve=FlatRateCurve(_RATE),
                    dividend_curve=FlatDividendCurve(_DIV),
                ),
                method="mc",
                cfg=MCConfig(n_paths=1_000, seed=_SEED),
            )


def test_price_is_deterministic_for_a_fixed_seed_and_moves_with_it() -> None:
    triple = _triple()
    cfg = MCConfig(n_paths=20_000, n_steps=1, seed=_SEED)
    first = price(*triple, method="mc", cfg=cfg)
    second = price(*triple, method="mc", cfg=cfg)
    assert first.value == second.value
    assert first.stderr == second.stderr

    other = price(*triple, method="mc", cfg=MCConfig(n_paths=20_000, n_steps=1, seed=999))
    assert other.value != first.value


@pytest.mark.parametrize("n_steps", [1, 4])
def test_multi_step_paths_price_the_same_contract(n_steps: int) -> None:
    """Only the endpoint matters, so more steps is the same law and more work.

    The two estimates differ because they consume different random numbers, not
    because they price different things; they agree well inside their combined
    standard error.
    """
    result = price(
        *_triple(),
        method="mc",
        cfg=MCConfig(n_paths=80_000, n_steps=n_steps, seed=_SEED),
    )
    assert result.stderr is not None
    assert abs(result.value - _closed_form()) <= 4.0 * result.stderr


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("spot", [95.0, 105.0])
def test_degenerate_limits_have_zero_standard_error(kind: str, spot: float) -> None:
    """Evidence class: CLOSED_FORM. No sampling happens at `T = 0` or `sigma = 0`."""
    for expiry, sigma in ((0.0, _SIGMA), (_EXPIRY, 0.0)):
        option = DigitalOption(kind=kind, strike=_STRIKE, expiry=expiry, cash=3.0)
        model = BlackScholesModel(sigma=sigma)
        market = Market(
            spot=spot,
            rate_curve=FlatRateCurve(_RATE),
            dividend_curve=FlatDividendCurve(0.02),
        )
        result = price(option, model, market, method="mc", cfg=MCConfig(n_paths=10, seed=_SEED))
        assert result.stderr == 0.0
        assert result.value == price(option, model, market).value


def test_config_validation() -> None:
    triple = _triple()
    with pytest.raises(InvalidInputError, match="n_paths must be >= 2"):
        price(*triple, method="mc", cfg=MCConfig(n_paths=1))
    with pytest.raises(InvalidInputError, match="n_steps must be >= 1"):
        price(*triple, method="mc", cfg=MCConfig(n_steps=0))


def test_the_vanilla_estimator_is_unchanged_by_the_payoff_hook() -> None:
    """`price_european_from_terminal` gained an optional `payoff` callable.

    Passing the vanilla payoff explicitly must reproduce the default branch
    bit-for-bit, and the existing vanilla MC prices must not have moved; the
    second half is covered by the pinned values in `tests/test_mc_pricing.py`,
    and the first is asserted here on both the value and the standard error.
    """
    rng = np.random.default_rng(7)
    terminal = 100.0 * np.exp(rng.normal(size=5_000) * 0.2)

    default = price_european_from_terminal(
        terminal, strike=100.0, discount_factor=0.95, kind="call"
    )
    explicit = price_european_from_terminal(
        terminal,
        strike=100.0,
        discount_factor=0.95,
        kind="call",
        payoff=lambda s: np.asarray(call_payoff(s, 100.0), dtype=float),
    )
    assert default.value == explicit.value
    assert default.stderr == explicit.stderr

    digital = price_european_from_terminal(
        terminal,
        strike=100.0,
        discount_factor=0.95,
        kind="call",
        payoff=lambda s: np.asarray(digital_payoff(s, 100.0, 1.0, "call"), dtype=float),
    )
    assert digital.value != default.value


def test_a_vanilla_still_gets_the_vanilla_mc_engine() -> None:
    """The registry keys on the instrument type, so nothing here leaked."""
    vanilla = EuropeanOption(kind="call", strike=_STRIKE, expiry=_EXPIRY)
    model = BlackScholesModel(sigma=_SIGMA)
    market = Market(
        spot=_SPOT,
        rate_curve=FlatRateCurve(_RATE),
        dividend_curve=FlatDividendCurve(_DIV),
    )
    result = price(vanilla, model, market, method="mc", cfg=MCConfig(n_paths=10_000, seed=_SEED))
    assert result.meta is not None
    assert "instrument" not in result.meta
    # And its Greeks still work, unlike the digital's.
    assert greeks(
        vanilla, model, market, method="mc", cfg=MCConfig(n_paths=10_000, seed=_SEED)
    ).delta > 0.0
