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

Greeks are refused, and `tests` below pin the refusal and its message; the
reasoning is in `qpl.engines.mc.digital`.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from qpl.engines.analytic.digital import digital_price
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


def test_greeks_are_refused_and_say_what_to_use_instead() -> None:
    """Evidence class: NEGATIVE_FINDING, encoded as an error rather than a number.

    The pathwise derivative of `cash * 1{S_T > K}` is a Dirac mass, so there is
    no pathwise estimator. A common-random-numbers bump is not a repair: the
    bumped and unbumped payoffs differ only on the `O(h)` fraction of paths
    that cross the strike, each by a full `cash`, so the difference quotient
    has variance `O(cash**2 / (N h))` and its standard deviation **grows** like
    `h**-1/2` as the bump shrinks.

    Measured: the CRN central-difference delta at `N = 40_000`, over 20 seeds,
    against a true delta of 0.018762 --

    | h     | mean     | sd across seeds | sd / true delta |
    |-------|----------|-----------------|-----------------|
    | 1     | 0.018751 | 0.00041         | 2.2%            |
    | 0.1   | 0.018781 | 0.00177         | 9.4%            |
    | 0.01  | 0.017060 | 0.00474         | 25.3%           |
    | 0.001 | 0.014859 | 0.01585         | 84.5%           |

    A factor of 1000 in `h` multiplies the noise by 38.6, against the 31.6 that
    `h**-1/2` predicts. Note the trap in the first row: a *coarse* bump is
    accurate here, because delta is smooth at this point and the `O(h**2)` bias
    is tiny. That is an accident of the point, not a method -- the estimator
    has no limit as `h -> 0`, which is what "estimating a derivative" means.

    The engine therefore refuses, naming the likelihood-ratio estimator that
    does work (Glasserman, *Monte Carlo Methods in Financial Engineering*,
    ch. 7) and is scheduled for Phase 3.
    """
    triple = _triple()
    with pytest.raises(NotSupportedError, match="likelihood-ratio"):
        greeks(*triple, method="mc", cfg=MCConfig(n_paths=10_000, seed=_SEED))

    option, model, _ = triple

    def bumped_delta(h: float, seed: int) -> float:
        def at(spot: float) -> float:
            market = Market(
                spot=spot,
                rate_curve=FlatRateCurve(_RATE),
                dividend_curve=FlatDividendCurve(_DIV),
            )
            return price(
                option,
                model,
                market,
                method="mc",
                cfg=MCConfig(n_paths=40_000, n_steps=1, seed=seed),
            ).value

        return (at(_SPOT + h) - at(_SPOT - h)) / (2.0 * h)

    seeds = range(1, 13)
    coarse = np.array([bumped_delta(0.1, s) for s in seeds])
    fine = np.array([bumped_delta(0.001, s) for s in seeds])

    coarse_sd = float(np.std(coarse, ddof=1))
    fine_sd = float(np.std(fine, ddof=1))

    # Shrinking the bump by 100x makes the estimator noisier, not sharper.
    assert fine_sd > 4.0 * coarse_sd, (coarse_sd, fine_sd)
    # And by then the noise is comparable to the quantity being estimated.
    from qpl.engines.analytic.digital import greeks_digital as _analytic

    true_delta = _analytic(*triple).delta
    assert fine_sd > 0.4 * true_delta, (fine_sd, true_delta)
    assert coarse_sd < 0.2 * true_delta, (coarse_sd, true_delta)


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
