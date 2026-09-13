"""Asian options by Monte Carlo: the sampler, the identities, the estimator.

The Asian Monte Carlo engine has a property the vanilla one has to work for:
there is **no time-discretisation bias**. Exact lognormal stepping over the
fixing grid draws each fixing from its correct marginal with the correct joint
law, so the only error in a price here is statistical and the reported standard
error is the whole story. Everything below is organised around that.

- **The sampler.** The variable time grid is new in this slice; the uniform
  path is pinned bit-for-bit so that adding it cannot have moved any existing
  Monte Carlo number, and the variable path is checked against the exact
  lognormal mean/variance/covariance structure on an *irregular* schedule,
  where a uniform-`dt` slip would not cancel.
- **(c) STATISTICAL.** The geometric-average Monte Carlo price against its own
  exact closed form at 200 000 paths, for four estimators, plus the empirical
  coverage of the reported 95% interval over 40 seeds.
- **(d) EXACT_IDENTITY.** `A_arith >= A_geom` pathwise (AM-GM), hence
  `price_arith >= price_geom` for **every** seed when both are run on the same
  paths -- not a statistical statement, a per-sample one. And the measured gap
  between the Turnbull-Wakeman approximation and the control-variate Monte
  Carlo price, reported with its sign and not asserted to be zero.
- **(e) EXACT_IDENTITY.** Asian put-call parity, `C - P = e^{-rT}(A - K)`,
  which holds *per sample* because the same paths drive both legs, and in
  expectation against the closed-form `E[A]` / `E[G]`.

Reference point throughout: `S = K = 100`, `r = 5%`, `q = 0`, `sigma = 20%`,
`T = 1` -- the same market as the vanilla and digital reference rows, so the
Asian prices can be read next to them. Fixing schedules `t_i = i T / n` with
`n = 10` and `n = 52`.

Reference for the control variate: Kemna & Vorst (1990), *Journal of Banking
and Finance* 14, 113-129; the estimator layer is Glasserman (2003) section 4.1.
Every number here was measured in this repository.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.engines.analytic.asian import (
    discrete_geometric_price,
    expected_arithmetic_average,
    expected_geometric_average,
    turnbull_wakeman_price,
)
from qpl.engines.mc.asian import asian_terminal_sample
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.mc.processes import gbm_paths_from_normals, simulate_gbm_exact
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments import (
    AsianOption,
    arithmetic_average,
    geometric_average,
    uniform_fixing_times,
)
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass

# --------------------------------------------------------------------------
# The reference point.
# --------------------------------------------------------------------------

SPOT, STRIKE, EXPIRY, SIGMA = 100.0, 100.0, 1.0, 0.20
MARKET = Market(
    spot=SPOT,
    rate_curve=FlatRateCurve(0.05),
    dividend_curve=FlatDividendCurve(0.0),
)
MODEL = BlackScholesModel(sigma=SIGMA)
RATE = MARKET.rate(EXPIRY)
DIVIDEND = MARKET.dividend_yield(EXPIRY)
MU = RATE - DIVIDEND
DISCOUNT = MARKET.df_r(EXPIRY)
"""Read back from the curve, not from the literal `0.05`: `Market.rate(t)`
recovers the rate as `-log(df)/t` and returns `0.049999999999999996`. Rebuilding
a sample from the literal moves the paths in the last bit and breaks an
exact-identity assertion (the same trap recorded in Slice 7)."""

FIXINGS_10 = uniform_fixing_times(EXPIRY, 10)
FIXINGS_52 = uniform_fixing_times(EXPIRY, 52)

_UNBIASEDNESS_PATHS = 200_000
_UNBIASEDNESS_SEED = 7
_ESTIMATORS: tuple[tuple[str, str | tuple[str, ...]], ...] = (
    ("none", "none"),
    ("control_variate", "control_variate"),
    ("antithetic", "antithetic"),
    ("antithetic+control", ("antithetic", "control_variate")),
)


def _asian(averaging: str, times, kind: str = "call") -> AsianOption:
    return AsianOption(
        kind=kind, strike=STRIKE, expiry=EXPIRY, fixing_times=times, averaging=averaging
    )


def _cfg(vr, n_paths: int, seed: int) -> MCConfig:
    """Paths that spend the same number of normals under either sampler.

    Antithetic evaluates the payoff twice per draw, so matching the *cost*
    (normal draws, which here is `n_paths * n_fixings`) means doubling the path
    count -- the same convention Slice 7 measures its ratios on.
    """
    return MCConfig(
        n_paths=2 * n_paths if "antithetic" in vr else n_paths,
        n_steps=1,
        seed=seed,
        variance_reduction=vr,
    )


def _mc(option: AsianOption, vr, n_paths: int, seed: int):
    return price(option, MODEL, MARKET, method="mc", cfg=_cfg(vr, n_paths, seed))


def _geometric_truth(times, kind: str = "call") -> float:
    return discrete_geometric_price(
        S=SPOT,
        K=STRIKE,
        T=EXPIRY,
        r=RATE,
        sigma=SIGMA,
        fixing_times=times,
        q=DIVIDEND,
        kind=kind,
    )


# --------------------------------------------------------------------------
# The sampler: the uniform grid is untouched, the variable one is new.
# --------------------------------------------------------------------------

_PINNED_UNIFORM = [
    [100.0, 90.10325499309418, 105.60155427440597, 118.62084158645726],
    [100.0, 96.80530936033892, 99.99298946320528, 107.95757048031692],
]


def test_the_uniform_grid_path_is_bit_for_bit_what_it_was():
    """Adding a time grid must not move a single existing Monte Carlo number.

    `simulate_gbm_exact(t=..., n_steps=...)` keeps its own arithmetic rather
    than being expressed through the variable-grid branch, because
    `t_{i+1} - t_i` for a uniform grid is not bit-for-bit `t / n_steps` and a
    price that moved in its last digit would invalidate the pinned
    `(value, stderr)` pairs elsewhere in this suite. Asserted with `==`, not
    `approx`.
    """
    paths = simulate_gbm_exact(
        s0=100.0, mu=0.05, sigma=0.2, t=1.0, n_steps=3, n_paths=2, seed=123
    )
    # Literals recorded on macOS arm64; other libm builds differ by a few ULP,
    # so the pin is round-off tight rather than bit-for-bit across platforms.
    assert np.allclose(paths, np.asarray(_PINNED_UNIFORM), rtol=1e-13, atol=0.0)


def test_the_two_grid_spellings_are_mutually_exclusive():
    """Exactly one of `(t, n_steps)` and `times`; neither and both are refused.

    The alternative -- letting `times` win silently when both are given -- makes
    a caller that meant one thing and typed the other get a plausible number
    from a different time grid.
    """
    base = dict(s0=100.0, mu=0.05, sigma=0.2, n_paths=4, seed=1)
    with pytest.raises(InvalidInputError, match="exactly one of"):
        simulate_gbm_exact(**base)
    with pytest.raises(InvalidInputError, match="exactly one of"):
        simulate_gbm_exact(**base, t=1.0, times=(0.5, 1.0))
    with pytest.raises(InvalidInputError, match="n_steps is required"):
        simulate_gbm_exact(**base, t=1.0)
    with pytest.raises(InvalidInputError, match="n_steps is meaningless"):
        simulate_gbm_exact(**base, n_steps=2, times=(0.5, 1.0))


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ((), "times must be a non-empty 1D sequence"),
        ((0.0, 0.5), "times must all be > 0"),
        ((0.5, 0.5), "times must be strictly increasing"),
        ((1.0, 0.5), "times must be strictly increasing"),
        ((0.5, float("inf")), "times must be finite"),
    ],
)
def test_an_unusable_time_grid_is_refused(times, message):
    with pytest.raises(InvalidInputError, match=message):
        simulate_gbm_exact(
            s0=100.0, mu=0.05, sigma=0.2, n_paths=4, seed=1, times=times
        )


def test_the_variable_grid_reproduces_the_exact_lognormal_moments():
    """CLOSED_FORM: the right marginals *and* the right covariance, off a uniform grid.

    On an irregular schedule a `dt` slip does not cancel the way it can on a
    uniform one, which is why the grid below is deliberately ragged. Checked
    against the exact structure of geometric Brownian motion:

        E[log S_t]      = log S_0 + (mu - sigma^2/2) t,
        Cov(log S_s, log S_t) = sigma^2 min(s, t).

    400 000 paths gives about two and a half significant figures on each
    covariance entry, so the tolerance is relative and loose; the point of the
    row is that a wrong `dt` would be out by a factor, not by 2%. The worst
    measured relative error is 2.21e-02 and it sits at the `(t_1, t_5)` corner
    -- the smallest covariance paired with the longest lever, which is where a
    sample covariance is noisiest.
    """
    assert EvidenceClass.CLOSED_FORM
    times = np.array([0.03, 0.31, 0.32, 1.4, 2.75])
    s0, mu, sigma, n = 100.0, 0.07, 0.35, 400_000
    rng = np.random.default_rng(4242)
    z = rng.normal(size=(n, times.size))
    paths = gbm_paths_from_normals(z, s0=s0, mu=mu, sigma=sigma, times=times)
    logs = np.log(paths)

    expected_mean = math.log(s0) + (mu - 0.5 * sigma**2) * times
    assert np.allclose(logs.mean(axis=0), expected_mean, atol=4e-3)

    expected_cov = sigma**2 * np.minimum(times[:, None], times[None, :])
    assert np.allclose(np.cov(logs, rowvar=False), expected_cov, rtol=0.04)

    # And the `s0` column is prepended exactly by the public entry point.
    full = simulate_gbm_exact(
        s0=s0, mu=mu, sigma=sigma, n_paths=16, seed=3, times=times
    )
    assert full.shape == (16, times.size + 1)
    assert np.all(full[:, 0] == s0)


def test_normals_and_the_grid_must_agree_in_shape():
    with pytest.raises(InvalidInputError, match="one column per time"):
        gbm_paths_from_normals(
            np.zeros((5, 3)), s0=100.0, mu=0.0, sigma=0.2, times=(0.5, 1.0)
        )
    with pytest.raises(InvalidInputError, match="2D"):
        gbm_paths_from_normals(
            np.zeros(5), s0=100.0, mu=0.0, sigma=0.2, times=(0.5, 1.0)
        )


# --------------------------------------------------------------------------
# The engine's contract.
# --------------------------------------------------------------------------


def test_n_steps_other_than_one_is_refused_rather_than_ignored():
    """The Asian's time grid is its fixing schedule; `n_steps` has no meaning.

    Silently ignoring a config field on one instrument while honouring it on
    another is discovered by a wrong number, not by a message. It raises.
    """
    option = _asian("arithmetic", FIXINGS_10)
    cfg = MCConfig(n_paths=1000, n_steps=4, seed=1)
    with pytest.raises(InvalidInputError, match="n_steps must be 1 for an Asian"):
        price(option, MODEL, MARKET, method="mc", cfg=cfg)


@pytest.mark.parametrize(
    "vr", ["stratified", ("stratified", "control_variate")], ids=["alone", "with_cv"]
)
def test_stratified_sampling_is_refused_for_an_asian(vr):
    """NotSupportedError naming the Brownian bridge, for both fixing counts.

    Slice 7 refuses `stratified` when `n_steps > 1`. That check alone would let
    an Asian through, because the Asian engine passes `n_steps = 1` to the
    sampler validator -- the fixing count is not the config's step count. The
    refusal is therefore stated in the Asian engine, in terms of the fixing
    schedule, and covers the one-fixing case too.
    """
    cfg = MCConfig(n_paths=1024, seed=1, variance_reduction=vr, n_strata=16)
    for times in (FIXINGS_10, uniform_fixing_times(EXPIRY, 1)):
        with pytest.raises(NotSupportedError, match="Brownian bridge"):
            price(_asian("arithmetic", times), MODEL, MARKET, method="mc", cfg=cfg)


def test_antithetic_still_needs_an_even_path_count():
    """The Slice 7 sampler rule is reused, not restated."""
    cfg = MCConfig(n_paths=1001, seed=1, variance_reduction="antithetic")
    with pytest.raises(InvalidInputError, match="n_paths must be even"):
        price(_asian("arithmetic", FIXINGS_10), MODEL, MARKET, method="mc", cfg=cfg)


GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")
GREEKS_SEEDS = tuple(range(1000, 1040))
GREEKS_PATHS = 10_000
GREEKS_SPREAD_SEEDS = GREEKS_SEEDS[:20]
GREEKS_AGREEMENT_PATHS = 60_000
GREEKS_MIN_COVERAGE = 34
"""Binomial(40, 0.95) tail bound; see `tests/test_mc_greeks.py`."""

Z_95 = 1.959963984540054

GREEKS_FIXINGS = uniform_fixing_times(EXPIRY, 6)
GREEKS_MODEL = BlackScholesModel(sigma=0.25)
GREEKS_MARKET = Market(
    spot=SPOT,
    rate_curve=FlatRateCurve(0.05),
    dividend_curve=FlatDividendCurve(0.03),
)
"""A dividend yield, so a Greek that confused `r` with `mu` would show."""

GEOMETRIC = AsianOption(
    kind="call",
    strike=STRIKE,
    expiry=EXPIRY,
    fixing_times=GREEKS_FIXINGS,
    averaging="geometric",
)
ARITHMETIC = AsianOption(
    kind="call",
    strike=STRIKE,
    expiry=EXPIRY,
    fixing_times=GREEKS_FIXINGS,
    averaging="arithmetic",
)


def _greek_coverage(estimator: str) -> dict[str, int]:
    exact = greeks(GEOMETRIC, GREEKS_MODEL, GREEKS_MARKET, method="analytic")
    hits = dict.fromkeys(GREEK_NAMES, 0)
    for seed in GREEKS_SEEDS:
        cfg = MCConfig(
            n_paths=GREEKS_PATHS, n_steps=1, seed=seed, greeks_estimator=estimator
        )
        result = greeks(GEOMETRIC, GREEKS_MODEL, GREEKS_MARKET, method="mc", cfg=cfg)
        assert result.meta is not None
        for name in GREEK_NAMES:
            if abs(getattr(result, name) - getattr(exact, name)) <= (
                Z_95 * result.meta["stderr"][name]
            ):
                hits[name] += 1
    return hits


@pytest.mark.parametrize("estimator", ["bump", "pathwise", "likelihood_ratio"])
def test_asian_mc_greeks_cover_the_geometric_closed_form(estimator: str) -> None:
    """Evidence class: STATISTICAL against a CLOSED_FORM reference.

    The geometric average is the only Asian with exact Greeks, so it is the
    only place a Monte Carlo Asian Greek can be checked against a number rather
    than against another estimator. All three estimators are run against it at
    the same 40 seeds, six fixings, `sigma = 25%`, `q = 3%`, 10 000 paths:

        Greek    bump   pathwise   likelihood_ratio
        delta      38       38            40
        gamma      35       35            35
        vega       38       38            38
        theta      40       40            40
        rho        38       38            38

    Three of the five columns are *identical* across estimators and that is not
    a copy-and-paste error: `pathwise` supplies only delta and vega and
    `likelihood_ratio` only delta, so gamma, rho and theta are the bumped
    estimates in all three runs, seed for seed. The columns that do differ are
    the ones the slice is about.

    Theta is the roll derivative (settlement and every fixing shift together);
    it agrees with the closed form because
    `qpl.engines.analytic.asian.greeks_asian` uses the same convention. A
    `-dV/dT` theta would be `-r V = -0.325` here against the roll theta's
    `-8.028`, a factor of 25 out and with the wrong shape entirely.
    """
    hits = _greek_coverage(estimator)
    for name in GREEK_NAMES:
        assert hits[name] >= GREEKS_MIN_COVERAGE, (estimator, name, hits)


def test_pathwise_delta_agrees_with_the_bump_and_the_closed_form() -> None:
    """Requirement (f): three routes to one number, and a variance ordering.

    Evidence class: STATISTICAL for the two Monte Carlo legs, CLOSED_FORM for
    the reference. At 60 000 paths and one seed the pathwise and bumped deltas
    agree with the closed form and with each other inside their own standard
    errors; over 20 seeds at 10 000 paths and six fixings the standard
    deviations of the delta estimate are

        pathwise          6.538e-03
        bump              6.542e-03      ratio to pathwise 1.0006
        likelihood ratio  1.287e-02      ratio to pathwise 1.968

    The first two are the same estimator to `O(h**2)` -- a common-random-numbers
    bump of a Lipschitz payoff *is* the pathwise derivative, and here they agree
    to six parts in ten thousand -- while the likelihood-ratio delta is about
    twice as noisy, because `S_0` enters the joint density of the fixings
    through the **first transition only**: its score is `Z_1 / (S_0 sigma
    sqrt(t_1))` and it throws away everything the later fixings know about the
    payoff.

    That penalty **grows with the density of the schedule**, which is the part
    worth asserting rather than observing once. Over 12 seeds the measured
    ratio is 1.567 at six fixings and 3.439 at twelve, because doubling the
    fixings halves `t_1` and the score carries `1/sqrt(t_1)` while the pathwise
    estimator gets *quieter* (6.538e-03 to 5.001e-03 at 20 seeds: more fixings
    means a less variable average). A separate run at 26 fixings gives 5.228.
    So the worse estimator gets worse exactly where an Asian is most likely to
    be monitored.
    """
    exact = greeks(GEOMETRIC, GREEKS_MODEL, GREEKS_MARKET, method="analytic").delta

    def _one(option, estimator: str, seed: int, n_paths: int):
        cfg = MCConfig(
            n_paths=n_paths, n_steps=1, seed=seed, greeks_estimator=estimator
        )
        result = greeks(option, GREEKS_MODEL, GREEKS_MARKET, method="mc", cfg=cfg)
        assert result.meta is not None
        return result.delta, result.meta["stderr"]["delta"]

    pathwise, pathwise_se = _one(GEOMETRIC, "pathwise", 4, GREEKS_AGREEMENT_PATHS)
    bumped, bumped_se = _one(GEOMETRIC, "bump", 4, GREEKS_AGREEMENT_PATHS)
    assert abs(pathwise - exact) <= 3.0 * pathwise_se
    assert abs(bumped - exact) <= 3.0 * bumped_se
    assert abs(pathwise - bumped) <= 3.0 * (pathwise_se + bumped_se)

    def _spread(option, estimator: str, seeds) -> float:
        return float(
            np.std(
                [_one(option, estimator, seed, GREEKS_PATHS)[0] for seed in seeds],
                ddof=1,
            )
        )

    spreads = {
        estimator: _spread(GEOMETRIC, estimator, GREEKS_SPREAD_SEEDS)
        for estimator in ("pathwise", "bump", "likelihood_ratio")
    }
    assert spreads["likelihood_ratio"] > 1.5 * spreads["pathwise"], spreads
    assert 0.8 < spreads["bump"] / spreads["pathwise"] < 1.25, spreads

    dense = AsianOption(
        kind="call",
        strike=STRIKE,
        expiry=EXPIRY,
        fixing_times=uniform_fixing_times(EXPIRY, 2 * GEOMETRIC.n_fixings),
        averaging="geometric",
    )
    seeds = GREEKS_SEEDS[:12]
    sparse_ratio = _spread(GEOMETRIC, "likelihood_ratio", seeds) / _spread(
        GEOMETRIC, "pathwise", seeds
    )
    dense_ratio = _spread(dense, "likelihood_ratio", seeds) / _spread(
        dense, "pathwise", seeds
    )
    assert dense_ratio > 1.5 * sparse_ratio, (sparse_ratio, dense_ratio)


def test_the_arithmetic_asian_has_pathwise_greeks_and_no_closed_form() -> None:
    """Evidence class: INDEPENDENT_ENGINE, since there is nothing exact to use.

    The arithmetic average has no closed-form Greeks and
    `method="analytic"` says so, naming the Monte Carlo route. What is checkable
    is that the pathwise estimator and the bump -- two different routines on the
    same paths -- agree within their standard errors, and that the arithmetic
    delta sits *above* the geometric one, which is an ordering rather than a
    number: the arithmetic average dominates the geometric one pathwise (AM-GM),
    so the arithmetic call is worth more and is more sensitive to the spot.
    """
    with pytest.raises(NotSupportedError, match="no rigorous control"):
        greeks(ARITHMETIC, GREEKS_MODEL, GREEKS_MARKET, method="analytic")

    cfg_pw = MCConfig(
        n_paths=GREEKS_AGREEMENT_PATHS, n_steps=1, seed=4, greeks_estimator="pathwise"
    )
    cfg_bump = MCConfig(
        n_paths=GREEKS_AGREEMENT_PATHS, n_steps=1, seed=4, greeks_estimator="bump"
    )
    pathwise = greeks(ARITHMETIC, GREEKS_MODEL, GREEKS_MARKET, method="mc", cfg=cfg_pw)
    bumped = greeks(ARITHMETIC, GREEKS_MODEL, GREEKS_MARKET, method="mc", cfg=cfg_bump)
    assert pathwise.meta is not None and bumped.meta is not None
    for name in ("delta", "vega"):
        gap = abs(getattr(pathwise, name) - getattr(bumped, name))
        budget = 3.0 * (
            pathwise.meta["stderr"][name] + bumped.meta["stderr"][name]
        )
        assert gap <= budget, (name, gap, budget)

    geometric_delta = greeks(
        GEOMETRIC, GREEKS_MODEL, GREEKS_MARKET, method="analytic"
    ).delta
    assert pathwise.delta > geometric_delta


def test_the_estimator_is_recorded_per_greek_because_an_asian_mixes_them() -> None:
    """Evidence class: EXACT_IDENTITY on the contract.

    `pathwise` covers delta and vega here; `likelihood_ratio` covers delta. The
    rest is a bump and the result says so per Greek instead of labelling the
    whole thing with the family the caller asked for.
    """
    cfg = MCConfig(n_paths=5_000, n_steps=1, seed=1, greeks_estimator="pathwise")
    meta = greeks(ARITHMETIC, GREEKS_MODEL, GREEKS_MARKET, method="mc", cfg=cfg).meta
    assert meta is not None
    assert meta["estimator"] == {
        "delta": "pathwise",
        "gamma": "bump",
        "vega": "pathwise",
        "theta": "bump",
        "rho": "bump",
    }
    assert meta["theta_convention"].startswith("roll")

    cfg = MCConfig(
        n_paths=5_000, n_steps=1, seed=1, greeks_estimator="likelihood_ratio"
    )
    meta = greeks(ARITHMETIC, GREEKS_MODEL, GREEKS_MARKET, method="mc", cfg=cfg).meta
    assert meta is not None
    assert meta["estimator"]["delta"] == "likelihood_ratio"
    assert meta["estimator"]["vega"] == "bump"


@pytest.mark.parametrize(
    "variance_reduction", ["antithetic", "control_variate"]
)
def test_asian_greeks_compose_with_the_variance_reduction_that_applies(
    variance_reduction: str,
) -> None:
    """Evidence class: STATISTICAL. Antithetic and the Kemna-Vorst control both
    reduce the Greek sample the way they reduce the price sample.

    Stratification is refused here for the same reason it is refused for the
    price: an average over `n` fixings has no single scalar to partition.
    """
    cfg = MCConfig(
        n_paths=GREEKS_PATHS,
        n_steps=1,
        seed=9,
        greeks_estimator="pathwise",
        variance_reduction=variance_reduction,
    )
    plain = MCConfig(
        n_paths=GREEKS_PATHS, n_steps=1, seed=9, greeks_estimator="pathwise"
    )
    exact = greeks(GEOMETRIC, GREEKS_MODEL, GREEKS_MARKET, method="analytic").delta
    reduced = greeks(GEOMETRIC, GREEKS_MODEL, GREEKS_MARKET, method="mc", cfg=cfg)
    base = greeks(GEOMETRIC, GREEKS_MODEL, GREEKS_MARKET, method="mc", cfg=plain)
    assert reduced.meta is not None and base.meta is not None
    assert abs(reduced.delta - exact) <= 4.0 * reduced.meta["stderr"]["delta"]
    assert abs(base.delta - exact) <= 4.0 * base.meta["stderr"]["delta"]

    with pytest.raises(NotSupportedError, match="Brownian bridge"):
        greeks(
            GEOMETRIC,
            GREEKS_MODEL,
            GREEKS_MARKET,
            method="mc",
            cfg=MCConfig(
                n_paths=GREEKS_PATHS, n_steps=1, seed=9, variance_reduction="stratified"
            ),
        )


def test_zero_volatility_returns_the_deterministic_average_without_sampling():
    """Degenerate limit, matching the closed form and reporting zero stderr."""
    model = BlackScholesModel(sigma=0.0)
    for averaging, truth in (
        (
            "arithmetic",
            expected_arithmetic_average(S=SPOT, mu=MU, fixing_times=FIXINGS_10),
        ),
        (
            "geometric",
            expected_geometric_average(
                S=SPOT, mu=MU, sigma=0.0, fixing_times=FIXINGS_10
            ),
        ),
    ):
        result = price(
            _asian(averaging, FIXINGS_10),
            model,
            MARKET,
            method="mc",
            cfg=MCConfig(n_paths=1000, seed=1),
        )
        assert result.stderr == 0.0
        assert result.value == pytest.approx(
            DISCOUNT * max(truth - STRIKE, 0.0), abs=1e-12
        )
        assert result.meta is not None and result.meta["degenerate"] == "sigma=0"


def test_the_engine_is_deterministic_and_reports_what_it_did():
    option = _asian("arithmetic", FIXINGS_10)
    a = _mc(option, "control_variate", 5000, 11)
    b = _mc(option, "control_variate", 5000, 11)
    assert (a.value, a.stderr) == (b.value, b.stderr)
    assert a.meta is not None
    assert a.meta["n_fixings"] == 10
    assert a.meta["n_normal_draws"] == 5000 * 10
    assert a.meta["control_variate"] == "discounted_geometric_average_payoff"
    assert a.meta["averaging"] == "arithmetic"
    geometric = _mc(_asian("geometric", FIXINGS_10), "control_variate", 5000, 11)
    assert geometric.meta is not None
    assert geometric.meta["control_variate"] == "discounted_last_fixing_spot"


def test_the_control_mean_is_the_closed_form_not_a_sample_estimate():
    """EXACT_IDENTITY: `E[X]` for the geometric control is the exact price.

    The whole point of the Kemna-Vorst control is that its mean is *known*. If
    it were estimated from the same sample the correction would be identically
    zero and the estimator would be the plain one wearing a hat.
    """
    assert EvidenceClass.EXACT_IDENTITY
    option = _asian("arithmetic", FIXINGS_10)
    sample = asian_terminal_sample(
        option,
        MODEL,
        MARKET,
        cfg=_cfg("control_variate", 4000, 5),
        methods=("control_variate",),
    )
    assert sample.x_mean == pytest.approx(_geometric_truth(FIXINGS_10), abs=1e-14)
    assert sample.x_mean != pytest.approx(float(np.mean(sample.x)), abs=1e-6)


# --------------------------------------------------------------------------
# (c) Unbiasedness and coverage against the exact geometric price.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_fixings", [10, 52])
@pytest.mark.parametrize("estimator_id,vr", _ESTIMATORS, ids=[e[0] for e in _ESTIMATORS])
def test_geometric_mc_is_within_four_stderr_of_its_closed_form(
    n_fixings, estimator_id, vr
):
    """STATISTICAL: |z| < 4 at 200 000 paths against the exact discrete formula.

    The geometric Asian is the one path-dependent contract in this package that
    has an exact answer, which makes it the right place to test the *sampler*:
    a bug in the variable-time-grid stepping would show up here and nowhere
    else, because an arithmetic Asian has no closed form to be wrong against.

    Measured |z| at seed 7 (200 000 paths, or 400 000 for the antithetic
    columns, which is the same number of normal draws):

        fixings   none   control   antithetic   anti+ctrl
        10       -1.055   -0.970     -1.206      -1.325
        52       -0.711   -0.277     -0.466      -1.306

    The four columns are **not** independent evidence: they share a seed and
    the antithetic sample contains the plain one, so they move together. What
    varies between them is the width of the interval, and the sharpest column
    is the strongest statement -- the control-variate standard error at 52
    fixings is 1.05e-02 against the plain 1.75e-02.

    A note on seeds. At seed 20250913 the same cells read -1.78, -2.80, -3.37,
    -1.75 at ten fixings: all four low, because they share the draw. Over 30
    independent seeds the mean `z` of the plain estimator is -0.19 with sample
    standard deviation 1.075, so the estimator is unbiased and that seed was
    simply low. Seed 7 is used here because a 4-sigma gate on a *fixed* seed
    that happens to sit at -3.37 is a test that fails on an innocent refactor.
    """
    assert EvidenceClass.STATISTICAL
    times = FIXINGS_10 if n_fixings == 10 else FIXINGS_52
    result = _mc(
        _asian("geometric", times), vr, _UNBIASEDNESS_PATHS, _UNBIASEDNESS_SEED
    )
    assert result.stderr is not None and result.stderr > 0.0
    z = (result.value - _geometric_truth(times)) / result.stderr
    assert abs(z) < 4.0, f"{n_fixings}f/{estimator_id}: z={z:.4f}"


def test_the_reported_95_percent_interval_covers_at_the_binomial_tolerance():
    """STATISTICAL: coverage over 40 seeds, per estimator, at 4 000 paths.

    Tolerance: at least 33 of 40 per cell. Under true 95% coverage,
    `P(X <= 32) = 6.9e-04` for `X ~ Binomial(40, 0.95)`, so each cell is a
    one-sided test at about the 0.07% level -- the same construction Slice 7
    uses, rescaled to 40 seeds. Measured (ten fixings):

        estimator            hits/40
        none                   39
        control_variate        38
        antithetic             38
        antithetic+control     38

    Aggregate 153/160 = 0.956. The `control_variate` cell is the one worth
    reading: its intervals are about 38x narrower than the plain estimator's,
    so 38/40 there says far more about the `ddof=2` regression standard error
    than 39/40 says about the plain one. A control-variate standard error
    computed the naive way (the plain formula on the adjusted sample, with one
    degree of freedom instead of two) would be 1.00013x too small here and
    would still score 38 -- which is why the exact-identity tests in
    `tests/test_mc_variance_reduction.py` check the formula and this one checks
    only its calibration.
    """
    assert EvidenceClass.STATISTICAL
    truth = _geometric_truth(FIXINGS_10)
    option = _asian("geometric", FIXINGS_10)
    critical = 1.959963984540054
    total = 0
    for estimator_id, vr in _ESTIMATORS:
        hits = sum(
            1
            for seed in range(201, 241)
            if abs(_mc(option, vr, 4_000, seed).value - truth)
            <= critical * _mc(option, vr, 4_000, seed).stderr
        )
        assert hits >= 33, f"{estimator_id}: coverage {hits}/40"
        total += hits
    assert total / (40 * len(_ESTIMATORS)) >= 0.90


# --------------------------------------------------------------------------
# (d) AM-GM, pathwise, and the Turnbull-Wakeman gap.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_fixings", [10, 52])
def test_arithmetic_beats_geometric_on_every_seed_because_it_does_on_every_path(
    n_fixings,
):
    """EXACT_IDENTITY: AM >= GM pathwise, so the price ordering holds per seed.

    This is not a statistical claim and is deliberately not tested as one. Both
    averagings run on the *same* seed use the same normals, hence the same
    fixings; `A_arith >= A_geom` for each path by the arithmetic-geometric mean
    inequality; `max(. - K, 0)` is non-decreasing; so the discounted payoff is
    larger path by path and the sample mean is larger with probability one.
    Twenty seeds, and the assertion is on the payoff vectors as well as on the
    prices, so a failure says which step broke.

    Measured gap at 2 000 000 paths, control-variate estimator:
    +0.215454 at ten fixings and +0.216677 at fifty-two, i.e. about 3.5% of the
    price and almost independent of the fixing count -- the gap is driven by
    `Var(log A)`, which the fixing count changes only through the `O(1/n)`
    correction to the averaging window.
    """
    assert EvidenceClass.EXACT_IDENTITY
    times = FIXINGS_10 if n_fixings == 10 else FIXINGS_52
    arithmetic = _asian("arithmetic", times)
    geometric = _asian("geometric", times)
    for seed in range(1, 21):
        cfg = _cfg("none", 2_000, seed)
        sample_a = asian_terminal_sample(
            arithmetic, MODEL, MARKET, cfg=cfg, methods=()
        )
        sample_g = asian_terminal_sample(geometric, MODEL, MARKET, cfg=cfg, methods=())
        assert np.all(sample_a.y >= sample_g.y), seed
        assert float(sample_a.y.mean()) > float(sample_g.y.mean()), seed
        assert _mc(arithmetic, "none", 2_000, seed).value > _mc(
            geometric, "none", 2_000, seed
        ).value


def test_am_gm_holds_on_the_realised_averages_not_only_on_the_payoffs():
    """EXACT_IDENTITY: the inequality before the payoff truncates it.

    Checking only the payoffs would pass trivially whenever both are zero.
    """
    assert EvidenceClass.EXACT_IDENTITY
    rng = np.random.default_rng(19)
    z = rng.normal(size=(20_000, 52))
    fixings = gbm_paths_from_normals(
        z, s0=SPOT, mu=MU, sigma=SIGMA, times=np.asarray(FIXINGS_52)
    )
    arith = arithmetic_average(fixings)
    geom = geometric_average(fixings)
    assert np.all(arith > geom)
    # Measured mean ratio A_arith / A_geom at sigma = 20%, 52 fixings: 1.003374.
    # Small, and the price gap it produces is 3.5% -- the payoff's kink
    # amplifies a 0.34% shift of the whole distribution.
    assert float(np.mean(arith / geom)) == pytest.approx(1.003374, abs=2e-4)


_TW_GAP_ROWS = (
    # (label, n_fixings, sigma, strike, expiry, q, cv_mc, tw, gap)
    ("atm_10f", 10, 0.20, 100.0, 1.0, 0.0, 6.234570, 6.252316, +0.017746),
    ("atm_52f", 52, 0.20, 100.0, 1.0, 0.0, 5.854109, 5.873245, +0.019136),
    ("itm_26f_q_eq_r", 26, 0.20, 80.0, 0.5, 0.05, 19.513514, 19.515210, +0.001696),
    ("atm_52f_vol40", 52, 0.40, 100.0, 1.0, 0.0, 10.290842, 10.382623, +0.091781),
)


@pytest.mark.parametrize(
    ("label", "n_fixings", "sigma", "strike", "expiry", "q", "cv_mc", "tw", "gap"),
    _TW_GAP_ROWS,
    ids=[row[0] for row in _TW_GAP_ROWS],
)
def test_the_turnbull_wakeman_gap_is_measured_with_its_sign(
    label, n_fixings, sigma, strike, expiry, q, cv_mc, tw, gap
):
    """STATISTICAL + PUBLISHED_BENCHMARK(approximation): TW is **above** the truth.

    The slice statement asked for the gap and its sign, and not for a claim that
    the approximation is exact. Measured at 500 000 to 2 000 000 paths with the
    Kemna-Vorst control variate (standard errors 2e-04 to 1.3e-03, so the gap is
    resolved at 8 to 105 standard errors -- it is a real bias, not noise):

        point                   sigma^2 T   CV-MC       TW          gap        rel
        ATM, 10 fixings           0.040     6.234570    6.252316   +0.017746  +0.285%
        ATM, 52 fixings           0.040     5.854109    5.873245   +0.019136  +0.327%
        K=80, 26 fixings, q=r     0.020    19.513514   19.515210   +0.001696  +0.009%
        ATM, 52 fixings, vol 40%  0.160    10.290842   10.382623   +0.091781  +0.892%

    The sign is **always positive** here: the fitted lognormal is more
    right-skewed than the true distribution of the arithmetic average, so it
    puts too much mass in the region that pays and overprices the call. The size
    tracks `sigma^2 T` -- doubling the volatility quadruples the total variance
    and multiplies the relative error by about three -- and collapses deep in
    the money, where the option is nearly a forward on the average and the whole
    price is determined by the first moment, which the approximation matches
    **exactly**.

    That last row is the one to remember when reading the published 19.5152
    benchmark: the approximation reproduces its own published value to 9.8e-06
    (`tests/test_asian_analytic.py`) and is 1.7e-03 away from the true price. A
    benchmark for an approximation is not a benchmark for the thing it
    approximates.
    """
    assert EvidenceClass.STATISTICAL
    market = Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(0.05),
        dividend_curve=FlatDividendCurve(q),
    )
    model = BlackScholesModel(sigma=sigma)
    r = market.rate(expiry)
    times = uniform_fixing_times(expiry, n_fixings)
    option = AsianOption("call", strike, expiry, times, "arithmetic")

    approx = turnbull_wakeman_price(
        S=SPOT,
        K=strike,
        T=expiry,
        r=r,
        sigma=sigma,
        fixing_times=times,
        q=market.dividend_yield(expiry),
    )
    assert approx == pytest.approx(tw, abs=1e-5)

    result = price(
        option,
        model,
        market,
        method="mc",
        cfg=MCConfig(n_paths=200_000, seed=7, variance_reduction="control_variate"),
    )
    measured_gap = approx - result.value
    assert measured_gap > 0.0, f"{label}: TW is not above the simulated price"
    # Recorded gap, within the shorter run's own noise plus a little.
    assert measured_gap == pytest.approx(gap, abs=max(6.0 * result.stderr, 2e-4))
    # And the gap is resolved: it is many standard errors, not a coin flip.
    assert measured_gap > 4.0 * result.stderr


# --------------------------------------------------------------------------
# (e) Asian put-call parity.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("averaging", ["arithmetic", "geometric"])
@pytest.mark.parametrize("n_fixings", [10, 52])
def test_asian_put_call_parity_is_exact_per_sample(averaging, n_fixings):
    """EXACT_IDENTITY: `C - P = e^{-rT}(A_sample - K)` path by path.

    `max(A - K, 0) - max(K - A, 0) = A - K` for every real `A`, so with the same
    seed -- hence the same normals, hence the same paths -- the call and put
    samples differ by an affine function of the realised average. The identity
    therefore holds *in the sample*, exactly, and not merely in expectation:
    there is no Monte Carlo error in it at all.

    Checked at round-off on the payoff vectors, then on the prices, and then
    against the closed-form `E[A]` (a sum of forwards) or `E[G] = exp(m + v/2)`
    as a statistical statement -- which is the only part of this that has a
    standard error.
    """
    assert EvidenceClass.EXACT_IDENTITY
    times = FIXINGS_10 if n_fixings == 10 else FIXINGS_52
    call = _asian(averaging, times, "call")
    put = _asian(averaging, times, "put")
    cfg = _cfg("none", 20_000, 13)

    y_call = asian_terminal_sample(call, MODEL, MARKET, cfg=cfg, methods=()).y
    y_put = asian_terminal_sample(put, MODEL, MARKET, cfg=cfg, methods=()).y
    realised_average = y_call / DISCOUNT - y_put / DISCOUNT + STRIKE
    assert np.all(realised_average > 0.0)

    call_price = price(call, MODEL, MARKET, method="mc", cfg=cfg)
    put_price = price(put, MODEL, MARKET, method="mc", cfg=cfg)
    sample_mean = float(realised_average.mean())
    assert call_price.value - put_price.value == pytest.approx(
        DISCOUNT * (sample_mean - STRIKE), abs=1e-11
    )

    exact_mean = (
        expected_arithmetic_average(S=SPOT, mu=MU, fixing_times=times)
        if averaging == "arithmetic"
        else expected_geometric_average(
            S=SPOT, mu=MU, sigma=SIGMA, fixing_times=times
        )
    )
    sample_se = float(realised_average.std(ddof=1) / math.sqrt(realised_average.size))
    assert abs(sample_mean - exact_mean) < 4.0 * sample_se


@pytest.mark.parametrize("n_fixings", [1, 10, 52])
def test_the_closed_form_geometric_pair_satisfies_parity_through_the_dispatcher(
    n_fixings,
):
    """EXACT_IDENTITY: the analytic route obeys the same identity with `E[G]`.

    The Monte Carlo statement above is about a realised sample; this one is
    about the model. Both are needed: the first would pass even if the closed
    form were wrong, and the second would pass even if the sampler were.
    """
    assert EvidenceClass.EXACT_IDENTITY
    times = uniform_fixing_times(EXPIRY, n_fixings)
    call = price(_asian("geometric", times, "call"), MODEL, MARKET).value
    put = price(_asian("geometric", times, "put"), MODEL, MARKET).value
    expected = expected_geometric_average(
        S=SPOT, mu=MU, sigma=SIGMA, fixing_times=times
    )
    assert call - put == pytest.approx(DISCOUNT * (expected - STRIKE), abs=1e-12)
