"""Mechanics, contract and degenerate limits of the least-squares MC engine.

This file checks that the engine *does what the algorithm says*, on inputs
small enough to verify by hand, plus the config contract, determinism, and the
two identities that need no reference value. The numbers that need a benchmark
-- Longstaff & Schwartz's table, the lattice Bermudan, the three-way
cross-method agreement, the exercise boundary -- live in
`tests/test_lsm_american.py`; the basis and in-sample bias measurements live in
`tests/test_lsm_american_bias.py`.

Evidence classes used here: EXACT_IDENTITY (the hand-computed rollback and the
determinism pins, where the tolerance is round-off and nothing else) and
STATISTICAL (the no-dividend American call, where the claim is that a
simulation estimate sits within a stated multiple of its own standard error of
a closed form).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.engines.mc.american import (
    LSM_BASES,
    basis_matrix,
    lsm_rollback,
    simulate_exercise_grid,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import AmericanOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel, bs_price
from qpl.pricing import greeks, price


def _market(spot: float, rate: float, div: float = 0.0) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(rate),
        dividend_curve=FlatDividendCurve(div),
    )


# --------------------------------------------------------------------------
# (a) The rollback, on an example small enough to do by hand.
# --------------------------------------------------------------------------


def test_three_paths_two_dates_regression_is_an_exact_interpolation() -> None:
    """Evidence class: EXACT_IDENTITY.

    Three paths, two exercise dates, `K = 1`, `r = 10%`, a degree-1 polynomial
    basis. At `t_1 = 0.5` the spots are `(0.8, 0.9, 1.4)`, so exactly **two**
    paths are in the money and the design matrix is `2 x 2` and invertible:
    the least-squares fit is an exact interpolation, and the fitted
    continuation value on each of those two paths is that path's own
    discounted terminal cashflow. Nothing is approximated, so every number
    below is arithmetic:

        terminal spots      (1.2, 0.6, 1.3)
        terminal payoffs    (0.0, 0.4, 0.0)
        cashflows at t=0    (0.0, 0.4 e^{-0.1}, 0.0)
        regressands at t_1  (0.0, 0.4 e^{-0.05})
        fitted slope        (0.4 e^{-0.05} - 0) / (0.9 - 0.8)
        intrinsic at t_1    (0.2, 0.1, 0.0)

    Path 0 exercises (`0.2 > 0`), path 1 does not (`0.1 < 0.4 e^{-0.05}`),
    path 2 is out of the money and is never even offered the choice. The price
    is the mean of `(0.2 e^{-0.05}, 0.4 e^{-0.1}, 0)`.

    This is the whole algorithm: the in-the-money filter, the regression on the
    current spot, the comparison against intrinsic, and the fact that the value
    reported is the **realised** cashflow and not the fitted one -- path 1's
    contribution is `0.4 e^{-0.1}`, its actual payoff, not the `0.4 e^{-0.05}`
    the regression fitted for it.
    """
    paths = np.array([[0.8, 1.2], [0.9, 0.6], [1.4, 1.3]])
    times = np.array([0.5, 1.0])
    rate = 0.10
    discounts = np.exp(-rate * times)

    fit = lsm_rollback(
        paths,
        times=times,
        discounts=discounts,
        kind="put",
        strike=1.0,
        basis="polynomial",
        degree=1,
    )

    d1, d2 = float(discounts[0]), float(discounts[1])
    expected_cashflows = np.array([0.2 * d1, 0.4 * d2, 0.0])
    assert fit.cashflows == pytest.approx(expected_cashflows, abs=1e-15)
    assert fit.stop_index.tolist() == [0, 1, 1]
    assert fit.early_exercise_fraction == pytest.approx(1.0 / 3.0)

    # The interpolating line through (0.8, 0) and (0.9, 0.4 e^{-0.05}).
    slope = 0.4 * d2 / d1 / (0.9 - 0.8)
    assert fit.coefficients[0] == pytest.approx(
        np.array([-0.8 * slope, slope]), rel=1e-12
    )
    # A put's exercise region is S <= B, so the boundary is the largest
    # exercising spot; only path 0 exercises at t_1.
    assert fit.boundary[0] == pytest.approx(0.8)
    # At expiry every in-the-money path is exercised, and the largest such
    # spot is path 1's 0.6.
    assert fit.boundary[1] == pytest.approx(0.6)


def test_applying_a_given_policy_reproduces_the_fitted_one_on_its_own_paths() -> None:
    """Evidence class: EXACT_IDENTITY.

    Feeding a fit's own coefficients back in as `coefficients=` must reproduce
    it bit for bit on the same paths: the fitting pass and the applying pass
    differ only in where `beta` comes from. Without this, an out-of-sample
    valuation could silently be running a *different* decision rule from the
    one that was trained, and the low-biased estimator would be low-biased for
    the wrong reason.
    """
    times = np.linspace(0.0, 1.0, 11)[1:]
    paths, _ = simulate_exercise_grid(
        s0=36.0, mu=0.06, sigma=0.2, times=times, n_paths=2_000, seed=5, methods=()
    )
    discounts = np.exp(-0.06 * times)
    common = dict(
        times=times, discounts=discounts, kind="put", strike=40.0,
        basis="laguerre", degree=3,
    )
    fitted = lsm_rollback(paths, **common)
    applied = lsm_rollback(paths, coefficients=fitted.coefficients, **common)

    assert np.array_equal(applied.cashflows, fitted.cashflows)
    assert np.array_equal(applied.stop_index, fitted.stop_index)


def test_one_exercise_date_is_the_european_option() -> None:
    """Evidence class: CLOSED_FORM.

    `exercise_dates=1` leaves only the terminal date, so no regression runs and
    the estimator is the plain discounted-terminal-payoff average. It must
    reproduce Black-Scholes within its own noise, and the early-exercise
    fraction must be exactly zero -- there is nothing to exercise early.
    """
    option = AmericanOption(kind="put", strike=100.0, expiry=1.0)
    result = price(
        option,
        BlackScholesModel(sigma=0.20),
        _market(100.0, 0.05),
        method="mc",
        cfg=MCConfig(n_paths=40_000, seed=3, exercise_dates=1),
    )
    closed_form = bs_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0, kind="put")

    assert result.meta is not None
    assert result.meta["early_exercise_fraction"] == 0.0
    assert result.meta["n_regressions"] == 0
    assert abs(result.value - closed_form) < 3.0 * result.stderr


# --------------------------------------------------------------------------
# The basis.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("basis", LSM_BASES)
@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_basis_has_degree_plus_one_columns_and_a_constant(basis: str, degree: int) -> None:
    """Both families have `degree + 1` columns, so degrees are comparable.

    The first column is the constant in both, which is what makes
    `lsm_degree=3, lsm_basis="laguerre"` the Longstaff-Schwartz basis exactly:
    a constant plus the first three weighted Laguerre functions.
    """
    spots = np.array([60.0, 90.0, 100.0, 130.0])
    design = basis_matrix(spots, strike=100.0, basis=basis, degree=degree)
    assert design.shape == (4, degree + 1)
    assert np.array_equal(design[:, 0], np.ones(4))


def test_weighted_laguerre_columns_match_the_closed_forms() -> None:
    """Evidence class: CLOSED_FORM.

    The recurrence is checked against the expanded first four weighted Laguerre
    functions, written out independently:

        w L_0 = w,  w L_1 = w (1 - x),  w L_2 = w (1 - 2x + x^2/2),
        w L_3 = w (1 - 3x + 3x^2/2 - x^3/6),      w = exp(-x/2).

    Standard forms, not quoted from Longstaff & Schwartz; the point of the
    check is that the recurrence in `_laguerre_columns` implements them.
    """
    strike = 40.0
    spots = np.array([10.0, 36.0, 40.0, 72.0, 120.0])
    x = spots / strike
    w = np.exp(-0.5 * x)
    design = basis_matrix(spots, strike=strike, basis="laguerre", degree=4)

    assert design[:, 1] == pytest.approx(w, rel=1e-14)
    assert design[:, 2] == pytest.approx(w * (1.0 - x), rel=1e-14)
    assert design[:, 3] == pytest.approx(w * (1.0 - 2.0 * x + x * x / 2.0), rel=1e-14)
    assert design[:, 4] == pytest.approx(
        w * (1.0 - 3.0 * x + 1.5 * x * x - x**3 / 6.0), rel=1e-13
    )


def test_scaling_by_the_strike_is_what_keeps_the_laguerre_basis_alive() -> None:
    """Evidence class: NEGATIVE_FINDING, in the form of the counterfactual.

    At a spot of 100 an unscaled weighted Laguerre basis carries `exp(-50)`,
    which is `2e-22`. The polynomial factors claw a little of that back -- `L_3`
    at `x = 120` is about `-3e+5` -- so the largest non-constant entry over the
    sample below is `1.3e-14`, against a constant column of exactly 1. The
    design is rank one to within 1e-14 relative, its condition number is
    `5.0e+23`, and a regression on it returns the sample mean and three
    coefficients of noise. Scaled by the strike the same sample has a condition
    number of `8.5e+04` -- not small, since four smooth functions on five
    points are genuinely close to dependent, but nineteen orders of magnitude
    better and inside what `lstsq` resolves. Asserted rather than described,
    because it is the reason `basis_matrix` takes a `strike` at all.
    """
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    scaled = basis_matrix(spots, strike=100.0, basis="laguerre", degree=3)
    unscaled = basis_matrix(spots, strike=1.0, basis="laguerre", degree=3)

    assert np.max(np.abs(unscaled[:, 1:])) < 1e-13
    assert np.linalg.cond(unscaled) > 1e20
    assert np.linalg.cond(scaled) < 1e6


# --------------------------------------------------------------------------
# (e) The no-dividend American call.
# --------------------------------------------------------------------------


def test_no_dividend_american_call_is_the_european_call_and_rarely_stops_early() -> None:
    """Evidence class: STATISTICAL, plus a measured early-exercise fraction.

    Merton (1973): an American call on a non-dividend-paying stock is never
    exercised early, because the continuation value dominates `S - K` by
    `K (1 - e^{-r(T-t)})` at every date. A lattice reproduces that as an exact
    identity (`tests/cases/test_american_black_scholes_cases.py` asserts a zero
    difference). LSM cannot: the exercise decision is made against a *fitted*
    continuation value, and wherever the fit sits below the truth by more than
    the true early-exercise premium, a path stops early.

    Measured at `S = K = 100, r = 5%, q = 0, sigma = 20%, T = 1`, 50 dates,
    50 000 antithetic paths, laguerre degree 3, seed 11: the early-exercise
    fraction is **1.9e-03** and the price is 10.4383 against the Black-Scholes
    10.4506, a gap of -1.2e-02 against a standard error of 4.6e-02.

    The bound below is 1e-02 on the fraction, five times the measured value.
    It is deliberately not `== 0`: that would be asserting the regression is
    exact. It is deliberately not loose either -- the same run at **250**
    dates gives a fraction of **0.219**, and the reason is worth writing down.
    Near expiry the true early-exercise premium of a call is
    `K (1 - e^{-r dt}) ~ K r dt`, which at `dt = 1/250` is 0.02 on a value of
    ten; the regression's own approximation error over the in-the-money range
    is of that size, so the comparison at the last few dates is a coin flip.
    It costs almost nothing in value (exercising at `T - dt` instead of `T` is
    worth `~ K r dt`), which is why the *price* stays inside its standard
    error while the *fraction* does not stay inside anything. Pinned in
    `test_call_early_exercise_fraction_blows_up_when_the_dates_get_dense`.
    """
    option = AmericanOption(kind="call", strike=100.0, expiry=1.0)
    result = price(
        option,
        BlackScholesModel(sigma=0.20),
        _market(100.0, 0.05),
        method="mc",
        cfg=MCConfig(
            n_paths=50_000, seed=11, variance_reduction="antithetic",
            exercise_dates=50, lsm_degree=3,
        ),
    )
    closed_form = bs_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0, kind="call")

    assert result.meta is not None
    fraction = result.meta["early_exercise_fraction"]
    assert fraction < 1e-2, fraction
    assert abs(result.value - closed_form) < 3.0 * result.stderr


def test_call_early_exercise_fraction_blows_up_when_the_dates_get_dense() -> None:
    """Evidence class: NEGATIVE_FINDING.

    The same call at 250 dates stops early on about a fifth of the paths while
    its price stays within one standard error of Black-Scholes. Pinned so that
    the early-exercise fraction is not read as a measure of whether the policy
    is right: it measures how often the regression's own error exceeds a
    premium that shrinks like `dt`, which is a statement about the basis and
    the date spacing, not about the value. See the sibling test's docstring for
    the mechanism.
    """
    option = AmericanOption(kind="call", strike=100.0, expiry=1.0)
    result = price(
        option,
        BlackScholesModel(sigma=0.20),
        _market(100.0, 0.05),
        method="mc",
        cfg=MCConfig(
            n_paths=50_000, seed=11, variance_reduction="antithetic",
            exercise_dates=250, lsm_degree=3,
        ),
    )
    closed_form = bs_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0, kind="call")

    assert result.meta is not None
    assert result.meta["early_exercise_fraction"] > 0.1
    assert abs(result.value - closed_form) < 3.0 * result.stderr


# --------------------------------------------------------------------------
# (h) Determinism.
# --------------------------------------------------------------------------


_DETERMINISM_CONFIGS = [
    MCConfig(n_paths=4_000, seed=1, exercise_dates=12),
    MCConfig(n_paths=4_000, seed=1, exercise_dates=12, lsm_in_sample=True),
    MCConfig(
        n_paths=4_000, seed=1, exercise_dates=12, variance_reduction="antithetic",
        lsm_basis="polynomial", lsm_degree=5,
    ),
]


@pytest.mark.parametrize("cfg", _DETERMINISM_CONFIGS, ids=["plain", "in_sample", "antithetic"])
def test_same_seed_gives_the_same_number(cfg: MCConfig) -> None:
    """Evidence class: EXACT_IDENTITY. Equality, not `approx`."""
    option = AmericanOption(kind="put", strike=40.0, expiry=1.0)
    model, market = BlackScholesModel(sigma=0.20), _market(36.0, 0.06)
    first = price(option, model, market, method="mc", cfg=cfg)
    second = price(option, model, market, method="mc", cfg=cfg)
    assert first.value == second.value
    assert first.stderr == second.stderr


def test_a_different_seed_gives_a_different_number() -> None:
    """The companion to determinism: the engine is not ignoring the seed."""
    option = AmericanOption(kind="put", strike=40.0, expiry=1.0)
    model, market = BlackScholesModel(sigma=0.20), _market(36.0, 0.06)
    values = {
        price(
            option, model, market, method="mc",
            cfg=MCConfig(n_paths=4_000, seed=seed, exercise_dates=12),
        ).value
        for seed in (1, 2, 3)
    }
    assert len(values) == 3


def test_the_training_and_valuation_streams_are_different_samples() -> None:
    """`SeedSequence.spawn` gives two streams, not the same one twice.

    If the two spawned generators produced the same normals, the
    "out-of-sample" estimator would be the in-sample one under another name and
    every bias measurement in `tests/test_lsm_american_bias.py` would be
    measuring zero.
    """
    times = np.linspace(0.0, 1.0, 6)[1:]
    train, value = (
        simulate_exercise_grid(
            s0=100.0, mu=0.05, sigma=0.2, times=times, n_paths=64,
            seed=np.random.default_rng(s), methods=(),
        )[0]
        for s in np.random.SeedSequence(123).spawn(2)
    )
    assert not np.allclose(train, value)


# --------------------------------------------------------------------------
# Contract: what the engine refuses, and the degenerate limits.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (MCConfig(n_paths=1), "n_paths must be >= 2"),
        (MCConfig(n_steps=4), "n_steps must be 1"),
        (MCConfig(exercise_dates=0), "exercise_dates must be >= 1"),
        (MCConfig(exercise_dates=2.5), "exercise_dates must be an int"),  # type: ignore[arg-type]
        (MCConfig(lsm_basis="hermite"), "unknown lsm_basis"),  # type: ignore[arg-type]
        (MCConfig(lsm_degree=0), "lsm_degree must be >= 1"),
        (MCConfig(lsm_degree="3"), "lsm_degree must be an int"),  # type: ignore[arg-type]
        (MCConfig(lsm_in_sample=1), "lsm_in_sample must be a bool"),  # type: ignore[arg-type]
        (MCConfig(n_paths=4_001, variance_reduction="antithetic"), "n_paths must be even"),
    ],
)
def test_invalid_config_is_rejected(cfg: MCConfig, message: str) -> None:
    option = AmericanOption(kind="put", strike=40.0, expiry=1.0)
    with pytest.raises(InvalidInputError, match=message):
        price(option, BlackScholesModel(sigma=0.2), _market(36.0, 0.06), method="mc", cfg=cfg)


@pytest.mark.parametrize("name", ["control_variate", "stratified"])
def test_the_two_reductions_that_do_not_compose_are_refused(name: str) -> None:
    """`NotSupportedError` with a message that says which one and why."""
    option = AmericanOption(kind="put", strike=40.0, expiry=1.0)
    with pytest.raises(NotSupportedError, match=name):
        price(
            option, BlackScholesModel(sigma=0.2), _market(36.0, 0.06), method="mc",
            cfg=MCConfig(n_paths=1_024, variance_reduction=name, exercise_dates=8),
        )


def test_antithetic_does_compose_and_halves_the_estimator_units() -> None:
    """The pair average is the unit, so the reported stderr is over pairs.

    What makes this legitimate for LSM and not merely arithmetic: one
    regression is fitted across the whole sample, each path makes its own
    exercise decision, and the pairing is applied to the **realised cashflow**
    afterwards. Reducing to pairs earlier would average two different stopping
    times before either had been decided.
    """
    option = AmericanOption(kind="put", strike=40.0, expiry=1.0)
    result = price(
        option, BlackScholesModel(sigma=0.2), _market(36.0, 0.06), method="mc",
        cfg=MCConfig(n_paths=8_000, seed=4, variance_reduction="antithetic", exercise_dates=10),
    )
    assert result.meta is not None
    assert result.meta["n_estimator_units"] == 4_000
    assert result.meta["n_antithetic_pairs"] == 4_000
    assert result.meta["n_normal_draws"] == 2 * 4_000 * 10


def test_greeks_are_refused_with_a_reason() -> None:
    """Registered-but-raising, so the message names the exercise policy."""
    option = AmericanOption(kind="put", strike=40.0, expiry=1.0)
    with pytest.raises(NotSupportedError, match="exercise policy"):
        greeks(
            option, BlackScholesModel(sigma=0.2), _market(36.0, 0.06), method="mc",
            cfg=MCConfig(n_paths=1_000, exercise_dates=8),
        )


def test_a_european_option_still_goes_to_the_european_engine() -> None:
    """The new key must not shadow the old one: `EuropeanOption` is a sibling."""
    european = EuropeanOption(kind="put", strike=40.0, expiry=1.0)
    result = price(
        european, BlackScholesModel(sigma=0.2), _market(36.0, 0.06), method="mc",
        cfg=MCConfig(n_paths=2_000, seed=9),
    )
    assert result.meta is not None
    assert "instrument" not in result.meta
    assert result.meta["n_steps"] == 1


@pytest.mark.parametrize("kind", ["put", "call"])
def test_zero_expiry_is_intrinsic(kind: str) -> None:
    option = AmericanOption(kind=kind, strike=100.0, expiry=0.0)  # type: ignore[arg-type]
    result = price(
        option, BlackScholesModel(sigma=0.2), _market(90.0, 0.05), method="mc",
        cfg=MCConfig(n_paths=1_000),
    )
    assert result.value == pytest.approx(10.0 if kind == "put" else 0.0)
    assert result.stderr == 0.0
    assert result.meta is not None and result.meta["degenerate"] == "T=0"


def test_zero_volatility_maximises_the_discounted_intrinsic_along_the_forward() -> None:
    """At `sigma = 0` there is nothing to simulate and nothing to regress.

    The spot is the deterministic forward `S_0 e^{(r-q)t}`, so the Bermudan
    problem collapses to a maximum of `e^{-rt} g(F(t))` over the exercise
    dates. With `q > r` the forward falls, so a put is worth most at the
    **last** date, and with `q = 0` the forward rises and the put is worth most
    at the first -- the same statement `qpl.engines.tree.american` makes over a
    continuum, restricted to this grid. Checked against a direct maximisation
    over the same dates.
    """
    spot, strike, rate, div, expiry, dates = 100.0, 120.0, 0.02, 0.10, 2.0, 8
    option = AmericanOption(kind="put", strike=strike, expiry=expiry)
    result = price(
        option, BlackScholesModel(sigma=0.0), _market(spot, rate, div), method="mc",
        cfg=MCConfig(n_paths=1_000, exercise_dates=dates),
    )
    times = np.linspace(0.0, expiry, dates + 1)[1:]
    expected = max(
        math.exp(-rate * t) * max(strike - spot * math.exp((rate - div) * t), 0.0)
        for t in times
    )
    assert result.value == pytest.approx(expected, rel=1e-12)
    assert result.stderr == 0.0
    assert result.meta is not None and result.meta["degenerate"] == "sigma=0"


def test_meta_reports_the_estimator_and_what_it_priced() -> None:
    """A Bermudan value must never be silently labelled American."""
    option = AmericanOption(kind="put", strike=40.0, expiry=1.0)
    result = price(
        option, BlackScholesModel(sigma=0.2), _market(36.0, 0.06), method="mc",
        cfg=MCConfig(n_paths=4_000, seed=2, exercise_dates=25, lsm_degree=3),
    )
    meta = result.meta
    assert meta is not None
    assert meta["estimator"] == "longstaff_schwartz"
    assert meta["exercise_style"] == "bermudan(25 dates)"
    assert meta["exercise_dates"] == 25
    assert meta["n_basis_functions"] == 4
    assert meta["lsm_in_sample"] is False
    assert meta["n_regressions"] == 24
    assert np.asarray(meta["exercise_boundary"]).shape == (25,)
    assert np.asarray(meta["exercise_boundary_times"])[-1] == pytest.approx(1.0)
    assert math.isfinite(meta["regression_condition_max"])
