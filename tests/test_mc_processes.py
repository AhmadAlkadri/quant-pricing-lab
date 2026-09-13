import math

import numpy as np
import pytest

from qpl.engines.mc.processes import (
    price_european_from_terminal,
    simulate_gbm_euler,
    simulate_gbm_exact,
)
from qpl.exceptions import InvalidInputError
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price


def _european_setup() -> tuple[EuropeanOption, BlackScholesModel, Market]:
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = Market(
        spot=100.0,
        rate_curve=FlatRateCurve(0.05),
        dividend_curve=FlatDividendCurve(0.01),
    )
    return option, model, market


def test_simulate_gbm_exact_deterministic() -> None:
    a = simulate_gbm_exact(s0=100.0, mu=0.04, sigma=0.2, t=1.0, n_steps=12, n_paths=100, seed=7)
    b = simulate_gbm_exact(s0=100.0, mu=0.04, sigma=0.2, t=1.0, n_steps=12, n_paths=100, seed=7)
    assert np.array_equal(a, b)
    assert a.shape == (100, 13)
    assert np.all(a > 0.0)


def test_simulate_gbm_euler_deterministic() -> None:
    a = simulate_gbm_euler(s0=100.0, mu=0.04, sigma=0.2, t=1.0, n_steps=12, n_paths=100, seed=7)
    b = simulate_gbm_euler(s0=100.0, mu=0.04, sigma=0.2, t=1.0, n_steps=12, n_paths=100, seed=7)
    assert np.array_equal(a, b)
    assert a.shape == (100, 13)


def test_simulate_gbm_exact_log_return_moments_one_step() -> None:
    s0 = 100.0
    mu = 0.03
    sigma = 0.25
    t = 1.0
    n_paths = 150_000
    paths = simulate_gbm_exact(
        s0=s0,
        mu=mu,
        sigma=sigma,
        t=t,
        n_steps=1,
        n_paths=n_paths,
        seed=123,
    )
    log_returns = np.log(paths[:, 1] / s0)

    expected_mean = (mu - 0.5 * sigma * sigma) * t
    expected_var = sigma * sigma * t
    sample_mean = float(np.mean(log_returns))
    sample_var = float(np.var(log_returns))

    mean_tol = 6.0 * math.sqrt(expected_var / n_paths)
    var_tol = 8.0 * expected_var * math.sqrt(2.0 / (n_paths - 1))

    assert abs(sample_mean - expected_mean) < mean_tol
    assert abs(sample_var - expected_var) < var_tol


def test_exact_terminal_price_matches_analytic_within_ci() -> None:
    option, model, market = _european_setup()
    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    mu = r - q

    n_paths = 120_000
    paths = simulate_gbm_exact(
        s0=market.spot,
        mu=mu,
        sigma=model.sigma,
        t=t,
        n_steps=12,
        n_paths=n_paths,
        seed=11,
    )
    mc = price_european_from_terminal(
        paths[:, -1],
        strike=option.strike,
        discount_factor=math.exp(-r * t),
        kind=option.kind,
    )

    analytic = price(option, model, market, method="analytic").value
    assert abs(mc.value - analytic) <= 4.0 * mc.stderr


def test_euler_strong_error_decreases_with_steps() -> None:
    option, model, market = _european_setup()
    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    mu = r - q

    step_grid = [5, 20, 50]
    n_paths = 50_000
    seed = 123
    rmses = []

    for n_steps in step_grid:
        exact = simulate_gbm_exact(
            s0=market.spot,
            mu=mu,
            sigma=model.sigma,
            t=t,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=seed,
        )[:, -1]
        euler = simulate_gbm_euler(
            s0=market.spot,
            mu=mu,
            sigma=model.sigma,
            t=t,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=seed,
        )[:, -1]
        rmses.append(float(np.sqrt(np.mean((euler - exact) ** 2))))

    assert rmses[1] < rmses[0]
    assert rmses[2] < rmses[1]


def test_euler_pricing_reasonable_with_ci_style_tolerance() -> None:
    option, model, market = _european_setup()
    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    mu = r - q

    n_paths = 120_000
    euler_terminal = simulate_gbm_euler(
        s0=market.spot,
        mu=mu,
        sigma=model.sigma,
        t=t,
        n_steps=50,
        n_paths=n_paths,
        seed=321,
    )[:, -1]
    euler_mc = price_european_from_terminal(
        euler_terminal,
        strike=option.strike,
        discount_factor=math.exp(-r * t),
        kind=option.kind,
    )

    analytic = price(option, model, market, method="analytic").value
    discretization_allowance = 0.03
    assert abs(euler_mc.value - analytic) <= 4.0 * euler_mc.stderr + discretization_allowance


def test_price_european_from_terminal_validation_errors() -> None:
    with pytest.raises(InvalidInputError):
        price_european_from_terminal(np.array([[100.0, 101.0]]), strike=100.0, discount_factor=0.95)
    with pytest.raises(InvalidInputError):
        price_european_from_terminal(np.array([100.0]), strike=100.0, discount_factor=0.95)
    with pytest.raises(InvalidInputError):
        price_european_from_terminal(np.array([100.0, 101.0]), strike=0.0, discount_factor=0.95)
    with pytest.raises(InvalidInputError):
        price_european_from_terminal(np.array([100.0, 101.0]), strike=100.0, discount_factor=0.0)
    with pytest.raises(InvalidInputError):
        price_european_from_terminal(
            np.array([100.0, 101.0]),
            strike=100.0,
            discount_factor=0.95,
            kind="digital",  # type: ignore[arg-type]
        )


def test_process_validation_errors() -> None:
    with pytest.raises(InvalidInputError):
        simulate_gbm_exact(s0=0.0, mu=0.0, sigma=0.2, t=1.0, n_steps=10, n_paths=10)
    with pytest.raises(InvalidInputError):
        simulate_gbm_exact(s0=100.0, mu=0.0, sigma=-0.1, t=1.0, n_steps=10, n_paths=10)
    with pytest.raises(InvalidInputError):
        simulate_gbm_exact(s0=100.0, mu=0.0, sigma=0.2, t=-1.0, n_steps=10, n_paths=10)
    with pytest.raises(InvalidInputError):
        simulate_gbm_exact(s0=100.0, mu=0.0, sigma=0.2, t=1.0, n_steps=0, n_paths=10)
    with pytest.raises(InvalidInputError):
        simulate_gbm_exact(s0=100.0, mu=0.0, sigma=0.2, t=1.0, n_steps=10, n_paths=0)

    with pytest.raises(InvalidInputError):
        simulate_gbm_euler(s0=0.0, mu=0.0, sigma=0.2, t=1.0, n_steps=10, n_paths=10)
    with pytest.raises(InvalidInputError):
        simulate_gbm_euler(s0=100.0, mu=0.0, sigma=-0.1, t=1.0, n_steps=10, n_paths=10)
    with pytest.raises(InvalidInputError):
        simulate_gbm_euler(s0=100.0, mu=0.0, sigma=0.2, t=-1.0, n_steps=10, n_paths=10)
    with pytest.raises(InvalidInputError):
        simulate_gbm_euler(s0=100.0, mu=0.0, sigma=0.2, t=1.0, n_steps=0, n_paths=10)
    with pytest.raises(InvalidInputError):
        simulate_gbm_euler(s0=100.0, mu=0.0, sigma=0.2, t=1.0, n_steps=10, n_paths=0)
