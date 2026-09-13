import math

import numpy as np
import pytest

from qpl.engines.dp import (
    BinomialDPConfig,
    backward_induction_optimal_stopping,
    build_recombining_spot_tree,
    price_american_put_binomial,
)
from qpl.exceptions import InvalidInputError
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price


def _market(spot: float, r: float, q: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(r, allow_negative=True),
        dividend_curve=FlatDividendCurve(q, allow_negative=True),
    )


def test_backward_induction_toy_recursion_closed_form() -> None:
    exercise_values = [
        np.array([0.0]),
        np.array([1.0, 2.0]),
        np.array([0.0, 4.0, 1.0]),
    ]

    def continuation(_level: int, next_values: np.ndarray) -> np.ndarray:
        return 0.5 * (next_values[:-1] + next_values[1:])

    value_tree, policy = backward_induction_optimal_stopping(
        exercise_values=exercise_values,
        continuation_operator=continuation,
    )

    assert np.allclose(value_tree[2], np.array([0.0, 4.0, 1.0]))
    assert np.allclose(value_tree[1], np.array([2.0, 2.5]))
    assert np.allclose(value_tree[0], np.array([2.25]))
    assert np.array_equal(policy[0], np.array([False]))
    assert np.array_equal(policy[1], np.array([False, False]))
    assert np.array_equal(policy[2], np.array([True, True, True]))


def test_backward_induction_validation_errors() -> None:
    with pytest.raises(InvalidInputError):
        backward_induction_optimal_stopping(exercise_values=[], continuation_operator=lambda _j, v: v)

    with pytest.raises(InvalidInputError):
        backward_induction_optimal_stopping(  # type: ignore[arg-type]
            exercise_values=[np.array([0.0])],
            continuation_operator=None,
        )

    with pytest.raises(InvalidInputError):
        backward_induction_optimal_stopping(
            exercise_values=[np.array([0.0]), np.array([1.0]), np.array([0.0, 1.0, 2.0])],
            continuation_operator=lambda _j, v: v[:-1],
        )

    with pytest.raises(InvalidInputError):
        backward_induction_optimal_stopping(
            exercise_values=[np.array([0.0]), np.array([1.0, 2.0])],
            continuation_operator=lambda _j, _v: np.array([1.0, np.nan]),
        )


def test_spot_tree_shape_and_values() -> None:
    tree = build_recombining_spot_tree(spot=100.0, up=1.1, down=0.9, n_steps=3)
    assert [len(level) for level in tree] == [1, 2, 3, 4]

    assert tree[0][0] == pytest.approx(100.0)
    assert tree[2][1] == pytest.approx(100.0 * 1.1 * 0.9)
    assert tree[3][0] == pytest.approx(100.0 * 0.9**3)
    assert tree[3][3] == pytest.approx(100.0 * 1.1**3)


def test_spot_tree_validation_errors() -> None:
    with pytest.raises(InvalidInputError):
        build_recombining_spot_tree(spot=0.0, up=1.1, down=0.9, n_steps=2)
    with pytest.raises(InvalidInputError):
        build_recombining_spot_tree(spot=100.0, up=0.0, down=0.9, n_steps=2)
    with pytest.raises(InvalidInputError):
        build_recombining_spot_tree(spot=100.0, up=1.1, down=-0.9, n_steps=2)
    with pytest.raises(InvalidInputError):
        build_recombining_spot_tree(spot=100.0, up=1.1, down=0.9, n_steps=0)


def test_one_step_american_put_matches_manual_bellman() -> None:
    strike = 100.0
    expiry = 1.0
    market = _market(spot=95.0, r=0.05, q=0.0)
    model = BlackScholesModel(sigma=0.2)
    cfg = BinomialDPConfig(n_steps=1)

    result = price_american_put_binomial(
        strike=strike,
        expiry=expiry,
        market=market,
        model=model,
        cfg=cfg,
    )

    dt = expiry
    u = math.exp(model.sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((0.05 - 0.0) * dt) - d) / (u - d)
    s_up = 95.0 * u
    s_dn = 95.0 * d
    payoff_up = max(strike - s_up, 0.0)
    payoff_dn = max(strike - s_dn, 0.0)
    continuation = math.exp(-0.05 * dt) * (p * payoff_up + (1.0 - p) * payoff_dn)
    manual = max(max(strike - 95.0, 0.0), continuation)

    assert result.value == pytest.approx(manual, abs=1e-12)


def test_american_put_price_bounds_and_monotonicity() -> None:
    strike = 100.0
    expiry = 1.0
    model = BlackScholesModel(sigma=0.2)
    cfg = BinomialDPConfig(n_steps=200)

    market = _market(spot=100.0, r=0.05, q=0.01)
    american = price_american_put_binomial(
        strike=strike,
        expiry=expiry,
        market=market,
        model=model,
        cfg=cfg,
    ).value
    european = price(
        EuropeanOption(kind="put", strike=strike, expiry=expiry),
        model,
        market,
        method="analytic",
    ).value
    assert american >= european - 1e-12

    spots = [90.0, 100.0, 110.0]
    spot_prices = [
        price_american_put_binomial(
            strike=strike,
            expiry=expiry,
            market=_market(spot=s, r=0.05, q=0.01),
            model=model,
            cfg=cfg,
        ).value
        for s in spots
    ]
    assert spot_prices[0] >= spot_prices[1] >= spot_prices[2]

    strikes = [90.0, 100.0, 110.0]
    strike_prices = [
        price_american_put_binomial(
            strike=k,
            expiry=expiry,
            market=market,
            model=model,
            cfg=cfg,
        ).value
        for k in strikes
    ]
    assert strike_prices[0] <= strike_prices[1] <= strike_prices[2]


def test_american_put_determinism_and_lattice_payload() -> None:
    market = _market(spot=100.0, r=0.05, q=0.01)
    model = BlackScholesModel(sigma=0.2)
    cfg = BinomialDPConfig(n_steps=40)

    result_a = price_american_put_binomial(
        strike=100.0,
        expiry=1.0,
        market=market,
        model=model,
        cfg=cfg,
        return_lattice=True,
    )
    result_b = price_american_put_binomial(
        strike=100.0,
        expiry=1.0,
        market=market,
        model=model,
        cfg=cfg,
        return_lattice=True,
    )

    assert result_a.value == result_b.value
    assert result_a.meta is not None
    assert result_a.meta["method"] == "dp_binomial"
    assert result_a.meta["p"] == pytest.approx(result_b.meta["p"])  # type: ignore[index]

    spot_tree = result_a.meta["spot_tree"]  # type: ignore[index]
    exercise_policy = result_a.meta["exercise_policy"]  # type: ignore[index]
    assert len(spot_tree) == cfg.n_steps + 1
    assert len(exercise_policy) == cfg.n_steps + 1
    assert spot_tree[0].shape == (1,)
    assert exercise_policy[0].dtype == bool


def test_american_put_validation_and_no_arbitrage_failure() -> None:
    market = _market(spot=100.0, r=0.05, q=0.01)
    model = BlackScholesModel(sigma=0.2)

    with pytest.raises(InvalidInputError):
        price_american_put_binomial(
            strike=0.0,
            expiry=1.0,
            market=market,
            model=model,
            cfg=BinomialDPConfig(n_steps=50),
        )
    with pytest.raises(InvalidInputError):
        price_american_put_binomial(
            strike=100.0,
            expiry=-1.0,
            market=market,
            model=model,
            cfg=BinomialDPConfig(n_steps=50),
        )
    with pytest.raises(InvalidInputError):
        price_american_put_binomial(
            strike=100.0,
            expiry=1.0,
            market=market,
            model=model,
            cfg=BinomialDPConfig(n_steps=0),
        )

    with pytest.raises(InvalidInputError):
        price_american_put_binomial(
            strike=100.0,
            expiry=1.0,
            market=_market(spot=100.0, r=1.0, q=0.0),
            model=BlackScholesModel(sigma=0.01),
            cfg=BinomialDPConfig(n_steps=1),
        )
