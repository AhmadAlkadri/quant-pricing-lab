"""American exercise on the CRR lattice, through the dispatcher.

Everything here goes in by the public route,
``price(AmericanOption(...), model, market, method="tree", cfg=TreeConfig(...))``,
because that route is part of what the slice delivers: exercise style is an
instrument property and the registry resolves it (ADR-0005).

The measured-convergence evidence lives in
`tests/test_tree_american_convergence.py`; this file covers what is exact.
"""

from __future__ import annotations

import math

import pytest

from qpl.engines.mc.pricers import MCConfig
from qpl.engines.pde.pricers import PDEConfig
from qpl.engines.tree import TreeConfig
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import AmericanOption, EuropeanOption
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


def _american(
    kind: str,
    *,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
    n_steps: int,
):
    return price(
        AmericanOption(kind=kind, strike=strike, expiry=expiry),  # type: ignore[arg-type]
        BlackScholesModel(sigma=sigma),
        _market(spot, rate, div),
        method="tree",
        cfg=TreeConfig(n_steps=n_steps),
    )


# --------------------------------------------------------------------------
# The Slice 1 dynamic program, moved onto the dispatcher
# --------------------------------------------------------------------------

# Values produced by `qpl.engines.dp.price_american_put_binomial` before that
# keyword-based entry point was retired in favour of
# `qpl.engines.tree.price_american`, recorded to full double precision. The
# Bellman step is unchanged expression by expression -- the continuation is
# still `disc * (p * V[1:] + (1 - p) * V[:-1])` and the step is still
# `np.maximum(intrinsic, continuation)` -- so these are expected to agree
# exactly, not merely to 1e-12.
_DP_PRE_REFACTOR = (
    ("atm_1y_q1_n200", 100.0, 100.0, 1.0, 0.05, 0.01, 0.2, 200, 6.362744790336522),
    ("otm_spot95_n1", 95.0, 100.0, 1.0, 0.05, 0.0, 0.2, 1, 8.930470562349573),
    ("itm_short_n401", 120.0, 90.0, 0.5, 0.03, 0.05, 0.35, 401, 1.6458008994012916),
    ("lowvol_2y_n1000", 100.0, 100.0, 2.0, 0.0, 0.0, 0.1, 1000, 5.635788658985444),
    ("zerovol_n200", 100.0, 100.0, 1.0, 0.05, 0.01, 0.0, 200, 0.0),
)


@pytest.mark.parametrize(
    ("spot", "strike", "expiry", "rate", "div", "sigma", "n_steps", "expected"),
    [row[1:] for row in _DP_PRE_REFACTOR],
    ids=[row[0] for row in _DP_PRE_REFACTOR],
)
def test_american_put_matches_the_retired_dp_engine_bit_for_bit(
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
    n_steps: int,
    expected: float,
) -> None:
    """Evidence class: EXACT_IDENTITY (a pinned refactoring invariant).

    Retiring `qpl.engines.dp.price_american_put_binomial` in favour of an
    instrument-dispatched engine must be a pure re-plumbing. The tolerance is
    zero: anything but bit equality means the numerics moved, not just the
    entry point.
    """
    assert _american(
        "put",
        spot=spot,
        strike=strike,
        expiry=expiry,
        rate=rate,
        div=div,
        sigma=sigma,
        n_steps=n_steps,
    ).value == expected


def test_american_put_price_bounds_and_monotonicity() -> None:
    common = {"strike": 100.0, "expiry": 1.0, "rate": 0.05, "div": 0.01, "sigma": 0.2}
    european = price(
        EuropeanOption(kind="put", strike=100.0, expiry=1.0),
        BlackScholesModel(sigma=0.2),
        _market(100.0, 0.05, 0.01),
        method="analytic",
    ).value
    american = _american("put", spot=100.0, n_steps=200, **common).value
    assert american >= european - 1e-12

    spot_prices = [
        _american("put", spot=s, n_steps=200, **common).value for s in (90.0, 100.0, 110.0)
    ]
    assert spot_prices[0] >= spot_prices[1] >= spot_prices[2]

    strike_prices = [
        _american(
            "put", spot=100.0, strike=k, expiry=1.0, rate=0.05, div=0.01, sigma=0.2, n_steps=200
        ).value
        for k in (90.0, 100.0, 110.0)
    ]
    assert strike_prices[0] <= strike_prices[1] <= strike_prices[2]


def test_american_put_is_deterministic_and_reports_its_lattice() -> None:
    kwargs = dict(spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.01, sigma=0.2, n_steps=40)
    a = _american("put", **kwargs)
    b = _american("put", **kwargs)
    assert a.value == b.value
    assert a.stderr is None

    meta = a.meta
    assert meta is not None
    assert meta["method"] == "tree"
    assert meta["exercise"] == "american"
    assert meta["n_steps"] == 40
    assert meta["dt"] == pytest.approx(1.0 / 40)
    assert meta["early_exercise_node_count"] > 0
    assert len(meta["exercise_boundary"]) == 41


def test_american_config_and_no_arbitrage_validation() -> None:
    with pytest.raises(InvalidInputError, match="n_steps must be >= 1"):
        _american(
            "put", spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.0, sigma=0.2, n_steps=0
        )
    with pytest.raises(InvalidInputError, match="No-arbitrage"):
        _american(
            "put", spot=100.0, strike=100.0, expiry=1.0, rate=1.0, div=0.0, sigma=0.01, n_steps=1
        )
    with pytest.raises(InvalidInputError, match="strike must be > 0"):
        AmericanOption(kind="put", strike=0.0, expiry=1.0)
    with pytest.raises(InvalidInputError, match="expiry must be >= 0"):
        AmericanOption(kind="put", strike=100.0, expiry=-1.0)


# --------------------------------------------------------------------------
# The registry says which engines can price early exercise
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "cfg"),
    [("analytic", None), ("mc", MCConfig(n_paths=64, n_steps=1, seed=1)), ("pde", PDEConfig())],
)
def test_non_tree_methods_reject_american_exercise(method: str, cfg: object) -> None:
    """Evidence class: EXACT_IDENTITY (an API contract, not a number).

    None of the analytic, Monte Carlo or PDE engines implements an
    early-exercise rule, and none of them is registered for `AmericanOption`.
    The refusal therefore comes out of the ordinary registry lookup with the
    ordinary message -- no engine needs an `if instrument.american: raise`.
    """
    option = AmericanOption(kind="put", strike=100.0, expiry=1.0)
    kwargs = {} if cfg is None else {"cfg": cfg}
    with pytest.raises(NotSupportedError, match="Unsupported instrument/model/market combination"):
        price(option, BlackScholesModel(sigma=0.2), _market(100.0, 0.05, 0.0), method=method, **kwargs)


def test_american_is_not_a_european_subclass() -> None:
    """Registry lookup walks the MRO, so inheriting would silently mis-price.

    If `AmericanOption` subclassed `EuropeanOption`, an American option handed
    to `method="analytic"` would resolve to the European closed form and return
    a number instead of refusing.
    """
    assert not issubclass(AmericanOption, EuropeanOption)
    assert not issubclass(EuropeanOption, AmericanOption)
    assert AmericanOption(kind="put", strike=100.0, expiry=1.0) != EuropeanOption(
        kind="put", strike=100.0, expiry=1.0
    )


def test_expiry_zero_is_intrinsic() -> None:
    """Evidence class: CLOSED_FORM. No time, no lattice, no optionality."""
    for kind, spot, expected in (("put", 95.0, 5.0), ("put", 105.0, 0.0), ("call", 105.0, 5.0)):
        result = _american(
            kind, spot=spot, strike=100.0, expiry=0.0, rate=0.05, div=0.01, sigma=0.2, n_steps=50
        )
        assert result.value == pytest.approx(expected, abs=1e-12)
        assert result.meta is not None
        assert result.meta["degenerate"] == "expiry"


def test_american_put_never_worth_less_than_intrinsic() -> None:
    for spot in (60.0, 80.0, 100.0, 120.0):
        value = _american(
            "put",
            spot=spot,
            strike=100.0,
            expiry=1.0,
            rate=0.05,
            div=0.01,
            sigma=0.2,
            n_steps=200,
        ).value
        assert value >= max(100.0 - spot, 0.0) - 1e-12


def test_sigma_zero_put_with_no_dividend_is_the_better_of_two_endpoints() -> None:
    """Evidence class: CLOSED_FORM.

    With `sigma = 0` the spot follows its forward exactly, so the value is a
    maximisation over the exercise date alone. With `q <= r` the objective
    `K e^{-rt} - S0 e^{-qt}` has no interior maximum, so the answer is
    `max(intrinsic now, discounted forward intrinsic)`.
    """
    spot, strike, expiry, rate, div = 95.0, 100.0, 1.0, 0.05, 0.0
    forward = spot * math.exp((rate - div) * expiry)
    expected = max(strike - spot, math.exp(-rate * expiry) * max(strike - forward, 0.0))
    value = _american(
        "put",
        spot=spot,
        strike=strike,
        expiry=expiry,
        rate=rate,
        div=div,
        sigma=0.0,
        n_steps=137,
    ).value
    assert value == pytest.approx(expected, abs=1e-12)
    assert value == pytest.approx(5.0, abs=1e-12)


def test_deprecated_dp_wrapper_returns_the_engine_value() -> None:
    """The Slice 1 keyword entry point is now a wrapper, not a second engine.

    Evidence class: EXACT_IDENTITY. It forwards to the dispatcher engine, so
    the values are the same object's output, not two implementations that
    happen to agree.
    """
    from qpl.engines.dp import BinomialDPConfig, price_american_put_binomial

    market = _market(100.0, 0.05, 0.01)
    model = BlackScholesModel(sigma=0.2)
    wrapped = price_american_put_binomial(
        strike=100.0,
        expiry=1.0,
        market=market,
        model=model,
        cfg=BinomialDPConfig(n_steps=60),
        return_lattice=True,
    )
    direct = _american(
        "put", spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.01, sigma=0.2, n_steps=60
    )
    assert wrapped.value == direct.value

    meta = wrapped.meta or {}
    assert len(meta["spot_tree"]) == 61
    assert len(meta["exercise_policy"]) == 61
