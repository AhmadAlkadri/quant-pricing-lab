"""CRR tree engine: configuration, degenerate limits, metadata, and the
bit-for-bit invariance of the American-put DP engine under the shared lattice.

The measured-convergence evidence lives in `tests/test_tree_convergence.py`;
this file covers the parts that are exact rather than asymptotic.
"""

from __future__ import annotations

import math

import pytest

from qpl.engines.dp import BinomialDPConfig, price_american_put_binomial
from qpl.engines.tree import TreeConfig, crr_parameters, crr_spot_level
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


def _tree(option: EuropeanOption, sigma: float, market: Market, n: int) -> float:
    return price(
        option, BlackScholesModel(sigma=sigma), market, method="tree", cfg=TreeConfig(n_steps=n)
    ).value


# --------------------------------------------------------------------------
# Lattice construction
# --------------------------------------------------------------------------


def test_crr_parameters_match_the_definition() -> None:
    """Evidence class: EXACT_IDENTITY (the defining formulas, recomputed here).

    u = exp(sigma sqrt(dt)), d = 1/u, p = (exp((r-q) dt) - d) / (u - d).
    """
    sigma, expiry, r, q, n = 0.2, 1.0, 0.05, 0.01, 40
    lattice = crr_parameters(
        sigma=sigma, expiry=expiry, rate=r, dividend_yield=q, n_steps=n
    )

    dt = expiry / n
    u = math.exp(sigma * math.sqrt(dt))
    assert lattice.dt == pytest.approx(dt, rel=0, abs=0)
    assert lattice.up == pytest.approx(u, rel=0, abs=0)
    assert lattice.down == pytest.approx(1.0 / u, rel=0, abs=0)
    assert lattice.up * lattice.down == pytest.approx(1.0, abs=1e-15)
    assert lattice.p == pytest.approx(
        (math.exp((r - q) * dt) - 1.0 / u) / (u - 1.0 / u), abs=1e-15
    )
    assert lattice.discount == pytest.approx(math.exp(-r * dt), rel=0, abs=0)
    assert not lattice.degenerate


def test_crr_no_arbitrage_violation_raises() -> None:
    """A coarse tree with a drift larger than the vol scale has no valid p.

    p in [0, 1] requires d <= e^{(r-q) dt} <= u. With sigma = 1% and one step
    of a year at r = 100%, the growth factor is far above u.
    """
    with pytest.raises(InvalidInputError, match="No-arbitrage"):
        crr_parameters(sigma=0.01, expiry=1.0, rate=1.0, dividend_yield=0.0, n_steps=1)


def test_crr_spot_level_is_sorted_and_recombines() -> None:
    lattice = crr_parameters(
        sigma=0.25, expiry=1.0, rate=0.02, dividend_yield=0.0, n_steps=8
    )
    level = crr_spot_level(spot=100.0, up=lattice.up, down=lattice.down, level=4)
    assert level.shape == (5,)
    assert list(level) == sorted(level)
    # d = 1/u, so an even level has the initial spot exactly at its centre.
    assert level[2] == pytest.approx(100.0, abs=1e-12)


def test_tree_config_validation() -> None:
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.0)

    with pytest.raises(InvalidInputError, match="n_steps must be >= 1"):
        price(option, model, market, method="tree", cfg=TreeConfig(n_steps=0))
    with pytest.raises(InvalidInputError, match="scheme must be 'crr'"):
        price(
            option,
            model,
            market,
            method="tree",
            cfg=TreeConfig(n_steps=10, scheme="lr"),  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------
# Degenerate limits, shared with the analytic / MC / PDE engines
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "spot", "expected"),
    [("call", 105.0, 5.0), ("call", 95.0, 0.0), ("put", 95.0, 5.0), ("put", 105.0, 0.0)],
)
def test_expiry_zero_is_intrinsic(kind: str, spot: float, expected: float) -> None:
    """Evidence class: CLOSED_FORM. At T=0 there is no lattice; the price is
    the payoff, as in every other engine."""
    option = EuropeanOption(kind=kind, strike=100.0, expiry=0.0)  # type: ignore[arg-type]
    assert _tree(option, 0.2, _market(spot, 0.05, 0.01), 50) == pytest.approx(
        expected, abs=1e-12
    )


@pytest.mark.parametrize("kind", ["call", "put"])
def test_zero_vol_is_discounted_forward_intrinsic(kind: str) -> None:
    """Evidence class: CLOSED_FORM. With sigma=0 the spot is deterministic and
    equals its forward, so the price is the discounted intrinsic of that
    forward -- matching the analytic, MC and PDE engines."""
    spot, k, t, r, q = 110.0, 100.0, 1.0, 0.05, 0.02
    option = EuropeanOption(kind=kind, strike=k, expiry=t)  # type: ignore[arg-type]
    forward = spot * math.exp((r - q) * t)
    payoff = max(forward - k, 0.0) if kind == "call" else max(k - forward, 0.0)
    expected = math.exp(-r * t) * payoff

    assert _tree(option, 0.0, _market(spot, r, q), 137) == pytest.approx(expected, abs=1e-12)


def test_meta_reports_the_realised_lattice() -> None:
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    result = price(
        option,
        BlackScholesModel(sigma=0.2),
        _market(100.0, 0.05, 0.0),
        method="tree",
        cfg=TreeConfig(n_steps=64),
    )
    meta = result.meta
    assert meta is not None
    assert meta["method"] == "tree"
    assert meta["scheme"] == "crr"
    assert meta["n_steps"] == 64
    assert meta["dt"] == pytest.approx(1.0 / 64)
    assert float(meta["u"]) * float(meta["d"]) == pytest.approx(1.0, abs=1e-15)
    assert 0.0 <= float(meta["p"]) <= 1.0
    assert result.stderr is None


def test_tree_is_deterministic() -> None:
    option = EuropeanOption(kind="put", strike=105.0, expiry=0.75)
    market = _market(100.0, 0.03, 0.01)
    a = _tree(option, 0.3, market, 333)
    b = _tree(option, 0.3, market, 333)
    assert a == b


# --------------------------------------------------------------------------
# The DP engine now consumes the shared lattice: it must not have moved.
# --------------------------------------------------------------------------

# Values produced by the American-put DP engine *before* it was rewritten onto
# `qpl.engines.tree.lattice`, recorded to full double precision. The shared
# builder performs the same arithmetic in the same order, so these are expected
# to agree exactly, not merely to 1e-12.
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
def test_american_put_dp_is_unchanged_by_the_shared_lattice(
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

    Moving the CRR parameter computation and the spot-lattice builder out of
    `qpl.engines.dp` and into `qpl.engines.tree.lattice` must be a pure
    refactor. The tolerance is zero: the arithmetic is unchanged expression by
    expression, so anything but bit equality means the lattice moved.
    """
    value = price_american_put_binomial(
        strike=strike,
        expiry=expiry,
        market=_market(spot, rate, div),
        model=BlackScholesModel(sigma=sigma),
        cfg=BinomialDPConfig(n_steps=n_steps),
    ).value
    assert value == expected
