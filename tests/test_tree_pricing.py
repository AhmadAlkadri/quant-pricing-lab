"""CRR tree engine: configuration, degenerate limits, metadata, lattice Greeks.

The measured-convergence evidence lives in `tests/test_tree_convergence.py`,
and everything about American exercise in `tests/test_tree_american.py`; this
file covers the European parts that are exact rather than asymptotic.
"""

from __future__ import annotations

import math

import pytest

from qpl.engines.tree import TreeConfig, crr_parameters, crr_spot_level
from qpl.exceptions import InvalidInputError
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price
from qpl.validation import fit_convergence_order


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
    with pytest.raises(InvalidInputError, match="scheme must be one of"):
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
# Lattice Greeks
# --------------------------------------------------------------------------

# Tolerances below are taken from measurement, not from habit. Worst absolute
# residual against the closed form over `n` in [1990, 2010] (the sweep covers
# both parities, so it captures the oscillation rather than one lucky `n`), on
# the three specification points used here:
#
#   greek   worst measured   tolerance   headroom
#   delta   6.3e-05          2e-04       3.2x
#   gamma   8.0e-06          3e-05       3.8x
#   theta   1.6e-03          5e-03       3.1x
#   vega    7.1e-02          2.5e-01     3.5x
#   rho     3.9e-03          1.2e-02     3.1x
#
# Delta, gamma and theta are read straight off the lattice and converge at
# measured order 1 in 1/n (ATM call, orders 1.004 / 1.002 / 1.001 on odd n and
# 0.999 / 1.010 / 1.010 on even n). Vega and rho are bump-and-revalue and are
# two to three orders of magnitude coarser relative to their own size, because
# bumping sigma moves the whole lattice and the tree's oscillating price error
# does not cancel between the two evaluations; see the bump constants in
# `qpl.engines.tree.pricers`.

_GREEK_TOLERANCES = {
    "delta": 2e-4,
    "gamma": 3e-5,
    "theta": 5e-3,
    "vega": 2.5e-1,
    "rho": 1.2e-2,
}

_GREEK_POINTS = (
    ("atm_1y", 100.0, 100.0, 1.0, 0.05, 0.00, 0.20),
    ("otm_9m_div", 100.0, 110.0, 0.75, 0.03, 0.01, 0.25),
    ("itm_6m_div", 120.0, 90.0, 0.5, 0.03, 0.05, 0.35),
)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(
    ("spot", "strike", "expiry", "rate", "div", "sigma"),
    [row[1:] for row in _GREEK_POINTS],
    ids=[row[0] for row in _GREEK_POINTS],
)
def test_tree_greeks_match_analytic(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: CLOSED_FORM, at `n_steps = 2000`.

    See the table above this test for where each tolerance comes from.
    """
    option = EuropeanOption(kind=kind, strike=strike, expiry=expiry)  # type: ignore[arg-type]
    model = BlackScholesModel(sigma=sigma)
    market = _market(spot, rate, div)

    exact = greeks(option, model, market, method="analytic")
    tree = greeks(option, model, market, method="tree", cfg=TreeConfig(n_steps=2000))

    for name, tol in _GREEK_TOLERANCES.items():
        assert getattr(tree, name) == pytest.approx(getattr(exact, name), abs=tol), name


def test_tree_greeks_converge_at_order_one() -> None:
    """Evidence class: CONVERGENCE_ORDER.

    Delta, gamma and theta come from nodes at steps 1 and 2, one and two `dt`
    away from valuation time, so their error inherits the `O(1/n)` of the
    price rather than improving on it. Odd and even `n` are fitted separately
    for the same reason the price is (see `tests/test_tree_convergence.py`);
    the band [0.8, 1.2] is the same one used for the price.
    """
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.0)
    exact = greeks(option, model, market, method="analytic")

    for levels in ((25, 51, 101, 201, 401, 801), (26, 50, 100, 200, 400, 800)):
        measured = [
            greeks(option, model, market, method="tree", cfg=TreeConfig(n_steps=n))
            for n in levels
        ]
        h = [1.0 / n for n in levels]
        for name in ("delta", "gamma", "theta"):
            errs = [abs(getattr(g, name) - getattr(exact, name)) for g in measured]
            fit = fit_convergence_order(h, errs)
            assert 0.8 <= fit.order <= 1.2, (name, levels, fit.order)
            assert fit.residual < 0.05, (name, levels, fit.residual)


def test_tree_greeks_metadata_and_validation() -> None:
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.2)
    market = _market(100.0, 0.05, 0.0)

    result = greeks(option, model, market, method="tree", cfg=TreeConfig(n_steps=64))
    assert result.meta is not None
    assert result.meta["method"] == "tree"
    assert result.meta["bumps"] == {"sigma": 1e-2, "r": 1e-4}

    # Gamma and theta read step-2 nodes, so a one-step tree cannot supply them.
    with pytest.raises(InvalidInputError, match="n_steps must be >= 2"):
        greeks(option, model, market, method="tree", cfg=TreeConfig(n_steps=1))

    # sigma = 0 collapses the lattice to a single path: no gamma to read.
    with pytest.raises(InvalidInputError, match="sigma must be > 0 for tree Greeks"):
        greeks(
            option,
            BlackScholesModel(sigma=0.0),
            market,
            method="tree",
            cfg=TreeConfig(n_steps=64),
        )

    # T = 0 is the flat case shared with the MC engine: every Greek is zero.
    expired = EuropeanOption(kind="call", strike=100.0, expiry=0.0)
    flat = greeks(expired, model, market, method="tree", cfg=TreeConfig(n_steps=64))
    assert (flat.delta, flat.gamma, flat.vega, flat.theta, flat.rho) == (0.0,) * 5
