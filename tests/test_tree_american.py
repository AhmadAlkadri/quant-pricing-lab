"""American exercise on the CRR lattice: the parts that are exact.

Everything here goes in by the public route,
``price(AmericanOption(...), model, market, method="tree", cfg=TreeConfig(...))``,
because that route is part of what the slice delivers: exercise style is an
instrument property and the registry resolves it (ADR-0005).

Claims and how each is justified:

(a) EXACT_IDENTITY -- a two-step American put equals a backward induction
    written out in plain arithmetic in the test, with no call to the engine.
(b) EXACT_IDENTITY -- American >= European on every lattice; the
    early-exercise premium is non-negative and increases with `K` for puts.
(c) EXACT_IDENTITY -- with `q = 0` an American call equals the European call
    *bit for bit* on the same lattice (Merton 1973: never exercise early); with
    `q > 0` it is strictly greater.
(g) CLOSED_FORM / EXACT_IDENTITY -- the extracted put boundary is monotone
    non-decreasing in calendar time within a node-grid parity, never falls by
    more than the offset between the two interleaved grids, and lands on the
    in-the-money node adjacent to `K` at expiry.

Plus the degenerate limits, the lattice Greeks, and the `NotSupportedError`
paths. The measured-convergence evidence is in
`tests/test_tree_american_convergence.py`.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from qpl.engines.mc.pricers import MCConfig
from qpl.engines.pde.pricers import PDEConfig
from qpl.engines.tree import TreeConfig
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import AmericanOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price


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


def _european_tree(
    kind: str,
    *,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
    n_steps: int,
) -> float:
    return price(
        EuropeanOption(kind=kind, strike=strike, expiry=expiry),  # type: ignore[arg-type]
        BlackScholesModel(sigma=sigma),
        _market(spot, rate, div),
        method="tree",
        cfg=TreeConfig(n_steps=n_steps),
    ).value


# --------------------------------------------------------------------------
# (a) Hand-computed backward induction
# --------------------------------------------------------------------------


def test_two_step_american_put_matches_hand_computed_backward_induction() -> None:
    """Evidence class: EXACT_IDENTITY.

    The expected value below is a two-step dynamic program written out in
    plain arithmetic -- five `max` calls and two discounted expectations, no
    loop, no numpy, no engine. The point of doing it by hand is that a sign
    error, an off-by-one in the node indexing, or a continuation value that
    forgot to discount would all survive a comparison against a finer run of
    the same engine and none of them survives this.

    The specification is chosen so that early exercise actually binds: the
    option is deep in the money and the rate is high, and the test asserts
    that at least one `max` picks the exercise branch rather than trusting
    that it does.
    """
    spot, strike, expiry, r, q, sigma = 90.0, 100.0, 1.0, 0.08, 0.0, 0.20

    dt = expiry / 2.0
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    # Level 2 (expiry): exercise value only.
    s_dd = spot * d * d
    s_ud = spot * u * d
    s_uu = spot * u * u
    v_dd = max(strike - s_dd, 0.0)
    v_ud = max(strike - s_ud, 0.0)
    v_uu = max(strike - s_uu, 0.0)

    # Level 1: Bellman.
    s_d = spot * d
    s_u = spot * u
    cont_d = disc * (p * v_ud + (1.0 - p) * v_dd)
    cont_u = disc * (p * v_uu + (1.0 - p) * v_ud)
    ex_d = max(strike - s_d, 0.0)
    ex_u = max(strike - s_u, 0.0)
    v_d = max(ex_d, cont_d)
    v_u = max(ex_u, cont_u)

    # Level 0: Bellman again.
    cont_0 = disc * (p * v_u + (1.0 - p) * v_d)
    ex_0 = max(strike - spot, 0.0)
    expected = max(ex_0, cont_0)

    # Not a vacuous test: somewhere the exercise branch wins.
    assert (ex_d > cont_d) or (ex_u > cont_u) or (ex_0 > cont_0)

    engine = _american(
        "put",
        spot=spot,
        strike=strike,
        expiry=expiry,
        rate=r,
        div=q,
        sigma=sigma,
        n_steps=2,
    ).value
    assert engine == pytest.approx(expected, abs=1e-12)


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


def test_deprecated_dp_wrapper_returns_the_engine_value() -> None:
    """The Slice 1 keyword entry point is now a wrapper, not a second engine.

    Evidence class: EXACT_IDENTITY. It forwards to the dispatcher engine, so
    the values are the same function's output, not two implementations that
    happen to agree.
    """
    from qpl.engines.dp import BinomialDPConfig, price_american_put_binomial

    wrapped = price_american_put_binomial(
        strike=100.0,
        expiry=1.0,
        market=_market(100.0, 0.05, 0.01),
        model=BlackScholesModel(sigma=0.2),
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


# --------------------------------------------------------------------------
# (b) American >= European, and the early-exercise premium
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("n_steps", [1, 2, 3, 5, 11, 40, 101, 500])
def test_american_is_never_worth_less_than_european_on_the_same_lattice(
    kind: str, n_steps: int
) -> None:
    """Evidence class: EXACT_IDENTITY.

    An American holder may always choose to do nothing until expiry, so the
    American value dominates the European one path by path -- and on a lattice
    that domination is structural: `max(intrinsic, continuation) >=
    continuation` at every node, and the continuation recursion is the European
    one. The inequality therefore holds at *every* `n`, coarse trees included,
    and is not an asymptotic statement. Zero slack is allowed.
    """
    spec = dict(spot=100.0, strike=105.0, expiry=1.0, rate=0.05, div=0.03, sigma=0.25)
    american = _american(kind, n_steps=n_steps, **spec).value
    european = _european_tree(kind, n_steps=n_steps, **spec)
    assert american >= european


_PREMIUM_STRIKES = (60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0, 160.0)


def test_put_early_exercise_premium_is_non_negative_and_increases_with_strike() -> None:
    """Evidence class: EXACT_IDENTITY (an ordering, measured on the lattice).

    The premium is what the right to exercise early is worth. For a put it is
    financed by interest earned on the strike received, so it grows with `K`:
    measured at `S = 100, T = 1, r = 5%, q = 1%, sigma = 20%, n = 500`, against
    the European price on the same lattice,

        K       60      80       90       100      110      120      140      160
        prem  3.0e-04  3.1e-02  1.4e-01  4.3e-01  1.0e+00  2.2e+00  5.1e+00  6.7e+00

    strictly increasing across the row. The assertion below is
    non-decreasing with zero slack, which is the claim the model actually
    supports; the measured numbers are strictly increasing with room to spare.
    """
    spec = dict(spot=100.0, expiry=1.0, rate=0.05, div=0.01, sigma=0.2, n_steps=500)
    premia = [
        _american("put", strike=k, **spec).value - _european_tree("put", strike=k, **spec)
        for k in _PREMIUM_STRIKES
    ]
    assert all(premium >= 0.0 for premium in premia), premia
    assert all(b >= a for a, b in pairwise(premia)), premia
    # And it is not a flat line: the right to exercise early is worth something.
    assert premia[-1] > 1.0


# --------------------------------------------------------------------------
# (c) The American call on a non-dividend-paying stock
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_steps", [1, 2, 7, 50, 201, 1000])
def test_american_call_without_dividends_equals_the_european_call_exactly(
    n_steps: int,
) -> None:
    """Evidence class: EXACT_IDENTITY. Merton (1973), Bell Journal of
    Economics and Management Science 4(1), 141-183.

    With `q = 0` and `r >= 0` it is never optimal to exercise a call early, so
    the American and European values coincide. On the lattice the argument is
    finite and checkable: at a node `tau` from expiry the European call value
    is a discounted expectation of `max(S_T - K, 0) >= S_T - K`, and the tree
    reprices the forward exactly, so

        C_euro(S, tau) >= e^{-r tau} (S e^{(r - q) tau} - K)
                        = S e^{-q tau} - K e^{-r tau}
                        = S - K e^{-r tau}      (q = 0)
                       >= S - K,                 (r >= 0)

    and also `C_euro >= 0`. So `max(intrinsic, continuation) = continuation` at
    every node and the Bellman maximum is a no-op.

    A no-op in exact arithmetic is a no-op in floating point too -- `max(a, b)`
    returns `b` itself, not a rounded copy of it -- so the two engines agree to
    the last bit, and the tolerance here is zero rather than 1e-12.
    """
    spec = dict(spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.0, sigma=0.2)
    assert _american("call", n_steps=n_steps, **spec).value == _european_tree(
        "call", n_steps=n_steps, **spec
    )


@pytest.mark.parametrize("n_steps", [50, 201])
def test_american_call_with_dividends_strictly_exceeds_the_european_call(
    n_steps: int,
) -> None:
    """Evidence class: EXACT_IDENTITY (a strict inequality, not a tolerance).

    The bound above needs `q = 0`: with `q > 0` the holder forgoes the dividend
    yield by waiting, and `S e^{-q tau} - K e^{-r tau}` can fall below
    `S - K`. Measured excess at `S = K = 100, T = 1, r = 5%, q = 6%,
    sigma = 20%`: 1.93e-01 at `n = 50` and 1.81e-01 at `n = 201`.
    """
    spec = dict(spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.06, sigma=0.2)
    american = _american("call", n_steps=n_steps, **spec).value
    european = _european_tree("call", n_steps=n_steps, **spec)
    assert american > european + 1e-3


# --------------------------------------------------------------------------
# (g) The exercise boundary
# --------------------------------------------------------------------------

_BOUNDARY_SPEC = dict(
    spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.01, sigma=0.2, n_steps=200
)

_BOUNDARY_ROUNDOFF = 1e-9
"""Slack on the boundary's monotonicity, in spot units.

A flat stretch of the boundary is the *same* spot level reported at successive
levels of the lattice, but computed as `S0 u^i d^(j-i)` with different `(i, j)`
each time. Those products agree to about 3e-14 on a spot of 100, not exactly,
so the measured differences along a flat stretch are +-2.8e-14 rather than 0.
1e-9 is a round-off budget nine orders of magnitude below the smallest
genuine step (2.256 at this `n`, since the boundary moves by whole nodes), so
it cannot hide a boundary that actually moved the wrong way.
"""


def _boundary(kind: str) -> np.ndarray:
    result = _american(kind, **_BOUNDARY_SPEC)
    assert result.meta is not None
    return np.asarray(result.meta["exercise_boundary"], dtype=float)


def test_put_exercise_boundary_reaches_the_strike_at_expiry() -> None:
    """Evidence class: CLOSED_FORM, up to node spacing.

    At expiry every in-the-money node is exercised, so the boundary is the
    largest node strictly below `K`. On an even level the nodes are
    `K u^{2m}` (because `d = 1/u` and `S0 = K` here), so that node is exactly
    `K / u**2` and the gap to the strike is one terminal-level node spacing --
    2.79 on a strike of 100 at `n = 200`, where `K (1 - u^{-2}) = 2.789`. The
    node *at* the strike is excluded on purpose: it is in the money by nothing,
    and including it would make the answer depend on whether
    `S0 u**100 d**100` rounds to just above or just below `K` (see
    `_INTRINSIC_FLOOR_REL` in `qpl.engines.tree.american`).
    """
    boundary = _boundary("put")
    strike = _BOUNDARY_SPEC["strike"]
    n_steps = _BOUNDARY_SPEC["n_steps"]
    u = math.exp(_BOUNDARY_SPEC["sigma"] * math.sqrt(_BOUNDARY_SPEC["expiry"] / n_steps))

    assert boundary.shape == (n_steps + 1,)
    assert boundary[-1] == pytest.approx(strike / (u * u), abs=1e-10)
    assert boundary[-1] < strike
    assert strike - boundary[-1] <= strike * (1.0 - 1.0 / (u * u)) + 1e-12


def test_put_exercise_boundary_rises_toward_the_strike() -> None:
    """Evidence class: EXACT_IDENTITY (an ordering), with the honest caveat.

    The continuous-time boundary `B(t)` of an American put is non-decreasing
    and tends to `K` as `t -> T`. The *extracted* boundary is not monotone
    level by level, and saying it is would be false: levels of the same parity
    share a node grid (`S0 u^{2i-j}`), and consecutive levels use the two
    interleaved grids, so the reported boundary alternates between them and
    can drop by up to the offset between the grids.

    What is measured and asserted instead, which is the same statement without
    the discretisation artefact:

    - each parity subsequence (`boundary[0::2]`, `boundary[1::2]`) is monotone
      non-decreasing, up to a round-off budget: the same economic node reached
      at two different levels is `S0 u^i d^(j-i)` for two different `(i, j)`,
      which are the same number only to about 3e-14 on a spot of 100, so a
      *flat* stretch of the boundary shows up as steps of +-2.8e-14;
    - no level-to-level step falls by more than the grid offset. Measured
      worst drop at `n = 200`: 1.3845, against an offset of
      `S (1 - 1/u) = 1.4043` near the strike.

    The boundary is NaN at early levels (the first 16 here) because the tree
    has not yet reached a node low enough for exercise to be optimal that
    early, not because the boundary does not exist there: it is below the
    lattice's own floor.
    """
    boundary = _boundary("put")
    strike = _BOUNDARY_SPEC["strike"]
    n_steps = _BOUNDARY_SPEC["n_steps"]
    u = math.exp(_BOUNDARY_SPEC["sigma"] * math.sqrt(_BOUNDARY_SPEC["expiry"] / n_steps))

    for parity in (0, 1):
        sub = boundary[parity::2]
        sub = sub[~np.isnan(sub)]
        assert sub.size > 10
        assert np.all(np.diff(sub) >= -_BOUNDARY_ROUNDOFF), (parity, sub)
        assert sub[-1] <= strike

    finite = boundary[~np.isnan(boundary)]
    drops = -np.diff(finite)
    # One fine-grid offset, evaluated at the largest boundary value reached.
    assert drops.max() <= finite.max() * (1.0 - 1.0 / u) + 1e-9

    # The NaN prefix is a prefix, not scattered holes: once exercise becomes
    # optimal it stays optimal at every later level.
    first_finite = int(np.flatnonzero(~np.isnan(boundary))[0])
    assert not np.any(np.isnan(boundary[first_finite:]))


def test_call_exercise_boundary_falls_toward_the_strike() -> None:
    """Evidence class: EXACT_IDENTITY (the mirror-image ordering).

    For a dividend-paying call the exercise region is `S >= B(t)` with `B`
    non-increasing toward `K`, so the extracted boundary is the *smallest*
    exercised node and each parity subsequence must be non-increasing.
    Measured at expiry: `K u**2 = 102.8688`, one terminal-level node spacing
    above the strike.
    """
    spec = dict(_BOUNDARY_SPEC, div=0.06)
    result = _american("call", **spec)
    assert result.meta is not None
    boundary = np.asarray(result.meta["exercise_boundary"], dtype=float)
    u = math.exp(spec["sigma"] * math.sqrt(spec["expiry"] / spec["n_steps"]))

    assert boundary[-1] == pytest.approx(spec["strike"] * u * u, abs=1e-10)
    for parity in (0, 1):
        sub = boundary[parity::2]
        sub = sub[~np.isnan(sub)]
        assert sub.size > 10
        assert np.all(np.diff(sub) <= _BOUNDARY_ROUNDOFF), (parity, sub)
        assert sub[-1] >= spec["strike"]


def test_early_exercise_node_count_tracks_the_boundary() -> None:
    result = _american("put", **_BOUNDARY_SPEC)
    assert result.meta is not None
    count = result.meta["early_exercise_node_count"]
    boundary = np.asarray(result.meta["exercise_boundary"], dtype=float)
    # Every level with a finite boundary before expiry contributes at least one
    # node, and no level contributes more nodes than it has.
    levels_with_exercise = int(np.count_nonzero(~np.isnan(boundary[:-1])))
    assert levels_with_exercise <= count <= sum(j + 1 for j in range(_BOUNDARY_SPEC["n_steps"]))

    # A call on a non-dividend-paying stock is never exercised early: no nodes,
    # and no boundary anywhere.
    never = _american(
        "call", spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.0, sigma=0.2, n_steps=200
    )
    assert never.meta is not None
    assert never.meta["early_exercise_node_count"] == 0
    assert np.all(np.isnan(np.asarray(never.meta["exercise_boundary"], dtype=float)[:-1]))


# --------------------------------------------------------------------------
# Degenerate limits
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "spot", "expected"),
    [("call", 105.0, 5.0), ("call", 95.0, 0.0), ("put", 95.0, 5.0), ("put", 105.0, 0.0)],
)
def test_expiry_zero_is_intrinsic(kind: str, spot: float, expected: float) -> None:
    """Evidence class: CLOSED_FORM. No time, no lattice, no optionality."""
    result = _american(
        kind, spot=spot, strike=100.0, expiry=0.0, rate=0.05, div=0.01, sigma=0.2, n_steps=50
    )
    assert result.value == pytest.approx(expected, abs=1e-12)
    assert result.meta is not None
    assert result.meta["degenerate"] == "expiry"
    assert result.meta["exercise_boundary"] is None


@pytest.mark.parametrize(
    ("kind", "spot", "strike", "rate", "div"),
    [
        ("put", 95.0, 100.0, 0.05, 0.00),
        ("put", 110.0, 100.0, 0.05, 0.02),
        ("call", 110.0, 100.0, 0.05, 0.00),
        ("call", 100.0, 105.0, 0.05, 0.02),
    ],
)
def test_sigma_zero_with_rate_above_yield_is_the_better_endpoint(
    kind: str, spot: float, strike: float, rate: float, div: float
) -> None:
    """Evidence class: CLOSED_FORM.

    With `sigma = 0` the spot follows its forward exactly, so the price is a
    maximisation over the exercise date alone, and with `r > q` the objective
    has no interior maximum for a put (nor for a call away from the money).
    The value is then `max(intrinsic now, discounted forward intrinsic)` --
    the European `sigma = 0` limit with the "exercise now" branch added, which
    is what makes it the American one.
    """
    expiry = 1.0
    forward = spot * math.exp((rate - div) * expiry)
    disc = math.exp(-rate * expiry)
    if kind == "call":
        now, later = max(spot - strike, 0.0), disc * max(forward - strike, 0.0)
    else:
        now, later = max(strike - spot, 0.0), disc * max(strike - forward, 0.0)

    value = _american(
        kind,
        spot=spot,
        strike=strike,
        expiry=expiry,
        rate=rate,
        div=div,
        sigma=0.0,
        n_steps=137,
    ).value
    assert value == pytest.approx(max(now, later), abs=1e-12)


def test_sigma_zero_put_can_beat_both_endpoints_when_the_yield_exceeds_the_rate() -> None:
    """Evidence class: NEGATIVE_FINDING (against this slice's own statement).

    The slice brief asserted that at `sigma = 0` an American put is worth
    `max(intrinsic now, discounted forward intrinsic)`. That is true whenever
    `r >= q` -- which includes every no-dividend case, so it is easy to believe
    -- and **false** when `q > r`.

    Why: the objective is `h(t) = K e^{-rt} - S0 e^{-qt}`, whose only turning
    point is at `t* = log(rK / (q S0)) / (r - q)` with
    `h''(t*) = (r - q) r K e^{-r t*}`. For `q > r` that is negative, so `t*` is
    an interior *maximum*, and when it lands inside `(0, T)` the best exercise
    date is neither now nor expiry.

    Measured here at `S0 = K = 100, r = 2%, q = 50%, T = 10`: `t* = 6.7060`
    and `h(t*) = 83.9506`, against `max(h(0), h(T)) = 81.1993`. The engine
    returns 83.9506; an endpoints-only rule would be 2.75 too low, which is
    3.4% of the price and nothing like round-off.
    """
    spot, strike, expiry, rate, div = 100.0, 100.0, 10.0, 0.02, 0.50

    def h(t: float) -> float:
        return strike * math.exp(-rate * t) - spot * math.exp(-div * t)

    turning = math.log((rate * strike) / (div * spot)) / (rate - div)
    assert 0.0 < turning < expiry
    endpoints = max(0.0, h(0.0), h(expiry))
    interior = h(turning)
    assert interior > endpoints

    value = _american(
        "put",
        spot=spot,
        strike=strike,
        expiry=expiry,
        rate=rate,
        div=div,
        sigma=0.0,
        n_steps=64,
    ).value
    assert value == pytest.approx(interior, abs=1e-12)
    assert value == pytest.approx(83.9505861332, abs=1e-9)
    assert value - endpoints == pytest.approx(2.7513055253, abs=1e-9)


def test_american_is_never_worth_less_than_intrinsic() -> None:
    for kind, spots in (("put", (60.0, 80.0, 100.0, 120.0)), ("call", (80.0, 100.0, 140.0))):
        for spot in spots:
            value = _american(
                kind,
                spot=spot,
                strike=100.0,
                expiry=1.0,
                rate=0.05,
                div=0.03,
                sigma=0.2,
                n_steps=200,
            ).value
            intrinsic = max(spot - 100.0, 0.0) if kind == "call" else max(100.0 - spot, 0.0)
            assert value >= intrinsic


# --------------------------------------------------------------------------
# Lattice Greeks
# --------------------------------------------------------------------------


def _greeks(kind: str, *, n_steps: int, american: bool, **spec):
    cls = AmericanOption if american else EuropeanOption
    return greeks(
        cls(kind=kind, strike=spec["strike"], expiry=spec["expiry"]),  # type: ignore[arg-type]
        BlackScholesModel(sigma=spec["sigma"]),
        _market(spec["spot"], spec["rate"], spec["div"]),
        method="tree",
        cfg=TreeConfig(n_steps=n_steps),
    )


_GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")


@pytest.mark.parametrize("n_steps", [2, 50, 500])
def test_american_call_greeks_equal_european_call_greeks_without_dividends(
    n_steps: int,
) -> None:
    """Evidence class: EXACT_IDENTITY.

    The value tree is bit-identical (see the price test above), the delta,
    gamma and theta estimators read the same nodes, and bumping `sigma` or `r`
    leaves `q = 0` alone -- so the bumped trees are bit-identical too and vega
    and rho follow. All five Greeks therefore agree exactly, not approximately.
    """
    spec = dict(spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.0, sigma=0.2)
    american = _greeks("call", n_steps=n_steps, american=True, **spec)
    european = _greeks("call", n_steps=n_steps, american=False, **spec)
    for name in _GREEK_NAMES:
        assert getattr(american, name) == getattr(european, name), name


def test_american_put_greeks_have_the_right_signs_and_beat_the_european_delta() -> None:
    """Evidence class: CLOSED_FORM for the signs, EXACT_IDENTITY for the
    ordering against the European put on the same lattice.

    There is no closed form for an American Greek, so the checks are the
    structural ones. A put's delta is negative and above -1, gamma positive,
    vega positive (more dispersion helps a convex payoff), theta negative
    (value decays), and rho negative (a higher rate makes the strike receipt
    worth less). The early-exercise right makes the American put *more*
    sensitive to spot than the European one -- its delta is more negative --
    because the value tracks intrinsic in the exercise region, where delta is
    exactly -1.
    """
    spec = dict(spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.0, sigma=0.2)
    american = _greeks("put", n_steps=2000, american=True, **spec)
    european = _greeks("put", n_steps=2000, american=False, **spec)

    assert -1.0 < american.delta < 0.0
    assert american.gamma > 0.0
    assert american.vega > 0.0
    assert american.theta < 0.0
    assert american.rho < 0.0
    assert american.delta < european.delta
    assert american.gamma > european.gamma
    # Rho: exercising early forgoes the discounting of the strike, so the
    # American put is less rate-sensitive in magnitude than the European one.
    assert european.rho < american.rho < 0.0


def test_american_lattice_delta_agrees_with_a_spot_bump_at_the_same_n() -> None:
    """Evidence class: INDEPENDENT_ENGINE (two routes through the same engine).

    The lattice delta is a slope across the two step-1 nodes -- a centred
    difference in spot evaluated at time `dt` rather than at `0`. Re-pricing at
    bumped spots is a centred difference at time `0`. They should agree to the
    `O(dt)` bias of reading the slope one step late plus the bump's own
    `O(h**2)` truncation, and for the call they do:

        call, n = 2000, q = 3%:  agreement 8.3e-06 at h = 0.01, 0.1 and 0.5

    The put does not behave that way, and the reason is worth recording.
    Bumping the spot rescales the *whole* lattice, so the exercise boundary
    moves relative to the node grid and the American value, as a function of
    `S0` at fixed `n`, is piecewise smooth with small slope jumps where a node
    crosses the boundary. A small bump then differences across that roughness
    instead of across the smooth part:

        put, n = 2000, q = 3%:   h = 0.01 -> 1.6e-03
                                 h = 0.1  -> 3.4e-04
                                 h = 0.5  -> 2.6e-05

    -- the opposite of the usual "smaller bump, smaller truncation" intuition.
    The bump below is therefore 0.5 (half a percent of spot), large enough to
    average over the lattice roughness and still small enough that the
    `O(h**2)` truncation of a central difference is negligible. Worst measured
    residual over `n` in {2000, 2001} and both kinds at that bump: 9.8e-05
    (put, `n = 2001`). Tolerance 3e-04, about a factor of three.
    """
    spec = dict(spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.03, sigma=0.2)
    bump = 0.5
    for n_steps in (2000, 2001):
        for kind in ("put", "call"):
            lattice_delta = _greeks(kind, n_steps=n_steps, american=True, **spec).delta
            up = _american(kind, n_steps=n_steps, **dict(spec, spot=spec["spot"] + bump)).value
            down = _american(kind, n_steps=n_steps, **dict(spec, spot=spec["spot"] - bump)).value
            bumped_delta = (up - down) / (2.0 * bump)
            assert lattice_delta == pytest.approx(bumped_delta, abs=3e-4), (kind, n_steps)


def test_american_greeks_validation_and_degenerate_cases() -> None:
    spec = dict(spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.0, sigma=0.2)

    result = _greeks("put", n_steps=64, american=True, **spec)
    assert result.meta is not None
    assert result.meta["exercise"] == "american"
    assert result.meta["bumps"] == {"sigma": 1e-2, "r": 1e-4}

    with pytest.raises(InvalidInputError, match="n_steps must be >= 2"):
        _greeks("put", n_steps=1, american=True, **spec)
    with pytest.raises(InvalidInputError, match="sigma must be > 0 for tree Greeks"):
        _greeks("put", n_steps=64, american=True, **dict(spec, sigma=0.0))

    flat = _greeks("put", n_steps=64, american=True, **dict(spec, expiry=0.0))
    assert tuple(getattr(flat, name) for name in _GREEK_NAMES) == (0.0,) * 5


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
    model, market = BlackScholesModel(sigma=0.2), _market(100.0, 0.05, 0.0)
    kwargs = {} if cfg is None else {"cfg": cfg}
    message = "Unsupported instrument/model/market combination"
    with pytest.raises(NotSupportedError, match=message):
        price(option, model, market, method=method, **kwargs)
    with pytest.raises(NotSupportedError, match=message):
        greeks(option, model, market, method=method, **kwargs)


def test_american_is_not_a_european_subclass() -> None:
    """Registry lookup walks the MRO, so inheriting would silently mis-price.

    If `AmericanOption` subclassed `EuropeanOption`, an American option handed
    to `method="analytic"` would resolve to the European closed form and return
    a number instead of refusing -- the worst possible failure mode, since the
    number looks perfectly reasonable.
    """
    assert not issubclass(AmericanOption, EuropeanOption)
    assert not issubclass(EuropeanOption, AmericanOption)
    assert AmericanOption(kind="put", strike=100.0, expiry=1.0) != EuropeanOption(
        kind="put", strike=100.0, expiry=1.0
    )


def test_american_option_validation() -> None:
    with pytest.raises(InvalidInputError, match="strike must be > 0"):
        AmericanOption(kind="put", strike=0.0, expiry=1.0)
    with pytest.raises(InvalidInputError, match="expiry must be >= 0"):
        AmericanOption(kind="put", strike=100.0, expiry=-1.0)
    with pytest.raises(InvalidInputError, match="kind must be"):
        AmericanOption(kind="straddle", strike=100.0, expiry=1.0)  # type: ignore[arg-type]


def test_american_config_and_no_arbitrage_validation() -> None:
    with pytest.raises(InvalidInputError, match="n_steps must be >= 1"):
        _american(
            "put", spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.0, sigma=0.2, n_steps=0
        )
    with pytest.raises(InvalidInputError, match="No-arbitrage"):
        _american(
            "put", spot=100.0, strike=100.0, expiry=1.0, rate=1.0, div=0.0, sigma=0.01, n_steps=1
        )


def test_american_put_is_deterministic_and_reports_its_lattice() -> None:
    kwargs = dict(spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.01, sigma=0.2, n_steps=40)
    a = _american("put", **kwargs)
    b = _american("put", **kwargs)
    assert a.value == b.value
    assert a.stderr is None

    meta = a.meta
    assert meta is not None
    assert meta["method"] == "tree"
    assert meta["scheme"] == "crr"
    assert meta["exercise"] == "american"
    assert meta["n_steps"] == 40
    assert meta["dt"] == pytest.approx(1.0 / 40)
    assert float(meta["u"]) * float(meta["d"]) == pytest.approx(1.0, abs=1e-15)
    assert meta["early_exercise_node_count"] > 0
    assert len(meta["exercise_boundary"]) == 41


def test_american_price_monotonicity_in_spot_and_strike() -> None:
    spec = dict(expiry=1.0, rate=0.05, div=0.01, sigma=0.2, n_steps=200)
    by_spot = [_american("put", spot=s, strike=100.0, **spec).value for s in (90.0, 100.0, 110.0)]
    assert by_spot[0] >= by_spot[1] >= by_spot[2]

    by_strike = [_american("put", spot=100.0, strike=k, **spec).value for k in (90.0, 100.0, 110.0)]
    assert by_strike[0] <= by_strike[1] <= by_strike[2]
