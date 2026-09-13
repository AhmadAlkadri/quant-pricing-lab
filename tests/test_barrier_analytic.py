"""The Reiner-Rubinstein closed forms, and the six things that pin them.

Every assertion names the `EvidenceClass` that justifies it.

1. **EXACT_IDENTITY** -- with a zero rebate, knock-in plus knock-out equals the
   vanilla, for all four in/out pairs and both kinds, to round-off. This holds
   by a pathwise partition of the sample space (a path either touches the
   barrier or it does not) and does not depend on the formulas being right,
   which is exactly what makes it worth asserting: a sign error in `C` or `D`
   breaks it, because the two assemblies use those blocks with opposite signs.
2. **EXACT_IDENTITY** -- with a rebate the same sum is *not* the vanilla, and
   the excess is exactly `E + F`, the value of the two rebate legs. Asserted
   with its sign, so the identity in (1) cannot be "restored" by a wrong rebate
   convention.
3. **CLOSED_FORM** -- the two barrier limits. `H -> 0` makes a down-and-out the
   vanilla; `H -> S` makes any knock-out worth exactly its rebate, at `H = S`
   itself and not only in the limit.
4. **CLOSED_FORM** -- the degenerate `T = 0` and `sigma = 0` limits, where the
   path is deterministic and the answer can be written down by hand.
5. **NEGATIVE_FINDING** -- a spot already at or beyond the barrier settles the
   contract, and the closed forms do **not** detect that on their own: they
   return a perfectly finite wrong number there, which is why the guard is a
   guard.
6. **CLOSED_FORM** / **NEGATIVE_FINDING** -- the finite-difference Greeks
   against the vanilla engine's analytic Greeks in the `H -> 0` limit, plus the
   two measured facts about where a barrier's gamma does and does not blow up.

Sources for the closed forms: Reiner, E. and Rubinstein, M. (1991), "Breaking
down the barriers", Risk 4(8), 28-35; Merton, R.C. (1973), Bell Journal of
Economics and Management Science 4(1), section 8, for the down-and-out call
alone; the joint law of the terminal value and the running extremum is derived
from the reflection principle in `qpl.engines.analytic.barrier` following
Shreve, *Stochastic Calculus for Finance II*, chapter 7. No expression, number
or table is reproduced from any of them.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from qpl.engines.analytic.barrier import (
    BGK_BETA,
    barrier_blocks,
    barrier_price,
    bgk_continuity_corrected_price,
    shifted_barrier,
)
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import (
    BARRIER_TYPES,
    BarrierOption,
    EuropeanOption,
    uniform_monitoring_times,
)
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel, bs_price
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass

# (id, spot, strike, expiry, rate, dividend, sigma). Five points: the Haug
# specification, two either side of the money, one with `q > r` and one long
# maturity. Barriers are placed at +-10% of the spot unless a test says
# otherwise, so every point has a live barrier on both sides.
_POINTS = (
    ("haug_6m", 100.0, 100.0, 0.50, 0.08, 0.04, 0.25),
    ("atm_1y", 100.0, 100.0, 1.00, 0.05, 0.00, 0.20),
    ("otm_9m_div", 100.0, 110.0, 0.75, 0.03, 0.01, 0.25),
    ("itm_1y_qgtr", 120.0, 90.0, 1.00, 0.03, 0.05, 0.35),
    ("otm_2y", 90.0, 100.0, 2.00, 0.02, 0.06, 0.40),
)
_ARGNAMES = ("spot", "strike", "expiry", "rate", "div", "sigma")
_ARGS = [row[1:] for row in _POINTS]
_IDS = [row[0] for row in _POINTS]

_PARITY_TOLERANCE = 1.0e-13
"""Round-off budget for the in-out identity.

Measured worst absolute residual over the five points, both kinds and both
barrier sides: **1.421e-14** (relative 1.810e-15). The tolerance keeps a factor
of 7; it is a round-off budget, not a model tolerance, and nothing about the
model can make it larger.
"""

_LIMIT_TOLERANCE = 1.0e-13
"""Same budget for the two barrier limits; worst measured 1.421e-14."""


def _barrier(spot: float, down: bool, offset: float = 0.10) -> float:
    return spot * (1.0 - offset) if down else spot * (1.0 + offset)


def _market(spot: float, rate: float, div: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(rate, allow_negative=True),
        dividend_curve=FlatDividendCurve(div, allow_negative=True),
    )


# --------------------------------------------------------------------------
# The instrument.
# --------------------------------------------------------------------------


def test_barrier_option_validates_its_fields() -> None:
    """Validation at construction, one message per rule."""
    ok = dict(kind="call", strike=100.0, expiry=1.0, barrier=95.0)
    BarrierOption(**ok)

    with pytest.raises(InvalidInputError, match="kind must be"):
        BarrierOption(**{**ok, "kind": "straddle"})
    with pytest.raises(InvalidInputError, match="strike must be > 0"):
        BarrierOption(**{**ok, "strike": 0.0})
    with pytest.raises(InvalidInputError, match="expiry must be >= 0"):
        BarrierOption(**{**ok, "expiry": -1.0})
    with pytest.raises(InvalidInputError, match="barrier must be finite and > 0"):
        BarrierOption(**{**ok, "barrier": 0.0})
    with pytest.raises(InvalidInputError, match="barrier must be finite and > 0"):
        BarrierOption(**{**ok, "barrier": math.inf})
    with pytest.raises(InvalidInputError, match="barrier_type must be one of"):
        BarrierOption(**{**ok, "barrier_type": "double-knock-out"})
    with pytest.raises(InvalidInputError, match="rebate must be finite and >= 0"):
        BarrierOption(**{**ok, "rebate": -1.0})
    with pytest.raises(InvalidInputError, match="monitoring must be 'continuous'"):
        BarrierOption(**{**ok, "monitoring": "daily"})
    with pytest.raises(InvalidInputError, match="monitoring must be strictly increasing"):
        BarrierOption(**{**ok, "monitoring": (0.5, 0.5, 1.0)})
    with pytest.raises(InvalidInputError, match="monitoring must all be <= expiry"):
        BarrierOption(**{**ok, "monitoring": (0.5, 1.5)})
    with pytest.raises(InvalidInputError, match="monitoring must all be > 0"):
        BarrierOption(**{**ok, "monitoring": (0.0, 1.0)})
    with pytest.raises(InvalidInputError, match="monitoring must contain at least one"):
        BarrierOption(**{**ok, "monitoring": ()})


def test_barrier_option_normalises_and_exposes_its_shape() -> None:
    """Lower-casing, the tuple conversion, and the four derived properties."""
    option = BarrierOption(
        kind="CALL",
        strike=100.0,
        expiry=1.0,
        barrier=95.0,
        barrier_type="Down-And-In",
        monitoring=[0.5, 1.0],
    )
    assert option.kind == "call"
    assert option.barrier_type == "down-and-in"
    assert option.monitoring == (0.5, 1.0)
    assert option.is_down and not option.is_knock_out
    assert not option.is_continuous
    assert option.monitoring_times == (0.5, 1.0)
    assert option.n_monitoring == 2
    assert hash(option)  # frozen and hashable, like every other instrument here

    continuous = BarrierOption(kind="put", strike=100.0, expiry=1.0, barrier=110.0,
                               barrier_type="up-and-out")
    assert continuous.is_continuous
    assert continuous.monitoring_times == ()
    assert continuous.n_monitoring == 0


@pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
def test_is_touched_uses_weak_inequalities(barrier_type: str) -> None:
    """Touching the level *is* reaching it, so the comparison is `<=` / `>=`."""
    option = BarrierOption(
        kind="call", strike=100.0, expiry=1.0, barrier=95.0, barrier_type=barrier_type
    )
    if option.is_down:
        assert option.is_touched(95.0) and option.is_touched(94.9)
        assert not option.is_touched(95.1)
    else:
        assert option.is_touched(95.0) and option.is_touched(95.1)
        assert not option.is_touched(94.9)


# --------------------------------------------------------------------------
# (1) and (2): in-out parity, with and without a rebate.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("down", [True, False])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_knock_in_plus_knock_out_is_the_vanilla(
    kind: str, down: bool, spot: float, strike: float, expiry: float,
    rate: float, div: float, sigma: float,
) -> None:
    """Evidence class: EXACT_IDENTITY.

    A path either touches the barrier before expiry or it does not, and the two
    contracts pay the vanilla on exactly those two complementary events. So
    their sum is the vanilla pathwise, hence in price, for every parameter set
    and with no approximation anywhere. Worst measured residual over the twenty
    cells: 1.421e-14.
    """
    assert EvidenceClass.EXACT_IDENTITY is EvidenceClass.EXACT_IDENTITY
    h = _barrier(spot, down)
    common = dict(S=spot, K=strike, T=expiry, r=rate, sigma=sigma, q=div, H=h,
                  rebate=0.0, kind=kind)
    knock_in = barrier_price(
        barrier_type="down-and-in" if down else "up-and-in", **common
    )
    knock_out = barrier_price(
        barrier_type="down-and-out" if down else "up-and-out", **common
    )
    vanilla = bs_price(S=spot, K=strike, T=expiry, r=rate, sigma=sigma, q=div, kind=kind)

    assert knock_in + knock_out == pytest.approx(vanilla, abs=_PARITY_TOLERANCE)
    # Non-negativity up to round-off: the closed form assembles a worthless leg
    # from cancelling building blocks, and Linux libm lands at -3.6e-15 where
    # macOS lands at +0.0 (CI failure on the Slice 12 push).
    assert knock_in >= -1e-12 and knock_out >= -1e-12
    assert vanilla > 1e-6


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("down", [True, False])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_a_rebate_breaks_parity_by_exactly_the_two_rebate_legs(
    kind: str, down: bool, spot: float, strike: float, expiry: float,
    rate: float, div: float, sigma: float,
) -> None:
    """Evidence class: EXACT_IDENTITY, on the *broken* identity.

    With a rebate the knock-in pays it at expiry when the barrier was never
    touched (`E`) and the knock-out pays it at the touch time when it was
    (`F`), on complementary events. So the sum overshoots the vanilla by
    `E + F` -- a strictly positive amount, not a small error. Pinning the
    excess with its value rather than only its sign is what stops a wrong
    rebate convention from looking like a rounding difference.
    """
    rebate = 3.0
    h = _barrier(spot, down)
    common = dict(S=spot, K=strike, T=expiry, r=rate, sigma=sigma, q=div, H=h,
                  rebate=rebate, kind=kind)
    knock_in = barrier_price(
        barrier_type="down-and-in" if down else "up-and-in", **common
    )
    knock_out = barrier_price(
        barrier_type="down-and-out" if down else "up-and-out", **common
    )
    vanilla = bs_price(S=spot, K=strike, T=expiry, r=rate, sigma=sigma, q=div, kind=kind)

    blocks = barrier_blocks(
        S=spot, K=strike, T=expiry, r=rate, sigma=sigma, q=div, H=h, rebate=rebate,
        phi=1.0 if kind == "call" else -1.0,
        eta=1.0 if down else -1.0,
    )
    excess = knock_in + knock_out - vanilla
    assert excess == pytest.approx(blocks.E + blocks.F, abs=_PARITY_TOLERANCE)
    assert excess > 0.0
    # Both legs are worth something: a rebate paid on either event is worth
    # strictly less than the rebate itself (it is discounted) and strictly more
    # than nothing.
    assert 0.0 < excess < rebate


# --------------------------------------------------------------------------
# (3): the two barrier limits.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_a_vanishing_barrier_recovers_the_vanilla(
    kind: str, spot: float, strike: float, expiry: float, rate: float,
    div: float, sigma: float,
) -> None:
    """Evidence class: CLOSED_FORM.

    `H -> 0` makes a down-and-out call impossible to knock out, and `H -> inf`
    does the same for an up-and-out. Both must return the Black-Scholes value.
    The reflected blocks do not merely become small: `(H/S)^{2mu}` can grow
    without bound when `mu < 0`, and it is the normal tail multiplying it that
    decays fast enough to win. Worst measured residual at `H = 1e-6 S` and
    `H = 1e+6 S`: 1.421e-14 for both.
    """
    vanilla = bs_price(S=spot, K=strike, T=expiry, r=rate, sigma=sigma, q=div, kind=kind)
    common = dict(S=spot, K=strike, T=expiry, r=rate, sigma=sigma, q=div,
                  rebate=0.0, kind=kind)
    down_out = barrier_price(H=spot * 1e-6, barrier_type="down-and-out", **common)
    up_out = barrier_price(H=spot * 1e6, barrier_type="up-and-out", **common)
    assert down_out == pytest.approx(vanilla, abs=_LIMIT_TOLERANCE)
    assert up_out == pytest.approx(vanilla, abs=_LIMIT_TOLERANCE)

    # ... and the matching knock-ins are worth nothing.
    assert barrier_price(H=spot * 1e-6, barrier_type="down-and-in", **common) == pytest.approx(
        0.0, abs=_LIMIT_TOLERANCE
    )
    assert barrier_price(H=spot * 1e6, barrier_type="up-and-in", **common) == pytest.approx(
        0.0, abs=_LIMIT_TOLERANCE
    )


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("barrier_type", ["down-and-out", "up-and-out"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_a_barrier_at_the_spot_leaves_exactly_the_rebate(
    kind: str, barrier_type: str, spot: float, strike: float, expiry: float,
    rate: float, div: float, sigma: float,
) -> None:
    """Evidence class: CLOSED_FORM, at the value and not in a limit.

    Two different mechanisms produce the same number and both are checked. The
    *assembled formula* collapses to the block `F` at `H = S` -- for a call
    block by block (`A = C`, `B = D`), for a put as a difference of sums (see
    `qpl.engines.analytic.barrier`) -- and `F -> K` there, so the formula
    itself says "rebate". The *guard* reaches the same answer by observing that
    the spot has already touched the barrier. Worst measured residual of the
    formula route: 1.421e-14.
    """
    rebate = 3.0
    phi = 1.0 if kind == "call" else -1.0
    eta = 1.0 if barrier_type.startswith("down") else -1.0
    blocks = barrier_blocks(
        S=spot, K=strike, T=expiry, r=rate, sigma=sigma, q=div, H=spot,
        rebate=rebate, phi=phi, eta=eta,
    )
    assert blocks.F == pytest.approx(rebate, abs=_LIMIT_TOLERANCE)

    # The guard route, through the public entry point.
    assert barrier_price(
        S=spot, K=strike, T=expiry, r=rate, sigma=sigma, q=div, H=spot,
        rebate=rebate, barrier_type=barrier_type, kind=kind,
    ) == rebate

    # Approaching the barrier from the live side, the value converges to the
    # rebate rather than jumping to it: continuity, which is what makes the
    # guard consistent with the formula instead of merely agreeing at a point.
    live_side = spot * (1.0 + 1e-8) if eta > 0 else spot * (1.0 - 1e-8)
    near = barrier_price(
        S=live_side, K=strike, T=expiry, r=rate, sigma=sigma, q=div, H=spot,
        rebate=rebate, barrier_type=barrier_type, kind=kind,
    )
    assert near == pytest.approx(rebate, abs=1e-5)


# --------------------------------------------------------------------------
# (4): the deterministic limits.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
def test_zero_expiry_is_the_payoff(barrier_type: str) -> None:
    """Evidence class: CLOSED_FORM. At `T = 0` there is no path to monitor."""
    common = dict(S=100.0, K=90.0, T=0.0, r=0.05, sigma=0.2, q=0.0, rebate=3.0,
                  barrier_type=barrier_type, kind="call")
    # `H = S` is touched for both sides (the comparison is weak); an untouched
    # barrier has to sit strictly on the far side of the spot.
    touched = barrier_price(H=100.0, **common)
    untouched = barrier_price(H=95.0 if "down" in barrier_type else 110.0, **common)
    if barrier_type.endswith("out"):
        assert touched == 3.0
        assert untouched == 10.0
    else:
        assert touched == 10.0
        assert untouched == 3.0


def test_zero_volatility_follows_the_deterministic_forward() -> None:
    """Evidence class: CLOSED_FORM, re-derived here rather than compared.

    At `sigma = 0` the spot is `S e^{(r-q)t}`, monotone, so the barrier is
    touched at most once and the touch time solves `S e^{(r-q)t} = H`. Each
    value below is written out independently from that observation.
    """
    s, k, t, r, q = 100.0, 100.0, 1.0, 0.06, 0.01
    b = r - q
    rebate = 3.0
    common = dict(S=s, K=k, T=t, r=r, sigma=0.0, q=q, rebate=rebate, kind="call")

    # The forward rises to 100 e^{0.05} = 105.127, so an up barrier at 103 is
    # touched at t* = log(1.03)/0.05, and one at 110 is never touched.
    t_star = math.log(103.0 / s) / b
    assert barrier_price(H=103.0, barrier_type="up-and-out", **common) == pytest.approx(
        rebate * math.exp(-r * t_star), abs=1e-14
    )
    assert barrier_price(H=110.0, barrier_type="up-and-out", **common) == pytest.approx(
        math.exp(-r * t) * max(s * math.exp(b * t) - k, 0.0), abs=1e-14
    )
    assert barrier_price(H=103.0, barrier_type="up-and-in", **common) == pytest.approx(
        math.exp(-r * t) * max(s * math.exp(b * t) - k, 0.0), abs=1e-14
    )
    assert barrier_price(H=110.0, barrier_type="up-and-in", **common) == pytest.approx(
        rebate * math.exp(-r * t), abs=1e-14
    )
    # A down barrier below the starting spot is never reached, because the
    # forward only rises.
    assert barrier_price(H=95.0, barrier_type="down-and-out", **common) == pytest.approx(
        math.exp(-r * t) * max(s * math.exp(b * t) - k, 0.0), abs=1e-14
    )


# --------------------------------------------------------------------------
# (5): the inception guard.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
def test_a_spot_beyond_the_barrier_settles_the_contract(kind: str) -> None:
    """Evidence class: CLOSED_FORM on the answer, NEGATIVE_FINDING on the need.

    A knock-out whose barrier has already been touched is dead and worth its
    rebate **now**, undiscounted; a knock-in in the same position is alive and
    worth the vanilla. Both are facts about the contract, so every engine in
    this package must report them, and this test is the reference the tree and
    Monte Carlo tests compare against.

    The negative half: the closed forms do not notice. Evaluated on the wrong
    side of the barrier the blocks are finite and the assembled number is
    plausible-*looking* -- and, at this point, **negative**: -1.0394 for a
    down-and-out call at `S = 90, H = 95` against a true value of 3.00. A
    negative price is not a rounding difference, and it is what a formula
    derived for `S > H` produces when handed `S < H`, so the guard cannot be
    replaced by a limit or by an assertion that the formula "handles it".
    """
    s, k, t, r, q, sigma, rebate = 90.0, 100.0, 0.5, 0.08, 0.04, 0.25, 3.0
    common = dict(S=s, K=k, T=t, r=r, sigma=sigma, q=q, rebate=rebate, kind=kind)
    vanilla = bs_price(S=s, K=k, T=t, r=r, sigma=sigma, q=q, kind=kind)

    assert barrier_price(H=95.0, barrier_type="down-and-out", **common) == rebate
    assert barrier_price(H=95.0, barrier_type="down-and-in", **common) == vanilla
    assert barrier_price(H=85.0, barrier_type="up-and-out", **common) == rebate
    assert barrier_price(H=85.0, barrier_type="up-and-in", **common) == vanilla
    # Exactly on the barrier counts as touched.
    assert barrier_price(H=s, barrier_type="down-and-out", **common) == rebate

    blocks = barrier_blocks(
        S=s, K=k, T=t, r=r, sigma=sigma, q=q, H=95.0, rebate=rebate,
        phi=1.0 if kind == "call" else -1.0, eta=1.0,
    )
    unguarded = blocks.A - blocks.C + blocks.F
    assert math.isfinite(unguarded)
    if kind == "call":
        assert unguarded == pytest.approx(-1.0394, abs=1e-3)
        assert unguarded < 0.0  # not a price at all


def test_the_dispatcher_reports_the_inception_state() -> None:
    """`meta["touched_at_inception"]` says which branch produced the number."""
    market = _market(90.0, 0.08, 0.04)
    model = BlackScholesModel(sigma=0.25)
    dead = BarrierOption("call", 100.0, 0.5, 95.0, "down-and-out", 3.0)
    alive = BarrierOption("call", 100.0, 0.5, 85.0, "down-and-out", 3.0)
    assert price(dead, model, market).meta["touched_at_inception"] is True
    assert price(alive, model, market).meta["touched_at_inception"] is False


# --------------------------------------------------------------------------
# Discrete monitoring is refused, by name.
# --------------------------------------------------------------------------


def test_analytic_refuses_a_discretely_monitored_barrier() -> None:
    """Evidence class: none -- a contract check, not a numerical claim.

    The Reiner-Rubinstein forms price the continuous contract. Answering a
    discrete one with them would be a silently different question, so the
    engine refuses and the message names both alternatives.
    """
    option = BarrierOption(
        "call", 100.0, 0.5, 95.0, "down-and-out", 3.0, uniform_monitoring_times(0.5, 50)
    )
    market = _market(100.0, 0.08, 0.04)
    model = BlackScholesModel(sigma=0.25)
    with pytest.raises(NotSupportedError) as excinfo:
        price(option, model, market)
    message = str(excinfo.value)
    assert "method='mc'" in message
    assert "bgk_continuity_corrected_price" in message
    assert "1/sqrt(m)" in message
    with pytest.raises(NotSupportedError):
        greeks(option, model, market)


def test_the_bgk_shift_moves_the_barrier_the_right_way() -> None:
    """Evidence class: CLOSED_FORM on the sign, which is the whole content.

    A discretely monitored knock-out is worth **more** than the continuous one
    (the path can dip past the barrier and return unobserved), so approximating
    the discrete price from the continuous formula moves the barrier *away*
    from the spot, and simulating the continuous price on a discrete grid moves
    it *toward* the spot. Getting these backwards doubles the bias instead of
    removing it, which is why the direction is a named argument rather than a
    sign.
    """
    h, sigma, dt = 95.0, 0.25, 0.5 / 50
    assert shifted_barrier(H=h, sigma=sigma, dt=dt, down=True, toward_spot=False) < h
    assert shifted_barrier(H=h, sigma=sigma, dt=dt, down=True, toward_spot=True) > h
    assert shifted_barrier(H=105.0, sigma=sigma, dt=dt, down=False, toward_spot=False) > 105.0
    assert shifted_barrier(H=105.0, sigma=sigma, dt=dt, down=False, toward_spot=True) < 105.0
    assert shifted_barrier(H=h, sigma=sigma, dt=0.0, down=True, toward_spot=True) == h

    # `beta` is the zeta value, not a fitted constant.
    assert BGK_BETA == pytest.approx(1.4603545088095868 / math.sqrt(2.0 * math.pi), rel=1e-15)

    continuous = barrier_price(
        S=100.0, K=100.0, T=0.5, r=0.08, sigma=sigma, q=0.04, H=h, rebate=0.0,
        barrier_type="down-and-out", kind="call",
    )
    corrected = bgk_continuity_corrected_price(
        S=100.0, K=100.0, T=0.5, r=0.08, sigma=sigma, q=0.04, H=h, rebate=0.0,
        barrier_type="down-and-out", kind="call", n_monitoring=50,
    )
    assert corrected > continuous
    with pytest.raises(InvalidInputError, match="n_monitoring must be >= 1"):
        bgk_continuity_corrected_price(
            S=100.0, K=100.0, T=0.5, r=0.08, sigma=sigma, q=0.04, H=h,
            barrier_type="down-and-out", kind="call", n_monitoring=0,
        )


# --------------------------------------------------------------------------
# (6): Greeks.
# --------------------------------------------------------------------------

_GREEK_RELATIVE_TOLERANCE = {
    "delta": 1e-8,
    "gamma": 1e-5,
    "vega": 1e-6,
    "theta": 1e-7,
    "rho": 1e-6,
}
"""Budgets for the `H -> 0` Greek comparison, derived from the measurement.

Worst measured relative residual against the vanilla engine's closed-form
Greeks over four points: 1.948e-10 (delta), 1.440e-06 (gamma), 2.011e-08
(vega), 4.291e-09 (theta), 2.286e-08 (rho). Each budget keeps between 7x and
50x. Gamma is the loose one because a second central difference divides by
`h**2 = 1e-06`.
"""


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_greeks_reduce_to_the_vanillas_when_the_barrier_vanishes(
    kind: str, spot: float, strike: float, expiry: float, rate: float,
    div: float, sigma: float,
) -> None:
    """Evidence class: CLOSED_FORM.

    With `H = 1e-6 S` a down-and-out option *is* the vanilla, so its five
    finite-difference Greeks must reproduce the vanilla engine's five
    closed-form ones. This checks the difference quotients, the bump sizes and
    the theta sign convention in one shot, against a formula that shares no
    code with the barrier blocks.
    """
    market = _market(spot, rate, div)
    model = BlackScholesModel(sigma=sigma)
    ours = greeks(
        BarrierOption(kind, strike, expiry, spot * 1e-6, "down-and-out", 0.0),
        model, market,
    )
    reference = greeks(EuropeanOption(kind, strike, expiry), model, market)

    for name, tol in _GREEK_RELATIVE_TOLERANCE.items():
        got, want = getattr(ours, name), getattr(reference, name)
        assert got == pytest.approx(want, rel=tol), name
    assert ours.meta["fd"] == "central"
    assert ours.meta["distance_to_barrier_in_bumps"] > 1.0


def test_a_knock_out_call_is_concave_near_its_barrier() -> None:
    """Evidence class: NEGATIVE_FINDING -- gamma does *not* blow up as S -> H.

    The slice this engine was written for expected a gamma singularity as the
    spot approaches the barrier. There is none. The value vanishes *linearly*
    in `S - H` and both derivatives converge: on the reference down-and-out
    call (`K = 100`, `H = 95`, `T = 0.5`, `r = 8%`, `q = 4%`, `sigma = 25%`,
    zero rebate) delta tends to 0.9265 and gamma to -0.0125 as `S -> H+`. What
    the barrier changes is gamma's **sign**: the same vanilla at `S = 100` has
    gamma +0.02168, so the knock-out is concave where the vanilla is convex.

    Asserted as a ratio rather than as a level, so the finding survives any
    later change of bump size: gamma must be *bounded* along the approach and
    must keep one sign.
    """
    model = BlackScholesModel(sigma=0.25)
    spots = (110.0, 105.0, 100.0, 96.0, 95.5, 95.1, 95.001)
    gammas = []
    values = []
    for s in spots:
        result = greeks(
            BarrierOption("call", 100.0, 0.5, 95.0, "down-and-out", 0.0),
            model, _market(s, 0.08, 0.04),
        )
        gammas.append(result.gamma)
        values.append(
            barrier_price(S=s, K=100.0, T=0.5, r=0.08, sigma=0.25, q=0.04, H=95.0,
                          rebate=0.0, barrier_type="down-and-out", kind="call")
        )

    assert max(abs(g) for g in gammas) < 0.05  # bounded, not blowing up
    assert gammas[-1] == pytest.approx(-0.01248, abs=5e-4)
    assert all(g < 0.0 for g in gammas[2:])  # concave from the money inward
    assert gammas[0] > 0.0  # and convex far from the barrier: the sign flips

    # The value vanishes linearly: halving `S - H` halves the price.
    assert values[-1] / values[-2] == pytest.approx(0.01, rel=0.05)

    vanilla_gamma = greeks(
        EuropeanOption("call", 100.0, 0.5), model, _market(100.0, 0.08, 0.04)
    ).gamma
    assert vanilla_gamma > 0.0
    assert gammas[2] < 0.0 < vanilla_gamma


def test_gamma_blows_up_at_the_corner_and_only_there() -> None:
    """Evidence class: NEGATIVE_FINDING -- the singularity is in `T`, not in `S`.

    The barrier problem's singularity sits at the corner `(S = H, t = T)`, and
    it exists only when the terminal payoff is discontinuous across the
    barrier, i.e. when `vanilla(H) != rebate`. Measured on a down-and-out call
    with `K = 90 < H = 95` (payoff jumps by 5 across the barrier) at `S = 96`:
    gamma is -0.018, -0.029, -0.123, -0.721, -3.842 at
    `T = 0.5, 0.1, 0.02, 0.005, 0.001`.

    With `K = 100 > H` the payoff is continuous across the barrier (zero on
    both sides) and there is no growth at all on the same ladder -- which is
    the half of the finding that says *why*.
    """
    model = BlackScholesModel(sigma=0.25)
    market = _market(96.0, 0.08, 0.04)
    maturities = (0.5, 0.1, 0.02, 0.005, 0.001)

    jump = [
        abs(greeks(BarrierOption("call", 90.0, t, 95.0, "down-and-out", 0.0),
                   model, market).gamma)
        for t in maturities
    ]
    smooth = [
        abs(greeks(BarrierOption("call", 100.0, t, 95.0, "down-and-out", 0.0),
                   model, market).gamma)
        for t in maturities
    ]

    assert all(b > a for a, b in pairwise(jump))  # monotone growth
    assert jump[-1] > 100.0 * jump[0]
    assert jump[-1] == pytest.approx(3.84, abs=0.2)
    # The smooth case does not grow: it decays to zero instead.
    assert smooth[-1] < smooth[0]
    assert smooth[-1] < 1e-3


def test_greeks_refuse_the_two_degenerate_limits() -> None:
    """`T = 0` and `sigma = 0` make the value a step function of the spot."""
    model = BlackScholesModel(sigma=0.25)
    market = _market(100.0, 0.08, 0.04)
    with pytest.raises(InvalidInputError, match="T must be > 0"):
        greeks(BarrierOption("call", 100.0, 0.0, 95.0, "down-and-out"), model, market)
    with pytest.raises(InvalidInputError, match="sigma must be > 0"):
        greeks(
            BarrierOption("call", 100.0, 0.5, 95.0, "down-and-out"),
            BlackScholesModel(sigma=0.0),
            market,
        )
