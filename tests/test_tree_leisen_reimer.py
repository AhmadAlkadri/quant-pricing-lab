"""Leisen-Reimer lattice: construction, symmetry, validation, limits, Greeks.

The measured price convergence -- the point of the scheme -- is in
`tests/test_tree_lr_convergence.py`, and the QuantLib comparison in
`tests/oracle/test_lr_vs_quantlib.py`. This file covers the parts that are
exact or structural rather than asymptotic, plus the one asymptotic claim that
belongs with the Greeks rather than with the price.

Five kinds of claim, kept apart because they are justified differently:

(a) EXACT_IDENTITY -- the Peizer-Pratt inversion is antisymmetric about 1/2,
    and that antisymmetry makes the lattice symmetric about the forward when
    the option is at the money forward. Recomputed here from `d1` and `d2`
    written out in the test, not read back from the engine.
(b) EXACT_IDENTITY -- the lattice reprices the one-step forward, and therefore
    put-call parity holds on it to round-off at every odd `n`.
(c) NEGATIVE_FINDING (of a sort: a refusal, pinned) -- even `n` has no
    Leisen-Reimer construction and the engine says so rather than rounding up.
(d) CLOSED_FORM -- the `T = 0` and `sigma = 0` limits are the same closed
    forms the CRR scheme returns, because neither goes near a lattice.
(e) CONVERGENCE_ORDER -- delta, gamma and theta off the Leisen-Reimer lattice
    converge at a measured order of **1**, not 2. That contradicts the
    expectation this slice was written with and is recorded as measured; see
    `test_lattice_greeks_are_first_order_not_second`.

Scheme: Leisen and Reimer (1996), "Binomial models for option valuation --
examining and improving convergence", Applied Mathematical Finance 3(4),
319-346. Every number below was measured in this repository; none is quoted
from that paper. Derivation and tables: `docs/notes/leisen_reimer.md`.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from qpl.engines.tree import (
    TreeConfig,
    crr_parameters,
    lattice_parameters,
    leisen_reimer_parameters,
    peizer_pratt_inversion,
)
from qpl.exceptions import InvalidInputError
from qpl.instruments.options import AmericanOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price
from qpl.validation import fit_convergence_order

LR = "leisen-reimer"

# (id, spot, strike, expiry, rate, dividend, sigma)
_POINTS = (
    ("atm_1y", 100.0, 100.0, 1.0, 0.05, 0.00, 0.20),
    ("otm_9m_div", 100.0, 110.0, 0.75, 0.03, 0.01, 0.25),
    ("deep_otm_2y", 80.0, 120.0, 2.0, 0.02, 0.03, 0.40),
    ("itm_1y_div", 120.0, 90.0, 1.0, 0.03, 0.05, 0.35),
)
_IDS = [row[0] for row in _POINTS]
_ARGS = [row[1:] for row in _POINTS]
_ARGNAMES = ("spot", "strike", "expiry", "rate", "div", "sigma")

ODD_LEVELS = (25, 51, 101, 201, 401)


def _market(spot: float, r: float, q: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(r, allow_negative=True),
        dividend_curve=FlatDividendCurve(q, allow_negative=True),
    )


def _lr(
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
        cfg=TreeConfig(n_steps=n_steps, scheme=LR),
    ).value


def _d1_d2(
    *, spot: float, strike: float, expiry: float, rate: float, div: float, sigma: float
) -> tuple[float, float]:
    """Black-Scholes `d1` and `d2`, written out here rather than imported.

    The lattice is built from these two numbers, so a test that reads them
    back from the engine would be checking the engine against itself.
    """
    vol = sigma * math.sqrt(expiry)
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * sigma * sigma) * expiry) / vol
    return d1, d1 - vol


# --------------------------------------------------------------------------
# (a) The Peizer-Pratt inversion, and the symmetry it buys
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_steps", [1, 3, 25, 101, 801, 8001])
@pytest.mark.parametrize("z", [0.001, 0.05, 0.3, 1.0, 2.5, 5.0])
def test_peizer_pratt_is_antisymmetric_about_one_half(z: float, n_steps: int) -> None:
    """Evidence class: EXACT_IDENTITY.

    `h(-z) = 1 - h(z)`, because `z` enters the method-2 formula only through
    `z**2` and an explicit sign. Measured worst residual over this grid:
    5.6e-17, i.e. one ULP of 1/2. The tolerance is a round-off budget.

    This is not decoration. It is the reason the lattice is symmetric at the
    money forward (next test) and, through that, the reason the scheme has no
    odd/even oscillation to remove there.
    """
    assert peizer_pratt_inversion(-z, n_steps) == pytest.approx(
        1.0 - peizer_pratt_inversion(z, n_steps), abs=1e-16
    )


@pytest.mark.parametrize("n_steps", [1, 3, 101, 801])
def test_peizer_pratt_is_one_half_at_zero_and_increasing(n_steps: int) -> None:
    """Evidence class: EXACT_IDENTITY.

    `h(0) = 1/2` exactly (special-cased, since the general expression would
    return `0.5 + 0.0` anyway but with a sign that depends on how `z == 0` is
    signed), and `h` is strictly increasing. The monotonicity is what makes
    `p' > p` -- since `d1 > d2` -- and therefore what makes the lattice
    arbitrage-free with no separate check.
    """
    assert peizer_pratt_inversion(0.0, n_steps) == 0.5

    grid = [-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0]
    values = [peizer_pratt_inversion(z, n_steps) for z in grid]
    assert all(0.0 < v < 1.0 for v in values), values
    assert all(a < b for a, b in pairwise(values)), values


@pytest.mark.parametrize("n_steps", [3, 25, 101, 801])
@pytest.mark.parametrize(
    ("spot", "expiry", "rate", "div", "sigma"),
    [(100.0, 1.0, 0.05, 0.00, 0.20), (80.0, 2.0, 0.03, 0.01, 0.35)],
    ids=["no_yield", "with_yield"],
)
def test_at_the_money_forward_the_lattice_is_symmetric(
    n_steps: int, spot: float, expiry: float, rate: float, div: float, sigma: float
) -> None:
    """Evidence class: EXACT_IDENTITY.

    "At the money" for this scheme means at the money *forward*,
    `K = S e^{(r-q)T}`, which is where `d2 = -d1`. Antisymmetry then gives
    `p' = 1 - p` exactly, and the two multipliers become mirror images about
    the forward:

        u d = growth**2 p'(1 - p') / (p (1 - p)) = growth**2,
        u / growth = growth / d.

    Measured: `p + p' - 1` is **exactly** 0.0 at every `n` tested, and
    `u d - growth**2` is at most 2.2e-16. With `r = q` the second identity
    collapses to `u d = 1`, i.e. the Leisen-Reimer tree is spot-centred
    exactly where CRR always is -- which is the sense in which it "reduces to
    a symmetric tree at the money".
    """
    strike = spot * math.exp((rate - div) * expiry)
    lattice = leisen_reimer_parameters(
        spot=spot,
        strike=strike,
        sigma=sigma,
        expiry=expiry,
        rate=rate,
        dividend_yield=div,
        n_steps=n_steps,
    )
    d1, d2 = _d1_d2(
        spot=spot, strike=strike, expiry=expiry, rate=rate, div=div, sigma=sigma
    )
    assert d2 == pytest.approx(-d1, abs=1e-14)

    p_share = peizer_pratt_inversion(d1, n_steps)
    assert lattice.p + p_share == 1.0
    assert lattice.up * lattice.down == pytest.approx(lattice.growth**2, abs=1e-15)
    assert lattice.up / lattice.growth == pytest.approx(
        lattice.growth / lattice.down, abs=1e-14
    )


@pytest.mark.parametrize("n_steps", [3, 101, 801])
def test_zero_carry_at_the_money_gives_a_spot_centred_lattice(n_steps: int) -> None:
    """Evidence class: EXACT_IDENTITY.

    The special case of the previous test that is easiest to check by eye:
    with `r = q = 0` and `K = S`, `u d = 1` to round-off, so an even time
    level has the initial spot exactly at its centre, exactly as CRR always
    does. `spot_centred` stays `False` all the same -- the flag records
    whether `u d = 1` *by construction*, which is what the Greek estimators
    need to know, not whether it happens to hold at one specification.
    """
    lattice = leisen_reimer_parameters(
        spot=100.0,
        strike=100.0,
        sigma=0.2,
        expiry=1.0,
        rate=0.0,
        dividend_yield=0.0,
        n_steps=n_steps,
    )
    assert lattice.up * lattice.down == pytest.approx(1.0, abs=1e-15)
    assert lattice.growth == 1.0
    assert not lattice.spot_centred


@pytest.mark.parametrize("n_steps", [1, 3, 51, 801])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_lattice_reprices_the_forward_and_is_arbitrage_free(
    n_steps: int,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: EXACT_IDENTITY.

    Two structural facts, both of which the construction is arranged to give
    rather than to check afterwards:

    - `p u + (1 - p) d = growth`, because `d` is solved from that equation.
      Measured worst residual 4.4e-16 relative.
    - `d < growth < u` and `0 < p < 1`. This follows from `p' > p`, which
      follows from `d1 > d2` and the inversion being increasing -- so unlike
      CRR, where a coarse tree with a large drift can produce a `p` outside
      `[0, 1]`, Leisen-Reimer has no no-arbitrage condition to violate.
    """
    lattice = leisen_reimer_parameters(
        spot=spot,
        strike=strike,
        sigma=sigma,
        expiry=expiry,
        rate=rate,
        dividend_yield=div,
        n_steps=n_steps,
    )
    reproduced = lattice.p * lattice.up + (1.0 - lattice.p) * lattice.down
    assert reproduced == pytest.approx(lattice.growth, rel=1e-14)

    assert 0.0 < lattice.p < 1.0
    assert lattice.down < lattice.growth < lattice.up
    assert lattice.n_steps == n_steps
    assert lattice.dt == pytest.approx(expiry / n_steps, rel=0, abs=0)
    assert not lattice.degenerate
    assert not lattice.spot_centred


def test_off_the_money_forward_the_lattice_is_not_symmetric() -> None:
    """Evidence class: EXACT_IDENTITY (a non-vacuity check).

    The symmetry tests above would pass on a lattice that was symmetric
    everywhere, which would mean the strike was not entering the geometry at
    all. It does: at `K = 120` against a forward of 105.1 the departures are
    `p + p' - 1 = 1.4e-01` and `u d / growth**2 - 1 = -7.6e-03` at `n = 101`,
    both enormous compared with the 2e-16 measured at the money forward.
    """
    common = {
        "spot": 100.0,
        "sigma": 0.20,
        "expiry": 1.0,
        "rate": 0.05,
        "dividend_yield": 0.0,
        "n_steps": 101,
    }
    lattice = leisen_reimer_parameters(strike=120.0, **common)  # type: ignore[arg-type]
    d1, _ = _d1_d2(
        spot=100.0, strike=120.0, expiry=1.0, rate=0.05, div=0.0, sigma=0.20
    )
    p_share = peizer_pratt_inversion(d1, 101)

    assert abs(lattice.p + p_share - 1.0) > 1e-2
    assert abs(lattice.up * lattice.down / lattice.growth**2 - 1.0) > 1e-3


def test_the_lattice_depends_on_the_strike() -> None:
    """Evidence class: EXACT_IDENTITY.

    Unlike CRR, a Leisen-Reimer call and put on *different* strikes do not
    share a lattice. Worth pinning, because it is the property that makes the
    scheme second order and also the property that makes it impossible to
    reuse one tree across a strike ladder.
    """
    common = {
        "spot": 100.0,
        "sigma": 0.20,
        "expiry": 1.0,
        "rate": 0.05,
        "dividend_yield": 0.0,
        "n_steps": 51,
    }
    a = leisen_reimer_parameters(strike=100.0, **common)  # type: ignore[arg-type]
    b = leisen_reimer_parameters(strike=110.0, **common)  # type: ignore[arg-type]
    assert a.up != b.up
    assert a.down != b.down
    assert a.p != b.p
    # ... but a call and a put on the SAME strike do, which is what makes
    # put-call parity below an identity rather than a coincidence.
    assert leisen_reimer_parameters(strike=100.0, **common) == a  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# (b) Put-call parity on the Leisen-Reimer lattice
# --------------------------------------------------------------------------

# Measured parity residuals on the Leisen-Reimer lattice grow like
# n * eps * price, exactly as on the CRR one and for the same reason -- the
# discount factor is applied n times as e^{-r dt} rather than once as e^{-rT}.
# Worst over the points and step counts below: 9.7e-12 at n = 2001. 1e-10 is a
# round-off budget with roughly a factor of ten of headroom, and is the same
# budget `tests/test_tree_convergence.py` uses for CRR.
_PARITY_TOLERANCE = 1e-10


@pytest.mark.parametrize("n_steps", [1, 3, 25, 101, 801, 2001])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_put_call_parity_holds_on_the_lr_tree(
    n_steps: int,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: EXACT_IDENTITY.

    `C - P = S e^{-qT} - K e^{-rT}` is a statement about the forward, and this
    lattice reprices the forward exactly: `p u + (1 - p) d = growth` by
    construction (`d` is solved from it), so the `n`-step expected terminal
    spot is `S e^{(r-q)T}` with no discretisation error.

    The call and the put share a lattice here only because they share a
    strike: `d1` and `d2` do not know the option's kind. That is worth saying
    out loud, since parity would be meaningless if the two legs were priced on
    two different trees.
    """
    common = {
        "spot": spot,
        "strike": strike,
        "expiry": expiry,
        "rate": rate,
        "div": div,
        "sigma": sigma,
        "n_steps": n_steps,
    }
    call = _lr("call", **common)  # type: ignore[arg-type]
    put = _lr("put", **common)  # type: ignore[arg-type]

    forward_leg = spot * math.exp(-div * expiry)
    strike_leg = strike * math.exp(-rate * expiry)
    residual = (call - put) - (forward_leg - strike_leg)
    assert abs(residual) <= _PARITY_TOLERANCE


# --------------------------------------------------------------------------
# (c) Even n is refused, not rounded
# --------------------------------------------------------------------------


def _option_kwargs() -> tuple[BlackScholesModel, Market]:
    return BlackScholesModel(sigma=0.2), _market(100.0, 0.05, 0.0)


@pytest.mark.parametrize("n_steps", [2, 10, 200, 800])
@pytest.mark.parametrize("exercise", ["european", "american"])
def test_even_n_steps_is_rejected(n_steps: int, exercise: str) -> None:
    """Evidence class: NEGATIVE_FINDING (a pinned refusal).

    `P(Bin(n, p) > n/2)` counts a whole number of outcomes only when `n` is
    odd, so there is no even-`n` Leisen-Reimer construction to round to. The
    documented, tested choice made here is to **reject**; QuantLib's
    `BinomialVanillaEngine` instead rounds `n` up to `n + 1` silently, which
    makes a convergence study report the wrong `h` and makes a "same n"
    comparison between schemes compare two different trees. The two engines
    still agree at odd `n`, which is all `tests/oracle/test_lr_vs_quantlib.py`
    ever asks them at.
    """
    model, market = _option_kwargs()
    option = (
        EuropeanOption(kind="call", strike=100.0, expiry=1.0)
        if exercise == "european"
        else AmericanOption(kind="put", strike=100.0, expiry=1.0)
    )
    cfg = TreeConfig(n_steps=n_steps, scheme=LR)

    with pytest.raises(InvalidInputError, match="requires an odd n_steps"):
        price(option, model, market, method="tree", cfg=cfg)
    with pytest.raises(InvalidInputError, match="requires an odd n_steps"):
        greeks(option, model, market, method="tree", cfg=cfg)


def test_even_n_is_rejected_before_the_degenerate_shortcuts() -> None:
    """Evidence class: NEGATIVE_FINDING (the ordering, pinned).

    Validation runs on the config, before any of the `T = 0` / `sigma = 0`
    shortcuts that would never build a lattice at all. Otherwise the same
    config would be an error at `sigma = 0.2` and silently fine at
    `sigma = 0`, which is worse than either answer on its own.
    """
    market = _market(100.0, 0.05, 0.0)
    cfg = TreeConfig(n_steps=200, scheme=LR)

    with pytest.raises(InvalidInputError, match="requires an odd n_steps"):
        price(
            EuropeanOption(kind="call", strike=100.0, expiry=0.0),
            BlackScholesModel(sigma=0.2),
            market,
            method="tree",
            cfg=cfg,
        )
    with pytest.raises(InvalidInputError, match="requires an odd n_steps"):
        price(
            EuropeanOption(kind="call", strike=100.0, expiry=1.0),
            BlackScholesModel(sigma=0.0),
            market,
            method="tree",
            cfg=cfg,
        )


def test_validation_order_puts_n_steps_before_the_parity_requirement() -> None:
    """`n_steps = 0` is a size problem, not a parity problem, and says so.

    Pinned because the two checks both fire on `TreeConfig(n_steps=0,
    scheme="leisen-reimer")` and the message the caller sees should be about
    the thing they most likely meant.
    """
    model, market = _option_kwargs()
    option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)

    with pytest.raises(InvalidInputError, match="n_steps must be >= 1"):
        price(
            option, model, market, method="tree", cfg=TreeConfig(n_steps=0, scheme=LR)
        )
    # Greeks need two levels; `n_steps = 2` clears that and then fails on
    # parity, so the smallest admissible Greek lattice here is n = 3.
    with pytest.raises(InvalidInputError, match="n_steps must be >= 2"):
        greeks(
            option, model, market, method="tree", cfg=TreeConfig(n_steps=1, scheme=LR)
        )
    with pytest.raises(InvalidInputError, match="requires an odd n_steps"):
        greeks(
            option, model, market, method="tree", cfg=TreeConfig(n_steps=2, scheme=LR)
        )
    assert greeks(
        option, model, market, method="tree", cfg=TreeConfig(n_steps=3, scheme=LR)
    ).delta > 0.0


def test_leisen_reimer_parameters_rejects_zero_volatility() -> None:
    """`d1` and `d2` do not exist at `sigma = 0`, so the constructor refuses.

    The pricers never reach it: `lattice_parameters` routes `sigma == 0` to
    the degenerate CRR lattice whatever the scheme, and the price itself comes
    from the closed form. The guard exists so that a direct caller gets an
    error rather than a `ZeroDivisionError` from inside the formula.
    """
    with pytest.raises(InvalidInputError, match="sigma must be finite and > 0"):
        leisen_reimer_parameters(
            spot=100.0,
            strike=100.0,
            sigma=0.0,
            expiry=1.0,
            rate=0.05,
            dividend_yield=0.0,
            n_steps=51,
        )
    with pytest.raises(InvalidInputError, match="odd n_steps"):
        leisen_reimer_parameters(
            spot=100.0,
            strike=100.0,
            sigma=0.2,
            expiry=1.0,
            rate=0.05,
            dividend_yield=0.0,
            n_steps=50,
        )
    with pytest.raises(InvalidInputError, match="odd positive integer"):
        peizer_pratt_inversion(0.5, 50)


# --------------------------------------------------------------------------
# (d) Degenerate limits, identical to the CRR scheme's
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "spot", "expected"),
    [("call", 105.0, 5.0), ("call", 95.0, 0.0), ("put", 95.0, 5.0), ("put", 105.0, 0.0)],
)
def test_expiry_zero_is_intrinsic_for_both_schemes(
    kind: str, spot: float, expected: float
) -> None:
    """Evidence class: CLOSED_FORM.

    At `T = 0` there is no lattice to parameterise, so the scheme cannot
    matter and the two must agree bit for bit -- they run the same three lines.
    """
    option = EuropeanOption(kind=kind, strike=100.0, expiry=0.0)  # type: ignore[arg-type]
    model, market = BlackScholesModel(sigma=0.2), _market(spot, 0.05, 0.01)

    crr = price(option, model, market, method="tree", cfg=TreeConfig(n_steps=51)).value
    lr = price(
        option, model, market, method="tree", cfg=TreeConfig(n_steps=51, scheme=LR)
    ).value
    assert lr == crr
    assert lr == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("exercise", ["european", "american"])
def test_zero_vol_matches_the_crr_scheme_and_the_closed_form(
    kind: str, exercise: str
) -> None:
    """Evidence class: CLOSED_FORM.

    At `sigma = 0` the Leisen-Reimer construction does not exist (`d1`, `d2`
    are undefined), and it is not needed: the spot is deterministic, so the
    European price is the discounted intrinsic value of the forward and the
    American price is the best discounted intrinsic along it. Both engines
    take the closed form and never touch a lattice, so the two schemes agree
    bit for bit, and the metadata still reports which scheme was asked for.
    """
    spot, strike, t, r, q = 110.0, 100.0, 1.0, 0.05, 0.02
    model, market = BlackScholesModel(sigma=0.0), _market(spot, r, q)
    option = (
        EuropeanOption(kind=kind, strike=strike, expiry=t)  # type: ignore[arg-type]
        if exercise == "european"
        else AmericanOption(kind=kind, strike=strike, expiry=t)  # type: ignore[arg-type]
    )

    crr_res = price(option, model, market, method="tree", cfg=TreeConfig(n_steps=137))
    lr_res = price(
        option, model, market, method="tree", cfg=TreeConfig(n_steps=137, scheme=LR)
    )
    assert lr_res.value == crr_res.value
    assert lr_res.meta is not None and lr_res.meta["scheme"] == LR
    assert lr_res.meta["degenerate"] == "zero_vol"

    if exercise == "european":
        forward = spot * math.exp((r - q) * t)
        payoff = (
            max(forward - strike, 0.0) if kind == "call" else max(strike - forward, 0.0)
        )
        assert lr_res.value == pytest.approx(math.exp(-r * t) * payoff, abs=1e-12)


def test_lattice_parameters_routes_zero_vol_to_the_degenerate_crr_lattice() -> None:
    """Evidence class: EXACT_IDENTITY.

    The dispatcher's one special case, pinned: `sigma = 0` collapses both
    branches onto the forward, which is the same lattice for every scheme, so
    asking for Leisen-Reimer there returns the CRR degenerate object rather
    than raising from inside a formula that needs `log(S/K) / 0`.
    """
    common = {
        "sigma": 0.0,
        "expiry": 1.0,
        "rate": 0.05,
        "dividend_yield": 0.01,
        "n_steps": 51,
    }
    routed = lattice_parameters(
        scheme=LR, spot=100.0, strike=90.0, **common  # type: ignore[arg-type]
    )
    assert routed == crr_parameters(**common)  # type: ignore[arg-type]
    assert routed.degenerate


def test_meta_reports_the_scheme_and_the_realised_lattice() -> None:
    result = price(
        EuropeanOption(kind="call", strike=100.0, expiry=1.0),
        BlackScholesModel(sigma=0.2),
        _market(100.0, 0.05, 0.0),
        method="tree",
        cfg=TreeConfig(n_steps=101, scheme=LR),
    )
    meta = result.meta
    assert meta is not None
    assert meta["method"] == "tree"
    assert meta["scheme"] == LR
    assert meta["n_steps"] == 101
    assert meta["dt"] == pytest.approx(1.0 / 101)
    # The signature of the scheme, right there in the metadata: unlike CRR,
    # u * d is not 1.
    assert float(meta["u"]) * float(meta["d"]) != pytest.approx(1.0, abs=1e-6)
    assert 0.0 < float(meta["p"]) < 1.0


def test_american_lr_dominates_european_lr_and_collapses_when_it_should() -> None:
    """Evidence class: EXACT_IDENTITY.

    The two American facts that are structural rather than numerical, checked
    on the Leisen-Reimer lattice because they are properties of the Bellman
    step and must survive a change of lattice:

    - an American put is worth at least the European put on the same lattice;
    - with `q = 0` an American call never exercises early, so it equals the
      European call on the same lattice.

    The call identity is bit-for-bit: the Bellman maximum is a no-op at every
    node and `max(a, b)` returns `b` itself. The put comparison only needs the
    ordering.
    """
    model, market = BlackScholesModel(sigma=0.2), _market(100.0, 0.05, 0.0)
    cfg = TreeConfig(n_steps=201, scheme=LR)

    american_put = price(
        AmericanOption(kind="put", strike=100.0, expiry=1.0),
        model,
        market,
        method="tree",
        cfg=cfg,
    ).value
    european_put = price(
        EuropeanOption(kind="put", strike=100.0, expiry=1.0),
        model,
        market,
        method="tree",
        cfg=cfg,
    ).value
    assert american_put > european_put

    american_call = price(
        AmericanOption(kind="call", strike=100.0, expiry=1.0),
        model,
        market,
        method="tree",
        cfg=cfg,
    ).value
    european_call = price(
        EuropeanOption(kind="call", strike=100.0, expiry=1.0),
        model,
        market,
        method="tree",
        cfg=cfg,
    ).value
    assert american_call == pytest.approx(european_call, abs=1e-12)


# --------------------------------------------------------------------------
# (e) Lattice Greeks: measured order 1, not 2
# --------------------------------------------------------------------------

_GREEK_POINTS = (
    ("atm", 100.0, 100.0, 1.0, 0.05, 0.00, 0.20),
    ("otm_k120", 100.0, 120.0, 1.0, 0.05, 0.00, 0.20),
    ("itm_div", 120.0, 90.0, 1.0, 0.03, 0.05, 0.35),
)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(
    _ARGNAMES,
    [row[1:] for row in _GREEK_POINTS],
    ids=[row[0] for row in _GREEK_POINTS],
)
def test_lattice_greeks_are_first_order_not_second(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: CONVERGENCE_ORDER. **Contradicts the slice statement.**

    This slice was written expecting delta off the Leisen-Reimer lattice to
    converge at about order 2 at the money. It does not: the measured order is
    **1.00**, over `n` in {25, 51, 101, 201, 401} against the closed-form
    Greeks, at every point and both kinds.

        point       kind   delta          gamma          theta
        atm         call   0.996/1.2e-04  1.003/2.8e-05  0.995/5.5e-04
        atm         put    0.996/1.2e-04  1.003/2.8e-05  1.066/4.0e-05
        otm_k120    call   0.998/2.8e-04  1.003/1.8e-05  1.000/2.0e-02
        otm_k120    put    0.998/2.8e-04  1.003/1.8e-05  1.000/1.9e-02
        itm_div     call   0.999/1.4e-04  1.008/2.9e-05  1.002/2.2e-02
        itm_div     put    1.000/2.5e-04  1.008/2.9e-05  1.005/1.9e-02

    (fitted order / absolute error at n = 401; log-space RMS residuals all
    below 0.021, the largest being the at-the-money put's theta, whose error
    is so small -- 4.0e-05 -- that two error terms are cancelling inside it.)

    The reason is that Leisen-Reimer fixes a *different* error. Its second
    order is about the terminal distribution: the strike is placed where the
    binomial and normal tails already agree. Delta, gamma and theta are not
    read at the root; they are read at time levels 1 and 2 and used as
    estimates at time 0. That substitution is an `O(dt)` error no matter how
    good the lattice is, and `O(dt) = O(1/n)` dominates the `O(1/n**2)` price
    error. Getting order 2 out of these Greeks would need a different
    estimator (an extended lattice below the root, say), not a better tree.

    What Leisen-Reimer does buy for the Greeks is the *constant*: at the money
    at `n = 401` the CRR delta error is 1.4e-03 against this scheme's 1.2e-04,
    about twelve times larger, and CRR's oscillates with the parity of `n`
    while this does not. That is measured in
    `tests/test_tree_lr_convergence.py`.
    """
    option = EuropeanOption(kind=kind, strike=strike, expiry=expiry)  # type: ignore[arg-type]
    model, market = BlackScholesModel(sigma=sigma), _market(spot, rate, div)
    exact = greeks(option, model, market, method="analytic")

    errors: dict[str, list[float]] = {"delta": [], "gamma": [], "theta": []}
    for n in ODD_LEVELS:
        got = greeks(
            option, model, market, method="tree", cfg=TreeConfig(n_steps=n, scheme=LR)
        )
        for name in errors:
            errors[name].append(abs(getattr(got, name) - getattr(exact, name)))

    h = [1.0 / n for n in ODD_LEVELS]
    for name, errs in errors.items():
        fit = fit_convergence_order(h, errs)
        assert 0.85 <= fit.order <= 1.15, (name, fit.order)
        assert fit.residual < 0.05, (name, fit.residual)
        # And emphatically not order 2, which is the claim being refuted.
        assert fit.order < 1.5, (name, fit.order)


def test_theta_needs_the_off_centre_correction_on_an_lr_lattice() -> None:
    """Evidence class: NEGATIVE_FINDING.

    Pinned because it is the one place the Leisen-Reimer lattice broke shared
    code, and because the failure is silent rather than loud: the uncorrected
    estimator returns a plausible-looking number.

    `lattice_delta_gamma_theta` reads theta as `(V(2,1) - V(0,0)) / (2 dt)`,
    which is a pure difference in *time* only when `S(2,1) = S(0,0)` -- true
    on a CRR lattice, where `u d = 1`, and false here. At `S = 100, K = 120,
    T = 1, n = 801` the middle step-2 node sits 4.6e-02 above the spot, and
    the uncorrected difference therefore also contains `offset * delta`.

    Measured: uncorrected, the error against the closed-form theta is 5.235
    at `n = 801` with a fitted order of **-0.002** -- it does not converge at
    all. Corrected, the error is 9.8e-03 with a fitted order of 1.000. The
    check below reproduces the uncorrected estimator in two lines from the
    engine's own delta and gamma and confirms the gap is real at this
    specification.
    """
    spot, strike, expiry, rate, div, sigma = 100.0, 120.0, 1.0, 0.05, 0.0, 0.20
    option = EuropeanOption(kind="call", strike=strike, expiry=expiry)
    model, market = BlackScholesModel(sigma=sigma), _market(spot, rate, div)

    n_steps = 801
    lattice = leisen_reimer_parameters(
        spot=spot,
        strike=strike,
        sigma=sigma,
        expiry=expiry,
        rate=rate,
        dividend_yield=div,
        n_steps=n_steps,
    )
    offset = spot * lattice.up * lattice.down - spot
    assert abs(offset) > 1e-2, offset

    got = greeks(
        option, model, market, method="tree", cfg=TreeConfig(n_steps=n_steps, scheme=LR)
    )
    exact = greeks(option, model, market, method="analytic")

    # Undo the correction the engine applied, recovering the naive estimator.
    naive = got.theta + (
        offset * got.delta + 0.5 * offset * offset * got.gamma
    ) / (2.0 * lattice.dt)

    assert abs(got.theta - exact.theta) < 2e-2
    assert abs(naive - exact.theta) > 1.0
