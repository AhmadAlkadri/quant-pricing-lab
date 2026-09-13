"""Measured evidence for the CRR binomial tree.

Four kinds of claim, kept apart because they are justified differently:

(a) EXACT_IDENTITY -- a one-step and a two-step tree reproduce the value of the
    replicating portfolio, computed here from the hedging equations rather than
    from the risk-neutral formula the engine uses.
(b) EXACT_IDENTITY -- put-call parity holds on the lattice to round-off, for
    every `n`, because the tree reprices the forward exactly.
(c) CONVERGENCE_ORDER -- the error against the closed form decays like `1/n`,
    separately on odd and on even `n`, and the two subsequences sit on
    opposite sides of the Black-Scholes value.
(d) EXACT_IDENTITY -- the American tree engine and this European tree agree
    bit-for-bit when early exercise is never optimal, which is the observable
    consequence of their sharing one lattice and one continuation expression.
    The complementary check, that the American numbers did not move when the
    Slice 1 keyword entry point was retired, is pinned in
    `tests/test_tree_american.py::test_american_put_matches_the_retired_dp_engine_bit_for_bit`.

A fitted slope is not a theorem: the orders below are measurements of this
implementation on these refinement sequences. The order-1 behaviour and the
odd/even oscillation are established in Leisen and Reimer (1996), "Binomial
models for option valuation -- examining and improving convergence", Applied
Mathematical Finance 3(4), 319-346; the tree itself is Cox, Ross and
Rubinstein (1979), Journal of Financial Economics 7, 229-263. The derivation
and the full tables are in `docs/notes/crr_tree_convergence.md`.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from qpl.engines.tree import TreeConfig
from qpl.instruments.options import AmericanOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price
from qpl.validation import fit_convergence_order

# Reference point for the convergence study, and the one used by
# `docs/notes/crr_tree_convergence.md` and `examples/tree_convergence.py`:
# at the money, so the strike sits where the terminal grid is densest and the
# odd/even effect is at its cleanest.
CONV_SPOT = 100.0
CONV_STRIKE = 100.0
CONV_EXPIRY = 1.0
CONV_RATE = 0.05
CONV_DIVIDEND = 0.0
CONV_SIGMA = 0.20

ODD_LEVELS = (25, 51, 101, 201, 401, 801)
EVEN_LEVELS = (26, 50, 100, 200, 400, 800)


def _market(spot: float, r: float, q: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(r, allow_negative=True),
        dividend_curve=FlatDividendCurve(q, allow_negative=True),
    )


def _tree(option: EuropeanOption, sigma: float, market: Market, n: int) -> float:
    return price(
        option,
        BlackScholesModel(sigma=sigma),
        market,
        method="tree",
        cfg=TreeConfig(n_steps=n),
    ).value


def _conv_setup(kind: str = "call") -> tuple[EuropeanOption, Market, float]:
    option = EuropeanOption(kind=kind, strike=CONV_STRIKE, expiry=CONV_EXPIRY)  # type: ignore[arg-type]
    market = _market(CONV_SPOT, CONV_RATE, CONV_DIVIDEND)
    analytic = price(
        option, BlackScholesModel(sigma=CONV_SIGMA), market, method="analytic"
    ).value
    return option, market, analytic


# --------------------------------------------------------------------------
# (a) Hand-computed replication
# --------------------------------------------------------------------------


def _replicating_value(
    *, v_up: float, v_down: float, spot: float, u: float, d: float, r: float, q: float, dt: float
) -> float:
    """Cost of the portfolio that replicates a one-period claim.

    Hold `delta` shares and `cash` in the money market. Over one step the share
    position grows by the reinvested dividend yield, `e^{q dt}`, and the cash
    position by `e^{r dt}`, so replication demands

        delta * spot * u * e^{q dt} + cash * e^{r dt} = v_up
        delta * spot * d * e^{q dt} + cash * e^{r dt} = v_down

    which solve to `delta = (v_up - v_down) / (spot (u - d) e^{q dt})` and
    `cash = e^{-r dt} (v_up - delta * spot * u * e^{q dt})`. The claim's value
    today is `delta * spot + cash`.

    This is the hedging argument, not the risk-neutral formula: no probability
    appears anywhere above. That the engine agrees is the content of the test.
    """
    carry = math.exp(q * dt)
    delta = (v_up - v_down) / (spot * (u - d) * carry)
    cash = math.exp(-r * dt) * (v_up - delta * spot * u * carry)
    return delta * spot + cash


# Measured residual between the replication value and the engine for these
# four cases: at most 2.3e-14 on prices of 6 to 16, i.e. round-off from a
# different order of operations. The tolerance is a round-off budget.
_REPLICATION_TOLERANCE = 1e-12


@pytest.mark.parametrize("kind", ["call", "put"])
def test_one_step_tree_matches_hand_replication(kind: str) -> None:
    """Evidence class: EXACT_IDENTITY."""
    spot, strike, expiry, r, q, sigma = 100.0, 95.0, 1.0, 0.05, 0.02, 0.25
    option = EuropeanOption(kind=kind, strike=strike, expiry=expiry)  # type: ignore[arg-type]

    dt = expiry
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u

    def payoff(s: float) -> float:
        return max(s - strike, 0.0) if kind == "call" else max(strike - s, 0.0)

    expected = _replicating_value(
        v_up=payoff(spot * u),
        v_down=payoff(spot * d),
        spot=spot,
        u=u,
        d=d,
        r=r,
        q=q,
        dt=dt,
    )
    engine = _tree(option, sigma, _market(spot, r, q), 1)
    assert engine == pytest.approx(expected, abs=_REPLICATION_TOLERANCE)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_two_step_tree_matches_hand_replication(kind: str) -> None:
    """Evidence class: EXACT_IDENTITY.

    Two steps means replicating twice: once at each step-1 node, then once at
    the root with the two step-1 replication costs as its payoffs. Nothing but
    the hedging equations is used.
    """
    spot, strike, expiry, r, q, sigma = 100.0, 95.0, 1.0, 0.05, 0.02, 0.25
    option = EuropeanOption(kind=kind, strike=strike, expiry=expiry)  # type: ignore[arg-type]

    dt = expiry / 2.0
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u

    def payoff(s: float) -> float:
        return max(s - strike, 0.0) if kind == "call" else max(strike - s, 0.0)

    v_uu, v_ud, v_dd = payoff(spot * u * u), payoff(spot * u * d), payoff(spot * d * d)
    common = {"u": u, "d": d, "r": r, "q": q, "dt": dt}
    v_u = _replicating_value(v_up=v_uu, v_down=v_ud, spot=spot * u, **common)
    v_d = _replicating_value(v_up=v_ud, v_down=v_dd, spot=spot * d, **common)
    expected = _replicating_value(v_up=v_u, v_down=v_d, spot=spot, **common)

    engine = _tree(option, sigma, _market(spot, r, q), 2)
    assert engine == pytest.approx(expected, abs=_REPLICATION_TOLERANCE)


# --------------------------------------------------------------------------
# (b) Put-call parity on the lattice
# --------------------------------------------------------------------------

_PARITY_POINTS = (
    ("atm_1y", 100.0, 100.0, 1.0, 0.05, 0.00, 0.20),
    ("otm_9m_div", 100.0, 110.0, 0.75, 0.03, 0.01, 0.25),
    ("deep_otm_2y", 80.0, 120.0, 2.0, 0.02, 0.03, 0.40),
)

# Measured parity residuals grow like n * eps * price -- the discount factor is
# applied n times as e^{-r dt} rather than once as e^{-rT} -- reaching 8.3e-12
# at n=800 and 1.5e-11 at n=2000 over these three points. 1e-10 is a round-off
# budget with roughly a factor of seven of headroom at the largest n tested.
_PARITY_TOLERANCE = 1e-10


@pytest.mark.parametrize("n_steps", [1, 2, 7, 50, 101, 800, 2000])
@pytest.mark.parametrize(
    ("spot", "strike", "expiry", "rate", "div", "sigma"),
    [row[1:] for row in _PARITY_POINTS],
    ids=[row[0] for row in _PARITY_POINTS],
)
def test_put_call_parity_holds_on_the_tree(
    n_steps: int,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: EXACT_IDENTITY.

    `C - P = S e^{-qT} - K e^{-rT}` is a statement about the forward, and the
    CRR lattice reprices the forward exactly: the one-step expected spot is
    `p u + (1-p) d = e^{(r-q) dt}` by construction of `p`, so the `n`-step
    expected terminal spot is `S e^{(r-q)T}` with no discretisation error at
    all. Parity therefore holds at *every* `n`, coarse trees included, and the
    residual is round-off rather than model error -- which is exactly why this
    test is worth having: it fails on a sign or discounting slip that the
    convergence tests would absorb into the error constant.
    """
    market = _market(spot, rate, div)
    call = _tree(EuropeanOption(kind="call", strike=strike, expiry=expiry), sigma, market, n_steps)
    put = _tree(EuropeanOption(kind="put", strike=strike, expiry=expiry), sigma, market, n_steps)

    forward_leg = spot * math.exp(-div * expiry)
    strike_leg = strike * math.exp(-rate * expiry)
    residual = (call - put) - (forward_leg - strike_leg)
    assert abs(residual) <= _PARITY_TOLERANCE


# --------------------------------------------------------------------------
# (c) Measured order, and the odd/even oscillation
# --------------------------------------------------------------------------


def _signed_errors(kind: str, levels: tuple[int, ...]) -> list[float]:
    option, market, analytic = _conv_setup(kind)
    return [_tree(option, CONV_SIGMA, market, n) - analytic for n in levels]


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(
    ("parity", "levels"), [("odd", ODD_LEVELS), ("even", EVEN_LEVELS)]
)
def test_crr_is_first_order_on_each_parity_subsequence(
    kind: str, parity: str, levels: tuple[int, ...]
) -> None:
    """Evidence class: CONVERGENCE_ORDER.

    Fitting odd and even `n` together would fit a single power law to two
    interleaved sequences with different constants *and* different signs; the
    slope would still come out near 1, but the log-space residual would be
    reporting the oscillation rather than the rate. Fitting them separately is
    what makes the number mean something.

    Measured on this point (`h = 1/n`): odd order 1.0010 with log-space RMS
    residual 0.0005, even order 0.9987 with residual 0.0007. The band
    [0.8, 1.2] matches the one used for the implicit-Euler time order in
    `tests/test_pde_ch4.py`; the residual bound of 0.01 is far tighter than
    that test's because these sequences are essentially exact power laws.
    """
    errs = [abs(e) for e in _signed_errors(kind, levels)]
    fit = fit_convergence_order([1.0 / n for n in levels], errs)

    assert 0.8 <= fit.order <= 1.2, (parity, fit.order)
    assert fit.residual < 0.01, (parity, fit.residual)
    assert fit.n_points == len(levels)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_odd_and_even_subsequences_bracket_black_scholes(kind: str) -> None:
    """Evidence class: CONVERGENCE_ORDER (the sign structure of the error).

    Measured: the tree price is *above* Black-Scholes at every odd `n` tested
    and *below* it at every even `n`, so the two subsequences bracket the true
    value. At the money, an even `n` puts a terminal node exactly at the
    strike (`d = 1/u`, so node `n/2` of level `n` is `S0 = K`); an odd `n`
    leaves the strike between the two nodes that straddle it. That is the
    source of the oscillation: the discrete terminal distribution resolves the
    payoff kink differently depending on the parity of `n`.

    The scaled errors `n * |error|` converge to about 1.7529 on odd `n` and
    about 1.9994 on even `n` -- two different order-1 constants, one per
    parity, which is precisely what "an oscillating coefficient" means here.
    """
    odd = _signed_errors(kind, ODD_LEVELS)
    even = _signed_errors(kind, EVEN_LEVELS)

    assert all(e > 0.0 for e in odd), odd
    assert all(e < 0.0 for e in even), even
    # Bracketing is the useful consequence: the true value lies between the
    # two subsequences at every refinement level.
    assert max(even) < 0.0 < min(odd)

    odd_constants = [n * abs(e) for n, e in zip(ODD_LEVELS, odd, strict=True)]
    even_constants = [n * abs(e) for n, e in zip(EVEN_LEVELS, even, strict=True)]
    assert odd_constants[-1] == pytest.approx(1.7529, abs=5e-3)
    assert even_constants[-1] == pytest.approx(1.9994, abs=5e-3)


def test_richardson_extrapolation_of_two_odd_n_improves_the_order() -> None:
    """Evidence class: CONVERGENCE_ORDER.

    Within one parity the error is `C/n + o(1/n)` with a constant that no
    longer oscillates, so two levels of the same parity can be combined to
    cancel the leading term: from `V(n1)` and `V(n2)`,

        V_ext = (n2 V(n2) - n1 V(n1)) / (n2 - n1).

    Extrapolating the consecutive odd pairs (25,51), (51,101), ..., (401,801)
    and fitting the extrapolated error against `h = 1/n1` gives a measured
    order of **1.959** with log-space residual 0.022 -- close to, but not
    exactly, the 2 that a clean `C/n + D/n**2` expansion would give. The
    shortfall is real and expected: the pairs are `(n, 2n+1)` rather than
    `(n, 2n)`, and the second-order coefficient within a parity is itself only
    asymptotically constant. The band below admits that, and would still catch
    an extrapolation that had failed to improve on the order-1 raw sequence.

    Errors after extrapolation: 1.30e-04 at n1=25 down to 5.71e-07 at n1=401,
    against 7.04e-02 and 4.37e-03 for the raw odd sequence at the same n1 --
    a factor of 540 and 7700 respectively.
    """
    option, market, analytic = _conv_setup("call")

    pairs = tuple(pairwise(ODD_LEVELS))
    h, errs = [], []
    for n1, n2 in pairs:
        v1 = _tree(option, CONV_SIGMA, market, n1)
        v2 = _tree(option, CONV_SIGMA, market, n2)
        extrapolated = (n2 * v2 - n1 * v1) / (n2 - n1)
        h.append(1.0 / n1)
        errs.append(abs(extrapolated - analytic))

    fit = fit_convergence_order(h, errs)
    assert 1.8 <= fit.order <= 2.2, fit.order
    assert fit.residual < 0.1, fit.residual

    # And it is an improvement, not merely a different slope: at the coarsest
    # and finest pairs the extrapolated error is orders of magnitude smaller
    # than the raw error at the same n1.
    raw = [abs(e) for e in _signed_errors("call", ODD_LEVELS)]
    assert errs[0] < raw[0] / 100.0
    assert errs[-1] < raw[-2] / 1000.0


# --------------------------------------------------------------------------
# (d) American and European exercise share one lattice
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_steps", [51, 200, 501])
def test_american_put_equals_european_tree_when_r_is_zero(n_steps: int) -> None:
    """Evidence class: EXACT_IDENTITY.

    Early exercise of a put pays for itself out of interest earned on the
    strike; at `r = 0` (and a positive dividend yield, which makes waiting
    strictly more attractive still) it is never optimal, so the American value
    equals the European value. Both engines build their lattice with
    `qpl.engines.tree.lattice`, and the Bellman step reduces to the same
    continuation expression, so the agreement is bit-for-bit -- measured at
    exactly 0.0 difference for all three `n`. A tolerance of 1e-12 is kept as
    the stated budget in case a future compiler or numpy reorders the
    reduction.
    """
    spot, strike, expiry, r, q, sigma = 100.0, 105.0, 1.0, 0.0, 0.03, 0.30
    market = _market(spot, r, q)

    american = price(
        AmericanOption(kind="put", strike=strike, expiry=expiry),
        BlackScholesModel(sigma=sigma),
        market,
        method="tree",
        cfg=TreeConfig(n_steps=n_steps),
    ).value
    european = _tree(
        EuropeanOption(kind="put", strike=strike, expiry=expiry), sigma, market, n_steps
    )

    assert american == pytest.approx(european, abs=1e-12)
    assert american >= european - 1e-12
