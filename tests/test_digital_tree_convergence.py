"""What a binomial lattice does to a discontinuous payoff, measured.

The slice that wrote this file expected **order 1 with large oscillation for
both schemes, with Leisen-Reimer failing to cure it**. Both halves of that are
wrong, and the truth is more interesting in both directions:

- **CRR is order 1 only at the money on odd `n`**, where the strike is the
  geometric mean of the two central terminal nodes and therefore sits exactly
  log-midway between them by symmetry. Off the money it is **order 1/2** on a
  block-RMS fit (measured 0.5020 and 0.5037) and the individual errors do not
  decay at all in any usable sense: over odd `n` from 25 to 801 they change
  sign ten and nineteen times, span 5.11e-02 down to 3.35e-06, and the error
  at `n = 801` can be larger than the error at `n = 401`. Pinned below as a
  NEGATIVE_FINDING.

- **Leisen-Reimer is order 2 on a digital, everywhere, with no oscillation.**
  Measured 1.9844, 1.9839 and 1.9836 at the three points, errors one-signed
  and strictly decreasing, and 1.3e+03 to 2.6e+05 times smaller than CRR's at
  `n = 801`. It does not merely survive the discontinuity: the digital is the
  contract the construction is *best* at, and the mechanism is exact rather
  than statistical --

      the strike always falls strictly between the two central terminal nodes,
      so the tree's digital price is exactly
      `cash e^{-rT} P(Bin(n, p) > n / 2)`,

  and `p` is chosen by the Peizer-Pratt inversion precisely so that this
  probability equals `N(d2)` to high order. For a vanilla, Leisen-Reimer gets
  order 2 by matching two tails that the price then combines; for a digital
  the price *is* one of those tails, with nothing else in it. Both statements
  -- the node bracket and the exact equality with the discounted binomial tail
  -- are asserted below to 1e-13, so the mechanism is measured and not
  narrated.

Evidence classes are named on each test. The schemes are Cox, Ross and
Rubinstein (1979) and Leisen and Reimer (1996); every number here was measured
in this repository.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
from scipy.stats import binom

from qpl.engines.analytic.digital import digital_price, greeks_digital as analytic_greeks
from qpl.engines.tree import TreeConfig
from qpl.engines.tree.lattice import crr_spot_level, lattice_parameters
from qpl.exceptions import InvalidInputError
from qpl.instruments.options import DigitalOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass, fit_convergence_order

ODD_LEVELS = (25, 51, 101, 201, 401, 801)
"""The odd step counts the slice asks for; `h = 1 / n`. Odd throughout, since
Leisen-Reimer has no even-`n` construction."""

# (id, spot, strike, expiry, rate, dividend, sigma)
_POINTS = (
    ("atm_1y", 100.0, 100.0, 1.00, 0.05, 0.00, 0.20),
    ("otm_9m_div", 100.0, 110.0, 0.75, 0.03, 0.01, 0.25),
    ("itm_1y_div", 120.0, 90.0, 1.00, 0.03, 0.05, 0.35),
)
_ARGNAMES = ("spot", "strike", "expiry", "rate", "div", "sigma")
_ARGS = [row[1:] for row in _POINTS]
_IDS = [row[0] for row in _POINTS]
_OFF_MONEY = [row for row in _POINTS if row[0] != "atm_1y"]


def _triple(kind, spot, strike, expiry, rate, div, sigma, cash=1.0):
    return (
        DigitalOption(kind=kind, strike=strike, expiry=expiry, cash=cash),
        BlackScholesModel(sigma=sigma),
        Market(
            spot=spot,
            rate_curve=FlatRateCurve(rate),
            dividend_curve=FlatDividendCurve(div),
        ),
    )


def _signed_errors(triple, scheme: str, levels=ODD_LEVELS) -> list[float]:
    option, model, market = triple
    exact = digital_price(
        S=market.spot,
        K=option.strike,
        T=option.expiry,
        r=market.rate(option.expiry),
        sigma=model.sigma,
        q=market.dividend_yield(option.expiry),
        cash=option.cash,
        kind=option.kind,
    )
    return [
        price(
            option,
            model,
            market,
            method="tree",
            cfg=TreeConfig(n_steps=n, scheme=scheme),  # type: ignore[arg-type]
        ).value
        - exact
        for n in levels
    ]


# --------------------------------------------------------------------------
# The Leisen-Reimer mechanism, stated exactly.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
@pytest.mark.parametrize("n_steps", [25, 101, 801])
def test_leisen_reimer_digital_is_the_discounted_binomial_tail(
    n_steps: int,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: EXACT_IDENTITY, on the discretisation rather than the model.

    Two claims, both algebraic rather than asymptotic:

    1. the strike lies strictly between terminal nodes `(n-1)/2` and `(n+1)/2`,
       so the set of in-the-money terminal nodes is exactly "more than `n/2` up
       moves", whatever the point;
    2. the lattice price is therefore `cash e^{-rT} P(Bin(n, p) > n/2)` exactly,
       which is checked against `scipy.stats.binom.sf` -- an independent
       evaluation of the same tail.

    Together these say the Leisen-Reimer digital price *is* the Peizer-Pratt
    approximation of `N(d2)`, discounted. Order 2 on the price then follows
    from the accuracy of that inversion, which is the scheme's own claim, and
    is measured separately below.
    """
    lattice = lattice_parameters(
        scheme="leisen-reimer",
        spot=spot,
        strike=strike,
        sigma=sigma,
        expiry=expiry,
        rate=rate,
        dividend_yield=div,
        n_steps=n_steps,
    )
    spots = crr_spot_level(
        spot=spot, up=lattice.up, down=lattice.down, level=n_steps
    )
    median = (n_steps + 1) // 2
    assert spots[median - 1] < strike < spots[median]

    tree = price(
        *_triple("call", spot, strike, expiry, rate, div, sigma),
        method="tree",
        cfg=TreeConfig(n_steps=n_steps, scheme="leisen-reimer"),
    ).value
    tail = float(binom.sf(n_steps / 2.0, n_steps, lattice.p))
    assert tree == pytest.approx(math.exp(-rate * expiry) * tail, abs=1e-13)


# --------------------------------------------------------------------------
# Measured orders.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_leisen_reimer_is_second_order_on_a_digital(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: CONVERGENCE_ORDER.

    Measured over `n` in (25, 51, 101, 201, 401, 801), call leg: order 1.9844
    (atm, residual 0.0085), 1.9839 (otm, 0.0088), 1.9836 (itm, 0.0089). The
    puts are the exact mirror -- their errors are the calls' negated, because
    the two lattice prices sum to `cash e^{-rT}` node by node -- so both are
    checked.

    The band is `2.0 +- 0.2`, the same band the vanilla Leisen-Reimer rows use
    in `qpl.cases`; like those, the fit sits just *below* 2 because the
    Peizer-Pratt tail match is high but finite order.
    """
    errors = _signed_errors(_triple(kind, spot, strike, expiry, rate, div, sigma), "leisen-reimer")
    fit = fit_convergence_order([1.0 / n for n in ODD_LEVELS], [abs(e) for e in errors])

    assert abs(fit.order - 2.0) <= 0.2, (fit.order, errors)
    assert fit.residual < 0.05, fit.residual
    # One-signed and strictly decreasing: no oscillation to average away.
    assert all(e > 0.0 for e in errors) or all(e < 0.0 for e in errors), errors
    assert all(abs(a) > abs(b) for a, b in pairwise(errors)), errors


@pytest.mark.parametrize("kind", ["call", "put"])
def test_crr_is_first_order_at_the_money_on_odd_steps(kind: str) -> None:
    """Evidence class: CONVERGENCE_ORDER, and a special case rather than a rule.

    At `S = K` on a CRR lattice the terminal nodes are `K u^j d^{n-j}` with
    `u d = 1`, so node `(n-1)/2` and node `(n+1)/2` have geometric mean exactly
    `K`: the strike is log-midway between them for **every** odd `n`. The
    fractional position that drives the error is therefore frozen, and the
    error decays cleanly: measured order 1.0014 with log-space residual 0.0008
    and no sign change over `n` in (25 ... 801).

    Nothing here generalises off the money; that is the next test.
    """
    errors = _signed_errors(_triple(kind, 100.0, 100.0, 1.0, 0.05, 0.0, 0.20), "crr")
    fit = fit_convergence_order([1.0 / n for n in ODD_LEVELS], [abs(e) for e in errors])

    assert abs(fit.order - 1.0) <= 0.2, (fit.order, errors)
    assert fit.residual < 0.01, fit.residual
    assert all(e > 0.0 for e in errors) or all(e < 0.0 for e in errors), errors

    # The symmetry that makes it work, asserted rather than described.
    for n in (25, 101, 801):
        lattice = lattice_parameters(
            scheme="crr",
            spot=100.0,
            strike=100.0,
            sigma=0.20,
            expiry=1.0,
            rate=0.05,
            dividend_yield=0.0,
            n_steps=n,
        )
        spots = crr_spot_level(spot=100.0, up=lattice.up, down=lattice.down, level=n)
        median = (n + 1) // 2
        assert math.sqrt(spots[median - 1] * spots[median]) == pytest.approx(100.0, rel=1e-12)


@pytest.mark.parametrize(
    _ARGNAMES, [row[1:] for row in _OFF_MONEY], ids=[row[0] for row in _OFF_MONEY]
)
def test_crr_off_the_money_is_half_order_with_sign_changing_oscillation(
    spot: float, strike: float, expiry: float, rate: float, div: float, sigma: float
) -> None:
    """Evidence class: NEGATIVE_FINDING. This is the pathology the slice exists for.

    A CRR lattice records only *which side of the strike* each terminal node
    fell on. Moving the strike across a node changes the discrete payoff there
    by a whole `cash`, and the price by `cash` times that node's risk-neutral
    probability, which is `O(1 / sqrt(n))`. So the error is `O(1 / sqrt(n))`
    with a coefficient that sweeps the full range as `n` increments -- not an
    odd/even parity effect, which averaging could remove, but a continuously
    drifting fractional position.

    Measured on a block-RMS fit over every odd `n` from 25 to 801, grouped into
    the five windows (25-50, 51-100, 101-200, 201-400, 401-801):

    | point      | order  | residual | RMS at 25-50 | RMS at 401-801 | sign changes |
    |------------|--------|----------|--------------|----------------|--------------|
    | otm_9m_div | 0.5020 | 0.1254   | 3.84e-02     | 8.39e-03       | 10           |
    | itm_1y_div | 0.5037 | 0.0728   | 3.38e-02     | 8.22e-03       | 19           |

    Individual errors span 5.11e-02 down to 3.35e-06 at otm and 5.67e-02 down
    to 6.76e-06 at itm: a single `n` tells you nothing.

    Seen at consecutive odd `n` the error is a **sawtooth**, which is the
    mechanism made visible. Over `n` in 101 ... 201 at otm it ramps smoothly
    from +1.5e-02 to +3.1e-02 as the strike drifts across a terminal cell, then
    drops to -2.9e-02 in a single step of `n` when the strike crosses a node
    and that node's whole probability mass changes sides; then it ramps again.
    The jump is `cash` times that node's risk-neutral probability -- `O(1 / sqrt(n))`
    -- and the ramp is the fractional position drifting. The window has 2 sign
    changes at otm and 3 at itm, and a largest-to-smallest error ratio of 113
    and 38.
    """
    triple = _triple("call", spot, strike, expiry, rate, div, sigma)

    errors = _signed_errors(triple, "crr")
    fit = fit_convergence_order([1.0 / n for n in ODD_LEVELS], [abs(e) for e in errors])
    # Emphatically not order 1, and not a power law at all on a single-`n` fit.
    assert fit.order < 0.7, (fit.order, errors)
    assert fit.residual > 0.05, fit.residual

    # The sawtooth, on consecutive odd `n`.
    window = tuple(range(101, 202, 2))
    windowed = np.array(_signed_errors(triple, "crr", levels=window))
    assert np.sum(np.diff(np.sign(windowed)) != 0) >= 2, windowed
    assert np.max(np.abs(windowed)) > 30.0 * np.min(np.abs(windowed)), windowed

    # A ramp with a jump in it: the largest step between neighbouring `n` is an
    # order of magnitude above the typical one, which a smooth error would not
    # produce.
    steps = np.abs(np.diff(windowed))
    assert np.max(steps) > 10.0 * np.median(steps), steps

    # And refining by 2x is not reliably an improvement.
    assert max(abs(errors[-1]), abs(errors[-2])) > 0.25 * abs(errors[0]), errors


@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_leisen_reimer_beats_crr_by_three_orders_of_magnitude(
    spot: float, strike: float, expiry: float, rate: float, div: float, sigma: float
) -> None:
    """Evidence class: CONVERGENCE_ORDER, as a ratio rather than two slopes.

    At `n = 801`, measured |error| CRR / Leisen-Reimer: 4.55e+03 (atm),
    4.68e+05 (otm), 6.34e+05 (itm). The floor asserted is 1e+03, which the
    weakest cell -- the at-the-money point, where CRR is at its *best* -- clears
    by 4.5x.
    """
    triple = _triple("call", spot, strike, expiry, rate, div, sigma)
    crr = abs(_signed_errors(triple, "crr", levels=(801,))[0])
    lr = abs(_signed_errors(triple, "leisen-reimer", levels=(801,))[0])
    assert crr > 1_000.0 * lr, (crr, lr)


# --------------------------------------------------------------------------
# Lattice Greeks on a digital: measured, not claimed.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("greek", ["delta", "gamma", "theta"])
def test_lattice_greeks_are_first_order_on_leisen_reimer(greek: str) -> None:
    """Evidence class: CONVERGENCE_ORDER.

    The order-2 price does **not** carry over to the Greeks, exactly as on a
    vanilla and for the same reason: `lattice_delta_gamma_theta` reads levels 1
    and 2 of the lattice, which is an `O(dt)` substitution no scheme fixes.
    Measured over `n` in (25 ... 801) at the reference ATM call: delta 1.003
    (residual 0.002), gamma 1.002 (0.001), theta 1.005 (0.002). Off the money
    (otm_9m_div) the same three are 1.004, 1.006 and 0.999.

    Rho is second order (1.895 and 1.902) because it is bump-and-revalue on a
    lattice whose geometry does not move with `r`, so it inherits the price's
    order rather than the estimators'. Vega is flat (order -0.011, error stuck
    at -1.08e-03) because `VEGA_BUMP`'s own `O(h**2)` bias binds -- the same
    finding Slice 3 recorded for the vanilla Leisen-Reimer vega.
    """
    triple = _triple("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.20)
    option, model, market = triple
    exact = getattr(analytic_greeks(option, model, market), greek)

    errors = [
        abs(
            getattr(
                greeks(
                    option,
                    model,
                    market,
                    method="tree",
                    cfg=TreeConfig(n_steps=n, scheme="leisen-reimer"),
                ),
                greek,
            )
            - exact
        )
        for n in ODD_LEVELS
    ]
    fit = fit_convergence_order([1.0 / n for n in ODD_LEVELS], errors)
    assert abs(fit.order - 1.0) <= 0.15, (greek, fit.order, errors)
    assert fit.residual < 0.02, fit.residual


def test_lattice_rho_is_second_order_and_vega_is_flat() -> None:
    """Evidence class: CONVERGENCE_ORDER for rho, NEGATIVE_FINDING for vega.

    Rho: measured 1.895, residual 0.071 -- bumping `r` leaves `u`, `d` and the
    node positions alone, so the two solves share their discretisation error
    and the difference inherits the order-2 price.

    Vega: measured order -0.011 with the error pinned at -1.03e-03 to
    -1.08e-03 across a 32x refinement. Bumping `sigma` *does* move the
    Leisen-Reimer lattice, and at `h = VEGA_BUMP = 1e-2` the central
    difference's own `O(h**2)` bias is larger than anything the refinement
    removes. This is not a digital-specific finding; Slice 3 measured the same
    floor on the vanilla.
    """
    triple = _triple("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.20)
    option, model, market = triple
    exact = analytic_greeks(option, model, market)

    def error(n: int, greek: str) -> float:
        value = getattr(
            greeks(
                option,
                model,
                market,
                method="tree",
                cfg=TreeConfig(n_steps=n, scheme="leisen-reimer"),
            ),
            greek,
        )
        return abs(value - getattr(exact, greek))

    rho = [error(n, "rho") for n in ODD_LEVELS]
    rho_fit = fit_convergence_order([1.0 / n for n in ODD_LEVELS], rho)
    assert abs(rho_fit.order - 1.9) <= 0.2, (rho_fit.order, rho)

    vega = [error(n, "vega") for n in ODD_LEVELS]
    vega_fit = fit_convergence_order([1.0 / n for n in ODD_LEVELS], vega)
    assert abs(vega_fit.order) < 0.1, (vega_fit.order, vega)
    assert max(vega) / min(vega) < 1.2, vega


# --------------------------------------------------------------------------
# Degenerate limits and validation.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["crr", "leisen-reimer"])
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("spot", [95.0, 105.0])
def test_degenerate_limits_match_the_analytic_engine(
    scheme: str, kind: str, spot: float
) -> None:
    """Evidence class: CLOSED_FORM. `T = 0` and `sigma = 0` have no lattice in them."""
    at_expiry = _triple(kind, spot, 100.0, 0.0, 0.05, 0.01, 0.20, 3.0)
    zero_vol = _triple(kind, spot, 100.0, 1.0, 0.05, 0.02, 0.0, 3.0)
    for triple in (at_expiry, zero_vol):
        tree = price(
            *triple, method="tree", cfg=TreeConfig(n_steps=51, scheme=scheme)  # type: ignore[arg-type]
        ).value
        assert tree == price(*triple).value


def test_a_terminal_node_on_the_strike_pays_nothing() -> None:
    """Evidence class: EXACT_IDENTITY on the convention, plus a NEGATIVE_FINDING.

    `digital_payoff` uses strict inequalities, so a point exactly on the strike
    pays nothing in either the call or the put. Asserted directly below.

    **The slice expected that to bite at the money on a CRR lattice with an
    even `n`**, where `u d = 1` makes the centre terminal node algebraically
    equal to the spot, and therefore to the strike. It does not, and the reason
    is round-off: `crr_spot_level` evaluates `S * u**j * d**(n-j)`, and at
    `S = K = 100` that lands at 100 +- 1.4e-13 rather than at 100. Which side
    it falls on is decided by the last bits and flips with `n` -- measured
    **below** the strike at `n = 50` (-1.4e-13) and **above** it at `n = 100`
    and `n = 200` (+1.4e-13, +1.3e-13). So the node is always counted, exactly
    once, and `call + put = cash e^{-rT}` holds to round-off on even `n` as
    well as odd.

    That is worth pinning for two reasons: the strict convention is never
    exercised by this lattice, so a test that assumed it was would be testing
    nothing; and the *side* the centre node falls on is a round-off coin flip
    that contributes to the at-the-money even-`n` error, which is one more
    reason the odd-`n` sequence is the one the order fits use.
    """
    from qpl.instruments.payoffs import digital_payoff

    assert digital_payoff(100.0, 100.0, 5.0, "call") == 0.0
    assert digital_payoff(100.0, 100.0, 5.0, "put") == 0.0

    offsets = {}
    for n in (50, 100, 200):
        lattice = lattice_parameters(
            scheme="crr",
            spot=100.0,
            strike=100.0,
            sigma=0.20,
            expiry=1.0,
            rate=0.05,
            dividend_yield=0.0,
            n_steps=n,
        )
        spots = crr_spot_level(spot=100.0, up=lattice.up, down=lattice.down, level=n)
        offsets[n] = spots[n // 2] - 100.0

    assert all(0.0 < abs(offset) < 1e-11 for offset in offsets.values()), offsets
    assert offsets[50] < 0.0 < offsets[100], offsets

    call = _triple("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.20)
    put = _triple("put", 100.0, 100.0, 1.0, 0.05, 0.0, 0.20)
    riskless = math.exp(-0.05)
    for n in (100, 101):
        cfg = TreeConfig(n_steps=n)
        total = (
            price(*call, method="tree", cfg=cfg).value
            + price(*put, method="tree", cfg=cfg).value
        )
        # 1e-14 rather than 1e-15: the even-`n` sum accumulates 2.2e-15 of
        # round-off over 100 discounted roll-back steps.
        assert total == pytest.approx(riskless, abs=1e-14), n


def test_config_validation_matches_the_vanilla_engine() -> None:
    triple = _triple("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.20)

    with pytest.raises(InvalidInputError, match="n_steps must be >= 1"):
        price(*triple, method="tree", cfg=TreeConfig(n_steps=0))
    with pytest.raises(InvalidInputError, match="n_steps must be >= 2"):
        greeks(*triple, method="tree", cfg=TreeConfig(n_steps=1))
    with pytest.raises(InvalidInputError, match="odd n_steps"):
        price(*triple, method="tree", cfg=TreeConfig(n_steps=100, scheme="leisen-reimer"))
    with pytest.raises(InvalidInputError, match="sigma must be > 0"):
        greeks(
            *_triple("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.0),
            method="tree",
            cfg=TreeConfig(n_steps=51),
        )


def test_evidence_classes_used_here_exist() -> None:
    for name in ("CONVERGENCE_ORDER", "NEGATIVE_FINDING", "EXACT_IDENTITY", "CLOSED_FORM"):
        assert isinstance(getattr(EvidenceClass, name), EvidenceClass)
