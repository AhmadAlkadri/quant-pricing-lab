"""Measured convergence of the American CRR tree.

Slice item (d): the American value converges as `n` grows with a fitted order
in [0.8, 1.2] against a fine-`n` reference from the same engine, with the
odd/even behaviour measured and reported truthfully rather than assumed.

What is different from the European study in `tests/test_tree_convergence.py`,
and why this file exists separately:

- There is no closed form for an American option, so the reference is this
  engine at `n = 8001` (0.16 s). A same-engine reference is not free: it
  carries its own error, and that error is *subtracted from every measured
  error*, which tilts the fitted constants. The size of the tilt is quantified
  below rather than ignored.
- The odd/even bracketing survives -- odd `n` above, even `n` below -- but the
  two constants are further apart than in the European case and they drift,
  because the American error has a second source: the exercise boundary is
  resolved only to the node spacing, and that error does not oscillate with
  the parity of `n`.
- **Richardson extrapolation does not work here.** For the European call it
  lifts the measured order from 1.00 to 1.96. For the American put it reduces
  the error by one to two orders of magnitude and then stops, at a fitted
  order of 0.30 (odd) and 0.64 (even) with log-space residuals of 0.27 and
  0.36 -- the fit's way of saying the extrapolated sequence is not a power law
  at all. See `test_richardson_extrapolation_does_not_restore_order_two`.

A fitted slope is not a theorem. Every number quoted here is a measurement of
this implementation on these refinement sequences, reproducible by running
this file. The tree is Cox, Ross and Rubinstein (1979), Journal of Financial
Economics 7, 229-263; the order-1 behaviour and the odd/even oscillation of
the European CRR tree are Leisen and Reimer (1996), Applied Mathematical
Finance 3(4), 319-346. Neither paper is the source of any number below.
Derivation and the full tables: `docs/notes/american_exercise_on_trees.md`.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import pairwise

import pytest

from qpl.engines.tree import TreeConfig
from qpl.instruments.options import AmericanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price
from qpl.validation import fit_convergence_order

ODD_LEVELS = (25, 51, 101, 201, 401, 801)
EVEN_LEVELS = (26, 50, 100, 200, 400, 800)

REFERENCE_N_STEPS = 8001
"""Step count of the same-engine reference.

Odd on purpose, so that the reference sits on the same side of the true value
as the odd subsequence and the odd constants are the less contaminated of the
two. Costs 0.16 s; the whole file runs in about 1.3 s.
"""

GREEK_REFERENCE_N_STEPS = 2001
"""Greeks reference. Smaller than the price reference because the Greek
refinement sequence stops at `n = 401`, and because `greeks` prices the tree
three times (value, sigma bump, rate bump)."""

# (id, kind, spot, strike, expiry, rate, dividend, sigma)
_POINTS = (
    ("atm_put", "put", 100.0, 100.0, 1.0, 0.05, 0.00, 0.20),
    ("atm_call_with_yield", "call", 100.0, 100.0, 1.0, 0.05, 0.06, 0.20),
)
_IDS = [row[0] for row in _POINTS]
_ARGS = [row[1:] for row in _POINTS]
_ARGNAMES = ("kind", "spot", "strike", "expiry", "rate", "div", "sigma")


@lru_cache(maxsize=None)
def _price(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
    n_steps: int,
) -> float:
    """Cached because the engine is deterministic and several tests share the
    `n = 8001` reference; without it the file spends most of its time
    recomputing the same lattice."""
    return price(
        AmericanOption(kind=kind, strike=strike, expiry=expiry),  # type: ignore[arg-type]
        BlackScholesModel(sigma=sigma),
        Market(
            spot=spot,
            rate_curve=FlatRateCurve(rate),
            dividend_curve=FlatDividendCurve(div),
        ),
        method="tree",
        cfg=TreeConfig(n_steps=n_steps),
    ).value


def _signed_errors(args: tuple, levels: tuple[int, ...]) -> tuple[float, list[float]]:
    reference = _price(*args, n_steps=REFERENCE_N_STEPS)
    return reference, [_price(*args, n_steps=n) - reference for n in levels]


# --------------------------------------------------------------------------
# (d) Order on each parity subsequence
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("parity", "levels"), [("odd", ODD_LEVELS), ("even", EVEN_LEVELS)])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_american_tree_is_first_order_on_each_parity_subsequence(
    parity: str,
    levels: tuple[int, ...],
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: CONVERGENCE_ORDER.

    Fitted orders against the `n = 8001` reference, with log-space RMS
    residuals:

        point                 odd                even
        atm_put               1.0141 (0.0174)    0.9722 (0.0242)
        atm_call_with_yield   1.0232 (0.0173)    0.9674 (0.0213)

    all inside the [0.8, 1.2] band this repository uses for order-1 claims.

    The residuals are an order of magnitude larger than the European tree's
    (0.0005 / 0.0007 in `tests/test_tree_convergence.py`) and that is not
    noise: the reference carries its own error, which is subtracted from every
    point in the fit and bends the tail. Quantified for the ATM put by
    re-running offline against a much better reference -- the average of
    `n = 64000` and `n = 64001`, 6.090376463020103, which brackets and so
    largely cancels the oscillation:

        reference used      odd order (resid)   even order (resid)
        n = 8001            1.0141 (0.0174)     0.9722 (0.0242)
        64000/64001 avg     0.9872 (0.0034)     1.0168 (0.0091)

    The `n = 8001` value is 1.80e-04 above that better reference, which is
    exactly the size of the tilt. Both readings are order 1; the cleaner one
    is the second, and the test uses the first because the slice asks for a
    reference from this same engine.
    """
    args = (kind, spot, strike, expiry, rate, div, sigma)
    _, signed = _signed_errors(args, levels)
    fit = fit_convergence_order([1.0 / n for n in levels], [abs(e) for e in signed])

    assert 0.8 <= fit.order <= 1.2, (parity, fit.order)
    assert fit.residual < 0.05, (parity, fit.residual)
    assert fit.n_points == len(levels)


@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_odd_and_even_subsequences_bracket_the_reference(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: CONVERGENCE_ORDER (the sign structure of the error).

    Measured: the American price is above the fine-`n` reference at every odd
    `n` tested and below it at every even `n`, so the bracketing the European
    tree shows at the money survives early exercise. The mechanism is the same
    -- at the money an even `n` puts a terminal node exactly on the strike and
    an odd `n` straddles it -- and it is visible here for the same reason
    (`S = K` at both points).

    The scaled constants `n |err|` do *not* settle the way the European ones
    do (1.7529 / 1.9994 to four figures). Measured:

        atm_put   odd  1.3907 1.3973 1.4151 1.3999 1.3763 1.3134
                  even 0.8222 0.8414 0.8202 0.8347 0.8602 0.9238
        call+q    odd  1.8741 1.8846 1.8728 1.8583 1.8127 1.7169
                  even 1.2864 1.2782 1.2921 1.3077 1.3538 1.4496

    Both parities drift toward each other, which is the reference's own error
    leaking in: against the 64000/64001 reference the ATM put's constants run
    1.395 -> 1.458 (odd) and 0.818 -> 0.780 (even) instead, i.e. they drift
    apart slightly rather than together. Either way they are two different
    constants that are only approximately constant -- unlike the European
    case, the American error is not a clean `C/n` within a parity, because the
    exercise boundary is resolved only to the node spacing and that error has
    no parity structure of its own.

    So: no numeric constants are pinned here, only the sign structure, which
    is the part that is robust.
    """
    args = (kind, spot, strike, expiry, rate, div, sigma)
    _, odd = _signed_errors(args, ODD_LEVELS)
    _, even = _signed_errors(args, EVEN_LEVELS)

    assert all(e > 0.0 for e in odd), odd
    assert all(e < 0.0 for e in even), even
    assert max(even) < 0.0 < min(odd)

    # Two distinct constants, not one: the odd and even scaled errors do not
    # overlap anywhere on these sequences.
    odd_scaled = [n * abs(e) for n, e in zip(ODD_LEVELS, odd, strict=True)]
    even_scaled = [n * abs(e) for n, e in zip(EVEN_LEVELS, even, strict=True)]
    assert min(odd_scaled) > max(even_scaled)


# --------------------------------------------------------------------------
# Richardson extrapolation: the negative finding
# --------------------------------------------------------------------------


@pytest.mark.parametrize("parity", ["odd", "even"])
def test_richardson_extrapolation_does_not_restore_order_two(parity: str) -> None:
    """Evidence class: NEGATIVE_FINDING.

    Within one parity the European tree's error is `C/n + D/n**2` with a
    constant `C` that no longer oscillates, so combining two levels,

        V_ext = (n2 V(n2) - n1 V(n1)) / (n2 - n1),

    cancels the leading term and lifts the measured order from 1.00 to 1.959
    (`tests/test_tree_convergence.py`). The same construction applied to the
    American put does not do that.

    Measured for the ATM put against the `n = 8001` reference, extrapolating
    consecutive pairs (25,51), (51,101), ..., (401,801) and (26,50), ...,
    (400,800):

        parity  order   log-resid   extrapolated errors
        odd     0.2960  0.2712      2.53e-04 3.55e-04 1.52e-04 1.18e-04 1.57e-04
        even    0.6447  0.3562      8.00e-04 4.24e-04 1.45e-04 1.27e-04 1.59e-04

    Two things are true at once and both are worth stating:

    1. The extrapolation *does* help in magnitude. Against the raw errors at
       the same `n1` (5.56e-02 ... 3.43e-03 odd) it is a factor of 220 at the
       coarsest pair and 22 at the finest.
    2. It does not converge. The extrapolated sequence flattens at 1.2e-04 to
       3.6e-04 and stops improving, and a log-log fit of a flat sequence
       returns a slope near zero with a large residual -- which is what the
       numbers above are.

    Where the floor comes from, checked offline so that the claim is not
    circular: the `n = 8001` reference is itself 1.80e-04 away from the
    64000/64001 average, so a flat floor at ~1.5e-04 is *at* the reference's
    own accuracy and this test cannot see past it. Repeating the measurement
    against that better reference does not rescue the order either -- odd
    1.1613 with residual 0.7097, even 1.3467 with residual 0.6349, extrapolated
    errors 4.33e-04, 5.35e-04, 2.83e-05, 6.19e-05, 2.28e-05: non-monotone, far
    from the clean order-2 sequence the European call gives.

    The reason is structural rather than numerical. Richardson assumes an
    asymptotic expansion in `1/n` with fixed coefficients. The American error
    does not have one: part of it comes from the exercise boundary being
    resolved to the nearest node, and that part moves in steps as `n` changes
    which node is nearest, so there is no smooth `C` to cancel. This is the
    expected failure mode, and it is the reason no Richardson-extrapolated
    American value appears anywhere in this package's public surface.
    """
    levels = ODD_LEVELS if parity == "odd" else EVEN_LEVELS
    args = _ARGS[0]
    reference, raw = _signed_errors(args, levels)
    values = [e + reference for e in raw]

    h, extrapolated = [], []
    for (n1, v1), (n2, v2) in pairwise(list(zip(levels, values, strict=True))):
        h.append(1.0 / n1)
        extrapolated.append(abs((n2 * v2 - n1 * v1) / (n2 - n1) - reference))

    fit = fit_convergence_order(h, extrapolated)

    # (1) It helps: every extrapolated error beats the raw error at the same n1
    # by at least a factor of ten.
    for n1, ext, rawerr in zip(levels[:-1], extrapolated, raw[:-1], strict=True):
        assert ext < abs(rawerr) / 10.0, (n1, ext, rawerr)

    # (2) It does not reach order 2, and the fit says the residual sequence is
    # not a power law.
    assert fit.order < 1.0, (parity, fit.order)
    assert fit.residual > 0.15, (parity, fit.residual)

    # (3) The flat floor is at the reference's own accuracy, not below it.
    assert max(extrapolated[-3:]) < 5e-4
    assert min(extrapolated) > 1e-5


# --------------------------------------------------------------------------
# Lattice Greeks for American payoffs
# --------------------------------------------------------------------------

_GREEK_ODD = (25, 51, 101, 201, 401)
_GREEK_EVEN = (26, 50, 100, 200, 400)


@pytest.mark.parametrize(("parity", "levels"), [("odd", _GREEK_ODD), ("even", _GREEK_EVEN)])
@pytest.mark.parametrize("name", ["delta", "gamma", "theta"])
def test_american_lattice_greeks_converge_at_order_one(
    parity: str, levels: tuple[int, ...], name: str
) -> None:
    """Evidence class: CONVERGENCE_ORDER.

    Delta, gamma and theta are read off nodes one and two steps from the root,
    so they inherit the price's `O(1/n)` rather than improving on it, exactly
    as in the European case. Measured for the ATM American put against the
    same engine at `n = 2001`:

        greek   odd order (resid)   even order (resid)
        delta   1.1644 (0.0265)     1.0743 (0.0043)
        gamma   1.1845 (0.0255)     1.0787 (0.0104)
        theta   1.2586 (0.0478)     1.0759 (0.0129)

    The odd orders sit above 1.2, so the [0.8, 1.2] band used for the price
    does not fit and is not forced to: the band below is [0.9, 1.4]. The
    excess is the same reference contamination as for the price, in the
    direction that steepens the fit, plus the exercise boundary's own
    node-resolution error. Errors at `n = 401`: 4.2e-05 (delta), 9.6e-06
    (gamma), 9.6e-04 (theta).
    """
    kind, spot, strike, expiry, rate, div, sigma = _ARGS[0]
    option = AmericanOption(kind=kind, strike=strike, expiry=expiry)  # type: ignore[arg-type]
    model = BlackScholesModel(sigma=sigma)
    market = Market(
        spot=spot,
        rate_curve=FlatRateCurve(rate),
        dividend_curve=FlatDividendCurve(div),
    )

    def _greek(n: int) -> float:
        return getattr(
            greeks(option, model, market, method="tree", cfg=TreeConfig(n_steps=n)), name
        )

    reference = _greek(GREEK_REFERENCE_N_STEPS)
    errs = [abs(_greek(n) - reference) for n in levels]
    fit = fit_convergence_order([1.0 / n for n in levels], errs)

    assert 0.9 <= fit.order <= 1.4, (name, parity, fit.order)
    assert fit.residual < 0.08, (name, parity, fit.residual)
