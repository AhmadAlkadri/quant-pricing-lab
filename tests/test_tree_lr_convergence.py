"""Measured convergence of the Leisen-Reimer tree, and its margin over CRR.

Two claims, both CONVERGENCE_ORDER, plus one comparison table:

(b) The European price converges at **order 2** in `1 / n` with no odd/even
    oscillation -- at the money and at the two off-the-money points where
    `tests/oracle/test_tree_vs_quantlib.py` measured the CRR error constant to
    be erratic in *both* engines.
(f) The American put converges at a measured order of **1.06**, not 2. The
    exercise boundary is resolved only to the node spacing, and that error
    does not care how well the terminal distribution is matched. Richardson
    extrapolation does not rescue it, exactly as it failed to for CRR in
    Slice 2.

Refinement grid: `n` in {25, 51, 101, 201, 401, 801}, all odd, because the
Leisen-Reimer construction has no even-`n` form. `h = 1 / n`. The whole file
runs in about 0.4 s.

References
----------
- European errors are measured against this repository's closed form, which
  `tests/oracle/test_tree_vs_quantlib.py` has checked against QuantLib's
  `AnalyticEuropeanEngine` to 1.1e-14.
- American errors are measured against `AMERICAN_BRACKETED_LIMIT` below, an
  in-repo value from the *CRR* engine, so it is independent of the scheme
  under test here.

A fitted slope is not a theorem: every number below is a measurement of this
implementation on this grid, reproducible by running this file. The scheme is
Leisen and Reimer (1996), "Binomial models for option valuation -- examining
and improving convergence", Applied Mathematical Finance 3(4), 319-346; no
number here is quoted from that paper. Derivation and the full tables:
`docs/notes/leisen_reimer.md`.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import pairwise

import pytest

from qpl.engines.tree import TreeConfig
from qpl.instruments.options import AmericanOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price
from qpl.validation import fit_convergence_order

LR = "leisen-reimer"
CRR = "crr"

ODD_LEVELS: tuple[int, ...] = (25, 51, 101, 201, 401, 801)
"""Refinement grid. Odd throughout, which the Leisen-Reimer scheme requires;
it is also the grid `tests/test_tree_convergence.py` fits CRR's odd branch on,
so the two studies are directly comparable."""

H = [1.0 / n for n in ODD_LEVELS]

AMERICAN_BRACKETED_LIMIT = 6.090376463020103
"""The American put value at `S = K = 100, r = 5%, q = 0, sigma = 20%, T = 1`.

Derived in-repo, not published: it is the average of the CRR engine at
`n = 64000` and `n = 64001`, which bracket the limit from below and above (the
odd/even bracketing measured in `tests/test_tree_american_convergence.py`), so
the average is accurate to well inside the 3.7e-04 that the coarsest error
measured against it here has to spare. Recorded in
`docs/notes/american_exercise_on_trees.md` and used here because it comes from
the *other* scheme and so is not a same-engine reference for Leisen-Reimer.
"""

# (id, kind, spot, strike, expiry, rate, dividend, sigma)
_EUROPEAN_POINTS = (
    ("atm_1y", 100.0, 100.0, 1.0, 0.05, 0.00, 0.20),
    ("otm_2y_div", 100.0, 110.0, 2.0, 0.03, 0.01, 0.25),
    ("itm_1y_div", 120.0, 90.0, 1.0, 0.03, 0.05, 0.35),
)
_EU_IDS = [row[0] for row in _EUROPEAN_POINTS]
_EU_ARGS = [row[1:] for row in _EUROPEAN_POINTS]
_EU_ARGNAMES = ("spot", "strike", "expiry", "rate", "div", "sigma")

AMERICAN_SPEC = (100.0, 100.0, 1.0, 0.05, 0.00, 0.20)


def _market(spot: float, r: float, q: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(r),
        dividend_curve=FlatDividendCurve(q),
    )


@lru_cache(maxsize=None)
def _european(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
    n_steps: int,
    scheme: str,
) -> float:
    """Cached: the engine is deterministic and several tests share these
    prices, most of all the CRR comparison legs."""
    return price(
        EuropeanOption(kind=kind, strike=strike, expiry=expiry),  # type: ignore[arg-type]
        BlackScholesModel(sigma=sigma),
        _market(spot, rate, div),
        method="tree",
        cfg=TreeConfig(n_steps=n_steps, scheme=scheme),  # type: ignore[arg-type]
    ).value


@lru_cache(maxsize=None)
def _closed_form(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> float:
    return price(
        EuropeanOption(kind=kind, strike=strike, expiry=expiry),  # type: ignore[arg-type]
        BlackScholesModel(sigma=sigma),
        _market(spot, rate, div),
        method="analytic",
    ).value


@lru_cache(maxsize=None)
def _american(n_steps: int, scheme: str) -> float:
    spot, strike, expiry, rate, div, sigma = AMERICAN_SPEC
    return price(
        AmericanOption(kind="put", strike=strike, expiry=expiry),
        BlackScholesModel(sigma=sigma),
        _market(spot, rate, div),
        method="tree",
        cfg=TreeConfig(n_steps=n_steps, scheme=scheme),  # type: ignore[arg-type]
    ).value


def _european_signed_errors(
    kind: str, args: tuple[float, ...], scheme: str
) -> list[float]:
    exact = _closed_form(kind, *args)
    return [_european(kind, *args, n_steps=n, scheme=scheme) - exact for n in ODD_LEVELS]


# --------------------------------------------------------------------------
# (b) European: measured order 2, at and away from the money
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_EU_ARGNAMES, _EU_ARGS, ids=_EU_IDS)
def test_leisen_reimer_european_is_second_order(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: CONVERGENCE_ORDER.

    Fitted order of `|tree - Black-Scholes|` against `h = 1/n` over
    `n` in {25, 51, 101, 201, 401, 801}, identical for the call and the put at
    each point (parity holds on the lattice to 1e-11, so the two error
    sequences coincide to that):

        point       order   log-RMS residual   |err| at n=801
        atm_1y      1.9840  0.0087             5.52e-07
        otm_2y_div  1.9842  0.0086             1.06e-06
        itm_1y_div  1.9709  0.0154             2.26e-07

    The band is [1.8, 2.2], as the slice specified. The measured orders sit
    just *below* 2 rather than straddling it, consistently, which is worth
    stating rather than rounding away: the Peizer-Pratt inversion matches the
    binomial tail to high but finite order, so the leading `1/n**2` term is
    accompanied by a slowly decaying correction, and over a sixfold refinement
    that drags the fitted slope down by about 0.02-0.03. The residual bound of
    0.02 (0.05 off the money) is set by that same correction, not by noise; a
    clean power law would fit with a residual near 1e-03, and the CRR fits on
    this grid do (0.0005).

    The off-the-money points are exactly the two where
    `tests/oracle/test_tree_vs_quantlib.py` measured the CRR error constant to
    be erratic in both `qpl` and QuantLib -- fitted orders 1.21 to 1.48 with
    log-space residuals of 0.37 to 1.52, non-monotone errors. Order 2 survives
    there, and the residual stays two orders of magnitude below CRR's. That is
    the central claim of this slice.
    """
    args = (spot, strike, expiry, rate, div, sigma)
    errs = [abs(e) for e in _european_signed_errors(kind, args, LR)]
    fit = fit_convergence_order(H, errs)

    assert 1.8 <= fit.order <= 2.2, fit.order
    assert fit.residual < 0.05, fit.residual
    assert fit.n_points == len(ODD_LEVELS)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_EU_ARGNAMES, _EU_ARGS, ids=_EU_IDS)
def test_the_error_does_not_oscillate_with_the_parity_of_n(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: CONVERGENCE_ORDER (the sign structure of the error).

    The CRR error at the money changes sign with every increment of `n`: on
    the 21 consecutive counts `n = 101 .. 121` it alternates
    `+1.74e-02, -1.96e-02, +1.70e-02, ...` -- 20 sign changes in 21 values.
    The Leisen-Reimer error on the 11 odd counts in the same range is
    one-signed and monotone: `-3.42e-05, -3.29e-05, ..., -2.39e-05`, a total
    variation of 36% of its own mean across the whole range.

    Only odd counts are available for comparison, since the scheme rejects
    even ones, so this is not the claim that Leisen-Reimer has "fixed" the
    even branch -- there is no even branch. It is the claim that within the
    sequence the scheme *does* define, consecutive refinements move the error
    smoothly instead of flipping it, which is what makes a single
    `C / n**2` fit meaningful and what the CRR study could only get by
    splitting the grid by parity.
    """
    args = (spot, strike, expiry, rate, div, sigma)
    signed = _european_signed_errors(kind, args, LR)

    # One sign across the whole refinement grid.
    assert all(e > 0.0 for e in signed) or all(e < 0.0 for e in signed), signed
    # And monotone in magnitude: every refinement helps, none flips.
    magnitudes = [abs(e) for e in signed]
    assert all(a > b for a, b in pairwise(magnitudes)), magnitudes

    # Non-vacuity: CRR on the same odd grid is one-signed too (that is the
    # point of splitting by parity), so the contrast that matters is on
    # CONSECUTIVE counts. Check it at the one point the CRR study used.
    if (spot, strike) == (100.0, 100.0):
        crr_consecutive = [
            _european(kind, *args, n_steps=n, scheme=CRR) - _closed_form(kind, *args)
            for n in range(101, 122)
        ]
        flips = sum(1 for a, b in pairwise(crr_consecutive) if a * b < 0.0)
        assert flips == 20, flips

        lr_consecutive = [
            _european(kind, *args, n_steps=n, scheme=LR) - _closed_form(kind, *args)
            for n in range(101, 122, 2)
        ]
        assert all(e < 0.0 for e in lr_consecutive), lr_consecutive


# --------------------------------------------------------------------------
# The comparison table: how much better, and where
# --------------------------------------------------------------------------

# Ratios `|CRR error| / |LR error|` at the same `n`, measured on the three
# European points above (identical for call and put):
#
#     point         n = 101   n = 801
#     atm_1y            507      3966
#     otm_2y_div         35      2673
#     itm_1y_div       1137      5229
#
# The asserted floors are a factor of about 3.5 below the worst measurement at
# each column. The worst cell is `otm_2y_div` at n = 101, and it is low for a
# reason that is CRR's rather than Leisen-Reimer's: off the money the CRR error
# constant is erratic (pinned in `tests/oracle/test_tree_vs_quantlib.py`), and
# at that particular `n` it happens to sit near a sign change, so CRR is
# accidentally good there. That is precisely why the floor is set from the
# worst cell and not from the at-the-money one.
_MIN_RATIO_AT_101 = 10.0
_MIN_RATIO_AT_801 = 500.0


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_EU_ARGNAMES, _EU_ARGS, ids=_EU_IDS)
def test_leisen_reimer_beats_crr_by_orders_of_magnitude(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: CONVERGENCE_ORDER (the practical consequence of it).

    An order is a statement about a limit; this is the statement about the
    numbers a caller actually gets. See the table above the test for the
    measured ratios and for why the floors are what they are.
    """
    args = (spot, strike, expiry, rate, div, sigma)
    exact = _closed_form(kind, *args)

    table: dict[int, tuple[float, float, float]] = {}
    for n in (101, 801):
        lr_err = abs(_european(kind, *args, n_steps=n, scheme=LR) - exact)
        crr_err = abs(_european(kind, *args, n_steps=n, scheme=CRR) - exact)
        assert lr_err > 0.0
        table[n] = (lr_err, crr_err, crr_err / lr_err)

    assert table[101][2] >= _MIN_RATIO_AT_101, table
    assert table[801][2] >= _MIN_RATIO_AT_801, table
    # The ratio grows with n, which is the order gap showing up as a ratio:
    # one scheme's error falls like 1/n and the other's like 1/n**2.
    assert table[801][2] > table[101][2], table


# --------------------------------------------------------------------------
# (f) American: measured order 1, and Richardson still does not help
# --------------------------------------------------------------------------


def _american_signed_errors(scheme: str) -> list[float]:
    return [_american(n, scheme) - AMERICAN_BRACKETED_LIMIT for n in ODD_LEVELS]


def test_american_put_on_the_lr_lattice_is_first_order() -> None:
    """Evidence class: CONVERGENCE_ORDER. **Not order 2, and not expected to be.**

    Measured against `AMERICAN_BRACKETED_LIMIT` over the odd grid:

        scheme          order   residual   signed error at n=801
        leisen-reimer   1.0641  0.0176     -3.66e-04
        crr             0.9872  0.0034     +1.82e-03

    Order 1, as the slice anticipated. The European order-2 argument does not
    transfer: it is about placing the strike where the binomial and normal
    terminal tails agree, and the American value's dominant discretisation
    error is not in the terminal distribution at all but in the location of
    the early-exercise boundary, which the lattice resolves only to its node
    spacing `O(1/sqrt(n))` in log-spot per step. That error is `O(1/n)` in the
    value and is indifferent to how the terminal grid was chosen.

    Two things Leisen-Reimer does buy here, both measured:

    - the **constant** is about 4 to 5 times better at every `n` on this grid
      (ratios 3.79, 4.05, 4.50, 4.55, 4.79, 4.97 from n=25 to n=801);
    - the error has the **opposite sign** to CRR's -- Leisen-Reimer approaches
      the limit from below at every `n` tested, CRR from above -- so the two
      schemes bracket the value at a shared `n` in the way CRR's own odd and
      even branches do. That is a cheap error bar and it is asserted below.

    The fitted order comes out slightly above 1 (1.0641, residual 0.0176)
    rather than exactly 1, because the boundary error and the smaller
    order-2 terminal error are both present and the second is shrinking
    faster. The band [0.85, 1.25] admits that without admitting order 2.
    """
    lr_signed = _american_signed_errors(LR)
    crr_signed = _american_signed_errors(CRR)

    lr_fit = fit_convergence_order(H, [abs(e) for e in lr_signed])
    crr_fit = fit_convergence_order(H, [abs(e) for e in crr_signed])

    assert 0.85 <= lr_fit.order <= 1.25, lr_fit.order
    assert lr_fit.residual < 0.05, lr_fit.residual
    assert lr_fit.order < 1.5, lr_fit.order
    assert 0.85 <= crr_fit.order <= 1.25, crr_fit.order

    # Opposite sides of the limit, at every n.
    assert all(e < 0.0 for e in lr_signed), lr_signed
    assert all(e > 0.0 for e in crr_signed), crr_signed

    # And a better constant, at every n.
    ratios = [
        abs(c) / abs(lr) for c, lr in zip(crr_signed, lr_signed, strict=True)
    ]
    assert min(ratios) >= 3.0, ratios


def test_richardson_extrapolation_still_does_not_restore_order_two() -> None:
    """Evidence class: NEGATIVE_FINDING.

    Slice 2 measured that Richardson extrapolation of consecutive odd `n`
    lifts the *European* CRR order from 1.00 to 1.96 but leaves the *American*
    one at 0.30, with a log-space residual of 0.27 -- the fit's way of saying
    the extrapolated sequence is not a power law. Changing the lattice does
    not change that. On the Leisen-Reimer lattice the same extrapolation,

        V_ext = (n2 V(n2) - n1 V(n1)) / (n2 - n1),

    over the pairs (25,51), (51,101), ..., (401,801) gives a fitted order of
    **1.34** with a log-space residual of **0.72**, and extrapolated errors of
    7.97e-04, 5.68e-04, 2.87e-05, 6.70e-05, 2.23e-05 -- non-monotone, which is
    where most of that residual comes from.

    So the cancellation does happen (the errors are one to two orders of
    magnitude below the raw ones) but what is left is not a cleaner power law:
    the boundary error is not `C/n` with a constant, it is `C(n)/n` with a
    constant that jumps as the node grid steps past the true boundary. The
    assertion is that the residual stays large, so this finding fails loudly
    if it ever quietly becomes true.
    """
    pair_h: list[float] = []
    pair_err: list[float] = []
    for n1, n2 in pairwise(ODD_LEVELS):
        v1 = _american(n1, LR)
        v2 = _american(n2, LR)
        extrapolated = (n2 * v2 - n1 * v1) / (n2 - n1)
        pair_h.append(1.0 / n1)
        pair_err.append(abs(extrapolated - AMERICAN_BRACKETED_LIMIT))

    fit = fit_convergence_order(pair_h, pair_err)

    # It does cancel something: the extrapolated error is far below the raw.
    raw = [abs(e) for e in _american_signed_errors(LR)]
    assert pair_err[0] < raw[0] / 10.0, (pair_err[0], raw[0])

    # But it is not a power law, and in particular not a clean order 2.
    assert fit.residual > 0.2, fit.residual
    assert not (1.8 <= fit.order <= 2.2), fit.order
    # Non-monotone: at least one pair is worse than the one before it.
    assert any(b > a for a, b in pairwise(pair_err)), pair_err
