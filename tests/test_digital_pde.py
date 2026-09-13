"""A jump at the strike on a finite-difference grid: pathology and remedies.

Reference point throughout: `S = K = 100, r = 5%, q = 0, sigma = 20%, T = 1`,
cash-or-nothing call, `cash = 1`, refined on `n_s = n_t = n` for `n` in
(100, 200, 400, 800) unless a test says otherwise. Every number below was
measured in this repository.

The standard refinement path (`n_s = n_t`)
------------------------------------------

| configuration                            | price  | delta  | gamma  |
|------------------------------------------|--------|--------|--------|
| plain CN, unaligned, no projection       | 1.0012 | 0.8632 | 1.0153 |
| CN, unaligned, `cell_average`            | 2.0153 | 2.0059 | 1.9979 |
| Rannacher, unaligned, `cell_average`     | 2.0154 | 2.0061 | 1.9980 |
| plain CN, midpoint-aligned               | 1.9291 | 2.0353 | 2.0313 |
| Rannacher, midpoint-aligned              | 1.9248 | 2.0360 | 2.0315 |

Row 1 is the NEGATIVE_FINDING: a discontinuous terminal condition costs the
scheme a full order in the price, because the terminal data at the node
nearest the strike is wrong by `O(cash)` for any `ds`. Rows 2-5 are the
remedy, and rows 4-5 against rows 2-3 are the no-op result -- on this path
alignment and the projection do the same job, and Rannacher does almost
nothing (price error 1.216e-05 against 1.296e-05 at `n = 100`).

The stressed path (`n_s = 80 n_t`, `n_t` in (10, 20, 40))
----------------------------------------------------------
This is the path Slice 4 used to expose the undamped-Crank-Nicolson gamma, and
it is where Rannacher stops being cosmetic.

| configuration                        | price   | delta   | gamma   |
|--------------------------------------|---------|---------|---------|
| plain CN, unaligned, no projection   | +0.0106 | -1.0008 | -1.9986 |
| CN, unaligned, `cell_average`        | +0.9915 | -1.0004 | -0.9996 |
| plain CN, midpoint-aligned           | +0.9912 | -0.9978 | -0.9980 |
| Rannacher, midpoint-aligned          | +2.0043 | +2.0583 | +2.0592 |
| Rannacher, unaligned, `cell_average` | +2.0044 | +2.0586 | +2.0586 |

Three configurations **diverge**: refining makes delta and gamma worse, by a
factor of two and four per level respectively. The plain-CN unaligned gamma
error reaches **52.19** against a true gamma of -3.283e-04. The two remedies
are therefore orthogonal and both are required: cell averaging (or alignment)
fixes how the jump is represented in *space*, Rannacher fixes the fact that
Crank-Nicolson does not damp it in *time*. Neither alone is enough here, which
is the sharpest statement this slice makes.

Citations: Pooley, Forsyth and Vetzal (2003), Journal of Computational Finance
6(4), for payoff projection as the remedy for non-smooth payoffs; Rannacher
(1984), Numerische Mathematik 43, and Giles and Carter (2006), Journal of
Computational Finance 9(4), for the implicit start-up. No number here is taken
from any of them.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import quad

from qpl.engines.analytic.digital import digital_price, greeks_digital as analytic_greeks
from qpl.engines.pde.digital import digital_payoff_on_grid
from qpl.engines.pde.grid import SpotGrid
from qpl.engines.pde.pricers import PDEConfig
from qpl.exceptions import InvalidInputError
from qpl.instruments.options import DigitalOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass, fit_convergence_order

LEVELS = (100, 200, 400, 800)
"""`n_s = n_t = n`. Deliberately *not* including `n = 50`: with the default
`s_max = 4 S = 400` the unaligned spacing is `8` there, which puts the strike
at `12.5` spacings -- accidentally midpoint-aligned. Every level in `LEVELS`
puts the strike exactly *on* a node instead, which is the unaligned worst case
and the one the pathology rows are about."""

STRESSED_LEVELS = (10, 20, 40)
"""`n_t`, with `n_s = 80 n_t`: `dt` large relative to `ds**2`."""

_SPOT = 100.0
_STRIKE = 100.0
_EXPIRY = 1.0
_RATE = 0.05
_DIV = 0.0
_SIGMA = 0.20
_CASH = 1.0

GAMMA_SIGN_CHANGE_SPOT = _STRIKE * math.exp(-(_RATE + 0.5 * _SIGMA * _SIGMA) * _EXPIRY)
"""The spot at which `d1 = 0`, so the digital's gamma is exactly zero and
changes sign. 93.2394 at the reference point."""


def _triple(kind: str = "call", spot: float = _SPOT, sigma: float = _SIGMA, expiry: float = _EXPIRY):
    return (
        DigitalOption(kind=kind, strike=_STRIKE, expiry=expiry, cash=_CASH),
        BlackScholesModel(sigma=sigma),
        Market(
            spot=spot,
            rate_curve=FlatRateCurve(_RATE),
            dividend_curve=FlatDividendCurve(_DIV),
        ),
    )


def _cfg(n: int, alignment: str, time_stepping: str, projection: str, *, stressed: bool = False):
    return PDEConfig(
        n_s=80 * n if stressed else n,
        n_t=n,
        theta=0.5,
        s_max_multiplier=4.0,
        strike_alignment=alignment,  # type: ignore[arg-type]
        time_stepping=time_stepping,  # type: ignore[arg-type]
        payoff_projection=projection,  # type: ignore[arg-type]
    )


def _errors(
    quantity: str,
    alignment: str,
    time_stepping: str,
    projection: str,
    *,
    levels=LEVELS,
    stressed: bool = False,
    spot: float = _SPOT,
) -> list[float]:
    """Signed errors of `quantity` against the closed form, level by level."""
    option, model, market = _triple(spot=spot)
    if quantity == "price":
        exact = digital_price(
            S=spot,
            K=_STRIKE,
            T=_EXPIRY,
            r=_RATE,
            sigma=_SIGMA,
            q=_DIV,
            cash=_CASH,
            kind="call",
        )
    else:
        exact = getattr(analytic_greeks(option, model, market), quantity)

    out = []
    for n in levels:
        cfg = _cfg(n, alignment, time_stepping, projection, stressed=stressed)
        if quantity == "price":
            value = price(option, model, market, method="pde", cfg=cfg).value
        else:
            value = getattr(greeks(option, model, market, method="pde", cfg=cfg), quantity)
        out.append(value - exact)
    return out


def _order(errors, levels=LEVELS):
    return fit_convergence_order([1.0 / n for n in levels], [abs(e) for e in errors])


# --------------------------------------------------------------------------
# The projection itself.
# --------------------------------------------------------------------------


def test_cell_average_is_the_exact_average_of_the_indicator() -> None:
    """Evidence class: EXACT_IDENTITY on the discretisation.

    For a step function the cell average has a closed form -- the fraction of
    the cell in the money -- so no quadrature and no smoothing width is
    involved. Three properties are checked against a brute-force average of
    the payoff over each cell on a fine sub-grid:

    - the projected values match that average to 1e-12;
    - every value lies in `[0, cash]`, and at most one interior cell is
      strictly between (the one the strike straddles);
    - call and put projections sum to `cash` at every node, so the static
      replication identity survives the projection exactly.
    """
    call = DigitalOption(kind="call", strike=100.0, expiry=1.0, cash=3.0)
    put = DigitalOption(kind="put", strike=100.0, expiry=1.0, cash=3.0)
    ds = 400.0 / 37.0  # deliberately not a divisor of the strike
    s_grid = ds * np.arange(38, dtype=float)
    grid = SpotGrid(s=s_grid, kind="uniform", uniform=True, ds=ds)

    projected = digital_payoff_on_grid(call, grid, "cell_average")

    # Reference: adaptive quadrature of the indicator over each cell, told
    # where the discontinuity is. Shares no code with the closed-form fraction.
    quadrature = np.array(
        [
            quad(
                lambda x: 3.0 * float(x > 100.0),
                s - 0.5 * ds,
                s + 0.5 * ds,
                points=[100.0],
            )[0]
            / ds
            for s in s_grid
        ]
    )
    assert np.max(np.abs(projected - quadrature)) < 1e-12 * 3.0

    assert np.all(projected >= 0.0) and np.all(projected <= 3.0)
    strictly_inside = np.sum((projected > 1e-12) & (projected < 3.0 - 1e-12))
    assert strictly_inside == 1, projected

    put_projected = digital_payoff_on_grid(put, grid, "cell_average")
    assert np.allclose(projected + put_projected, 3.0, rtol=0.0, atol=1e-15)


def test_midpoint_alignment_makes_the_projection_a_no_op() -> None:
    """Evidence class: NEGATIVE_FINDING -- the slice expected these to stack.

    With `strike_alignment="midpoint"` the spacing satisfies `K = (j + 1/2) ds`,
    so node `j`'s cell `[(j - 1/2) ds, (j + 1/2) ds]` ends exactly *at* the
    strike. No cell straddles `K`, every cell is wholly in or wholly out of the
    money, and the cell average of the indicator is its point sample. The
    projection therefore has nothing to do, and the measured prices agree to
    1.11e-16 at `n = 100` and are bit-for-bit identical at 200, 400 and 800.

    The two remedies are not additive; they are the same remedy reached two
    ways, and the projection's job is on the *unaligned* grid where alignment
    is not available.
    """
    option, model, model_market = _triple()
    for n in LEVELS:
        plain = price(
            option,
            model,
            model_market,
            method="pde",
            cfg=_cfg(n, "midpoint", "rannacher", "none"),
        ).value
        projected = price(
            option,
            model,
            model_market,
            method="pde",
            cfg=_cfg(n, "midpoint", "rannacher", "cell_average"),
        ).value
        assert abs(plain - projected) <= 1e-15, (n, plain, projected)

    # The geometric statement behind it: the straddling cell's face is the strike.
    cfg = _cfg(400, "midpoint", "rannacher", "none")
    ds = price(option, model, model_market, method="pde", cfg=cfg).meta["ds"]
    assert (_STRIKE / ds) % 1.0 == pytest.approx(0.5, abs=1e-12)


# --------------------------------------------------------------------------
# The pathology.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quantity", "expected_order", "band"),
    [("price", 1.0012, 0.15), ("delta", 0.8632, 0.15), ("gamma", 1.0153, 0.15)],
)
def test_plain_crank_nicolson_on_an_unaligned_grid_loses_an_order(
    quantity: str, expected_order: float, band: float
) -> None:
    """Evidence class: NEGATIVE_FINDING. The discontinuity pathology, pinned.

    Plain Crank-Nicolson, `strike_alignment="none"`, `payoff_projection="none"`,
    `n_s = n_t = n` over (100, 200, 400, 800). At each of those levels the
    default `s_max = 400` makes the strike land exactly on a node, so the node
    carries the full `cash` jump on one side and nothing on the other.

    Measured orders and log-space residuals: price **1.0012** (0.0007), delta
    **0.8632** (0.0390), gamma **1.0153** (0.0050). The price errors are
    -3.761e-02, -1.876e-02, -9.377e-03, -4.689e-03 -- a clean halving, i.e.
    genuinely first order rather than noise, and 169 times larger at `n = 800`
    than the aligned configuration's 2.23e-07.

    A vanilla on the same grid is second order in the price (Slice 0). Losing
    exactly one order is what a jump costs over a kink.
    """
    errors = _errors(quantity, "none", "theta", "none")
    fit = _order(errors)

    assert abs(fit.order - expected_order) <= band, (fit.order, errors)
    assert fit.residual < 0.05, fit.residual
    # Nowhere near second order, which is the whole point of the row.
    assert fit.order < 1.3, fit.order


def test_the_pathology_is_a_price_error_not_only_a_greek_error() -> None:
    """Evidence class: NEGATIVE_FINDING, and the difference from Slice 4.

    Slice 4's kink pathology showed up in gamma and left the price alone: the
    price is an average over the grid and barely notices a ripple. A jump is
    one derivative worse and the damage reaches the price itself. At
    `n = 800` the plain-CN unaligned price is out by **4.689e-03** on a value
    of 0.5323 -- 0.88% -- against **2.229e-07** for the remedied grid, a factor
    of 21_000.
    """
    unaligned = abs(_errors("price", "none", "theta", "none", levels=(800,))[0])
    remedied = abs(_errors("price", "midpoint", "rannacher", "none", levels=(800,))[0])

    assert unaligned > 4.0e-03
    assert unaligned / remedied > 10_000.0, (unaligned, remedied)


# --------------------------------------------------------------------------
# The remedy.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alignment", "projection"), [("none", "cell_average"), ("midpoint", "none")]
)
@pytest.mark.parametrize(
    ("quantity", "expected_order"), [("price", 2.0), ("delta", 2.0), ("gamma", 2.0)]
)
def test_rannacher_plus_a_consistent_jump_representation_restores_order_two(
    alignment: str, projection: str, quantity: str, expected_order: float
) -> None:
    """Evidence class: CONVERGENCE_ORDER.

    Two ways to represent the jump consistently -- cell-average it onto an
    unaligned grid, or align the grid so the jump sits on a cell face -- both
    with Rannacher start-up. Measured:

    | configuration                        | price  | delta  | gamma  |
    |--------------------------------------|--------|--------|--------|
    | Rannacher, unaligned, `cell_average` | 2.0154 | 2.0061 | 1.9980 |
    | Rannacher, midpoint-aligned          | 1.9248 | 2.0360 | 2.0315 |

    The band is `2.0 +- 0.2`. The aligned price fit sits at 1.92 rather than
    2.00 because its errors are already down at 2.2e-07 by `n = 800`, where the
    `O(ds**2)` term is competing with the `O(dt**2)` one; the residual (0.023)
    is small, so it is a clean power law at a slightly different slope, not a
    broken one.
    """
    errors = _errors(quantity, alignment, "rannacher", projection)
    fit = _order(errors)

    assert abs(fit.order - expected_order) <= 0.2, (fit.order, errors)
    assert fit.residual < 0.05, fit.residual
    # Monotone: no oscillation left to average away.
    assert all(abs(a) > abs(b) for a, b in pairwise(errors)), errors


def test_the_remedy_agrees_with_the_closed_form_at_the_finest_grid() -> None:
    """Evidence class: CLOSED_FORM.

    At `n_s = n_t = 800`, Rannacher, midpoint-aligned: price error 2.229e-07,
    delta error -7.413e-07 on 1.876e-02, gamma error 9.955e-08 on -3.283e-04.
    The tolerances below keep a factor of 3 to 4 over each.
    """
    option, model, market = _triple()
    cfg = _cfg(800, "midpoint", "rannacher", "none")
    exact_price = digital_price(
        S=_SPOT, K=_STRIKE, T=_EXPIRY, r=_RATE, sigma=_SIGMA, q=_DIV, kind="call"
    )
    exact = analytic_greeks(option, model, market)

    assert price(option, model, market, method="pde", cfg=cfg).value == pytest.approx(
        exact_price, abs=7.0e-07
    )
    numeric = greeks(option, model, market, method="pde", cfg=cfg)
    assert numeric.delta == pytest.approx(exact.delta, abs=3.0e-06)
    assert numeric.gamma == pytest.approx(exact.gamma, abs=4.0e-07)


# --------------------------------------------------------------------------
# The stressed grid: the two remedies are orthogonal.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alignment", "projection", "label"),
    [
        ("none", "none", "plain CN, unaligned"),
        ("none", "cell_average", "CN, unaligned, projected"),
        ("midpoint", "none", "CN, aligned"),
    ],
)
def test_on_a_stressed_grid_crank_nicolson_diverges_whatever_the_jump_representation(
    alignment: str, projection: str, label: str
) -> None:
    """Evidence class: NEGATIVE_FINDING.

    `n_s = 80 n_t`, `n_t` in (10, 20, 40), so `dt` is large relative to
    `ds**2` -- the regime Slice 4 used to expose undamped Crank-Nicolson on a
    vanilla's kink. Measured delta and gamma orders:

    | configuration                      | delta   | gamma   |
    |------------------------------------|---------|---------|
    | plain CN, unaligned, no projection | -1.0008 | -1.9986 |
    | CN, unaligned, `cell_average`      | -1.0004 | -0.9996 |
    | plain CN, midpoint-aligned         | -0.9978 | -0.9980 |

    Every one is **negative**: refining the grid makes the Greek worse. The
    plain-CN unaligned gamma error grows 3.268 -> 13.05 -> 52.19 against a true
    gamma of -3.283e-04. Fixing how the jump is represented in space does not
    help, because the problem is in time: Crank-Nicolson's amplification factor
    tends to -1 for the stiffest modes and a jump is made of them.
    """
    for quantity in ("delta", "gamma"):
        errors = _errors(
            quantity, alignment, "theta", projection, levels=STRESSED_LEVELS, stressed=True
        )
        fit = _order(errors, levels=STRESSED_LEVELS)
        assert fit.order < -0.5, (label, quantity, fit.order, errors)
        assert fit.residual < 0.05, fit.residual
        # Divergence, stated as growth rather than only as a slope.
        assert abs(errors[-1]) > 3.0 * abs(errors[0]), (label, quantity, errors)


@pytest.mark.parametrize(
    ("alignment", "projection"), [("midpoint", "none"), ("none", "cell_average")]
)
@pytest.mark.parametrize("quantity", ["price", "delta", "gamma"])
def test_rannacher_rescues_the_stressed_grid_when_the_jump_is_represented_consistently(
    alignment: str, projection: str, quantity: str
) -> None:
    """Evidence class: CONVERGENCE_ORDER. Both remedies, together.

    Same stressed grid, same three quantities, with Rannacher start-up added:

    | configuration                        | price  | delta  | gamma  |
    |--------------------------------------|--------|--------|--------|
    | Rannacher, midpoint-aligned          | 2.0043 | 2.0583 | 2.0592 |
    | Rannacher, unaligned, `cell_average` | 2.0044 | 2.0586 | 2.0586 |

    The gamma error at `n_t = 40` is 4.4e-08 where undamped Crank-Nicolson on
    the same grid is at 52.19: a factor of 1.2e+09. Rannacher alone is not
    enough either -- it is tested against the unaligned, unprojected grid in
    the next test.
    """
    errors = _errors(
        quantity, alignment, "rannacher", projection, levels=STRESSED_LEVELS, stressed=True
    )
    fit = _order(errors, levels=STRESSED_LEVELS)
    assert abs(fit.order - 2.0) <= 0.2, (quantity, fit.order, errors)
    assert fit.residual < 0.05, fit.residual


def test_rannacher_alone_does_not_fix_an_unaligned_unprojected_grid() -> None:
    """Evidence class: NEGATIVE_FINDING. The other half of the orthogonality.

    On the standard `n_s = n_t` path with `strike_alignment="none"` and
    `payoff_projection="none"`, adding Rannacher moves the price by 2.358e-06
    at `n = 100`, falling to 1.539e-08 at `n = 800` -- against an *error* of
    3.761e-02 and 4.689e-03 at those levels, i.e. a correction 16_000 to
    300_000 times smaller than what is wrong. The fitted price order is
    1.0012 with damping and 1.0012 without.

    Damping a jump that is already mis-represented on the grid removes ripples
    it never had: the missing order comes from the terminal data, not from the
    time march. This is the converse of the stressed-grid result above, and
    together the two say the remedies are orthogonal.
    """
    plain = _errors("price", "none", "theta", "none")
    damped = _errors("price", "none", "rannacher", "none")

    corrections = [abs(a - b) for a, b in zip(plain, damped, strict=True)]
    assert all(c < 1e-05 for c in corrections), corrections
    assert all(
        c < 1e-03 * abs(e) for c, e in zip(corrections, plain, strict=True)
    ), (corrections, plain)

    for errors in (plain, damped):
        fit = _order(errors)
        assert abs(fit.order - 1.0) <= 0.15, (fit.order, errors)


# --------------------------------------------------------------------------
# Gamma, reported honestly.
# --------------------------------------------------------------------------


def test_gamma_is_second_order_in_absolute_terms_but_has_no_relative_accuracy_at_its_zero() -> None:
    """Evidence class: CONVERGENCE_ORDER plus NEGATIVE_FINDING.

    A digital's gamma carries the factor `-d1`, so it changes sign at
    `S = K exp(-(r + sigma**2/2) T)` = 93.2394 at this point, where the true
    gamma is -2.97e-19, i.e. zero.

    At the money (`S = K = 100`, true gamma -3.283e-04) the remedied grid gives
    gamma at order **2.0315** with residual 0.0239 -- the same order as delta,
    with no penalty for the sign change elsewhere on the grid.

    At the sign-change spot the *absolute* error still falls at order **2.1226**
    (residual 0.0837, errors 9.67e-06 down to 1.09e-07), but the relative error
    is infinite at every level, because the quantity being approximated is
    zero. Both are asserted: gamma is second order, and "second order" says
    nothing about relative accuracy near its zero.
    """
    at_the_money = _errors("gamma", "midpoint", "rannacher", "none")
    fit = _order(at_the_money)
    assert abs(fit.order - 2.0) <= 0.2, (fit.order, at_the_money)

    at_zero = _errors(
        "gamma", "midpoint", "rannacher", "none", spot=GAMMA_SIGN_CHANGE_SPOT
    )
    zero_fit = _order(at_zero)
    assert abs(zero_fit.order - 2.1) <= 0.25, (zero_fit.order, at_zero)

    option, model, market = _triple(spot=GAMMA_SIGN_CHANGE_SPOT)
    exact = analytic_greeks(option, model, market).gamma
    assert abs(exact) < 1e-15
    # Relative accuracy is not a thing here: the grid value is 1.09e-07 and the
    # truth is 0, so the ratio is unbounded however fine the grid.
    assert abs(at_zero[-1]) > 1e-08


# --------------------------------------------------------------------------
# Identities, degenerate limits, validation, and the vanilla's bit-for-bit.
# --------------------------------------------------------------------------


def test_grid_prices_preserve_the_static_replication_identity() -> None:
    """Evidence class: EXACT_IDENTITY where it holds, NEGATIVE_FINDING where it does not.

    The projection and the boundary data are call/put symmetric and the
    operator is linear, so the two grid solves should sum to `cash e^{-rT}`.
    Measured at `n_s = n_t = 200`, `cash = 2.5`:

    | configuration                  | theta      | rannacher  |
    |--------------------------------|------------|------------|
    | unaligned, no projection       | -9.393e-02 | -9.393e-02 |
    | unaligned, `cell_average`      | -6.193e-10 | +7.370e-08 |
    | midpoint-aligned               | -6.193e-10 | +7.370e-08 |

    Two things to read off it.

    **The unaligned, unprojected grid breaks the identity by 3.8% of the
    riskless value.** At these levels a node sits exactly on the strike, and
    the strict payoff convention pays nothing there in *either* leg, so that
    node's whole weight is lost from the sum. Cell averaging repairs it
    exactly, because the call and put projections sum to `cash` at every node
    by construction. This is an argument for the projection that has nothing to
    do with convergence order.

    **The surviving residual is the time scheme's own discounting error, not
    round-off.** The sum is a function of `tau` alone (constant in `S`), so the
    scheme integrates `V' = -rV` and returns the Pade approximant of
    `e^{-r dt}` rather than the exponential: `O(dt**3)` per Crank-Nicolson step,
    which accumulates to 6.2e-10, and `O(dt**2)` for each of the four fully
    implicit Rannacher half steps, which is the larger 7.4e-08. Rannacher is
    *less* exact on this identity and more accurate on everything else.
    """
    call = DigitalOption(kind="call", strike=_STRIKE, expiry=_EXPIRY, cash=2.5)
    put = DigitalOption(kind="put", strike=_STRIKE, expiry=_EXPIRY, cash=2.5)
    model = BlackScholesModel(sigma=_SIGMA)
    market = Market(
        spot=_SPOT,
        rate_curve=FlatRateCurve(_RATE),
        dividend_curve=FlatDividendCurve(_DIV),
    )
    riskless = 2.5 * math.exp(-_RATE * _EXPIRY)

    def total(alignment: str, time_stepping: str, projection: str) -> float:
        cfg = _cfg(200, alignment, time_stepping, projection)
        return (
            price(call, model, market, method="pde", cfg=cfg).value
            + price(put, model, market, method="pde", cfg=cfg).value
        )

    for time_stepping, budget in (("theta", 1e-09), ("rannacher", 1e-07)):
        for alignment, projection in (("none", "cell_average"), ("midpoint", "none")):
            residual = total(alignment, time_stepping, projection) - riskless
            assert abs(residual) <= budget, (alignment, projection, time_stepping, residual)

        broken = total("none", time_stepping, "none") - riskless
        assert broken == pytest.approx(-9.393e-02, rel=1e-3), (time_stepping, broken)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("spot", [95.0, 105.0])
def test_degenerate_limits_match_the_analytic_engine(kind: str, spot: float) -> None:
    """Evidence class: CLOSED_FORM. There is no grid at `T = 0` or `sigma = 0`."""
    cfg = _cfg(100, "midpoint", "rannacher", "cell_average")
    for triple in (_triple(kind, spot=spot, expiry=0.0), _triple(kind, spot=spot, sigma=0.0)):
        assert price(*triple, method="pde", cfg=cfg).value == price(*triple).value


def test_grid_greeks_refuse_the_degenerate_limits() -> None:
    cfg = _cfg(100, "midpoint", "rannacher", "none")
    with pytest.raises(InvalidInputError, match="expiry must be > 0"):
        greeks(*_triple(expiry=0.0), method="pde", cfg=cfg)
    with pytest.raises(InvalidInputError, match="sigma must be > 0"):
        greeks(*_triple(sigma=0.0), method="pde", cfg=cfg)


def test_bad_payoff_projection_is_rejected() -> None:
    cfg = PDEConfig(n_s=100, n_t=100, payoff_projection="smoothed")  # type: ignore[arg-type]
    with pytest.raises(
        InvalidInputError, match="payoff_projection must be 'none' or 'cell_average'"
    ):
        price(*_triple(), method="pde", cfg=cfg)


def test_payoff_projection_defaults_to_none_and_the_vanilla_ignores_it() -> None:
    """The field is read only by the digital engine, like `psor` for American.

    `PDEConfig()` defaults to `"none"`, so every pre-Slice-6 vanilla call is
    unaffected; and a vanilla priced with `"cell_average"` returns the *same*
    number bit-for-bit, because the vanilla terminal condition is still sampled
    at the nodes. Projecting a kink is a real remedy too, but it is not what
    this slice measured and it is not silently switched on.
    """
    assert PDEConfig().payoff_projection == "none"

    vanilla = EuropeanOption(kind="call", strike=_STRIKE, expiry=_EXPIRY)
    model = BlackScholesModel(sigma=_SIGMA)
    market = Market(
        spot=_SPOT,
        rate_curve=FlatRateCurve(_RATE),
        dividend_curve=FlatDividendCurve(_DIV),
    )
    plain = price(vanilla, model, market, method="pde", cfg=_cfg(200, "midpoint", "rannacher", "none"))
    projected = price(
        vanilla, model, market, method="pde", cfg=_cfg(200, "midpoint", "rannacher", "cell_average")
    )
    assert plain.value == projected.value
    assert plain.meta is not None and plain.meta["payoff_projection"] == "none"


def test_evidence_classes_used_here_exist() -> None:
    for name in ("CONVERGENCE_ORDER", "NEGATIVE_FINDING", "EXACT_IDENTITY", "CLOSED_FORM"):
        assert isinstance(getattr(EvidenceClass, name), EvidenceClass)
