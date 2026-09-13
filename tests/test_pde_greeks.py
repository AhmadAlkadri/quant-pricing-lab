"""Greeks read off the finite-difference grid, and what Rannacher fixes.

Every number quoted in a docstring here was measured by running the code in
this file; none is taken from a source. The reference point throughout is
S = K = 100, r = 5%, q = 0, sigma = 20%, European call, at T = 1 for the
smooth-case order measurements and T = 0.05 for the start-up pathology, which
needs a short maturity to be visible.

Evidence classes used: CONVERGENCE_ORDER for every measured order,
NEGATIVE_FINDING for the plain-Crank-Nicolson gamma pathology and for the
stalling of the bump path, CLOSED_FORM for the single-point comparisons.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from qpl.engines.pde.pricers import PDEConfig
from qpl.exceptions import InvalidInputError
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks
from qpl.validation import fit_convergence_order


def _market(spot: float, r: float, q: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(r),
        dividend_curve=FlatDividendCurve(q),
    )


_SPOT = 100.0
_STRIKE = 100.0
_RATE = 0.05
_DIVIDEND = 0.0
_SIGMA = 0.2

_SMOOTH_EXPIRY = 1.0
"""Long enough that the payoff kink has diffused away by `tau = T`."""

_STARTUP_EXPIRY = 0.05
"""Short enough that the Crank-Nicolson start-up ripple is still there."""


def _setup(expiry: float, kind: str = "call"):
    option = EuropeanOption(kind=kind, strike=_STRIKE, expiry=expiry)
    model = BlackScholesModel(sigma=_SIGMA)
    market = _market(_SPOT, _RATE, _DIVIDEND)
    analytic = greeks(option, model, market, method="analytic")
    return option, model, market, analytic


def _pde_greeks(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    n_s: int,
    n_t: int,
    time_stepping: str = "rannacher",
    alignment: str = "midpoint",
    greeks_method: str = "grid",
):
    cfg = PDEConfig(
        n_s=n_s,
        n_t=n_t,
        theta=0.5,
        s_max_multiplier=4.0,
        strike_alignment=alignment,  # type: ignore[arg-type]
        time_stepping=time_stepping,  # type: ignore[arg-type]
        greeks_method=greeks_method,  # type: ignore[arg-type]
    )
    return greeks(option, model, market, method="pde", cfg=cfg)


_JOINT_LEVELS = (50, 100, 200, 400, 800)
"""`n_s = n_t = n`; ds and dt shrink together, so one fitted slope is the
joint order of the scheme."""


# --------------------------------------------------------------------------
# (a) Delta, gamma and theta converge at order two on an aligned grid.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("time_stepping", ["theta", "rannacher"])
@pytest.mark.parametrize(
    ("greek", "expected_order"),
    [("delta", 2.0), ("gamma", 2.0), ("theta", 2.0)],
)
def test_grid_greeks_converge_at_order_two(
    time_stepping: str, greek: str, expected_order: float
) -> None:
    """Delta, gamma and theta read from the grid are second order.

    Evidence class: CONVERGENCE_ORDER, against the closed-form Greeks.

    Delta and gamma are central stencils, `O(ds**2)` where the solution is
    smooth, and the spot is not a node here -- with `strike_alignment=
    "midpoint"` and S = K the spot sits exactly halfway between two nodes,
    which is the *worst* case for interpolation. Interpolating the Greek
    rather than the price is what keeps that second order: linear
    interpolation of a smooth function between nodes a distance ds apart adds
    an `O(ds**2)` term, so nothing is lost.

    Theta is the PDE identity, built from those two stencils and the price,
    so it is second order for the same reason.

    Measured over n in (50, 100, 200, 400, 800), reference ATM call, T = 1:

    | greek | theta-scheme    | rannacher       |
    |-------|-----------------|-----------------|
    | delta | 2.001 (r 0.021) | 2.001 (r 0.021) |
    | gamma | 2.064 (r 0.035) | 2.067 (r 0.036) |
    | theta | 2.024 (r 0.018) | 2.024 (r 0.019) |

    **This contradicts the slice statement that wrote this test.** The slice
    expected plain Crank-Nicolson gamma to show start-up pollution here and
    Rannacher to repair it. On *this* refinement path it does not: with
    n_s = n_t = n the time step shrinks as fast as the spacing, so
    `lambda * dt` at the strike stays around 1 and Crank-Nicolson damps the
    stiff modes perfectly well. The pathology is real, but it needs a time
    step that is large relative to `ds**2`, which n_s = n_t = n hides; see
    `test_plain_crank_nicolson_gamma_diverges_under_refinement` for the
    refinement path on which it appears.
    """
    option, model, market, analytic = _setup(_SMOOTH_EXPIRY)
    reference = getattr(analytic, greek)

    errs = [
        abs(
            getattr(
                _pde_greeks(
                    option, model, market, n_s=n, n_t=n, time_stepping=time_stepping
                ),
                greek,
            )
            - reference
        )
        for n in _JOINT_LEVELS
    ]

    fit = fit_convergence_order([1.0 / n for n in _JOINT_LEVELS], errs)
    assert abs(fit.order - expected_order) <= 0.2, (greek, time_stepping, fit.order)
    assert fit.residual < 0.1, fit.residual
    # Monotone: the sequence really is a decaying power law, not a lucky fit.
    assert all(b < a for a, b in pairwise(errs)), errs


# --------------------------------------------------------------------------
# (b) Theta: the identity is the answer, the time difference is the check.
# --------------------------------------------------------------------------


def test_theta_identity_is_second_order_and_backward_difference_is_first() -> None:
    """The two ways to read theta, and why only one of them is reported.

    Evidence class: CONVERGENCE_ORDER.

    `greeks_european` reports the PDE identity

        V_t = -(1/2 sigma**2 S**2 V_SS + (r - q) S V_S - r V),

    whose right-hand side is built entirely from second-order spatial
    quantities. It therefore carries no time-discretisation error of its own:
    it inherits whatever error the *solution* has. The alternative, a
    difference of the last two time levels, is a one-sided difference in
    calendar time and is `O(dt)` no matter how accurate the scheme is.

    That is measured here by holding the spatial grid fixed and fine
    (n_s = 800, aligned) and refining only n_t over (25, 50, 100, 200), with
    Rannacher stepping:

    | n_t | identity error | backward-difference error |
    |-----|----------------|---------------------------|
    | 25  | -1.076e-03     | -4.432e-02                |
    | 50  | -8.362e-05     | -2.130e-02                |
    | 100 | +1.789e-04     | -1.031e-02                |
    | 200 | +2.440e-04     | -4.944e-03                |

    The backward difference fits order **1.054** with a log-space RMS residual
    of **0.0020** -- a clean first-order sequence, and it is 20x to 200x larger
    than the identity's error at every level. The identity's own column is not
    fitted: by n_t = 50 it has already reached the spatial floor of that grid
    and is no longer measuring anything temporal. That asymmetry is the whole
    point, so the test asserts the first-order fit on the cross-check and a
    ceiling on the identity, not an order for both.
    """
    option, model, market, analytic = _setup(_SMOOTH_EXPIRY)

    levels = (25, 50, 100, 200)
    identity_errs = []
    backward_errs = []
    for n_t in levels:
        res = _pde_greeks(option, model, market, n_s=800, n_t=n_t)
        identity_errs.append(abs(res.theta - analytic.theta))
        backward_errs.append(
            abs(res.meta["theta_backward_difference"] - analytic.theta)
        )

    fit = fit_convergence_order([1.0 / n for n in levels], backward_errs)
    assert 0.8 <= fit.order <= 1.2, fit.order
    assert fit.residual < 0.05, fit.residual

    # The identity is uniformly better, and already at the spatial floor.
    assert max(identity_errs) < 2e-3, identity_errs
    assert all(b > 15.0 * i for i, b in zip(identity_errs, backward_errs, strict=True))


def test_theta_identity_and_backward_difference_agree_on_a_fine_grid() -> None:
    """Cross-check: the two routes to theta must meet where both are accurate.

    Evidence class: CLOSED_FORM. Two estimators with different leading error
    terms agreeing to 1e-3 on a fine grid is evidence that neither has a sign
    error or a missing term; it says nothing about which is more accurate,
    which the test above measures.
    """
    option, model, market, analytic = _setup(_SMOOTH_EXPIRY)
    res = _pde_greeks(option, model, market, n_s=800, n_t=800)

    assert res.theta == pytest.approx(analytic.theta, abs=1e-3)
    assert res.meta["theta_backward_difference"] == pytest.approx(
        analytic.theta, abs=2e-3
    )
    assert res.meta["theta_source"] == "PDE identity at the spot"


# --------------------------------------------------------------------------
# (c) The Crank-Nicolson start-up pathology, and the Rannacher repair.
# --------------------------------------------------------------------------


_STARTUP_TIME_LEVELS = (5, 10, 20, 40)
"""n_t for the start-up study; the spatial grid is `n_s = 80 * n_t`.

Holding `n_s / n_t` fixed makes `dt` shrink like `ds`, so `lambda * dt` at the
strike -- which is what governs Crank-Nicolson's amplification factor -- grows
linearly with the refinement instead of staying put. That is the regime in
which the start-up ripple is a real problem, and it is the regime a
practitioner lands in whenever the spatial grid is the expensive resolution and
the time grid is not.
"""


@pytest.mark.parametrize("alignment", ["none", "midpoint"])
def test_plain_crank_nicolson_gamma_diverges_under_refinement(alignment: str) -> None:
    """Plain Crank-Nicolson gamma gets *worse* the finer the grid.

    Evidence class: NEGATIVE_FINDING. This pins a documented failure mode so
    that it cannot silently change and cannot be mistaken for a bug elsewhere.

    Mechanism. One theta step multiplies an eigenmode by
    `R(z) = (1 + (1-theta) z) / (1 - theta z)` with `z = -lambda dt`. At
    theta = 1/2 this tends to **-1** as `z -> -infinity`: the stiffest modes
    are not damped, only flipped in sign each step. The payoff kink is exactly
    high-frequency content in that basis, so it survives as a sign-alternating
    ripple in the solution. The price is an average over the grid and hardly
    notices. Gamma is a second difference of neighbouring values and is
    dominated by it. Rannacher (1984), Numerische Mathematik 43, 309-327;
    Giles and Carter (2006), Journal of Computational Finance 9(4), 89-112.

    Measured, T = 0.05, S = K = 100, `n_s = 80 * n_t`, closed-form gamma
    0.08893343, relative error `(gamma_pde - gamma_bs) / gamma_bs`:

    | n_t | n_s  | unaligned CN | unaligned Rannacher |
    |-----|------|--------------|---------------------|
    | 5   | 400  | -2.591e-01   | +1.303e-02          |
    | 10  | 800  | +5.570e-01   | +4.161e-03          |
    | 20  | 1600 | +1.134e+00   | +9.893e-04          |
    | 40  | 3200 | +2.277e+00   | +2.416e-04          |

    Fitted orders: Crank-Nicolson **-1.043** (residual 0.018) -- a *negative*
    order, i.e. refinement makes it worse at rate n -- against Rannacher's
    **1.933** (residual 0.076). At the finest level plain Crank-Nicolson
    returns a gamma 3.28 times the true value while its price is fine.

    On a strike-aligned grid the same thing happens more slowly: orders
    **-0.915** and **1.888**, relative errors running -3.949e-02, +5.887e-02,
    +1.258e-01, +2.538e-01 for Crank-Nicolson and 2.159e-03 down to 4.467e-05
    for Rannacher. Alignment is not a substitute for damping: it fixes where
    the kink sits, not what the scheme does to it.
    """
    option, model, market, analytic = _setup(_STARTUP_EXPIRY)
    reference = analytic.gamma

    def _errors(time_stepping: str) -> list[float]:
        return [
            abs(
                _pde_greeks(
                    option,
                    model,
                    market,
                    n_s=80 * n_t,
                    n_t=n_t,
                    time_stepping=time_stepping,
                    alignment=alignment,
                ).gamma
                - reference
            )
            for n_t in _STARTUP_TIME_LEVELS
        ]

    cn_errs = _errors("theta")
    rannacher_errs = _errors("rannacher")

    h = [1.0 / n for n in _STARTUP_TIME_LEVELS]
    cn_fit = fit_convergence_order(h, cn_errs)
    rannacher_fit = fit_convergence_order(h, rannacher_errs)

    # The pathology: a negative fitted order. Refining makes gamma worse.
    assert cn_fit.order < -0.5, cn_fit.order
    # It is a clean power law in the wrong direction, not noise.
    assert cn_fit.residual < 0.2, cn_fit.residual
    # And it is large, not a wobble: the final gamma is off by more than 20%.
    assert cn_errs[-1] > 0.2 * reference, cn_errs

    # The repair: order restored, and monotone.
    assert 1.7 <= rannacher_fit.order <= 2.2, rannacher_fit.order
    assert all(
        b < a for a, b in pairwise(rannacher_errs)
    ), rannacher_errs
    # Better at every level, and by a growing margin.
    assert all(r < c for r, c in zip(rannacher_errs, cn_errs, strict=True))
    assert rannacher_errs[-1] * 1000.0 < cn_errs[-1]


def test_the_price_barely_notices_what_destroys_gamma() -> None:
    """The pathology is in the Greeks, not the price.

    Evidence class: NEGATIVE_FINDING (the same one, seen from the other side).
    At `n_s = 1600, n_t = 20, T = 0.05` on an unaligned grid, plain
    Crank-Nicolson returns a gamma **2.13 times** the closed-form value --
    0.189743 against 0.088933, a relative error of 113% -- while its *price* is
    1.90699917 against a closed form of 1.90937494, a relative error of 0.12%.
    That is a factor of about 900 between the two, and it is what makes this
    failure mode dangerous: nothing about the price flags it, and a test suite
    that only checks prices would pass.

    Rannacher on the same grid: gamma 0.089021 (relative error 0.1%) and price
    1.90825869 (relative error 0.058%). It fixes gamma by three orders of
    magnitude and improves the price by a factor of two on the way.
    """
    from qpl.pricing import price

    option = EuropeanOption(kind="call", strike=_STRIKE, expiry=_STARTUP_EXPIRY)
    model = BlackScholesModel(sigma=_SIGMA)
    market = _market(_SPOT, _RATE, _DIVIDEND)
    analytic_price = price(option, model, market, method="analytic").value
    analytic_gamma = greeks(option, model, market, method="analytic").gamma

    rel_price = {}
    rel_gamma = {}
    for time_stepping in ("theta", "rannacher"):
        cfg = PDEConfig(
            n_s=1600,
            n_t=20,
            theta=0.5,
            strike_alignment="none",
            time_stepping=time_stepping,  # type: ignore[arg-type]
        )
        value = price(option, model, market, method="pde", cfg=cfg).value
        gamma = greeks(option, model, market, method="pde", cfg=cfg).gamma
        rel_price[time_stepping] = abs(value - analytic_price) / analytic_price
        rel_gamma[time_stepping] = abs(gamma - analytic_gamma) / analytic_gamma

    # Both prices are respectable; only one of the gammas is.
    assert rel_price["theta"] < 5e-3
    assert rel_price["rannacher"] < 5e-3
    assert rel_gamma["theta"] > 1.0
    assert rel_gamma["rannacher"] < 5e-3
    # The gap between "price is fine" and "gamma is not" is two orders of
    # magnitude or more.
    assert rel_gamma["theta"] > 100.0 * rel_price["theta"]


# --------------------------------------------------------------------------
# (f) The bump path, kept as a named alternative, and where it is worse.
# --------------------------------------------------------------------------


def test_bump_delta_stalls_while_grid_delta_keeps_converging() -> None:
    """`greeks_method="bump"` stops converging; the grid path does not.

    Evidence class: NEGATIVE_FINDING plus CONVERGENCE_ORDER.

    The bump path evaluates three full solves at S, S(1+h) and S(1-h) with
    h = 1% of spot and differences them. Two things go wrong and neither is
    fixed by refining the grid:

    1. the central difference has its own `O(h**2)` bias, and h is fixed at 1%
       of spot, so that bias is a floor;
    2. `s_max` defaults to `s_max_multiplier * spot`, so the three solves are
       on three *different* grids with different `ds` and a different strike
       position -- the one thing `strike_alignment` exists to control. Their
       discretisation errors do not cancel.

    Measured, reference ATM call, T = 1, aligned, Rannacher, delta error:

    | n    | grid       | bump       |
    |------|------------|------------|
    | 50   | -9.038e-03 | +4.165e-04 |
    | 100  | -2.394e-03 | -6.222e-05 |
    | 200  | -5.658e-04 | -7.176e-05 |
    | 400  | -1.430e-04 | -8.248e-05 |
    | 800  | -3.594e-05 | -8.509e-05 |
    | 1600 | -9.007e-06 | -8.574e-05 |

    Fitted orders over n in (50 ... 800): **2.001** (residual 0.021) for the
    grid path against **0.418** (residual 0.563) for the bump path. The bump
    path is *better* on the coarsest grids -- its bias happens to have the
    opposite sign to the discretisation error there -- and then stops, so by
    n = 1600 the grid path is 9.5x closer and pulling away. The bump gamma is
    worse still: its error sequence is not a power law at all (log-space RMS
    residual 0.73 against the grid path's 0.04) and changes sign three times.
    """
    option, model, market, analytic = _setup(_SMOOTH_EXPIRY)

    grid_errs = []
    bump_errs = []
    for n in _JOINT_LEVELS:
        grid_errs.append(
            abs(_pde_greeks(option, model, market, n_s=n, n_t=n).delta - analytic.delta)
        )
        bump_errs.append(
            abs(
                _pde_greeks(
                    option, model, market, n_s=n, n_t=n, greeks_method="bump"
                ).delta
                - analytic.delta
            )
        )

    h = [1.0 / n for n in _JOINT_LEVELS]
    grid_fit = fit_convergence_order(h, grid_errs)
    bump_fit = fit_convergence_order(h, bump_errs)

    assert abs(grid_fit.order - 2.0) <= 0.2, grid_fit.order
    assert grid_fit.residual < 0.1

    # The bump path has stalled: nowhere near order 2, and the fit is not even
    # a power law.
    assert bump_fit.order < 1.0, bump_fit.order
    assert bump_fit.residual > 0.3, bump_fit.residual
    # A floor, stated directly: the last two bump errors are within 5% of each
    # other while the grid errors over the same step fall by a factor of four.
    tail = bump_errs[-2:]
    assert max(tail) / min(tail) < 1.05, tail
    assert grid_errs[-2] / grid_errs[-1] > 3.0, grid_errs
    # And by the finest level the grid path has overtaken it outright.
    assert grid_errs[-1] < 0.5 * bump_errs[-1]


def test_bump_gamma_sequence_is_not_a_power_law() -> None:
    """The bump gamma error is erratic, for a reason that is not the bump size.

    Evidence class: NEGATIVE_FINDING. Bumping the spot moves `s_max`, hence
    `ds`, hence the strike's position among the nodes. The three solves
    therefore sit on three differently-aligned grids and their errors do not
    cancel. Measured log-space RMS residual 0.73 for the bump path against
    0.04 for the grid path over the same five refinements, with three sign
    changes in the bump column.
    """
    option, model, market, analytic = _setup(_SMOOTH_EXPIRY)

    signed = [
        _pde_greeks(
            option, model, market, n_s=n, n_t=n, greeks_method="bump"
        ).gamma
        - analytic.gamma
        for n in _JOINT_LEVELS
    ]
    fit = fit_convergence_order([1.0 / n for n in _JOINT_LEVELS], [abs(e) for e in signed])

    assert fit.residual > 0.3, fit.residual
    sign_changes = sum(
        1 for a, b in pairwise(signed) if a * b < 0.0
    )
    assert sign_changes >= 2, signed

    grid_fit = fit_convergence_order(
        [1.0 / n for n in _JOINT_LEVELS],
        [
            abs(_pde_greeks(option, model, market, n_s=n, n_t=n).gamma - analytic.gamma)
            for n in _JOINT_LEVELS
        ],
    )
    assert grid_fit.residual < 0.1, grid_fit.residual
    assert fit.residual > 5.0 * grid_fit.residual


def test_bump_path_still_returns_nan_for_vega_theta_rho() -> None:
    """The named alternative is the old behaviour, unchanged.

    Evidence class: CLOSED_FORM in the weak sense of pinning an interface: the
    point is that `greeks_method="bump"` is the pre-Slice-4 path verbatim, NaN
    and all, so that comparisons against it are comparisons against what this
    package actually shipped.
    """
    option, model, market, _ = _setup(_SMOOTH_EXPIRY)
    res = _pde_greeks(
        option, model, market, n_s=200, n_t=200, greeks_method="bump", alignment="none"
    )

    assert math.isnan(res.vega)
    assert math.isnan(res.theta)
    assert math.isnan(res.rho)
    assert res.meta["greeks_method"] == "bump"
    assert res.meta["bump_size"] == 1.0


# --------------------------------------------------------------------------
# Vega and rho, and the interface.
# --------------------------------------------------------------------------


def test_grid_vega_and_rho_match_the_closed_form() -> None:
    """Vega and rho are bump-and-revalue and inherit the grid's error.

    Evidence class: CLOSED_FORM. Bumping sigma or r does not move the grid --
    `ds`, `s_max` and the strike alignment depend only on the spot and the
    strike -- so the two solves share their discretisation error and much of it
    cancels. What is left is the grid's own error, not the stencils': these two
    are no better than the price, which is why they are not claimed to be
    second order here.

    Measured at n_s = n_t = 400, aligned, Rannacher: vega residual -7.89e-04
    against a closed-form 37.524035, rho residual -5.67e-03 against 53.232482.
    The tolerances below keep a factor of about four.
    """
    option, model, market, analytic = _setup(_SMOOTH_EXPIRY)
    res = _pde_greeks(option, model, market, n_s=400, n_t=400)

    assert res.vega == pytest.approx(analytic.vega, abs=3e-3)
    assert res.rho == pytest.approx(analytic.rho, abs=2e-2)
    assert res.meta["bumps"] == {"sigma": 1e-2, "r": 1e-4}


@pytest.mark.parametrize("kind", ["call", "put"])
def test_grid_greeks_signs_and_put_call_relations(kind: str) -> None:
    """Sign conventions and the two exact Greek parities.

    Evidence class: EXACT_IDENTITY for the two parities. Differentiating
    put-call parity `C - P = S e^{-qT} - K e^{-rT}` twice in S gives
    `gamma_C = gamma_P` exactly, and once gives
    `delta_C - delta_P = e^{-qT}`. Both hold node-by-node on the grid because
    the two solves share it, so the residual is a round-off budget plus the
    difference between the two boundary conditions, not a model-error budget.
    """
    call, model, market, _ = _setup(_SMOOTH_EXPIRY, "call")
    put = EuropeanOption(kind="put", strike=_STRIKE, expiry=_SMOOTH_EXPIRY)

    res_call = _pde_greeks(call, model, market, n_s=400, n_t=400)
    res_put = _pde_greeks(put, model, market, n_s=400, n_t=400)

    target = res_call if kind == "call" else res_put
    if kind == "call":
        assert target.delta > 0.5
        assert target.rho > 0.0
    else:
        assert target.delta < -0.3
        assert target.rho < 0.0
    assert target.gamma > 0.0
    assert target.vega > 0.0
    assert target.theta < 0.0

    df_q = market.df_q(_SMOOTH_EXPIRY)
    assert res_call.gamma == pytest.approx(res_put.gamma, abs=1e-9)
    assert res_call.delta - res_put.delta == pytest.approx(df_q, abs=1e-6)
    assert res_call.vega == pytest.approx(res_put.vega, abs=1e-6)


def test_grid_greeks_meta_says_where_each_number_came_from() -> None:
    option, model, market, _ = _setup(_SMOOTH_EXPIRY)
    res = _pde_greeks(option, model, market, n_s=200, n_t=200)
    meta = res.meta

    assert meta["greeks_method"] == "grid"
    assert "central stencil" in meta["delta_source"]
    assert "central stencil" in meta["gamma_source"]
    assert meta["theta_source"] == "PDE identity at the spot"
    assert "bump" in meta["vega_source"]
    assert "bump" in meta["rho_source"]
    assert meta["pde_meta"]["time_stepping"] == "rannacher"
    assert meta["pde_meta"]["implicit_startup_steps"] == 4
    # S = K = 100 on a midpoint-aligned grid: the spot is a half-integer number
    # of spacings from the origin, i.e. exactly between two nodes. That is the
    # worst case for the interpolation, and it is the default case at the money.
    assert meta["spot_is_node"] is False
    assert meta["spot_node_weight"] == pytest.approx(0.5, abs=1e-9)


def test_grid_greeks_are_deterministic() -> None:
    option, model, market, _ = _setup(_SMOOTH_EXPIRY)
    a = _pde_greeks(option, model, market, n_s=150, n_t=120)
    b = _pde_greeks(option, model, market, n_s=150, n_t=120)
    for greek in ("delta", "gamma", "vega", "theta", "rho"):
        assert getattr(a, greek) == getattr(b, greek)


def test_grid_greeks_reject_the_degenerate_limits() -> None:
    """T = 0 and sigma = 0 have no grid to differentiate.

    The price is still well defined in both limits and `price_european`
    returns it; gamma is a point mass at the strike in both, so the grid
    engine refuses rather than returning a grid-spacing-dependent number. This
    matches the analytic engine, which raises for the same two inputs.
    """
    model = BlackScholesModel(sigma=_SIGMA)
    market = _market(_SPOT, _RATE, _DIVIDEND)
    cfg = PDEConfig(n_s=100, n_t=100, strike_alignment="midpoint")

    with pytest.raises(InvalidInputError):
        greeks(
            EuropeanOption(kind="call", strike=_STRIKE, expiry=0.0),
            model,
            market,
            method="pde",
            cfg=cfg,
        )
    with pytest.raises(InvalidInputError):
        greeks(
            EuropeanOption(kind="call", strike=_STRIKE, expiry=1.0),
            BlackScholesModel(sigma=0.0),
            market,
            method="pde",
            cfg=cfg,
        )


def test_pde_rejects_unknown_greeks_method() -> None:
    option, model, market, _ = _setup(_SMOOTH_EXPIRY)
    with pytest.raises(InvalidInputError):
        greeks(
            option,
            model,
            market,
            method="pde",
            cfg=PDEConfig(n_s=50, n_t=50, greeks_method="pathwise"),  # type: ignore[arg-type]
        )
