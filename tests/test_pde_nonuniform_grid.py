"""The non-uniform spot grid: the stencil, its order, and the uniform grid intact.

Two claims are measured here and nothing else; the barrier that motivated the
mesh is in `tests/test_pde_barrier.py`.

(a) **The three-point stencil.** Its second-derivative arm is first order
    *pointwise* on a general grid -- the leading term is proportional to
    `h+ - h-` -- and the global error is nonetheless second order when the grid
    is the image of a uniform grid under a smooth map. Both halves are measured:
    a deliberately non-smooth grid (spacings alternating `h`, `2h`) fits order 1
    on the pointwise second derivative, the `sinh` grid fits order 2, and the
    prices and Greeks the scheme produces on the `sinh` grid fit order 2 against
    closed forms.

(h) **`grid="uniform"` is untouched.** The grid construction, the operator
    coefficients and the Greek stencils are compared with `==` against the
    pre-Slice-13 expressions, written out here rather than imported, so that a
    refactor of the shared code cannot change the default path by a single bit.

Evidence classes: CONVERGENCE_ORDER for every fitted order, EXACT_IDENTITY for
the bit-for-bit comparisons, CLOSED_FORM for the stencils checked against
analytic derivatives.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.engines.analytic.black_scholes import greeks_european as analytic_greeks
from qpl.engines.analytic.digital import price_digital as analytic_digital
from qpl.engines.pde.grid import (
    SpotGrid,
    build_spot_grid,
    delta_gamma_nodes,
    inverse_sinh_coordinate,
    operator_coefficients,
    sinh_coordinate,
    stencil_weights,
)
from qpl.engines.pde.pricers import PDEConfig, _build_grid, _operator
from qpl.exceptions import InvalidInputError
from qpl.instruments.options import DigitalOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel, bs_price
from qpl.pricing import greeks, price
from qpl.validation import fit_convergence_order

SPOT = 100.0
STRIKE = 100.0
EXPIRY = 1.0
RATE = 0.05
DIVIDEND = 0.0
SIGMA = 0.20

LEVELS = (50, 100, 200, 400)
"""Refinement ladder, `n_s = n_t = n`. Four levels is enough for a slope and
cheap enough to run in the suite; the fifth level (800) was measured while
writing this file and is quoted in the docstrings, not run."""

H = tuple(1.0 / n for n in LEVELS)

SINH_CONCENTRATION = 0.05


def _market() -> Market:
    return Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )


def _model() -> BlackScholesModel:
    return BlackScholesModel(sigma=SIGMA)


def _cfg(n: int, **kwargs: object) -> PDEConfig:
    base: dict[str, object] = {
        "n_s": n,
        "n_t": n,
        "strike_alignment": "midpoint",
        "time_stepping": "rannacher",
    }
    base.update(kwargs)
    return PDEConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# (h) The uniform grid, bit for bit.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alignment", ["none", "midpoint"])
@pytest.mark.parametrize("n_s", [37, 200])
def test_uniform_grid_construction_is_the_legacy_arithmetic(
    alignment: str, n_s: int
) -> None:
    """Evidence class: EXACT_IDENTITY.

    The pre-Slice-13 `_build_grid` body is written out below and compared with
    `==` -- not `approx` -- against what the shared builder returns. A single
    reassociation (`s_max / n_s` computed from a different `s_max`, say) would
    move the last bits of every price the default configuration produces, and
    that is exactly the kind of change a shared grid module invites.
    """
    cfg = PDEConfig(n_s=n_s, n_t=10, strike_alignment=alignment)  # type: ignore[arg-type]

    # --- the legacy body, copied here as the reference ---
    s_max = cfg.s_max if cfg.s_max is not None else cfg.s_max_multiplier * SPOT
    ds = s_max / cfg.n_s
    if alignment == "midpoint":
        j = max(round(STRIKE / ds - 0.5), 0)
        ds = STRIKE / (j + 0.5)
        s_max = ds * cfg.n_s
        expected = ds * np.arange(cfg.n_s + 1, dtype=float)
    else:
        expected = np.linspace(0.0, s_max, cfg.n_s + 1)
    # --- end of the legacy body ---

    grid = _build_grid(STRIKE, SPOT, cfg)
    assert grid.uniform is True
    assert grid.kind == "uniform"
    assert grid.ds == ds
    assert grid.s_max == s_max
    assert np.array_equal(grid.s, expected)


def test_uniform_operator_coefficients_are_the_legacy_expressions() -> None:
    """Evidence class: EXACT_IDENTITY, on the coefficients themselves.

    `_operator` now branches on whether the grid is uniform. The uniform arm
    must evaluate the same expressions in the same order as the pre-Slice-13
    engine; the general three-point arm is algebraically equal to it on a
    uniform grid but **not** bit-for-bit, because `2 / (h (h + h))` and
    `1 / h**2` round differently. The test therefore checks the uniform arm
    with `==` and the general arm with a round-off tolerance, which is the
    honest pair of statements.
    """
    cfg = PDEConfig(n_s=200, n_t=10, strike_alignment="midpoint")
    grid = _build_grid(STRIKE, SPOT, cfg)
    s_inner = grid.s[1:-1]
    ds = grid.ds
    r, q = 0.05, 0.01

    diffusion = 0.5 * SIGMA * SIGMA * (s_inner**2) / (ds * ds)
    drift = (r - q) * s_inner / (2.0 * ds)
    legacy = (
        diffusion - drift,
        -(SIGMA * SIGMA) * (s_inner**2) / (ds * ds) - r,
        diffusion + drift,
    )

    got = _operator(grid, SIGMA, r, q)
    for actual, expected in zip(got, legacy, strict=True):
        assert np.array_equal(actual, expected)

    # The same nodes, declared non-uniform, go through the general weights.
    forced = SpotGrid(s=grid.s, kind="uniform", uniform=False, ds=ds)
    general = operator_coefficients(forced, SIGMA, r, q)
    for actual, expected in zip(general, legacy, strict=True):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert not np.array_equal(actual, expected)


def test_uniform_greek_stencils_are_the_legacy_central_differences() -> None:
    """Evidence class: EXACT_IDENTITY, on the delta and gamma node arrays."""
    cfg = PDEConfig(n_s=200, n_t=10, strike_alignment="midpoint")
    grid = _build_grid(STRIKE, SPOT, cfg)
    v = np.exp(-0.01 * grid.s) * np.sin(0.03 * grid.s)
    ds = grid.ds

    delta, gamma = delta_gamma_nodes(grid, v)
    assert np.array_equal(delta, (v[2:] - v[:-2]) / (2.0 * ds))
    assert np.array_equal(gamma, (v[2:] - 2.0 * v[1:-1] + v[:-2]) / (ds * ds))


def test_the_default_configuration_is_the_uniform_grid() -> None:
    """The new fields default to the pre-Slice-13 behaviour, and say so."""
    cfg = PDEConfig()
    assert cfg.grid == "uniform"
    assert cfg.barrier_alignment == "node"
    assert cfg.grid_points is None
    assert _build_grid(STRIKE, SPOT, cfg).uniform is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("grid", "log"),
        ("concentration", 0.0),
        ("concentration", -1.0),
        ("concentration", math.inf),
        ("grid_points", ()),
        ("grid_points", (0.0,)),
        ("barrier_alignment", "nearest"),
    ],
)
def test_the_new_fields_are_validated(field: str, value: object) -> None:
    call = EuropeanOption("call", STRIKE, EXPIRY)
    cfg = PDEConfig(**{field: value})  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError):
        price(call, _model(), _market(), method="pde", cfg=cfg)


# --------------------------------------------------------------------------
# (a) The stencil: pointwise order, and order under a smooth map.
# --------------------------------------------------------------------------


def _smooth(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`f`, `f'`, `f''` for `f(S) = exp(S / 80) sin(S / 40)`, all in closed form."""
    a, b = 1.0 / 80.0, 1.0 / 40.0
    e, sn, cs = np.exp(a * s), np.sin(b * s), np.cos(b * s)
    f = e * sn
    d1 = e * (a * sn + b * cs)
    d2 = e * ((a * a - b * b) * sn + 2.0 * a * b * cs)
    return f, d1, d2


def _alternating_grid(n: int) -> SpotGrid:
    """Spacings `h, 2h, h, 2h, ...` on `[50, 150]`: a grid from no smooth map.

    `h+ - h-` is `O(h)` at every interior node here rather than `O(h**2)`, so
    the second-derivative stencil's pointwise first-order term is not damped by
    anything. This is the control that makes the `sinh` measurement mean
    something.
    """
    pattern = np.tile([1.0, 2.0], n)
    s = 50.0 + np.concatenate([[0.0], np.cumsum(pattern)])
    s = 50.0 + (s - 50.0) * (100.0 / (s[-1] - 50.0))
    s.setflags(write=False)
    return SpotGrid(s=s, kind="sinh", uniform=False, ds=float(np.diff(s).mean()))


def _sinh_grid(n: int) -> SpotGrid:
    cfg = PDEConfig(
        n_s=n,
        n_t=10,
        grid="sinh",
        concentration=SINH_CONCENTRATION,
        strike_alignment="midpoint",
    )
    return build_spot_grid(
        strike=STRIKE, spot=SPOT, cfg=cfg, concentration_points=(STRIKE,)
    )


def test_second_derivative_stencil_is_only_first_order_on_a_non_smooth_grid() -> None:
    """Evidence class: NEGATIVE_FINDING, and the reason the mesh must be smooth.

    On spacings alternating `h, 2h` the pointwise error of the second-derivative
    stencil is `(h+ - h-) f'''/3 + O(h**2)` with `h+ - h- = O(h)`, so it is
    first order. Measured order 1.0000 (log-space residual 0.0000) over
    `n = 50, 100, 200, 400` half-periods, against the first-derivative
    stencil's 2.0000 on the same grids -- the two arms of the same three-point
    formula, one second order on any grid and one not.
    """
    first_errs, second_errs = [], []
    for n in LEVELS:
        grid = _alternating_grid(n)
        f, d1, d2 = _smooth(grid.s)
        delta, gamma = delta_gamma_nodes(grid, f)
        first_errs.append(float(np.max(np.abs(delta - d1[1:-1]))))
        second_errs.append(float(np.max(np.abs(gamma - d2[1:-1]))))

    h = tuple(1.0 / n for n in LEVELS)
    second = fit_convergence_order(h, second_errs)
    first = fit_convergence_order(h, first_errs)
    assert second.order == pytest.approx(1.0, abs=0.05)
    assert second.residual < 0.05
    assert first.order == pytest.approx(2.0, abs=0.05)
    assert first.residual < 0.05


def test_second_derivative_stencil_is_second_order_on_the_sinh_grid() -> None:
    """Evidence class: CONVERGENCE_ORDER, and the content of the mesh argument.

    The same stencil on a grid generated by a smooth map has
    `h+ - h- = g'' dxi**2 + O(dxi**4)`, so the pointwise first-order term is
    multiplied by a spacing that is itself shrinking and the result is second
    order. Measured order ~2 on both derivatives; the *pointwise* claim is the
    one that would fail if the mesh were built by an arbitrary rule rather than
    by a map.
    """
    first_errs, second_errs = [], []
    for n in LEVELS:
        grid = _sinh_grid(n)
        interior = (grid.s >= 40.0) & (grid.s <= 200.0)
        mask = interior[1:-1]
        f, d1, d2 = _smooth(grid.s)
        delta, gamma = delta_gamma_nodes(grid, f)
        first_errs.append(float(np.max(np.abs(delta - d1[1:-1])[mask])))
        second_errs.append(float(np.max(np.abs(gamma - d2[1:-1])[mask])))

    h = tuple(1.0 / n for n in LEVELS)
    for errs in (first_errs, second_errs):
        fit = fit_convergence_order(h, errs)
        assert fit.order == pytest.approx(2.0, abs=0.12), fit.order
        assert fit.residual < 0.05


def test_stencil_weights_are_exact_on_quadratics() -> None:
    """Evidence class: EXACT_IDENTITY -- a three-point stencil reproduces a
    quadratic's derivatives exactly on any grid, which is what makes it
    consistent at all."""
    grid = _sinh_grid(80)
    (w1m, w1_0, w1p), (w2m, w2_0, w2p) = stencil_weights(grid)
    s = grid.s
    for coeffs in ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, -3.0, 2.0)):
        a2, a1, a0 = coeffs
        v = a2 * s * s + a1 * s + a0
        delta = w1m * v[:-2] + w1_0 * v[1:-1] + w1p * v[2:]
        gamma = w2m * v[:-2] + w2_0 * v[1:-1] + w2p * v[2:]
        scale = max(abs(a2), abs(a1), 1.0) * float(np.max(np.abs(s)))
        assert np.max(np.abs(delta - (2.0 * a2 * s[1:-1] + a1))) < 1e-9 * scale
        assert np.max(np.abs(gamma - 2.0 * a2)) < 1e-9 * max(abs(a2), 1.0)


# --------------------------------------------------------------------------
# (a) Prices and Greeks on the sinh grid.
# --------------------------------------------------------------------------


def test_vanilla_price_is_second_order_on_the_sinh_grid() -> None:
    """Evidence class: CONVERGENCE_ORDER, against the Black-Scholes closed form.

    Measured order **1.9915** (log-space residual 0.0088) at
    `concentration = 0.05` over `n = 50 ... 800`, errors 6.910e-03, 1.768e-03,
    4.474e-04, 1.112e-04, 2.771e-05 -- against the uniform aligned grid's
    1.9973 and 8.027e-03 ... 3.229e-05 on the same ladder. The orders agree;
    the *constants* barely differ, which is the honest reading for a vanilla:
    concentrating nodes at the strike is worth about 15% here, not a factor.
    A barrier is where the mesh earns its keep (`tests/test_pde_barrier.py`).
    """
    call = EuropeanOption("call", STRIKE, EXPIRY)
    model, market = _model(), _market()
    exact = bs_price(
        S=SPOT, K=STRIKE, T=EXPIRY, r=RATE, sigma=SIGMA, q=DIVIDEND, kind="call"
    )
    errs = [
        abs(
            price(
                call,
                model,
                market,
                method="pde",
                cfg=_cfg(n, grid="sinh", concentration=SINH_CONCENTRATION),
            ).value
            - exact
        )
        for n in LEVELS
    ]
    fit = fit_convergence_order(H, errs)
    assert fit.order == pytest.approx(2.0, abs=0.1), fit.order
    assert fit.residual < 0.05


def test_digital_price_is_second_order_on_the_sinh_grid() -> None:
    """Evidence class: CONVERGENCE_ORDER, against the cash-or-nothing closed form.

    Measured **1.9907** (residual 0.0095) at `concentration = 0.05` over
    `n = 50 ... 800`. Slice 6 measured order 1.0012 for the same contract on an
    unaligned uniform grid; a jump costs a full order unless the grid does
    something about where it falls, and the `sinh` grid inherits the midpoint
    alignment that fixes it rather than replacing it.
    """
    digital = DigitalOption(kind="call", strike=STRIKE, expiry=EXPIRY, cash=1.0)
    model, market = _model(), _market()
    exact = analytic_digital(digital, model, market).value
    errs = [
        abs(
            price(
                digital,
                model,
                market,
                method="pde",
                cfg=_cfg(n, grid="sinh", concentration=SINH_CONCENTRATION),
            ).value
            - exact
        )
        for n in LEVELS
    ]
    fit = fit_convergence_order(H, errs)
    assert fit.order == pytest.approx(2.0, abs=0.1), fit.order
    assert fit.residual < 0.05


@pytest.mark.parametrize("greek", ["delta", "gamma"])
def test_vanilla_greeks_keep_order_two_on_the_sinh_grid(greek: str) -> None:
    """Evidence class: CONVERGENCE_ORDER, against the closed-form Greeks.

    This is the claim the non-uniform stencil had to earn: a Greek is a
    *difference* of the grid values, so it is where a first-order pointwise
    truncation error would show up first. Measured over `n = 50 ... 800` at
    `concentration = 0.05`: delta **1.9911** (errors 8.736e-04 down to
    3.509e-06), gamma **1.9924** (1.844e-05 down to 7.376e-08), against the
    uniform aligned grid's 2.0014 and 2.0668 with errors 2.6x and 5.8x larger
    at every level.
    """
    call = EuropeanOption("call", STRIKE, EXPIRY)
    model, market = _model(), _market()
    exact = getattr(analytic_greeks(call, model, market), greek)
    errs = [
        abs(
            getattr(
                greeks(
                    call,
                    model,
                    market,
                    method="pde",
                    cfg=_cfg(n, grid="sinh", concentration=SINH_CONCENTRATION),
                ),
                greek,
            )
            - exact
        )
        for n in LEVELS
    ]
    fit = fit_convergence_order(H, errs)
    assert fit.order == pytest.approx(2.0, abs=0.12), fit.order
    assert fit.residual < 0.05


# --------------------------------------------------------------------------
# Grid geometry: the anchors are exact, and the strike is a midpoint.
# --------------------------------------------------------------------------


def test_the_sinh_grid_puts_the_strike_at_a_cell_midpoint() -> None:
    """Evidence class: EXACT_IDENTITY, to the round-off of a bisection.

    The half-integer rule in `xi` misses the `S`-midpoint by `O(h**2)`, so the
    spacing is refined until the arithmetic midpoint of the straddling cell is
    the strike. What is asserted is the `S`-space condition, since that is the
    one the payoff symmetry argument needs.
    """
    for n in (61, 200):
        grid = _sinh_grid(n)
        index = int(np.searchsorted(grid.s, STRIKE, side="right")) - 1
        midpoint = 0.5 * (grid.s[index] + grid.s[index + 1])
        assert math.isclose(midpoint, STRIKE, rel_tol=1e-12)
        assert grid.meta["strike_cell_offset"] < 1e-12


def test_required_node_points_land_on_nodes_exactly() -> None:
    """Evidence class: EXACT_IDENTITY.

    An anchor is written back as the literal value it was asked for, so
    `grid.index_of(level)` can use `==`. Converging to the barrier is not the
    same as being on it, and the whole point of the mesh is the difference.
    """
    for kind, concentration in (("uniform", 0.05), ("sinh", 0.05)):
        cfg = PDEConfig(
            n_s=160,
            n_t=10,
            grid=kind,  # type: ignore[arg-type]
            concentration=concentration,
            strike_alignment="midpoint",
        )
        grid = build_spot_grid(
            strike=STRIKE,
            spot=SPOT,
            cfg=cfg,
            node_points=(95.0,),
            concentration_points=(STRIKE, 95.0),
        )
        index = grid.index_of(95.0)
        assert index > 0
        assert grid.s[index] == 95.0
        assert np.all(np.diff(grid.s) > 0.0)


def test_the_truncated_domain_starts_exactly_at_its_lower_bound() -> None:
    """A knock-out's domain begins at the barrier, so the barrier is node 0."""
    cfg = PDEConfig(n_s=120, n_t=10, grid="sinh", strike_alignment="midpoint")
    grid = build_spot_grid(
        strike=STRIKE,
        spot=SPOT,
        cfg=cfg,
        lower=95.0,
        concentration_points=(STRIKE, 95.0),
    )
    assert grid.s[0] == 95.0
    assert grid.s_max > STRIKE


def test_the_sinh_coordinate_and_its_inverse_agree() -> None:
    """Evidence class: CLOSED_FORM for one point, round-trip for the rest.

    With a single critical point the map inverts in closed form to
    `S = c + alpha sinh(xi)`, which is the published formula; the bisection is
    checked against it there and by round-trip with two points, where no closed
    form exists.
    """
    alpha = 15.0
    s = np.linspace(1.0, 400.0, 51)

    one = sinh_coordinate(s, (100.0,), alpha)
    assert np.allclose(100.0 + alpha * np.sinh(one), s, rtol=1e-12, atol=1e-9)

    two = sinh_coordinate(s, (95.0, 110.0), alpha)
    back = inverse_sinh_coordinate(
        two, (95.0, 110.0), alpha, lower=0.5, upper=500.0
    )
    assert np.allclose(back, s, rtol=1e-11, atol=1e-9)
    assert np.all(np.diff(two) > 0.0)
