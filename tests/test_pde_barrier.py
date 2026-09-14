"""The barrier on a grid: a boundary, a projection, and where each one fails.

Reference point throughout (the Slice 12 study point, zero rebate so that
in-out parity is exact): `S = K = 100`, `H = 95`, `r = 8%`, `q = 4%`,
`sigma = 25%`, `T = 0.5`, down-and-out call. Its continuously monitored
closed-form value is 4.5125986078 and is checked to 3e-14 against QuantLib in
`tests/oracle/test_barrier_vs_quantlib.py`, so it is used here as an exact
reference rather than as another engine's opinion.

What is measured, in order:

* **on a node** -- the continuous knock-out converges at order 2 when the
  barrier is the domain boundary;
* **off a node** -- and at order one, with an erratic constant, when it is
  rounded to the nodes instead (Zvan, Vetzal and Forsyth (2000));
* **the mesh** -- the `sinh` grid concentrated at the barrier and the strike
  has a 13x-18x smaller error constant at equal node count, with the gain
  measured across four concentrations rather than quoted at the best one;
* **discrete monitoring** -- the PDE agrees with the plain Monte Carlo
  estimator of the same contract within its standard error, and its
  discrete-to-continuous gap is order 1/2 in `1/m` and agrees with the
  Broadie-Glasserman-Kou barrier shift to better than 1%;
* **the projection** -- sampling the monitoring projection at nodes is first
  order and biased low; weighting it by the cell fraction is second order;
* **in-out parity** -- exact on the grid, to the round-off of the banded solve;
* **Greeks** -- delta and gamma from the non-uniform stencil against central
  differences of the closed form, and Slice 12's gamma sign flip reproduced.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.engines.analytic.barrier import (
    barrier_price,
    bgk_continuity_corrected_price,
    greeks_barrier as analytic_barrier_greeks,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.pde.barrier import KNOCK_IN, KNOCK_OUT, VANILLA_LEG, solve_leg
from qpl.engines.pde.pricers import PDEConfig
from qpl.exceptions import InvalidInputError
from qpl.instruments.options import BarrierOption, uniform_monitoring_times
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel, bs_price
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass, fit_convergence_order

SPOT = 100.0
STRIKE = 100.0
BARRIER = 95.0
EXPIRY = 0.5
RATE = 0.08
DIVIDEND = 0.04
SIGMA = 0.25

CONCENTRATION = 0.05
LEVELS = (100, 200, 400, 800)
H_LEVELS = tuple(1.0 / n for n in LEVELS)


def _market(spot: float = SPOT) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )


def _model() -> BlackScholesModel:
    return BlackScholesModel(sigma=SIGMA)


def _option(
    barrier_type: str = "down-and-out",
    *,
    monitoring: int | None = None,
    rebate: float = 0.0,
) -> BarrierOption:
    schedule = (
        "continuous"
        if monitoring is None
        else uniform_monitoring_times(EXPIRY, monitoring)
    )
    return BarrierOption(
        kind="call",
        strike=STRIKE,
        expiry=EXPIRY,
        barrier=BARRIER,
        barrier_type=barrier_type,  # type: ignore[arg-type]
        rebate=rebate,
        monitoring=schedule,
    )


def _closed_form(barrier_type: str = "down-and-out") -> float:
    return barrier_price(
        S=SPOT,
        K=STRIKE,
        T=EXPIRY,
        r=RATE,
        sigma=SIGMA,
        q=DIVIDEND,
        H=BARRIER,
        barrier_type=barrier_type,
        kind="call",
    )


def _cfg(n: int, **kwargs: object) -> PDEConfig:
    base: dict[str, object] = {
        "n_s": n,
        "n_t": n,
        "strike_alignment": "midpoint",
        "time_stepping": "rannacher",
    }
    base.update(kwargs)
    return PDEConfig(**base)  # type: ignore[arg-type]


def _pde(option: BarrierOption, cfg: PDEConfig) -> float:
    return price(option, _model(), _market(), method="pde", cfg=cfg).value


# --------------------------------------------------------------------------
# (b) The barrier on a node, and off it.
# --------------------------------------------------------------------------


def test_continuous_knock_out_is_second_order_with_the_barrier_on_a_node() -> None:
    """Evidence class: CONVERGENCE_ORDER, against the Reiner-Rubinstein form.

    The domain is truncated at the barrier, so the Dirichlet condition is
    imposed at `S = H` exactly and the scheme prices the contract it was given
    rather than a displaced one. Measured order **2.0668** (log-space residual
    0.0883) over `n_s = n_t = 100, 200, 400, 800` on the uniform grid, errors
    -4.676e-03, -8.861e-04, -2.568e-04, -5.959e-05 -- all of one sign, the
    scheme biased low, which is the truncation error of the far boundary and
    the time march rather than anything to do with the barrier.
    """
    option = _option()
    exact = _closed_form()
    errors = [_pde(option, _cfg(n)) - exact for n in LEVELS]

    fit = fit_convergence_order(H_LEVELS, [abs(e) for e in errors])
    assert fit.order == pytest.approx(2.0, abs=0.15), fit.order
    assert fit.residual < 0.15
    assert all(e < 0.0 for e in errors)


def test_the_barrier_is_reported_as_a_node_and_priced_exactly() -> None:
    """The grid says where the barrier landed, and it landed on it."""
    for kwargs in ({}, {"grid": "sinh", "concentration": CONCENTRATION}):
        result = price(
            _option(), _model(), _market(), method="pde", cfg=_cfg(200, **kwargs)
        )
        assert result.meta["barrier_on_node"] is True
        assert result.meta["domain_truncated_at_barrier"] is True
        assert result.meta["effective_barrier"] == BARRIER
        assert result.meta["effective_barrier_ratio"] == 1.0
        assert result.meta["s_min"] == BARRIER


def test_off_node_barrier_is_first_order_with_an_erratic_constant() -> None:
    """Evidence class: NEGATIVE_FINDING, with the numbers.

    `PDEConfig(barrier_alignment="none")` keeps the ordinary `[0, s_max]` grid
    and kills every node at or below the barrier, which is what a lattice does
    and what a grid does if nothing is said about placement. The scheme then
    prices the barrier at the **largest dead node**, which sits up to one
    spacing below `H`, and the price is locally linear in the barrier level,
    so the error is `O(ds)` -- with a constant set by the fractional part of
    `H / ds`, which is why it is erratic rather than monotone.

    Measured over `n_s = n_t = 100, 200, 400, 800`: errors +8.515e-01,
    +1.182e+00, +3.924e-01, +2.396e-01, a fit of **0.7079** with a log-space
    residual of **0.3068** (the diagnostic saying it is not a power law at
    fixed `n`), and `|error| * n` in **[85.2, 245.7]** across the ladder plus
    `n = 1600` -- bounded and not shrinking, which is the assumption-free form
    of "first order with an erratic constant". Against the aligned grid at the
    same node count the errors are **182x** to **4021x** larger.

    Every error is **positive**: the effective barrier is below the contract's,
    and a knock-out whose barrier is further away is worth more. That sign is
    the check that the mechanism is the displacement and not something else.

    Zvan, R., Vetzal, K. and Forsyth, P. (2000), "PDE methods for pricing
    barrier options", Journal of Economic Dynamics and Control 24, 1563-1590,
    is where the barrier-on-a-node requirement is stated; every number above is
    measured here.
    """
    option = _option()
    exact = _closed_form()

    errors, aligned_errors, effective = [], [], []
    for n in LEVELS:
        result = price(
            option,
            _model(),
            _market(),
            method="pde",
            cfg=_cfg(n, barrier_alignment="none"),
        )
        errors.append(result.value - exact)
        effective.append(float(result.meta["effective_barrier"]))
        aligned_errors.append(abs(_pde(option, _cfg(n)) - exact))
        assert result.meta["barrier_on_node"] is False
        assert result.meta["domain_truncated_at_barrier"] is False

    # The mechanism: a barrier strictly below the contract's, by less than one
    # spacing, and a price that is therefore too high.
    for value in effective:
        assert value < BARRIER
    assert all(e > 0.0 for e in errors)

    fit = fit_convergence_order(H_LEVELS, [abs(e) for e in errors])
    assert fit.order < 1.3
    assert fit.residual > 0.15  # not a power law, and the fit says so

    scaled = [abs(e) * n for e, n in zip(errors, LEVELS, strict=True)]
    assert 50.0 < min(scaled) and max(scaled) < 400.0

    ratios = [
        abs(e) / a for e, a in zip(errors, aligned_errors, strict=True)
    ]
    assert min(ratios) > 100.0


# --------------------------------------------------------------------------
# (c) What the mesh is worth, at equal node count.
# --------------------------------------------------------------------------


def test_the_sinh_mesh_shrinks_the_barrier_error_constant() -> None:
    """Evidence class: CONVERGENCE_ORDER plus a measured constant ratio.

    Same order, smaller constant: that is all a mesh can buy, and on a barrier
    it buys a lot. Uniform against `sinh` at `concentration = 0.05`, aligned
    and Rannacher, `n_s = n_t = n`:

        n        uniform        sinh       ratio
        100     4.676e-03    2.594e-04     18.0
        200     8.861e-04    6.838e-05     13.0
        400     2.568e-04    1.665e-05     15.4
        800     5.959e-05    4.108e-06     14.5

    and the fitted `sinh` order is 1.9981 (residual 0.0195) against the uniform
    grid's 2.0668.

    The concentration scan at `n = 100`, reported rather than tuned away:

        concentration   0.02      0.05      0.10      0.20
        error        2.114e-04 2.594e-04 4.355e-04 9.397e-04
        ratio           22.1      18.0      10.7       5.0

    So the *default* `concentration = 0.05` is not the best cell on this
    contract -- 0.02 is 1.2x better here -- and it is left where it is because
    on a plain vanilla the ranking reverses (6.910e-03 at 0.05 against
    1.099e-02 at 0.02, `n = 50`; `tests/test_pde_nonuniform_grid.py`). A mesh
    that starves the tails to feed the barrier is not free.
    """
    option = _option()
    exact = _closed_form()

    uniform = [abs(_pde(option, _cfg(n)) - exact) for n in LEVELS]
    concentrated = [
        abs(_pde(option, _cfg(n, grid="sinh", concentration=CONCENTRATION)) - exact)
        for n in LEVELS
    ]

    fit = fit_convergence_order(H_LEVELS, concentrated)
    assert fit.order == pytest.approx(2.0, abs=0.15), fit.order
    assert fit.residual < 0.1

    ratios = [u / c for u, c in zip(uniform, concentrated, strict=True)]
    assert min(ratios) > 8.0, ratios
    assert max(ratios) < 30.0, ratios

    # The scan, run so that the number in the docstring is not a memory.
    scan = {
        concentration: abs(
            _pde(option, _cfg(100, grid="sinh", concentration=concentration)) - exact
        )
        for concentration in (0.02, 0.05, 0.10, 0.20)
    }
    assert scan[0.02] < scan[0.05] < scan[0.10] < scan[0.20]
    assert scan[0.20] < uniform[0]  # even the worst concentration beats uniform


# --------------------------------------------------------------------------
# (e) In-out parity, on one grid.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grid_kwargs",
    [{}, {"grid": "sinh", "concentration": CONCENTRATION}],
    ids=["uniform", "sinh"],
)
@pytest.mark.parametrize("n", [100, 400])
def test_in_out_parity_holds_on_the_grid_to_solver_round_off(
    grid_kwargs: dict[str, object], n: int
) -> None:
    """Evidence class: EXACT_IDENTITY, on the discrete system rather than the model.

    The knock-out, the knock-in and the summed-data leg are three right-hand
    sides of the **same** matrix on the **same** grid, so their solutions add
    exactly. This is a stronger statement than "in + out = vanilla in the
    limit": it holds at `n = 100`, where the knock-in is still 3.96e-03 away
    from its own closed form.

    Measured relative residuals: 1.13e-16, 3.40e-16 (uniform, `n = 100, 400`)
    and 3.40e-16, 1.13e-15 (`sinh`) -- a handful of ulps of the ~7.85 leg
    value, i.e. the banded solve's round-off and nothing else.
    """
    option = _option()
    model, market, cfg = _model(), _market(), _cfg(n, **grid_kwargs)

    knock_out = solve_leg(option, model, market, cfg, leg=KNOCK_OUT).price
    knock_in = solve_leg(option, model, market, cfg, leg=KNOCK_IN).price
    combined = solve_leg(option, model, market, cfg, leg=VANILLA_LEG).price

    residual = knock_out + knock_in - combined
    assert abs(residual) <= 1e-13 * abs(combined)

    # The summed leg is the vanilla solved on the truncated domain with the
    # exact Black-Scholes value imposed at the barrier, so it had better be the
    # vanilla to the grid's own accuracy.
    vanilla = bs_price(
        S=SPOT, K=STRIKE, T=EXPIRY, r=RATE, sigma=SIGMA, q=DIVIDEND, kind="call"
    )
    assert combined == pytest.approx(vanilla, abs=1e-3)


def test_the_knock_in_converges_to_its_own_closed_form() -> None:
    """Evidence class: CONVERGENCE_ORDER -- the direct formulation is right.

    The knock-in is solved as its own boundary problem: terminal data the
    rebate (zero here), the Black-Scholes value of the vanilla imposed at the
    barrier, and `rebate e^{-r tau}` at the far boundary. With a zero rebate
    the terminal data is identically zero and the whole value arrives through
    the barrier, which is a sharp test of that boundary condition. Measured on
    the `sinh` grid: errors -5.911e-05 (`n = 100`) and -3.986e-06 (`n = 400`),
    fitted order ~2.
    """
    option = _option("down-and-in")
    exact = _closed_form("down-and-in")
    errors = [
        abs(
            _pde(option, _cfg(n, grid="sinh", concentration=CONCENTRATION)) - exact
        )
        for n in LEVELS
    ]
    fit = fit_convergence_order(H_LEVELS, errors)
    assert fit.order == pytest.approx(2.0, abs=0.2), fit.order
    assert errors[-1] < 1e-5


def test_an_up_and_out_call_prices_on_the_mirrored_domain() -> None:
    """The up barrier is the same construction reflected, and is checked as one.

    Its domain is `[0, H]` with **both** ends pinned, so the strike cannot also
    be nudged to a cell midpoint; the realised offset is reported rather than
    silently accepted, and the price still converges.
    """
    option = BarrierOption(
        kind="call",
        strike=100.0,
        expiry=1.0,
        barrier=120.0,
        barrier_type="up-and-out",
    )
    market = Market(
        spot=100.0,
        rate_curve=FlatRateCurve(0.05),
        dividend_curve=FlatDividendCurve(0.0),
    )
    model = BlackScholesModel(sigma=0.20)
    exact = barrier_price(
        S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0, H=120.0,
        barrier_type="up-and-out", kind="call",
    )
    errors = []
    for n in (100, 200, 400):
        result = price(option, model, market, method="pde", cfg=_cfg(n))
        assert result.meta["effective_barrier"] == 120.0
        assert result.meta["s_max"] == 120.0
        assert result.meta["strike_cell_offset"] >= 0.0
        errors.append(abs(result.value - exact))
    fit = fit_convergence_order([1 / n for n in (100, 200, 400)], errors)
    assert fit.order > 1.0, fit.order
    assert errors[-1] < 5e-4


# --------------------------------------------------------------------------
# (d) Discrete monitoring.
# --------------------------------------------------------------------------

DISCRETE_CFG = {"grid": "sinh", "concentration": CONCENTRATION}
MC_PATHS = 200_000
MC_SEED = 20_260_913
MC_STDERR_MULTIPLE = 3.5


@pytest.mark.parametrize("n_monitoring", [20, 80])
def test_discrete_pde_agrees_with_the_plain_monte_carlo_estimator(
    n_monitoring: int,
) -> None:
    """Evidence class: STATISTICAL, against an unbiased estimator of the same contract.

    `MCConfig(barrier_correction='none')` observes the barrier on exactly the
    contract's schedule and is unbiased for the **discrete** contract, so the
    comparison is of two routes to one number rather than of two contracts.
    Measured `z` at 400 000 paths and the study seed: -0.57, -2.31, -1.15,
    -1.25, -0.32 over `m = 10, 20, 40, 80, 160`. The budget here is 3.5 of the
    simulation's own standard error, which is a statistical statement and not
    an accuracy claim.
    """
    option = _option(monitoring=n_monitoring)
    model, market = _model(), _market()

    grid_value = _pde(option, _cfg(400, **DISCRETE_CFG))
    simulation = price(
        option,
        model,
        market,
        method="mc",
        cfg=MCConfig(
            n_paths=MC_PATHS,
            seed=MC_SEED,
            variance_reduction=("antithetic", "control_variate"),
        ),
    )
    assert simulation.meta["estimates"] == "discrete"
    assert grid_value == pytest.approx(
        simulation.value, abs=MC_STDERR_MULTIPLE * simulation.stderr
    )


def test_the_discrete_gap_is_order_one_half_and_matches_the_bgk_shift() -> None:
    """Evidence class: CONVERGENCE_ORDER, cross-checked against a published shift.

    The knock-out is worth more when it is observed less often, and the excess
    over the continuous price decays like `m**-1/2`. Measured **0.4431**
    (log-space residual 0.0101) over `m = 10, 20, 40, 80, 160` on the `sinh`
    grid at `n_s = n_t = 400`, gaps +1.625746, +1.213371, +0.900876, +0.658822,
    +0.475119.

    That it is below 0.5 is not noise and not a grid artefact: the
    Broadie-Glasserman-Kou barrier shift, evaluated in closed form on the same
    `m` ladder, has fitted order **0.4361** -- the same `o(1/sqrt(m))`
    contamination at finite `m` that Slice 12 measured from the simulation
    side (0.4603 +- 0.0125 paired). The two orders agree to 0.007.

    Ratio of the measured gap to the BGK-predicted gap: 1.0147, 0.9981, 0.9974,
    0.9963, 0.9915. So the continuity correction predicts the grid's own
    discrete-versus-continuous difference to better than 1.5% at `m = 10` and
    better than 0.4% from `m = 20` on -- a genuinely independent check, since
    nothing in this engine knows about `beta = -zeta(1/2)/sqrt(2 pi)`.
    """
    exact = _closed_form()
    ladder = (10, 20, 40, 80, 160)

    gaps, predicted = [], []
    for n_monitoring in ladder:
        option = _option(monitoring=n_monitoring)
        gaps.append(_pde(option, _cfg(400, **DISCRETE_CFG)) - exact)
        predicted.append(
            bgk_continuity_corrected_price(
                S=SPOT, K=STRIKE, T=EXPIRY, r=RATE, sigma=SIGMA, q=DIVIDEND,
                H=BARRIER, barrier_type="down-and-out", kind="call",
                n_monitoring=n_monitoring,
            )
            - exact
        )

    assert all(gap > 0.0 for gap in gaps)
    h = tuple(1.0 / n for n in ladder)
    fit = fit_convergence_order(h, gaps)
    bgk_fit = fit_convergence_order(h, predicted)

    assert fit.order == pytest.approx(0.5, abs=0.1), fit.order
    assert fit.residual < 0.05
    # The deviation from one half is the correction's, not the grid's.
    assert fit.order == pytest.approx(bgk_fit.order, abs=0.05)

    ratios = [g / p for g, p in zip(gaps, predicted, strict=True)]
    assert all(abs(ratio - 1.0) < 0.02 for ratio in ratios), ratios


def test_the_monitoring_projection_must_be_cell_weighted() -> None:
    """Evidence class: NEGATIVE_FINDING -- the digital pathology, once per date.

    Observing the barrier turns the value function into `V * 1{S > H}`, which
    is **discontinuous**, and Slice 6 measured what a discontinuity costs a
    grid that samples it at nodes: a full order. Here it is worse than usual,
    because on a grid where `H` is a node the node's own cell is half alive and
    the node is nonetheless set to the rebate -- so the scheme throws away half
    a cell of value at every monitoring date and is biased **low**.

    Measured on exactly the same grid (`s_max = 380`, `n_s = n_t = n`, so
    `H = 95` is node `n/4` for every `n`), `m = 10`, against a converged
    reference:

        n        node-sampled     cell-weighted
        100      -6.339e-01        +3.639e-02
        200      -3.365e-01        +7.067e-03
        400      -1.711e-01        +1.304e-03
        800      -8.604e-02        +3.261e-04

    fitted orders **0.9619** (residual 0.0140) and **2.2844** (residual
    0.0748). Every node-sampled error is negative, which is the half-cell
    argument's signature.

    The weighting is the `L2` projection: each node takes the average of the
    post-observation function over its own cell, with the live part's average
    read off the pre-observation `V` -- which is smooth across `H`, the
    previous jump having diffused away -- by linear interpolation. It needs no
    node placement at all, which is the point.
    """
    option = _option(monitoring=10)
    reference = _pde(option, _cfg(600, **DISCRETE_CFG))

    levels = (100, 200, 400, 800)
    sampled, weighted = [], []
    for n in levels:
        common = {"s_max": 380.0, "strike_alignment": "none"}
        sampled.append(
            _pde(option, _cfg(n, barrier_alignment="none", **common)) - reference
        )
        weighted.append(
            _pde(option, _cfg(n, barrier_alignment="node", **common)) - reference
        )

    assert all(e < 0.0 for e in sampled)
    h = tuple(1.0 / n for n in levels)
    sampled_fit = fit_convergence_order(h, [abs(e) for e in sampled])
    weighted_fit = fit_convergence_order(h, [abs(e) for e in weighted])

    assert sampled_fit.order == pytest.approx(1.0, abs=0.15), sampled_fit.order
    assert weighted_fit.order > 1.8, weighted_fit.order
    assert abs(weighted[-1]) * 100.0 < abs(sampled[-1])


def test_the_time_grid_contains_every_monitoring_date() -> None:
    """The construction, asserted rather than described.

    The levels are the union of the `n_t` uniform ones and the `m` monitoring
    ones, so a projection lands on a time level and is never interpolated onto
    one. A schedule whose dates are **not** a sub-multiple of `n_t` is the case
    that would expose a merge that silently dropped one.
    """
    for n_monitoring, n_t in ((7, 20), (10, 40), (13, 13)):
        option = _option(monitoring=n_monitoring)
        result = price(
            option,
            _model(),
            _market(),
            method="pde",
            cfg=PDEConfig(
                n_s=100,
                n_t=n_t,
                strike_alignment="midpoint",
                time_stepping="rannacher",
            ),
        )
        assert result.meta["n_projections"] == n_monitoring
        assert result.meta["monitoring"] == "discrete"
        assert result.meta["barrier_projection"] == "cell_weighted"
        # Rannacher splits the first two intervals, so two extra steps.
        assert result.meta["n_steps_taken"] >= n_t + 2


# --------------------------------------------------------------------------
# (f) Greeks.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spot", [110.0, 105.0, 100.0])
def test_grid_greeks_match_the_closed_form_away_from_the_barrier(spot: float) -> None:
    """Evidence class: CLOSED_FORM, against central differences of the formula.

    The reference is `qpl.engines.analytic.barrier.greeks_barrier`, itself
    central differences of a closed form checked to 3e-14 against QuantLib, so
    the comparison inherits that formula's accuracy and the difference
    quotient's `O(h^2)`.

    Measured at `n_s = n_t = 400`, aligned, Rannacher, on the `sinh` grid:

        S       delta residual   gamma residual
        110      +2.83e-06        -1.51e-07
        105      +3.87e-06        -1.14e-07
        100      +5.66e-06        -2.60e-07

    and on the uniform grid one to two decimal orders worse (+9.62e-05,
    +6.43e-05, +1.82e-04 in delta). Fitted orders at the reference spot:
    delta 1.9980, gamma 1.9558 on the `sinh` grid; 2.0830 and 2.0833 on the
    uniform one, with errors 39x and 41x larger at `n = 800`.
    """
    option = _option()
    model, market = _model(), _market(spot)
    reference = analytic_barrier_greeks(option, model, market)

    grid = greeks(
        option,
        model,
        market,
        method="pde",
        cfg=_cfg(400, grid="sinh", concentration=CONCENTRATION),
    )
    assert grid.delta == pytest.approx(reference.delta, abs=5e-5)
    assert grid.gamma == pytest.approx(reference.gamma, abs=5e-6)
    assert math.isfinite(grid.theta)
    assert grid.meta["greeks_method"] == "grid"


@pytest.mark.parametrize("greek", ["delta", "gamma"])
def test_barrier_grid_greeks_are_second_order(greek: str) -> None:
    """Evidence class: CONVERGENCE_ORDER, on the non-uniform stencil's output."""
    option = _option()
    model, market = _model(), _market()
    exact = getattr(analytic_barrier_greeks(option, model, market), greek)

    errors = [
        abs(
            getattr(
                greeks(
                    option,
                    model,
                    market,
                    method="pde",
                    cfg=_cfg(n, grid="sinh", concentration=CONCENTRATION),
                ),
                greek,
            )
            - exact
        )
        for n in (100, 200, 400)
    ]
    fit = fit_convergence_order([1 / n for n in (100, 200, 400)], errors)
    assert fit.order == pytest.approx(2.0, abs=0.15), fit.order
    assert fit.residual < 0.05


@pytest.mark.slow
def test_gamma_flips_sign_approaching_the_barrier() -> None:
    """Evidence class: NEGATIVE_FINDING -- Slice 12's finding, reproduced on a grid.

    Slice 12's slice statement said to document gamma blowing up as `S -> H`.
    It does not: the continuously monitored value is smooth at the barrier for
    any `T > 0`, and what the barrier does to the shape is flip the **sign** of
    gamma -- a knock-out near its barrier is being pinned toward the rebate
    from one side rather than bending away from the strike, so it is concave
    where the vanilla is convex. The singularity is at the corner
    `(S = H, t = T)`.

    Measured off the grid at `n_s = n_t = 400`, `sinh`, with the closed form's
    own values in brackets:

        S        110       105       100        98        96      95.5
        gamma +0.002874 +0.000322 -0.004628 -0.007407 -0.010671 -0.011563
        (cf)  +0.002874 +0.000322 -0.004628 -0.007407 -0.010671 -0.011562

    The sign change is between `S = 105` and `S = 100`, and delta stays finite
    and increasing all the way down (0.8842, 0.8754, 0.8850, 0.8970, 0.9150,
    0.9205), which is the other half of "no blow-up".
    """
    option = _option()
    model = _model()
    spots = (110.0, 105.0, 100.0, 98.0, 96.0, 95.5)

    gammas, deltas = [], []
    for spot in spots:
        grid = greeks(
            option,
            model,
            _market(spot),
            method="pde",
            cfg=_cfg(400, grid="sinh", concentration=CONCENTRATION),
        )
        gammas.append(grid.gamma)
        deltas.append(grid.delta)

    assert gammas[0] > 0.0 and gammas[1] > 0.0
    assert all(g < 0.0 for g in gammas[2:])
    # Finite, and monotonically more negative rather than diverging.
    assert gammas[-1] > -0.02
    assert all(math.isfinite(d) for d in deltas)
    assert deltas[-1] > deltas[0]

    # The vanilla is convex at the same point, which is the contrast that makes
    # the sign meaningful.
    from qpl.engines.analytic.black_scholes import greeks_european
    from qpl.instruments.options import EuropeanOption

    vanilla = greeks_european(
        EuropeanOption("call", STRIKE, EXPIRY), model, _market(96.0)
    )
    assert vanilla.gamma > 0.0


# --------------------------------------------------------------------------
# Contract-level behaviour.
# --------------------------------------------------------------------------


def test_a_touched_spot_settles_at_inception_like_every_other_engine() -> None:
    """A fact about the contract, so the grid must return the same number."""
    model = _model()
    market = _market(BARRIER - 1.0)

    knock_out = price(
        _option(rebate=3.0), model, market, method="pde", cfg=_cfg(100)
    )
    assert knock_out.value == 3.0
    assert knock_out.meta["degenerate"] == "touched_at_inception"

    knock_in = price(
        _option("down-and-in"), model, market, method="pde", cfg=_cfg(100)
    )
    vanilla = bs_price(
        S=market.spot, K=STRIKE, T=EXPIRY, r=RATE, sigma=SIGMA, q=DIVIDEND,
        kind="call",
    )
    assert knock_in.value == pytest.approx(vanilla, rel=1e-13)

    with pytest.raises(InvalidInputError, match="already touched"):
        greeks(_option(), model, market, method="pde", cfg=_cfg(100))


@pytest.mark.parametrize("monitoring", [None, 8])
def test_degenerate_limits_delegate_to_the_model(monitoring: int | None) -> None:
    """`T = 0` and `sigma = 0` are statements about the model, not the grid."""
    market = _market()
    zero_vol = BlackScholesModel(sigma=0.0)
    option = _option(monitoring=monitoring)

    result = price(option, zero_vol, market, method="pde", cfg=_cfg(100))
    assert result.meta["degenerate"] == "zero_vol"
    forward = SPOT * math.exp((RATE - DIVIDEND) * EXPIRY)
    # The forward rises away from the down barrier, so the contract survives.
    assert result.value == pytest.approx(
        math.exp(-RATE * EXPIRY) * max(forward - STRIKE, 0.0), rel=1e-12
    )

    at_expiry = BarrierOption(
        kind="call",
        strike=STRIKE,
        expiry=0.0,
        barrier=BARRIER,
        barrier_type="down-and-out",
    )
    assert (
        price(at_expiry, _model(), market, method="pde", cfg=_cfg(100)).value
        == pytest.approx(max(SPOT - STRIKE, 0.0), abs=0.0)
    )


def test_the_evidence_classes_used_here_exist() -> None:
    """The claims above are labelled; this pins the labels as real."""
    for name in (
        "CONVERGENCE_ORDER",
        "NEGATIVE_FINDING",
        "STATISTICAL",
        "EXACT_IDENTITY",
        "CLOSED_FORM",
    ):
        assert isinstance(getattr(EvidenceClass, name), EvidenceClass)


def test_the_pde_barrier_meta_describes_the_discretisation() -> None:
    """Whatever else changes, the result says what it did."""
    result = price(
        _option(monitoring=12),
        _model(),
        _market(),
        method="pde",
        cfg=_cfg(200, grid="sinh", concentration=CONCENTRATION),
    )
    meta = result.meta
    assert meta["method"] == "pde"
    assert meta["instrument"] == "barrier"
    assert meta["leg"] == KNOCK_OUT
    assert meta["grid"] == "sinh"
    assert meta["monitoring"] == "discrete"
    assert meta["n_monitoring"] == 12
    assert meta["barrier"] == BARRIER
    assert meta["concentration"] == CONCENTRATION
    assert set(meta["concentration_points"]) == {STRIKE, BARRIER}
    assert meta["barrier_node_index"] > 0
    assert np.isfinite(meta["ds_min"]) and meta["ds_min"] < meta["ds_max"]
