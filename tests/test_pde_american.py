"""The American PDE engine: the solver, the LCP, the identities, the limits.

Slice 5, items (a), (c), (f) and (g). The measured convergence order, the
boundary comparison and the iteration-growth study are the separate file
`tests/test_pde_american_convergence.py`; the QuantLib leg is
`tests/oracle/test_pde_american_vs_quantlib.py`.

Evidence classes used here:

- EXACT_IDENTITY for the structural facts -- `V >= g` holds bit-for-bit
  because the projection assigns the obstacle value itself; the no-dividend
  American call is the European call once the constraint is inactive; the
  registry now resolves `AmericanOption` + `method="pde"`.
- EXACT_IDENTITY *up to solver tolerance* for the complementarity check and
  the unprojected-equals-direct-solve check, where the permitted residual is
  derived from `PSORConfig.tol` and the measured accumulation, not chosen.
- CLOSED_FORM for the `sigma = 0` and `T = 0` limits, which are re-derived here
  by brute-force maximisation over the deterministic forward path rather than
  by comparing against the lattice engine -- the two engines *share* that code,
  so agreeing with it would be checking nothing.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from qpl.engines.pde.american import (
    PSORConfig,
    _psor_sweeps,
    _solve_grid_american,
)
from qpl.engines.pde.pricers import (
    PDEConfig,
    _build_grid,
    _dirichlet,
    _operator,
    _payoff,
    _solve_grid,
    _solve_tridiagonal,
)
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import AmericanOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.numerics.linear_systems import sor_solve
from qpl.pricing import greeks, price

_REFERENCE = dict(kind="put", spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div=0.0, sigma=0.20)
"""The ATM American put the whole curriculum uses as its reference point."""


def _market(spot: float, rate: float, div: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(rate, allow_negative=True),
        dividend_curve=FlatDividendCurve(div, allow_negative=True),
    )


def _cfg(n: int, **overrides) -> PDEConfig:
    """The grid this file works on: aligned, Rannacher, square.

    Alignment and damping are not the subject of any test here -- Slice 4
    measured both -- but they are the settings a caller should use, so the
    American engine is exercised on them rather than on the bare defaults.
    """
    kwargs = dict(
        n_s=n, n_t=n, strike_alignment="midpoint", time_stepping="rannacher"
    )
    kwargs.update(overrides)
    return PDEConfig(**kwargs)  # type: ignore[arg-type]


def _american(cfg: PDEConfig, **spec):
    merged = {**_REFERENCE, **spec}
    return price(
        AmericanOption(
            kind=merged["kind"], strike=merged["strike"], expiry=merged["expiry"]
        ),
        BlackScholesModel(sigma=merged["sigma"]),
        _market(merged["spot"], merged["rate"], merged["div"]),
        method="pde",
        cfg=cfg,
    )


def _grids(cfg: PDEConfig, **spec):
    """American and European grid solutions on the *same* grid."""
    merged = {**_REFERENCE, **spec}
    model = BlackScholesModel(sigma=merged["sigma"])
    market = _market(merged["spot"], merged["rate"], merged["div"])
    kwargs = dict(kind=merged["kind"], strike=merged["strike"], expiry=merged["expiry"])
    american = _solve_grid_american(AmericanOption(**kwargs), model, market, cfg)
    european = _solve_grid(EuropeanOption(**kwargs), model, market, cfg)
    return american, european


# --------------------------------------------------------------------------
# The registry: `AmericanOption` + `method="pde"` resolves now
# --------------------------------------------------------------------------


def test_the_dispatcher_prices_american_options_by_pde() -> None:
    """Evidence class: EXACT_IDENTITY (an API contract).

    Until Slice 5 this raised `NotSupportedError` out of the registry lookup,
    and `tests/test_tree_american.py` asserted that it did. The replacement
    assertion is here: both `price` and `greeks` resolve, and the American
    answer is *not* the European one -- a registration that silently fell back
    to the European engine would pass a "does not raise" test.
    """
    cfg = _cfg(200)
    american = _american(cfg)
    european = price(
        EuropeanOption(kind="put", strike=100.0, expiry=1.0),
        BlackScholesModel(sigma=0.20),
        _market(100.0, 0.05, 0.0),
        method="pde",
        cfg=cfg,
    )
    assert american.value > european.value + 0.4
    assert (american.meta or {})["exercise"] == "american"
    assert (american.meta or {})["scheme"] == "psor"

    greek = greeks(
        AmericanOption(kind="put", strike=100.0, expiry=1.0),
        BlackScholesModel(sigma=0.20),
        _market(100.0, 0.05, 0.0),
        method="pde",
        cfg=cfg,
    )
    assert -1.0 < greek.delta < 0.0
    assert greek.gamma > 0.0
    # Monte Carlo and the closed form still refuse; only the tree and the PDE
    # are registered for early exercise.
    with pytest.raises(NotSupportedError, match="Unsupported instrument"):
        price(
            AmericanOption(kind="put", strike=100.0, expiry=1.0),
            BlackScholesModel(sigma=0.20),
            _market(100.0, 0.05, 0.0),
            method="analytic",
        )


# --------------------------------------------------------------------------
# (a) The solver: validation, the stopping criterion, and the iteration cap
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("psor", "message"),
    [
        (PSORConfig(omega=0.0), "omega must satisfy 0 < omega < 2"),
        (PSORConfig(omega=2.0), "omega must satisfy 0 < omega < 2"),
        (PSORConfig(omega=math.nan), "omega must satisfy 0 < omega < 2"),
        (PSORConfig(tol=0.0), "tol must be > 0"),
        (PSORConfig(tol=-1e-8), "tol must be > 0"),
        (PSORConfig(max_iter=0), "max_iter must be >= 1"),
        (PSORConfig(on_max_iter="ignore"), "on_max_iter must be 'raise' or 'flag'"),
    ],
)
def test_psor_config_is_validated(psor: PSORConfig, message: str) -> None:
    """`0 < omega < 2` is Kahan's necessary condition for SOR, not a taste.

    Outside it the iteration matrix has spectral radius at least
    `|omega - 1| >= 1`. The message is the one
    `qpl.numerics.linear_systems.sor_solve` already uses, so the repository's
    two SOR implementations refuse the same input the same way.
    """
    with pytest.raises(InvalidInputError, match=message):
        _american(_cfg(20, psor=psor))


def test_the_iteration_cap_raises_rather_than_returning_a_half_solve() -> None:
    """Evidence class: EXACT_IDENTITY (a contract about failure).

    An LCP solve that ran out of sweeps is not a price, and the default
    behaviour says so. `max_iter=1` cannot meet `tol=1e-8` on any grid here, so
    the first time step raises and names itself.
    """
    with pytest.raises(InvalidInputError, match=r"PSOR did not converge within max_iter=1\b"):
        _american(_cfg(50, psor=PSORConfig(max_iter=1)))


def test_the_iteration_cap_can_be_flagged_instead_of_raised() -> None:
    """The alternative is loud in `meta`, never silent.

    With `on_max_iter="flag"` the march continues and every step that hit the
    cap is listed. The returned number is then measurably wrong -- asserted
    here, so that "flag" cannot be mistaken for "it was fine anyway".
    """
    flagged = _american(_cfg(50, psor=PSORConfig(max_iter=2, on_max_iter="flag")))
    meta = flagged.meta or {}
    assert meta["psor_converged"] is False
    assert len(meta["psor_unconverged_steps"]) == meta["n_steps_taken"]
    assert np.all(np.asarray(meta["psor_iterations"]) == 2)

    good = _american(_cfg(50))
    assert (good.meta or {})["psor_converged"] is True
    assert abs(flagged.value - good.value) > 1e-3


def _one_step_system(n: int) -> tuple[np.ndarray, ...]:
    """A Crank-Nicolson step taken from the engine's own assembly.

    Built out of `_build_grid`, `_payoff`, `_operator` and `_dirichlet`, i.e.
    the same four calls `_solve_grid_american` makes, so the system below is a
    real one and not a hand-written stand-in.
    """
    cfg = _cfg(n)
    grid = _build_grid(100.0, 100.0, cfg)
    s_grid, s_max = grid.s, grid.s_max
    v = _payoff("put", 100.0, s_grid)
    dt = 1.0 / n
    a, b, c = _operator(grid, 0.20, 0.05, 0.0)
    lower = -0.5 * dt * a
    diag = 1.0 - 0.5 * dt * b
    upper = -0.5 * dt * c
    rhs = (1.0 + 0.5 * dt * b) * v[1:-1] + 0.5 * dt * (a * v[:-2] + c * v[2:])
    v0, vmax = _dirichlet("put", 100.0, s_max, math.exp(-0.05 * dt), 1.0)
    rhs[0] -= lower[0] * v0
    rhs[-1] -= upper[-1] * vmax
    lower[0] = 0.0
    upper[-1] = 0.0
    return lower, diag, upper, rhs, v[1:-1], v[1:-1].copy()


@pytest.mark.parametrize("omega", [1.0, 1.2, 1.5])
@pytest.mark.parametrize("n", [20, 50, 100])
def test_unprojected_psor_reproduces_the_direct_solve_and_sor_solve(
    n: int, omega: float
) -> None:
    """Evidence class: EXACT_IDENTITY up to solver tolerance. Slice item (a).

    Three solvers, one system, one answer:

    - `_solve_tridiagonal`, the European engine's LAPACK banded solve;
    - `_psor_sweeps` with the obstacle at `-inf`, i.e. plain red-black SOR;
    - `qpl.numerics.linear_systems.sor_solve`, the in-repo *natural-order*,
      dense SOR from the Chapter 5 lab.

    The last of these is the reason the projection is written here rather than
    bolted onto `sor_solve`: it is used as the independent reference for the
    ordering. Red-black and natural order are different iterations -- the
    measured sweep counts differ, 21 against 19 at `omega = 1.2` -- and a
    tridiagonal matrix is consistently ordered in Young's sense under both, so
    they share a spectral radius and a limit. Measured agreement at
    `tol = 1e-12`, worst node over the nine cases: **3.2e-13** against the
    banded solve and 2.7e-13 against `sor_solve`, on values up to about 92 --
    3.5e-15 relative, i.e. round-off.

    `tol` is 1e-12 rather than 1e-14 deliberately. The max-update criterion is
    absolute, so it cannot be pushed below the round-off scale of the values
    themselves: at the fixed point the Gauss-Seidel value differs from the
    iterate by about `eps * |x| ~ 1e-12` here, the relaxation multiplies that
    by `omega`, and at `tol = 1e-14` the `omega = 1.5` sweep never terminates
    (measured: 100000 sweeps, already at the answer to 1.4e-14).
    """
    lower, diag, upper, rhs, x0, x = _one_step_system(n)
    direct = _solve_tridiagonal(lower, diag, upper, rhs)

    obstacle = np.full(diag.size, -np.inf)
    sweeps, _ = _psor_sweeps(
        x, lower, diag, upper, rhs, obstacle, omega=omega, tol=1e-12, max_iter=100_000
    )
    dense = np.diag(diag) + np.diag(upper[:-1], 1) + np.diag(lower[1:], -1)
    natural = sor_solve(dense, rhs, omega=omega, x0=x0.copy(), tol=1e-12, max_iter=100_000)

    assert sweeps < 100_000
    assert np.max(np.abs(x - direct)) < 1e-12
    assert np.max(np.abs(x - natural.x)) < 1e-12
    assert natural.converged


@pytest.mark.parametrize("n", [50, 100, 200])
def test_projected_psor_matches_a_natural_order_psor_written_out_here(n: int) -> None:
    """Evidence class: INDEPENDENT_ENGINE for the *ordering*, on the real LCP.

    The test above removes the constraint; this one keeps it and replaces the
    red-black sweep with a plain lexicographic one written as a Python loop --
    the textbook PSOR, slow and obviously correct. Agreement to 1e-11 on every
    node says the vectorised colouring is a reordering of the same iteration
    and not a different algorithm.
    """
    lower, diag, upper, rhs, x0, x = _one_step_system(n)
    cfg = _cfg(n)
    s_grid = _build_grid(100.0, 100.0, cfg).s
    obstacle = _payoff("put", 100.0, s_grid)[1:-1]

    _psor_sweeps(
        x, lower, diag, upper, rhs, obstacle, omega=1.2, tol=1e-13, max_iter=100_000
    )

    naive = x0.copy()
    for _ in range(100_000):
        update = 0.0
        for i in range(naive.size):
            left = naive[i - 1] if i > 0 else 0.0
            right = naive[i + 1] if i + 1 < naive.size else 0.0
            gauss_seidel = (rhs[i] - lower[i] * left - upper[i] * right) / diag[i]
            new = max(obstacle[i], naive[i] + 1.2 * (gauss_seidel - naive[i]))
            update = max(update, abs(new - naive[i]))
            naive[i] = new
        if update <= 1e-13:
            break

    assert np.max(np.abs(x - naive)) < 1e-11


# --------------------------------------------------------------------------
# (c) American >= European, and the constraint that is never active
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [100, 200, 400])
def test_no_dividend_american_call_equals_the_european_call(n: int) -> None:
    """Evidence class: EXACT_IDENTITY up to solver tolerance. Slice item (c).

    Merton (1973): an American call on a non-dividend-paying stock is never
    exercised early. On this grid that is visible directly -- the active set is
    *empty* at every time level, so the LCP degenerates to the linear system
    the European engine solves, and the upper Dirichlet value
    `max(s_max - K, s_max e^{-q tau} - K e^{-r tau})` picks the European branch
    because `q = 0` makes it the larger.

    The two prices therefore differ only by the PSOR tolerance. Measured at
    `n = 50 ... 800`: 2.7e-08, 1.0e-08, 2.2e-11, 3.4e-14, 7.6e-14. The
    tolerance below is 1e-06, derived as a round number above the worst of
    those; it is a statement about the solver, not about early exercise.

    Note the sweep count for this case: 12, 12, 16, 33, 59 over the same grids,
    against 11-12 flat for the put. With no active set the iteration has to
    converge the whole linear system, including the poorly dominant rows at the
    top of the grid. The projection makes PSOR *faster*, not slower -- the
    opposite of the intuition that a constraint is extra work.
    """
    cfg = _cfg(n)
    american, european = _grids(cfg, kind="call")
    assert american.meta["early_exercise_node_count"] == 0
    assert american.meta["exercise_time_level_count"] == 0
    # NaN at every level *before* expiry. At expiry itself the value is the
    # payoff by construction, so every in-the-money node is trivially "active"
    # and the boundary is the node next to the strike -- the same convention
    # the lattice engine uses at its terminal level.
    boundary = np.asarray(american.meta["exercise_boundary"])
    assert np.all(np.isnan(boundary[:-1]))
    assert boundary[-1] == pytest.approx(100.0 + 0.5 * american.ds, abs=1e-10)
    assert american.price == pytest.approx(european.price, abs=1e-6)


@pytest.mark.parametrize("div", [0.0, 0.04])
@pytest.mark.parametrize("n", [200, 400])
def test_american_dominates_european_everywhere_on_the_grid(n: int, div: float) -> None:
    """Evidence class: EXACT_IDENTITY up to solver tolerance. Slice item (c).

    Two claims, with different budgets:

    - `V >= g` at every node, with **zero** slack permitted. This one really is
      exact: `np.maximum(obstacle, relaxed)` returns the obstacle value itself
      when the constraint binds, so an active node carries the payoff to the
      last bit.
    - `V_american >= V_european` at every node on the same grid. Not exact,
      because the American side is an iteration and the European side a direct
      solve. Measured worst violation over `n` in (200, 400, 800) and both
      dividends: `0.0`, `-7.4e-12`, `-1.0e-07` at the default `tol = 1e-8`, and
      `0.0`, `-2.2e-14`, `-5.2e-12` at `tol = 1e-12`. The violation is the
      solver's accumulated tolerance and nothing else, which is why it shrinks
      with `tol` and not with `n`. Budget below: `1e-6`.

    And the premium is not vacuously small -- it reaches 4.88 on this grid, so
    a bug that made the American solve silently European would fail loudly.
    """
    cfg = _cfg(n)
    american, european = _grids(cfg, div=div)
    payoff = _payoff("put", 100.0, american.s_grid)

    assert np.min(american.v - payoff) == 0.0
    assert american.meta["lcp_min_constraint_slack"] == 0.0

    premium = american.v - european.v
    assert np.min(premium) > -1e-6
    assert np.max(premium) > 1.0


# --------------------------------------------------------------------------
# (f) The complementarity condition, checked rather than assumed
# --------------------------------------------------------------------------


_LCP_BUDGET = 1e-6
"""Permitted worst complementarity residual, derived from `PSORConfig.tol`.

At every interior node and every time step the engine measures
`min(V - g, |A V - b|)`, which is zero exactly when one of the two LCP
inequalities is tight. Worst value over both kinds, `q` in (0, 6%) and
`n_s = n_t` in (50 ... 800), at the default `tol = 1e-8`: **1.10e-07**, and
that worst case is the no-dividend call, where the constraint is never active
and the quantity is really just the linear residual of a slowly converging
solve. For the put it stays between 1.3e-09 and 1.8e-08. The budget keeps a
factor of about nine over the worst measured value and is two orders of
magnitude below the smallest discretisation error any grid here produces.
"""


@pytest.mark.parametrize("div", [0.0, 0.06])
@pytest.mark.parametrize("kind", ["put", "call"])
@pytest.mark.parametrize("n", [100, 400])
def test_the_lcp_holds_at_every_node_and_every_time_step(
    n: int, kind: str, div: float
) -> None:
    """Evidence class: EXACT_IDENTITY up to solver tolerance. Slice item (f).

    The three LCP conditions, each checked separately over the whole march:

    - `V - g >= 0` everywhere: exact, `lcp_min_constraint_slack == 0.0`.
    - `A V - b >= 0` everywhere, which is the discrete `L V <= 0`: measured
      down to `-1.1e-07`, i.e. violated only at the solver's own tolerance.
    - complementarity, `min(V - g, |A V - b|) = 0` at every node: measured at
      most `1.1e-07`.

    That is a direct test of the free-boundary condition. It does not depend on
    knowing where the boundary is, and it fails if the projection is applied to
    the wrong array, or after the wrong step, or with the wrong sign.
    """
    result = _american(_cfg(n), kind=kind, div=div)
    meta = result.meta or {}
    assert meta["psor_converged"] is True
    assert meta["lcp_min_constraint_slack"] == 0.0
    assert meta["lcp_min_operator_residual"] > -_LCP_BUDGET
    assert meta["lcp_max_complementarity"] < _LCP_BUDGET


def test_tightening_the_solver_tightens_the_complementarity_residual() -> None:
    """The residual is the solver's, not the scheme's -- shown, not asserted.

    If the `1e-07` above were a discretisation artefact it would not move when
    `tol` moves. It moves by four orders of magnitude for four orders of
    magnitude of `tol`, which identifies it.
    """
    residuals = [
        (_american(_cfg(400, psor=PSORConfig(tol=tol))).meta or {})["lcp_max_complementarity"]
        for tol in (1e-6, 1e-8, 1e-10, 1e-12)
    ]
    assert all(b < a for a, b in pairwise(residuals)), residuals
    assert residuals[0] / residuals[-1] > 1e3


# --------------------------------------------------------------------------
# (g) The degenerate limits
# --------------------------------------------------------------------------


def _best_discounted_intrinsic(
    kind: str, spot: float, strike: float, expiry: float, rate: float, div: float
) -> float:
    """`max_t e^{-rt} * intrinsic(S0 e^{(r-q)t})`, by brute force.

    Deliberately *not* the closed-form candidate set the engines use. At
    `sigma = 0` the spot is deterministic, so the American value is a
    one-dimensional maximisation over the exercise date, and scanning 200001
    dates is a re-derivation rather than a restatement. That matters here:
    `qpl.engines.pde.american` imports its `sigma = 0` answer from
    `qpl.engines.tree.american`, so asserting that the two engines agree would
    be asserting that shared code equals itself.
    """
    best = 0.0
    for i in range(200_001):
        t = expiry * i / 200_000
        forward = spot * math.exp((rate - div) * t)
        payoff = max(forward - strike, 0.0) if kind == "call" else max(strike - forward, 0.0)
        best = max(best, math.exp(-rate * t) * payoff)
    return best


@pytest.mark.parametrize(
    ("kind", "spot", "strike", "expiry", "rate", "div"),
    [
        ("put", 100.0, 100.0, 1.0, 0.05, 0.00),
        ("put", 130.0, 100.0, 1.0, 0.05, 0.00),
        ("call", 100.0, 90.0, 1.0, 0.05, 0.00),
        ("call", 100.0, 100.0, 2.0, 0.03, 0.08),
        ("put", 60.0, 100.0, 3.0, 0.02, 0.10),
    ],
)
def test_zero_volatility_matches_the_deterministic_forward_path(
    kind: str, spot: float, strike: float, expiry: float, rate: float, div: float
) -> None:
    """Evidence class: CLOSED_FORM. Slice item (g).

    At `sigma = 0` there is nothing to diffuse and the engine short-circuits
    the grid. The value is the best discounted intrinsic value along the
    deterministic forward path, re-derived here by a 200001-point scan.
    """
    value = _american(
        _cfg(50),
        kind=kind,
        spot=spot,
        strike=strike,
        expiry=expiry,
        rate=rate,
        div=div,
        sigma=0.0,
    )
    expected = _best_discounted_intrinsic(kind, spot, strike, expiry, rate, div)
    assert value.value == pytest.approx(expected, abs=1e-10)
    assert (value.meta or {})["degenerate"] == "zero_vol"
    assert (value.meta or {})["exercise_boundary"] is None


def test_zero_volatility_put_with_q_above_r_exercises_at_an_interior_time() -> None:
    """Evidence class: CLOSED_FORM -- the Slice 2 finding, on the PDE engine.

    `h(t) = K e^{-rt} - S0 e^{-qt}` has one turning point, at
    `t* = log(rK / (q S0)) / (r - q)`, and `h''(t*) = (r - q) r K e^{-r t*}`.
    For `q > r` that is negative, so `t*` is an interior *maximum*, and when it
    lands inside `(0, T)` the best exercise date is neither now nor expiry --
    which means `max(intrinsic now, discounted forward intrinsic)`, the formula
    that works whenever `r >= q`, is simply wrong there.

    At `S0 = K = 100, r = 2%, q = 50%, T = 10`: `t* = 6.7060`, value 83.9506,
    against 81.1993 for the better endpoint. The same numbers Slice 2 pinned on
    the lattice engine (`tests/test_tree_american.py`), reached here through a
    different entry point.
    """
    spot = strike = 100.0
    expiry, rate, div = 10.0, 0.02, 0.50

    def h(t: float) -> float:
        return strike * math.exp(-rate * t) - spot * math.exp(-div * t)

    turning = math.log((rate * strike) / (div * spot)) / (rate - div)
    assert 0.0 < turning < expiry
    endpoints = max(0.0, h(0.0), h(expiry))

    value = _american(
        _cfg(50),
        kind="put",
        spot=spot,
        strike=strike,
        expiry=expiry,
        rate=rate,
        div=div,
        sigma=0.0,
    ).value
    assert value == pytest.approx(h(turning), abs=1e-12)
    assert value == pytest.approx(83.9505861332, abs=1e-9)
    assert value - endpoints == pytest.approx(2.7513055253, abs=1e-9)


@pytest.mark.parametrize(("kind", "spot"), [("put", 90.0), ("put", 110.0), ("call", 130.0)])
def test_expiry_zero_is_the_payoff(kind: str, spot: float) -> None:
    """Evidence class: EXACT_IDENTITY. Slice item (g)."""
    value = _american(_cfg(50), kind=kind, spot=spot, expiry=0.0).value
    expected = max(spot - 100.0, 0.0) if kind == "call" else max(100.0 - spot, 0.0)
    assert value == expected


@pytest.mark.parametrize("sigma", [0.0, 0.20])
def test_grid_greeks_refuse_the_degenerate_limits(sigma: float) -> None:
    """Same refusal as the European engine, for the same reason.

    Gamma is a point mass at both `T = 0` and `sigma = 0`, so there is nothing
    for a stencil to read. The *price* is still returned in both cases.
    """
    expiry = 0.0 if sigma > 0.0 else 1.0
    message = "expiry must be > 0" if expiry == 0.0 else "sigma must be > 0"
    with pytest.raises(InvalidInputError, match=message):
        greeks(
            AmericanOption(kind="put", strike=100.0, expiry=expiry),
            BlackScholesModel(sigma=sigma),
            _market(100.0, 0.05, 0.0),
            method="pde",
            cfg=_cfg(50),
        )


# --------------------------------------------------------------------------
# Greeks: what is shared with the European engine, and what is not
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_grid_greeks_agree_with_the_fine_lattice() -> None:
    """Evidence class: INDEPENDENT_ENGINE.

    There is no closed form for an American Greek, so the check is against the
    lattice engine at `n = 8001`. Measured at `n_s = n_t = 800`:

        quantity   PDE          tree(8001)   difference
        delta      -0.411107    -0.411062    -4.5e-05
        gamma       0.022990     0.022989     1.1e-06
        vega       37.4874      37.4866       8.3e-04
        theta      -2.237925    -2.237981     5.6e-05
        rho       -30.2211     -30.2198      -1.3e-03

    The tolerances below are those differences with a factor of about three.
    They are not accuracy claims: the lattice Greeks are themselves order 1
    (Slice 3), so this is two approximations agreeing, which is exactly what
    `INDEPENDENT_ENGINE` means.
    """
    from qpl.engines.tree import TreeConfig

    option = AmericanOption(kind="put", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.20)
    market = _market(100.0, 0.05, 0.0)
    pde = greeks(option, model, market, method="pde", cfg=_cfg(800))
    tree = greeks(option, model, market, method="tree", cfg=TreeConfig(n_steps=8001))

    assert pde.delta == pytest.approx(tree.delta, abs=1.5e-4)
    assert pde.gamma == pytest.approx(tree.gamma, abs=5e-6)
    assert pde.vega == pytest.approx(tree.vega, abs=3e-3)
    assert pde.theta == pytest.approx(tree.theta, abs=2e-4)
    assert pde.rho == pytest.approx(tree.rho, abs=5e-3)
    assert (pde.meta or {})["spot_in_exercise_region"] is False


def test_theta_is_zero_in_the_exercise_region_not_the_pde_identity() -> None:
    """Evidence class: EXACT_IDENTITY -- and a trap the European code walks into.

    Deep in the money the American put *is* the payoff: `V = K - S`,
    independent of `t`, so `theta = 0` exactly and `delta = -1`, `gamma = 0`.
    The PDE identity `V_t = -(1/2 sigma^2 S^2 V_SS + (r-q) S V_S - r V)` does
    not hold there -- inside the exercise region the PDE is a strict inequality,
    which is the whole content of complementarity -- and evaluating it anyway
    returns `rK - qS`, here `+5.0`, for a quantity whose true value is zero.

    The engine detects the exercise region from the obstacle and reports zero,
    naming the source. Measured at `S = 40, K = 100`: delta `-1.0`, gamma
    `4.3e-15`, theta `0.0`.
    """
    result = greeks(
        AmericanOption(kind="put", strike=100.0, expiry=1.0),
        BlackScholesModel(sigma=0.20),
        _market(40.0, 0.05, 0.0),
        method="pde",
        cfg=_cfg(400, s_max=400.0),
    )
    meta = result.meta or {}
    assert meta["spot_in_exercise_region"] is True
    assert meta["theta_source"].startswith("exercise region")
    assert result.delta == pytest.approx(-1.0, abs=1e-12)
    assert result.gamma == pytest.approx(0.0, abs=1e-12)
    assert result.theta == 0.0

    # And the quantity the identity would have produced is not small.
    rate, strike = 0.05, 100.0
    assert abs(rate * strike) > 1.0


def test_the_bump_greeks_path_prices_american_options_not_european_ones() -> None:
    """`greeks_method="bump"` is kept for the American engine too.

    It is the pre-Slice-4 path -- three solves at three spots -- and it still
    returns NaN for vega, theta and rho. What is asserted here is only that it
    bumps the *American* pricer: its delta must match the grid path's to the
    accuracy of a 1% central difference, and must not match the European
    engine's delta, which at this point differs by about 0.03.
    """
    option = AmericanOption(kind="put", strike=100.0, expiry=1.0)
    model = BlackScholesModel(sigma=0.20)
    market = _market(100.0, 0.05, 0.0)
    bumped = greeks(option, model, market, method="pde", cfg=_cfg(400, greeks_method="bump"))
    grid = greeks(option, model, market, method="pde", cfg=_cfg(400))
    european = greeks(
        EuropeanOption(kind="put", strike=100.0, expiry=1.0),
        model,
        market,
        method="pde",
        cfg=_cfg(400),
    )

    assert (bumped.meta or {})["greeks_method"] == "bump"
    assert math.isnan(bumped.vega) and math.isnan(bumped.theta) and math.isnan(bumped.rho)
    assert bumped.delta == pytest.approx(grid.delta, abs=2e-3)
    assert abs(grid.delta - european.delta) > 1e-2


# --------------------------------------------------------------------------
# Metadata contract
# --------------------------------------------------------------------------


def test_meta_reports_the_grid_the_solver_and_the_boundary() -> None:
    """The engine's observable surface, pinned so it cannot quietly shrink."""
    result = _american(_cfg(200))
    meta = result.meta or {}

    assert meta["method"] == "pde"
    assert meta["scheme"] == "psor"
    assert meta["strike_alignment"] == "midpoint"
    assert meta["time_stepping"] == "rannacher"
    assert meta["implicit_startup_steps"] == 4
    assert meta["n_steps_taken"] == 202
    assert meta["psor_omega"] == 1.2
    assert meta["psor_tol"] == 1e-8

    iterations = np.asarray(meta["psor_iterations"])
    assert iterations.shape == (202,)
    assert iterations.min() >= 1
    assert meta["psor_iterations_total"] == int(iterations.sum())
    assert meta["psor_iterations_mean"] == pytest.approx(float(iterations.mean()))
    assert meta["psor_iterations_max"] == int(iterations.max())

    boundary = np.asarray(meta["exercise_boundary"])
    times = np.asarray(meta["exercise_boundary_times"])
    assert boundary.shape == times.shape == (203,)
    assert times[0] == 0.0
    assert times[-1] == 1.0
    # Rannacher's four half steps make the last three intervals half-width.
    assert np.all(np.diff(times) > 0.0)
    assert times[-1] - times[-2] == pytest.approx(0.5 * (times[1] - times[0]))

    # Read-only, so a caller cannot corrupt another caller's view.
    for array in (meta["psor_iterations"], meta["exercise_boundary"], times):
        assert not np.asarray(array).flags.writeable


def test_the_strike_must_lie_inside_the_grid() -> None:
    """Same guard, same message, as the European engine."""
    with pytest.raises(InvalidInputError, match="must lie strictly inside the spot grid"):
        _american(_cfg(50, s_max=50.0))


def test_the_engine_is_deterministic() -> None:
    """No randomness anywhere: two identical calls are bit-identical."""
    a = _american(_cfg(200))
    b = _american(_cfg(200))
    assert a.value == b.value
    assert (a.meta or {})["psor_iterations_total"] == (b.meta or {})["psor_iterations_total"]
