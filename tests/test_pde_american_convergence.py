"""Measured convergence of the American PDE engine, its boundary, its sweeps.

Slice 5, items (b), (e) and (h). The engine itself, the LCP check and the
degenerate limits are `tests/test_pde_american.py`.

Every grid used here is `n_s = n_t = n`, `strike_alignment="midpoint"`,
`time_stepping="rannacher"`, `s_max = 4 S`, PSOR at the default
`omega = 1.2, tol = 1e-8`. Both of the first two settings are load-bearing and
the file measures why: unaligned, the same sequence fits order 1.69 with a
log-space RMS residual of **0.27** -- it is not a power law at all, because the
strike and the free boundary move relative to the nodes as `n` changes.

The reference is `qpl.cases.AMERICAN_BRACKETED_LIMIT = 6.090376463020103`, the
average of the CRR lattice at `n = 64000` and `n = 64001`. It comes from a
completely different discretisation, so using it here is not circular; its own
distance from the true value is below 1e-08 by the odd/even bracketing measured
in `tests/test_tree_american_convergence.py`, four orders of magnitude below
the smallest PDE error fitted below.

Runtime: about 5.6 s, dominated by the `n = 1600` solves.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from qpl.cases import AMERICAN_BRACKETED_LIMIT, AMERICAN_REFERENCE_SPEC
from qpl.engines.pde.american import PSORConfig
from qpl.engines.pde.pricers import PDEConfig
from qpl.engines.tree import TreeConfig
from qpl.instruments.options import AmericanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price
from qpl.validation import fit_convergence_order

LEVELS = (50, 100, 200, 400, 800)
"""Refinement path for the price order. `n_s = n_t = n`; 0.28 s in total."""

ITERATION_LEVELS = (100, 200, 400, 800, 1600)
"""Refinement path for the sweep-count study on `n_s = n_t`."""

STIFF_TIME_LEVELS = (25, 50, 100, 200)
"""Time levels for the `n_s = 8 n_t` study, where `dt / ds**2` grows like `n_t`."""


def _cfg(n: int, **overrides) -> PDEConfig:
    kwargs = dict(n_s=n, n_t=n, strike_alignment="midpoint", time_stepping="rannacher")
    kwargs.update(overrides)
    return PDEConfig(**kwargs)  # type: ignore[arg-type]


def _pde(
    cfg: PDEConfig,
    *,
    kind: str = "put",
    spot: float = 100.0,
    strike: float = 100.0,
    expiry: float = 1.0,
    rate: float = 0.05,
    div: float = 0.0,
    sigma: float = 0.20,
):
    return price(
        AmericanOption(kind=kind, strike=strike, expiry=expiry),  # type: ignore[arg-type]
        BlackScholesModel(sigma=sigma),
        Market(
            spot=spot,
            rate_curve=FlatRateCurve(rate),
            dividend_curve=FlatDividendCurve(div),
        ),
        method="pde",
        cfg=cfg,
    )


def _reference_errors(levels: tuple[int, ...], **overrides) -> list[float]:
    """Signed errors of the ATM put against the lattice-bracketed limit."""
    return [
        _pde(_cfg(n, **overrides)).value - AMERICAN_BRACKETED_LIMIT for n in levels
    ]


# --------------------------------------------------------------------------
# (b) The price order
# --------------------------------------------------------------------------


EXPECTED_ORDER = 1.85
ORDER_BAND = 0.12
"""The measured order and the band, both justified in the test below."""


def test_american_pde_price_order_against_the_lattice_bracketed_limit() -> None:
    """Evidence class: CONVERGENCE_ORDER. Slice item (b).

    Measured on the ATM American put (`S = K = 100, r = 5%, q = 0,
    sigma = 20%, T = 1`), aligned, Rannacher, against
    `AMERICAN_BRACKETED_LIMIT`:

        n         50         100        200        400        800       1600
        err   -6.963e-02 -2.086e-02 -5.637e-03 -1.515e-03 -4.240e-04 -1.246e-04
        local      --      1.739      1.888      1.895      1.837      1.766

    Fitted order **1.8502** (log-space RMS residual 0.0265) over
    `n = 50 ... 800`, and **1.8506** (0.0261) over `n = 100 ... 1600` -- the
    same number from a disjoint tail, which is what makes it a rate rather than
    an artefact of where the window sits.

    **Why the band is 1.85 +- 0.12 and not "about 2" or "about 1".** Two error
    terms are present and they have different orders.

    - The Crank-Nicolson/Rannacher scheme on a smooth solution is `O(ds**2) +
      O(dt**2)`, and `strike_alignment="midpoint"` is what stops the payoff
      kink spoiling that constant (Slice 4). On `n_s = n_t = n` both are
      `O(1/n**2)`.
    - The free boundary is located only to the node spacing. Whether a given
      node is in the exercise region at a given time level is a yes/no
      decision, so the boundary the grid represents differs from the true one
      by `O(ds)`, and the value error that follows is `O(ds)` times the size of
      the mismatch region -- first order, and untouched by how good the time
      stepping is.

    The second term must eventually dominate, and the *local* orders above show
    it beginning to: 1.895 at `n = 200`, 1.766 at `n = 800`, and 1.48 over the
    `n = 1600 -> 3200` doubling (measured; not run here, because a 3200-grid
    solve costs 1.8 s). So the honest statement is "between 1 and 2, currently
    1.85 and falling", and the band admits exactly that: it excludes 2.0, it
    excludes 1.0, and the test separately asserts that the tail order is below
    the middle of the range so the drift cannot silently disappear.

    All six errors are **negative**: this engine approaches the limit from
    below, as the Leisen-Reimer lattice does and the CRR lattice does not. Two
    engines on opposite sides of the limit bracket it, which is asserted here
    because it is what makes the cross-engine tolerances in `qpl.cases`
    derivable rather than guessed.
    """
    errors = _reference_errors(LEVELS)
    fit = fit_convergence_order([1.0 / n for n in LEVELS], [abs(e) for e in errors])

    assert abs(fit.order - EXPECTED_ORDER) <= ORDER_BAND, (fit.order, fit.residual)
    assert fit.residual < 0.05, fit.residual
    # Explicitly not order 2, and explicitly not order 1.
    assert 1.0 < fit.order < 2.0, fit.order
    # From below, monotonically.
    assert all(e < 0.0 for e in errors), errors
    assert all(abs(b) < abs(a) for a, b in pairwise(errors)), errors

    # The boundary term is taking over: the last local order is below the best.
    local = [np.log2(abs(a / b)) for a, b in pairwise(errors)]
    assert local[-1] < max(local) - 0.05, local


def test_the_order_survives_a_disjoint_refinement_window() -> None:
    """A rate measured on one window and confirmed on another.

    `n = 100 ... 1600` shares only its endpoints' shape with `n = 50 ... 800`
    and fits 1.8506 against 1.8502. A fit that moved by more than the band
    between the two windows would mean the sequence is pre-asymptotic and the
    number should not be quoted as an order at all.
    """
    errors = _reference_errors(ITERATION_LEVELS)
    fit = fit_convergence_order(
        [1.0 / n for n in ITERATION_LEVELS], [abs(e) for e in errors]
    )
    assert abs(fit.order - EXPECTED_ORDER) <= ORDER_BAND, (fit.order, fit.residual)
    assert fit.residual < 0.05, fit.residual


def test_strike_alignment_is_what_makes_the_sequence_a_power_law() -> None:
    """Evidence class: NEGATIVE_FINDING -- the unaligned grid is not a power law.

    With `strike_alignment="none"` the strike, and with it the free boundary,
    lands in a different place relative to the nodes at every `n`. The errors
    (-6.96e-02, -4.70e-02, -1.16e-02, -3.01e-03, -7.90e-04) fit order 1.6885
    with a log-space RMS residual of **0.2688**, ten times the aligned fit's
    0.0265. The assertion is on the *residual*: the order it happens to report
    is not meaningful, and pinning it would be pinning noise.

    This is the Slice 4 strike-alignment result reappearing under early
    exercise, and it is the reason every other grid in this file is aligned.
    """
    errors = _reference_errors(LEVELS, strike_alignment="none")
    fit = fit_convergence_order([1.0 / n for n in LEVELS], [abs(e) for e in errors])
    assert fit.residual > 0.15, fit.residual

    aligned = fit_convergence_order(
        [1.0 / n for n in LEVELS], [abs(e) for e in _reference_errors(LEVELS)]
    )
    assert fit.residual > 5.0 * aligned.residual, (fit.residual, aligned.residual)


@pytest.mark.parametrize(
    ("label", "kind", "spot", "strike", "rate", "div"),
    [
        ("ls2001_row1_put", "put", 36.0, 40.0, 0.06, 0.00),
        ("call_with_yield", "call", 100.0, 100.0, 0.05, 0.06),
    ],
)
def test_the_order_is_not_special_to_the_reference_point(
    label: str, kind: str, spot: float, strike: float, rate: float, div: float
) -> None:
    """Evidence class: CONVERGENCE_ORDER, fitted without a reference value.

    There is no bracketed limit for these two points, so the order is fitted on
    *successive differences* `|v(2n) - v(n)|`, which behave like
    `C n^{-p} (1 - 2^{-p})` and therefore carry the same slope as the error.
    The same trick `tests/oracle/test_american_vs_quantlib.py` uses on
    QuantLib's grid, and for the same reason: using this engine's own fine
    answer as the limit would make the fit circular.

    Measured over `n = 50 ... 1600`:

        point             differences                                    order  residual
        ls2001_row1_put   3.63e-02 6.23e-03 1.50e-03 4.43e-04 1.33e-04   1.9992  0.166
        call_with_yield   1.45e-02 5.60e-03 1.59e-03 4.67e-04 1.35e-04   1.7074  0.081

    The band below is deliberately wider than the reference point's -- these
    are noisier fits on five points with no limit to anchor them -- but it
    still says the same thing: between 1 and 2, nearer 2 at these grid sizes.
    """
    levels = (50, 100, 200, 400, 800, 1600)
    values = [
        _pde(_cfg(n), kind=kind, spot=spot, strike=strike, rate=rate, div=div).value
        for n in levels
    ]
    diffs = [abs(b - a) for a, b in pairwise(values)]
    fit = fit_convergence_order([1.0 / n for n in levels[:-1]], diffs)

    assert 1.5 <= fit.order <= 2.1, (label, fit.order, fit.residual)
    assert fit.residual < 0.25, (label, fit.residual)
    # Monotone approach, which is what makes a successive-difference fit
    # interpretable as a rate rather than as a sign-flipping wobble.
    assert all(b > a for a, b in pairwise(values)), values


# --------------------------------------------------------------------------
# (e) The exercise boundary
# --------------------------------------------------------------------------


BOUNDARY_N = 400
BOUNDARY_TREE_N_STEPS = 2000
"""Grids for the boundary comparison: `ds = 0.9950`, tree node gap 0.4462."""


def _boundaries(kind: str, div: float):
    pde = _pde(_cfg(BOUNDARY_N), kind=kind, div=div)
    tree = price(
        AmericanOption(kind=kind, strike=100.0, expiry=1.0),  # type: ignore[arg-type]
        BlackScholesModel(sigma=0.20),
        Market(
            spot=100.0,
            rate_curve=FlatRateCurve(0.05),
            dividend_curve=FlatDividendCurve(div),
        ),
        method="tree",
        cfg=TreeConfig(n_steps=BOUNDARY_TREE_N_STEPS),
    )
    return pde, tree


def test_the_put_boundary_is_monotone_at_every_level_not_only_within_a_parity() -> None:
    """Evidence class: EXACT_IDENTITY (an ordering). Slice item (e).

    The continuous-time boundary `B(t)` of an American put is non-decreasing
    and tends to `K` as `t -> T`. Slice 2 found that the *lattice* boundary is
    monotone only within a parity subsequence: consecutive levels of a
    binomial tree live on the two interleaved node grids `S0 u^{2i-j}`, so the
    reported boundary alternates between them and zigzags by up to one
    fine-grid offset.

    That failure mode does not exist here, and the reason is structural rather
    than lucky: a finite-difference grid is the *same* set of nodes at every
    time level. There is one subsequence, the whole sequence, and it is
    non-decreasing with zero tolerance -- the boundary either stays on a node
    or steps up to the next one. Measured at `n_s = n_t = 400`: 402 steps, no
    step negative, minimum step exactly 0.0.

    Two more facts the grid gets for free:

    - **No NaN prefix.** The lattice's boundary is NaN for the first 47 of 2000
      levels (up to `t = 0.0235`) because no lattice node sits low enough that
      early; the grid reaches down to `S = 0` at every level, so the boundary
      exists everywhere, starting at 80.5970 at `t = 0`.
    - **At expiry it is the node next to the strike**, `K - ds/2 = 99.5025`,
      exactly, because `strike_alignment="midpoint"` puts the strike halfway
      between two nodes. That is "equals `K` up to node spacing", stated
      precisely.
    """
    pde, tree = _boundaries("put", 0.0)
    meta = pde.meta or {}
    boundary = np.asarray(meta["exercise_boundary"])
    steps = np.diff(boundary)

    assert not np.any(np.isnan(boundary))
    assert np.min(steps) == 0.0
    assert np.max(steps) > 0.0
    assert boundary[-1] == pytest.approx(100.0 - 0.5 * meta["ds"], abs=1e-10)
    assert boundary[0] == pytest.approx(80.5970, abs=1e-3)

    tree_boundary = np.asarray((tree.meta or {})["exercise_boundary"], dtype=float)
    assert np.any(np.isnan(tree_boundary))
    assert not np.any(np.isnan(boundary))


def test_the_call_boundary_is_monotone_downward_toward_the_strike() -> None:
    """The mirror statement, so the convention is not put-specific.

    With `q = 6% > r = 5%` an American call is exercised for large `S`, so the
    exercise region is `S >= B(t)` and the reported boundary is the *smallest*
    active node and non-increasing in `t`. Measured at `n_s = n_t = 400`:
    134.3284 at `t = 0` down to `K + ds/2 = 100.4975` at expiry, no step
    positive.
    """
    pde, _ = _boundaries("call", 0.06)
    meta = pde.meta or {}
    boundary = np.asarray(meta["exercise_boundary"])

    assert not np.any(np.isnan(boundary))
    assert np.max(np.diff(boundary)) == 0.0
    assert boundary[0] == pytest.approx(134.3284, abs=1e-3)
    assert boundary[-1] == pytest.approx(100.0 + 0.5 * meta["ds"], abs=1e-10)


BOUNDARY_TIMES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def test_the_pde_boundary_matches_the_lattice_boundary_within_node_spacing() -> None:
    """Evidence class: INDEPENDENT_ENGINE. Slice item (e).

    Neither engine can place the boundary better than its own node spacing, so
    the most that can be asserted is that they agree to the *sum* of the two:
    `ds = 0.9950` on the `n_s = 400` grid and `S (1 - 1/u) = 0.4462` between
    adjacent terminal nodes of the `n = 2000` lattice, i.e. 1.4412. Measured
    differences `B_pde - B_tree`:

        t        0.10    0.25    0.50    0.75    0.90    0.95    0.99
        diff   +0.1858 +0.4494 -0.0381 -0.0989 +0.7245 +0.2717 -0.1040

    worst 0.7245, about half the combined spacing, and the sign changes -- the
    two boundaries interleave rather than one sitting consistently above the
    other, which is what agreement inside a shared quantisation looks like.

    The tolerance is that combined spacing, derived and not chosen. Tightening
    it would assert that two different quantisations of the same curve land on
    the same side of it, which is not true and should not be asserted.
    """
    pde, tree = _boundaries("put", 0.0)
    meta = pde.meta or {}
    boundary = np.asarray(meta["exercise_boundary"])
    times = np.asarray(meta["exercise_boundary_times"])

    tree_meta = tree.meta or {}
    tree_boundary = np.asarray(tree_meta["exercise_boundary"], dtype=float)
    tree_times = np.linspace(0.0, 1.0, BOUNDARY_TREE_N_STEPS + 1)
    combined = meta["ds"] + 100.0 * (1.0 - 1.0 / tree_meta["u"])

    differences = []
    for t in BOUNDARY_TIMES:
        i = int(np.argmin(np.abs(times - t)))
        j = int(np.argmin(np.abs(tree_times - t)))
        assert not np.isnan(tree_boundary[j])
        differences.append(boundary[i] - tree_boundary[j])

    assert max(abs(d) for d in differences) < combined, (differences, combined)
    # Interleaved, not offset: both signs appear.
    assert max(differences) > 0.0 and min(differences) < 0.0, differences


# --------------------------------------------------------------------------
# (h) How the sweep count grows
# --------------------------------------------------------------------------


def _mean_sweeps(cfg: PDEConfig) -> float:
    return float((_pde(cfg).meta or {})["psor_iterations_mean"])


def test_sweep_counts_grow_with_dt_over_ds_squared_not_with_n() -> None:
    """Evidence class: CONVERGENCE_ORDER (of a cost, not of an error). Item (h).

    Forsyth and Vetzal (2002) motivate their penalty method partly by the
    observation that PSOR iteration counts grow as the grid is refined. That is
    confirmed here and also *qualified*: what drives the growth is the
    stiffness ratio `dt / ds**2`, and a refinement path can hold it down.

    **Path A, `n_s = n_t = n`** (`dt / ds**2` grows like `n`), mean sweeps per
    time step at the default `omega = 1.2, tol = 1e-8`:

        n        100    200    400    800   1600   3200
        mean    11.33  11.17  10.65  10.18  12.75  21.80

    Flat to `n = 800` and rising after. The two effects nearly cancel there:
    the matrix gets stiffer, but the previous time level -- the starting
    iterate -- gets closer to the answer, because the solution moves by `O(dt)`
    over a step. A least-squares fit over 100...3200 is not a power law
    (residual 0.195) and the exponent it reports, 0.10, should not be quoted.

    **Path B, `n_s = 8 n_t`** (`dt / ds**2` grows like `n_t`, with the constant
    64x larger), where the starting-iterate effect is much weaker:

        n_t       25     50    100    200
        n_s      200    400    800   1600
        mean   19.15  32.88  56.97  98.26

    a clean power law: fitted exponent **0.787** in `n_t`, log-space RMS
    residual **0.0018**. Optimal SOR on a consistently ordered matrix would
    give 0.5 and Gauss-Seidel 1.0; 0.787 at a fixed `omega = 1.2` sits between
    them, which is what a fixed relaxation away from the per-grid optimum
    should do (see `qpl.engines.pde.american.PSOR_OMEGA_DEFAULT`).

    The test asserts the power law on path B and the flatness on path A. It
    stops at `n = 1600` on both; the `n = 3200` row above is measured but not
    run, because a 3200-grid solve costs 1.8 s and buys one point.
    """
    flat = [_mean_sweeps(_cfg(n)) for n in ITERATION_LEVELS]
    assert 9.0 < min(flat) and max(flat) < 16.0, flat
    # The stiffness ratio grows sixteen-fold across that path while the sweep
    # count moves by less than 40%.
    assert max(flat) / min(flat) < 1.4, flat

    stiff = [_mean_sweeps(_cfg(nt, n_s=8 * nt)) for nt in STIFF_TIME_LEVELS]
    fit = fit_convergence_order([1.0 / nt for nt in STIFF_TIME_LEVELS], stiff)

    # `fit_convergence_order` fits against `h = 1/n_t`, so a cost that *grows*
    # with `n_t` comes back as a negative slope.
    assert fit.order == pytest.approx(-0.787, abs=0.08), fit.order
    assert fit.residual < 0.02, fit.residual
    assert all(b > a for a, b in pairwise(stiff)), stiff
    # And the growth is real, not a rounding wobble: five-fold over eight-fold
    # refinement.
    assert stiff[-1] / stiff[0] > 4.0, stiff


def test_raising_omega_is_worth_an_order_of_magnitude_on_a_stiff_grid() -> None:
    """The default is a choice, and this is what it costs where it is wrong.

    On `n_s = 80 n_t, n_t = 20` -- a grid fine in space and coarse in time, the
    regime the Slice 4 gamma pathology lives in -- mean sweeps per step run
    1367, 952, 640, 391, 173 at `omega = 1.0, 1.2, 1.4, 1.6, 1.8`, still
    falling at 1.8. On the `n_s = n_t` path the ordering is the other way round
    and `omega = 1.8` costs about seven times `omega = 1.2`.

    Only three relaxations are run below, and only at `n_t = 10`, to keep the
    file under four seconds; the point is the *direction*, which reverses
    between the two paths.
    """
    stiff_cheap = _mean_sweeps(_cfg(10, n_s=800, psor=PSORConfig(omega=1.2)))
    stiff_fast = _mean_sweeps(_cfg(10, n_s=800, psor=PSORConfig(omega=1.8)))
    square_cheap = _mean_sweeps(_cfg(400, psor=PSORConfig(omega=1.2)))
    square_slow = _mean_sweeps(_cfg(400, psor=PSORConfig(omega=1.8)))

    assert stiff_fast < 0.5 * stiff_cheap, (stiff_cheap, stiff_fast)
    assert square_slow > 2.0 * square_cheap, (square_cheap, square_slow)


def test_the_reference_spec_is_the_one_the_cases_layer_pins() -> None:
    """No second copy of the specification: the numbers above describe this one."""
    spec = AMERICAN_REFERENCE_SPEC
    assert (spec.spot, spec.strike, spec.expiry) == (100.0, 100.0, 1.0)
    assert (spec.rate, spec.dividend, spec.sigma, spec.kind) == (0.05, 0.0, 0.20, "put")
