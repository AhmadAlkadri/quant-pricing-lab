"""The barrier on a lattice: the Boyle-Lau sawtooth, and what fixes it.

A lattice cannot knock out *at* the barrier, only at the first node beyond it,
so the tree prices an **effective** barrier `H_eff <= H` (down) whose distance
from `H` depends on the fractional part of `lambda = log(S/H)/(sigma sqrt(dt))`.
Since `lambda` grows like `sqrt(n)`, that fractional part cycles and the price
error sawtooths in `n`. Every assertion below names its `EvidenceClass`.

1. **NEGATIVE_FINDING** -- the sawtooth, pinned by amplitude over a full period
   at three refinement levels. The amplitude decays at order **0.4566**, not 1:
   the barrier displacement is `O(sigma sqrt(dt))` and the price is locally
   linear in it, so the error is `O(n**-1/2)`. This is the same one-half the
   discrete-monitoring Monte Carlo bias has, and for the same reason.
2. **NEGATIVE_FINDING** -- Leisen-Reimer does not help. Measured amplitude
   ratio LR/CRR 0.977, 0.978, 0.980 at the three levels, and a fitted order of
   0.4542 against CRR's 0.4566. Its construction aligns the *strike*, and a
   barrier is a second level it says nothing about.
3. **CONVERGENCE_ORDER** -- along the Boyle-Lau subsequence
   `n_k = floor(k^2 sigma^2 T / log(S/H)^2)` the error drops by three to four
   decimal orders and becomes first order: block-RMS fit **1.1258** (residual
   0.0041). It is *not* monotone, which the slice expected and which the signs
   disprove.
4. **NEGATIVE_FINDING** -- the Boyle-Lau step counts do nothing for
   Leisen-Reimer (block order 0.4254, errors three decimal orders worse than
   CRR's at the same `n`), because `n_k` is derived from the CRR ladder and an
   LR lattice has no such ladder.
5. **EXACT_IDENTITY** -- in-out parity holds on the lattice itself, not only in
   the limit, because the knock-in is assembled from the same three roll-backs.

Sources: Boyle, P.P. and Lau, S.H. (1994), "Bumping up against the barrier with
the binomial method", Journal of Derivatives 1(4), 6-14, for the sawtooth and
the step-count remedy; Derman, Kani, Ergener and Bardhan (1995), Risk 8(6), for
the interpolation remedy, which is cited and not implemented. Cox, Ross and
Rubinstein (1979) and Leisen and Reimer (1996) for the lattices. Every number
here was measured in this repository.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.engines.analytic.barrier import barrier_price
from qpl.engines.tree.barrier import barrier_layer_index, boyle_lau_steps
from qpl.engines.tree.pricers import TreeConfig
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import (
    BarrierOption,
    EuropeanOption,
    uniform_monitoring_times,
)
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price
from qpl.validation import fit_convergence_order

_SPOT, _STRIKE, _EXPIRY = 100.0, 100.0, 0.5
_RATE, _DIV, _SIGMA, _BARRIER = 0.08, 0.04, 0.25, 95.0

_LAMBDA_0 = math.log(_SPOT / _BARRIER) / (_SIGMA * math.sqrt(_EXPIRY))
"""0.29015869 -- the barrier's distance from the spot in units of
`sigma sqrt(T)`. The sawtooth's period in `n` is `2 sqrt(n) / lambda_0`, which
is why the windows below widen with `n`: a fixed-width window of 20 steps
covers a whole period at `n = 100` and 9% of one at `n = 1600`, so amplitudes
measured on a fixed window would appear to decay for a reason that has nothing
to do with convergence."""

_WINDOW_STARTS: tuple[int, ...] = (100, 200, 400)
"""Where each full-period sawtooth window begins. Three levels, not five: the
two further levels were measured (amplitudes 0.34032 at `n0 = 800` and 0.25807
at 1600, fitted order 0.4625 over all five) and cost 6 seconds for 0.006 of
fitted order, which is inside the three-point fit's own residual."""

_BOYLE_LAU_LAYERS: tuple[int, ...] = tuple(range(4, 17))
"""Layer indices `k`; `n_k` runs from 190 to 3040 at this specification."""

_SAWTOOTH_ORDER_BAND = 0.12
"""Band on the fitted sawtooth-amplitude order around one half.

Measured 0.4566 (CRR) and 0.4542 (LR) over the three windows, with log-space
residuals of 0.030; over five windows the same fits give 0.4625 and 0.4549. The
band has to cover the 0.043 shortfall from 0.5 -- which is a real
pre-asymptotic effect, since the amplitude is the maximum of a quantity whose
period is itself growing -- plus the 0.006 level-count sensitivity, so 0.12
keeps about 2.5x over the observed spread.
"""


def _market(spot: float = _SPOT) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(_RATE),
        dividend_curve=FlatDividendCurve(_DIV),
    )


def _model() -> BlackScholesModel:
    return BlackScholesModel(sigma=_SIGMA)


def _option(barrier_type: str = "down-and-out", rebate: float = 0.0) -> BarrierOption:
    return BarrierOption(
        "call", _STRIKE, _EXPIRY, _BARRIER, barrier_type, rebate
    )


_CONTINUOUS = barrier_price(
    S=_SPOT, K=_STRIKE, T=_EXPIRY, r=_RATE, sigma=_SIGMA, q=_DIV, H=_BARRIER,
    rebate=0.0, barrier_type="down-and-out", kind="call",
)


def _tree(n: int, scheme: str = "crr", option: BarrierOption | None = None):
    return price(
        option if option is not None else _option(), _model(), _market(),
        method="tree", cfg=TreeConfig(n_steps=n, scheme=scheme),  # type: ignore[arg-type]
    )


def _period(n0: int) -> int:
    """Steps needed for `lambda` to advance by one at `n0`: `2 sqrt(n0)/lambda_0`."""
    return math.ceil(2.0 * math.sqrt(n0) / _LAMBDA_0) + 1


@pytest.fixture(scope="module")
def sawtooth() -> dict[int, dict[str, float]]:
    """Full-period error windows at each level, for CRR and Leisen-Reimer."""
    out: dict[int, dict[str, float]] = {}
    for n0 in _WINDOW_STARTS:
        period = _period(n0)
        crr = [_tree(n).value - _CONTINUOUS for n in range(n0, n0 + period)]
        first_odd = n0 if n0 % 2 else n0 + 1
        lr = [
            _tree(n, "leisen-reimer").value - _CONTINUOUS
            for n in range(first_odd, n0 + period, 2)
        ]
        out[n0] = {
            "period": period,
            "crr_min": min(crr),
            "crr_max": max(crr),
            "crr_amplitude": max(crr) - min(crr),
            "lr_amplitude": max(lr) - min(lr),
        }
    return out


@pytest.fixture(scope="module")
def boyle_lau() -> dict[str, list[float]]:
    """The Boyle-Lau subsequence, and Leisen-Reimer evaluated at the same `n`."""
    steps, crr, lr, misalignment = [], [], [], []
    for k in _BOYLE_LAU_LAYERS:
        n = boyle_lau_steps(
            k, spot=_SPOT, barrier=_BARRIER, sigma=_SIGMA, expiry=_EXPIRY
        )
        result = _tree(n)
        steps.append(n)
        crr.append(result.value - _CONTINUOUS)
        misalignment.append(1.0 - result.meta["effective_barrier_ratio"])
        lr.append(_tree(n if n % 2 else n + 1, "leisen-reimer").value - _CONTINUOUS)
    return {"n": steps, "crr": crr, "lr": lr, "misalignment": misalignment}


def _block_rms(values: list[float], steps: list[int], groups) -> tuple[np.ndarray, np.ndarray]:
    h = np.array([float(np.mean([1.0 / steps[i] for i in range(a, b)])) for a, b in groups])
    rms = np.array([float(np.sqrt(np.mean(np.square(values[a:b])))) for a, b in groups])
    return h, rms


_GROUPS = ((0, 4), (4, 8), (8, 13))
"""Three blocks of the thirteen Boyle-Lau levels. A block-RMS fit rather than a
per-`n` fit for the same reason Slice 6 used one on the CRR digital: the
per-`n` sequence is not a power law -- its constant carries the leftover
misalignment, which is uniform-ish in `[0, 1)` -- while its envelope is. A
straight per-`n` fit returns 0.5716 with a log-space residual of 1.49, i.e. the
diagnostic saying the fit is meaningless."""


# --------------------------------------------------------------------------
# The contract.
# --------------------------------------------------------------------------


def test_tree_refuses_discrete_monitoring_and_greeks() -> None:
    """Both refusals name what to use instead."""
    cfg = TreeConfig(n_steps=100)
    discrete = BarrierOption(
        "call", _STRIKE, _EXPIRY, _BARRIER, "down-and-out", 0.0,
        uniform_monitoring_times(_EXPIRY, 50),
    )
    with pytest.raises(NotSupportedError) as excinfo:
        price(discrete, _model(), _market(), method="tree", cfg=cfg)
    assert "method='mc'" in str(excinfo.value)

    with pytest.raises(NotSupportedError) as excinfo:
        greeks(_option(), _model(), _market(), method="tree", cfg=cfg)
    message = str(excinfo.value)
    assert "sawtooth" in message
    assert "method='analytic'" in message


def test_boyle_lau_steps_validates_and_places_the_layer() -> None:
    """`n_k` puts layer `k` at or just below the barrier, never above it."""
    for k in (1, 4, 9, 16):
        n = boyle_lau_steps(
            k, spot=_SPOT, barrier=_BARRIER, sigma=_SIGMA, expiry=_EXPIRY
        )
        lam = barrier_layer_index(
            spot=_SPOT, barrier=_BARRIER, sigma=_SIGMA, expiry=_EXPIRY, n_steps=n
        )
        # `lambda` lands just below the integer `k`, so `ceil(lambda) == k` and
        # layer `k` is the first dead one.
        assert k - 1.0 < lam <= k
        assert math.ceil(lam) == k
        # The effective barrier is therefore at or below the contractual one.
        assert _tree(n).meta["effective_barrier"] <= _BARRIER

    with pytest.raises(InvalidInputError, match="layer must be >= 1"):
        boyle_lau_steps(0, spot=_SPOT, barrier=_BARRIER, sigma=_SIGMA, expiry=_EXPIRY)
    with pytest.raises(InvalidInputError, match="positive and distinct"):
        boyle_lau_steps(4, spot=_SPOT, barrier=_SPOT, sigma=_SIGMA, expiry=_EXPIRY)
    with pytest.raises(InvalidInputError, match="too far from"):
        boyle_lau_steps(1, spot=_SPOT, barrier=10.0, sigma=_SIGMA, expiry=_EXPIRY)


def test_the_degenerate_limits_delegate_to_the_closed_form() -> None:
    """`T = 0` and `sigma = 0` are facts about the model, not about a lattice."""
    market = _market()
    cfg = TreeConfig(n_steps=50)
    at_expiry = BarrierOption("call", 90.0, 0.0, _BARRIER, "down-and-out", 3.0)
    assert price(at_expiry, _model(), market, method="tree", cfg=cfg).value == 10.0

    zero_vol = _option(rebate=3.0)
    tree = price(zero_vol, BlackScholesModel(sigma=0.0), market, method="tree", cfg=cfg)
    analytic = barrier_price(
        S=_SPOT, K=_STRIKE, T=_EXPIRY, r=_RATE, sigma=0.0, q=_DIV, H=_BARRIER,
        rebate=3.0, barrier_type="down-and-out", kind="call",
    )
    assert tree.value == analytic
    assert tree.meta["degenerate"] == "zero_vol"


@pytest.mark.parametrize("barrier_type", ["down-and-out", "down-and-in"])
def test_a_touched_spot_matches_the_closed_form_exactly(barrier_type: str) -> None:
    """Same inception guard as every other engine, same number."""
    market = _market(90.0)
    option = BarrierOption("call", _STRIKE, _EXPIRY, _BARRIER, barrier_type, 3.0)
    tree = price(option, _model(), market, method="tree", cfg=TreeConfig(n_steps=50))
    assert tree.value == price(option, _model(), market).value
    assert tree.meta["degenerate"] == "touched_at_inception"


def test_an_unreachable_barrier_gives_the_vanilla_lattice_price_bit_for_bit() -> None:
    """Evidence class: EXACT_IDENTITY.

    With `H` far enough below the lattice's reach no node is ever knocked, so
    the barrier roll-back *is* the vanilla roll-back and the two must agree to
    the last bit -- not approximately. `meta["effective_barrier"]` is `None`,
    which is how the caller is told nothing was knocked rather than having to
    infer it from the price.
    """
    cfg = TreeConfig(n_steps=101)
    far = BarrierOption("call", _STRIKE, _EXPIRY, 1e-6, "down-and-out", 0.0)
    barrier = price(far, _model(), _market(), method="tree", cfg=cfg)
    vanilla = price(
        EuropeanOption("call", _STRIKE, _EXPIRY), _model(), _market(),
        method="tree", cfg=cfg,
    )
    assert barrier.value == vanilla.value
    assert barrier.meta["effective_barrier"] is None
    assert barrier.meta["effective_barrier_ratio"] is None


@pytest.mark.parametrize("scheme", ["crr", "leisen-reimer"])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_in_out_parity_holds_on_the_lattice_itself(scheme: str, kind: str) -> None:
    """Evidence class: EXACT_IDENTITY, on the discretisation and not in its limit.

    The knock-in is assembled from the same lattice's vanilla and knock-out
    roll-backs, so their sum is the lattice's vanilla price identically -- at
    `n = 51`, where every one of the three numbers is several percent wrong.
    That is the point: an identity that only holds in the limit cannot
    distinguish a correct assembly from a convergent one.
    """
    cfg = TreeConfig(n_steps=51, scheme=scheme)  # type: ignore[arg-type]
    market, model = _market(), _model()
    common = (kind, _STRIKE, _EXPIRY, _BARRIER)
    knock_in = price(
        BarrierOption(*common, "down-and-in", 0.0), model, market, method="tree", cfg=cfg
    )
    knock_out = price(
        BarrierOption(*common, "down-and-out", 0.0), model, market, method="tree", cfg=cfg
    )
    vanilla = price(
        EuropeanOption(kind, _STRIKE, _EXPIRY), model, market, method="tree", cfg=cfg
    )
    assert knock_in.value + knock_out.value == pytest.approx(vanilla.value, abs=1e-13)
    assert knock_in.value > 1e-6 and knock_out.value > 1e-6


def test_a_knock_in_rebate_is_the_survival_claim() -> None:
    """Evidence class: CLOSED_FORM, against the analytic engine at a fine lattice.

    The knock-in rebate leg is a roll-back of the terminal payoff `1` with dead
    nodes zeroed -- the discounted lattice probability of never touching the
    barrier -- and adding `rebate` times it must reproduce the closed form's
    `E` block. Checked at the Boyle-Lau `n` for layer 12, where the barrier is
    essentially on a layer and the remaining error is the lattice's own.
    """
    n = boyle_lau_steps(12, spot=_SPOT, barrier=_BARRIER, sigma=_SIGMA, expiry=_EXPIRY)
    for rebate in (0.0, 3.0):
        option = _option("down-and-in", rebate)
        tree = _tree(n, option=option).value
        exact = barrier_price(
            S=_SPOT, K=_STRIKE, T=_EXPIRY, r=_RATE, sigma=_SIGMA, q=_DIV,
            H=_BARRIER, rebate=rebate, barrier_type="down-and-in", kind="call",
        )
        assert tree == pytest.approx(exact, abs=2e-3)
    # The rebate leg is worth something and is bounded by the rebate itself.
    bare = _tree(n, option=_option("down-and-in", 0.0)).value
    with_rebate = _tree(n, option=_option("down-and-in", 3.0)).value
    assert 0.0 < with_rebate - bare < 3.0


# --------------------------------------------------------------------------
# (1) and (2): the sawtooth.
# --------------------------------------------------------------------------


def test_the_lattice_barrier_error_sawtooths_at_order_one_half(sawtooth) -> None:
    """Evidence class: NEGATIVE_FINDING -- half an order, not one.

    Measured over a full sawtooth period at each level:

        n0        period   min err    max err   amplitude
        100         70     +0.01314   +0.94823   0.93510
        200         99     +0.00494   +0.64461   0.63967
        400        139     +0.00336   +0.49990   0.49654

    fitted amplitude order **0.4566** (log-space residual 0.030). The mechanism
    is arithmetic: the effective barrier is `H exp(-frac(lambda) sigma
    sqrt(dt))`, the displacement is `O(sigma sqrt(dt)) = O(n**-1/2)`, and the
    price is locally linear in the barrier level. It is the same one-half the
    discretely monitored Monte Carlo estimator shows in `tests/test_barrier_mc.py`,
    because it is the same displacement seen from the other side.

    The amplitude is not a rounding effect: at `n0 = 100` the error swings over
    21% of the option's value as `n` walks through 70 consecutive step counts,
    and the *best* `n` in each window is three decimal orders better than the
    worst.
    """
    amplitudes = [sawtooth[n0]["crr_amplitude"] for n0 in _WINDOW_STARTS]
    assert amplitudes == sorted(amplitudes, reverse=True)
    assert amplitudes[0] / _CONTINUOUS > 0.15  # a fifth of the option's value

    fit = fit_convergence_order(
        np.array([1.0 / n0 for n0 in _WINDOW_STARTS]), np.array(amplitudes)
    )
    assert fit.order == pytest.approx(0.5, abs=_SAWTOOTH_ORDER_BAND)
    assert fit.residual < 0.10
    # Every window's error is one-signed and positive: the effective barrier is
    # always at or below `H`, so the lattice always over-values the knock-out.
    for n0 in _WINDOW_STARTS:
        assert sawtooth[n0]["crr_min"] > 0.0


def test_leisen_reimer_does_not_fix_the_barrier(sawtooth) -> None:
    """Evidence class: NEGATIVE_FINDING, and the prediction held.

    Leisen-Reimer buys two orders of magnitude on a European vanilla and a full
    order on a digital (Slices 3 and 6). On a barrier it buys **2%**: measured
    amplitude ratio LR/CRR 0.977, 0.978, 0.980 at `n0 = 100, 200, 400`, and a
    fitted amplitude order of 0.4542 against CRR's 0.4566.

    The reason is structural. The Peizer-Pratt inversion arranges the terminal
    grid around the *strike*, which is the level a vanilla payoff bends at. A
    barrier is a second level, at a different place, that the construction
    never looks at -- and because `u d != 1` the LR node spots do not even form
    a single geometric ladder, so there is no step count that puts a layer on
    the barrier by construction. Both schemes are limited by the same
    displacement, so both are order one half.
    """
    ratios = [
        sawtooth[n0]["lr_amplitude"] / sawtooth[n0]["crr_amplitude"]
        for n0 in _WINDOW_STARTS
    ]
    assert all(0.95 < ratio < 1.02 for ratio in ratios), ratios
    fit = fit_convergence_order(
        np.array([1.0 / n0 for n0 in _WINDOW_STARTS]),
        np.array([sawtooth[n0]["lr_amplitude"] for n0 in _WINDOW_STARTS]),
    )
    assert fit.order == pytest.approx(0.5, abs=_SAWTOOTH_ORDER_BAND)
    # Not a claim that the two schemes are the same engine: at a fixed `n` they
    # differ by 1.06e-02 at n = 101 and 1.39e-03 at n = 801.
    assert abs(_tree(101).value - _tree(101, "leisen-reimer").value) > 1e-3


# --------------------------------------------------------------------------
# (3) and (4): the Boyle-Lau subsequence.
# --------------------------------------------------------------------------


def test_the_boyle_lau_subsequence_is_first_order_but_not_monotone(boyle_lau) -> None:
    """Evidence class: CONVERGENCE_ORDER, with the slice's expectation corrected.

    Choosing `n_k = floor(k^2 sigma^2 T / log(S/H)^2)` puts layer `k` within a
    hair of the barrier and removes three to four decimal orders of error:

        k       4        7        10        13        16
        n     190      582      1187      2007      3040
        err +1.38e-04 -6.38e-05 +1.43e-03 +4.92e-04 +3.37e-04

    against +0.64 for the unaligned `n = 200`. The block-RMS fit over three
    groups of the thirteen levels gives order **1.1258** (log-space residual
    0.0041) -- first order, as it should be.

    **It is not monotone**, which is what the slice expected. The signs alone
    disprove it (negative at `k = 7` and `k = 14`, positive elsewhere), and the
    reason is that `floor` leaves a residual misalignment: `1 - H_eff/H` is
    measured at 1.14e-07 for `k = 7` and 8.14e-05 for `k = 5`, a spread of
    nearly three decimal orders that behaves like a uniform draw on the
    fractional part. Since the residual misalignment is itself `O(1/n)` with a
    coefficient in `[0, 1)`, the error is `O(1/n)` with an erratic constant --
    `|err * n|` stays inside `[0.025, 1.70]` across all thirteen levels, which
    is the assumption-free form of the same statement and is asserted below.
    """
    steps, errors = boyle_lau["n"], boyle_lau["crr"]
    assert min(errors) < 0.0 < max(errors)  # not one-signed, hence not monotone
    assert max(abs(e) for e in errors) < 6e-3
    assert abs(_tree(200).value - _CONTINUOUS) > 50.0 * max(abs(e) for e in errors)

    scaled = [abs(e) * n for e, n in zip(errors, steps, strict=True)]
    assert max(scaled) < 3.0
    assert min(scaled) > 1e-3

    h, rms = _block_rms(errors, steps, _GROUPS)
    fit = fit_convergence_order(h, rms)
    assert fit.order == pytest.approx(1.0, abs=0.25)
    assert fit.residual < 0.05

    # The per-`n` fit is reported as uninformative rather than suppressed.
    raw = fit_convergence_order(
        np.array([1.0 / n for n in steps]), np.abs(errors)
    )
    assert raw.residual > 1.0


def test_the_boyle_lau_counts_do_nothing_for_leisen_reimer(boyle_lau) -> None:
    """Evidence class: NEGATIVE_FINDING -- the remedy is scheme-specific.

    `n_k` is derived from the CRR ladder `S u^i`, `u = exp(sigma sqrt(dt))`. An
    LR lattice has `u d != 1`, so its nodes are not on that ladder and the same
    step count aligns nothing. Measured at the same `n_k`, the LR errors are
    +0.718, +0.592, +0.110, +0.430, ... -- three decimal orders worse than
    CRR's at the identical step count -- and the block-RMS order is 0.4254,
    i.e. the sawtooth is still there and untouched.

    This is worth pinning because the two halves are easy to conflate: "choose
    `n` to align the barrier" is a statement about a lattice geometry, not
    about a step count, and carrying the count across schemes carries nothing.
    """
    lr_errors = boyle_lau["lr"]
    crr_errors = boyle_lau["crr"]
    assert min(abs(e) for e in lr_errors) > 100.0 * min(abs(e) for e in crr_errors)
    assert np.mean(np.abs(lr_errors)) > 300.0 * np.mean(np.abs(crr_errors))

    h, rms = _block_rms(lr_errors, boyle_lau["n"], _GROUPS)
    fit = fit_convergence_order(h, rms)
    assert fit.order < 0.7  # still the sawtooth, nowhere near first order


def test_the_residual_error_tracks_the_leftover_misalignment(boyle_lau) -> None:
    """Evidence class: CLOSED_FORM on the mechanism, measured on the data.

    If the Boyle-Lau residual is the leftover barrier misalignment and nothing
    else, then `err / (1 - H_eff/H)` should be roughly constant across `k`
    while `err` itself is not. Measured ratios span 24 to 162 over the twelve
    levels where the misalignment is resolvable (a factor of 7), while the
    errors themselves span 1.1e-05 to 4.9e-03 (a factor of 460). That is the
    mechanism accounting for the great majority of the scatter, which is what
    the claim needs and all it claims.
    """
    errors = boyle_lau["crr"]
    misalignment = boyle_lau["misalignment"]
    resolvable = [
        (e, m) for e, m in zip(errors, misalignment, strict=True) if m > 1e-6
    ]
    ratios = [abs(e) / m for e, m in resolvable]
    assert max(ratios) / min(ratios) < 12.0
    error_spread = max(abs(e) for e in errors) / min(abs(e) for e in errors)
    assert error_spread > 100.0
    assert error_spread > 8.0 * (max(ratios) / min(ratios))
