"""Asians and barriers under Heston: what the Black-Scholes engines relied on.

Both path-dependent Monte Carlo engines in this package were built on one
property of geometric Brownian motion, and both lose it here.

**The Asian lost its control variate.** Kemna and Vorst's geometric average is
lognormal under Black-Scholes, so its option has a closed form and makes a
control variate with correlation 0.9996 and a measured variance factor of 1277
(Slice 8). Under Heston the geometric average of the fixings is lognormal only
*conditional on the variance path*, so there is no closed form and no exact
mean to subtract. The fallback is the discounted terminal spot, measured
correlation **0.63**, factor **1.7** -- three decimal orders of magnitude worse,
and the engine says so in `meta["geometric_control_unavailable"]` rather than
quietly reporting a smaller number.

**The barrier's Brownian bridge stopped being exact.** Under Black-Scholes the
bridge estimator is *unbiased for the continuous contract at any number of
sampling dates* -- Slice 12 measured it pricing the continuous barrier from a
**single** observation. That rests on the bridge between two sampled points
having constant volatility, which is what makes the reflection-principle
crossing probability exact. Under Heston the volatility varies inside the step,
the engine feeds the bridge the trapezoidal integrated variance
`dt (v_i + v_{i+1}) / 2`, and the result is a one-term approximation. Measured
on a near-the-money barrier (`H = 95`, `S_0 = 100`, `K = 100`, `T = 1`, the
reference parameter set) at 400,000 antithetic paths with the terminal-spot
control:

    n_steps     price     stderr    mean survival
        25     4.63838   0.02376       0.08711
        50     4.51679   0.02424       0.08402
       100     4.41176   0.02452       0.08228
       200     4.38643   0.02496       0.08142
       400     4.31406   0.02508       0.08025
       800     4.31367   0.02528       0.07994

-- a residual of **+0.325 at `n_steps = 25`** against the finest grid, 7.5% of
the price and thirteen standard errors, falling at roughly first order in the
step size. A caller who carried the Black-Scholes reading of
`barrier_correction="brownian_bridge"` across to Heston would believe that row
was already the continuous price.

The discretely monitored contract is a different matter: `"none"` is unbiased
for it *given the scheme*, and the discrete-versus-continuous gap is the
contract's, not the estimator's -- measured +1.083 at `m = 50`, `H = 80` on the
same set, which is the Slice 12 monitoring bias reappearing under a different
model.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from heston_points import LEWIS

from qpl.engines.mc.pricers import MCConfig
from qpl.exceptions import NotSupportedError
from qpl.instruments.options import (
    AsianOption,
    BarrierOption,
    uniform_fixing_times,
    uniform_monitoring_times,
)
from qpl.pricing import price

PATHS = 60_000
BRIDGE_PATHS = 160_000
"""The bridge residual is the one claim here whose signal is not large next to
its own noise: the barrier payoff's standard error at 60,000 paths is 6e-02 and
the residual being measured is 2e-01, so this family alone pays for a larger
sample. Everything else in this file is a gap of 1.08 or a correlation, and
60,000 paths resolve those with room to spare."""

SEED = 31
_VARIANCE_REDUCTION = ("antithetic", "control_variate")


def _cfg(n_steps: int, **changes) -> MCConfig:
    base = {
        "n_paths": PATHS,
        "n_steps": n_steps,
        "seed": SEED,
        "variance_reduction": _VARIANCE_REDUCTION,
    }
    base.update(changes)
    return MCConfig(**base)


# --------------------------------------------------------------------------
# The Asian.
# --------------------------------------------------------------------------

ASIAN = AsianOption(
    kind="call",
    strike=100.0,
    expiry=1.0,
    fixing_times=uniform_fixing_times(1.0, 12),
    averaging="arithmetic",
)


@lru_cache(maxsize=None)
def _asian(n_steps: int):
    return price(ASIAN, LEWIS.model(), LEWIS.market(), method="mc", cfg=_cfg(n_steps))


def test_the_asian_control_variate_is_three_orders_weaker_than_kemna_vorst() -> None:
    """Evidence class: NEGATIVE_FINDING, pinned as a measurement.

    0.63 against 0.9996, a predicted variance factor of 1.7 against 1277. The
    number is not a defect of this engine -- it is what is left once the model
    stops making the geometric average lognormal -- and it is surfaced on the
    result so a caller comparing an Asian's standard error across models is not
    surprised by two decimal orders.
    """
    result = _asian(48)
    correlation = float(result.meta["control_correlation"])
    factor = float(result.meta["control_variance_factor_predicted"])
    assert 0.5 < correlation < 0.75
    assert 1.4 < factor < 2.2
    assert result.meta["control_variate"] == "discounted_terminal_spot"
    assert "no closed form" in result.meta["geometric_control_unavailable"]


def test_the_asian_price_has_converged_by_four_steps_per_fixing() -> None:
    """Evidence class: STATISTICAL.

    Measured at 200,000 paths: 8.71591 / 8.76645 / 8.78044 / 8.75624 / 8.76054
    at `n_steps` = 12 / 24 / 48 / 96 / 192 with a standard error of 1.93e-02, so
    only the one-step-per-fixing grid is measurably off and everything from 24
    steps up sits inside the noise. The *fixings* are read exactly wherever the
    schedule falls (`heston_time_grid` unions the two grids), so what is
    converging here is the scheme, not the observation dates.
    """
    coarse = _asian(12)
    fine = _asian(48)
    spread = (float(coarse.stderr) ** 2 + float(fine.stderr) ** 2) ** 0.5
    assert abs(coarse.value - fine.value) < 4.0 * spread
    assert fine.meta["n_fixings"] == 12
    assert fine.meta["n_grid"] >= 48


# --------------------------------------------------------------------------
# The barrier.
# --------------------------------------------------------------------------

NEAR_BARRIER = BarrierOption(
    kind="call",
    strike=100.0,
    expiry=1.0,
    barrier=95.0,
    barrier_type="down-and-out",
    monitoring=uniform_monitoring_times(1.0, 25),
)
"""Close enough to the spot that the bridge does most of the work: the mean
survival probability is about 8%, so almost every path's value is decided by a
crossing probability rather than by an observed touch."""

FAR_BARRIER = BarrierOption(
    kind="call",
    strike=100.0,
    expiry=1.0,
    barrier=80.0,
    barrier_type="down-and-out",
    monitoring=uniform_monitoring_times(1.0, 50),
)


@lru_cache(maxsize=None)
def _barrier(option_id: str, n_steps: int, correction: str):
    near = option_id == "near"
    option = NEAR_BARRIER if near else FAR_BARRIER
    return price(
        option,
        LEWIS.model(),
        LEWIS.market(),
        method="mc",
        cfg=_cfg(
            n_steps,
            barrier_correction=correction,
            n_paths=BRIDGE_PATHS if near else PATHS,
        ),
    )


def test_the_brownian_bridge_is_not_unbiased_under_heston() -> None:
    """Evidence class: NEGATIVE_FINDING, pinned.

    The headline of this file. Under Black-Scholes the bridge estimator's mean
    does not depend on the sampling grid at all; under Heston it does, because
    the crossing probability assumes a constant volatility over the step and
    gets an integrated one instead. The residual measured here is against a
    grid four times finer, which is the same comparison
    `docs/notes/heston_monte_carlo_qe.md` runs out to 800 steps.
    """
    coarse = _barrier("near", 25, "brownian_bridge")
    fine = _barrier("near", 100, "brownian_bridge")
    residual = coarse.value - fine.value
    spread = (float(coarse.stderr) ** 2 + float(fine.stderr) ** 2) ** 0.5
    assert residual > 3.0 * spread
    assert residual > 0.1
    # And the direction is the finding, not just the size: the coarse grid
    # *overstates* survival (it cannot see the crossings inside a step that the
    # integrated variance understates), so a knock-out is priced too high.
    assert float(coarse.meta["mean_survival_probability"]) > float(
        fine.meta["mean_survival_probability"]
    )
    assert "one-term approximation" in coarse.meta["brownian_bridge_caveat"]


def test_the_bridge_residual_shrinks_with_the_step_size() -> None:
    """Evidence class: CONVERGENCE_ORDER, read qualitatively.

    Measured at 400,000 paths the residual against `n_steps = 800` runs +0.325
    / +0.203 / +0.098 / +0.073 / +0.0004 over `n_steps` = 25 / 50 / 100 / 200 /
    400, i.e. roughly first order in the step size with a noisy tail. The claim
    asserted here is the monotone part: halving the step halves the residual,
    so the gap is a discretisation of the bridge and not a wiring error that no
    refinement would remove.
    """
    reference = _barrier("near", 100, "brownian_bridge").value
    coarse = _barrier("near", 25, "brownian_bridge").value - reference
    middle = _barrier("near", 50, "brownian_bridge").value - reference
    assert coarse > middle > 0.0


def test_the_discrete_estimator_and_the_bridge_price_different_contracts() -> None:
    """Evidence class: NEGATIVE_FINDING for the size, EXACT_IDENTITY for the sign.

    `"none"` estimates the contract as written -- observed at its 50 monitoring
    dates -- and `"brownian_bridge"` estimates the continuously monitored one.
    A discrete knock-out survives more often than a continuous one, so the
    first is worth more, always; the measured gap is +1.083 at `m = 50` and
    `H = 80`, which is Slice 12's monitoring bias under a second model.
    `meta["estimates"]` says which contract each number belongs to, so a caller
    never has to infer it from the configuration.
    """
    discrete = _barrier("far", 50, "none")
    continuous = _barrier("far", 50, "brownian_bridge")
    assert discrete.meta["estimates"] == "discrete"
    assert continuous.meta["estimates"] == "continuous"
    gap = discrete.value - continuous.value
    spread = (float(discrete.stderr) ** 2 + float(continuous.stderr) ** 2) ** 0.5
    assert gap > 5.0 * spread
    assert gap == pytest.approx(1.08, abs=0.25)


def test_a_touched_barrier_is_refused_rather_than_simulated() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the contract).

    The Black-Scholes engine settles a touched knock-in by falling back to the
    closed-form vanilla. Under Heston the vanilla leg is a *transform* price,
    not a simulated one, so returning it from a Monte Carlo engine would mean
    silently swapping methods; the refusal names the route instead.
    """
    touched = BarrierOption(
        kind="call",
        strike=100.0,
        expiry=1.0,
        barrier=95.0,
        barrier_type="up-and-out",
        monitoring=uniform_monitoring_times(1.0, 12),
    )
    with pytest.raises(NotSupportedError, match="already touched at inception"):
        price(
            touched, LEWIS.model(), LEWIS.market(), method="mc", cfg=_cfg(12)
        )


def test_a_knock_out_rebate_is_refused_on_the_bridge_route() -> None:
    """Evidence class: EXACT_IDENTITY (a statement about the contract).

    The bridge supplies the probability that the path crossed, not the
    distribution of when it did, and a knock-out rebate is paid at the touch
    time. Same refusal as the Black-Scholes engine gives, for the same reason.
    """
    with_rebate = BarrierOption(
        kind="call",
        strike=100.0,
        expiry=1.0,
        barrier=80.0,
        barrier_type="down-and-out",
        rebate=2.0,
        monitoring=uniform_monitoring_times(1.0, 12),
    )
    with pytest.raises(NotSupportedError, match="crossing probability, not the"):
        price(
            with_rebate,
            LEWIS.model(),
            LEWIS.market(),
            method="mc",
            cfg=_cfg(24, barrier_correction="brownian_bridge"),
        )
    # ...and it prices perfectly well on the discrete route, where the touch
    # time is observed.
    result = price(
        with_rebate,
        LEWIS.model(),
        LEWIS.market(),
        method="mc",
        cfg=_cfg(24, n_paths=4_000),
    )
    assert result.value > 0.0
