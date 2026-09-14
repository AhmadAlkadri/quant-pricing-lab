"""The smile a simulated Heston price makes, against the transform's own.

A Monte Carlo price is a number with an error bar; an implied volatility is a
number obtained from it by a nonlinear inversion. The error bar has to travel
through the inversion, and it does so through vega: to first order

    sd(sigma_imp) = sd(price) / vega(sigma_imp),

with vega the Black-Scholes vega at the implied volatility itself (not at any
model volatility -- the whole point of an implied volatility is that the
Black-Scholes model is being used as a quoting convention). Every comparison
below is stated as a multiple of that propagated standard error, which is what
makes "the simulated smile matches the transform smile" a checkable claim
rather than a picture.

What is measured, at 100,000 antithetic conditional paths and `dt = 1/32` on
the reference set (`rho = -0.5`, `T = 1`):

    K      MC price          iv_MC      iv_transform   diff        iv stderr
     80    26.75794+-0.0133  0.445924   0.446474       -5.50e-04   4.35e-04
     90    20.91794+-0.0117  0.434197   0.434630       -4.33e-04   3.28e-04
    100    16.05615+-0.0099  0.424121   0.424485       -3.64e-04   2.57e-04
    110    12.11959+-0.0080  0.415483   0.415806       -3.23e-04   2.04e-04
    120     9.01369+-0.0062  0.408108   0.408406       -2.97e-04   1.63e-04

-- every cell within 1.9 propagated standard errors, and every one **negative**,
which is the residual time-discretisation bias showing through: at `dt = 1/64`
and 400,000 paths the same column reads -1.99e-04 / -1.72e-04 / -1.42e-04 /
-1.13e-04 / -8.7e-05, halved along with the step size while the standard errors
fall by only a factor of two as well. The agreement is therefore *statistical*
at these settings and would become a measurable bias at a larger path count,
which is stated rather than hidden behind a passing test.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from heston_points import LEWIS

from qpl.engines.analytic.black_scholes import (
    greeks_european as black_scholes_greeks,
    implied_volatility,
)
from qpl.engines.fourier.pricers import FourierConfig
from qpl.engines.fourier.smile import implied_vol
from qpl.engines.mc.pricers import MCConfig
from qpl.instruments.options import EuropeanOption
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price

SMILE_STRIKES: tuple[float, ...] = (80.0, 90.0, 100.0, 110.0, 120.0)
SMILE_EXPIRY = 1.0
SMILE_PATHS = 100_000
SMILE_STEPS = 32
SMILE_SEED = 4242

TRANSFORM = FourierConfig(method="lewis")
"""Slice 15's conclusion: the two contour methods are the ones with no
parameter to get wrong. Lewis is used here, and the implied volatilities it
produces are the ones `tests/test_heston_smile.py` already pins the shape of."""

STDERR_MULTIPLE = 4.0
"""Worst measured |z| over the five strikes is 1.82; this keeps a factor of 2."""


@lru_cache(maxsize=None)
def _mc_implied_vol(strike: float) -> tuple[float, float]:
    """`(implied vol, its propagated standard error)` of the simulated price.

    Cached: the same five simulations feed three tests, and at a fixed seed
    they are the same numbers every time, so paying for them once is the same
    evidence at a third of the runtime.
    """
    option = EuropeanOption(kind="call", strike=strike, expiry=SMILE_EXPIRY)
    market = LEWIS.market()
    result = price(
        option,
        LEWIS.model(),
        market,
        method="mc",
        cfg=MCConfig(
            n_paths=SMILE_PATHS,
            n_steps=SMILE_STEPS,
            seed=SMILE_SEED,
            variance_reduction="antithetic",
            heston_conditional=True,
        ),
    )
    vol = implied_volatility(result.value, option, market)
    vega = black_scholes_greeks(option, BlackScholesModel(sigma=vol), market).vega
    return vol, float(result.stderr) / vega


@lru_cache(maxsize=None)
def _transform_implied_vol(strike: float) -> float:
    return implied_vol(
        LEWIS.model(),
        LEWIS.market(),
        strike=strike,
        expiry=SMILE_EXPIRY,
        cfg=TRANSFORM,
    )


@pytest.mark.parametrize("strike", SMILE_STRIKES)
def test_the_simulated_implied_vol_matches_the_transform_within_its_own_noise(
    strike: float,
) -> None:
    """Evidence class: STATISTICAL, with the tolerance propagated through vega."""
    mc_vol, mc_stderr = _mc_implied_vol(strike)
    assert abs(mc_vol - _transform_implied_vol(strike)) < STDERR_MULTIPLE * mc_stderr


def test_the_simulated_smile_has_the_same_skew_sign_as_the_transform_smile() -> None:
    """Evidence class: STATISTICAL for the levels, CLOSED_FORM for the shape.

    `rho < 0` makes down moves coincide with volatility-up moves, which fattens
    the left tail and lifts low-strike implied volatilities: the smile is
    strictly decreasing in strike (Gatheral 2006, chapter 2; measured for the
    transform in `tests/test_heston_smile.py`). The simulated smile has to
    reproduce that **shape**, and not merely land near the transform strike by
    strike -- a set of five independent prices each within noise could still be
    non-monotone, and that would say the simulation had lost the correlation.
    """
    mc_vols = np.array([_mc_implied_vol(k)[0] for k in SMILE_STRIKES])
    transform_vols = np.array([_transform_implied_vol(k) for k in SMILE_STRIKES])
    assert np.all(np.diff(mc_vols) < 0.0)
    assert np.all(np.diff(transform_vols) < 0.0)
    # And the slopes agree, not just the signs: the whole smile is tilted the
    # same way by the same amount.
    mc_slope = float(mc_vols[-1] - mc_vols[0])
    transform_slope = float(transform_vols[-1] - transform_vols[0])
    assert mc_slope == pytest.approx(transform_slope, rel=0.02)


def test_the_residual_smile_error_is_a_bias_and_not_noise() -> None:
    """Evidence class: NEGATIVE_FINDING, pinned as a measurement.

    All five differences are negative at `dt = 1/32`. Five independent
    statistical errors would agree in sign with probability 1/16, so this is
    the discretisation bias leaking through the inversion rather than a run of
    luck, and it halves when the step size does (module docstring). A caller
    calibrating to simulated prices would inherit it.
    """
    differences = [
        _mc_implied_vol(k)[0] - _transform_implied_vol(k) for k in SMILE_STRIKES
    ]
    assert all(difference < 0.0 for difference in differences)
    assert max(abs(d) for d in differences) < 1e-03
