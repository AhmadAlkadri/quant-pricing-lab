"""Carr-Madan under Black-Scholes: the damping, the FFT grid, and the rules.

Four claims are measured here.

(a) **The direct quadrature is exact.** At the package defaults (trapezoid, 512
    intervals, reach derived from the model's variance) the damped-transform
    integral reproduces the Black-Scholes price to 1.9e-10 at worst over ten
    (point, kind) cells, and to ~2e-14 at nine of them. CLOSED_FORM.

(b) **The FFT variant's error is interpolation, not integration.** At Carr and
    Madan's recommended settings the same integral evaluated *on* the FFT's
    log-strike grid is accurate to 2.2e-07, and the linearly interpolated value
    at a strike between two nodes is 6.3e-04 -- a factor of 2900. The knob that
    fixes one breaks the other. NEGATIVE_FINDING.

(c) **The damping parameter alpha.** The slice statement predicted a failure at
    large alpha over {0.5, 1, 1.5, 3, 10}. There is no failure anywhere in that
    range: all five are at the floating-point floor at every point tested. The
    two real failure modes are *outside* it, they are asymmetric, and only one
    of them is about alpha being large. NEGATIVE_FINDING.

(d) **The Chapter 6 quadrature rules, on a pricing integral.** The composite
    trapezoid, composite Simpson and Gauss-Legendre rules from
    `qpl.numerics.quadrature` have never been measured on anything but smooth
    finite-interval integrands, where they are order 2, order 4 and
    exponentially accurate. On *this* integrand -- real, even, analytic, and
    Gaussian-decaying -- the trapezoid rule is spectrally accurate and Simpson's
    rule is six decimal orders of magnitude **worse** than it at the same node
    count. Neither of those is the nominal order, and the reason is derived.
    CONVERGENCE_ORDER and NEGATIVE_FINDING.

Source: Carr and Madan (1999), Journal of Computational Finance 2(4), 61-73.
Every number below is measured here; none is quoted from that paper.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
from fourier_points import KINDS, POINTS, Point

from qpl.engines.fourier import (
    U_MAX_STANDARD_DEVIATIONS,
    FourierConfig,
    carr_madan_fft,
    characteristic_function_model,
    damped_call_transform,
    default_u_max,
)
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import DigitalOption, EuropeanOption
from qpl.pricing import price
from qpl.validation import EvidenceClass, fit_convergence_order

QUADRATURE_TOLERANCE = 5e-10
"""Worst measured 1.93e-10, at `short_atm`. That point is the one whose derived
reach is largest (T = 0.05 and sigma = 30% give `12 / sqrt(c2) = 178.9`), so at
a fixed 512 intervals its step is 0.35 against 0.12 at the reference point and
it is the only cell not yet at the floating-point floor. Nine of the ten cells
measure ~2e-14."""


def _cm(point: Point, kind: str = "call", **kwargs) -> float:
    return price(
        point.instrument(kind, "vanilla"),
        point.model(),
        point.market(),
        method="fourier",
        cfg=FourierConfig(method="carr_madan", **kwargs),
    ).value


def _quad(point: Point, kind: str = "call", **kwargs) -> float:
    return _cm(point, kind, carr_madan_transform="quadrature", **kwargs)


def _exact(point: Point, kind: str = "call", strike: float | None = None) -> float:
    option = EuropeanOption(
        kind=kind,
        strike=point.strike if strike is None else strike,
        expiry=point.expiry,
    )
    return price(option, point.model(), point.market()).value


def _grid(point: Point, *, eta: float = 0.25, weights: str = "simpson", alpha: float = 1.5):
    return carr_madan_fft(
        characteristic_function_model(point.model()),
        s0=point.spot,
        expiry=point.expiry,
        rate=point.rate,
        dividend=point.dividend,
        alpha=alpha,
        n_grid=4096,
        eta=eta,
        weights=weights,
    )


# --------------------------------------------------------------------------
# (a) The integral.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("kind", KINDS)
def test_direct_quadrature_matches_the_closed_form(point: Point, kind: str) -> None:
    """Evidence class: CLOSED_FORM."""
    assert _quad(point, kind) == pytest.approx(
        _exact(point, kind), abs=QUADRATURE_TOLERANCE
    )


def test_the_put_is_parity_and_the_test_says_so() -> None:
    """NOT evidence that the method is right -- a statement about the code.

    Carr-Madan transforms the *call* price, so this package's Carr-Madan put is
    `C - S e^{-qT} + K e^{-rT}` and put-call parity holds to round-off by
    construction. The same is true of Lewis and Gil-Pelaez. COS is the only
    method here whose put comes from its own payoff coefficients, and therefore
    the only one where a parity test is evidence rather than arithmetic. The
    residual is asserted to be at round-off precisely so that this stays a
    documented property of the implementation and not a claim about the method.
    """
    point = POINTS[0]
    call, put = _quad(point, "call"), _quad(point, "put")
    forward = point.spot * math.exp(-point.dividend * point.expiry)
    strike_leg = point.strike * math.exp(-point.rate * point.expiry)
    assert abs((call - put) - (forward - strike_leg)) < 1e-13


def test_default_reach_is_derived_from_the_models_variance() -> None:
    """A fixed `u_max` would be wrong by an order of magnitude across the points."""
    reaches = {}
    for point in POINTS:
        cf = characteristic_function_model(point.model())
        reach = default_u_max(
            cf, point.expiry, rate=point.rate, dividend=point.dividend
        )
        expected = U_MAX_STANDARD_DEVIATIONS / (point.sigma * math.sqrt(point.expiry))
        assert math.isclose(reach, expected, rel_tol=1e-12)
        reaches[point.name] = reach
    assert max(reaches.values()) / min(reaches.values()) > 8.0


def test_the_integrand_is_even_which_is_what_makes_trapezoid_spectral() -> None:
    """CLOSED_FORM, and it is the mechanism behind claim (d).

    `psi_T(-v) = conj(psi_T(v))` because the damped price is real, so the
    integrand `Re[e^{-i v k} psi_T(v)]` is an **even** function of `v`. Every
    odd derivative therefore vanishes at `v = 0`, and the Gaussian decay kills
    the integrand and all its derivatives long before the truncation point. The
    Euler-Maclaurin correction terms of the trapezoid rule are exactly those two
    sets of boundary derivatives, so every one of them is zero and the rule's
    error is the aliasing term alone -- which for an analytic integrand decays
    faster than any power of the step. That is the whole explanation of the
    order table below, and it is a property of the integrand, not of the rule.
    """
    point = POINTS[0]
    cf = characteristic_function_model(point.model())
    log_strike = math.log(point.strike)

    def integrand(v: np.ndarray) -> np.ndarray:
        transform = damped_call_transform(
            cf,
            v,
            s0=point.spot,
            expiry=point.expiry,
            rate=point.rate,
            dividend=point.dividend,
            alpha=1.5,
        )
        return np.real(np.exp(-1j * v * log_strike) * transform)

    v = np.array([0.25, 1.0, 4.0, 9.0])
    assert np.allclose(integrand(v), integrand(-v), rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------------
# (b) The FFT grid.
# --------------------------------------------------------------------------

ETA_LEVELS = (0.05, 0.10, 0.25, 0.50)


def test_the_fft_grid_is_the_reciprocal_of_the_transform_grid() -> None:
    """`lambda = 2 pi / (N eta)` and the grid spans `[-pi/eta, pi/eta)`.

    This is not a convention, it is what makes the sampled integral a DFT: the
    strike spacing is forced by the transform spacing, and that forcing is the
    source of every FFT error measured below.
    """
    point = POINTS[0]
    grid = _grid(point, eta=0.25)
    assert math.isclose(grid.spacing, 2.0 * math.pi / (4096 * 0.25), rel_tol=1e-12)
    assert math.isclose(grid.log_strikes[0], -math.pi / 0.25, rel_tol=1e-12)
    assert math.isclose(grid.reach(), 4096 * 0.25, rel_tol=1e-12)


def test_a_strike_outside_the_grid_is_refused_with_the_remedy() -> None:
    """`eta = 1` puts the half-width at 3.14, and `ln 100 = 4.61`."""
    point = POINTS[0]
    with pytest.raises(InvalidInputError, match="outside the FFT grid"):
        _cm(point, eta=1.0)


def test_off_grid_interpolation_dominates_the_fft_error() -> None:
    """NEGATIVE_FINDING, and the headline one for this variant.

    At the recommended `N = 4096, eta = 0.25`, measured at `atm_1y`:

        on the grid (strike = exp(node))      -2.17e-07
        interpolated between two nodes        +6.30e-04

    a factor of 2900. Nothing about the transform got worse between those two
    lines -- it is the *same FFT output*. The linear interpolation of a convex
    function over a spacing of `lambda = 6.14e-03` in log-strike is the entire
    difference, and its size is about `C''(k) lambda^2 / 8`.
    """
    point = POINTS[0]
    grid = _grid(point, eta=0.25)
    node = int(np.argmin(np.abs(grid.log_strikes - math.log(point.strike))))
    on_grid_strike = math.exp(grid.log_strikes[node])

    on_grid = abs(grid.values[node] - _exact(point, "call", strike=on_grid_strike))
    off_grid = abs(_cm(point, eta=0.25) - _exact(point, "call"))

    assert on_grid < 1e-06
    assert off_grid > 1e-04
    assert off_grid / on_grid > 100.0


def test_the_engine_reports_whether_it_interpolated() -> None:
    """A caller has to be able to tell which of the two numbers above they got."""
    point = POINTS[0]
    grid = _grid(point, eta=0.25)
    node = int(np.argmin(np.abs(grid.log_strikes - math.log(point.strike))))
    on_grid_strike = math.exp(grid.log_strikes[node])

    aligned = price(
        EuropeanOption(kind="call", strike=on_grid_strike, expiry=point.expiry),
        point.model(),
        point.market(),
        method="fourier",
        cfg=FourierConfig(method="carr_madan"),
    )
    assert aligned.meta is not None
    assert aligned.meta["on_grid"] is True

    requested = price(
        point.instrument("call", "vanilla"),
        point.model(),
        point.market(),
        method="fourier",
        cfg=FourierConfig(method="carr_madan"),
    )
    assert requested.meta is not None
    assert requested.meta["on_grid"] is False
    assert 0.0 < requested.meta["interpolation_weight"] < 1.0


def test_eta_trades_the_two_fft_errors_against_each_other() -> None:
    """NEGATIVE_FINDING: at fixed `N` there is no setting that makes both small.

    `eta` is the transform-grid spacing and `lambda = 2 pi / (N eta)` is the
    strike-grid spacing, so they move in opposite directions. Measured at
    `atm_1y`, `N = 4096`:

        eta    lambda     on grid     interpolated
        0.05   3.07e-02  -1.33e-13      +6.11e-03
        0.10   1.53e-02  +1.03e-13      +2.65e-03
        0.25   6.14e-03  -2.17e-07      +6.30e-04
        0.50   3.07e-03  -2.68e-03      -2.65e-03

    Reading down the two columns: the integral gets worse and the interpolation
    gets better, and by `eta = 0.5` the integration error has caught up with the
    interpolation error, so the interpolated column stops improving. `N` is the
    only way out, and it costs an FFT of length `N`.
    """
    point = POINTS[0]
    on_grid, interpolated = [], []
    for eta in ETA_LEVELS:
        grid = _grid(point, eta=eta)
        node = int(np.argmin(np.abs(grid.log_strikes - math.log(point.strike))))
        on_grid.append(
            abs(grid.values[node] - _exact(point, "call", strike=math.exp(grid.log_strikes[node])))
        )
        interpolated.append(abs(_cm(point, eta=eta) - _exact(point, "call")))

    # Integration error grows with eta over the last three levels (the first is
    # already at the floating-point floor, where the ordering is noise).
    assert all(a < b for a, b in pairwise(on_grid[1:])), on_grid
    # Interpolation error falls with eta until the integration error catches up.
    assert interpolated[0] > interpolated[1] > interpolated[2]
    assert interpolated[3] > interpolated[2]


def test_carr_madans_own_simpson_weighting_is_the_dominant_fft_error() -> None:
    """NEGATIVE_FINDING, and the same root cause as the rule study below.

    The paper weights the sampled integral by Simpson's rule. Measured on-grid
    error at `atm_1y`, `N = 4096`, against the plain trapezoid weighting of the
    same samples:

        eta     Simpson      trapezoid    ratio
        0.10   +1.03e-13    +1.03e-13     1.0
        0.25   -2.17e-07    +7.46e-14     2.9e+06
        0.50   -2.68e-03    +6.51e-07     4.1e+03
        1.00   -2.89e-01    +8.07e-03     3.6e+01

    The rule that is nominally two orders higher is up to six decimal orders of
    magnitude worse, and it is worse for a reason that has nothing to do with
    smoothness: see `test_simpson_is_a_romberg_combination_and_pays_for_it`.
    At `eta = 0.1` both are at the floor, which is why the ratio there is 1.
    """
    point = POINTS[0]
    ratios = []
    for eta in (0.25, 0.5):
        errors = {}
        for weights in ("simpson", "trapezoid"):
            grid = _grid(point, eta=eta, weights=weights)
            node = int(np.argmin(np.abs(grid.log_strikes - math.log(point.strike))))
            errors[weights] = abs(
                grid.values[node]
                - _exact(point, "call", strike=math.exp(grid.log_strikes[node]))
            )
        ratios.append(errors["simpson"] / errors["trapezoid"])
    assert all(ratio > 100.0 for ratio in ratios), ratios


# --------------------------------------------------------------------------
# (c) The damping parameter.
# --------------------------------------------------------------------------

SLICE_ALPHAS = (0.5, 1.0, 1.5, 3.0, 10.0)
"""The range the slice statement named and expected a failure inside."""


@pytest.mark.parametrize("point", POINTS[:3], ids=[p.name for p in POINTS[:3]])
def test_no_failure_anywhere_in_the_predicted_alpha_range(point: Point) -> None:
    """NEGATIVE_FINDING: the predicted failure does not happen.

    Measured error at `alpha` in {0.5, 1, 1.5, 3, 10}, direct quadrature at the
    defaults, over the three points: every cell is at or below 5.5e-10, and all
    but the two `alpha = 0.5` cells are at ~1e-14. Under Black-Scholes
    `E[S_T^{alpha+1}]` is finite for **every** alpha, so the integrability
    condition that usually limits the damping never binds; what limits it here
    is floating point, and that is a different statement with a different
    threshold (the next two tests).
    """
    errors = [abs(_quad(point, alpha=alpha) - _exact(point)) for alpha in SLICE_ALPHAS]
    assert max(errors) < 1e-09, dict(zip(SLICE_ALPHAS, errors, strict=True))


@pytest.mark.parametrize("point", POINTS[:3], ids=[p.name for p in POINTS[:3]])
def test_small_alpha_is_the_failure_mode_the_slice_missed(point: Point) -> None:
    """NEGATIVE_FINDING: the damping fails *below* the named range, not above.

    As `alpha -> 0` the denominator `alpha^2 + alpha - v^2 + i(2 alpha + 1) v`
    approaches `-v^2 + i v`, which vanishes at `v = 0`: the transform of the
    undamped call price has a pole there, and for small alpha the integrand has
    a spike of height `~1/alpha` and width `~alpha` sitting at the left endpoint.
    A uniform grid of 512 intervals over a reach of 60 cannot see a feature of
    width 0.05, so the trapezoid rule misses most of its mass. Measured at
    `atm_1y`: 7.4e+00 at alpha = 0.05, 4.7e-01 at 0.1, 1.5e-04 at 0.25,
    2.3e-10 at 0.5. This is a resolution failure, not an integrability one.
    """
    tiny = abs(_quad(point, alpha=0.05) - _exact(point))
    small = abs(_quad(point, alpha=0.1) - _exact(point))
    usable = abs(_quad(point, alpha=1.5) - _exact(point))
    assert tiny > 0.1
    assert tiny > small > usable
    assert usable < 1e-12


def test_large_alpha_fails_by_cancellation_and_the_threshold_is_moneyness() -> None:
    """NEGATIVE_FINDING with a derived cause, measured at three moneyness levels.

    The price is `e^{-alpha ln K} / pi` times an integral whose size is set by
    `psi_T(0) ~ e^{-rT} E[S_T^{alpha+1}] / (alpha^2 + alpha)`. Their product is
    the price, but each factor separately is enormous: the relative round-off
    of the difference is about

        (S_0 / K)^alpha * exp(alpha^2 c2 / 2) * S_0 / price * eps,

    with `c2 = sigma^2 T`. That is an upper bound on the damage and it is
    *moneyness-dependent*, which is why the three points fail at three different
    alphas. Measured error and the bound:

        point          alpha=20            alpha=30           alpha=40
        atm_1y        -1.03e-13 (6e-12)   -1.60e-09 (1e-07)  -3.83e-03 (2e-01)
        otm_9m_div    +1.63e-13 (7e-12)   -3.36e-09 (3e-07)  +3.14e-02 (2e+00)
        itm_1y_div    +3.04e-04 (1e-02)   +4.16e+10 (4e+12)  -2.41e+31 (3e+32)

    The in-the-money point (`S_0 / K = 1.333`) is unusable by alpha = 20 while
    the at-the-money one is still exact; the bound predicts that ordering and
    every measured error sits inside it, one to two orders of magnitude below.
    """
    eps = float(np.finfo(float).eps)
    for point in POINTS[:3]:
        exact = _exact(point)
        for alpha in (20.0, 30.0, 40.0):
            error = abs(_quad(point, alpha=alpha) - exact)
            bound = (
                (point.spot / point.strike) ** alpha
                * math.exp(0.5 * alpha**2 * point.sigma**2 * point.expiry)
                * point.spot
                / exact
                * eps
            )
            assert error <= bound, (point.name, alpha, error, bound)

    itm = POINTS[2]
    assert abs(_quad(itm, alpha=20.0) - _exact(itm)) > 1e-05
    assert abs(_quad(POINTS[0], alpha=20.0) - _exact(POINTS[0])) < 1e-11


# --------------------------------------------------------------------------
# (d) The Chapter 6 rules on this pricing integral.
# --------------------------------------------------------------------------

RULE_LEVELS = (8, 16, 32, 64, 128, 256)

NOMINAL_ORDERS = {"trapezoid": 2.0, "simpson": 4.0}
"""What the composite rules are on a generic smooth integrand."""


@pytest.mark.parametrize("rule", ("trapezoid", "simpson", "gauss_legendre"))
def test_no_rule_converges_at_its_nominal_order_on_this_integral(rule: str) -> None:
    """CONVERGENCE_ORDER, and the answer is "faster than any order".

    Measured at `atm_1y`, alpha = 1.5, reach 60:

        n         8          16         32         64        128        256
        trap   +2.64e+01  +7.35e+00  +6.39e-01  +4.30e-03  +1.85e-07  +1.60e-14
        simp   +1.41e+01  +9.91e-01  -1.60e+00  -2.07e-01  -1.43e-03  -6.18e-08
        gauss  +9.34e-01  -1.00e-01  -8.59e-05  +2.22e-11  +2.90e-12  +9.68e-13

    Fitted orders over the above-floor points: 6.49 (trapezoid, nominal 2), 4.86
    (Simpson, nominal 4) and 9.31 (Gauss-Legendre). Every one of those fits has
    a log-space residual above 2.4, which for a genuine power law would be near
    zero -- the fits are reported because the slice statement asked for them and
    are meaningless as orders. The honest statement is the ratio test asserted
    below: the improvement per doubling keeps growing, which no `C / n^p` does.
    This is the Fusai Chapter 6 toolkit meeting a pricing integral, and the
    smooth-function test that toolkit was built against could not have shown it.
    """
    point = POINTS[0]
    errors = [
        abs(_quad(point, quadrature=rule, n_quad=n) - _exact(point))
        for n in RULE_LEVELS
    ]
    above_floor = [(n, e) for n, e in zip(RULE_LEVELS, errors, strict=True) if e > 1e-10]
    assert len(above_floor) >= 3, errors

    fit = fit_convergence_order(
        [1.0 / n for n, _ in above_floor], [e for _, e in above_floor]
    )
    assert fit.order > NOMINAL_ORDERS.get(rule, 4.0), (rule, fit.order)
    assert fit.residual > 1.0, (rule, fit.residual)

    tail = [e for _, e in above_floor[-3:]]
    drops = [math.log10(a / b) for a, b in pairwise(tail)]
    assert drops[-1] > drops[0], (rule, drops)


def test_simpson_is_a_romberg_combination_and_pays_for_it() -> None:
    """EXACT_IDENTITY, and it is the whole explanation of Simpson's failure.

    On a uniform grid, composite Simpson at `n` intervals is exactly the Romberg
    combination of two trapezoid rules:

        S_n = (4 T_n - T_{n/2}) / 3,

    so its *error* is `(4 (T_n - I) - (T_{n/2} - I)) / 3`. That combination is
    worth having when the trapezoid error is `O(h^2)`, because the leading terms
    cancel. Here the trapezoid error is not `O(h^2)` -- it is spectrally small --
    so `T_n - I` is negligible against `T_{n/2} - I` and Simpson's error is just

        S_n - I ~= -(T_{n/2} - I) / 3.

    In other words Simpson throws away the accurate rule and keeps a third of
    the inaccurate one. Measured at `atm_1y`:

        n      Simpson error    -(trapezoid error at n/2)/3
        64     -2.071e-01       -2.129e-01
        128    -1.433e-03       -1.433e-03
        256    -6.178e-08       -6.178e-08

    The identity itself is checked to 3.6e-15.
    """
    point = POINTS[0]
    exact = _exact(point)
    for n in (64, 128, 256):
        simpson = _quad(point, quadrature="simpson", n_quad=n)
        fine = _quad(point, quadrature="trapezoid", n_quad=n)
        coarse = _quad(point, quadrature="trapezoid", n_quad=n // 2)
        assert abs(simpson - (4.0 * fine - coarse) / 3.0) < 1e-13
        predicted = -(coarse - exact) / 3.0
        assert simpson - exact == pytest.approx(predicted, rel=0.05)


def test_trapezoid_beats_simpson_by_six_decimal_orders_at_the_same_cost() -> None:
    """NEGATIVE_FINDING: the nominally higher-order rule is the wrong default.

    At `n = 256` the trapezoid error is 1.60e-14 and Simpson's is 6.18e-08, at
    exactly the same number of integrand evaluations. That is why
    `FourierConfig.quadrature` defaults to `"trapezoid"` and the default is a
    measurement rather than a habit.
    """
    point = POINTS[0]
    exact = _exact(point)
    trapezoid = abs(_quad(point, quadrature="trapezoid", n_quad=256) - exact)
    simpson = abs(_quad(point, quadrature="simpson", n_quad=256) - exact)
    assert simpson / trapezoid > 1e05
    assert FourierConfig().quadrature == "trapezoid"


def test_gauss_legendre_reaches_the_floor_first_and_then_stops() -> None:
    """Spectral too, and cheapest to the floor -- but its floor is higher.

    Gauss-Legendre is at 2.2e-11 by 64 nodes, where the trapezoid rule is still
    at 4.3e-03; but it then stalls at ~1e-12 while the trapezoid rule reaches
    1.6e-14 at 256. The nodes are irrational and the weights are computed, so
    the rule carries its own evaluation error; the trapezoid rule's abscissae
    and weights are exact.
    """
    point = POINTS[0]
    exact = _exact(point)
    assert abs(_quad(point, quadrature="gauss_legendre", n_quad=64) - exact) < 1e-10
    assert abs(_quad(point, quadrature="trapezoid", n_quad=64) - exact) > 1e-04
    assert abs(_quad(point, quadrature="gauss_legendre", n_quad=256) - exact) > 1e-13
    assert abs(_quad(point, quadrature="trapezoid", n_quad=256) - exact) < 1e-13


# --------------------------------------------------------------------------
# Refusals and configuration.
# --------------------------------------------------------------------------


def test_carr_madan_has_no_digital_payoff() -> None:
    point = POINTS[0]
    option = DigitalOption(kind="call", strike=100.0, expiry=1.0, cash=1.0)
    with pytest.raises(NotSupportedError, match="prices the call transform"):
        price(
            option,
            point.model(),
            point.market(),
            method="fourier",
            cfg=FourierConfig(method="carr_madan"),
        )


def test_carr_madan_has_no_greeks_and_the_message_names_cos() -> None:
    from qpl.pricing import greeks

    point = POINTS[0]
    with pytest.raises(NotSupportedError, match="FourierConfig\\(method='cos'\\)"):
        greeks(
            point.instrument("call", "vanilla"),
            point.model(),
            point.market(),
            method="fourier",
            cfg=FourierConfig(method="carr_madan"),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha must be finite and > 0"),
        ({"alpha": -1.0}, "alpha must be finite and > 0"),
        ({"carr_madan_transform": "dft"}, "carr_madan_transform must be one of"),
        ({"n_grid": 3}, "n_grid must be an even integer >= 4"),
        ({"n_grid": 4095}, "n_grid must be an even integer >= 4"),
        ({"eta": 0.0}, "eta must be finite and > 0"),
        ({"fft_weights": "gauss"}, "fft_weights must be one of"),
        ({"quadrature": "romberg"}, "quadrature must be one of"),
        ({"n_quad": 1}, "n_quad must be an integer >= 2"),
        ({"quadrature": "simpson", "n_quad": 65}, "n_quad must be even"),
        ({"u_max": 0.0}, "u_max must be None or finite and > 0"),
    ],
)
def test_config_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        FourierConfig(**kwargs)


def test_zero_volatility_is_refused_by_the_reach_rule() -> None:
    from qpl.models.black_scholes import BlackScholesModel

    point = POINTS[0]
    with pytest.raises(InvalidInputError, match="positive variance"):
        price(
            point.instrument("call", "vanilla"),
            BlackScholesModel(sigma=0.0),
            point.market(),
            method="fourier",
            cfg=FourierConfig(method="carr_madan", carr_madan_transform="quadrature"),
        )


def test_metadata_records_the_variant_and_its_settings() -> None:
    point = POINTS[0]
    result = price(
        point.instrument("call", "vanilla"),
        point.model(),
        point.market(),
        method="fourier",
        cfg=FourierConfig(
            method="carr_madan",
            carr_madan_transform="quadrature",
            quadrature="gauss_legendre",
            n_quad=128,
            alpha=1.25,
        ),
    )
    assert result.meta is not None
    assert result.meta["fourier_method"] == "carr_madan"
    assert result.meta["carr_madan_transform"] == "quadrature"
    assert result.meta["quadrature"] == "gauss_legendre"
    assert result.meta["n_quad"] == 128
    assert result.meta["alpha"] == 1.25
    assert math.isclose(result.meta["u_max"], U_MAX_STANDARD_DEVIATIONS / 0.2, rel_tol=1e-12)


def test_evidence_classes_referenced_by_this_module_exist() -> None:
    assert EvidenceClass.NEGATIVE_FINDING.value == "negative_finding"
    assert EvidenceClass.CONVERGENCE_ORDER.value == "convergence_order"
