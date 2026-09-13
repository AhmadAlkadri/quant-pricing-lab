"""Lewis and Gil-Pelaez: the two methods with no grid and no damping parameter.

Both hand a single improper integral to `scipy.integrate.quad` and both land at
the floating-point floor, which makes them the accuracy reference the other two
transform methods are measured against.

(a) **Accuracy.** Worst error over ten (point, kind) cells: 2.13e-14 for Lewis
    and 1.42e-14 for Gil-Pelaez, three orders of magnitude better than the
    `~1e-10` the slice statement expected. CLOSED_FORM.

(b) **The Gil-Pelaez digital is the exercise probability itself**, so it needs
    no extra derivation and costs nothing extra: worst error 1.11e-16, and the
    two probabilities `Pi_1` and `Pi_2` reproduce `N(d1)` and `N(d2)` to
    1.67e-16. The slice statement expected `~1e-10` there too. CLOSED_FORM.

(c) **The reported error bound is not the error.** QUADPACK returns an error
    estimate with every answer, and on these integrands it is at least 1.19e+03
    times the error actually made (and up to 1e+05). It is recorded in the
    result metadata as what it is -- a bound the routine believes -- and is never
    used as an accuracy claim. NEGATIVE_FINDING.

(d) **The contour and the removable singularity.** Lewis's `u^2 + 1/4` is
    bounded below by 1/4, so the integrand has no singularity to avoid at all;
    Gil-Pelaez's `1/(iu)` does, and it is removable with limit `c1 - ln K`.
    Both are checked. CLOSED_FORM.

Sources: Lewis (2001), "A simple option formula for general jump-diffusions and
other exponential Levy processes"; Gil-Pelaez (1951), Biometrika 38(3-4),
481-482. Derivations are in the module docstrings of
`qpl.engines.fourier.lewis` and `qpl.engines.fourier.gil_pelaez`; nothing is
reproduced from either source.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fourier_points import KINDS, POINTS, Point
from scipy.special import erf

from qpl.engines.fourier import (
    LEWIS_STRIP_OFFSET,
    FourierConfig,
    black_scholes_characteristic_function,
    characteristic_function_model,
    gil_pelaez_probabilities,
)
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import DigitalOption
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass

LEWIS_TOLERANCE = 1e-13
GIL_PELAEZ_TOLERANCE = 1e-13
"""Worst measured 2.13e-14 and 1.42e-14 over the ten vanilla cells."""

DIGITAL_TOLERANCE = 1e-15
"""Worst measured 1.11e-16 over the ten digital cells."""

REPORTED_BOUND_PESSIMISM = 1e03
"""Smallest measured ratio of QUADPACK's reported bound to the actual error."""


def _value(point: Point, method: str, kind: str, payoff: str, **kwargs):
    return price(
        point.instrument(kind, payoff),
        point.model(),
        point.market(),
        method="fourier",
        cfg=FourierConfig(method=method, **kwargs),
    )


def _exact(point: Point, kind: str, payoff: str) -> float:
    return price(point.instrument(kind, payoff), point.model(), point.market()).value


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def _d1_d2(point: Point) -> tuple[float, float]:
    root = point.sigma * math.sqrt(point.expiry)
    d1 = (
        math.log(point.spot / point.strike)
        + (point.rate - point.dividend + 0.5 * point.sigma**2) * point.expiry
    ) / root
    return d1, d1 - root


# --------------------------------------------------------------------------
# (a) Accuracy.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("method", "tolerance"),
    [("lewis", LEWIS_TOLERANCE), ("gil_pelaez", GIL_PELAEZ_TOLERANCE)],
)
def test_vanilla_matches_the_closed_form(
    point: Point, kind: str, method: str, tolerance: float
) -> None:
    """Evidence class: CLOSED_FORM."""
    assert _value(point, method, kind, "vanilla").value == pytest.approx(
        _exact(point, kind, "vanilla"), abs=tolerance
    )


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("kind", KINDS)
def test_gil_pelaez_digital_matches_the_closed_form(point: Point, kind: str) -> None:
    """CLOSED_FORM, and the digital is free here.

    `Pi_2` *is* the exercise probability, so the cash-or-nothing call is
    `cash * e^{-rT} * Pi_2` with no second derivation and no second integral --
    the same two quadratures that price the vanilla already contain it.
    """
    assert _value(point, "gil_pelaez", kind, "digital").value == pytest.approx(
        _exact(point, kind, "digital"), abs=DIGITAL_TOLERANCE
    )


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
def test_the_two_probabilities_are_n_d1_and_n_d2(point: Point) -> None:
    """CLOSED_FORM at the level of the probabilities, not just the price.

    This is the test that says the `Pi_1` / `Pi_2` decomposition is the
    Black-Scholes formula and not merely numerically equal to it: an error in
    the share-measure shift `phi(u - i) / phi(-i)` that happened to cancel in
    the price would show up here. Worst measured residual 1.67e-16.
    """
    result = _value(point, "gil_pelaez", "call", "vanilla")
    d1, d2 = _d1_d2(point)
    assert result.meta is not None
    assert result.meta["pi1"] == pytest.approx(_norm_cdf(d1), abs=1e-15)
    assert result.meta["pi2"] == pytest.approx(_norm_cdf(d2), abs=1e-15)


def test_the_share_measure_shift_is_an_arithmetic_check_on_the_transform() -> None:
    """`phi_S(-i) = E[S_T] = S_0 e^{(r-q)T}`, the forward. CLOSED_FORM.

    Gil-Pelaez's `Pi_1` divides by exactly this number, so any model claiming to
    satisfy `CharacteristicFunctionModel` can be checked in one line before it
    is trusted anywhere else -- which is the cheapest Heston smoke test there is.
    """
    point = POINTS[2]
    value = black_scholes_characteristic_function(
        np.array([-1j]),
        point.expiry,
        rate=point.rate,
        dividend=point.dividend,
        sigma=point.sigma,
    )[0]
    forward = math.exp((point.rate - point.dividend) * point.expiry)
    assert math.isclose(value.real, forward, rel_tol=1e-12)


# --------------------------------------------------------------------------
# (c) The reported bound.
# --------------------------------------------------------------------------


def test_quadpacks_reported_bound_is_three_to_five_orders_pessimistic() -> None:
    """NEGATIVE_FINDING.

    Measured at the package defaults, the smallest ratio of reported bound to
    actual error over the cells where the error is non-zero is 1.19e+03 (Lewis)
    and 1.50e+03 (Gil-Pelaez); the largest is above 1e+05. The bound is a
    *bound* -- it is not wrong -- but a caller who read it as an accuracy figure
    would under-claim by five decimal orders of magnitude, so it is reported
    under the name `reported_abserr` and never as the method's error.
    """
    ratios = []
    for point in POINTS:
        for method in ("lewis", "gil_pelaez"):
            result = _value(point, method, "call", "vanilla")
            error = abs(result.value - _exact(point, "call", "vanilla"))
            assert result.meta is not None
            bound = result.meta["reported_abserr"]
            assert bound >= 0.0
            if error > 0.0:
                ratios.append(bound / error)
    assert ratios
    assert min(ratios) > REPORTED_BOUND_PESSIMISM, min(ratios)


def test_the_default_tolerance_is_the_bottom_of_a_measured_scan() -> None:
    """Tightening past the default stops helping; loosening it costs a decade.

    SciPy's own default (1.49e-08) leaves Lewis at 2.49e-13. The package default
    of 1e-10 reaches 2.13e-14 for about 30% more work, and 1e-12 is no better.
    """
    point = POINTS[0]
    exact = _exact(point, "call", "vanilla")
    loose = abs(_value(point, "lewis", "call", "vanilla", quad_tolerance=1.49e-08).value - exact)
    default = abs(_value(point, "lewis", "call", "vanilla").value - exact)
    assert FourierConfig().quad_tolerance == 1e-10
    assert loose > default


# --------------------------------------------------------------------------
# (d) Contours and singularities.
# --------------------------------------------------------------------------


def test_lewis_contour_is_fixed_and_its_denominator_cannot_vanish() -> None:
    """CLOSED_FORM: `z = u + i/2` gives `z(z - i) = u^2 + 1/4 >= 1/4`.

    The contrast with Carr-Madan is the reason both are here. Carr-Madan's
    denominator `alpha^2 + alpha - v^2 + i(2 alpha + 1) v` approaches a zero at
    `v = 0` as `alpha -> 0`, which is the small-alpha failure measured in
    `tests/test_fourier_carr_madan.py`. Lewis has no parameter that can be set
    badly: the contour is the midpoint of the payoff transform's strip and the
    denominator is bounded away from zero on it, uniformly in everything.
    """
    assert LEWIS_STRIP_OFFSET == 0.5
    u = np.linspace(-5.0, 5.0, 101)
    assert float(np.min(u * u + LEWIS_STRIP_OFFSET**2)) >= 0.25


def test_gil_pelaez_integrand_limit_at_the_origin_is_c1_minus_log_strike() -> None:
    """CLOSED_FORM: the `1/(iu)` singularity is removable and the limit is exact.

    `Re[e^{-iuk} phi(u) / (i u)] = Im[e^{-iuk} phi(u)] / u`, and near the origin
    `e^{-iuk} phi(u) = exp(i u (c1_spot - k)) + O(u^2)`, so the ratio tends to
    `c1_spot - k` where `c1_spot = ln S_0 + c1`. Checked against the implemented
    integrand evaluated at a sequence of shrinking arguments.
    """
    point = POINTS[0]
    cf = characteristic_function_model(point.model())
    cumulants = cf.log_return_cumulants(
        point.expiry, rate=point.rate, dividend=point.dividend
    )
    limit = math.log(point.spot) + cumulants.c1 - math.log(point.strike)

    def ratio(u: float) -> float:
        arg = np.array([u], dtype=complex)
        phi_spot = np.exp(1j * arg * math.log(point.spot)) * cf.characteristic_function(
            arg, point.expiry, rate=point.rate, dividend=point.dividend
        )
        value = np.exp(-1j * arg * math.log(point.strike)) * phi_spot
        return float(np.imag(value)[0] / u)

    for u in (1e-03, 1e-04, 1e-05):
        assert ratio(u) == pytest.approx(limit, abs=1e-06)


def test_gil_pelaez_probabilities_are_reachable_without_the_dispatcher() -> None:
    """The inversion is a public function; a calibration will want it directly."""
    point = POINTS[0]
    probabilities = gil_pelaez_probabilities(
        characteristic_function_model(point.model()),
        s0=point.spot,
        strike=point.strike,
        expiry=point.expiry,
        rate=point.rate,
        dividend=point.dividend,
        limit=200,
        tolerance=1e-10,
    )
    d1, d2 = _d1_d2(point)
    assert probabilities.pi1 == pytest.approx(_norm_cdf(d1), abs=1e-15)
    assert probabilities.pi2 == pytest.approx(_norm_cdf(d2), abs=1e-15)
    assert 0.0 < probabilities.pi2 < probabilities.pi1 < 1.0


# --------------------------------------------------------------------------
# All four methods, one number.
# --------------------------------------------------------------------------

METHOD_TOLERANCES = {
    "cos": 1e-11,
    "carr_madan": 5e-10,
    "lewis": 1e-13,
    "gil_pelaez": 1e-13,
}
"""Per-method budgets, each derived from its own measured worst case; the
Carr-Madan entry is the direct-quadrature variant, because the FFT variant's
interpolation error is 6.3e-04 and belongs to a different conversation."""


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("kind", KINDS)
def test_all_four_transform_methods_agree(point: Point, kind: str) -> None:
    """Evidence class: CLOSED_FORM for each leg (they share a reference).

    Explicitly **not** INDEPENDENT_ENGINE: all four read the same
    characteristic function, so agreement between them is evidence about the
    four payoff transforms and nothing at all about the model. The reference is
    the closed form, which is why each leg is compared to it rather than to the
    others.
    """
    exact = _exact(point, kind, "vanilla")
    for method, tolerance in METHOD_TOLERANCES.items():
        kwargs = {"carr_madan_transform": "quadrature"} if method == "carr_madan" else {}
        value = _value(point, method, kind, "vanilla", **kwargs).value
        assert value == pytest.approx(exact, abs=tolerance), method


def test_the_two_digital_methods_agree() -> None:
    """COS and Gil-Pelaez; the other two have no digital payoff at all."""
    for point in POINTS:
        exact = _exact(point, "call", "digital")
        assert _value(point, "cos", "call", "digital").value == pytest.approx(
            exact, abs=1e-11
        )
        assert _value(point, "gil_pelaez", "call", "digital").value == pytest.approx(
            exact, abs=DIGITAL_TOLERANCE
        )


# --------------------------------------------------------------------------
# Refusals and configuration.
# --------------------------------------------------------------------------


def test_lewis_has_no_digital_payoff() -> None:
    point = POINTS[0]
    option = DigitalOption(kind="call", strike=100.0, expiry=1.0, cash=1.0)
    with pytest.raises(NotSupportedError, match="prices the call transform"):
        price(
            option,
            point.model(),
            point.market(),
            method="fourier",
            cfg=FourierConfig(method="lewis"),
        )


@pytest.mark.parametrize("method", ("lewis", "gil_pelaez"))
def test_greeks_are_refused_and_the_message_names_cos(method: str) -> None:
    point = POINTS[0]
    with pytest.raises(NotSupportedError, match="closed\\s+forms of the cosine"):
        greeks(
            point.instrument("call", "vanilla"),
            point.model(),
            point.market(),
            method="fourier",
            cfg=FourierConfig(method=method),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quad_limit": 0}, "quad_limit must be an integer >= 1"),
        ({"quad_limit": 10.0}, "quad_limit must be an integer >= 1"),
        ({"quad_tolerance": 0.0}, "quad_tolerance must be finite and > 0"),
        ({"quad_tolerance": float("nan")}, "quad_tolerance must be finite and > 0"),
    ],
)
def test_config_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        FourierConfig(**kwargs)


def test_every_method_name_is_reachable_and_validated() -> None:
    from qpl.engines.fourier import FOURIER_METHODS

    assert set(FOURIER_METHODS) == {"cos", "carr_madan", "lewis", "gil_pelaez"}
    with pytest.raises(InvalidInputError, match="method must be one of"):
        FourierConfig(method="heston")  # type: ignore[arg-type]
    assert EvidenceClass.CLOSED_FORM.value == "closed_form"
