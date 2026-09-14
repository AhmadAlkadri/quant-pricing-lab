"""Calibrating Heston: the gradient, the constraints, and the pricer's envelope.

This file starts with the two claims that are about the *machinery* rather than
about identifiability -- the analytic Jacobian and the constraint handling --
because everything later depends on them being right.

The claims about identifiability -- synthetic recovery (a), the conditioning
and the flat direction (b), the objective choice (c) and initialisation (d) --
follow in the second half.

(f) **The analytic gradient.** `heston_charfn_gradient` returns `phi` together
    with its five parameter derivatives from the closed-form affine solution.
    `phi` reproduces `qpl.models.heston.heston_characteristic_function` to
    2.5e-16 absolute; the derivatives agree with central differences of that
    function to **1.3e-09 to 4.6e-08** relative over three parameter sets and
    three maturities, which is the difference quotient's own floor and not a
    property of the formula. Carried through the COS sum, the price gradient
    agrees with central differences **of the full pricer, truncation-range
    motion included**, to 6.8e-07 relative on the price objective. It is worth
    **9.4x** against a five-parameter central-difference Jacobian at the same
    point, and **3.4x** end to end on a real solve (41 ms against 142 ms).
    CLOSED_FORM for the identity, and a measured tolerance for the rest.

(e) **Constraints.** The bounded solver (`method="trf"`) stays strictly inside
    `DEFAULT_BOUNDS`, which keep `v0, kappa, theta, xi > 0` and `|rho| < 1` and
    impose nothing else. The unconstrained route (`method="lm"`) **leaves the
    domain**: from a start at `(0.005, 0.01, 0.005, 0.01, 0.0)` it terminates
    on its own `xtol` at `rho = +1.0140`, a correlation above one, and the
    result reports `in_domain = False` and `model = None` rather than coercing
    it into a `HestonModel` that cannot exist. The bounded run from the same
    start recovers the truth at an RMSE of 2.56e-07. A Feller-violating optimum
    is **reported** and not forbidden: the Feller-violating study set is
    recovered to 1e-06 with `feller_number = 0.0800` and
    `feller_satisfied = False`. EXACT_IDENTITY for the bound arithmetic,
    NEGATIVE_FINDING for what the unconstrained route does with the same data.

(g) **The COS envelope, and the payoff leg.** Slice 15 measured the COS
    call/put asymmetry twice and got opposite answers. Both are right: a call's
    payoff coefficient carries `e^b` and a put's carries `e^{z*} = K / S_0`, so
    the call loses to round-off when the range is wide and the put loses to the
    truncated left tail when it is narrow. The crossover is at a range endpoint
    of about 7.5 and it is sharp -- over eight cells the call is better at every
    `b` below 6.69 and the put at every `b` above 8.49, by up to **seven decimal
    orders**. Choosing by `b` also makes the residual function *total*, because
    the put leg's coefficients cannot overflow: over all fourteen starts of
    `qpl.cases.heston_calibration` and 200 uniform draws inside the bounds,
    every price and gradient is finite and inside the no-arbitrage strip, where
    a fixed call leg reports a 100-strike call at `-3.2e+43` in the same corner.
    That one rule removed a start-grid failure rate of 3/14 outright.

Sources: Gatheral (2006) chapter 3; Cui, del Bano Rollin and Germano (2017)
EJOR 263(2) section 3; Nocedal and Wright (2006) chapter 10. Every number in
this file is measured here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.calibration import (
    COS_PUT_LEG_THRESHOLD,
    DEFAULT_BOUNDS,
    FELLER_SATISFIED_COS,
    FELLER_VIOLATED_COS,
    HESTON_PARAMETERS,
    OptionQuote,
    calibrate_heston,
    cos_call_prices,
    default_cos_settings,
    heston_charfn_gradient,
    heston_quote_values,
    parameter_covariance,
    residual_jacobian,
    vega_weights,
)
from qpl.calibration.heston import VEGA_FLOOR, _Problem
from qpl.engines.analytic.black_scholes import implied_volatility
from qpl.engines.fourier.cos import cos_price
from qpl.engines.fourier.lewis import lewis_call
from qpl.exceptions import InvalidInputError
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import bs_price
from qpl.models.heston import (
    HestonModel,
    heston_characteristic_function,
    heston_log_return_cumulants,
)
from qpl.validation import EvidenceClass

# --------------------------------------------------------------------------
# The two study points, shared with qpl.cases.heston_calibration.
# --------------------------------------------------------------------------

MARKET = Market(
    spot=100.0,
    rate_curve=FlatRateCurve(0.01),
    dividend_curve=FlatDividendCurve(0.02),
)
"""The Alan Lewis reference market used everywhere else in this repository's
Heston tests, so a calibration number is comparable with a pricing one."""

REFERENCE = HestonModel(v0=0.04, kappa=4.0, theta=0.25, xi=1.0, rho=-0.5)
"""Feller number 4.0: the condition holds."""

FELLER_VIOLATED = HestonModel(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9)
"""Feller number 0.08: this repository's own violating set, shared with the
Slice 9 CIR study, the Slice 15 transform study and the Slice 16 QE study."""

STRIKES = (80.0, 90.0, 100.0, 110.0, 120.0)
SIX_MATURITIES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
THREE_MATURITIES = (0.25, 1.0, 2.0)
ONE_MATURITY = (1.0,)

TRUE_REFERENCE = tuple(getattr(REFERENCE, name) for name in HESTON_PARAMETERS)
TRUE_FELLER_VIOLATED = tuple(
    getattr(FELLER_VIOLATED, name) for name in HESTON_PARAMETERS
)


def synthetic_quotes(model, maturities, strikes=STRIKES, *, as_price=False):
    """Quotes generated from `model` by the module's own pricer.

    Deliberately the same pricer the calibration will use: that makes this a
    test of the *inverse problem* and not of the transform, which Slice 15
    already tested against six published values and against QuantLib. A
    synthetic-recovery test whose forward model differs from the calibrator's
    measures the difference between two pricers, which is a different claim.
    """
    quotes = [
        OptionQuote(strike=k, expiry=t, kind="call", value=1.0, value_type="price")
        for t in maturities
        for k in strikes
    ]
    prices = heston_quote_values(model, MARKET, quotes)
    out = []
    for quote, price in zip(quotes, prices, strict=True):
        if as_price:
            out.append(
                OptionQuote(
                    strike=quote.strike,
                    expiry=quote.expiry,
                    kind="call",
                    value=float(price),
                    value_type="price",
                )
            )
        else:
            option = EuropeanOption(
                kind="call", strike=quote.strike, expiry=quote.expiry
            )
            out.append(
                OptionQuote(
                    strike=quote.strike,
                    expiry=quote.expiry,
                    kind="call",
                    value=float(implied_volatility(float(price), option, MARKET)),
                    value_type="implied_vol",
                )
            )
    return out


# --------------------------------------------------------------------------
# (f) The analytic gradient.
# --------------------------------------------------------------------------

GRADIENT_POINTS = (
    TRUE_REFERENCE,
    TRUE_FELLER_VIOLATED,
    (0.09, 1.5, 0.06, 0.4, -0.7),
)
GRADIENT_MATURITIES = (0.25, 1.0, 3.0)

CHARFN_IDENTITY_TOLERANCE = 1e-15
"""Worst measured `|phi_here - phi_models|` over the grid below: 2.46e-16, i.e.
round-off. The two are the same formula written twice, so this is an identity
and the tolerance is a floating-point budget."""

CHARFN_GRADIENT_TOLERANCE = 1e-06
"""Worst measured relative difference against central differences: 4.56e-08, at
the Feller-violating set and `T = 3`. The budget is two decimal orders above
that because the *difference quotient* is the inaccurate side -- its own error
is `O(h^2) + O(eps/h)` at `h = 1e-05`, which is around 1e-08 relative here, and
the analytic formula has no `h` at all."""


@pytest.mark.parametrize("parameters", GRADIENT_POINTS, ids=["reference", "feller_violated", "third"])
@pytest.mark.parametrize("expiry", GRADIENT_MATURITIES, ids=lambda t: f"T{t}")
def test_the_gradient_transform_is_the_models_transform(parameters, expiry) -> None:
    """Evidence class: EXACT_IDENTITY.

    `heston_charfn_gradient` recomputes the characteristic function rather than
    calling `qpl.models.heston` -- every intermediate it needs is needed again
    by the derivatives, and evaluating the transform twice would double the
    cost of the thing this module does most often. That is only safe if the two
    are the same function, which is what this asserts.
    """
    assert EvidenceClass.EXACT_IDENTITY is EvidenceClass.EXACT_IDENTITY
    kwargs = dict(zip(HESTON_PARAMETERS, parameters, strict=True))
    u = np.linspace(0.0, 40.0, 97)
    phi, _ = heston_charfn_gradient(u, expiry, rate=0.01, dividend=0.02, **kwargs)
    reference = heston_characteristic_function(
        u, expiry, rate=0.01, dividend=0.02, **kwargs
    )
    assert np.max(np.abs(phi - reference)) < CHARFN_IDENTITY_TOLERANCE


@pytest.mark.parametrize("parameters", GRADIENT_POINTS, ids=["reference", "feller_violated", "third"])
@pytest.mark.parametrize("expiry", GRADIENT_MATURITIES, ids=lambda t: f"T{t}")
def test_the_charfn_gradient_matches_central_differences(parameters, expiry) -> None:
    """Evidence class: CLOSED_FORM, checked against an independent difference.

    The five derivatives are hand-derived from the affine solution used in this
    repository (Cui et al. (2017) section 3 for the strategy; their
    rearrangement has no `g` in it, so their formulas are not these). The check
    is a central difference of the *transform itself*, which shares no code
    with the gradient beyond the transform.
    """
    assert EvidenceClass.CLOSED_FORM is EvidenceClass.CLOSED_FORM
    kwargs = dict(zip(HESTON_PARAMETERS, parameters, strict=True))
    u = np.linspace(0.0, 40.0, 97)
    _, gradient = heston_charfn_gradient(u, expiry, rate=0.01, dividend=0.02, **kwargs)
    for index, name in enumerate(HESTON_PARAMETERS):
        step = 1e-05 * max(1.0, abs(kwargs[name]))
        up = dict(kwargs)
        up[name] += step
        down = dict(kwargs)
        down[name] -= step
        difference = (
            heston_characteristic_function(u, expiry, rate=0.01, dividend=0.02, **up)
            - heston_characteristic_function(
                u, expiry, rate=0.01, dividend=0.02, **down
            )
        ) / (2.0 * step)
        scale = max(float(np.max(np.abs(difference))), 1e-300)
        worst = float(np.max(np.abs(gradient[index] - difference))) / scale
        assert worst < CHARFN_GRADIENT_TOLERANCE, (name, worst)


PRICE_JACOBIAN_TOLERANCE = 5e-06
"""Worst measured relative difference between the analytic price Jacobian and a
central difference **of the full residual function**, range motion included:
6.79e-07 on the price objective and 1.15e-05 on the implied-volatility one.

The second number is larger for a reason worth writing down: the
implied-volatility residual is the output of a Brent inversion with an `xtol`
of 1e-07, so differencing it at `h = 1e-05` divides that tolerance by `2e-05`.
The analytic Jacobian goes through the chain rule instead -- one division by a
closed-form vega, no second inversion -- so it is the *difference* that carries
the error here, not the formula. The price objective, which has no inversion in
it at all, is the clean comparison and is the one this tolerance is set from;
the implied-volatility leg gets its own looser budget below."""

IMPLIED_VOL_JACOBIAN_TOLERANCE = 1e-04


@pytest.mark.parametrize(
    ("objective", "tolerance"),
    [("price", PRICE_JACOBIAN_TOLERANCE), ("implied_vol", IMPLIED_VOL_JACOBIAN_TOLERANCE)],
)
def test_the_price_jacobian_matches_central_differences(objective, tolerance) -> None:
    """Evidence class: CLOSED_FORM.

    The analytic price gradient holds the COS truncation range fixed while the
    difference quotient lets it move with the parameters. The agreement below
    is what says the omitted range-motion term is the derivative of a 5e-12
    truncation error rather than something that matters.
    """
    assert EvidenceClass.CLOSED_FORM is EvidenceClass.CLOSED_FORM
    quotes = synthetic_quotes(REFERENCE, SIX_MATURITIES)
    problem = _Problem(
        quotes,
        MARKET,
        objective=objective,
        weights=None,
        settings=FELLER_SATISFIED_COS,
    )
    x = np.array(TRUE_REFERENCE)
    analytic = problem.jacobian(x)
    difference = np.empty_like(analytic)
    for index in range(5):
        step = 1e-05 * max(1.0, abs(x[index]))
        up = x.copy()
        up[index] += step
        down = x.copy()
        down[index] -= step
        difference[:, index] = (problem.residuals(up) - problem.residuals(down)) / (
            2.0 * step
        )
    scale = np.maximum(np.abs(difference).max(axis=0), 1e-300)
    worst = float(np.max(np.abs(analytic - difference) / scale[None, :]))
    assert worst < tolerance, worst


def test_the_analytic_jacobian_and_the_differenced_one_reach_the_same_optimum() -> None:
    """The speed-up is not bought with accuracy.

    Measured on this grid: analytic 41.3 ms, `'2-point'` 141.5 ms, `'3-point'`
    179.2 ms for the price objective -- 3.4x and 4.3x -- and 80.0 / 275.4 /
    391.5 ms for the implied-volatility one. Timings are not asserted (a test
    that pins a wall clock fails on someone else's machine); what is asserted
    is that the three land on the same parameters, which is what makes the
    timing comparison meaningful in the first place.
    """
    quotes = synthetic_quotes(REFERENCE, SIX_MATURITIES)
    start = (0.08, 2.0, 0.15, 0.6, -0.2)
    fits = {
        mode: calibrate_heston(quotes, MARKET, initial=start, objective="price", jac=mode)
        for mode in ("analytic", "2-point", "3-point")
    }
    reference = np.array(fits["analytic"].parameters)
    for mode, fit in fits.items():
        assert fit.success, mode
        assert np.max(np.abs(np.array(fit.parameters) - reference)) < 1e-05, mode
        assert np.max(np.abs(reference - np.array(TRUE_REFERENCE))) < 1e-05


def test_a_put_quote_and_a_call_quote_share_one_gradient() -> None:
    """Parity's two legs carry no parameters, so the derivative is the same.

    This is the one place the put route is visible in the Jacobian, and it is
    an identity rather than an approximation: `P = C - S e^{-qT} + K e^{-rT}`
    and neither correction depends on `(v0, kappa, theta, xi, rho)`.
    """
    call = OptionQuote(strike=100.0, expiry=1.0, kind="call", value=0.3)
    put = OptionQuote(strike=100.0, expiry=1.0, kind="put", value=0.3)
    _, jac_call = residual_jacobian(
        [call], MARKET, TRUE_REFERENCE, objective="price"
    )
    _, jac_put = residual_jacobian([put], MARKET, TRUE_REFERENCE, objective="price")
    assert np.allclose(jac_call, jac_put, rtol=0.0, atol=0.0)


# --------------------------------------------------------------------------
# (g) What the COS pricer is worth, and where it stops being worth anything.
# --------------------------------------------------------------------------

LEG_CROSSOVER = (
    # (parameters, expiry, b, call-leg error, put-leg error)
    (TRUE_REFERENCE, 0.25, 1.730, 2.13e-13, 3.61e-07),
    (TRUE_FELLER_VIOLATED, 0.25, 2.946, 2.08e-12, 1.67e-10),
    (TRUE_REFERENCE, 1.0, 4.558, 1.33e-12, 2.70e-08),
    (TRUE_FELLER_VIOLATED, 1.0, 6.694, 1.48e-10, 4.75e-08),
    (TRUE_REFERENCE, 3.0, 8.489, 1.30e-10, 1.55e-11),
    (TRUE_FELLER_VIOLATED, 3.0, 14.128, 2.90e-05, 1.85e-07),
    (TRUE_REFERENCE, 10.0, 15.357, 1.81e-08, 4.97e-14),
    (TRUE_FELLER_VIOLATED, 10.0, 31.106, 6.45e03, 7.57e-07),
)
"""Worst error over strikes 80-120 against a Lewis integral of the same
transform, by payoff leg, at this module's settings. The `b` column is the
range's upper end and is what decides which leg wins."""


@pytest.mark.parametrize(
    ("parameters", "expiry", "endpoint", "call_error", "put_error"),
    LEG_CROSSOVER,
    ids=[f"b{row[2]:g}" for row in LEG_CROSSOVER],
)
def test_the_payoff_leg_crosses_over_with_the_range_endpoint(
    parameters, expiry, endpoint, call_error, put_error
) -> None:
    """NEGATIVE_FINDING: neither leg is uniformly better, and `b` decides.

    Slice 15 measured this asymmetry twice and got opposite answers -- the call
    four decimal orders better at the package defaults, the call diverging
    hopelessly on a Feller-violating ten-year cell -- and both measurements are
    right. A call's payoff coefficient integrates `e^z` up to `b`, so the
    cosine sum has to cancel `e^b` against a price of order `S_0`; a put's
    integrates up from `a` and carries only `e^{z*} = K / S_0`, but pays for it
    by missing whatever tail mass sits below `a`, which for `rho < 0` is the
    fat one.

    The written plan for this slice named the put as the reliable leg. That is
    right above the crossover and wrong below it by up to six decimal orders,
    so the module chooses per maturity instead of once.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    model = HestonModel(*parameters)
    settings = default_cos_settings(parameters)
    v0, kappa, theta, xi, rho = parameters
    c1, c2 = heston_log_return_cumulants(
        expiry, rate=0.01, dividend=0.02, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho
    )
    assert c1 + settings.truncation_l * math.sqrt(c2) == pytest.approx(
        endpoint, rel=0.02
    )

    worst_call = 0.0
    worst_put = 0.0
    for strike in STRIKES:
        reference, _ = lewis_call(
            model,
            s0=MARKET.spot,
            strike=strike,
            expiry=expiry,
            rate=0.01,
            dividend=0.02,
            limit=800,
            tolerance=1e-13,
        )
        direct_call = cos_price(
            model,
            s0=MARKET.spot,
            strike=strike,
            expiry=expiry,
            rate=0.01,
            dividend=0.02,
            kind="call",
            n_terms=settings.n_terms,
            truncation_l=settings.truncation_l,
        ).value
        direct_put = cos_price(
            model,
            s0=MARKET.spot,
            strike=strike,
            expiry=expiry,
            rate=0.01,
            dividend=0.02,
            kind="put",
            n_terms=settings.n_terms,
            truncation_l=settings.truncation_l,
        ).value
        spot_leg = MARKET.spot * math.exp(-0.02 * expiry)
        strike_leg = strike * math.exp(-0.01 * expiry)
        worst_call = max(worst_call, abs(direct_call - reference))
        worst_put = max(
            worst_put, abs(direct_put + spot_leg - strike_leg - reference)
        )

    # The published table reproduces, to within a decimal order.
    assert worst_call == pytest.approx(call_error, rel=10.0)
    assert worst_put == pytest.approx(put_error, rel=10.0)
    # ... and the threshold picks the better of the two at every endpoint.
    if endpoint < COS_PUT_LEG_THRESHOLD:
        assert worst_call < worst_put
    else:
        assert worst_put < worst_call


@pytest.mark.parametrize(
    ("parameters", "expiry"),
    [(row[0], row[1]) for row in LEG_CROSSOVER],
    ids=[f"b{row[2]:g}" for row in LEG_CROSSOVER],
)
def test_the_module_delivers_the_better_leg_at_every_cell(parameters, expiry) -> None:
    """The rule, end to end: `cos_call_prices` is at the better of the two."""
    model = HestonModel(*parameters)
    settings = default_cos_settings(parameters)
    values, _ = cos_call_prices(
        parameters,
        s0=MARKET.spot,
        strikes=np.array(STRIKES),
        expiry=expiry,
        rate=0.01,
        dividend=0.02,
        settings=settings,
    )
    worst = 0.0
    for index, strike in enumerate(STRIKES):
        reference, _ = lewis_call(
            model,
            s0=MARKET.spot,
            strike=strike,
            expiry=expiry,
            rate=0.01,
            dividend=0.02,
            limit=800,
            tolerance=1e-13,
        )
        worst = max(worst, abs(float(values[index]) - reference))
    row = next(r for r in LEG_CROSSOVER if r[0] == parameters and r[1] == expiry)
    assert worst < 10.0 * min(row[3], row[4])


def test_the_put_leg_keeps_the_whole_parameter_box_finite() -> None:
    """NEGATIVE_FINDING repaired: no clip, no overflow, no dead starts.

    The call leg's `e^b` is not merely inaccurate in the far corner of
    `DEFAULT_BOUNDS` -- it overflows. At `(0.5, 0.1, 0.8, 3.0, -0.99)` and
    `T = 3`, where the Feller branch asks for `L = 28` and the range's upper end
    is 113.8, a direct COS call reports a 100-strike call as **-3.2e+43**. A
    price like that clamps to the edge of the no-arbitrage strip, its implied
    volatility clamps with it, the residual goes flat and a local solver stalls
    on its first step; that was measured as three dead starts out of fourteen.

    The put leg has no `e^b` in it, so the same cell comes back as 11.61
    against a true 11.98 -- wrong, because the range is far too wide for the
    term count there, but a *price*. This is asserted over every start in the
    published grid at every study maturity, and it is why this module needs no
    range clip and no penalty on the pricer.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    corner = (0.5, 0.1, 0.8, 3.0, -0.99)
    settings = default_cos_settings(corner)
    overflowing = cos_price(
        HestonModel(*corner),
        s0=MARKET.spot,
        strike=100.0,
        expiry=3.0,
        rate=0.01,
        dividend=0.02,
        kind="call",
        n_terms=settings.n_terms,
        truncation_l=settings.truncation_l,
    ).value
    assert abs(overflowing) > 1e30

    for parameters in START_GRID:
        leg_settings = default_cos_settings(parameters)
        for expiry in SIX_MATURITIES:
            values, gradients = cos_call_prices(
                parameters,
                s0=MARKET.spot,
                strikes=np.array(STRIKES),
                expiry=expiry,
                rate=0.01,
                dividend=0.02,
                settings=leg_settings,
                gradient=True,
            )
            assert np.all(np.isfinite(values)), (parameters, expiry)
            assert np.all(np.isfinite(gradients)), (parameters, expiry)
            ceiling = MARKET.spot * math.exp(-0.02 * expiry)
            assert np.all(values >= -1e-06)
            assert np.all(values <= ceiling + 1e-06)


def test_the_feller_branch_picks_the_settings_and_neither_setting_does_both() -> None:
    """NEGATIVE_FINDING: no one COS range setting serves both parameter sets.

    At the money at `T = 1`, against a Lewis integral of the same transform,
    through `cos_call_prices` (so with the leg rule in force):

        settings          reference set (Feller 4)   violating set (Feller 0.08)
        L=10,  N=256              1.2e-12                    8.0e-04
        L=28, N=4096              4.3e-14                    1.2e-10

    The narrow setting truncates the Feller-violating left tail by eight
    decimal orders, so a Feller-violating fit **must** widen the range. The
    wide setting is not worse on the reference set at all once the leg rule is
    in force (before it, the same cell read 1.4e-08 and a `T = 2` cell read
    1.3e-05), so what is left is a **cost** argument and not an accuracy one:
    `L = 28, N = 4096` is 6.5x the run time of `L = 10, N = 256` for the same
    30 quotes. `default_cos_settings` branches on the Feller number for that
    reason, and the branch is stated here as what it is.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    assert default_cos_settings(TRUE_REFERENCE) == FELLER_SATISFIED_COS
    assert default_cos_settings(TRUE_FELLER_VIOLATED) == FELLER_VIOLATED_COS

    errors = {}
    for label, model, parameters in (
        ("reference", REFERENCE, TRUE_REFERENCE),
        ("violating", FELLER_VIOLATED, TRUE_FELLER_VIOLATED),
    ):
        reference_value, _ = lewis_call(
            model,
            s0=MARKET.spot,
            strike=100.0,
            expiry=1.0,
            rate=0.01,
            dividend=0.02,
            limit=800,
            tolerance=1e-13,
        )
        for name, settings in (
            ("default", FELLER_SATISFIED_COS),
            ("wide", FELLER_VIOLATED_COS),
        ):
            value, _ = cos_call_prices(
                parameters,
                s0=MARKET.spot,
                strikes=np.array([100.0]),
                expiry=1.0,
                rate=0.01,
                dividend=0.02,
                settings=settings,
            )
            errors[(label, name)] = abs(float(value[0]) - reference_value)

    # The narrow setting is fine on the reference set and eight decimal orders
    # out on the violating one: that is the accuracy half of the branch.
    assert errors[("reference", "default")] < 1e-11
    assert errors[("violating", "default")] > 1e-05
    assert errors[("violating", "wide")] < 1e-09
    assert errors[("violating", "default")] / errors[("violating", "wide")] > 1e05
    # The wide setting is NOT worse on the reference set once the leg rule is
    # in force, which is why the branch is justified by cost and says so.
    assert errors[("reference", "wide")] < 1e-11


# --------------------------------------------------------------------------
# (e) Constraints: the bounds, the domain, and the Feller flag.
# --------------------------------------------------------------------------


def test_the_bounded_solver_stays_inside_the_bounds_and_the_domain() -> None:
    """Evidence class: EXACT_IDENTITY (an arithmetic property of the answer)."""
    assert EvidenceClass.EXACT_IDENTITY is EvidenceClass.EXACT_IDENTITY
    quotes = synthetic_quotes(REFERENCE, SIX_MATURITIES)
    lower, upper = DEFAULT_BOUNDS
    for start in (
        (0.08, 2.0, 0.15, 0.6, -0.2),
        (0.04, 0.05, 0.25, 3.0, -0.95),
        (0.5, 15.0, 0.8, 4.0, 0.9),
    ):
        fit = calibrate_heston(quotes, MARKET, initial=start, objective="price")
        for index, name in enumerate(HESTON_PARAMETERS):
            assert lower[index] <= fit.parameters[index] <= upper[index], name
        assert fit.in_domain
        assert fit.model is not None


UNCONSTRAINED_RMSE_PENALTY = 1e05
"""From the start `(0.005, 0.01, 0.005, 0.01, 0.0)`, the bounded run recovers
the truth at a root-mean-square residual of 2.56e-07, while the unconstrained
one terminates at **`rho = +1.0140`** -- a correlation above one, outside the
Heston domain entirely -- with a residual of 6.83. From
`(0.04, 0.05, 0.25, 3.0, -0.95)` under the implied-volatility objective it
lands on `rho = -1.0000` exactly, on the boundary. The budget below is many
orders inside those ratios, because the point is that the answer is
inadmissible and not how large its residual is."""


def test_the_unconstrained_route_leaves_the_domain() -> None:
    """NEGATIVE_FINDING: `method="lm"` takes no bounds and it shows.

    `scipy.optimize.least_squares(method='lm')` has no bound support at all, so
    `calibrate_heston` refuses `bounds=` there rather than ignoring them, and
    the only thing between the search and an inadmissible parameter vector is
    `DOMAIN_PENALTY` -- which keeps the residual *evaluable* outside the
    domain, not the search inside it. From a start already pressed against the
    domain, not the search inside it. From a start in the low corner of the box
    the unconstrained run converges (by its own `xtol`) to a **correlation
    above one**, so `CalibrationResult.model` is `None` and `in_domain` is
    `False`: the answer is reported as what it is rather than coerced into a
    `HestonModel` that cannot exist. The bounded run from the same start
    recovers the truth.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    quotes = synthetic_quotes(REFERENCE, SIX_MATURITIES)
    start = (0.005, 0.01, 0.005, 0.01, 0.0)
    bounded = calibrate_heston(quotes, MARKET, initial=start, objective="price")
    free = calibrate_heston(
        quotes, MARKET, initial=start, objective="price", method="lm"
    )
    assert not free.in_domain
    assert free.model is None
    assert abs(free.parameter("rho")) > 1.0
    assert free.rmse > UNCONSTRAINED_RMSE_PENALTY * bounded.rmse
    assert bounded.rmse < 1e-05


def test_lm_refuses_bounds_rather_than_ignoring_them() -> None:
    quotes = synthetic_quotes(REFERENCE, THREE_MATURITIES)
    with pytest.raises(InvalidInputError, match="unconstrained route"):
        calibrate_heston(
            quotes,
            MARKET,
            initial=TRUE_REFERENCE,
            method="lm",
            bounds=DEFAULT_BOUNDS,
        )


def test_a_feller_violating_optimum_is_reported_and_not_forbidden() -> None:
    """Evidence class: NEGATIVE_FINDING for the flag, CLOSED_FORM for the fit.

    `DEFAULT_BOUNDS` keeps `kappa, theta, xi > 0` and `|rho| < 1` and imposes
    nothing else. `2 kappa theta >= xi^2` is a statement about whether the
    variance can reach zero, not about whether the parameters are admissible,
    and equity smiles routinely calibrate to the violating side. The fit is
    reported with `feller_satisfied = False` and a `feller_number` of 0.0800,
    and the model object is still built.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    quotes = synthetic_quotes(FELLER_VIOLATED, SIX_MATURITIES)
    fit = calibrate_heston(
        quotes, MARKET, initial=(0.08, 0.9, 0.09, 0.6, -0.6), objective="implied_vol"
    )
    assert fit.success
    assert not fit.feller_satisfied
    assert fit.feller_number == pytest.approx(0.08, abs=1e-03)
    assert fit.model is not None
    assert not fit.model.feller_satisfied
    assert np.max(np.abs(np.array(fit.parameters) - np.array(TRUE_FELLER_VIOLATED))) < 1e-06


def test_the_residual_is_total_outside_the_domain() -> None:
    """`DOMAIN_PENALTY` keeps a wandering unconstrained step evaluable.

    Outside the domain the transform is not a characteristic function of
    anything, so there is no price to report; the residual is instead the price
    at the clamped parameters plus a penalty proportional to the excursion,
    which is continuous, zero inside and gradient-positive outside. Without it
    an unconstrained solve aborts on a NaN instead of stepping back.
    """
    quotes = synthetic_quotes(REFERENCE, ONE_MATURITY)
    problem = _Problem(
        quotes,
        MARKET,
        objective="price",
        weights=None,
        settings=FELLER_SATISFIED_COS,
    )
    inside = problem.residuals(np.array(TRUE_REFERENCE))
    outside = problem.residuals(np.array([0.04, 4.0, 0.25, 1.0, -1.2]))
    assert np.all(np.isfinite(inside))
    assert np.all(np.isfinite(outside))
    assert np.min(np.abs(outside)) > 100.0 * np.max(np.abs(inside))


def test_the_vega_floor_and_the_implied_vol_clamp_do_not_bind_at_a_fit() -> None:
    """Both guards exist for the search path, and neither touches the answer."""
    quotes = synthetic_quotes(REFERENCE, SIX_MATURITIES)
    fit = calibrate_heston(
        quotes, MARKET, initial=(0.08, 2.0, 0.15, 0.6, -0.2), objective="implied_vol"
    )
    problem = _Problem(
        quotes,
        MARKET,
        objective="implied_vol",
        weights=None,
        settings=fit.cos_settings,
    )
    prices, _ = problem.prices(np.array(fit.parameters))
    vols = problem.implied_vols(prices)
    assert np.all(vols > 1e-02)
    assert np.min(np.abs(fit.jacobian)) > 0.0
    weights = vega_weights(quotes, MARKET)
    assert np.all(weights < 1.0 / VEGA_FLOOR)


# --------------------------------------------------------------------------
# Validation of the public surface.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"strike": 0.0}, "strike"),
        ({"expiry": 0.0}, "expiry"),
        ({"kind": "straddle"}, "kind"),
        ({"value_type": "variance"}, "value_type"),
        ({"value": -1.0}, "value"),
        ({"weight": 0.0}, "weight"),
    ],
)
def test_option_quote_validates_at_construction(kwargs, match) -> None:
    base = {
        "strike": 100.0,
        "expiry": 1.0,
        "kind": "call",
        "value": 0.2,
        "value_type": "implied_vol",
        "weight": 1.0,
    }
    base.update(kwargs)
    with pytest.raises(InvalidInputError, match=match):
        OptionQuote(**base)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"objective": "variance"}, "objective"),
        ({"method": "newton"}, "method"),
        ({"jac": "5-point"}, "jac"),
        ({"initial": (0.04, 4.0, 0.25)}, "initial must be 5"),
        ({"initial": (0.04, 4.0, 0.25, 1.0, -50.0)}, "inside bounds"),
        ({"n_starts": 0}, "n_starts"),
        ({"weights": [1.0, 2.0]}, "weights must have shape"),
    ],
)
def test_calibrate_heston_validates_its_arguments(kwargs, match) -> None:
    quotes = synthetic_quotes(REFERENCE, ONE_MATURITY)
    call = {"initial": TRUE_REFERENCE}
    call.update(kwargs)
    with pytest.raises(InvalidInputError, match=match):
        calibrate_heston(quotes, MARKET, **call)


def test_an_empty_quote_list_is_refused() -> None:
    with pytest.raises(InvalidInputError, match="non-empty"):
        calibrate_heston([], MARKET, initial=TRUE_REFERENCE)


def test_the_covariance_reports_infinities_rather_than_a_pseudo_inverse() -> None:
    """A rank-deficient Jacobian has no error bars, and says so.

    Reporting a pseudo-inverse there would hand back a finite standard error
    for a direction the data does not determine, which is the one thing this
    slice exists to avoid.
    """
    jacobian = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    residuals = np.array([1e-03, -1e-03, 2e-03])
    covariance, condition, singular_values, standard_errors = parameter_covariance(
        jacobian, residuals
    )
    assert not math.isfinite(condition)
    assert np.all(np.isinf(covariance))
    assert np.all(np.isinf(standard_errors))
    assert singular_values[-1] == pytest.approx(0.0, abs=1e-14)


# --------------------------------------------------------------------------
# (a) Synthetic recovery, clean and noisy.
# --------------------------------------------------------------------------

CLEAN_START_REFERENCE = (0.08, 2.0, 0.15, 0.6, -0.2)
"""Perturbed initial guess for the reference set: `v0` doubled, `kappa` halved,
`theta` cut by 40%, `xi` cut by 40% and `rho` moved to -0.2. Deliberately not a
small perturbation -- a recovery from a 5% bump measures the solver's last
iteration and nothing else."""

CLEAN_START_FELLER_VIOLATED = (0.08, 0.9, 0.09, 0.6, -0.6)

RESIDUAL_TO_PARAMETER_SAFETY = 10.0
"""How much slack the clean-recovery bound below carries over the bound the
Jacobian's smallest singular value supplies.

The bound itself is not wishful: a residual perturbation `dr` moves the
Gauss-Newton solution by at most `||dr|| / s_min`, so the parameter error at a
converged fit is at most `||r|| / s_min` -- the residual the solver actually
stopped at, divided by the smallest singular value of the Jacobian there. On
the reference set with the implied-volatility objective that is
`5.45e-13 / 1.00e-02 = 5.43e-11`, against a worst measured parameter error of
7.0e-12, so the bound is tight to a factor of 8. The factor of 10 below is
for the two places the linearisation is not exact (the model is nonlinear in
the parameters, and the solver's stopping point is not the exact minimiser),
not for headroom against a number that was guessed."""


@pytest.mark.parametrize(
    ("label", "model", "truth", "start"),
    [
        ("reference", REFERENCE, TRUE_REFERENCE, CLEAN_START_REFERENCE),
        (
            "feller_violated",
            FELLER_VIOLATED,
            TRUE_FELLER_VIOLATED,
            CLEAN_START_FELLER_VIOLATED,
        ),
    ],
)
@pytest.mark.parametrize("objective", ["price", "implied_vol"])
def test_clean_synthetic_recovery_is_bounded_by_the_smallest_singular_value(
    label, model, truth, start, objective
) -> None:
    """Evidence class: CLOSED_FORM, with a tolerance derived from the Jacobian.

    Quotes generated from a known model on a 5-strike by 6-maturity grid are
    recovered from the perturbed start above. The per-parameter errors,
    measured:

        set               objective      v0        kappa      theta      xi        rho
        reference         price         2.9e-09   5.4e-07   3.3e-09   4.1e-07   1.3e-07
        reference         implied_vol   5.4e-14   7.0e-12   1.5e-13   1.2e-12   2.4e-13
        Feller-violating  price         4.2e-10   7.6e-08   7.7e-09   2.9e-08   2.3e-09
        Feller-violating  implied_vol   7.6e-13   6.2e-12   5.7e-13   3.5e-11   1.2e-11

    The bound asserted is `||r|| / s_min` times `RESIDUAL_TO_PARAMETER_SAFETY`,
    not a round number: the recovery is only ever as good as the residual the
    solver stopped at divided by how flat the objective is in its flattest
    direction, and both of those come back in the result.

    Note what this does **not** say. Every error above is 1e-07 or smaller, and
    none of it is evidence that these five parameters are identifiable from
    this data -- the quotes are exact to 1e-12 and the tolerances are the
    solver's. What identifiability costs is measured with noise, below, and
    what it costs at one maturity is measured after that.
    """
    assert EvidenceClass.CLOSED_FORM is EvidenceClass.CLOSED_FORM
    quotes = synthetic_quotes(model, SIX_MATURITIES)
    fit = calibrate_heston(quotes, MARKET, initial=start, objective=objective)
    assert fit.success
    assert fit.model is not None

    smallest = float(fit.singular_values[-1])
    budget = (
        RESIDUAL_TO_PARAMETER_SAFETY
        * float(np.linalg.norm(fit.residuals))
        / smallest
    )
    errors = np.abs(np.array(fit.parameters) - np.array(truth))
    assert float(np.max(errors)) < budget, (label, objective, errors, budget)
    # ... and the bound is worth having: it is not vacuous.
    assert budget < 1e-03


NOISE_SEEDS = 20
NOISE_LEVELS_BP = (5.0, 20.0)
STDERR_AGREEMENT_BAND = (0.8, 1.5)
"""The factor by which the empirical root-mean-square parameter error may
differ from the mean Jacobian-based standard error, over `NOISE_SEEDS` noise
draws. Measured ratios:

    noise      v0     kappa   theta    xi     rho
     5 bp     1.07     1.08    0.99    1.14   1.16
    20 bp     1.09     1.10    1.00    1.15   1.23

so the linearised error bars are right to about 20%, and are consistently on
the *small* side -- which is the expected direction, because the Gauss-Newton
covariance ignores the second-order term and the model is not linear in
`kappa`. The band is set from that measurement and not from a hoped-for 1."""

NOISE_SCALING_BAND = (3.5, 4.8)
"""Quadrupling the quote noise should quadruple the parameter error if the
problem is locally linear. Measured ratios of the 20 bp RMSE to the 5 bp one:
4.05 (v0), 4.08 (kappa), 4.03 (theta), 4.03 (xi), 4.58 (rho). `rho` is the one
that drifts, which is the nonlinearity showing up first in the parameter that
enters the transform through `beta = kappa - rho xi i u`."""


def _noisy_quotes(model, maturities, *, noise_bp: float, seed: int):
    """`model`'s own smile with independent Gaussian implied-volatility noise.

    Noise is added in **volatility** units because that is how a bid/ask is
    quoted; the price objective then sees the same perturbed surface converted
    through Black-Scholes, so the two objectives are compared on one data set
    rather than on two.
    """
    rng = np.random.default_rng(seed)
    clean = synthetic_quotes(model, maturities)
    return [
        OptionQuote(
            strike=q.strike,
            expiry=q.expiry,
            kind="call",
            value=float(q.value + rng.normal(0.0, noise_bp * 1e-04)),
            value_type="implied_vol",
        )
        for q in clean
    ]


@pytest.mark.parametrize("noise_bp", NOISE_LEVELS_BP, ids=["5bp", "20bp"])
def test_the_recovery_error_matches_the_jacobian_standard_errors(noise_bp) -> None:
    """Evidence class: STATISTICAL -- the honest identifiability statement.

    Twenty independent draws of Gaussian implied-volatility noise on the same
    5-by-6 grid, each calibrated from the same perturbed start. What is
    compared is the **spread of the fitted parameters across draws** against
    the **standard errors the Jacobian predicts within a single draw**. Those
    are two different computations of the same quantity, and their agreement
    (0.99 to 1.23, table in `STDERR_AGREEMENT_BAND`) is what licenses quoting
    a `CalibrationResult.standard_errors` as an error bar at all.

    At 20 bp -- two-tenths of a volatility point, a narrow real spread -- the
    fitted `kappa` carries a standard error of **0.19 on a true value of 4**
    and `xi` **0.086 on 1.0**, while `theta` is at 1.6e-03 on 0.25 and `rho` at
    0.033 on -0.5. That is the slice's result in one line: a fit that
    reproduces the surface to 20 bp pins `theta` to 0.7% and `kappa` to 5%.
    """
    assert EvidenceClass.STATISTICAL is EvidenceClass.STATISTICAL
    errors = []
    standard_errors = []
    for seed in range(NOISE_SEEDS):
        quotes = _noisy_quotes(
            REFERENCE, SIX_MATURITIES, noise_bp=noise_bp, seed=1000 + seed
        )
        fit = calibrate_heston(
            quotes,
            MARKET,
            initial=CLEAN_START_REFERENCE,
            objective="implied_vol",
        )
        assert fit.success
        errors.append(np.array(fit.parameters) - np.array(TRUE_REFERENCE))
        standard_errors.append(fit.standard_errors)

    empirical = np.sqrt((np.array(errors) ** 2).mean(axis=0))
    predicted = np.array(standard_errors).mean(axis=0)
    low, high = STDERR_AGREEMENT_BAND
    for index, name in enumerate(HESTON_PARAMETERS):
        ratio = empirical[index] / predicted[index]
        assert low < ratio < high, (name, ratio)


def test_the_recovery_error_scales_with_the_noise() -> None:
    """Evidence class: STATISTICAL.

    Four times the quote noise, four times the parameter error. Run as its own
    test rather than folded into the one above so that a failure says which of
    the two claims broke.
    """
    assert EvidenceClass.STATISTICAL is EvidenceClass.STATISTICAL
    rms = {}
    for noise_bp in NOISE_LEVELS_BP:
        errors = []
        for seed in range(NOISE_SEEDS):
            quotes = _noisy_quotes(
                REFERENCE, THREE_MATURITIES, noise_bp=noise_bp, seed=2000 + seed
            )
            fit = calibrate_heston(
                quotes,
                MARKET,
                initial=CLEAN_START_REFERENCE,
                objective="implied_vol",
            )
            errors.append(np.array(fit.parameters) - np.array(TRUE_REFERENCE))
        rms[noise_bp] = np.sqrt((np.array(errors) ** 2).mean(axis=0))
    low, high = NOISE_SCALING_BAND
    ratios = rms[NOISE_LEVELS_BP[1]] / rms[NOISE_LEVELS_BP[0]]
    for index, name in enumerate(HESTON_PARAMETERS):
        assert low < ratios[index] < high, (name, ratios[index])


# --------------------------------------------------------------------------
# (b) Identifiability: the flat direction and what maturities buy.
# --------------------------------------------------------------------------

CONDITION_NUMBERS_IMPLIED_VOL = {1: 6.7137e07, 3: 5.6614e02, 6: 4.7754e02}
CONDITION_NUMBERS_PRICE = {1: 6.4804e07, 3: 7.8250e02, 6: 9.5237e02}
"""Condition number of the residual Jacobian at the **true** parameters, on a
5-strike grid, as maturities are added. Measured, not predicted.

The implied-volatility column falls monotonically and the price column does
**not** -- it improves by five decimal orders from one maturity to three and
then gets *worse* by 22% going to six. That contradicts the written plan for
this slice, which asserted the ordering for the objective without saying
which. The reason is in the next test: a price objective weights a cell by its
vega, so adding long-dated at-the-money cells inflates the largest singular
value faster than the smallest. Gatheral (2006) chapter 3 argues for the
implied-volatility objective on exactly this ground; here it is the difference
between a condition number that improves with data and one that does not."""

CONDITION_TOLERANCE = 0.05
"""Relative budget on the pinned condition numbers above. They are pure linear
algebra on a deterministic Jacobian, so the only drift is BLAS/round-off; 5%
is generous and exists so that a different LAPACK does not fail the suite."""


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("implied_vol", CONDITION_NUMBERS_IMPLIED_VOL),
        ("price", CONDITION_NUMBERS_PRICE),
    ],
)
def test_the_condition_number_at_the_true_parameters(objective, expected) -> None:
    """Evidence class: CLOSED_FORM (deterministic linear algebra)."""
    assert EvidenceClass.CLOSED_FORM is EvidenceClass.CLOSED_FORM
    for maturities in (ONE_MATURITY, THREE_MATURITIES, SIX_MATURITIES):
        quotes = synthetic_quotes(REFERENCE, maturities)
        residuals, jacobian = residual_jacobian(
            quotes, MARKET, TRUE_REFERENCE, objective=objective
        )
        _, condition, _, _ = parameter_covariance(jacobian, residuals)
        target = expected[len(maturities)]
        assert condition == pytest.approx(target, rel=CONDITION_TOLERANCE)


def test_more_maturities_condition_the_implied_vol_problem_but_not_the_price_one() -> None:
    """NEGATIVE_FINDING: the slice statement's ordering holds for one objective.

    Implied volatility: 6.71e+07 -> 5.66e+02 -> 4.78e+02, strictly falling.
    Price: 6.48e+07 -> 7.83e+02 -> 9.52e+02, falling and then rising.

    Both are measured at the same parameters on the same quotes; the only
    difference is the units the residual is stated in.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    conditions = {}
    for objective in ("implied_vol", "price"):
        for maturities in (ONE_MATURITY, THREE_MATURITIES, SIX_MATURITIES):
            quotes = synthetic_quotes(REFERENCE, maturities)
            residuals, jacobian = residual_jacobian(
                quotes, MARKET, TRUE_REFERENCE, objective=objective
            )
            _, condition, _, _ = parameter_covariance(jacobian, residuals)
            conditions[(objective, len(maturities))] = condition

    implied = [conditions[("implied_vol", n)] for n in (1, 3, 6)]
    priced = [conditions[("price", n)] for n in (1, 3, 6)]
    assert implied[0] > implied[1] > implied[2]
    assert priced[0] > priced[1]
    assert priced[2] > priced[1]
    # The one-maturity problem is five decimal orders worse under either.
    assert implied[0] / implied[2] > 1e04
    assert priced[0] / priced[2] > 1e04


def test_vega_weighting_a_price_objective_is_the_implied_vol_objective() -> None:
    """Evidence class: EXACT_IDENTITY -- Gatheral's argument, as arithmetic.

    `d sigma_i / dp = (d V_i / dp) / vega_i` is the chain rule, so a price
    residual divided by vega has **the same Jacobian** as an implied-volatility
    residual whenever the vega is evaluated at the same volatility. On
    noise-free quotes the market volatility and the model volatility coincide,
    so the two Jacobians agree to **2.2e-16** relative and the condition
    numbers agree to fifteen digits. That is why vega weighting repairs the
    price objective's conditioning: it is not a heuristic, it is the same
    problem.

    On *noisy* quotes the two stop being identical -- the weight is frozen at
    the market volatility while the chain rule uses the model's -- and the
    difference is second order in the residual. The objective comparison below
    measures what that is worth.
    """
    assert EvidenceClass.EXACT_IDENTITY is EvidenceClass.EXACT_IDENTITY
    quotes = synthetic_quotes(REFERENCE, SIX_MATURITIES, as_price=True)
    weights = vega_weights(quotes, MARKET)
    _, weighted = residual_jacobian(
        quotes, MARKET, TRUE_REFERENCE, objective="price", weights=weights
    )
    _, volatility = residual_jacobian(
        quotes, MARKET, TRUE_REFERENCE, objective="implied_vol"
    )
    scale = np.maximum(np.abs(volatility), 1e-300)
    assert float(np.max(np.abs(weighted - volatility) / scale)) < 1e-14


FLAT_DIRECTION_KAPPA_WEIGHT = 0.90
"""At one maturity the right singular vector of the smallest singular value is
`v0 +0.2755, kappa -0.9259, theta -0.0750, xi -0.2474, rho +0.0009`: it is a
`kappa` direction with a `v0`/`xi` admixture, and `rho` is not in it at all.
That is the flat direction Gatheral (2006) chapter 3 and Cui et al. (2017)
section 3 both name, measured here rather than quoted."""


def test_the_flat_direction_is_kappa_against_xi_and_v0() -> None:
    """Evidence class: CLOSED_FORM.

    The smallest singular value at one maturity is 2.95e-08 against a largest
    of 1.98 -- seven and a half decimal orders. Its direction says which
    combination of parameters the smile at that maturity cannot see.
    """
    assert EvidenceClass.CLOSED_FORM is EvidenceClass.CLOSED_FORM
    quotes = synthetic_quotes(REFERENCE, ONE_MATURITY)
    _, jacobian = residual_jacobian(
        quotes, MARKET, TRUE_REFERENCE, objective="implied_vol"
    )
    _, _, right = np.linalg.svd(jacobian)
    direction = right[-1]
    weights = dict(zip(HESTON_PARAMETERS, np.abs(direction), strict=True))
    assert weights["kappa"] > FLAT_DIRECTION_KAPPA_WEIGHT
    assert weights["xi"] > weights["theta"]
    assert weights["v0"] > weights["theta"]
    assert weights["rho"] < 0.01
    # theta is in the flat direction too, but an order of magnitude behind:
    # at T = 1 the average variance is still far from its long-run level.
    assert 0.01 < weights["theta"] < 0.2


SINGLE_MATURITY_STARTS = (
    (0.04, 0.5, 0.25, 0.30, -0.5),
    (0.04, 1.0, 0.25, 0.45, -0.5),
    (0.04, 2.0, 0.25, 0.70, -0.5),
    (0.04, 8.0, 0.25, 1.40, -0.5),
    (0.04, 12.0, 0.25, 1.80, -0.5),
    (0.05, 6.0, 0.10, 1.20, -0.4),
)

SINGLE_MATURITY_SMILE_BUDGET = 1e-05
SINGLE_MATURITY_KAPPA_SPREAD = 2.0
"""Measured: those six starts fit the same one-maturity smile to a worst
implied-volatility RMSE of **2.88e-06** -- three-hundredths of a basis point --
while landing on `kappa` anywhere from **2.918 to 7.288** (a factor of 2.50,
against a true 4.0) and `xi` from **0.806 to 1.573** (true 1.0). Two of them
drive `v0` to the lower bound. Over the same six fits `rho` lands in
[-0.5012, -0.4961] and `theta` in [0.2302, 0.2494]: the two parameters the
smile's slope and level see are recovered, and the two the *term structure*
sees are not."""


@pytest.mark.slow
def test_one_maturity_recovers_the_smile_but_not_kappa_and_xi() -> None:
    """NEGATIVE_FINDING: the slice's central claim, measured.

    A calibration that finds parameters is not evidence that the parameters
    are identified. Six starts, one maturity, exact quotes: every fit
    reproduces the smile to better than 1e-05 in implied volatility, the
    objective values span one decimal order around 1e-11 (i.e. all of them are
    at the numerical floor), and the fitted `kappa` spans a factor of 2.5.
    Nothing in any single one of those six results says so; the condition
    number does, before the fit is run.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    quotes = synthetic_quotes(REFERENCE, ONE_MATURITY)
    fits = [
        calibrate_heston(quotes, MARKET, initial=start, objective="implied_vol")
        for start in SINGLE_MATURITY_STARTS
    ]
    for fit in fits:
        assert fit.rmse < SINGLE_MATURITY_SMILE_BUDGET

    kappas = np.array([fit.parameter("kappa") for fit in fits])
    xis = np.array([fit.parameter("xi") for fit in fits])
    rhos = np.array([fit.parameter("rho") for fit in fits])
    assert kappas.max() / kappas.min() > SINGLE_MATURITY_KAPPA_SPREAD
    assert xis.max() / xis.min() > 1.5
    # `rho` is fine: it is the smile's slope, and one smile determines it.
    assert float(np.max(np.abs(rhos - TRUE_REFERENCE[4]))) < 5e-03


def test_three_maturities_recover_the_same_six_starts_exactly() -> None:
    """The other half of the finding: it is the data, not the optimiser.

    The identical six starts, on three maturities instead of one, all land on
    `kappa = 4.00000` and `xi = 1.00000` to better than 1e-05. Nothing about
    the solver changed.
    """
    quotes = synthetic_quotes(REFERENCE, THREE_MATURITIES)
    for start in SINGLE_MATURITY_STARTS:
        fit = calibrate_heston(
            quotes, MARKET, initial=start, objective="implied_vol"
        )
        assert fit.success
        assert fit.parameter("kappa") == pytest.approx(4.0, abs=1e-05)
        assert fit.parameter("xi") == pytest.approx(1.0, abs=1e-05)


# --------------------------------------------------------------------------
# (c) The objective choice.
# --------------------------------------------------------------------------

OBJECTIVE_SEEDS = 10
OBJECTIVE_NOISE_BP = 20.0
OBJECTIVE_COMPARISON = {
    "price": (0.002029, 0.079817),
    "vega_price": (0.001908, 0.081344),
    "implied_vol": (0.001908, 0.081348),
}
"""`(implied-volatility RMSE, price RMSE)` of the fit, averaged over ten noise
draws at 20 bp on the 5-by-6 grid.

Each objective wins on its own metric and loses on the other, and it is a small
win both ways: the price fit is 1.9% better in price RMSE and 6.3% worse in
implied-volatility RMSE. That ordering is stable -- it holds on every seed set
and grid tried (10, 12 and 16 draws; three maturities and six).

What is **not** stable is the parameter accuracy, and that is the finding. The
mean absolute parameter errors of the three fits are within 0.87x to 1.63x of
each other with no consistent sign: over 10 draws the price objective is 1.50x
worse on `kappa` and 1.13x *better* on `xi`; over 16 draws it is 1.22x worse on
`kappa` and 1.07x better on `xi`; over 12 draws at three maturities it is
better on `v0`, `kappa` and `theta` and 1.63x worse on `rho`. At 20 bp on this
grid the objective choice is worth a measurable amount on the **fit** and
nothing that survives a change of seed on the **parameters**.

Vega-weighted prices and implied volatilities agree to 0.4% on every parameter
and to four significant figures on both metrics, which is the previous test's
exact identity surviving the noise: the two objectives differ only at second
order in the residual."""


def _fit_rmse(parameters, quotes) -> tuple[float, float]:
    """`(implied-volatility RMSE, price RMSE)` of one parameter vector."""
    model = HestonModel(*parameters)
    prices = heston_quote_values(model, MARKET, quotes, settings=FELLER_SATISFIED_COS)
    vol_squared = 0.0
    price_squared = 0.0
    for quote, price in zip(quotes, prices, strict=True):
        option = EuropeanOption(kind="call", strike=quote.strike, expiry=quote.expiry)
        model_vol = implied_volatility(float(price), option, MARKET)
        target_price = float(
            bs_price(
                S=MARKET.spot,
                K=quote.strike,
                T=quote.expiry,
                r=MARKET.rate(quote.expiry),
                sigma=quote.value,
                q=MARKET.dividend_yield(quote.expiry),
                kind="call",
            )
        )
        vol_squared += (model_vol - quote.value) ** 2
        price_squared += (float(price) - target_price) ** 2
    n = len(quotes)
    return math.sqrt(vol_squared / n), math.sqrt(price_squared / n)


def test_each_objective_wins_on_its_own_metric() -> None:
    """STATISTICAL for the fit, NEGATIVE_FINDING for the parameters.

    The same noisy surface fitted three ways -- to prices, to vega-weighted
    prices, and to implied volatilities. There is no free lunch: the price
    objective is the best fit *to prices* and the worst fit to implied
    volatilities, and the other two are the mirror image. That much is stable.

    The part that contradicts the written plan for this slice is what it is
    worth. The plan said to "report which objective wins on which metric",
    which presumes the metric is what decides. On these quotes it is not: the
    implied-volatility fit is 6.3% better in implied-volatility RMSE and the
    *parameters* are not separated at all, drifting between 0.87x and 1.63x of
    each other with no consistent sign across seed sets. Gatheral (2006)
    chapter 3's argument for the implied-volatility objective is about
    conditioning, and conditioning is where it shows (the previous section):
    the price objective's condition number gets *worse* with more maturities
    and the vega-weighted one does not. At 20 bp the residual noise swamps the
    difference in the answer.
    """
    assert EvidenceClass.STATISTICAL is EvidenceClass.STATISTICAL
    totals = {name: np.zeros(2) for name in OBJECTIVE_COMPARISON}
    parameter_errors = {name: np.zeros(5) for name in OBJECTIVE_COMPARISON}
    for seed in range(OBJECTIVE_SEEDS):
        volatility_quotes = _noisy_quotes(
            REFERENCE, SIX_MATURITIES, noise_bp=OBJECTIVE_NOISE_BP, seed=3000 + seed
        )
        price_quotes = [
            OptionQuote(
                strike=q.strike,
                expiry=q.expiry,
                kind="call",
                value=float(
                    bs_price(
                        S=MARKET.spot,
                        K=q.strike,
                        T=q.expiry,
                        r=MARKET.rate(q.expiry),
                        sigma=q.value,
                        q=MARKET.dividend_yield(q.expiry),
                        kind="call",
                    )
                ),
                value_type="price",
            )
            for q in volatility_quotes
        ]
        fits = {
            "price": calibrate_heston(
                price_quotes,
                MARKET,
                initial=CLEAN_START_REFERENCE,
                objective="price",
            ),
            "vega_price": calibrate_heston(
                price_quotes,
                MARKET,
                initial=CLEAN_START_REFERENCE,
                objective="price",
                weights=vega_weights(price_quotes, MARKET),
            ),
            "implied_vol": calibrate_heston(
                volatility_quotes,
                MARKET,
                initial=CLEAN_START_REFERENCE,
                objective="implied_vol",
            ),
        }
        for name, fit in fits.items():
            totals[name] += np.array(_fit_rmse(fit.parameters, volatility_quotes))
            parameter_errors[name] += np.abs(
                np.array(fit.parameters) - np.array(TRUE_REFERENCE)
            )

    means = {name: totals[name] / OBJECTIVE_SEEDS for name in totals}
    errors = {
        name: parameter_errors[name] / OBJECTIVE_SEEDS for name in parameter_errors
    }

    # Each objective is best on the metric it optimises.
    assert means["price"][1] < means["implied_vol"][1]
    assert means["implied_vol"][0] < means["price"][0]
    # Vega weighting tracks the implied-volatility objective, not the price one.
    assert means["vega_price"][0] == pytest.approx(means["implied_vol"][0], rel=0.05)
    # And the parameters, which is what a calibration is actually for: the
    # price objective is NOT uniformly worse, which is the finding.
    ratios = errors["price"] / errors["implied_vol"]
    assert float(np.max(ratios)) > 1.2
    assert float(np.min(ratios)) < 1.0
    # Vega weighting and implied volatility give the same parameters.
    vega_ratios = errors["vega_price"] / errors["implied_vol"]
    assert np.all(np.abs(vega_ratios - 1.0) < 0.02)


# --------------------------------------------------------------------------
# (d) Local minima, initialisation and multi-start.
# --------------------------------------------------------------------------

START_GRID = (
    (0.04, 4.0, 0.25, 1.0, -0.5),
    (0.08, 2.0, 0.15, 0.6, -0.2),
    (0.01, 0.5, 0.05, 0.3, -0.9),
    (0.20, 10.0, 0.50, 2.0, -0.1),
    (0.04, 1.0, 0.04, 1.5, 0.5),
    (0.10, 0.3, 0.10, 1.8, 0.8),
    (0.50, 0.1, 0.80, 3.0, -0.99),
    (0.02, 15.0, 0.02, 0.05, -0.05),
    (0.30, 6.0, 0.30, 0.2, 0.9),
    (0.005, 0.01, 0.005, 0.01, 0.0),
    (0.9, 19.0, 0.9, 4.5, 0.95),
    (0.06, 3.0, 0.90, 2.5, -0.75),
    (0.04, 0.2, 0.60, 4.0, -0.3),
    (0.15, 8.0, 0.02, 1.2, 0.3),
)
"""Fourteen starts: the truth, a mild perturbation, four corners of the box,
five with the **wrong sign** of `rho`, and six that are Feller-violating."""

GLOBAL_BASIN_TOLERANCE = 1e-02
"""Relative slack on the best objective for a start to count as having reached
it. Not doing any work: the successful starts agree on the objective to 1e-10
relative, so any tolerance between 1e-09 and 1e+02 gives the same count."""

MEASURED_GLOBAL_COUNT = 14
"""Out of 14, at every maturity count tried (one, three and six), on quotes
carrying 20 bp of implied-volatility noise.

This number was **11/14** before the payoff-leg rule went in, and the three
failures were not local minima: all three had a huge `theta` or `xi`, where a
fixed call leg's `e^b` made the pricer wrong by three decimal orders and the
residual flat, so each solver stalled within 1e-02 of its start. Fixing the
pricer fixed the optimiser. That is the lesson this row exists to carry: an
"optimisation failure rate" measured on a broken pricer measures the pricer.

What 14/14 does **not** mean is in the next test."""

SINGLE_MATURITY_KAPPA_SPREAD_NOISY = 2.0
"""At one maturity, all fourteen starts also reach the best objective within
1% -- and land on `kappa` anywhere from **7.197 to 15.004** against a true 4.0,
a factor of 2.08. Every convergence criterion a solver has is satisfied by all
fourteen. At three and six maturities the same fourteen agree on `kappa` to
five decimal places (4.059 and 4.005)."""


@pytest.mark.slow
@pytest.mark.parametrize(
    ("maturities", "label"),
    [(ONE_MATURITY, "1mat"), (THREE_MATURITIES, "3mat"), (SIX_MATURITIES, "6mat")],
    ids=["1mat", "3mat", "6mat"],
)
def test_every_start_reaches_the_best_objective(maturities, label) -> None:
    """NEGATIVE_FINDING: 14/14 everywhere, and at one maturity that is empty.

    The honest form of "did the calibration work" is two questions, and only
    the first has a good answer here. Every one of fourteen spread starts --
    five with the wrong sign of `rho`, six outside the Feller region, four in
    the corners of the box -- reaches the best objective found, at one, three
    and six maturities alike. The optimiser is not the problem.

    At three and six maturities they also agree on the answer. At one they do
    not: they agree on the objective to 1% and disagree on `kappa` by a factor
    of 2.08. No stopping rule can see that, which is why this slice's product
    is a condition number and not a success rate.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    quotes = _noisy_quotes(
        REFERENCE, maturities, noise_bp=OBJECTIVE_NOISE_BP, seed=3000
    )
    fits = [
        calibrate_heston(quotes, MARKET, initial=start, objective="implied_vol")
        for start in START_GRID
    ]
    best = min(fit.objective for fit in fits)
    reached = [
        fit
        for fit in fits
        if fit.objective <= best * (1.0 + GLOBAL_BASIN_TOLERANCE)
    ]
    assert len(reached) == MEASURED_GLOBAL_COUNT

    kappas = np.array([fit.parameter("kappa") for fit in reached])
    if len(maturities) == 1:
        # Same objective, different model.
        assert kappas.max() / kappas.min() > SINGLE_MATURITY_KAPPA_SPREAD_NOISY
        assert kappas.min() > 4.0
    else:
        assert kappas.max() / kappas.min() < 1.0 + 1e-05
        assert kappas.mean() == pytest.approx(4.0, rel=0.02)


@pytest.mark.slow
def test_multi_start_ties_the_best_single_start_rather_than_beating_it() -> None:
    """`n_starts` finds the best objective, and buys nothing here.

    The slice statement asked for a multi-start option and for a demonstration
    that it finds the best objective. It does -- from the worst start in the
    grid, `MULTISTART_COUNT = 6` uniform draws inside the bounds reach
    3.116984e-05, the same objective every single start reaches, to 1e-09
    relative. What it does not do is *improve* on a single start, because on
    this problem there is nothing to improve: 14/14 already get there.

    That is the measured fact and it is worth stating plainly rather than
    dressing a tie up as a rescue. Multi-start was worth something when the
    pricer was broken (it recovered the global optimum from all three of the
    stalled starts); once the pricer was fixed there were no stalls left.
    """
    quotes = _noisy_quotes(
        REFERENCE, SIX_MATURITIES, noise_bp=OBJECTIVE_NOISE_BP, seed=3000
    )
    dead = (0.50, 0.1, 0.80, 3.0, -0.99)
    single = calibrate_heston(quotes, MARKET, initial=dead, objective="implied_vol")
    multi = calibrate_heston(
        quotes,
        MARKET,
        initial=dead,
        objective="implied_vol",
        n_starts=MULTISTART_COUNT,
    )
    assert len(multi.starts) == MULTISTART_COUNT
    assert multi.starts[0].initial == dead
    assert multi.objective <= single.objective
    assert multi.objective == pytest.approx(single.objective, rel=1e-06)
    assert np.max(
        np.abs(np.array(multi.parameters) - np.array(single.parameters))
    ) < 1e-05


MULTISTART_COUNT = 6
"""Draws used by the multi-start test. Measured: `n_starts` of 4, 6, 8, 12 and
16 all reach the same objective to six digits, at 1.5, 2.6, 3.4, 5.4 and 7.2
seconds. Six is the cheapest that is not the minimum tried."""


def test_the_multi_start_draw_is_reproducible() -> None:
    """The same seed draws the same points, so a reported rate is repeatable."""
    quotes = synthetic_quotes(REFERENCE, THREE_MATURITIES)
    first = calibrate_heston(
        quotes,
        MARKET,
        initial=CLEAN_START_REFERENCE,
        objective="implied_vol",
        n_starts=MULTISTART_COUNT,
    )
    second = calibrate_heston(
        quotes,
        MARKET,
        initial=CLEAN_START_REFERENCE,
        objective="implied_vol",
        n_starts=MULTISTART_COUNT,
    )
    assert [s.initial for s in first.starts] == [s.initial for s in second.starts]
    assert first.objective == second.objective


def test_explicit_starts_take_precedence_over_the_draw() -> None:
    quotes = synthetic_quotes(REFERENCE, THREE_MATURITIES)
    fit = calibrate_heston(
        quotes,
        MARKET,
        initial=CLEAN_START_REFERENCE,
        objective="implied_vol",
        starts=SINGLE_MATURITY_STARTS,
        n_starts=99,
    )
    assert len(fit.starts) == len(SINGLE_MATURITY_STARTS)
    assert fit.starts[0].initial == SINGLE_MATURITY_STARTS[0]
    assert fit.objective == min(s.objective for s in fit.starts)
