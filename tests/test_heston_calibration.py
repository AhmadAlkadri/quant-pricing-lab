"""Calibrating Heston: the gradient, the constraints, and the pricer's envelope.

This file starts with the two claims that are about the *machinery* rather than
about identifiability -- the analytic Jacobian and the constraint handling --
because everything later depends on them being right.

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
    domain**: from a start at `(0.04, 0.05, 0.25, 3.0, -0.95)` it terminates
    on its own `xtol` at `theta = -1.70`, a negative long-run variance, and the
    result reports `in_domain = False` and `model = None` rather than coercing
    it into a `HestonModel` that cannot exist. The bounded run from the same
    start recovers the truth at an RMSE of 2.56e-07. A Feller-violating optimum
    is **reported** and not forbidden: the Feller-violating study set is
    recovered to 1e-06 with `feller_number = 0.0800` and
    `feller_satisfied = False`. EXACT_IDENTITY for the bound arithmetic,
    NEGATIVE_FINDING for what the unconstrained route does with the same data.

(g) **The COS envelope.** Two facts about the pricer this calibrator uses, both
    of which a calibration hits and a single pricing call does not. First, the
    Slice 15 call/put asymmetry: at the package defaults the COS *call* is at
    1.2e-12 against a Lewis integral of the same transform while the COS *put*
    is at 2.7e-08, so this module prices the call and takes the put by parity.
    That **contradicts** the written plan for this slice, which had them the
    other way round. Second, the range's two-sided failure and the
    `COS_LOG_RANGE_CAP` guard: without the clip a parameter search in the far
    corner of the bounds reports a 100-strike call at **-3.2e+43**, and a
    solver started there dies on its first step.

Sources: Gatheral (2006) chapter 3; Cui, del Bano Rollin and Germano (2017)
EJOR 263(2) section 3; Nocedal and Wright (2006) chapter 10. Every number in
this file is measured here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpl.calibration import (
    COS_LOG_RANGE_CAP,
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

COS_CALL_BUDGET = 1e-11
COS_PUT_FLOOR = 1e-09
"""The call leg is at 1.24e-12 and the put leg at 2.70e-08 on the reference set
at `T = 1`, `K = 100`, at the package COS defaults. Slice 15 measured the same
asymmetry and its cause (`rho < 0` puts the missing tail mass at the range's
lower end, which is the put's end); what is new here is that a calibrator has
to choose, and the written plan for this slice chose the wrong leg."""


@pytest.mark.parametrize("strike", STRIKES)
def test_the_calibrator_prices_the_call_because_the_put_is_four_orders_worse(
    strike: float,
) -> None:
    """NEGATIVE_FINDING, and a contradicted slice statement.

    The plan for this slice said to price the **put** by COS and take the call
    by parity. It is the other way round. Both legs are the same cosine sum
    against the same coefficients and differ only in which end of `[a, b]` the
    payoff coefficient integrates from; with `rho = -0.5` the density is
    left-skewed and the missing mass sits at the lower end. Measured here at
    every strike, so the module prices the call.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    expiry = 1.0
    reference_call, _ = lewis_call(
        REFERENCE,
        s0=MARKET.spot,
        strike=strike,
        expiry=expiry,
        rate=0.01,
        dividend=0.02,
        limit=800,
        tolerance=1e-13,
    )
    spot_leg = MARKET.spot * math.exp(-0.02 * expiry)
    strike_leg = strike * math.exp(-0.01 * expiry)
    reference_put = reference_call - spot_leg + strike_leg

    calls, _ = cos_call_prices(
        TRUE_REFERENCE,
        s0=MARKET.spot,
        strikes=np.array([strike]),
        expiry=expiry,
        rate=0.01,
        dividend=0.02,
        settings=FELLER_SATISFIED_COS,
    )
    direct_put = cos_price(
        REFERENCE,
        s0=MARKET.spot,
        strike=strike,
        expiry=expiry,
        rate=0.01,
        dividend=0.02,
        kind="put",
        n_terms=FELLER_SATISFIED_COS.n_terms,
        truncation_l=FELLER_SATISFIED_COS.truncation_l,
    ).value

    call_error = abs(float(calls[0]) - reference_call)
    put_error = abs(direct_put - reference_put)
    assert call_error < COS_CALL_BUDGET
    assert put_error > COS_PUT_FLOOR
    assert put_error > 1000.0 * call_error


def test_the_feller_branch_picks_the_settings_and_neither_setting_does_both() -> None:
    """NEGATIVE_FINDING: the COS range fails on **both** sides, so it branches.

    At the money, against a Lewis integral of the same transform:

        settings          reference set (Feller 4)   violating set (Feller 0.08)
        L=10,  N=256              1.2e-12                    8.0e-04
        L=28, N=4096              1.4e-08                    1.2e-10

    Too narrow truncates the left tail; too wide amplifies round-off, because
    the payoff coefficients carry `e^b`. There is no single setting that is
    machine-accurate on both, which is why `default_cos_settings` is a branch.
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

    # Each set is better at its own branch, and by decimal orders.
    assert errors[("reference", "default")] < 1e-11
    assert errors[("reference", "wide")] > 1e-09
    assert errors[("violating", "wide")] < 1e-09
    assert errors[("violating", "default")] > 1e-05
    # ... and neither column is uniformly better than the other.
    assert errors[("reference", "default")] < errors[("violating", "default")]
    assert errors[("violating", "wide")] < errors[("reference", "wide")]


def test_the_log_range_cap_turns_an_overflow_into_a_truncation_error() -> None:
    """NEGATIVE_FINDING: the corner of the bounds overflows without the clip.

    At `(0.5, 0.1, 0.8, 3.0, -0.99)` and `T = 3` the Feller branch asks for
    `L = 28`, which puts the range's upper end at 113.8 in log-return units.
    The payoff coefficients carry `e^b`, so the 100-strike call comes back as
    **-3.2e+43** against a true value of 11.98. Clipping `[a, b]` to
    `[-10, 10]` makes it 1.8e+03 -- still wrong, because the range is now far
    too narrow there, but finite, of a plausible order and carrying a gradient
    the solver can use to walk back out.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    corner = (0.5, 0.1, 0.8, 3.0, -0.99)
    expiry = 3.0
    reference, _ = lewis_call(
        HestonModel(*corner),
        s0=MARKET.spot,
        strike=100.0,
        expiry=expiry,
        rate=0.01,
        dividend=0.02,
        limit=400,
        tolerance=1e-11,
    )
    clipped, _ = cos_call_prices(
        corner,
        s0=MARKET.spot,
        strikes=np.array([100.0]),
        expiry=expiry,
        rate=0.01,
        dividend=0.02,
        settings=default_cos_settings(corner),
    )
    unclipped = cos_price(
        HestonModel(*corner),
        s0=MARKET.spot,
        strike=100.0,
        expiry=expiry,
        rate=0.01,
        dividend=0.02,
        kind="call",
        n_terms=FELLER_VIOLATED_COS.n_terms,
        truncation_l=FELLER_VIOLATED_COS.truncation_l,
    ).value

    assert abs(unclipped) > 1e30
    assert abs(float(clipped[0])) < 1e06
    assert abs(float(clipped[0]) - reference) < abs(unclipped - reference) / 1e30
    assert COS_LOG_RANGE_CAP == 16.0


def test_the_cap_does_not_bind_on_either_study_set() -> None:
    """The guard is for the corners; it costs the study points nothing.

    Checked directly on the range rather than on a price: the widest cell used
    anywhere in this file is the Feller-violating set at `T = 3`, whose upper
    endpoint is `c1 + 28 sqrt(c2) = 14.13` against a cap of 16. The reference
    set's widest is 8.49 at the same maturity. So every other number here is a
    statement about the calibration and not about the guard.
    """
    for parameters, maturities in (
        (TRUE_REFERENCE, SIX_MATURITIES),
        (TRUE_FELLER_VIOLATED, SIX_MATURITIES),
    ):
        settings = default_cos_settings(parameters)
        v0, kappa, theta, xi, rho = parameters
        for expiry in maturities:
            c1, c2 = heston_log_return_cumulants(
                expiry,
                rate=0.01,
                dividend=0.02,
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
            )
            half_width = settings.truncation_l * math.sqrt(c2)
            assert c1 + half_width < COS_LOG_RANGE_CAP
            assert c1 - half_width > -COS_LOG_RANGE_CAP


def test_the_cosine_sum_amplifies_the_transforms_own_round_off() -> None:
    """NEGATIVE_FINDING: `e^b` is a condition number on the transform, too.

    `cos_call_prices` recomputes the characteristic function inside
    `heston_charfn_gradient`; `qpl.engines.fourier.cos.cos_price` reads
    `HestonModel.characteristic_function`. The two transforms agree to
    **2.5e-16** absolute -- they are the same formula -- and the two prices
    then differ by up to **3.5e-10** relative. The gap is not a disagreement
    about the model; it is the `e^b` cancellation in the payoff coefficients
    turning 1e-16 in `phi` into 1e-10 in a price, measured at the widest cell
    used here (`b = 14.13`). It is the same mechanism as the overflow above,
    seen at a survivable size, and it is why `COS_LOG_RANGE_CAP` exists.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    ratios = []
    for parameters in (TRUE_REFERENCE, TRUE_FELLER_VIOLATED):
        settings = default_cos_settings(parameters)
        model = HestonModel(*parameters)
        for expiry in SIX_MATURITIES:
            here, _ = cos_call_prices(
                parameters,
                s0=MARKET.spot,
                strikes=np.array(STRIKES),
                expiry=expiry,
                rate=0.01,
                dividend=0.02,
                settings=settings,
            )
            engine = np.array(
                [
                    cos_price(
                        model,
                        s0=MARKET.spot,
                        strike=k,
                        expiry=expiry,
                        rate=0.01,
                        dividend=0.02,
                        kind="call",
                        n_terms=settings.n_terms,
                        truncation_l=settings.truncation_l,
                    ).value
                    for k in STRIKES
                ]
            )
            ratios.append(float(np.max(np.abs(here - engine) / np.abs(engine))))
    assert max(ratios) < 1e-08
    assert max(ratios) > 1e-13


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
"""From the start `(0.04, 0.05, 0.25, 3.0, -0.95)`, the bounded run recovers the
truth at a root-mean-square residual of 2.56e-07, while the unconstrained one
terminates at `theta = -1.70` -- a **negative long-run variance**, outside the
Heston domain entirely -- with a residual of 2.4e+03. The budget below is many
orders inside that ratio, because the point is the sign of the answer and not
its size."""


def test_the_unconstrained_route_leaves_the_domain() -> None:
    """NEGATIVE_FINDING: `method="lm"` takes no bounds and it shows.

    `scipy.optimize.least_squares(method='lm')` has no bound support at all, so
    `calibrate_heston` refuses `bounds=` there rather than ignoring them, and
    the only thing between the search and an inadmissible parameter vector is
    `DOMAIN_PENALTY` -- which keeps the residual *evaluable* outside the
    domain, not the search inside it. From a start already pressed against the
    correlation bound, the unconstrained run converges (by its own `xtol`) to a
    **negative `theta`**, so `CalibrationResult.model` is `None` and
    `in_domain` is `False`: the answer is reported as what it is rather than
    coerced into a `HestonModel` that cannot exist. The bounded run from the
    same start recovers the truth.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    quotes = synthetic_quotes(REFERENCE, SIX_MATURITIES)
    start = (0.04, 0.05, 0.25, 3.0, -0.95)
    bounded = calibrate_heston(quotes, MARKET, initial=start, objective="price")
    free = calibrate_heston(
        quotes, MARKET, initial=start, objective="price", method="lm"
    )
    assert not free.in_domain
    assert free.model is None
    assert free.parameter("theta") < 0.0
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
