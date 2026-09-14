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
`2.67e-12 / 1.00e-02 = 2.67e-10`, against a worst measured parameter error of
1.48e-10, so the bound is tight to a factor of 1.8. The factor of 10 below is
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
        reference         price        +2.9e-09  -5.4e-07  +3.3e-09  -4.1e-07  -1.3e-07
        reference         implied_vol  +2.2e-12  -1.5e-10  +2.1e-12  -9.7e-12  +2.9e-12
        Feller-violating  price        -2.5e-10  +8.2e-08  -5.8e-09  +1.3e-09  +5.8e-09
        Feller-violating  implied_vol  -2.2e-10  -9.1e-09  +1.6e-09  +5.3e-09  +1.0e-09

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
it. Not a formality: the eleven successful starts agree on the objective to
1e-06 relative and on every parameter to 1e-04, while the three failures are
**four to seven decimal orders** away. Any tolerance between 1e-06 and 1e+02
gives the same count, so the number is not doing any work."""

MEASURED_GLOBAL_COUNT = 11
"""Out of 14. The three that fail are `(0.10, 0.3, 0.10, 1.8, +0.8)`,
`(0.50, 0.1, 0.80, 3.0, -0.99)` and `(0.04, 0.2, 0.60, 4.0, -0.3)`, and their
cause is measured rather than assumed: all three have a huge `theta` or `xi`,
which is exactly where `COS_LOG_RANGE_CAP` is binding hard, and each terminates
within 1e-03 of where it started. They are not local minima of the true
objective -- they are places where the *pricer* is wrong by three decimal
orders and the residual is therefore flat. Before the clip was added the same
grid scored 5/14 and the failures returned their starting vector exactly."""


@pytest.mark.slow
def test_a_minority_of_starts_stall_and_the_cause_is_the_pricer() -> None:
    """NEGATIVE_FINDING: 11/14, and the three failures have a diagnosis.

    The honest form of "did the calibration work". Eleven of fourteen starts
    reach the same optimum, to 1e-04 in every parameter. The other three stop
    essentially where they started, with objectives four to seven decimal
    orders worse, and all three sit where the COS range clip is binding.
    """
    assert EvidenceClass.NEGATIVE_FINDING is EvidenceClass.NEGATIVE_FINDING
    quotes = _noisy_quotes(
        REFERENCE, SIX_MATURITIES, noise_bp=OBJECTIVE_NOISE_BP, seed=3000
    )
    fits = [
        calibrate_heston(quotes, MARKET, initial=start, objective="implied_vol")
        for start in START_GRID
    ]
    best = min(fit.objective for fit in fits)
    reached = [fit for fit in fits if fit.objective <= best * (1.0 + GLOBAL_BASIN_TOLERANCE)]
    stalled = [
        (start, fit)
        for start, fit in zip(START_GRID, fits, strict=True)
        if fit.objective > best * (1.0 + GLOBAL_BASIN_TOLERANCE)
    ]
    assert len(reached) == MEASURED_GLOBAL_COUNT
    assert len(stalled) == len(START_GRID) - MEASURED_GLOBAL_COUNT

    # The winners agree with each other, not merely with the tolerance.
    winner = np.array(reached[0].parameters)
    for fit in reached:
        assert np.max(np.abs(np.array(fit.parameters) - winner)) < 1e-04
    # The losers went nowhere.
    for start, fit in stalled:
        assert np.max(np.abs(np.array(fit.parameters) - np.array(start))) < 1e-02
        assert fit.objective > 1e03 * best


def test_multi_start_finds_the_best_objective_from_a_dead_start() -> None:
    """`n_starts` is the cheapest answer to the failure rate above.

    Started at `(0.50, 0.1, 0.80, 3.0, -0.99)`, which on its own is one of the
    three stalls, six uniform draws inside the bounds recover the same optimum
    the eleven good starts reach -- and so do 4, 8, 12 and 16 draws, to the
    same six digits of the objective, so `MULTISTART_COUNT = 6` is chosen for
    its run time and not to make the test pass. `result.starts` carries every
    start so the success rate is inspectable rather than implied.
    """
    quotes = _noisy_quotes(
        REFERENCE, SIX_MATURITIES, noise_bp=OBJECTIVE_NOISE_BP, seed=3000
    )
    dead = (0.50, 0.1, 0.80, 3.0, -0.99)
    single = calibrate_heston(quotes, MARKET, initial=dead, objective="implied_vol")
    multi = calibrate_heston(
        quotes, MARKET, initial=dead, objective="implied_vol", n_starts=MULTISTART_COUNT
    )
    reference = calibrate_heston(
        quotes, MARKET, initial=CLEAN_START_REFERENCE, objective="implied_vol"
    )
    assert len(multi.starts) == MULTISTART_COUNT
    assert multi.starts[0].initial == dead
    assert multi.objective < single.objective / 1e03
    assert multi.objective == pytest.approx(reference.objective, rel=1e-03)
    assert np.max(
        np.abs(np.array(multi.parameters) - np.array(reference.parameters))
    ) < 1e-03


MULTISTART_COUNT = 6
"""Draws used by the multi-start test. Measured: `n_starts` of 4, 6, 8, 12 and
16 all reach `3.116984e-05` on the same data, at 1.5, 2.6, 3.4, 5.4 and 7.2
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
