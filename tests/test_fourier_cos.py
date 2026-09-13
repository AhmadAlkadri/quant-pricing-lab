"""The COS method under Black-Scholes: what it costs in N, in L, and in Greeks.

Five claims are measured here.

(a) **The price is the closed form.** Vanilla calls and puts and cash-or-nothing
    digitals, at five specification points, agree with the Black-Scholes
    formulas to about 1e-12 at the package defaults. CLOSED_FORM.

(b) **The error decays faster than any power of 1/N, and the rate is derivable.**
    For a Gaussian log-return law the cosine coefficient at index k carries the
    factor `|phi(u_k)| = exp(-sigma^2 T u_k^2 / 2)` with `u_k = k pi / (b - a)`
    and `b - a = 2 L sigma sqrt(T)`, so

        log10 err(N) ~ -pi^2 N^2 / (8 L^2 ln 10),

    a *Gaussian* decay in N, not merely an exponential one. That prediction is
    checked against the measured slope at three values of L. CONVERGENCE_ORDER,
    with the rate itself CLOSED_FORM. A power-law fit is reported alongside and
    is deliberately shown to be meaningless. NEGATIVE_FINDING.

(c) **The truncation range L.** Too small and the error floors at a level that
    no amount of N removes; too large and the `L^2` in the denominator above
    slows the decay. Both halves measured. NEGATIVE_FINDING for the floor.

(d) **The identities.** Put-call parity and the digital's static replication
    hold to round-off. Parity is a real check here and only here: the COS put
    is computed from its own payoff coefficients, not from the call by parity.
    EXACT_IDENTITY.

(e) **Greeks.** Delta and gamma are exact derivatives of the same truncated sum
    and land within 2e-14 and 1e-17 of the closed forms; vega, theta and rho
    are central differences of the transform price and land within 2e-07.
    CLOSED_FORM.

Source for the method: Fang and Oosterlee (2008), SIAM J. Sci. Comput. 31(2),
826-848. Every number below is measured in this repository; none is quoted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pytest

from qpl.engines.fourier import (
    FourierConfig,
    LogReturnCumulants,
    black_scholes_characteristic_function,
    black_scholes_log_return_cumulants,
    characteristic_function_model,
    cos_truncation_range,
)
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import DigitalOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass, fit_convergence_order


@dataclass(frozen=True)
class Point:
    """One pricing point, as scalars."""

    name: str
    spot: float
    strike: float
    expiry: float
    rate: float
    dividend: float
    sigma: float

    def market(self) -> Market:
        return Market(
            spot=self.spot,
            rate_curve=FlatRateCurve(self.rate),
            dividend_curve=FlatDividendCurve(self.dividend),
        )

    def model(self) -> BlackScholesModel:
        return BlackScholesModel(sigma=self.sigma)

    def instrument(self, kind: str, payoff: str):
        if payoff == "digital":
            return DigitalOption(
                kind=kind, strike=self.strike, expiry=self.expiry, cash=1.0
            )
        return EuropeanOption(kind=kind, strike=self.strike, expiry=self.expiry)


POINTS: tuple[Point, ...] = (
    Point("atm_1y", 100.0, 100.0, 1.0, 0.05, 0.00, 0.20),
    Point("otm_9m_div", 100.0, 110.0, 0.75, 0.03, 0.01, 0.25),
    Point("itm_1y_div", 120.0, 90.0, 1.0, 0.03, 0.05, 0.35),
    Point("short_atm", 100.0, 100.0, 0.05, 0.01, 0.00, 0.30),
    Point("long_high_vol", 80.0, 120.0, 2.0, 0.02, 0.03, 0.40),
)

KINDS = ("call", "put")
PAYOFFS = ("vanilla", "digital")

PRICE_TOLERANCE = 1e-11
"""Worst measured |COS - closed form| over the 20 (point, kind, payoff) cells at
the package defaults is 2.91e-12, at `long_high_vol` (T = 2, sigma = 40%, so the
widest range and the largest `e^b` weight in the call's payoff coefficients).
The tolerance keeps a factor of about 3.4 over that."""

DELTA_TOLERANCE = 2e-13
GAMMA_TOLERANCE = 1e-16
"""Worst measured 1.74e-14 and 9.54e-18. Both are derivatives of the *same*
truncated sum as the price, so they inherit its accuracy and nothing else."""

BUMPED_TOLERANCE = 2e-07
"""vega, theta and rho are central differences (see
`qpl.engines.fourier.pricers.VEGA_BUMP`). Worst measured 1.34e-07, the theta at
`short_atm`, where T = 0.05 makes the third derivative in T large."""


def _cos(point: Point, kind: str, payoff: str, **cfg_kwargs) -> float:
    return price(
        point.instrument(kind, payoff),
        point.model(),
        point.market(),
        method="fourier",
        cfg=FourierConfig(**cfg_kwargs),
    ).value


def _exact(point: Point, kind: str, payoff: str) -> float:
    return price(point.instrument(kind, payoff), point.model(), point.market()).value


def _cos_error(point: Point, kind: str, payoff: str, **cfg_kwargs) -> float:
    return abs(_cos(point, kind, payoff, **cfg_kwargs) - _exact(point, kind, payoff))


# --------------------------------------------------------------------------
# (a) The price.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("payoff", PAYOFFS)
def test_cos_price_matches_the_closed_form(point: Point, kind: str, payoff: str) -> None:
    """Evidence class: CLOSED_FORM."""
    assert EvidenceClass.CLOSED_FORM is EvidenceClass.CLOSED_FORM
    assert _cos(point, kind, payoff) == pytest.approx(
        _exact(point, kind, payoff), abs=PRICE_TOLERANCE
    )


def test_price_result_metadata_records_what_was_run() -> None:
    point = POINTS[0]
    result = price(
        point.instrument("call", "vanilla"),
        point.model(),
        point.market(),
        method="fourier",
        cfg=FourierConfig(n_terms=64, truncation_l=8.0),
    )
    assert result.stderr is None
    assert result.meta is not None
    assert result.meta["method"] == "fourier"
    assert result.meta["fourier_method"] == "cos"
    assert result.meta["n_terms"] == 64
    assert result.meta["truncation_l"] == 8.0
    lower, upper = result.meta["truncation_range"]
    # The range is a statement about the log return, so it must not move with
    # the spot -- which is exactly what makes the COS delta a closed form.
    assert lower < 0.0 < upper
    cumulants = black_scholes_log_return_cumulants(
        point.expiry, rate=point.rate, dividend=point.dividend, sigma=point.sigma
    )
    width = 8.0 * cumulants.truncation_width()
    assert math.isclose(upper - lower, 2.0 * width, rel_tol=1e-12)


# --------------------------------------------------------------------------
# (b) Convergence in N.
# --------------------------------------------------------------------------

CONVERGENCE_TERMS = (8, 12, 16, 20, 24, 28, 32)
"""The window where the error is above the floating-point floor at L = 10.

The slice statement proposed fitting over N in {16, 32, 64, 128}. That window
is **wrong for this problem** and the test below says so: by N = 48 the error is
already at 1e-13, so two of those four points measure round-off rather than
truncation and the fit is a fit through noise. The honest window stops at 32."""

FLOOR_TERMS = 48
"""The first `n_terms` at which the vanilla error is at the floating-point floor."""


@pytest.mark.parametrize("point", POINTS[:3], ids=[p.name for p in POINTS[:3]])
@pytest.mark.parametrize("payoff", PAYOFFS)
def test_cos_error_is_not_a_power_law_and_beats_every_power(
    point: Point, payoff: str
) -> None:
    """Evidence class: CONVERGENCE_ORDER, plus NEGATIVE_FINDING on the fit itself.

    Two things are asserted and they say opposite-looking things on purpose.
    The *fitted* power-law order is large -- around 9 to 10, well past the 4 the
    slice statement asked for -- and the fit's log-space residual is around 1.4,
    which for a genuine power law would be ~0. A large order with a large
    residual is the signature of something that is not a power law at all, and
    the ratio test below is the direct statement: each doubling of the term
    count multiplies the *rate* of improvement, which no `C / N^p` can do.
    """
    errors = [_cos_error(point, "call", payoff, n_terms=n) for n in CONVERGENCE_TERMS]
    assert all(e > 0.0 for e in errors)

    fit = fit_convergence_order([1.0 / n for n in CONVERGENCE_TERMS], errors)
    assert fit.order > 4.0, fit.order
    assert fit.residual > 0.5, fit.residual

    # Strictly decreasing, and the decrement in log-error grows: super-linear
    # convergence in the exponent, not a straight line in log-log.
    assert all(a > b for a, b in pairwise(errors)), errors
    drops = [math.log10(a / b) for a, b in pairwise(errors)]
    assert all(a < b for a, b in pairwise(drops)), drops


@pytest.mark.parametrize("truncation_l", (10.0, 20.0, 40.0))
def test_cos_decay_rate_matches_the_gaussian_prediction(truncation_l: float) -> None:
    """Evidence class: CONVERGENCE_ORDER against a CLOSED_FORM rate.

    For Black-Scholes the log return is Gaussian, so
    `|phi(u)| = exp(-sigma^2 T u^2 / 2)`. On the COS grid `u_k = k pi / (b - a)`
    with `b - a = 2 L sigma sqrt(T)`, so the k-th coefficient is damped by
    `exp(-k^2 pi^2 / (8 L^2))`, independent of sigma, T, the spot and the strike.
    The truncation error is set by the first dropped coefficient, hence

        log10 err(N) ~ -pi^2 N^2 / (8 L^2 ln 10).

    The `sigma` and `T` cancelling is the reason the same slope appears at every
    specification point, and the `L^2` in the denominator is the reason a larger
    range costs term count: doubling L quadruples the N needed for a given error.
    Measured/predicted slope ratios: 1.085 (L = 10), 1.071 (L = 20), 1.037
    (L = 40) -- the prediction is the first dropped coefficient and ignores the
    payoff weight, so it is expected to be slightly optimistic, and it is.
    """
    point = POINTS[0]
    predicted = -(math.pi**2) / (8.0 * truncation_l**2) / math.log(10.0)

    terms, logs = [], []
    for n in range(8, 401, 8):
        err = _cos_error(point, "call", "vanilla", n_terms=n, truncation_l=truncation_l)
        if FIT_FLOOR < err < 1.0:
            terms.append(float(n))
            logs.append(math.log10(err))
    assert len(terms) >= 3, terms

    slope, _ = np.polyfit(np.array(terms) ** 2, np.array(logs), 1)
    assert 1.0 <= slope / predicted <= 1.2, (slope, predicted)


def test_cos_reaches_the_floating_point_floor_and_stays_there() -> None:
    """NEGATIVE_FINDING: the slice statement's convergence window straddles it.

    The statement asked for "the error at N = 256 below 1e-12". It is -- but so
    is the error at N = 48, and the value at N = 256 is *the same double* as the
    value at N = 64. Beyond the floor the term count buys nothing, and a study
    that samples there is measuring the closed form's own round-off.
    """
    point = POINTS[0]
    at_floor = [
        _cos(point, "call", "vanilla", n_terms=n) for n in (FLOOR_TERMS * 2, 128, 256, 512)
    ]
    assert all(value == at_floor[0] for value in at_floor)
    assert _cos_error(point, "call", "vanilla", n_terms=FLOOR_TERMS) < 1e-12
    assert _cos_error(point, "call", "vanilla", n_terms=256) < 1e-12


# --------------------------------------------------------------------------
# (c) The truncation range.
# --------------------------------------------------------------------------

FIT_FLOOR = 1e-9
"""Errors below this are excluded from the decay fit.

Not a round number chosen for tidiness: the round-off floor itself moves with
L, because the call's payoff coefficient integrates `e^z` over `[a, b]` and so
carries a factor `e^b`. Measured floors at `atm_1y`: 2.7e-14 at L = 10,
5.9e-13 at L = 20, 5.5e-11 at L = 40. A fit that included those points would
report the floor's slope (zero) rather than the truncation rate."""

RANGE_LEVELS = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 20.0)
"""The L ladder. The slice statement asked for {4, 6, 8, 10, 12}; 2 and 20 are
added at the ends because the failure modes are at the ends."""


def test_cos_truncation_range_has_a_valley_around_the_recommended_l() -> None:
    """Evidence class: NEGATIVE_FINDING (both ends), CLOSED_FORM in the middle.

    Measured at N = 256, `atm_1y`, vanilla call:

        L     2        4        6        8       10       12       14       20
        err  4.98e-01 6.25e-04 2.04e-08 2.66e-14 2.66e-14 1.60e-14 7.90e-13 5.88e-13

    Two different failure modes bracket the usable band. Below L = 8 the range
    is too narrow and the missing tail mass is an error the term count cannot
    touch (see the next test). Above L = 12 the error starts *growing* again,
    for the reason the decay-rate test derives: the Gaussian damping of the k-th
    cosine coefficient is `exp(-k^2 pi^2 / (8 L^2))`, so a wider range needs more
    terms for the same accuracy, and at fixed N the accumulated round-off of a
    longer sum is what is left. Fang and Oosterlee's L = 10 sits in the middle of
    the valley rather than at its edge, which is what makes it a good default
    rather than a lucky one.
    """
    point = POINTS[0]
    errors = {
        level: _cos_error(point, "call", "vanilla", n_terms=256, truncation_l=level)
        for level in RANGE_LEVELS
    }
    # The narrow end: monotone improvement, and still far from the floor at 6.
    assert errors[2.0] > errors[4.0] > errors[6.0] > errors[8.0]
    assert errors[4.0] > 1e-5
    assert errors[6.0] > 1e-10
    # The valley.
    assert max(errors[8.0], errors[10.0], errors[12.0]) < 1e-13
    # The wide end: worse than the valley, by more than an order of magnitude.
    assert errors[14.0] > 10.0 * errors[10.0]
    assert errors[20.0] > 10.0 * errors[10.0]


def test_a_too_narrow_range_is_an_error_the_term_count_cannot_remove() -> None:
    """NEGATIVE_FINDING: at L = 4 the price is the *same double* for N = 32..1024.

    Range truncation and series truncation are two different errors. The series
    error vanishes with N; the range error is a statement about how much of the
    density was left outside `[a, b]` and does not depend on N at all. At L = 4
    the value is -6.246e-04 from the closed form and stays there.
    """
    point = POINTS[0]
    values = [_cos(point, "call", "vanilla", n_terms=n, truncation_l=4.0) for n in (32, 64, 128, 256, 512, 1024)]
    assert all(math.isclose(v, values[0], rel_tol=1e-12) for v in values)
    assert abs(values[0] - _exact(point, "call", "vanilla")) > 1e-5


def test_the_digital_needs_far_less_range_than_the_vanilla() -> None:
    """The indicator payoff has no `e^z` weight, so the far tail matters less.

    At L = 4 and N = 256 the vanilla call is 6.25e-04 from the closed form and
    the digital is already at the floating-point floor. The call's payoff
    coefficient is `S_0 chi_k - K psi_k`, and `chi_k` integrates `e^z` over the
    range, so its value is dominated by the top of the range and the mass left
    outside is weighted by `e^b`. The digital's is `cash * psi_k`, bounded by the
    range width. Evidence class: CLOSED_FORM, and the comparison is the point.
    """
    point = POINTS[0]
    vanilla = _cos_error(point, "call", "vanilla", n_terms=256, truncation_l=4.0)
    digital = _cos_error(point, "call", "digital", n_terms=256, truncation_l=4.0)
    assert vanilla > 1e-5
    assert digital < 1e-13


# --------------------------------------------------------------------------
# (d) Identities.
# --------------------------------------------------------------------------

PARITY_TOLERANCE = 5e-12
"""Worst measured residual 2.94e-12, at `long_high_vol`."""


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
def test_cos_put_call_parity(point: Point) -> None:
    """Evidence class: EXACT_IDENTITY, and here it is a real check.

    Worth one sentence because it is not true of the other three methods in
    this package: Carr-Madan, Lewis and Gil-Pelaez all transform the *call*, so
    their put is the call plus a forward and parity holds by construction. The
    COS put is built from its own payoff coefficients (`K psi_k(a, z*) - S_0
    chi_k(a, z*)`, a different pair of integrals over a different sub-interval),
    so a sign or an endpoint error in either branch shows up here.
    """
    call = _cos(point, "call", "vanilla")
    put = _cos(point, "put", "vanilla")
    forward = point.spot * math.exp(-point.dividend * point.expiry)
    strike_leg = point.strike * math.exp(-point.rate * point.expiry)
    assert abs((call - put) - (forward - strike_leg)) <= PARITY_TOLERANCE


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
def test_cos_digital_static_replication(point: Point) -> None:
    """A digital call plus a digital put is a zero-coupon bond. EXACT_IDENTITY.

    Under the COS sum this is the statement that the expansion integrates the
    density to one: the two payoff coefficients are `psi_k(z*, b)` and
    `psi_k(a, z*)`, which add to `psi_k(a, b)` whatever `z*` is. Worst measured
    residual 1.11e-16.
    """
    total = _cos(point, "call", "digital") + _cos(point, "put", "digital")
    assert abs(total - math.exp(-point.rate * point.expiry)) <= 1e-15


@pytest.mark.parametrize("point", POINTS[:3], ids=[p.name for p in POINTS[:3]])
def test_the_jump_costs_the_cosine_expansion_nothing(point: Point) -> None:
    """NEGATIVE_FINDING against the expectation a discontinuity is expensive.

    Slice 6 measured that a cash-or-nothing payoff costs the finite-difference
    scheme a **full order** of convergence, because the grid represents the
    payoff. The COS method represents the *density*, which is smooth whatever
    the payoff is, and the payoff enters only through an exact integral. So the
    digital does not converge more slowly -- it converges **faster**, by a
    factor of 26 to 54 in the error at the same term count, for the reason the
    range test above gives (no `e^z` weight in its coefficients).

    Measured at `atm_1y`: N = 16, vanilla 5.38e-02 against digital 9.95e-04;
    N = 24, 5.41e-04 against 1.54e-05; N = 32, 1.37e-06 against 5.21e-08.
    """
    for n in (16, 24, 32):
        vanilla = _cos_error(point, "call", "vanilla", n_terms=n)
        digital = _cos_error(point, "call", "digital", n_terms=n)
        assert digital < vanilla, (n, vanilla, digital)


# --------------------------------------------------------------------------
# (e) Greeks.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("point", POINTS, ids=[p.name for p in POINTS])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("payoff", PAYOFFS)
def test_cos_greeks_against_the_closed_forms(point: Point, kind: str, payoff: str) -> None:
    """Evidence class: CLOSED_FORM.

    Delta and gamma are exact derivatives of the truncated sum in `S_0` -- the
    truncation range does not move with the spot, so differentiating the price
    means differentiating `V_k` and nothing else -- and they land three to four
    decimal orders of magnitude inside the `~1e-10` the slice statement asked
    for. vega, theta and rho are central differences of the transform price and
    land at `~1e-08`, which is the bump's own `h^2` bias.
    """
    instrument = point.instrument(kind, payoff)
    model, market = point.model(), point.market()
    exact = greeks(instrument, model, market)
    cos = greeks(instrument, model, market, method="fourier", cfg=FourierConfig())

    assert cos.delta == pytest.approx(exact.delta, abs=DELTA_TOLERANCE)
    assert cos.gamma == pytest.approx(exact.gamma, abs=GAMMA_TOLERANCE)
    assert cos.vega == pytest.approx(exact.vega, abs=BUMPED_TOLERANCE)
    assert cos.theta == pytest.approx(exact.theta, abs=BUMPED_TOLERANCE)
    assert cos.rho == pytest.approx(exact.rho, abs=BUMPED_TOLERANCE)

    assert cos.meta is not None
    assert cos.meta["greeks"]["delta"] == "closed_form"
    assert cos.meta["greeks"]["other"] == "central_difference"


def test_cos_delta_and_gamma_do_not_improve_with_more_terms_past_the_floor() -> None:
    """They are the same sum as the price, so they hit the same floor at N = 48."""
    point = POINTS[0]
    instrument = point.instrument("call", "vanilla")
    model, market = point.model(), point.market()
    values = [
        greeks(instrument, model, market, method="fourier", cfg=FourierConfig(n_terms=n))
        for n in (64, 128, 256, 512)
    ]
    assert all(g.delta == values[0].delta for g in values)
    assert all(g.gamma == values[0].gamma for g in values)


# --------------------------------------------------------------------------
# The characteristic-function interface itself.
# --------------------------------------------------------------------------


def test_black_scholes_characteristic_function_is_the_gaussian_transform() -> None:
    """CLOSED_FORM: `phi(u) = exp(i u c1 - c2 u^2 / 2)` with the stated cumulants."""
    expiry, rate, dividend, sigma = 1.5, 0.04, 0.01, 0.3
    cumulants = black_scholes_log_return_cumulants(
        expiry, rate=rate, dividend=dividend, sigma=sigma
    )
    assert math.isclose(
        cumulants.c1, (rate - dividend - 0.5 * sigma**2) * expiry, rel_tol=1e-12
    )
    assert math.isclose(cumulants.c2, sigma**2 * expiry, rel_tol=1e-12)
    assert cumulants.c4 == 0.0

    u = np.array([0.0, 0.5, 1.0, 3.0])
    phi = black_scholes_characteristic_function(
        u, expiry, rate=rate, dividend=dividend, sigma=sigma
    )
    expected = np.exp(1j * u * cumulants.c1 - 0.5 * cumulants.c2 * u**2)
    assert np.allclose(phi, expected, rtol=0.0, atol=1e-15)
    # phi(0) = 1 exactly: the law is a probability measure.
    assert phi[0] == 1.0 + 0.0j


def test_characteristic_function_accepts_complex_arguments() -> None:
    """The other three methods need it off the real line; Heston will too.

    `phi(u - i)` is the transform under the share measure (Gil-Pelaez's `P1`)
    and must equal `E[S_T / S_0 * e^{i u ln(S_T/S_0)}]`. At `u = 0` that is
    `E[S_T]/S_0 = e^{(r - q) T}`.
    """
    expiry, rate, dividend, sigma = 0.75, 0.05, 0.02, 0.25
    value = black_scholes_characteristic_function(
        np.array([-1j]), expiry, rate=rate, dividend=dividend, sigma=sigma
    )[0]
    assert math.isclose(value.real, math.exp((rate - dividend) * expiry), rel_tol=1e-12)
    assert abs(value.imag) < 1e-15


def test_truncation_range_comes_from_the_cumulants() -> None:
    point = POINTS[0]
    cf = characteristic_function_model(point.model())
    lower, upper = cos_truncation_range(
        cf, point.expiry, rate=point.rate, dividend=point.dividend, truncation_l=10.0
    )
    cumulants = cf.log_return_cumulants(
        point.expiry, rate=point.rate, dividend=point.dividend
    )
    width = 10.0 * cumulants.truncation_width()
    assert math.isclose(lower, cumulants.c1 - width, rel_tol=1e-12)
    assert math.isclose(upper, cumulants.c1 + width, rel_tol=1e-12)


def test_truncation_width_uses_the_fourth_cumulant() -> None:
    """`sqrt(c2 + sqrt(c4))`. Zero for Black-Scholes, the slot Heston fills."""
    assert math.isclose(
        LogReturnCumulants(c1=0.0, c2=0.04, c4=0.0016).truncation_width(),
        math.sqrt(0.04 + 0.04),
        rel_tol=1e-12,
    )


def test_a_model_without_a_transform_is_refused() -> None:
    with pytest.raises(NotSupportedError, match="provides no characteristic function"):
        characteristic_function_model(object())


# --------------------------------------------------------------------------
# Validation.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "nonsense"}, "method must be one of"),
        ({"n_terms": 1}, "n_terms must be an integer >= 2"),
        ({"n_terms": 64.0}, "n_terms must be an integer >= 2"),
        ({"truncation_l": 0.0}, "truncation_l must be finite and > 0"),
        ({"truncation_l": float("inf")}, "truncation_l must be finite and > 0"),
    ],
)
def test_config_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        FourierConfig(**kwargs)


def test_zero_expiry_is_refused_rather_than_returned_as_intrinsic() -> None:
    """Deliberately different from every other engine (see the module docstring
    of `qpl.engines.fourier.pricers`): at T = 0 the terminal law is a point mass
    and there is no density to expand."""
    point = POINTS[0]
    option = EuropeanOption(kind="call", strike=100.0, expiry=0.0)
    with pytest.raises(InvalidInputError, match="Fourier methods require T > 0"):
        price(option, point.model(), point.market(), method="fourier", cfg=FourierConfig())


def test_zero_volatility_is_refused_with_a_message_that_says_why() -> None:
    point = POINTS[0]
    with pytest.raises(InvalidInputError, match="zero variance"):
        price(
            point.instrument("call", "vanilla"),
            BlackScholesModel(sigma=0.0),
            point.market(),
            method="fourier",
            cfg=FourierConfig(),
        )


def test_theta_refuses_a_maturity_shorter_than_its_own_bump() -> None:
    point = POINTS[0]
    option = EuropeanOption(kind="call", strike=100.0, expiry=1e-6)
    with pytest.raises(InvalidInputError, match="theta needs T >"):
        greeks(option, point.model(), point.market(), method="fourier", cfg=FourierConfig())
