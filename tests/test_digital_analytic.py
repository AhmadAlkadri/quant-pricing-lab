"""The closed-form cash-or-nothing digital, and the four things that pin it.

Every assertion here names the `EvidenceClass` that justifies it:

1. **EXACT_IDENTITY** -- a digital call plus a digital put is a certain payment
   of `cash` at `T`, so their prices must sum to `cash * e^{-rT}` to round-off,
   for any spot, strike, volatility, rate and dividend yield. This holds by
   static replication and does not depend on the formula being right.
2. **CLOSED_FORM** -- the digital call equals `-dC/dK` of the vanilla call (and
   the digital put equals `+dP/dK`), checked by a central difference in the
   strike against this package's own vanilla engine. Two engines, one identity.
3. **INDEPENDENT_ENGINE** -- the price equals `cash * e^{-rT}` times a
   risk-neutral probability computed by numerically integrating the lognormal
   density with `scipy.integrate.quad`, which shares no code with the `erf`
   evaluation the closed form uses.
4. **CLOSED_FORM** -- all five analytic Greeks against central finite
   differences of the analytic price, with tolerances derived from the measured
   residuals below rather than chosen.

Source for the closed forms: Reiner and Rubinstein (1991), "Unscrambling the
binary code", Risk 4(9), 75-83. The derivation is written out in
`qpl.engines.analytic.digital`; no expression, number or table is reproduced
from that article.
"""

from __future__ import annotations

import math

import pytest
from scipy.integrate import quad

from qpl.engines.analytic.digital import digital_price
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments.options import DigitalOption
from qpl.instruments.payoffs import digital_payoff
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel, bs_price
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass

# (id, spot, strike, expiry, rate, dividend, sigma). Five points spanning the
# money, two maturities either side of a year, and both signs of `r - q`.
_POINTS = (
    ("atm_1y", 100.0, 100.0, 1.00, 0.05, 0.00, 0.20),
    ("otm_9m_div", 100.0, 110.0, 0.75, 0.03, 0.01, 0.25),
    ("itm_1y_div", 120.0, 90.0, 1.00, 0.03, 0.05, 0.35),
    ("short_atm", 100.0, 100.0, 0.05, 0.01, 0.00, 0.30),
    ("deep_otm_2y", 80.0, 120.0, 2.00, 0.02, 0.03, 0.40),
)
_ARGNAMES = ("spot", "strike", "expiry", "rate", "div", "sigma")
_ARGS = [row[1:] for row in _POINTS]
_IDS = [row[0] for row in _POINTS]

_STRIKE_BUMP = 1e-2
"""Central-difference step for the `-dC/dK` identity, in strike units.

Derived, not chosen. The residual against the closed form over the five points
and both kinds is 1.375e-06 at `h = 1e-1`, **1.375e-08** at `h = 1e-2`,
1.424e-10 at `h = 1e-3` and 1.367e-10 at `h = 1e-4` -- a clean `O(h**2)` decay
that flattens onto a round-off floor near 1.4e-10 by `h = 1e-3`. `h = 1e-2` is
taken because it is the largest step still two orders of magnitude clear of
that floor, so the test measures the identity rather than the conditioning of
the difference.
"""

_STRIKE_IDENTITY_TOLERANCE = 5.0e-8
"""Worst measured residual at `_STRIKE_BUMP` is 1.375e-08; this keeps 3.6x."""

_CASH = 2.5
"""A `cash` that is neither 1 nor a power of two, so a dropped factor shows."""


def _market(spot: float, rate: float, div: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(rate),
        dividend_curve=FlatDividendCurve(div),
    )


def _triple(kind, spot, strike, expiry, rate, div, sigma, cash=1.0):
    return (
        DigitalOption(kind=kind, strike=strike, expiry=expiry, cash=cash),
        BlackScholesModel(sigma=sigma),
        _market(spot, rate, div),
    )


# --------------------------------------------------------------------------
# (1) The static-replication identity.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_digital_call_plus_put_is_a_riskless_payment(
    spot: float, strike: float, expiry: float, rate: float, div: float, sigma: float
) -> None:
    """Evidence class: EXACT_IDENTITY.

    `1{S_T > K} + 1{S_T < K} = 1` almost surely, so the two digitals together
    are a zero-coupon bond paying `cash`. The tolerance is a floating-point
    round-off budget, not a model-error budget: no discretisation is involved
    and the identity is independent of `sigma`.
    """
    assert EvidenceClass.EXACT_IDENTITY.value == "exact_identity"

    call = price(*_triple("call", spot, strike, expiry, rate, div, sigma, _CASH)).value
    put = price(*_triple("put", spot, strike, expiry, rate, div, sigma, _CASH)).value
    riskless = _CASH * math.exp(-rate * expiry)

    assert call + put == pytest.approx(riskless, rel=0.0, abs=1e-15 * _CASH)
    # Not vacuous: both legs must actually be worth something at these points.
    assert 0.0 < call < riskless
    assert 0.0 < put < riskless


# --------------------------------------------------------------------------
# (2) The digital is the strike-derivative of the vanilla.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_digital_is_the_strike_derivative_of_the_vanilla(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: CLOSED_FORM.

    `C(K) = S e^{-qT} N(d1) - K e^{-rT} N(d2)` gives `-dC/dK = e^{-rT} N(d2)`
    once the `phi(d1)`/`phi(d2)` terms cancel, which is the digital call per
    unit of cash; the put identity is the same with the opposite sign, because
    `P(K)` is increasing in `K`. Mechanically this is the limit of a short call
    spread, so it is a statement about replication rather than about algebra.

    The reference is this repository's *vanilla* engine, evaluated by a central
    difference with `_STRIKE_BUMP`, so the check ties two engines together
    rather than restating one of them.
    """
    digital = price(*_triple(kind, spot, strike, expiry, rate, div, sigma)).value

    h = _STRIKE_BUMP
    up = bs_price(S=spot, K=strike + h, T=expiry, r=rate, sigma=sigma, q=div, kind=kind)
    dn = bs_price(S=spot, K=strike - h, T=expiry, r=rate, sigma=sigma, q=div, kind=kind)
    slope = (up - dn) / (2.0 * h)
    expected = -slope if kind == "call" else slope

    assert digital == pytest.approx(expected, abs=_STRIKE_IDENTITY_TOLERANCE)


# --------------------------------------------------------------------------
# (3) The price as a probability, computed a second way.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_price_is_the_discounted_risk_neutral_probability(
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: INDEPENDENT_ENGINE.

    The closed form reduces to `cash * e^{-rT} Q(S_T > K)`, and `Q` is an
    integral of the lognormal density. Integrating that density numerically
    with `scipy.integrate.quad` reaches the same number through adaptive
    Gauss-Kronrod quadrature rather than through `erf`, so an error in the
    `d2` algebra could not hide in both.

    The tolerance is `quad`'s own reported absolute error times ten, which is
    what makes this an independent measurement rather than a second pin.
    """
    mu = math.log(spot) + (rate - div - 0.5 * sigma * sigma) * expiry
    s = sigma * math.sqrt(expiry)

    def density(x: float) -> float:
        return math.exp(-0.5 * ((math.log(x) - mu) / s) ** 2) / (
            x * s * math.sqrt(2.0 * math.pi)
        )

    if kind == "call":
        probability, abserr = quad(density, strike, math.inf, limit=200)
    else:
        probability, abserr = quad(density, 0.0, strike, limit=200)

    expected = _CASH * math.exp(-rate * expiry) * probability
    actual = price(*_triple(kind, spot, strike, expiry, rate, div, sigma, _CASH)).value
    assert actual == pytest.approx(expected, abs=max(10.0 * abserr, 1e-13))


# --------------------------------------------------------------------------
# (4) The Greeks, against central differences of the price.
# --------------------------------------------------------------------------

# (greek, relative tolerance, worst measured relative residual over the five
# points and both kinds). The finite differences below carry their own
# `O(h**2)` truncation and a round-off term, so these tolerances bound the
# *difference scheme's* error, not the closed form's: every one keeps a factor
# of at least 25 over what was measured.
_GREEK_TOLERANCES = {
    "delta": (1.0e-5, 3.687e-07),
    "gamma": (5.0e-5, 1.448e-06),
    "vega": (1.0e-7, 1.969e-09),
    "rho": (1.0e-8, 1.226e-10),
    "theta": (1.0e-6, 3.132e-08),
}


@pytest.mark.parametrize("greek", sorted(_GREEK_TOLERANCES))
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(_ARGNAMES, _ARGS, ids=_IDS)
def test_analytic_greeks_match_central_differences(
    greek: str,
    kind: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    div: float,
    sigma: float,
) -> None:
    """Evidence class: CLOSED_FORM.

    Each analytic Greek in `qpl.engines.analytic.digital` is differentiated by
    hand in that module's docstring; here each is differenced numerically from
    the price the same module returns. That checks the differentiation, not the
    price -- the price itself is pinned by the three tests above.

    `theta` is `dV/dt`, so the difference is taken in `-T`, matching the sign
    convention the vanilla analytic engine uses.
    """
    analytic = getattr(
        greeks(*_triple(kind, spot, strike, expiry, rate, div, sigma, _CASH)), greek
    )

    def value(**overrides: float) -> float:
        args = {
            "S": spot,
            "K": strike,
            "T": expiry,
            "r": rate,
            "sigma": sigma,
            "q": div,
            "cash": _CASH,
            "kind": kind,
        }
        args.update(overrides)
        return digital_price(**args)  # type: ignore[arg-type]

    if greek == "delta":
        h = 1e-4 * spot
        numeric = (value(S=spot + h) - value(S=spot - h)) / (2.0 * h)
    elif greek == "gamma":
        h = 1e-4 * spot
        numeric = (value(S=spot + h) - 2.0 * value() + value(S=spot - h)) / (h * h)
    elif greek == "vega":
        h = 1e-5
        numeric = (value(sigma=sigma + h) - value(sigma=sigma - h)) / (2.0 * h)
    elif greek == "rho":
        h = 1e-6
        numeric = (value(r=rate + h) - value(r=rate - h)) / (2.0 * h)
    else:
        h = 1e-6
        numeric = -(value(T=expiry + h) - value(T=expiry - h)) / (2.0 * h)

    tolerance, measured = _GREEK_TOLERANCES[greek]
    assert abs(analytic - numeric) <= tolerance * max(abs(numeric), 1e-12), (
        greek,
        analytic,
        numeric,
    )
    # The tolerance is derived from a measurement; keep that measurement honest.
    assert measured < tolerance


def test_gamma_changes_sign_across_the_strike() -> None:
    """Evidence class: CLOSED_FORM -- and the reason a digital is hard.

    `gamma = -A phi(d2) d1 / (S**2 sigma**2 T)` carries the factor `-d1`, so it
    is positive below the point where `d1 = 0` and negative above it. A digital
    call is convex on one side of the strike and concave on the other, which is
    exactly why the PDE and tree studies in this slice report gamma separately
    and do not expect it to behave like a vanilla's.
    """
    below = greeks(*_triple("call", 80.0, 100.0, 1.0, 0.05, 0.0, 0.20)).gamma
    above = greeks(*_triple("call", 130.0, 100.0, 1.0, 0.05, 0.0, 0.20)).gamma

    assert below > 0.0
    assert above < 0.0
    # And the zero sits where d1 does, i.e. at the forward-adjusted point.
    zero_spot = 100.0 * math.exp(-(0.05 + 0.5 * 0.04) * 1.0)
    near = greeks(*_triple("call", zero_spot, 100.0, 1.0, 0.05, 0.0, 0.20)).gamma
    assert abs(near) < 1e-15


# --------------------------------------------------------------------------
# Degenerate limits, scaling, and validation.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("spot", [95.0, 105.0])
def test_expiry_zero_is_the_payoff(kind: str, spot: float) -> None:
    """Evidence class: CLOSED_FORM. At `T = 0` the price is the payoff itself."""
    value = price(*_triple(kind, spot, 100.0, 0.0, 0.05, 0.01, 0.20, _CASH)).value
    assert value == digital_payoff(spot, 100.0, _CASH, kind)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_zero_volatility_is_the_discounted_forward_indicator(kind: str) -> None:
    """Evidence class: CLOSED_FORM.

    With `sigma = 0` the terminal spot is its forward `S e^{(r-q)T}` with
    certainty, so the digital is a zero-coupon bond if that forward is in the
    money and worth nothing otherwise. The point below has forward
    `100 e^{0.03} = 103.05`, which straddles the two strikes used.
    """
    for strike, expected_in_money in ((100.0, kind == "call"), (110.0, kind == "put")):
        value = price(*_triple(kind, 100.0, strike, 1.0, 0.05, 0.02, 0.0, _CASH)).value
        expected = _CASH * math.exp(-0.05) if expected_in_money else 0.0
        assert value == pytest.approx(expected, abs=1e-15)


def test_price_and_greeks_are_linear_in_cash() -> None:
    """Evidence class: EXACT_IDENTITY. `cash` is a pure scaling of the payoff."""
    one = _triple("call", 100.0, 100.0, 1.0, 0.05, 0.01, 0.2, 1.0)
    seven = _triple("call", 100.0, 100.0, 1.0, 0.05, 0.01, 0.2, 7.0)

    assert price(*seven).value == pytest.approx(7.0 * price(*one).value, rel=1e-15)
    g1, g7 = greeks(*one), greeks(*seven)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert getattr(g7, name) == pytest.approx(7.0 * getattr(g1, name), rel=1e-14)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "straddle"}, "kind must be"),
        ({"strike": 0.0}, "strike must be > 0"),
        ({"expiry": -1.0}, "expiry must be >= 0"),
        ({"cash": 0.0}, "cash must be finite and > 0"),
        ({"cash": -1.0}, "cash must be finite and > 0"),
        ({"cash": math.inf}, "cash must be finite and > 0"),
    ],
)
def test_construction_rejects_bad_inputs(kwargs: dict, message: str) -> None:
    base = {"kind": "call", "strike": 100.0, "expiry": 1.0, "cash": 1.0}
    base.update(kwargs)
    with pytest.raises(InvalidInputError, match=message):
        DigitalOption(**base)  # type: ignore[arg-type]


def test_kind_is_normalised_and_the_instrument_is_frozen() -> None:
    option = DigitalOption(kind="PUT", strike=100.0, expiry=1.0)
    assert option.kind == "put"
    assert option.cash == 1.0
    with pytest.raises(Exception):  # noqa: B017 -- frozen dataclass, any raise will do
        option.strike = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("sigma_and_expiry", [(0.0, 1.0), (0.2, 0.0)])
def test_greeks_refuse_the_degenerate_limits(sigma_and_expiry: tuple) -> None:
    """Evidence class: CLOSED_FORM.

    At `sigma = 0` or `T = 0` the price is a step function of the spot: delta
    is a Dirac mass and gamma its derivative. The vanilla engine refuses the
    same two limits one derivative lower, and the price is still returned.
    """
    sigma, expiry = sigma_and_expiry
    triple = _triple("call", 100.0, 100.0, expiry, 0.05, 0.0, sigma)
    with pytest.raises(InvalidInputError):
        greeks(*triple)
    assert price(*triple).value >= 0.0


def test_a_digital_is_not_a_vanilla_to_the_registry() -> None:
    """`DigitalOption` is outside the `VanillaOption` hierarchy on purpose.

    If it were a subclass, registry lookup -- which walks the MRO -- would
    resolve it to the European engines and price a step function as a hockey
    stick. The check below is on the type graph, and the `NotSupportedError`
    is what the lookup does with an unregistered model type.
    """
    from qpl.instruments.options import AmericanOption, EuropeanOption, VanillaOption

    assert not issubclass(DigitalOption, VanillaOption)
    assert not issubclass(DigitalOption, (EuropeanOption, AmericanOption))

    class _NotAModel:
        pass

    with pytest.raises(NotSupportedError):
        price(
            DigitalOption(kind="call", strike=100.0, expiry=1.0),
            _NotAModel(),
            _market(100.0, 0.05, 0.0),
        )
