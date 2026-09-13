"""Closed-form cash-or-nothing digital options under Black-Scholes.

Derivation
----------
A cash-or-nothing digital call pays `cash` at `T` if `S_T > K`. Its value is
the discounted risk-neutral expectation of that payoff, and the expectation of
an indicator is a probability:

    V_call = e^{-rT} E^Q[ cash * 1{S_T > K} ] = cash * e^{-rT} * Q(S_T > K).

Under the risk-neutral measure with continuous dividend yield `q`,

    S_T = S exp( (r - q - sigma^2 / 2) T + sigma sqrt(T) Z ),   Z ~ N(0, 1),

so `S_T > K` is `Z > -d2` with

    d2 = ( log(S / K) + (r - q - sigma^2 / 2) T ) / ( sigma sqrt(T) ),

and by the symmetry of the normal distribution `Q(S_T > K) = N(d2)`. Hence

    V_call = cash * e^{-rT} * N(d2),      V_put = cash * e^{-rT} * N(-d2),

the put following from `Q(S_T < K) = 1 - Q(S_T > K)` (the boundary event
`S_T = K` has probability zero). Adding the two gives the static-replication
identity this module is checked against: a digital call plus a digital put is
a certain payment of `cash` at `T`, worth `cash * e^{-rT}` today, whatever
`sigma`, `S` and `K` are.

Source for the closed forms: Reiner and Rubinstein (1991), "Unscrambling the
binary code", Risk 4(9), 75-83, who give the four binary payoffs (cash- and
asset-or-nothing, calls and puts) in this form. The derivation above is
written out independently here; nothing is reproduced from that article. Hull,
*Options, Futures, and Other Derivatives*, treats the same contract in its
exotics chapter and is used here only as a concept map.

The second identity worth stating, because it is what ties this module to the
vanilla engine: the digital call is minus the strike-derivative of the vanilla
call. Differentiating `C = S e^{-qT} N(d1) - K e^{-rT} N(d2)` in `K` and using
the standard identity `S e^{-qT} phi(d1) = K e^{-rT} phi(d2)`, the two terms
in `dd1/dK` and `dd2/dK` cancel and

    -dC/dK = e^{-rT} N(d2),

which is `V_call / cash`. That is not a coincidence: a digital call is the
limit of a short call spread `(C(K) - C(K + eps)) / eps` as `eps -> 0`, and
the spread pays one unit exactly when the spot ends above the strike.

Greeks
------
Write `A = cash * e^{-rT}` and `phi` for the standard normal density. Only
`d2` carries the spot, volatility and rate dependence, so every Greek is `A`
times a derivative of `N(+-d2)`. With `dd2/dS = 1 / (S sigma sqrt(T))`:

    delta_call = A phi(d2) / (S sigma sqrt(T))
    gamma_call = -A phi(d2) d1 / (S^2 sigma^2 T)          [since d1 = d2 + sigma sqrt(T)]
    vega_call  = -A phi(d2) d1 / sigma                     [since dd2/dsigma = -d1 / sigma]
    rho_call   = -T * V_call + A phi(d2) sqrt(T) / sigma
    theta_call = r * A * N(d2) - A phi(d2) * dd2/dT
    dd2/dT     = ( -log(S / K) / T + (r - q - sigma^2 / 2) ) / ( 2 sigma sqrt(T) )

and the put's are obtained by `N(-d2)` in place of `N(d2)` and a sign flip on
every term that came from differentiating the indicator (delta, gamma and vega
change sign; `rho` and `theta` keep their `-T V` and `r A N(.)` terms). All
ten expressions are checked against central finite differences of the price in
`tests/test_digital_analytic.py`.

Two of these are worth a sentence because they are what makes a digital hard
numerically:

- `gamma` **changes sign at the strike**, since it carries the factor `-d1`
  and `d1` passes through zero near the money. A digital is not convex; it is
  convex on one side of the strike and concave on the other.
- `delta` is largest at the money and collapses as `T -> 0`, where it behaves
  like `1 / sqrt(T)` -- the price is converging to a step function and its
  slope to a Dirac mass.
"""

from __future__ import annotations

import math

from ...exceptions import InvalidInputError
from ...instruments.options import DigitalOption
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult
from .black_scholes import _d1_d2, _norm_cdf, _norm_pdf

__all__ = ["digital_price", "greeks_digital", "price_digital"]


def digital_price(
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    cash: float = 1.0,
    kind: str = "call",
) -> float:
    """Closed-form cash-or-nothing digital value (module docstring for the derivation).

    Handles the two degenerate limits in the same way the rest of the package
    does: at `T = 0` the price is the payoff, and at `sigma = 0` the terminal
    spot is its forward, so the price is `cash * e^{-rT}` exactly when that
    forward is in the money.
    """
    if T < 0:
        raise InvalidInputError("T must be >= 0")
    if sigma < 0:
        raise InvalidInputError("sigma must be >= 0")
    if K <= 0:
        raise InvalidInputError("K must be > 0")
    if S <= 0:
        raise InvalidInputError("S must be > 0")
    kind_l = kind.lower()
    if kind_l not in {"call", "put"}:
        raise InvalidInputError("kind must be 'call' or 'put'")

    if T == 0.0:
        in_money = S > K if kind_l == "call" else S < K
        return cash if in_money else 0.0

    disc = math.exp(-r * T)
    if sigma == 0.0:
        forward = S * math.exp((r - q) * T)
        in_money = forward > K if kind_l == "call" else forward < K
        return disc * cash if in_money else 0.0

    sqrtT = math.sqrt(T)
    d2 = (math.log(S / K) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return cash * disc * _norm_cdf(d2 if kind_l == "call" else -d2)


def price_digital(
    option: DigitalOption,
    model: BlackScholesModel,
    market: Market,
) -> PriceResult:
    """Price a cash-or-nothing digital in closed form."""
    t = option.expiry
    return PriceResult(
        value=digital_price(
            S=market.spot,
            K=option.strike,
            T=t,
            r=market.rate(t),
            sigma=model.sigma,
            q=market.dividend_yield(t),
            cash=option.cash,
            kind=option.kind,
        ),
        meta={"method": "analytic", "model": "BlackScholes", "instrument": "digital"},
    )


def greeks_digital(
    option: DigitalOption,
    model: BlackScholesModel,
    market: Market,
) -> GreeksResult:
    """Closed-form digital Greeks (module docstring for the derivations).

    Raises
    ------
    InvalidInputError
        If `T = 0` or `sigma = 0`. Both limits make the price a step function
        of the spot, so delta is a Dirac mass and gamma its derivative; the
        vanilla engine refuses the same two limits for the same reason
        (one derivative lower).
    """
    s = market.spot
    k = option.strike
    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    sigma = model.sigma

    d1, d2, sqrtT = _d1_d2(S=s, K=k, T=t, r=r, q=q, sigma=sigma)
    amount = option.cash * math.exp(-r * t)
    pdf_d2 = _norm_pdf(d2)
    sign = 1.0 if option.kind == "call" else -1.0
    cdf = _norm_cdf(sign * d2)

    value = amount * cdf
    delta = sign * amount * pdf_d2 / (s * sigma * sqrtT)
    gamma = -sign * amount * pdf_d2 * d1 / (s * s * sigma * sigma * t)
    vega = -sign * amount * pdf_d2 * d1 / sigma
    rho = -t * value + sign * amount * pdf_d2 * sqrtT / sigma
    dd2_dt = (-math.log(s / k) / t + (r - q - 0.5 * sigma * sigma)) / (2.0 * sigma * sqrtT)
    theta = r * value - sign * amount * pdf_d2 * dd2_dt

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        meta={"method": "analytic", "model": "BlackScholes", "instrument": "digital"},
    )
