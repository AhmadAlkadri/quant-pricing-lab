from __future__ import annotations

import math

from ...exceptions import InvalidInputError
from ...instruments.options import EuropeanOption
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel, bs_price
from ..base import GreeksResult, PriceResult



def _norm_cdf(x: float) -> float:
    # TODO(CDF-DUP): unify with qpl.models.black_scholes._norm_cdf (vectorized).
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))



def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)



def _d1_d2(
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> tuple[float, float, float]:
    if T <= 0:
        raise InvalidInputError("T must be > 0 for Greeks")
    if sigma <= 0:
        raise InvalidInputError("sigma must be > 0 for Greeks")
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2, sqrtT



def price_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
) -> PriceResult:
    T = option.expiry
    r = market.rate(T)
    q = market.dividend_yield(T)
    value = float(
        bs_price(
            S=market.spot,
            K=option.strike,
            T=T,
            r=r,
            sigma=model.sigma,
            q=q,
            kind=option.kind,
        )
    )
    return PriceResult(value=value, meta={"method": "analytic", "model": "BlackScholes"})



def greeks_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
) -> GreeksResult:
    S = market.spot
    K = option.strike
    T = option.expiry
    r = market.rate(T)
    q = market.dividend_yield(T)
    sigma = model.sigma

    d1, d2, sqrtT = _d1_d2(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    pdf_d1 = _norm_pdf(d1)

    gamma = df_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * df_q * pdf_d1 * sqrtT

    if option.kind == "call":
        delta = df_q * nd1
        theta = (
            -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrtT)
            - r * K * df_r * nd2
            + q * S * df_q * nd1
        )
        rho = K * T * df_r * nd2
    else:
        nmd1 = _norm_cdf(-d1)
        nmd2 = _norm_cdf(-d2)
        delta = df_q * (nd1 - 1.0)
        theta = (
            -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrtT)
            + r * K * df_r * nmd2
            - q * S * df_q * nmd1
        )
        rho = -K * T * df_r * nmd2

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        meta={"method": "analytic", "model": "BlackScholes"},
    )


def implied_volatility(
    price: float,
    option: EuropeanOption,
    market: Market,
    *,
    lower: float = 1e-6,
    upper: float = 5.0,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> float:
    """Compute implied volatility via Brent's method.

    Raises:
        InvalidInputError: if price is outside arbitrage bounds or inputs invalid.
        RuntimeError: if solver fails to converge.
    """
    from scipy.optimize import brentq

    if price < 0:
        raise InvalidInputError("Option price must be non-negative")

    T = option.expiry
    # If T=0, IV is undefined (or requires intrinsic match).
    if T <= 0.0:
        raise InvalidInputError("Cannot compute implied vol for T<=0")

    S = market.spot
    K = option.strike
    r = market.rate(T)
    q = market.dividend_yield(T)

    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)

    # Arbitrage bounds
    if option.kind == "call":
        # Call value C >= max(0, S*df_q - K*df_r) and C < S*df_q
        intrinsic = max(0.0, S * df_q - K * df_r)
        max_val = S * df_q
        if not (intrinsic <= price < max_val):
             raise InvalidInputError(
                 f"Call price {price} outside bounds [{intrinsic:.4f}, {max_val:.4f})"
             )
    else:
        # Put value P >= max(0, K*df_r - S*df_q) and P < K*df_r
        intrinsic = max(0.0, K * df_r - S * df_q)
        max_val = K * df_r
        if not (intrinsic <= price < max_val):
             raise InvalidInputError(
                 f"Put price {price} outside bounds [{intrinsic:.4f}, {max_val:.4f})"
             )

    # If price is extremely close to intrinsic, sigma -> 0.
    if abs(price - intrinsic) < 1e-9:
        return 0.0

    def objective(sigma: float) -> float:
        val = float(
            bs_price(
                S=S,
                K=K,
                T=T,
                r=r,
                sigma=sigma,
                q=q,
                kind=option.kind,
            )
        )
        return val - price

    # Check bounds
    y_low = objective(lower)
    y_high = objective(upper)

    if y_low * y_high > 0:
        # If both same sign, we can't bracket.
        # If y_low > 0, model price at min vol is already > target price (shouldn't happen if bound checked?)
        # If y_high < 0, model price at max vol is < target price (vol > upper)
        msg = f"Cannot bracket implied vol in [{lower}, {upper}]. Errs: low={y_low:.2e}, high={y_high:.2e}"
        raise InvalidInputError(msg)

    try:
        iv = brentq(objective, lower, upper, xtol=tol, maxiter=max_iter)
        return float(iv)
    except Exception as e:
        raise RuntimeError(f"Implied vol solver failed: {e}")
