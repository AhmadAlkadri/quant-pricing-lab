from __future__ import annotations

from typing import Any, Literal

from .engines.analytic.black_scholes import greeks_european, price_european
from .engines.base import GreeksResult, PriceResult
from .exceptions import InvalidInputError, NotSupportedError
from .instruments.options import EuropeanOption
from .market.market import Market
from .models.black_scholes import BlackScholesModel



def price(
    instrument: Any,
    model: Any,
    market: Any,
    *,
    method: Literal["analytic", "mc", "pde"] = "analytic",
    **kwargs: Any,
) -> PriceResult:
    if method != "analytic":
        raise NotSupportedError(f"method '{method}' is not supported")
    if kwargs:
        raise InvalidInputError("Unexpected keyword arguments for method 'analytic'")

    if (
        isinstance(instrument, EuropeanOption)
        and isinstance(model, BlackScholesModel)
        and isinstance(market, Market)
    ):
        return price_european(instrument, model, market)

    raise NotSupportedError("Unsupported instrument/model/market combination")



def greeks(
    instrument: Any,
    model: Any,
    market: Any,
    *,
    method: Literal["analytic", "mc", "pde"] = "analytic",
    **kwargs: Any,
) -> GreeksResult:
    if method != "analytic":
        raise NotSupportedError(f"method '{method}' is not supported")
    if kwargs:
        raise InvalidInputError("Unexpected keyword arguments for method 'analytic'")

    if (
        isinstance(instrument, EuropeanOption)
        and isinstance(model, BlackScholesModel)
        and isinstance(market, Market)
    ):
        return greeks_european(instrument, model, market)

    raise NotSupportedError("Unsupported instrument/model/market combination")
