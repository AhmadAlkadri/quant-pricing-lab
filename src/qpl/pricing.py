from __future__ import annotations

from typing import Any, Literal

from .engines.analytic.black_scholes import greeks_european, price_european as price_european_analytic
from .engines.base import GreeksResult, PriceResult
from .engines.mc.pricers import (
    MCConfig,
    greeks_european as greeks_european_mc,
    price_european as price_european_mc,
)
from .engines.pde.pricers import PDEConfig, price_european as price_european_pde
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
    if method == "analytic":
        if kwargs:
            raise InvalidInputError("Unexpected keyword arguments for method 'analytic'")

        if (
            isinstance(instrument, EuropeanOption)
            and isinstance(model, BlackScholesModel)
            and isinstance(market, Market)
        ):
            return price_european_analytic(instrument, model, market)

        raise NotSupportedError("Unsupported instrument/model/market combination")

    if method == "mc":
        cfg = kwargs.get("cfg")
        extra_kwargs = {key: value for key, value in kwargs.items() if key != "cfg"}
        if cfg is None:
            raise InvalidInputError("cfg is required for method 'mc'")
        if extra_kwargs:
            raise InvalidInputError("Unexpected keyword arguments for method 'mc'")
        if not isinstance(cfg, MCConfig):
            raise InvalidInputError("cfg must be an instance of MCConfig")

        if (
            isinstance(instrument, EuropeanOption)
            and isinstance(model, BlackScholesModel)
            and isinstance(market, Market)
        ):
            return price_european_mc(instrument, model, market, cfg=cfg)

        raise NotSupportedError("Unsupported instrument/model/market combination")

    if method == "pde":
        cfg = kwargs.get("cfg")
        extra_kwargs = {key: value for key, value in kwargs.items() if key != "cfg"}
        if cfg is None:
            raise InvalidInputError("cfg is required for method 'pde'")
        if extra_kwargs:
            raise InvalidInputError("Unexpected keyword arguments for method 'pde'")
        if not isinstance(cfg, PDEConfig):
            raise InvalidInputError("cfg must be an instance of PDEConfig")

        if (
            isinstance(instrument, EuropeanOption)
            and isinstance(model, BlackScholesModel)
            and isinstance(market, Market)
        ):
            return price_european_pde(instrument, model, market, cfg=cfg)

        raise NotSupportedError("Unsupported instrument/model/market combination")

    raise NotSupportedError(f"method '{method}' is not supported")



def greeks(
    instrument: Any,
    model: Any,
    market: Any,
    *,
    method: Literal["analytic", "mc", "pde"] = "analytic",
    **kwargs: Any,
) -> GreeksResult:
    if method == "analytic":
        if kwargs:
            raise InvalidInputError("Unexpected keyword arguments for method 'analytic'")

        if (
            isinstance(instrument, EuropeanOption)
            and isinstance(model, BlackScholesModel)
            and isinstance(market, Market)
        ):
            return greeks_european(instrument, model, market)

        raise NotSupportedError("Unsupported instrument/model/market combination")

    if method == "mc":
        cfg = kwargs.get("cfg")
        bumps = kwargs.get("bumps")
        fd = kwargs.get("fd")
        extra_kwargs = {
            key: value for key, value in kwargs.items() if key not in {"cfg", "bumps", "fd"}
        }
        if cfg is None:
            raise InvalidInputError("cfg is required for method 'mc'")
        if extra_kwargs:
            raise InvalidInputError("Unexpected keyword arguments for method 'mc'")
        if not isinstance(cfg, MCConfig):
            raise InvalidInputError("cfg must be an instance of MCConfig")
        if bumps is not None and fd is not None:
            raise InvalidInputError("Use only one of bumps or fd for method 'mc'")
        if bumps is None:
            bumps = fd
        if bumps is not None and not isinstance(bumps, dict):
            raise InvalidInputError("bumps must be a dict of bump sizes")

        if (
            isinstance(instrument, EuropeanOption)
            and isinstance(model, BlackScholesModel)
            and isinstance(market, Market)
        ):
            return greeks_european_mc(instrument, model, market, cfg=cfg, bumps=bumps)

        raise NotSupportedError("Unsupported instrument/model/market combination")

    if method == "pde":
        raise NotSupportedError("Greeks not supported for method 'pde' yet")

    raise NotSupportedError(f"method '{method}' is not supported")
