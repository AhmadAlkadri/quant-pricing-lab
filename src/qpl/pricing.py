"""Public pricing dispatcher.

`price` and `greeks` do three things and nothing else: look up the keyword
contract for the requested `method`, validate the caller's keyword arguments
against it, and look up the engine registered for
`(instrument type, model type, method)`. The lookup tables live in
`qpl.engines.registry`; the wiring below is the whole list of engines this
package ships. See `.agents/brain/adr/0005-engine-registry.md`.
"""

from __future__ import annotations

from typing import Any, Literal

from .engines.analytic.black_scholes import (
    ANALYTIC_METHOD_SPEC,
    greeks_european as greeks_european_analytic,
    price_european as price_european_analytic,
)
from .engines.base import GreeksResult, PriceResult
from .engines.mc.pricers import (
    MC_METHOD_SPEC,
    greeks_european as greeks_european_mc,
    price_european as price_european_mc,
)
from .engines.pde.pricers import (
    PDE_METHOD_SPEC,
    greeks_european as greeks_european_pde,
    price_european as price_european_pde,
)
from .engines.registry import (
    method_spec,
    register,
    resolve_greeks,
    resolve_price,
)
from .instruments.options import EuropeanOption
from .models.black_scholes import BlackScholesModel

Method = Literal["analytic", "mc", "pde"]


def _register_builtin_engines() -> None:
    """Wire the engines this package ships into the registry.

    Explicit and ordinary: a new engine adds one `register(...)` call here (or
    the owning package calls `register` itself), and nothing scans the import
    graph looking for engines to discover.
    """
    common = {"instrument_type": EuropeanOption, "model_type": BlackScholesModel}
    register(
        **common,
        spec=ANALYTIC_METHOD_SPEC,
        price=price_european_analytic,
        greeks=greeks_european_analytic,
    )
    register(
        **common,
        spec=MC_METHOD_SPEC,
        price=price_european_mc,
        greeks=greeks_european_mc,
    )
    register(
        **common,
        spec=PDE_METHOD_SPEC,
        price=price_european_pde,
        greeks=greeks_european_pde,
    )


_register_builtin_engines()


def price(
    instrument: Any,
    model: Any,
    market: Any,
    *,
    method: Method = "analytic",
    **kwargs: Any,
) -> PriceResult:
    """Dispatch option pricing to the selected engine.

    Parameters
    ----------
    instrument
        Instrument instance. Currently `EuropeanOption` is supported.
    model
        Model instance. Currently `BlackScholesModel` is supported.
    market
        Market instance. Currently `Market` is supported.
    method
        Pricing engine selector: `"analytic"`, `"mc"`, or `"pde"`.
    **kwargs
        Method-specific keyword arguments:
        - analytic: no extra kwargs
        - mc: `cfg=MCConfig`
        - pde: `cfg=PDEConfig`

    Returns
    -------
    PriceResult
        Price estimate and optional metadata.

    Raises
    ------
    InvalidInputError
        If required kwargs are missing/invalid or unexpected kwargs are passed.
    NotSupportedError
        If the method is unknown or the instrument/model/market combination is unsupported.
    """
    bound = method_spec(method).bind_price_kwargs(kwargs)
    engine = resolve_price(instrument, model, market, method)
    return engine(instrument, model, market, **bound)


def greeks(
    instrument: Any,
    model: Any,
    market: Any,
    *,
    method: Method = "analytic",
    **kwargs: Any,
) -> GreeksResult:
    """Dispatch Greeks computation to the selected engine.

    Parameters
    ----------
    instrument
        Instrument instance. Currently `EuropeanOption` is supported.
    model
        Model instance. Currently `BlackScholesModel` is supported.
    market
        Market instance. Currently `Market` is supported.
    method
        Greeks engine selector: `"analytic"`, `"mc"`, or `"pde"`.
    **kwargs
        Method-specific keyword arguments:
        - analytic: no extra kwargs
        - mc: `cfg=MCConfig` and optional `bumps=dict[str, float]`
        - pde: `cfg=PDEConfig`

    Returns
    -------
    GreeksResult
        Greeks values and optional metadata.

    Raises
    ------
    InvalidInputError
        If required kwargs are missing/invalid or unexpected kwargs are passed.
    NotSupportedError
        If the method is unknown or the instrument/model/market combination is unsupported.
    """
    bound = method_spec(method).bind_greeks_kwargs(kwargs)
    engine = resolve_greeks(instrument, model, market, method)
    return engine(instrument, model, market, **bound)
