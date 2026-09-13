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
from .engines.analytic.digital import (
    greeks_digital as greeks_digital_analytic,
    price_digital as price_digital_analytic,
)
from .engines.base import GreeksResult, PriceResult
from .engines.mc.pricers import (
    MC_METHOD_SPEC,
    greeks_european as greeks_european_mc,
    price_european as price_european_mc,
)
from .engines.pde.american import (
    greeks_american as greeks_american_pde,
    price_american as price_american_pde,
)
from .engines.pde.digital import (
    greeks_digital as greeks_digital_pde,
    price_digital as price_digital_pde,
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
from .engines.tree.american import (
    greeks_american as greeks_american_tree,
    price_american as price_american_tree,
)
from .engines.tree.digital import (
    greeks_digital as greeks_digital_tree,
    price_digital as price_digital_tree,
)
from .engines.tree.pricers import (
    TREE_METHOD_SPEC,
    greeks_european as greeks_european_tree,
    price_european as price_european_tree,
)
from .instruments.options import AmericanOption, DigitalOption, EuropeanOption
from .models.black_scholes import BlackScholesModel

Method = Literal["analytic", "mc", "pde", "tree"]


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
    register(
        **common,
        spec=TREE_METHOD_SPEC,
        price=price_european_tree,
        greeks=greeks_european_tree,
    )
    # Exercise style is an instrument property (see `qpl.instruments.options`),
    # so an American entry is another key on the same method, not another
    # method. The tree engine (Slice 2) and the PDE engine's PSOR path
    # (Slice 5) each register one. Nothing registers the analytic or MC engines
    # for `AmericanOption`: asking them for one raises `NotSupportedError`
    # through the ordinary lookup, which is the point of keying on the
    # instrument type.
    american = {"instrument_type": AmericanOption, "model_type": BlackScholesModel}
    register(
        **american,
        spec=TREE_METHOD_SPEC,
        price=price_american_tree,
        greeks=greeks_american_tree,
    )
    register(
        **american,
        spec=PDE_METHOD_SPEC,
        price=price_american_pde,
        greeks=greeks_american_pde,
    )
    # Slice 6: the cash-or-nothing digital, a *discontinuous* payoff rather
    # than a different exercise rule. Same pattern again -- a new instrument
    # type is a new set of keys, not a new method string, and an engine that
    # has nothing sensible to do with a jump simply does not register.
    digital = {"instrument_type": DigitalOption, "model_type": BlackScholesModel}
    register(
        **digital,
        spec=ANALYTIC_METHOD_SPEC,
        price=price_digital_analytic,
        greeks=greeks_digital_analytic,
    )
    register(
        **digital,
        spec=TREE_METHOD_SPEC,
        price=price_digital_tree,
        greeks=greeks_digital_tree,
    )
    register(
        **digital,
        spec=PDE_METHOD_SPEC,
        price=price_digital_pde,
        greeks=greeks_digital_pde,
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
        Instrument instance. `EuropeanOption` is supported by every method;
        `AmericanOption` by `method="tree"` and `method="pde"`.
    model
        Model instance. Currently `BlackScholesModel` is supported.
    market
        Market instance. Currently `Market` is supported.
    method
        Pricing engine selector: `"analytic"`, `"mc"`, `"pde"`, or `"tree"`.
    **kwargs
        Method-specific keyword arguments:
        - analytic: no extra kwargs
        - mc: `cfg=MCConfig`
        - pde: `cfg=PDEConfig`
        - tree: `cfg=TreeConfig`

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
        Instrument instance. `EuropeanOption` is supported by every method;
        `AmericanOption` by `method="tree"` and `method="pde"`.
    model
        Model instance. Currently `BlackScholesModel` is supported.
    market
        Market instance. Currently `Market` is supported.
    method
        Greeks engine selector: `"analytic"`, `"mc"`, `"pde"`, or `"tree"`.
    **kwargs
        Method-specific keyword arguments:
        - analytic: no extra kwargs
        - mc: `cfg=MCConfig` and optional `bumps=dict[str, float]`
        - pde: `cfg=PDEConfig`
        - tree: `cfg=TreeConfig`

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
