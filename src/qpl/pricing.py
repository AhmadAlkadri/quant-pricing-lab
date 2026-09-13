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

from .engines.analytic.asian import (
    greeks_asian as greeks_asian_analytic,
    price_asian as price_asian_analytic,
)
from .engines.analytic.barrier import (
    greeks_barrier as greeks_barrier_analytic,
    price_barrier as price_barrier_analytic,
)
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
from .engines.mc.american import (
    greeks_american as greeks_american_mc,
    price_american as price_american_mc,
)
from .engines.mc.asian import (
    greeks_asian as greeks_asian_mc,
    price_asian as price_asian_mc,
)
from .engines.mc.barrier import (
    greeks_barrier as greeks_barrier_mc,
    price_barrier as price_barrier_mc,
)
from .engines.mc.digital import (
    greeks_digital as greeks_digital_mc,
    price_digital as price_digital_mc,
)
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
from .engines.tree.barrier import (
    greeks_barrier as greeks_barrier_tree,
    price_barrier as price_barrier_tree,
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
from .instruments.options import (
    AmericanOption,
    AsianOption,
    BarrierOption,
    DigitalOption,
    EuropeanOption,
)
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
    # method. The tree engine (Slice 2), the PDE engine's PSOR path (Slice 5)
    # and the least-squares Monte Carlo engine (Slice 11) each register one.
    # Nothing registers the analytic engine for `AmericanOption`: asking it for
    # one raises `NotSupportedError` through the ordinary lookup, which is the
    # point of keying on the instrument type.
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
    # Slice 11: Longstaff-Schwartz. The third discretisation of early exercise
    # -- a path sample rather than a lattice or a grid -- and the first engine
    # whose American answer carries a standard error. Its `greeks` callable
    # always raises, for the same reason the MC digital's did before Slice 10:
    # not registering would report "Unsupported instrument/model/market
    # combination", which is false here (the price engine right next to it
    # prices exactly that combination) and says nothing about why.
    register(
        **american,
        spec=MC_METHOD_SPEC,
        price=price_american_mc,
        greeks=greeks_american_mc,
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
    # The MC entry registers a `greeks` callable that always raises
    # `NotSupportedError`, rather than registering no callable at all: the two
    # differ in the message the caller gets. Not registering would report
    # "Unsupported instrument/model/market combination", which is true but
    # says nothing about *why* and nothing about the likelihood-ratio
    # estimator that will replace it in Phase 3.
    register(
        **digital,
        spec=MC_METHOD_SPEC,
        price=price_digital_mc,
        greeks=greeks_digital_mc,
    )
    # Slice 8: the fixed-strike Asian, a *path-dependent* payoff. Only the two
    # engines that can see a path register: the analytic one for the geometric
    # average (which is exactly lognormal) and Monte Carlo for both averagings.
    # The tree and the grid do not register at all -- pricing an average needs a
    # second state variable, which is a different discretisation and not a
    # different branch inside these ones.
    #
    # Both analytic entries are registered even though `price_asian` refuses an
    # *arithmetic* Asian: the refusal has to be the one that names
    # Turnbull-Wakeman and Monte Carlo, and an unregistered key would report
    # "Unsupported instrument/model/market combination" instead. Same reasoning
    # as the MC digital Greeks above.
    asian = {"instrument_type": AsianOption, "model_type": BlackScholesModel}
    register(
        **asian,
        spec=ANALYTIC_METHOD_SPEC,
        price=price_asian_analytic,
        greeks=greeks_asian_analytic,
    )
    register(
        **asian,
        spec=MC_METHOD_SPEC,
        price=price_asian_mc,
        greeks=greeks_asian_mc,
    )
    # Slice 12: the single barrier, path-dependent AND discontinuous at once.
    # The analytic entry prices the continuously monitored contract and refuses
    # a discrete schedule by inspecting the instrument, which is the whole
    # reason the monitoring convention lives on the contract rather than in a
    # config object.
    barrier = {"instrument_type": BarrierOption, "model_type": BlackScholesModel}
    register(
        **barrier,
        spec=ANALYTIC_METHOD_SPEC,
        price=price_barrier_analytic,
        greeks=greeks_barrier_analytic,
    )
    # The Monte Carlo entry prices the DISCRETE contract on the schedule the
    # instrument names, with `MCConfig.barrier_correction` selecting which of
    # the two contracts the answer estimates. Its `greeks` callable is
    # registered and always raises, for the same reason the MC digital's did
    # before Slice 10: an unregistered key would report an unsupported
    # combination, which is false when the price engine next to it prices
    # exactly that combination.
    register(
        **barrier,
        spec=MC_METHOD_SPEC,
        price=price_barrier_mc,
        greeks=greeks_barrier_mc,
    )
    # The lattice approximates the CONTINUOUS contract by knocking out at every
    # time level, and refuses a discrete schedule; its Greeks are registered
    # and always raise, because the lattice estimators would report the
    # Boyle-Lau sawtooth amplified by 1/dt under a Greek's name.
    register(
        **barrier,
        spec=TREE_METHOD_SPEC,
        price=price_barrier_tree,
        greeks=greeks_barrier_tree,
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
        `AmericanOption` by `method="tree"`, `method="pde"` and (Slice 11,
        least-squares Monte Carlo on a Bermudan exercise grid) `method="mc"`.
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
        `AmericanOption` by `method="tree"` and `method="pde"`;
        `method="mc"` prices one but refuses its Greeks.
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
