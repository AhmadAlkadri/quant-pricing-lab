"""Engine registry for the `qpl.pricing` dispatcher.

The dispatcher used to be an ``isinstance`` ladder repeated once per method,
so every new engine meant another copy of the same four checks. This module
replaces the ladder with two plain dictionaries keyed by
``(instrument type, model type, method)`` -- one for prices, one for Greeks --
plus a :class:`MethodSpec` per ``method`` string that states which keyword
arguments that method accepts and how to validate them.

There is no metaclass, no ``supports()`` predicate, and no import-time plugin
scan: `qpl.pricing` imports the engine modules it knows about and calls
:func:`register` explicitly. The registry is mechanism only; the wiring table
lives with the dispatcher, and the per-method keyword contract lives with the
engine that defines it.

``Market`` is deliberately *not* part of the key; see
`.agents/brain/adr/0005-engine-registry.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..exceptions import InvalidInputError, NotSupportedError
from ..market.market import Market

__all__ = [
    "EngineKey",
    "MethodSpec",
    "known_methods",
    "method_spec",
    "register",
    "resolve_greeks",
    "resolve_price",
]

EngineKey = tuple[type, type, str]
"""``(instrument type, model type, method)``."""

Validator = Callable[[Any], None]
"""Raises :class:`~qpl.exceptions.InvalidInputError` if the value is unusable."""

_UNSUPPORTED_COMBINATION = "Unsupported instrument/model/market combination"


@dataclass(frozen=True)
class MethodSpec:
    """The keyword-argument contract for one ``method`` string.

    Parameters
    ----------
    method
        The dispatcher's ``method=`` selector, e.g. ``"pde"``.
    cfg_type
        Type that the required ``cfg=`` argument must be an instance of, or
        ``None`` when the method takes no configuration object at all.
    greeks_optional_kwargs
        Extra keyword arguments accepted by :func:`qpl.pricing.greeks` only
        (never by :func:`qpl.pricing.price`), mapped to their validators. The
        bound value is passed through as ``None`` when the caller omits it, so
        the engine keeps its own default.
    """

    method: str
    cfg_type: type | None = None
    greeks_optional_kwargs: Mapping[str, Validator] = field(default_factory=dict)

    def bind_price_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Validate ``price`` kwargs and return what to forward to the engine."""
        return self._bind(kwargs, optional=())

    def bind_greeks_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Validate ``greeks`` kwargs and return what to forward to the engine."""
        return self._bind(kwargs, optional=tuple(self.greeks_optional_kwargs))

    def _bind(self, kwargs: Mapping[str, Any], *, optional: tuple[str, ...]) -> dict[str, Any]:
        takes_cfg = self.cfg_type is not None
        cfg = kwargs.get("cfg")

        # Order of the three checks below is load-bearing: it is the order the
        # pre-registry ladder used, and the error messages are part of the
        # public behaviour.
        if takes_cfg and cfg is None:
            raise InvalidInputError(f"cfg is required for method '{self.method}'")

        allowed = set(optional)
        if takes_cfg:
            allowed.add("cfg")
        if any(name not in allowed for name in kwargs):
            raise InvalidInputError(f"Unexpected keyword arguments for method '{self.method}'")

        if takes_cfg and not isinstance(cfg, self.cfg_type):  # type: ignore[arg-type]
            raise InvalidInputError(
                f"cfg must be an instance of {self.cfg_type.__name__}"  # type: ignore[union-attr]
            )

        bound: dict[str, Any] = {}
        if takes_cfg:
            bound["cfg"] = cfg
        for name in optional:
            value = kwargs.get(name)
            if value is not None:
                self.greeks_optional_kwargs[name](value)
            bound[name] = value
        return bound


_SPECS: dict[str, MethodSpec] = {}
_PRICE_ENGINES: dict[EngineKey, Callable[..., Any]] = {}
_GREEKS_ENGINES: dict[EngineKey, Callable[..., Any]] = {}


def register(
    *,
    instrument_type: type,
    model_type: type,
    spec: MethodSpec,
    price: Callable[..., Any] | None = None,
    greeks: Callable[..., Any] | None = None,
) -> None:
    """Bind engine callables to ``(instrument_type, model_type, spec.method)``.

    Parameters
    ----------
    instrument_type, model_type
        Types matched against the caller's arguments. Lookup walks the MRO, so
        a subclass of a registered type resolves to the same engine, matching
        the ``isinstance`` semantics this registry replaced.
    spec
        Keyword contract for ``spec.method``. Registering the same method name
        twice with a different contract is a programming error and raises.
    price, greeks
        Engine callables invoked as ``fn(instrument, model, market, **bound)``.
        At least one must be given.

    Raises
    ------
    InvalidInputError
        On a conflicting :class:`MethodSpec`, a duplicate key, or no callable.
    """
    existing = _SPECS.get(spec.method)
    if existing is not None and existing != spec:
        raise InvalidInputError(
            f"method '{spec.method}' is already registered with a different MethodSpec"
        )
    if price is None and greeks is None:
        raise InvalidInputError("register() needs at least one of price= or greeks=")

    _SPECS[spec.method] = spec
    key: EngineKey = (instrument_type, model_type, spec.method)
    for table, fn in ((_PRICE_ENGINES, price), (_GREEKS_ENGINES, greeks)):
        if fn is None:
            continue
        if key in table:
            raise InvalidInputError(f"engine already registered for {key}")
        table[key] = fn


def method_spec(method: str) -> MethodSpec:
    """Return the contract for ``method``.

    Raises
    ------
    NotSupportedError
        If no engine has registered that method name.
    """
    spec = _SPECS.get(method)
    if spec is None:
        raise NotSupportedError(f"method '{method}' is not supported")
    return spec


def known_methods() -> tuple[str, ...]:
    """Registered method names, sorted."""
    return tuple(sorted(_SPECS))


def _resolve(
    table: Mapping[EngineKey, Callable[..., Any]],
    instrument: Any,
    model: Any,
    market: Any,
    method: str,
) -> Callable[..., Any]:
    if not isinstance(market, Market):
        raise NotSupportedError(_UNSUPPORTED_COMBINATION)
    for instrument_type in type(instrument).__mro__:
        for model_type in type(model).__mro__:
            fn = table.get((instrument_type, model_type, method))
            if fn is not None:
                return fn
    raise NotSupportedError(_UNSUPPORTED_COMBINATION)


def resolve_price(instrument: Any, model: Any, market: Any, method: str) -> Callable[..., Any]:
    """Look up the pricing engine, or raise `NotSupportedError`."""
    return _resolve(_PRICE_ENGINES, instrument, model, market, method)


def resolve_greeks(instrument: Any, model: Any, market: Any, method: str) -> Callable[..., Any]:
    """Look up the Greeks engine, or raise `NotSupportedError`."""
    return _resolve(_GREEKS_ENGINES, instrument, model, market, method)
