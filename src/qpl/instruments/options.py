"""Vanilla option instruments, split by *exercise style*.

Exercise style is a property of the instrument, not of the engine: whether the
holder may exercise before expiry changes what is being priced, not how it is
computed. Encoding it as the instrument's type is what lets the engine
registry (`.agents/brain/adr/0005-engine-registry.md`) answer "can this engine
price this?" by lookup rather than by an ``if instrument.american:`` branch
inside each engine. The analytic, Monte Carlo and PDE engines register for
`EuropeanOption` only, so asking them for an `AmericanOption` raises
`NotSupportedError` through the ordinary dispatcher path; the tree engine
registers for both.

`AmericanOption` is deliberately *not* a subclass of `EuropeanOption`. Registry
lookup walks the MRO, so a subclass would silently resolve to the European
engines and be priced with the wrong exercise rule. The shared validation lives
in a common base, `VanillaOption`, which no engine registers for.

`DigitalOption` (Slice 6) is not in that hierarchy at all. It shares three
fields with a vanilla option and validates them the same way -- through the
module-level `_validated_kind` helper, so there is one copy of the rules and
one copy of the messages -- but its payoff is a step function rather than a
hockey stick, so nothing that prices a vanilla can price it. Making it a
subclass of `VanillaOption` would be harmless today (no engine registers for
that base) and a trap tomorrow, the moment anything does; the shared *code* is
a function call, not an inheritance edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..exceptions import InvalidInputError

__all__ = ["AmericanOption", "DigitalOption", "EuropeanOption", "VanillaOption"]


def _validated_kind(kind: str, strike: float, expiry: float) -> str:
    """Check the fields every option instrument here shares; return the kind.

    The order of the three checks and the exact messages are pinned by
    `tests/test_pricing_analytic.py` and are what `VanillaOption` raised
    before this was a function; `DigitalOption` reuses it rather than
    inheriting from `VanillaOption`.
    """
    kind_l = kind.lower()
    if kind_l not in {"call", "put"}:
        raise InvalidInputError("kind must be 'call' or 'put'")
    if strike <= 0:
        raise InvalidInputError("strike must be > 0")
    if expiry < 0:
        raise InvalidInputError("expiry must be >= 0")
    return kind_l


@dataclass(frozen=True)
class VanillaOption:
    """Common data and validation for a vanilla call or put.

    This base carries no exercise style of its own and is not registered with
    any engine; instantiate `EuropeanOption` or `AmericanOption` instead.

    Parameters
    ----------
    kind
        Option type: `"call"` or `"put"`. Normalised to lower case.
    strike
        Positive strike price.
    expiry
        Time to expiry in years (`>= 0`).
    """

    kind: Literal["call", "put"]
    strike: float
    expiry: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _validated_kind(self.kind, self.strike, self.expiry)
        )


@dataclass(frozen=True)
class EuropeanOption(VanillaOption):
    """European vanilla option: exercisable only at `expiry`."""


@dataclass(frozen=True)
class AmericanOption(VanillaOption):
    """American vanilla option: exercisable at any time up to `expiry`.

    Only the tree engine prices this; `method="analytic"`, `"mc"` and `"pde"`
    raise `NotSupportedError`, since none of them implements an early-exercise
    rule.
    """


@dataclass(frozen=True)
class DigitalOption:
    """European **cash-or-nothing** digital option.

    Pays a fixed amount `cash` at `expiry` if the option finishes in the
    money, and nothing otherwise:

        call:  cash * 1{S_T > K},        put:  cash * 1{S_T < K}.

    The two indicators are both *strict*, so at `S_T == K` neither pays. That
    event has probability zero under the model, which is why the convention is
    free to be symmetric between the call and the put; it matters only at a
    node or sample that lands exactly on the strike. QuantLib's
    `CashOrNothingPayoff` uses the same strict pair, which keeps
    `tests/oracle/test_digital_vs_quantlib.py` comparing the same contract.

    Why this is a separate type rather than a `VanillaOption` subclass: the
    payoff is a **jump**, not a kink. Every engine in this package that prices
    a vanilla assumes a continuous terminal condition somewhere -- the tree
    reads a payoff that is Lipschitz in the node spacing, the finite-difference
    grid samples a function whose one-sided limits agree, the Monte Carlo
    estimator has a payoff with bounded variation in the sample. None of that
    survives a step function, and each engine needs its own registered entry
    (see `.agents/brain/adr/0005-engine-registry.md`), which is exactly what
    keying the registry on the instrument type buys.

    Parameters
    ----------
    kind
        `"call"` (pays above the strike) or `"put"` (pays below). Normalised
        to lower case.
    strike
        Positive strike price: the level at which the payoff jumps.
    expiry
        Time to expiry in years (`>= 0`).
    cash
        Positive cash amount paid when in the money. The payoff and every
        price here are linear in `cash`, so it is a pure scaling; it is a
        field rather than a caller-side multiplication because the engines
        report prices and Greeks, and a Greek that silently assumed
        `cash = 1` would be wrong by that factor.
    """

    kind: Literal["call", "put"]
    strike: float
    expiry: float
    cash: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _validated_kind(self.kind, self.strike, self.expiry)
        )
        if not math.isfinite(self.cash) or self.cash <= 0:
            raise InvalidInputError("cash must be finite and > 0")
