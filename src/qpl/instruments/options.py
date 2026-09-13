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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..exceptions import InvalidInputError

__all__ = ["AmericanOption", "EuropeanOption", "VanillaOption"]


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
        kind_l = self.kind.lower()
        if kind_l not in {"call", "put"}:
            raise InvalidInputError("kind must be 'call' or 'put'")
        if self.strike <= 0:
            raise InvalidInputError("strike must be > 0")
        if self.expiry < 0:
            raise InvalidInputError("expiry must be >= 0")
        object.__setattr__(self, "kind", kind_l)


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
