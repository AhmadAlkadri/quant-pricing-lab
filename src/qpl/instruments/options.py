from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..exceptions import InvalidInputError


@dataclass(frozen=True)
class EuropeanOption:
    """European vanilla option definition.

    Parameters
    ----------
    kind
        Option type: `"call"` or `"put"`.
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
