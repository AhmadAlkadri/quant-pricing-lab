from __future__ import annotations

import math
from dataclasses import dataclass

from ..exceptions import InvalidInputError



@dataclass(frozen=True)
class FlatRateCurve:
    rate: float
    allow_negative: bool = False

    def __post_init__(self) -> None:
        if self.rate < 0 and not self.allow_negative:
            raise InvalidInputError("rate must be >= 0 unless allow_negative=True")

    def df(self, t: float) -> float:
        if t < 0:
            raise InvalidInputError("t must be >= 0")
        return math.exp(-self.rate * t)



@dataclass(frozen=True)
class FlatDividendCurve:
    yield_: float
    allow_negative: bool = False

    def __post_init__(self) -> None:
        if self.yield_ < 0 and not self.allow_negative:
            raise InvalidInputError("yield_ must be >= 0 unless allow_negative=True")

    def df(self, t: float) -> float:
        if t < 0:
            raise InvalidInputError("t must be >= 0")
        return math.exp(-self.yield_ * t)
