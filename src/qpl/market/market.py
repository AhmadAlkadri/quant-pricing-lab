from __future__ import annotations

from dataclasses import dataclass

from ..exceptions import InvalidInputError
from .curves import FlatDividendCurve, FlatRateCurve



@dataclass(frozen=True)
class Market:
    spot: float
    rate_curve: FlatRateCurve
    dividend_curve: FlatDividendCurve

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise InvalidInputError("spot must be > 0")
