from __future__ import annotations

import math
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

    def df_r(self, t: float) -> float:
        return self.rate_curve.df(t)

    def df_q(self, t: float) -> float:
        return self.dividend_curve.df(t)

    def rate(self, t: float) -> float:
        if t == 0:
            return 0.0
        return -math.log(self.df_r(t)) / t

    def dividend_yield(self, t: float) -> float:
        if t == 0:
            return 0.0
        return -math.log(self.df_q(t)) / t
