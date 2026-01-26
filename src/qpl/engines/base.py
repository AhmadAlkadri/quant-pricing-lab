from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol



class PricingEngine(Protocol):
    def price(self, instrument: Any, model: Any) -> float: ...


@dataclass(frozen=True)
class PriceResult:
    """Pricing result.

    stderr is None for deterministic/analytic methods; for Monte Carlo it is the
    standard error of the estimator (0.0 for deterministic edge cases).
    """
    value: float
    stderr: float | None = None
    meta: dict[str, Any] | None = None



@dataclass(frozen=True)
class GreeksResult:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    meta: dict[str, Any] | None = None
