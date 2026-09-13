from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol



class PricingEngine(Protocol):
    def price(self, instrument: Any, model: Any) -> float: ...


@dataclass(frozen=True)
class PriceResult:
    """Container for pricing outputs.

    Parameters
    ----------
    value
        Point estimate for the price.
    stderr
        Standard error for stochastic estimators. `None` for deterministic methods.
    meta
        Optional method-specific metadata.
    """
    value: float
    stderr: float | None = None
    meta: dict[str, Any] | None = None



@dataclass(frozen=True)
class GreeksResult:
    """Container for option Greeks outputs.

    Parameters
    ----------
    delta, gamma, vega, theta, rho
        Greek values in model units.
    meta
        Optional method-specific metadata.
    """
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    meta: dict[str, Any] | None = None
