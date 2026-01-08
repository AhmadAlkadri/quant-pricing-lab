from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any

class PricingEngine(Protocol):
    def price(self, instrument: Any, model: Any) -> float: ...

@dataclass(frozen=True)
class PriceResult:
    value: float
    stderr: float | None = None
    meta: dict[str, Any] | None = None
