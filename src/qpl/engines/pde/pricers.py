from __future__ import annotations
from dataclasses import dataclass
from ..base import PriceResult

@dataclass(frozen=True)
class PDEConfig:
    n_space: int = 400
    n_time: int = 200
    theta: float = 0.5  # Crank–Nicolson

def price_bs_fd_stub(*, cfg: PDEConfig) -> PriceResult:
    # Placeholder to be implemented in Milestone B
    raise NotImplementedError("PDE pricer not implemented yet (Milestone B).")
