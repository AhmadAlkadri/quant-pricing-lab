from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..base import PriceResult

@dataclass(frozen=True)
class MCConfig:
    n_paths: int = 50_000
    n_steps: int = 252
    seed: int = 123

def price_european_stub(*, cfg: MCConfig) -> PriceResult:
    # Placeholder to be implemented in Milestone A
    rng = np.random.default_rng(cfg.seed)
    _ = rng.normal(size=10)
    raise NotImplementedError("Monte Carlo pricer not implemented yet (Milestone A).")
