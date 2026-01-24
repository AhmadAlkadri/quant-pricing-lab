from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.options import EuropeanOption
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import PriceResult


@dataclass(frozen=True)
class MCConfig:
    n_paths: int = 50_000
    n_steps: int = 252
    seed: int = 123


def price_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
) -> PriceResult:
    if cfg.n_paths < 2:
        raise InvalidInputError("n_paths must be >= 2 for MC stderr with ddof=1")

    s0 = market.spot
    k = option.strike
    t = option.expiry
    r = market.rate_curve.rate
    q = market.dividend_curve.yield_
    sigma = model.sigma

    meta = {
        "method": "mc",
        "model": "BlackScholes",
        "n_paths": cfg.n_paths,
        "seed": cfg.seed,
    }

    if t == 0.0:
        if option.kind == "call":
            value = max(s0 - k, 0.0)
        else:
            value = max(k - s0, 0.0)
        return PriceResult(value=float(value), stderr=0.0, meta=meta)

    if sigma == 0.0:
        forward = s0 * math.exp((r - q) * t)
        disc = math.exp(-r * t)
        if option.kind == "call":
            value = disc * max(forward - k, 0.0)
        else:
            value = disc * max(k - forward, 0.0)
        return PriceResult(value=float(value), stderr=0.0, meta=meta)

    rng = np.random.default_rng(cfg.seed)
    z = rng.normal(size=cfg.n_paths)
    drift = (r - q - 0.5 * sigma * sigma) * t
    vol = sigma * math.sqrt(t)
    s_t = s0 * np.exp(drift + vol * z)

    if option.kind == "call":
        payoff = np.maximum(s_t - k, 0.0)
    else:
        payoff = np.maximum(k - s_t, 0.0)

    pv = math.exp(-r * t) * payoff
    value = float(np.mean(pv))
    stderr = float(np.std(pv, ddof=1) / math.sqrt(cfg.n_paths))

    return PriceResult(value=value, stderr=stderr, meta=meta)
