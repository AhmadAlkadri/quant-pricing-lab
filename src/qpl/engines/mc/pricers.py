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
    """Monte Carlo configuration.

    n_steps controls time discretization; n_steps=1 uses terminal sampling.
    """
    n_paths: int = 50_000
    n_steps: int = 1
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
    if cfg.n_steps < 1:
        raise InvalidInputError("n_steps must be >= 1")

    s0 = market.spot
    k = option.strike
    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    sigma = model.sigma
    df_r = market.df_r(t)

    meta = {
        "method": "mc",
        "model": "BlackScholes",
        "n_paths": cfg.n_paths,
        "n_steps": cfg.n_steps,
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
        if option.kind == "call":
            value = df_r * max(forward - k, 0.0)
        else:
            value = df_r * max(k - forward, 0.0)
        return PriceResult(value=float(value), stderr=0.0, meta=meta)

    rng = np.random.default_rng(cfg.seed)
    if cfg.n_steps == 1:
        z = rng.normal(size=cfg.n_paths)
        drift = (r - q - 0.5 * sigma * sigma) * t
        vol = sigma * math.sqrt(t)
        s_t = s0 * np.exp(drift + vol * z)
    else:
        dt = t / cfg.n_steps
        drift_step = (r - q - 0.5 * sigma * sigma) * dt
        vol_step = sigma * math.sqrt(dt)
        z = rng.normal(size=(cfg.n_paths, cfg.n_steps))
        log_s_t = math.log(s0) + drift_step * cfg.n_steps + vol_step * np.sum(z, axis=1)
        s_t = np.exp(log_s_t)

    if option.kind == "call":
        payoff = np.maximum(s_t - k, 0.0)
    else:
        payoff = np.maximum(k - s_t, 0.0)

    pv = df_r * payoff
    value = float(np.mean(pv))
    stderr = float(np.std(pv, ddof=1) / math.sqrt(cfg.n_paths))

    return PriceResult(value=value, stderr=stderr, meta=meta)
