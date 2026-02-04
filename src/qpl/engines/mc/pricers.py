from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.options import EuropeanOption
from ...market.curves import FlatDividendCurve, FlatRateCurve
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult


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


def greeks_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None = None,
) -> GreeksResult:
    if cfg.n_paths < 2:
        raise InvalidInputError("n_paths must be >= 2 for MC stderr with ddof=1")
    if cfg.n_steps < 1:
        raise InvalidInputError("n_steps must be >= 1")

    s0 = market.spot
    t = option.expiry
    sigma = model.sigma

    def _bump(name: str, default: float) -> float:
        if bumps is None or name not in bumps:
            return default
        value = float(bumps[name])
        if value <= 0.0:
            raise InvalidInputError(f"{name} bump must be > 0")
        return value

    dS = _bump("spot", max(abs(s0) * 1e-4, 1e-6))
    if s0 <= dS:
        dS = 0.5 * s0
    dsigma = _bump("sigma", 1e-4)
    if sigma > 0.0 and sigma <= dsigma:
        dsigma = 0.5 * sigma
    dr = _bump("r", 1e-5)

    meta = {
        "method": "mc",
        "model": "BlackScholes",
        "n_paths": cfg.n_paths,
        "n_steps": cfg.n_steps,
        "seed": cfg.seed,
        "fd": "central",
        "bumps": {"spot": dS, "sigma": dsigma, "r": dr},
    }

    if t == 0.0:
        meta["t0"] = True
        return GreeksResult(
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=0.0,
            meta=meta,
        )

    def _price(mkt: Market, mdl: BlackScholesModel) -> float:
        return price_european(option, mdl, mkt, cfg=cfg).value

    base = _price(market, model)

    def _price_spot(spot: float) -> float:
        mkt = Market(
            spot=spot,
            rate_curve=market.rate_curve,
            dividend_curve=market.dividend_curve,
        )
        return _price(mkt, model)

    value_up = _price_spot(s0 + dS)
    value_dn = _price_spot(s0 - dS)

    delta = (value_up - value_dn) / (2.0 * dS)
    gamma = (value_up - 2.0 * base + value_dn) / (dS * dS)

    if sigma == 0.0:
        vega = 0.0
    else:
        model_up = BlackScholesModel(sigma=sigma + dsigma)
        model_dn = BlackScholesModel(sigma=sigma - dsigma)
        value_up = _price(market, model_up)
        value_dn = _price(market, model_dn)
        vega = (value_up - value_dn) / (2.0 * dsigma)

    r = market.rate(t)
    q = market.dividend_yield(t)

    def _price_rate(rate: float) -> float:
        mkt = Market(
            spot=s0,
            rate_curve=FlatRateCurve(rate, allow_negative=True),
            dividend_curve=FlatDividendCurve(q, allow_negative=True),
        )
        return _price(mkt, model)

    value_up = _price_rate(r + dr)
    value_dn = _price_rate(r - dr)
    rho = (value_up - value_dn) / (2.0 * dr)

    # Theta implementation via finite difference (backward or central)
    # We use backward diff (V(t) - V(t-dt))/dt to avoid looking into the future
    # which might exceed expiry if we used forward diff, though t+dt is usually fine.
    # However, for Theta, we explicitly want -dV/dt.
    
    # We'll use a simple approach: if t is large enough, bump time backwards.
    # If t is too close to 0, bump forward?
    # Actually, standard definition Theta = -dV/dt.
    # Approximation: -(V(t + dt) - V(t)) / dt  (forward diff in time)
    # Or: -(V(t) - V(t - dt)) / dt (backward diff in time)
    
    # Let's use Backward Diff: Theta ~ (V(t - dt) - V(t)) / dt
    # This corresponds to "passage of time" reducing time-to-expiry.
    # New expiry t' = t - dt.
    
    dt = _bump("time", min(1e-4, t / 2.0))
    # Safety: ensure we don't go negative or too small
    if t <= dt:
         dt = t * 0.5

    def _price_time_shifted(new_t: float) -> float:
        # We need to construct a new option with modified expiry
        # OR just call price_european with a modified option.
        # However, option is immutable (frozen).
        # We can use dataclasses.replace
        from dataclasses import replace
        new_opt = replace(option, expiry=new_t)
        return price_european(new_opt, model, market, cfg=cfg).value

    # Calculate V(t-dt)
    # Note: price_european handles t=0 logic if dt~t
    val_minus = _price_time_shifted(t - dt)
    
    # Theta = (val_minus - base) / dt
    # Explain: 
    #   V(t_now) = base (time to expiry T)
    #   V(t_later) = val_minus (time to expiry T - dt)
    #   Theta approx (V(t_later) - V(t_now)) / dt
    #   = (val_minus - base) / dt
    theta = (val_minus - base) / dt

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        meta=meta,
    )
