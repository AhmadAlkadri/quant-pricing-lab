"""Cash-or-nothing digitals by Monte Carlo: price only, and why only price.

The price is the easy half. A digital's payoff is an indicator, so the
estimator is a sample proportion: draw `N` terminal spots, count how many
finish in the money, discount. Nothing about the discontinuity troubles it --
the payoff is bounded, its variance is finite and known exactly
(`cash**2 p (1 - p)` for a Bernoulli), and the central limit theorem applies
as it does to a vanilla. The estimator is unbiased for **any** `N`, and the
standard error falls at `N**-1/2` with the constant `cash * e^{-rT} sqrt(p(1-p))`.
Measured in `tests/test_digital_mc.py`.

That is the point worth making about Monte Carlo here: unlike the lattice and
the grid, it does not care that the payoff jumps. A tree has to decide which
side of the strike each node is on; a finite-difference grid has to represent
a step function in a space of grid functions. A sample path either finishes
above the strike or it does not, and the estimator asks nothing more of the
payoff than that it be measurable and square-integrable.

**Greeks are a different matter, and this engine refuses them.** The pathwise
method differentiates the payoff along the path,

    d/dS  cash * 1{S_T > K}  =  cash * delta_Dirac(S_T - K) * dS_T/dS,

which is not a function, so the pathwise estimator does not exist for a
digital -- and a finite-difference bump is not a repair either: with common
random numbers the bumped and unbumped payoffs differ only on the paths that
cross the strike, a fraction `O(h)` of them, each contributing a full `cash`,
so the difference quotient has variance `O(cash**2 / (N h))` and the estimator
is *inconsistent* as `h -> 0` at fixed `N`. The correct estimator is the
likelihood-ratio (score function) one, which differentiates the lognormal
density instead of the payoff and therefore does not care that the payoff is
discontinuous. That belongs to Phase 3 (Glasserman, *Monte Carlo Methods in
Financial Engineering*, chapter 7), and `greeks_digital` raises
`NotSupportedError` naming it rather than returning a number that looks like a
Greek.
"""

from __future__ import annotations

import math

import numpy as np

from ...exceptions import InvalidInputError, NotSupportedError
from ...instruments.options import DigitalOption
from ...instruments.payoffs import digital_payoff
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import GreeksResult, PriceResult
from .pricers import MCConfig
from .processes import price_european_from_terminal, simulate_gbm_exact

__all__ = ["greeks_digital", "price_digital"]


def price_digital(
    option: DigitalOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
) -> PriceResult:
    """Price a cash-or-nothing digital by Monte Carlo.

    Parameters
    ----------
    option, model, market
        As for `qpl.engines.mc.pricers.price_european`.
    cfg
        Monte Carlo settings. `n_steps = 1` is terminal sampling and is all a
        European digital needs; larger `n_steps` simulates the path and uses
        its endpoint, which is the same law and more work.

    Returns
    -------
    PriceResult
        Value and standard error, with `meta` reporting paths, steps and seed.
        Deterministic for a fixed seed.

    Raises
    ------
    InvalidInputError
        If `n_paths < 2` (the `ddof=1` standard error needs two samples) or
        `n_steps < 1`.
    """
    if cfg.n_paths < 2:
        raise InvalidInputError("n_paths must be >= 2 for MC stderr with ddof=1")
    if cfg.n_steps < 1:
        raise InvalidInputError("n_steps must be >= 1")

    s0 = market.spot
    t = option.expiry
    r = market.rate(t)
    q = market.dividend_yield(t)
    df_r = market.df_r(t)

    meta = {
        "method": "mc",
        "model": "BlackScholes",
        "instrument": "digital",
        "n_paths": cfg.n_paths,
        "n_steps": cfg.n_steps,
        "seed": cfg.seed,
    }

    if t == 0.0:
        value = float(digital_payoff(s0, option.strike, option.cash, option.kind))
        return PriceResult(value=value, stderr=0.0, meta=meta)

    if model.sigma == 0.0:
        forward = s0 * math.exp((r - q) * t)
        value = df_r * float(
            digital_payoff(forward, option.strike, option.cash, option.kind)
        )
        return PriceResult(value=value, stderr=0.0, meta=meta)

    paths = simulate_gbm_exact(
        s0=s0,
        mu=r - q,
        sigma=model.sigma,
        t=t,
        n_steps=cfg.n_steps,
        n_paths=cfg.n_paths,
        seed=cfg.seed,
    )
    estimate = price_european_from_terminal(
        paths[:, -1],
        strike=option.strike,
        discount_factor=df_r,
        kind=option.kind,
        payoff=lambda s_t: np.asarray(
            digital_payoff(s_t, option.strike, option.cash, option.kind), dtype=float
        ),
    )
    return PriceResult(value=estimate.value, stderr=estimate.stderr, meta=meta)


def greeks_digital(
    option: DigitalOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: MCConfig,
    bumps: dict[str, float] | None = None,
) -> GreeksResult:
    """Always raises: a digital has no pathwise Greek and no usable bump.

    Raises
    ------
    NotSupportedError
        Always. See the module docstring: the payoff's derivative is a Dirac
        mass, so the pathwise estimator does not exist, and a common-random-
        numbers bump has variance `O(1 / (N h))` and is inconsistent as the
        bump shrinks. The estimator that works is the likelihood-ratio one,
        scheduled for Phase 3.
    """
    raise NotSupportedError(
        "Monte Carlo Greeks are not available for a cash-or-nothing digital: "
        "the payoff's derivative is a Dirac mass, so there is no pathwise "
        "estimator, and a bump-and-revalue difference has variance O(1/(N h)) "
        "and does not converge as the bump shrinks. Use the likelihood-ratio "
        "(score function) estimator, scheduled for Phase 3 "
        "(docs/CURRICULUM.md); until then use method='analytic', 'pde' or "
        "'tree' for digital Greeks."
    )
