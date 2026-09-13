"""European option pricing and Greeks on a Cox-Ross-Rubinstein binomial tree.

The lattice itself lives in `qpl.engines.tree.lattice`; this module adds the
terminal payoff, the backward induction, and the lattice Greek estimators.

Assumptions
-----------
- Flat curves: the rate and dividend yield are read from `Market` once, at the
  option's expiry, and held constant over every step of the tree.
- Black-Scholes dynamics: the lattice is the CRR discretisation of geometric
  Brownian motion with constant volatility.

Accuracy
--------
The scheme is first order in ``1 / n`` with a coefficient that oscillates with
the parity of ``n``, because the strike's position between the two terminal
nodes that straddle it changes as ``n`` increments. See
`docs/notes/crr_tree_convergence.md` and Leisen and Reimer (1996), "Binomial
models for option valuation -- examining and improving convergence", Applied
Mathematical Finance 3(4), 319-346, which both establishes the order-1
behaviour and constructs the tree that removes the oscillation (a later
slice; not implemented here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...exceptions import InvalidInputError
from ...instruments.options import EuropeanOption
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import PriceResult
from ..registry import MethodSpec
from .lattice import CRRLattice, crr_parameters, crr_spot_level

__all__ = [
    "TREE_METHOD_SPEC",
    "TreeConfig",
    "price_european",
]

@dataclass(frozen=True)
class TreeConfig:
    """Binomial lattice configuration.

    Parameters
    ----------
    n_steps
        Number of time steps in the lattice. Price error decays like
        ``1 / n_steps`` with an oscillating coefficient; Greeks read off the
        lattice inherit that oscillation.
    scheme
        Lattice parameterisation. Only ``"crr"`` exists so far; the field is
        present because Leisen-Reimer is a planned sibling and adding it must
        not change this config's identity.
    """

    n_steps: int = 200
    scheme: Literal["crr"] = "crr"


TREE_METHOD_SPEC = MethodSpec(method="tree", cfg_type=TreeConfig)
"""Keyword contract for `method="tree"`: a required `TreeConfig`, nothing else."""


def _validate(cfg: TreeConfig, *, min_steps: int) -> None:
    if cfg.scheme != "crr":
        raise InvalidInputError("scheme must be 'crr'")
    if cfg.n_steps < min_steps:
        raise InvalidInputError(f"n_steps must be >= {min_steps}")


def _payoff(spots: np.ndarray, *, kind: str, strike: float) -> np.ndarray:
    if kind == "call":
        return np.maximum(spots - strike, 0.0)
    return np.maximum(strike - spots, 0.0)


def _degenerate_value(option: EuropeanOption, market: Market) -> float:
    """Price at ``T = 0`` or at ``sigma = 0``.

    Matches the analytic, MC and PDE engines: intrinsic value at expiry, and
    the discounted intrinsic value of the forward when the spot is
    deterministic.
    """
    s0 = market.spot
    k = option.strike
    t = option.expiry

    if t == 0.0:
        if option.kind == "call":
            return max(s0 - k, 0.0)
        return max(k - s0, 0.0)

    r = market.rate(t)
    q = market.dividend_yield(t)
    forward = s0 * math.exp((r - q) * t)
    disc = market.df_r(t)
    if option.kind == "call":
        return disc * max(forward - k, 0.0)
    return disc * max(k - forward, 0.0)


def _meta(
    lattice: CRRLattice | None, cfg: TreeConfig, *, degenerate: str | None
) -> dict[str, object]:
    meta: dict[str, object] = {
        "method": "tree",
        "model": "BlackScholes",
        "scheme": cfg.scheme,
        "n_steps": cfg.n_steps,
    }
    if lattice is None:
        meta.update({"dt": 0.0, "u": 1.0, "d": 1.0, "p": 0.5})
    else:
        meta.update(
            {
                "dt": lattice.dt,
                "u": lattice.up,
                "d": lattice.down,
                "p": lattice.p,
            }
        )
    if degenerate is not None:
        meta["degenerate"] = degenerate
    return meta


def _backward_induction(
    option: EuropeanOption,
    market: Market,
    lattice: CRRLattice,
    *,
    capture: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Roll the terminal payoff back to the root, keeping the named levels.

    The payoff vector at one time level is updated in a single numpy
    expression, so the only Python-level loop is over time levels:
    ``V_j = e^{-r dt} (p V_{j+1}[1:] + (1 - p) V_{j+1}[:-1])``, where index
    ``j`` counts up moves.
    """
    n = lattice.n_steps
    values = _payoff(
        crr_spot_level(spot=market.spot, up=lattice.up, down=lattice.down, level=n),
        kind=option.kind,
        strike=option.strike,
    )

    kept: dict[int, np.ndarray] = {}
    if n in capture:
        kept[n] = values.copy()

    disc, p = lattice.discount, lattice.p
    for level in range(n - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if level in capture:
            kept[level] = values.copy()

    return kept


def price_european(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    *,
    cfg: TreeConfig,
) -> PriceResult:
    """Price a European option by backward induction on a CRR binomial tree.

    Parameters
    ----------
    option
        European option (`call` or `put`).
    model
        Black-Scholes model with constant volatility.
    market
        Market object providing spot and flat rate/dividend curves. The rate
        and dividend yield are sampled once, at the option's expiry.
    cfg
        Lattice settings.

    Returns
    -------
    PriceResult
        Price, with metadata reporting the method, scheme, `n_steps`, and the
        realised lattice parameters `u`, `d`, `p`, `dt`.

    Raises
    ------
    InvalidInputError
        If `cfg` is out of range, or if the lattice violates the no-arbitrage
        condition (see `qpl.engines.tree.lattice.crr_parameters`).
    """
    _validate(cfg, min_steps=1)

    t = option.expiry
    if t == 0.0 or model.sigma == 0.0:
        value = _degenerate_value(option, market)
        degenerate = "expiry" if t == 0.0 else "zero_vol"
        lattice = (
            None
            if t == 0.0
            else crr_parameters(
                sigma=0.0,
                expiry=t,
                rate=market.rate(t),
                dividend_yield=market.dividend_yield(t),
                n_steps=cfg.n_steps,
            )
        )
        return PriceResult(value=float(value), meta=_meta(lattice, cfg, degenerate=degenerate))

    lattice = crr_parameters(
        sigma=model.sigma,
        expiry=t,
        rate=market.rate(t),
        dividend_yield=market.dividend_yield(t),
        n_steps=cfg.n_steps,
    )
    kept = _backward_induction(option, market, lattice, capture=(0,))
    return PriceResult(value=float(kept[0][0]), meta=_meta(lattice, cfg, degenerate=None))
