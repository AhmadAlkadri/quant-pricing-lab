from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...exceptions import InvalidInputError
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import PriceResult
from ..tree.lattice import build_recombining_spot_tree, crr_parameters
from .optimal_stopping import backward_induction_optimal_stopping


@dataclass(frozen=True)
class BinomialDPConfig:
    """Configuration for CRR binomial dynamic programming.

    Parameters
    ----------
    n_steps
        Number of time steps in the recombining lattice.
    """

    n_steps: int = 200


def price_american_put_binomial(
    *,
    strike: float,
    expiry: float,
    market: Market,
    model: BlackScholesModel,
    cfg: BinomialDPConfig,
    return_lattice: bool = False,
) -> PriceResult:
    """Price an American put with a CRR lattice and Bellman backward induction.

    Parameters
    ----------
    strike
        Put strike.
    expiry
        Time to maturity in years.
    market
        Market object for spot/rate/dividend inputs.
    model
        Black-Scholes model providing volatility.
    cfg
        Binomial dynamic-programming configuration.
    return_lattice
        If `True`, include spot/value/exercise lattices in result metadata.

    Returns
    -------
    PriceResult
        American put price and metadata (including early-exercise diagnostics).
    """
    if not math.isfinite(strike) or strike <= 0.0:
        raise InvalidInputError("strike must be finite and > 0")
    if not math.isfinite(expiry) or expiry < 0.0:
        raise InvalidInputError("expiry must be finite and >= 0")
    if cfg.n_steps < 1:
        raise InvalidInputError("n_steps must be >= 1")

    s0 = market.spot
    sigma = model.sigma

    if expiry == 0.0:
        intrinsic = max(strike - s0, 0.0)
        meta = {
            "method": "dp_binomial",
            "model": "BlackScholes",
            "instrument": "AmericanPut",
            "n_steps": cfg.n_steps,
            "dt": 0.0,
        }
        if return_lattice:
            spot_tree = [np.array([s0], dtype=float)]
            value_tree = [np.array([intrinsic], dtype=float)]
            exercise_policy = [np.array([True], dtype=bool)]
            exercise_values = [np.array([intrinsic], dtype=float)]
            meta.update(
                {
                    "spot_tree": spot_tree,
                    "value_tree": value_tree,
                    "exercise_values": exercise_values,
                    "exercise_policy": exercise_policy,
                }
            )
        return PriceResult(value=float(intrinsic), meta=meta)

    n_steps = cfg.n_steps
    r = market.rate(expiry)
    q = market.dividend_yield(expiry)

    # Shared with the European tree engine: one definition of "the CRR
    # lattice" for both. The arithmetic is unchanged from the inline version
    # this replaced, so prices are bit-identical.
    lattice = crr_parameters(
        sigma=sigma, expiry=expiry, rate=r, dividend_yield=q, n_steps=n_steps
    )
    dt = lattice.dt
    up = lattice.up
    down = lattice.down
    p = lattice.p
    discount = lattice.discount

    spot_tree = build_recombining_spot_tree(spot=s0, up=up, down=down, n_steps=n_steps)
    exercise_values = [np.maximum(strike - level_spots, 0.0) for level_spots in spot_tree]

    def _continuation_operator(_level: int, next_values: np.ndarray) -> np.ndarray:
        return discount * (p * next_values[1:] + (1.0 - p) * next_values[:-1])

    value_tree, exercise_policy = backward_induction_optimal_stopping(
        exercise_values=exercise_values,
        continuation_operator=_continuation_operator,
    )

    meta: dict[str, object] = {
        "method": "dp_binomial",
        "model": "BlackScholes",
        "instrument": "AmericanPut",
        "n_steps": n_steps,
        "dt": dt,
        "r": r,
        "q": q,
        "sigma": sigma,
        "u": up,
        "d": down,
        "p": p,
        "early_exercise_node_count": int(sum(np.count_nonzero(level) for level in exercise_policy[:-1])),
    }

    if return_lattice:
        meta.update(
            {
                "spot_tree": spot_tree,
                "value_tree": value_tree,
                "exercise_values": exercise_values,
                "exercise_policy": exercise_policy,
            }
        )

    return PriceResult(value=float(value_tree[0][0]), meta=meta)
