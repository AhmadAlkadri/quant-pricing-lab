"""Compatibility wrapper for the retired keyword-based American-put entry point.

The American put is now priced through the dispatcher, because exercise style
is a property of the instrument rather than a separate engine::

    price(AmericanOption(kind="put", strike=K, expiry=T), model, market,
          method="tree", cfg=TreeConfig(n_steps=n))

`price_american_put_binomial` below takes `strike`/`expiry` instead of an
instrument, which is exactly why it could never be dispatched on. It is kept
as a thin wrapper -- the value comes from `qpl.engines.tree.price_american`,
not from a second copy of the numerics -- so that lab notebooks written against
the Slice 1 signature keep running. New code should use the dispatcher.

The one thing this wrapper still computes itself is the optional full-lattice
payload (`return_lattice=True`), which the engine deliberately does not return:
three `O(n**2)` arrays are a teaching aid for a notebook, not something a
pricing call should allocate. It is built here from the two primitives this
package keeps -- the shared CRR lattice and
`backward_induction_optimal_stopping` -- and the wrapper asserts nothing about
it beyond shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...instruments.options import AmericanOption
from ...market.market import Market
from ...models.black_scholes import BlackScholesModel
from ..base import PriceResult
from ..tree.american import price_american
from ..tree.lattice import build_recombining_spot_tree, crr_parameters
from ..tree.pricers import TreeConfig
from .optimal_stopping import backward_induction_optimal_stopping

__all__ = ["BinomialDPConfig", "price_american_put_binomial"]


@dataclass(frozen=True)
class BinomialDPConfig:
    """Configuration for CRR binomial dynamic programming.

    Deprecated alias for `qpl.engines.tree.TreeConfig`, kept for the wrapper
    below.

    Parameters
    ----------
    n_steps
        Number of time steps in the recombining lattice.
    """

    n_steps: int = 200


def _lattice_payload(
    *,
    strike: float,
    expiry: float,
    market: Market,
    model: BlackScholesModel,
    n_steps: int,
) -> dict[str, object]:
    """Spot / value / exercise lattices, for plotting."""
    lattice = crr_parameters(
        sigma=model.sigma,
        expiry=expiry,
        rate=market.rate(expiry),
        dividend_yield=market.dividend_yield(expiry),
        n_steps=n_steps,
    )
    spot_tree = build_recombining_spot_tree(
        spot=market.spot, up=lattice.up, down=lattice.down, n_steps=n_steps
    )
    exercise_values = [np.maximum(strike - level, 0.0) for level in spot_tree]

    def _continuation(_level: int, next_values: np.ndarray) -> np.ndarray:
        return lattice.discount * (
            lattice.p * next_values[1:] + (1.0 - lattice.p) * next_values[:-1]
        )

    value_tree, exercise_policy = backward_induction_optimal_stopping(
        exercise_values=exercise_values,
        continuation_operator=_continuation,
    )
    return {
        "spot_tree": spot_tree,
        "value_tree": value_tree,
        "exercise_values": exercise_values,
        "exercise_policy": exercise_policy,
    }


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

    Deprecated: use
    ``price(AmericanOption(kind="put", ...), model, market, method="tree",
    cfg=TreeConfig(n_steps=...))``. The value returned here is the value that
    call returns, bit for bit.

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
    option = AmericanOption(kind="put", strike=strike, expiry=expiry)
    result = price_american(option, model, market, cfg=TreeConfig(n_steps=cfg.n_steps))

    engine_meta = result.meta or {}
    meta: dict[str, object] = {
        "method": "dp_binomial",
        "model": "BlackScholes",
        "instrument": "AmericanPut",
        "n_steps": cfg.n_steps,
        "dt": engine_meta.get("dt"),
        "r": market.rate(expiry) if expiry > 0.0 else 0.0,
        "q": market.dividend_yield(expiry) if expiry > 0.0 else 0.0,
        "sigma": model.sigma,
        "u": engine_meta.get("u"),
        "d": engine_meta.get("d"),
        "p": engine_meta.get("p"),
        "early_exercise_node_count": engine_meta.get("early_exercise_node_count", 0),
    }

    if return_lattice:
        if expiry == 0.0:
            intrinsic = max(strike - market.spot, 0.0)
            meta.update(
                {
                    "spot_tree": [np.array([market.spot], dtype=float)],
                    "value_tree": [np.array([intrinsic], dtype=float)],
                    "exercise_values": [np.array([intrinsic], dtype=float)],
                    "exercise_policy": [np.array([True], dtype=bool)],
                }
            )
        else:
            meta.update(
                _lattice_payload(
                    strike=strike,
                    expiry=expiry,
                    market=market,
                    model=model,
                    n_steps=cfg.n_steps,
                )
            )

    return PriceResult(value=result.value, meta=meta)
