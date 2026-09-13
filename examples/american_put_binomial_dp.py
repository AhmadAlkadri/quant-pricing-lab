"""American put on a CRR binomial tree, through the pricing dispatcher.

Prints the American and European values at the same specification, the
early-exercise premium between them, and a few points of the extracted
exercise boundary. Deterministic: the tree has no randomness and every number
below is a function of the printed inputs alone.

Run:  PYTHONPATH=src python examples/american_put_binomial_dp.py
"""

from __future__ import annotations

import math

import numpy as np

from qpl.engines.tree import TreeConfig
from qpl.instruments.options import AmericanOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price

SPOT = 100.0
STRIKE = 100.0
EXPIRY = 1.0
RATE = 0.05
DIVIDEND = 0.01
SIGMA = 0.20
N_STEPS = 200


def main() -> None:
    market = Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )
    model = BlackScholesModel(sigma=SIGMA)
    cfg = TreeConfig(n_steps=N_STEPS)

    # Exercise style is a property of the instrument: the same method, the same
    # config, a different instrument type, and the registry picks the engine.
    american = price(
        AmericanOption(kind="put", strike=STRIKE, expiry=EXPIRY),
        model,
        market,
        method="tree",
        cfg=cfg,
    )
    european = price(
        EuropeanOption(kind="put", strike=STRIKE, expiry=EXPIRY),
        model,
        market,
        method="tree",
        cfg=cfg,
    )

    meta = american.meta or {}
    boundary = np.asarray(meta["exercise_boundary"], dtype=float)
    times = np.linspace(0.0, EXPIRY, N_STEPS + 1)

    print("example=american_put_binomial_dp")
    print(f"spot={SPOT:.2f} strike={STRIKE:.2f} expiry={EXPIRY:.2f}")
    print(f"rate={RATE:.4f} dividend={DIVIDEND:.4f} sigma={SIGMA:.4f} n_steps={N_STEPS}")
    print(f"american_put={american.value:.6f}")
    print(f"european_put={european.value:.6f}")
    print(f"early_exercise_premium={american.value - european.value:.6f}")
    print(f"early_exercise_nodes={int(meta['early_exercise_node_count'])}")

    # The boundary is NaN early in the tree, where no node is low enough for
    # exercise to be optimal yet; report where it starts and how it rises.
    finite = np.flatnonzero(~np.isnan(boundary))
    print(f"boundary_first_time={times[finite[0]]:.4f}")
    for level in (finite[0], N_STEPS // 2, 3 * N_STEPS // 4, N_STEPS - 2, N_STEPS):
        value = boundary[level]
        shown = "nan" if math.isnan(value) else f"{value:.4f}"
        print(f"boundary t={times[level]:.4f} spot={shown}")


if __name__ == "__main__":
    main()
