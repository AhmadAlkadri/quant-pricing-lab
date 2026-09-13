"""The same American put on a lattice and on a PSOR finite-difference grid.

Prints, for one specification:

- the American value from each engine and the gap between them;
- the European value on the same PDE grid and the early-exercise premium;
- what the PSOR solve cost (sweeps per time step) and that the linear
  complementarity conditions actually hold;
- the exercise boundary from both engines at a few times, next to the node
  spacing that limits how well either can locate it.

The point of the comparison is that the two engines have nothing in common
except the model: a binomial lattice with geometric node spacing and a Bellman
maximum at every node, against a grid uniform in spot with the exercise
condition imposed as an LCP solved inside each time step. Agreeing to 1.4e-03
on a value of 6.09 -- with each engine's own error against the converged value
printed next to it, -1.5e-03 and -1.4e-04 -- is evidence; agreeing bit for bit
would have meant they were the same thing twice.

Deterministic: neither engine uses randomness, and every number below is a
function of the printed inputs alone.

Run:  PYTHONPATH=src python examples/american_put_cross_method.py
"""

from __future__ import annotations

import math

import numpy as np

from qpl.engines.pde.pricers import PDEConfig
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
DIVIDEND = 0.0
SIGMA = 0.20

PDE_N = 400
"""`n_s = n_t`, aligned and Rannacher-damped. About 0.07 s."""

TREE_N_STEPS = 2001
"""Odd, so the Leisen-Reimer scheme accepts it. About 0.04 s."""

BRACKETED_LIMIT = 6.090376463020103
"""The converged value, from `qpl.cases.AMERICAN_BRACKETED_LIMIT`: the average
of the CRR lattice at n = 64000 and n = 64001, which bracket it."""

REPORT_TIMES = (0.00, 0.25, 0.50, 0.75, 0.95)


def main() -> None:
    market = Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )
    model = BlackScholesModel(sigma=SIGMA)
    american = AmericanOption(kind="put", strike=STRIKE, expiry=EXPIRY)
    european = EuropeanOption(kind="put", strike=STRIKE, expiry=EXPIRY)

    pde_cfg = PDEConfig(
        n_s=PDE_N, n_t=PDE_N, strike_alignment="midpoint", time_stepping="rannacher"
    )
    tree_cfg = TreeConfig(n_steps=TREE_N_STEPS, scheme="leisen-reimer")

    pde = price(american, model, market, method="pde", cfg=pde_cfg)
    tree = price(american, model, market, method="tree", cfg=tree_cfg)
    pde_european = price(european, model, market, method="pde", cfg=pde_cfg)

    pde_meta = pde.meta or {}
    tree_meta = tree.meta or {}

    print("example=american_put_cross_method")
    print(f"spot={SPOT:.2f} strike={STRIKE:.2f} expiry={EXPIRY:.2f}")
    print(f"rate={RATE:.4f} dividend={DIVIDEND:.4f} sigma={SIGMA:.4f}")
    print(f"pde_grid n_s={PDE_N} n_t={PDE_N} alignment=midpoint time_stepping=rannacher")
    print(f"tree_grid n_steps={TREE_N_STEPS} scheme=leisen-reimer")

    # ---- prices -------------------------------------------------------
    print(f"pde_american={pde.value:.8f}")
    print(f"tree_american={tree.value:.8f}")
    print(f"engine_gap={abs(pde.value - tree.value):.3e}")
    print(f"pde_error_vs_limit={pde.value - BRACKETED_LIMIT:+.3e}")
    print(f"tree_error_vs_limit={tree.value - BRACKETED_LIMIT:+.3e}")
    print(f"pde_european={pde_european.value:.8f}")
    print(f"early_exercise_premium={pde.value - pde_european.value:.8f}")

    # ---- what the solve cost, and that the LCP holds -------------------
    print(f"psor_omega={pde_meta['psor_omega']:.2f} psor_tol={pde_meta['psor_tol']:.0e}")
    print(f"psor_mean_sweeps={pde_meta['psor_iterations_mean']:.2f}")
    print(f"psor_max_sweeps={pde_meta['psor_iterations_max']}")
    print(f"psor_converged={pde_meta['psor_converged']}")
    # At every interior node and every time step, either V - payoff or the
    # operator residual is zero. This is the complementarity condition, and it
    # is measured during the solve rather than asserted afterwards.
    print(f"lcp_max_complementarity={pde_meta['lcp_max_complementarity']:.3e}")
    print(f"lcp_min_constraint_slack={pde_meta['lcp_min_constraint_slack']:.3e}")

    # ---- the free boundary, from both engines -------------------------
    pde_boundary = np.asarray(pde_meta["exercise_boundary"], dtype=float)
    pde_times = np.asarray(pde_meta["exercise_boundary_times"], dtype=float)
    tree_boundary = np.asarray(tree_meta["exercise_boundary"], dtype=float)
    tree_times = np.linspace(0.0, EXPIRY, TREE_N_STEPS + 1)

    node_gap = SPOT * (1.0 - 1.0 / tree_meta["u"])
    print(f"pde_node_spacing={pde_meta['ds']:.4f} tree_node_spacing={node_gap:.4f}")
    # Neither engine can locate the boundary better than its own node spacing,
    # so the combined spacing is the scale at which the two columns below can
    # be expected to agree.
    print(f"combined_node_spacing={pde_meta['ds'] + node_gap:.4f}")

    for t in REPORT_TIMES:
        i = int(np.argmin(np.abs(pde_times - t)))
        j = int(np.argmin(np.abs(tree_times - t)))
        tree_value = tree_boundary[j]
        shown = "nan" if math.isnan(tree_value) else f"{tree_value:.4f}"
        print(f"boundary t={t:.2f} pde={pde_boundary[i]:.4f} tree={shown}")

    # The lattice has no boundary at all early on: no node sits low enough for
    # exercise to be optimal yet. The grid reaches down to S = 0 at every time
    # level, so it always has one.
    tree_first = int(np.flatnonzero(~np.isnan(tree_boundary))[0])
    print(f"tree_boundary_first_time={tree_times[tree_first]:.4f}")
    print(f"pde_boundary_first_time={pde_times[0]:.4f}")


if __name__ == "__main__":
    main()
