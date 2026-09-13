"""The same American put on a lattice, a PSOR grid, and a path sample.

Prints, for one specification:

- the American value from each engine and the gap between them;
- the European value on the same PDE grid and the early-exercise premium;
- what the PSOR solve cost (sweeps per time step) and that the linear
  complementarity conditions actually hold;
- the least-squares Monte Carlo value in both its estimators, in-sample and
  out-of-sample, each with its standard error, next to the two deterministic
  engines and next to the lattice Bermudan it is actually estimating;
- the exercise boundary from all three engines at a few times, next to the
  node spacing that limits how well any of them can locate it.

The point of the comparison is that the three engines have nothing in common
except the model: a binomial lattice with geometric node spacing and a Bellman
maximum at every node; a grid uniform in spot with the exercise condition
imposed as an LCP solved inside each time step; and a sample of paths on which
the continuation value is *regressed* rather than computed. The first two agree
to 1.4e-03 on a value of 6.09 -- with each one's error against the converged
value printed next to it, -1.5e-03 and -1.4e-04 -- and agreeing bit for bit
would have meant they were the same thing twice.

The simulation agrees to about 1%, and the three lines that explain why are
printed rather than buried: it prices a **Bermudan** option (`bermudan_gap`),
its policy is fitted on a finite sample and is therefore suboptimal
(`lsm_finite_sample_bias`), and it has a standard error. Read in that order the
column is evidence; read as "6.06 against 6.09" it looks like a bad engine.

Deterministic: the two grid engines use no randomness, and the Monte Carlo
column is a pure function of the printed seed.

Run:  PYTHONPATH=src python examples/american_put_cross_method.py
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from qpl.cases import AmericanBSSpec, bermudan_value_on_lattice
from qpl.engines.mc.pricers import MCConfig
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

LSM_PATHS = 100_000
LSM_DATES = 50
LSM_DEGREE = 3
LSM_SEED = 7
"""Monte Carlo column. 50 exercise dates rather than 250 keeps the run near a
second while leaving the Bermudan gap (1.2e-02) comfortably above the printed
standard error, so the two corrections can be told apart in the output."""

LSM_LATTICE_N_STEPS = 5_000
"""Lattice size for the Bermudan reference the simulation is really estimating.
Divisible by `LSM_DATES`, so the exercise dates land on lattice levels."""

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

    lsm_cfg = MCConfig(
        n_paths=LSM_PATHS,
        seed=LSM_SEED,
        variance_reduction="antithetic",
        exercise_dates=LSM_DATES,
        lsm_basis="laguerre",
        lsm_degree=LSM_DEGREE,
    )

    pde = price(american, model, market, method="pde", cfg=pde_cfg)
    tree = price(american, model, market, method="tree", cfg=tree_cfg)
    pde_european = price(european, model, market, method="pde", cfg=pde_cfg)
    lsm = price(american, model, market, method="mc", cfg=lsm_cfg)
    lsm_in_sample = price(
        american, model, market, method="mc",
        cfg=replace(lsm_cfg, lsm_in_sample=True),
    )

    pde_meta = pde.meta or {}
    tree_meta = tree.meta or {}
    lsm_meta = lsm.meta or {}

    print("example=american_put_cross_method")
    print(f"spot={SPOT:.2f} strike={STRIKE:.2f} expiry={EXPIRY:.2f}")
    print(f"rate={RATE:.4f} dividend={DIVIDEND:.4f} sigma={SIGMA:.4f}")
    print(f"pde_grid n_s={PDE_N} n_t={PDE_N} alignment=midpoint time_stepping=rannacher")
    print(f"tree_grid n_steps={TREE_N_STEPS} scheme=leisen-reimer")
    print(
        f"lsm_settings n_paths={LSM_PATHS} exercise_dates={LSM_DATES} "
        f"basis=laguerre degree={LSM_DEGREE} seed={LSM_SEED} antithetic=True"
    )

    # ---- prices -------------------------------------------------------
    print(f"pde_american={pde.value:.8f}")
    print(f"tree_american={tree.value:.8f}")
    print(f"engine_gap={abs(pde.value - tree.value):.3e}")
    print(f"pde_error_vs_limit={pde.value - BRACKETED_LIMIT:+.3e}")
    print(f"tree_error_vs_limit={tree.value - BRACKETED_LIMIT:+.3e}")
    print(f"pde_european={pde_european.value:.8f}")
    print(f"early_exercise_premium={pde.value - pde_european.value:.8f}")

    # ---- the third discretisation: a sample, not a mesh ----------------
    #
    # Printed in the order the corrections have to be applied. The simulation
    # does not estimate `tree_american`: it estimates the Bermudan value on its
    # own exercise grid, which is `lsm_lattice_bermudan` and is below the
    # continuous value by `bermudan_gap`. What is left after that correction is
    # the estimator's own error -- a low bias from a policy fitted on finitely
    # many paths, plus noise of size `lsm_stderr`.
    bermudan = bermudan_value_on_lattice(
        AmericanBSSpec(SPOT, STRIKE, EXPIRY, RATE, DIVIDEND, SIGMA, "put"),
        n_steps=LSM_LATTICE_N_STEPS,
        n_exercise=LSM_DATES,
    )
    print(f"lsm_out_of_sample={lsm.value:.8f} stderr={lsm.stderr:.2e}")
    print(f"lsm_in_sample={lsm_in_sample.value:.8f} stderr={lsm_in_sample.stderr:.2e}")
    print(f"lsm_lattice_bermudan={bermudan:.8f}")
    print(f"bermudan_gap={BRACKETED_LIMIT - bermudan:.3e}")
    print(f"lsm_finite_sample_bias={lsm.value - bermudan:+.3e}")
    print(f"lsm_vs_tree={lsm.value - tree.value:+.3e}")
    print(f"lsm_z_vs_bermudan={(lsm.value - bermudan) / lsm.stderr:+.2f}")
    print(f"lsm_early_exercise_fraction={lsm_meta['early_exercise_fraction']:.4f}")
    print(f"lsm_regressions={lsm_meta['n_regressions']}")
    print(f"lsm_condition_median={lsm_meta['regression_condition_median']:.2e}")
    print(f"lsm_condition_max={lsm_meta['regression_condition_max']:.2e}")

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
    lsm_boundary = np.asarray(lsm_meta["exercise_boundary"], dtype=float)
    lsm_times = np.asarray(lsm_meta["exercise_boundary_times"], dtype=float)

    node_gap = SPOT * (1.0 - 1.0 / tree_meta["u"])
    print(f"pde_node_spacing={pde_meta['ds']:.4f} tree_node_spacing={node_gap:.4f}")
    # Neither engine can locate the boundary better than its own node spacing,
    # so the combined spacing is the scale at which the two columns below can
    # be expected to agree.
    print(f"combined_node_spacing={pde_meta['ds'] + node_gap:.4f}")

    # The LSM boundary is the largest sampled spot at which exercising beat the
    # fitted continuation. It is an upper order statistic of a sample, so it is
    # biased upward, and it carries the regression's error at exactly the place
    # where intrinsic and continuation nearly coincide. Two per cent of the
    # strike is what it is worth, and the column is printed so that is visible.
    for t in REPORT_TIMES:
        i = int(np.argmin(np.abs(pde_times - t)))
        j = int(np.argmin(np.abs(tree_times - t)))
        k = int(np.argmin(np.abs(lsm_times - t)))
        tree_value = tree_boundary[j]
        shown = "nan" if math.isnan(tree_value) else f"{tree_value:.4f}"
        # `t = 0` is not an exercise date for the Bermudan the simulation
        # prices -- its grid is `t_i = i T / m` for `i >= 1` -- so the column
        # says so instead of quoting the nearest date as if it were this one.
        lsm_value = lsm_boundary[k]
        if t < lsm_times[0]:
            lsm_shown = "n/a"
        else:
            lsm_shown = "nan" if math.isnan(lsm_value) else f"{lsm_value:.4f}"
        print(
            f"boundary t={t:.2f} pde={pde_boundary[i]:.4f} tree={shown} "
            f"lsm={lsm_shown}"
        )

    # The lattice has no boundary at all early on: no node sits low enough for
    # exercise to be optimal yet. The grid reaches down to S = 0 at every time
    # level, so it always has one.
    tree_first = int(np.flatnonzero(~np.isnan(tree_boundary))[0])
    print(f"tree_boundary_first_time={tree_times[tree_first]:.4f}")
    print(f"pde_boundary_first_time={pde_times[0]:.4f}")


if __name__ == "__main__":
    main()
