"""One down-and-out call on a grid, and the three things node placement decides.

A finite-difference grid is the one discretisation that can put a node on the
barrier for every time step at once. Slice 12 measured what happens when it
cannot: the lattice knocks out at the first node beyond the barrier and the
simulation observes it on a schedule, and both are order **one half** because
both displace the barrier by `O(sigma sqrt(dt))` and the price is locally
linear in the barrier level.

Three cases, one per placement decision.

`--case grid` is the convergence table. For each of

- `uniform` with the barrier as the domain boundary,
- `uniform` with the barrier rounded to the nodes
  (`PDEConfig(barrier_alignment="none")`),
- `sinh` concentrated at the barrier and the strike,

it prints the price, the error against the Reiner-Rubinstein closed form, the
local order between consecutive levels, and the effective barrier the scheme
actually priced. The first and third are order two; the second is not, and the
`effective_barrier` column says why.

`--case mesh` is the concentration scan: what `PDEConfig(concentration=...)`
buys on this contract, and what it costs on a plain vanilla, so that the
package default is visible as a compromise rather than as a best value.

`--case discrete` is the monitoring table. For `m` equally spaced observation
dates it prints the grid price of the *discrete* contract, its gap to the
continuous closed form, the Broadie-Glasserman-Kou prediction of that gap, and
their ratio -- the continuity correction checking a scheme that knows nothing
about it. It also prints the in-out parity residual on one grid, which is a
statement about the discrete linear system and not about the model.

Deterministic: there is no randomness anywhere in this example.

Run:  PYTHONPATH=src python examples/barrier_pde_grid.py
      PYTHONPATH=src python examples/barrier_pde_grid.py --case mesh
      PYTHONPATH=src python examples/barrier_pde_grid.py --case discrete
"""

from __future__ import annotations

import argparse
import math

from qpl.engines.analytic.barrier import (
    barrier_price,
    bgk_continuity_corrected_price,
)
from qpl.engines.pde.barrier import KNOCK_IN, KNOCK_OUT, VANILLA_LEG, solve_leg
from qpl.engines.pde.pricers import PDEConfig
from qpl.instruments.options import (
    BarrierOption,
    EuropeanOption,
    uniform_monitoring_times,
)
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel, bs_price
from qpl.pricing import price
from qpl.validation import fit_convergence_order

SPOT = 100.0
STRIKE = 100.0
EXPIRY = 0.5
RATE = 0.08
DIVIDEND = 0.04
SIGMA = 0.25
BARRIER = 95.0
BARRIER_TYPE = "down-and-out"
KIND = "call"

LEVELS = (100, 200, 400, 800)
CONCENTRATION = 0.05
CONCENTRATIONS = (0.02, 0.05, 0.10, 0.20)
MESH_LEVELS = (100, 400)
MONITORING_LEVELS = (10, 20, 40, 80, 160)
DISCRETE_N = 400
PARITY_LEVELS = (100, 400)

_GRID_CASES: tuple[tuple[str, dict[str, object]], ...] = (
    ("uniform_on_node", {}),
    ("uniform_off_node", {"barrier_alignment": "none"}),
    ("sinh_on_node", {"grid": "sinh", "concentration": CONCENTRATION}),
)


def _market() -> Market:
    return Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )


def _model() -> BlackScholesModel:
    return BlackScholesModel(sigma=SIGMA)


def _option(monitoring: int | None = None) -> BarrierOption:
    schedule = (
        "continuous"
        if monitoring is None
        else uniform_monitoring_times(EXPIRY, monitoring)
    )
    return BarrierOption(
        KIND, STRIKE, EXPIRY, BARRIER, BARRIER_TYPE, 0.0, schedule
    )


def _cfg(n: int, **kwargs: object) -> PDEConfig:
    base: dict[str, object] = {
        "n_s": n,
        "n_t": n,
        "strike_alignment": "midpoint",
        "time_stepping": "rannacher",
    }
    base.update(kwargs)
    return PDEConfig(**base)  # type: ignore[arg-type]


def _continuous() -> float:
    return barrier_price(
        S=SPOT, K=STRIKE, T=EXPIRY, r=RATE, sigma=SIGMA, q=DIVIDEND, H=BARRIER,
        rebate=0.0, barrier_type=BARRIER_TYPE, kind=KIND,
    )


def _order(levels, errors) -> float:
    return fit_convergence_order(
        [1.0 / n for n in levels], [abs(e) for e in errors]
    ).order


def _fit(levels, errors):
    return fit_convergence_order([1.0 / n for n in levels], [abs(e) for e in errors])


def _header(exact: float) -> None:
    print("example=barrier_pde_grid")
    print(
        f"spot={SPOT:.2f} strike={STRIKE:.2f} barrier={BARRIER:.2f} "
        f"type={BARRIER_TYPE} kind={KIND} expiry={EXPIRY:.2f}"
    )
    print(f"rate={RATE:.4f} dividend={DIVIDEND:.4f} sigma={SIGMA:.4f}")
    print(f"analytic_continuous={exact:.10f}")
    print("strike_alignment=midpoint time_stepping=rannacher")


def _grid_case(exact: float) -> None:
    print("case=grid")
    errors_by_case: dict[str, list[float]] = {}
    for label, kwargs in _GRID_CASES:
        errors: list[float] = []
        previous: float | None = None
        for n in LEVELS:
            result = price(
                _option(), _model(), _market(), method="pde", cfg=_cfg(n, **kwargs)
            )
            error = result.value - exact
            local = (
                math.nan
                if previous is None
                else math.log(abs(previous) / abs(error)) / math.log(2.0)
            )
            previous = error
            errors.append(error)
            effective = float(result.meta["effective_barrier"])
            print(
                f"grid {label:17s} n={n:5d}"
                f" price={result.value:.10f}"
                f" error={error:+.5e}"
                f" local_order={local:+.4f}"
                f" effective_barrier={effective:.6f}"
                f" on_node={str(result.meta['barrier_on_node']):5s}"
                f" ds_min={float(result.meta['ds_min']):.5f}"
            )
        errors_by_case[label] = errors
        fit = _fit(LEVELS, errors)
        print(
            f"grid_order {label}={fit.order:+.4f} residual={fit.residual:.4f}"
        )

    on_node = errors_by_case["uniform_on_node"]
    off_node = errors_by_case["uniform_off_node"]
    concentrated = errors_by_case["sinh_on_node"]
    print(
        "off_node_error_times_n="
        + " ".join(f"{abs(e) * n:.2f}" for e, n in zip(off_node, LEVELS, strict=True))
    )
    print(
        "off_node_over_on_node="
        + " ".join(
            f"{abs(o) / abs(a):.1f}" for o, a in zip(off_node, on_node, strict=True)
        )
    )
    print(
        "uniform_over_sinh="
        + " ".join(
            f"{abs(u) / abs(c):.1f}"
            for u, c in zip(on_node, concentrated, strict=True)
        )
    )


def _mesh_case(exact: float) -> None:
    """What the mesh buys on a barrier, and what it costs on a vanilla.

    Both are printed at two node counts, so that the reader can see the ratios
    are a property of the mesh rather than of one lucky grid.
    """
    print("case=mesh")
    vanilla = EuropeanOption("call", STRIKE, EXPIRY)
    vanilla_exact = bs_price(
        S=SPOT, K=STRIKE, T=EXPIRY, r=RATE, sigma=SIGMA, q=DIVIDEND, kind="call"
    )

    def barrier_error(n: int, **kwargs: object) -> float:
        return abs(
            price(
                _option(), _model(), _market(), method="pde", cfg=_cfg(n, **kwargs)
            ).value
            - exact
        )

    def vanilla_error(n: int, **kwargs: object) -> float:
        return abs(
            price(
                vanilla, _model(), _market(), method="pde", cfg=_cfg(n, **kwargs)
            ).value
            - vanilla_exact
        )

    baselines = {n: (barrier_error(n), vanilla_error(n)) for n in MESH_LEVELS}
    for n in MESH_LEVELS:
        barrier_base, vanilla_base = baselines[n]
        print(
            f"uniform_baseline n={n:5d}"
            f" barrier_error={barrier_base:.5e}"
            f" vanilla_error={vanilla_base:.5e}"
        )
    for concentration in CONCENTRATIONS:
        kwargs = {"grid": "sinh", "concentration": concentration}
        for n in MESH_LEVELS:
            barrier_base, vanilla_base = baselines[n]
            on_barrier = barrier_error(n, **kwargs)
            on_vanilla = vanilla_error(n, **kwargs)
            print(
                f"mesh concentration={concentration:.2f} n={n:5d}"
                f" barrier_error={on_barrier:.5e}"
                f" barrier_gain={barrier_base / on_barrier:6.2f}"
                f" vanilla_error={on_vanilla:.5e}"
                f" vanilla_gain={vanilla_base / on_vanilla:6.2f}"
            )
    print(f"default_concentration={CONCENTRATION:.2f}")
    print(
        "mesh_reading=the mesh pays on the barrier and costs on the vanilla; "
        "it is a placement remedy, not free accuracy"
    )


def _discrete_case(exact: float) -> None:
    print("case=discrete")
    kwargs = {"grid": "sinh", "concentration": CONCENTRATION}
    gaps, predicted = [], []
    for m in MONITORING_LEVELS:
        result = price(
            _option(monitoring=m),
            _model(),
            _market(),
            method="pde",
            cfg=_cfg(DISCRETE_N, **kwargs),
        )
        shift = bgk_continuity_corrected_price(
            S=SPOT, K=STRIKE, T=EXPIRY, r=RATE, sigma=SIGMA, q=DIVIDEND,
            H=BARRIER, rebate=0.0, barrier_type=BARRIER_TYPE, kind=KIND,
            n_monitoring=m,
        )
        gaps.append(result.value - exact)
        predicted.append(shift - exact)
        print(
            f"discrete m={m:4d}"
            f" pde={result.value:.6f}"
            f" gap={gaps[-1]:+.6f}"
            f" bgk_gap={predicted[-1]:+.6f}"
            f" gap_over_bgk={gaps[-1] / predicted[-1]:.4f}"
            f" projections={int(result.meta['n_projections'])}"
            f" steps={int(result.meta['n_steps_taken'])}"
        )
    print(f"discrete_gap_order={_order(MONITORING_LEVELS, gaps):+.4f}")
    print(f"bgk_gap_order={_order(MONITORING_LEVELS, predicted):+.4f}")

    vanilla_value = bs_price(
        S=SPOT, K=STRIKE, T=EXPIRY, r=RATE, sigma=SIGMA, q=DIVIDEND, kind="call"
    )
    for n in PARITY_LEVELS:
        cfg = _cfg(n, **kwargs)
        option, model, market = _option(), _model(), _market()
        out = solve_leg(option, model, market, cfg, leg=KNOCK_OUT).price
        inside = solve_leg(option, model, market, cfg, leg=KNOCK_IN).price
        both = solve_leg(option, model, market, cfg, leg=VANILLA_LEG).price
        print(
            f"parity n={n:5d}"
            f" knock_out={out:.10f} knock_in={inside:.10f} summed_leg={both:.10f}"
            f" residual={out + inside - both:+.3e}"
            f" relative={abs(out + inside - both) / both:.3e}"
            f" summed_minus_black_scholes={both - vanilla_value:+.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("grid", "mesh", "discrete"), default="grid"
    )
    args = parser.parse_args()

    exact = _continuous()
    _header(exact)
    if args.case == "grid":
        _grid_case(exact)
    elif args.case == "mesh":
        _mesh_case(exact)
    else:
        _discrete_case(exact)


if __name__ == "__main__":
    main()
