"""One down-and-out call, and the two places a barrier gets discretised.

A barrier contract has a level the underlying must not reach, and every
numerical method has to decide *when* it checks. Monte Carlo checks on the
contract's monitoring dates; a lattice checks at its own time levels and can
only knock out at a node. Both substitute an **effective barrier** for the
contractual one, displaced by `O(sigma sqrt(dt))`, and because the price is
locally linear in the barrier level both errors are `O(dt**1/2)` -- half an
order, not one.

Two cases, one per discretisation.

`--case monitoring` prints, for `m` equally spaced monitoring dates:

- the plain Monte Carlo estimate, which is unbiased for the **discretely
  monitored** contract and therefore biased against the continuous closed form;
- the Broadie-Glasserman-Kou barrier shift, which moves the simulated barrier
  toward the spot by `exp(-+ beta sigma sqrt(dt))`, `beta = -zeta(1/2)/sqrt(2 pi)`;
- the Brownian-bridge estimator, which does not kill paths at all but weights
  each by its conditional survival probability, and is unbiased for the
  **continuous** contract at every `m` -- including `m = 1`;
- the BGK closed form, which is the same correction read the other way round:
  an approximation of the discrete price from the continuous formula.

`--case sawtooth` prints the lattice error over consecutive step counts, and
the Boyle-Lau step counts `n_k = floor(k**2 sigma**2 T / log(S/H)**2)` that put
a layer of nodes on the barrier. The sawtooth's *period* in `n` grows like
`sqrt(n)`, which is why the windows widen: a fixed-width window would report
decay that is only the period stretching.

Deterministic: every Monte Carlo leg is seeded and the lattice legs use no
randomness.

Run:  PYTHONPATH=src python examples/barrier_option_monitoring_bias.py
      PYTHONPATH=src python examples/barrier_option_monitoring_bias.py --case sawtooth
"""

from __future__ import annotations

import argparse
import math

from qpl.engines.analytic.barrier import (
    BGK_BETA,
    barrier_price,
    bgk_continuity_corrected_price,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.tree import TreeConfig, boyle_lau_steps
from qpl.instruments.options import BarrierOption, uniform_monitoring_times
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
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

MONITORING_LEVELS = (25, 50, 100, 200, 400)
MC_PATHS = 20_000
MC_SEED = 20_260_913
MC_VARIANCE_REDUCTION = ("antithetic", "control_variate")

WINDOW_STARTS = (100, 200, 400)
"""Each window spans one full sawtooth period, `2 sqrt(n) / lambda_0` steps.

Starting at 100 rather than lower: at `n0 = 50` the lattice is coarse enough
that the amplitude is still pre-asymptotic, and including it drags the fitted
order from 0.4566 to 0.3999 -- a real effect, but one that makes the example
teach the wrong number. The test carries the same three levels plus two more.
"""

BOYLE_LAU_LAYERS = (4, 7, 10, 13, 16)

LAMBDA_0 = math.log(SPOT / BARRIER) / (SIGMA * math.sqrt(EXPIRY))


def _market() -> Market:
    return Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )


def _model() -> BlackScholesModel:
    return BlackScholesModel(sigma=SIGMA)


def _continuous() -> float:
    return barrier_price(
        S=SPOT, K=STRIKE, T=EXPIRY, r=RATE, sigma=SIGMA, q=DIVIDEND, H=BARRIER,
        rebate=0.0, barrier_type=BARRIER_TYPE, kind=KIND,
    )


def _order(levels, errors) -> float:
    return fit_convergence_order(
        [1.0 / n for n in levels], [abs(e) for e in errors]
    ).order


def _header(exact: float) -> None:
    print("example=barrier_option_monitoring_bias")
    print(
        f"spot={SPOT:.2f} strike={STRIKE:.2f} barrier={BARRIER:.2f} "
        f"type={BARRIER_TYPE} kind={KIND} expiry={EXPIRY:.2f}"
    )
    print(f"rate={RATE:.4f} dividend={DIVIDEND:.4f} sigma={SIGMA:.4f}")
    print(f"analytic_continuous={exact:.10f}")
    print(f"bgk_beta={BGK_BETA:.10f}")
    print(f"lambda_0={LAMBDA_0:.6f}")


def _mc(m: int, correction: str):
    option = BarrierOption(
        KIND, STRIKE, EXPIRY, BARRIER, BARRIER_TYPE, 0.0,
        uniform_monitoring_times(EXPIRY, m),
    )
    return price(
        option, _model(), _market(), method="mc",
        cfg=MCConfig(
            n_paths=MC_PATHS,
            seed=MC_SEED,
            variance_reduction=MC_VARIANCE_REDUCTION,
            barrier_correction=correction,  # type: ignore[arg-type]
        ),
    )


def _monitoring_case(exact: float) -> None:
    print("case=monitoring")
    print(
        f"mc_paths={MC_PATHS} seed={MC_SEED} "
        f"variance_reduction={'+'.join(MC_VARIANCE_REDUCTION)}"
    )
    plain_errors, bgk_closed_errors = [], []
    for m in MONITORING_LEVELS:
        plain = _mc(m, "none")
        shifted = _mc(m, "bgk")
        bridged = _mc(m, "brownian_bridge")
        closed = bgk_continuity_corrected_price(
            S=SPOT, K=STRIKE, T=EXPIRY, r=RATE, sigma=SIGMA, q=DIVIDEND,
            H=BARRIER, rebate=0.0, barrier_type=BARRIER_TYPE, kind=KIND,
            n_monitoring=m,
        )
        plain_errors.append(plain.value - exact)
        bgk_closed_errors.append(closed - plain.value)
        print(
            f"monitoring m={m:4d}"
            f" plain={plain.value:.6f} plain_bias={plain.value - exact:+.6f}"
            f" bgk={shifted.value:.6f} bgk_bias={shifted.value - exact:+.6f}"
            f" bridge={bridged.value:.6f} bridge_z={(bridged.value - exact) / bridged.stderr:+.3f}"
            f" stderr={plain.stderr:.6f}"
            f" bgk_closed_form={closed:.6f}"
            f" effective_barrier={shifted.meta['effective_barrier']:.6f}"
            f" knock_fraction={plain.meta['knock_fraction']:.4f}"
        )
    print(f"plain_bias_order={_order(MONITORING_LEVELS, plain_errors):+.4f}")
    print(
        "bgk_closed_form_vs_plain_mc="
        + " ".join(f"{e:+.5f}" for e in bgk_closed_errors)
    )

    # One observation date, and the bridge already prices the continuous
    # contract: its mean does not depend on how many points it conditions on.
    one = _mc(1, "brownian_bridge")
    one_plain = _mc(1, "none")
    print(
        f"single_observation bridge={one.value:.6f}"
        f" z={(one.value - exact) / one.stderr:+.3f}"
        f" plain={one_plain.value:.6f}"
    )


def _tree(n: int, scheme: str = "crr"):
    option = BarrierOption(KIND, STRIKE, EXPIRY, BARRIER, BARRIER_TYPE, 0.0)
    return price(
        option, _model(), _market(), method="tree",
        cfg=TreeConfig(n_steps=n, scheme=scheme),  # type: ignore[arg-type]
    )


def _period(n0: int) -> int:
    return math.ceil(2.0 * math.sqrt(n0) / LAMBDA_0) + 1


def _sawtooth_case(exact: float) -> None:
    print("case=sawtooth")
    amplitudes, lr_amplitudes = [], []
    for n0 in WINDOW_STARTS:
        period = _period(n0)
        crr = [_tree(n).value - exact for n in range(n0, n0 + period)]
        first_odd = n0 if n0 % 2 else n0 + 1
        lr = [
            _tree(n, "leisen-reimer").value - exact
            for n in range(first_odd, n0 + period, 2)
        ]
        amplitudes.append(max(crr) - min(crr))
        lr_amplitudes.append(max(lr) - min(lr))
        print(
            f"window n0={n0:5d} period={period:4d}"
            f" crr_min={min(crr):+.5f} crr_max={max(crr):+.5f}"
            f" crr_amplitude={amplitudes[-1]:.5f}"
            f" lr_amplitude={lr_amplitudes[-1]:.5f}"
            f" lr_over_crr={lr_amplitudes[-1] / amplitudes[-1]:.4f}"
        )
    print(f"sawtooth_order crr={_order(WINDOW_STARTS, amplitudes):+.4f}")
    print(f"sawtooth_order leisen-reimer={_order(WINDOW_STARTS, lr_amplitudes):+.4f}")

    print("boyle_lau layers -> step counts")
    for layer in BOYLE_LAU_LAYERS:
        n = boyle_lau_steps(
            layer, spot=SPOT, barrier=BARRIER, sigma=SIGMA, expiry=EXPIRY
        )
        aligned = _tree(n)
        unaligned = _tree(n + max(1, _period(n) // 4))
        ratio = aligned.meta["effective_barrier_ratio"]
        print(
            f"boyle_lau k={layer:3d} n={n:6d}"
            f" err={aligned.value - exact:+.5e}"
            f" err_times_n={(aligned.value - exact) * n:+.4f}"
            f" misalignment={1.0 - ratio:.3e}"
            f" nearby_unaligned_err={unaligned.value - exact:+.5e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("monitoring", "sawtooth"), default="monitoring"
    )
    args = parser.parse_args()

    exact = _continuous()
    _header(exact)
    if args.case == "monitoring":
        _monitoring_case(exact)
    else:
        _sawtooth_case(exact)


if __name__ == "__main__":
    main()
