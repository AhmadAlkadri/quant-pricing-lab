"""CRR binomial tree: measured convergence to Black-Scholes, and the
odd/even oscillation that comes with it.

Prints one row per step count, then the least-squares convergence order fitted
separately on odd and on even `n`, then the same fit after Richardson
extrapolation of consecutive odd pairs. Deterministic; no arguments needed.

    PYTHONPATH=src python examples/tree_convergence.py
    PYTHONPATH=src python examples/tree_convergence.py --plot

The reference point is at the money (S = K = 100, r = 5%, q = 0, sigma = 20%,
T = 1, European call), which is where the parity effect is cleanest: with
`d = 1/u`, an even `n` puts a terminal node exactly on the strike and an odd
`n` leaves the strike between two nodes. Derivation and tables:
`docs/notes/crr_tree_convergence.md`.
"""

from __future__ import annotations

import sys
from itertools import pairwise

from qpl.engines.tree import TreeConfig
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price
from qpl.validation import fit_convergence_order

SPOT = 100.0
STRIKE = 100.0
EXPIRY = 1.0
RATE = 0.05
DIVIDEND = 0.0
SIGMA = 0.20

ODD_LEVELS = (25, 51, 101, 201, 401, 801)
EVEN_LEVELS = (26, 50, 100, 200, 400, 800)


def main() -> None:
    option = EuropeanOption(kind="call", strike=STRIKE, expiry=EXPIRY)
    model = BlackScholesModel(sigma=SIGMA)
    market = Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )

    analytic = price(option, model, market, method="analytic").value

    def tree(n: int) -> float:
        return price(
            option, model, market, method="tree", cfg=TreeConfig(n_steps=n)
        ).value

    print("example=tree_convergence")
    print(
        f"case=S={SPOT:g},K={STRIKE:g},T={EXPIRY:g},r={RATE:g},"
        f"q={DIVIDEND:g},sigma={SIGMA:g},kind=call"
    )
    print(f"black_scholes={analytic:.10f}")
    print()
    print(f"{'parity':>6} {'n':>5} {'tree':>14} {'signed error':>14} {'n*|error|':>10}")

    fits: dict[str, tuple[float, float]] = {}
    signed_by_parity: dict[str, list[float]] = {}
    for label, levels in (("odd", ODD_LEVELS), ("even", EVEN_LEVELS)):
        signed = []
        for n in levels:
            value = tree(n)
            error = value - analytic
            signed.append(error)
            print(f"{label:>6} {n:5d} {value:14.10f} {error:+14.3e} {n * abs(error):10.4f}")
        fit = fit_convergence_order([1.0 / n for n in levels], [abs(e) for e in signed])
        fits[label] = (fit.order, fit.residual)
        signed_by_parity[label] = signed

    print()
    print(f"odd_order={fits['odd'][0]:.4f}")
    print(f"odd_residual={fits['odd'][1]:.4f}")
    print(f"even_order={fits['even'][0]:.4f}")
    print(f"even_residual={fits['even'][1]:.4f}")
    print(f"odd_sign={'above' if all(e > 0 for e in signed_by_parity['odd']) else 'mixed'}")
    print(f"even_sign={'below' if all(e < 0 for e in signed_by_parity['even']) else 'mixed'}")

    # Richardson extrapolation within one parity: the error is C/n with a
    # constant that no longer oscillates, so two odd levels cancel it.
    pair_h, pair_err = [], []
    for n1, n2 in pairwise(ODD_LEVELS):
        extrapolated = (n2 * tree(n2) - n1 * tree(n1)) / (n2 - n1)
        pair_h.append(1.0 / n1)
        pair_err.append(abs(extrapolated - analytic))
    richardson = fit_convergence_order(pair_h, pair_err)
    print(f"richardson_order={richardson.order:.4f}")
    print(f"richardson_residual={richardson.residual:.4f}")
    print(f"richardson_error_at_n401={pair_err[-1]:.3e}")

    if "--plot" in sys.argv[1:]:
        _plot(analytic, signed_by_parity)


def _plot(analytic: float, signed_by_parity: dict[str, list[float]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for label, levels in (("odd", ODD_LEVELS), ("even", EVEN_LEVELS)):
        ax.loglog(
            [1.0 / n for n in levels],
            [abs(e) for e in signed_by_parity[label]],
            marker="o",
            label=f"{label} n",
        )
    ax.set_xlabel("h = 1 / n_steps")
    ax.set_ylabel("|tree - Black-Scholes|")
    ax.set_title(f"CRR convergence, ATM call (BS = {analytic:.6f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
