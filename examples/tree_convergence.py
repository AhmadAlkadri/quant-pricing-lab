"""Binomial trees: measured convergence to Black-Scholes, for two schemes.

Prints one row per step count, then the least-squares convergence order, then
the same fit after Richardson extrapolation of consecutive pairs.
Deterministic.

    PYTHONPATH=src python examples/tree_convergence.py
    PYTHONPATH=src python examples/tree_convergence.py --scheme leisen-reimer
    PYTHONPATH=src python examples/tree_convergence.py --plot

The reference point is at the money (S = K = 100, r = 5%, q = 0, sigma = 20%,
T = 1, European call).

`--scheme crr` (the default) fits odd and even `n` separately, because the CRR
error oscillates with the parity of `n`: with `d = 1/u`, an even `n` puts a
terminal node exactly on the strike and an odd `n` leaves the strike between
two nodes. Measured order 1.00 on each branch, and the two branches bracket
Black-Scholes. Derivation and tables: `docs/notes/crr_tree_convergence.md`.

`--scheme leisen-reimer` fits one sequence, because the construction has no
even-`n` form and its error has no parity to oscillate with. Measured order
1.98, with the error 507x smaller than CRR's at `n = 101` and 3966x smaller at
`n = 801`. The table it prints carries the CRR error alongside, so the
comparison is on the page rather than in the reader's head. Derivation and
tables: `docs/notes/leisen_reimer.md`.
"""

from __future__ import annotations

import argparse
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


def _setup() -> tuple[EuropeanOption, BlackScholesModel, Market, float]:
    option = EuropeanOption(kind="call", strike=STRIKE, expiry=EXPIRY)
    model = BlackScholesModel(sigma=SIGMA)
    market = Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )
    return option, model, market, price(option, model, market, method="analytic").value


def _header(analytic: float) -> None:
    print("example=tree_convergence")
    print(
        f"case=S={SPOT:g},K={STRIKE:g},T={EXPIRY:g},r={RATE:g},"
        f"q={DIVIDEND:g},sigma={SIGMA:g},kind=call"
    )
    print(f"black_scholes={analytic:.10f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheme",
        choices=("crr", "leisen-reimer"),
        default="crr",
        help="lattice parameterisation (default: crr)",
    )
    parser.add_argument("--plot", action="store_true", help="show a log-log error plot")
    args = parser.parse_args()

    if args.scheme == "crr":
        _crr_report(plot=args.plot)
    else:
        _leisen_reimer_report(plot=args.plot)


def _crr_report(*, plot: bool) -> None:
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

    _header(analytic)
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

    if plot:
        _plot(analytic, signed_by_parity)


def _leisen_reimer_report(*, plot: bool) -> None:
    """One sequence, order two, and the CRR error beside it.

    There is no even branch to fit: `TreeConfig(scheme="leisen-reimer")`
    rejects an even `n_steps`, because the Peizer-Pratt inversion matches
    `P(Bin(n, p) > n/2)`, which counts a whole number of outcomes only for odd
    `n`. So the table below is the odd sequence, and the comparison that
    replaces the odd/even one is against CRR at the same `n`.
    """
    option, model, market, analytic = _setup()

    def tree(n: int, scheme: str) -> float:
        return price(
            option,
            model,
            market,
            method="tree",
            cfg=TreeConfig(n_steps=n, scheme=scheme),  # type: ignore[arg-type]
        ).value

    _header(analytic)
    print("scheme=leisen-reimer")
    print()
    print(
        f"{'n':>5} {'leisen-reimer':>16} {'signed error':>14} "
        f"{'n^2*|error|':>12} {'crr |error|':>12} {'ratio':>9}"
    )

    signed: list[float] = []
    ratios: dict[int, float] = {}
    for n in ODD_LEVELS:
        value = tree(n, "leisen-reimer")
        error = value - analytic
        crr_error = abs(tree(n, "crr") - analytic)
        signed.append(error)
        ratios[n] = crr_error / abs(error)
        print(
            f"{n:5d} {value:16.10f} {error:+14.3e} {n * n * abs(error):12.4f} "
            f"{crr_error:12.3e} {ratios[n]:9.1f}"
        )

    fit = fit_convergence_order(
        [1.0 / n for n in ODD_LEVELS], [abs(e) for e in signed]
    )
    print()
    print(f"order={fit.order:.4f}")
    print(f"residual={fit.residual:.4f}")
    print(f"sign={'above' if all(e > 0 for e in signed) else 'below'}")
    print(f"error_ratio_vs_crr_at_n101={ratios[101]:.1f}")
    print(f"error_ratio_vs_crr_at_n801={ratios[801]:.1f}")

    # Richardson on an order-2 sequence should cancel the 1/n**2 term and
    # leave something like 1/n**3; the measured order says whether it does.
    pair_h, pair_err = [], []
    for n1, n2 in pairwise(ODD_LEVELS):
        v1, v2 = tree(n1, "leisen-reimer"), tree(n2, "leisen-reimer")
        extrapolated = (n2 * n2 * v2 - n1 * n1 * v1) / (n2 * n2 - n1 * n1)
        pair_h.append(1.0 / n1)
        pair_err.append(abs(extrapolated - analytic))
    richardson = fit_convergence_order(pair_h, pair_err)
    print(f"richardson_order={richardson.order:.4f}")
    print(f"richardson_residual={richardson.residual:.4f}")
    print(f"richardson_error_at_n401={pair_err[-1]:.3e}")

    if plot:
        _plot_lr(analytic, signed, [abs(tree(n, "crr") - analytic) for n in ODD_LEVELS])


def _plot_lr(analytic: float, lr_signed: list[float], crr_errs: list[float]) -> None:
    import matplotlib.pyplot as plt

    h = [1.0 / n for n in ODD_LEVELS]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.loglog(h, [abs(e) for e in lr_signed], marker="o", label="Leisen-Reimer")
    ax.loglog(h, crr_errs, marker="s", label="CRR (odd n)")
    ax.set_xlabel("h = 1 / n_steps")
    ax.set_ylabel("|tree - Black-Scholes|")
    ax.set_title(f"Leisen-Reimer vs CRR, ATM call (BS = {analytic:.6f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


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
