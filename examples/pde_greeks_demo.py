"""PDE Greeks read off the grid, and what Rannacher start-up fixes.

Prints two tables. The first is the smooth case: an ATM call at T = 1 on a
strike-aligned grid with `n_s = n_t = n`, where delta, gamma and theta all
converge at order two and the choice of time stepping barely matters. The
second is the start-up case: a short-dated ATM call with the spatial grid
refined 80x faster than the time grid, where plain Crank-Nicolson's gamma
*diverges* and Rannacher's converges.

    PYTHONPATH=src python examples/pde_greeks_demo.py
    PYTHONPATH=src python examples/pde_greeks_demo.py --case startup

Deterministic and fast (under two seconds for either case).

Why the second table needs its own grid family: Crank-Nicolson's amplification
factor tends to -1 for modes with `lambda dt` large, so the payoff kink's
high-frequency content is flipped rather than damped and survives the march.
With `n_s = n_t` the time step shrinks as fast as the spacing and `lambda dt`
at the strike stays near 1, so nothing goes wrong. With `n_s` growing faster,
`lambda dt` grows with the refinement and the ripple takes over -- in gamma,
which is a second difference of neighbouring values, long before it shows in
the price. Rannacher (1984), Numerische Mathematik 43, 309-327; Giles and
Carter (2006), Journal of Computational Finance 9(4), 89-112. Measured tables
and the derivation: `docs/notes/pde_greeks_and_rannacher.md`.
"""

from __future__ import annotations

import argparse

from qpl.engines.pde.pricers import PDEConfig
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks
from qpl.validation import fit_convergence_order

SPOT = 100.0
STRIKE = 100.0
RATE = 0.05
DIVIDEND = 0.0
SIGMA = 0.20

SMOOTH_EXPIRY = 1.0
SMOOTH_LEVELS = (50, 100, 200, 400)

STARTUP_EXPIRY = 0.05
STARTUP_TIME_LEVELS = (5, 10, 20)
STARTUP_SPACE_RATIO = 80

TIME_STEPPINGS = ("theta", "rannacher")


def _setup(expiry: float):
    option = EuropeanOption(kind="call", strike=STRIKE, expiry=expiry)
    model = BlackScholesModel(sigma=SIGMA)
    market = Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )
    return option, model, market, greeks(option, model, market, method="analytic")


def _errors(option, model, market, analytic, *, n_s, n_t, time_stepping, alignment):
    cfg = PDEConfig(
        n_s=n_s,
        n_t=n_t,
        theta=0.5,
        s_max_multiplier=4.0,
        strike_alignment=alignment,
        time_stepping=time_stepping,
    )
    res = greeks(option, model, market, method="pde", cfg=cfg)
    return {
        greek: getattr(res, greek) - getattr(analytic, greek)
        for greek in ("delta", "gamma", "theta")
    }


def _header(case: str, expiry: float, analytic) -> None:
    print("example=pde_greeks_demo")
    print(f"case={case}")
    print(
        f"point=S={SPOT:g},K={STRIKE:g},T={expiry:g},r={RATE:g},"
        f"q={DIVIDEND:g},sigma={SIGMA:g},kind=call"
    )
    print(
        f"analytic delta={analytic.delta:.10f} gamma={analytic.gamma:.10f} "
        f"theta={analytic.theta:.10f}"
    )


def _print_table(rows: list[tuple[int, int, dict[str, dict[str, float]]]]) -> None:
    print(
        f"{'n_s':>6} {'n_t':>6} | "
        + " | ".join(
            f"{ts[:4]} {greek:<5}".rjust(12)
            for ts in TIME_STEPPINGS
            for greek in ("delta", "gamma", "theta")
        )
    )
    for n_s, n_t, errs in rows:
        cells = " | ".join(
            f"{errs[ts][greek]:+12.3e}"
            for ts in TIME_STEPPINGS
            for greek in ("delta", "gamma", "theta")
        )
        print(f"{n_s:>6} {n_t:>6} | {cells}")


def _print_orders(levels, rows, label: str) -> None:
    h = [1.0 / n for n in levels]
    for ts in TIME_STEPPINGS:
        for greek in ("delta", "gamma", "theta"):
            errs = [abs(errs_by_ts[ts][greek]) for _, _, errs_by_ts in rows]
            fit = fit_convergence_order(h, errs)
            print(
                f"{label}_order {ts}_{greek}={fit.order:+.3f} "
                f"residual={fit.residual:.4f}"
            )


def _smooth() -> None:
    option, model, market, analytic = _setup(SMOOTH_EXPIRY)
    _header("smooth", SMOOTH_EXPIRY, analytic)
    print(f"grid=uniform-in-spot, strike_alignment=midpoint, n_s=n_t=n in {SMOOTH_LEVELS}")

    rows = [
        (
            n,
            n,
            {
                ts: _errors(
                    option,
                    model,
                    market,
                    analytic,
                    n_s=n,
                    n_t=n,
                    time_stepping=ts,
                    alignment="midpoint",
                )
                for ts in TIME_STEPPINGS
            },
        )
        for n in SMOOTH_LEVELS
    ]
    _print_table(rows)
    _print_orders(SMOOTH_LEVELS, rows, "smooth")
    print(
        "note=on this refinement path the two time steppings are "
        "indistinguishable; both are order two in all three Greeks"
    )


def _startup() -> None:
    option, model, market, analytic = _setup(STARTUP_EXPIRY)
    _header("startup", STARTUP_EXPIRY, analytic)
    print(
        f"grid=uniform-in-spot, strike_alignment=none, "
        f"n_s={STARTUP_SPACE_RATIO}*n_t, n_t in {STARTUP_TIME_LEVELS}"
    )

    rows = [
        (
            STARTUP_SPACE_RATIO * n_t,
            n_t,
            {
                ts: _errors(
                    option,
                    model,
                    market,
                    analytic,
                    n_s=STARTUP_SPACE_RATIO * n_t,
                    n_t=n_t,
                    time_stepping=ts,
                    alignment="none",
                )
                for ts in TIME_STEPPINGS
            },
        )
        for n_t in STARTUP_TIME_LEVELS
    ]
    _print_table(rows)
    _print_orders(STARTUP_TIME_LEVELS, rows, "startup")
    print(
        "note=the plain theta scheme fits a NEGATIVE gamma order here -- "
        "refining makes it worse -- and rannacher restores order two"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("smooth", "startup"),
        default="smooth",
        help="which table to print (default: smooth)",
    )
    args = parser.parse_args()

    if args.case == "smooth":
        _smooth()
    else:
        _startup()


if __name__ == "__main__":
    main()
