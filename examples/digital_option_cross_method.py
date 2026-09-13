"""One cash-or-nothing digital, four engines, and what a jump costs each.

Prints, for a single specification:

- the closed-form price and the static-replication identity that pins it;
- a Leisen-Reimer and a CRR lattice at matched step counts, with their errors;
- the finite-difference grid **with and without** the two remedies, so the
  discontinuity pathology and its repair appear in the same table;
- a Monte Carlo estimate with its standard error and z-score.

The point is that the four methods fail in four different ways, and only one
of them fails at all badly:

- **Monte Carlo does not notice.** An indicator is bounded and
  square-integrable, so the sample proportion is unbiased and its standard
  error is the usual `N**-1/2`.
- **Leisen-Reimer is at its best.** Its construction picks the per-step
  probability so that `P(Bin(n, p) > n/2)` matches `N(d2)`, and a digital's
  price *is* that probability discounted -- so it is order 2 with a tiny
  constant.
- **CRR is at its worst.** It records only which side of the strike each
  terminal node fell on, so the error is a sawtooth of size `O(1/sqrt(n))`.
- **The finite-difference grid depends entirely on how the jump is placed.**
  Plain Crank-Nicolson on an unaligned grid is first order in the *price*;
  aligning the strike to a cell face, or cell-averaging the payoff, restores
  second order.

Deterministic: the lattice and grid legs use no randomness, and the Monte
Carlo leg is seeded.

Run:  PYTHONPATH=src python examples/digital_option_cross_method.py
      PYTHONPATH=src python examples/digital_option_cross_method.py --case pde
"""

from __future__ import annotations

import argparse
import math
from itertools import pairwise

from qpl.engines.analytic.digital import digital_price
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.pde.pricers import PDEConfig
from qpl.engines.tree import TreeConfig
from qpl.instruments.options import DigitalOption
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
CASH = 1.0

TREE_LEVELS = (101, 201, 401, 801)
"""Odd, so both schemes accept them."""

PDE_LEVELS = (100, 200, 400, 800)
"""`n_s = n_t`. Every one of these puts the strike exactly *on* a node when the
grid is unaligned, which is the worst case and the one worth showing."""

MC_LEVELS = (5_000, 20_000, 80_000, 320_000)
MC_SEED = 123

# An off-the-money point, where CRR's sawtooth is visible. At the money the
# strike is the geometric mean of the two central terminal nodes for every odd
# `n` (because `u d = 1`), which freezes the fractional position and makes CRR
# look like a clean order-1 scheme. That symmetry is a special case, not the
# rule, and this second point is here so the example does not imply otherwise.
OFF_MONEY = {"spot": 100.0, "strike": 110.0, "expiry": 0.75, "rate": 0.03,
             "dividend": 0.01, "sigma": 0.25}
SAWTOOTH_LEVELS = tuple(range(121, 202, 2))

PDE_CONFIGURATIONS = (
    # (label, strike_alignment, time_stepping, payoff_projection)
    ("plain_cn_unaligned", "none", "theta", "none"),
    ("cn_unaligned_projected", "none", "theta", "cell_average"),
    ("rannacher_aligned", "midpoint", "rannacher", "none"),
    ("rannacher_unaligned_projected", "none", "rannacher", "cell_average"),
)


def _market() -> Market:
    return Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )


def _order(levels, errors) -> float:
    return fit_convergence_order(
        [1.0 / n for n in levels], [abs(e) for e in errors]
    ).order


def _header(exact: float) -> None:
    print("example=digital_option_cross_method")
    print(f"spot={SPOT:.2f} strike={STRIKE:.2f} expiry={EXPIRY:.2f} cash={CASH:.2f}")
    print(f"rate={RATE:.4f} dividend={DIVIDEND:.4f} sigma={SIGMA:.4f}")
    print(f"analytic={exact:.10f}")


def _identity(model: BlackScholesModel, market: Market) -> None:
    call = price(
        DigitalOption(kind="call", strike=STRIKE, expiry=EXPIRY, cash=CASH), model, market
    ).value
    put = price(
        DigitalOption(kind="put", strike=STRIKE, expiry=EXPIRY, cash=CASH), model, market
    ).value
    riskless = CASH * math.exp(-RATE * EXPIRY)
    print(f"analytic_put={put:.10f}")
    print(f"riskless={riskless:.10f}")
    # Exact in the model: the two digitals together are a zero-coupon bond.
    print(f"replication_residual={call + put - riskless:+.3e}")


def _trees(option, model, market, exact: float) -> None:
    print("tree_table scheme n price error")
    for scheme in ("crr", "leisen-reimer"):
        errors = []
        for n in TREE_LEVELS:
            value = price(
                option, model, market, method="tree", cfg=TreeConfig(n_steps=n, scheme=scheme)
            ).value
            errors.append(value - exact)
            print(f"tree {scheme} n={n} price={value:.10f} error={value - exact:+.3e}")
        print(f"tree_order {scheme}={_order(TREE_LEVELS, errors):+.4f}")

    finest_crr = abs(
        price(option, model, market, method="tree", cfg=TreeConfig(n_steps=801)).value - exact
    )
    finest_lr = abs(
        price(
            option,
            model,
            market,
            method="tree",
            cfg=TreeConfig(n_steps=801, scheme="leisen-reimer"),
        ).value
        - exact
    )
    print(f"tree_error_ratio_crr_over_lr_at_n801={finest_crr / finest_lr:.1f}")
    _crr_sawtooth()


def _crr_sawtooth() -> None:
    """CRR off the money, at consecutive odd `n`: a ramp with a jump in it.

    The error drifts smoothly while the strike moves inside one terminal cell,
    then flips sign in a single step of `n` when the strike crosses a node and
    that node's whole probability mass -- `O(1/sqrt(n))` of it -- changes
    sides. Averaging over parities does not remove this, because the position
    that drives it varies continuously with `n`.
    """
    option = DigitalOption(
        kind="call", strike=OFF_MONEY["strike"], expiry=OFF_MONEY["expiry"], cash=CASH
    )
    model = BlackScholesModel(sigma=OFF_MONEY["sigma"])
    market = Market(
        spot=OFF_MONEY["spot"],
        rate_curve=FlatRateCurve(OFF_MONEY["rate"]),
        dividend_curve=FlatDividendCurve(OFF_MONEY["dividend"]),
    )
    exact = digital_price(
        S=OFF_MONEY["spot"],
        K=OFF_MONEY["strike"],
        T=OFF_MONEY["expiry"],
        r=OFF_MONEY["rate"],
        sigma=OFF_MONEY["sigma"],
        q=OFF_MONEY["dividend"],
        cash=CASH,
        kind="call",
    )
    print(
        "sawtooth_point spot={spot:.2f} strike={strike:.2f} expiry={expiry:.2f} "
        "sigma={sigma:.2f}".format(**OFF_MONEY)
    )
    errors = []
    for n in SAWTOOTH_LEVELS:
        value = price(
            option, model, market, method="tree", cfg=TreeConfig(n_steps=n)
        ).value
        errors.append(value - exact)
        print(f"sawtooth crr n={n} error={value - exact:+.3e}")
    signs = [1 if e > 0 else -1 for e in errors]
    changes = sum(1 for a, b in pairwise(signs) if a != b)
    print(f"sawtooth_sign_changes={changes}")
    print(f"sawtooth_max_over_min={max(map(abs, errors)) / min(map(abs, errors)):.1f}")


def _pde(option, model, market, exact: float) -> None:
    print("pde_table configuration n price error")
    for label, alignment, time_stepping, projection in PDE_CONFIGURATIONS:
        errors = []
        for n in PDE_LEVELS:
            cfg = PDEConfig(
                n_s=n,
                n_t=n,
                theta=0.5,
                strike_alignment=alignment,  # type: ignore[arg-type]
                time_stepping=time_stepping,  # type: ignore[arg-type]
                payoff_projection=projection,  # type: ignore[arg-type]
            )
            value = price(option, model, market, method="pde", cfg=cfg).value
            errors.append(value - exact)
            print(f"pde {label} n={n} price={value:.10f} error={value - exact:+.3e}")
        print(f"pde_order {label}={_order(PDE_LEVELS, errors):+.4f}")


def _monte_carlo(option, model, market, exact: float) -> None:
    print("mc_table n price stderr z")
    stderrs = []
    for n_paths in MC_LEVELS:
        result = price(
            option,
            model,
            market,
            method="mc",
            cfg=MCConfig(n_paths=n_paths, n_steps=1, seed=MC_SEED),
        )
        stderr = result.stderr or 0.0
        stderrs.append(stderr)
        z = (result.value - exact) / stderr if stderr > 0.0 else 0.0
        print(f"mc n={n_paths} price={result.value:.10f} stderr={stderr:.3e} z={z:+.3f}")
    print(f"mc_stderr_order={_order(MC_LEVELS, stderrs):+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("all", "tree", "pde", "mc"),
        default="all",
        help="which table to print (default: all)",
    )
    args = parser.parse_args()

    market = _market()
    model = BlackScholesModel(sigma=SIGMA)
    option = DigitalOption(kind="call", strike=STRIKE, expiry=EXPIRY, cash=CASH)
    exact = digital_price(
        S=SPOT, K=STRIKE, T=EXPIRY, r=RATE, sigma=SIGMA, q=DIVIDEND, cash=CASH, kind="call"
    )

    print(f"case={args.case}")
    _header(exact)
    _identity(model, market)

    if args.case in ("all", "tree"):
        _trees(option, model, market, exact)
    if args.case in ("all", "pde"):
        _pde(option, model, market, exact)
    if args.case in ("all", "mc"):
        _monte_carlo(option, model, market, exact)

    if args.case == "all":
        # Monte Carlo has no Greeks here, and says so rather than returning a
        # number: the payoff's derivative is a Dirac mass.
        from qpl.exceptions import NotSupportedError
        from qpl.pricing import greeks

        try:
            greeks(option, model, market, method="mc", cfg=MCConfig(n_paths=1_000, seed=MC_SEED))
        except NotSupportedError:
            print("mc_greeks=not_supported")


if __name__ == "__main__":
    main()
