"""Three Monte Carlo Greek estimators, one table, two payoffs.

Prints, for a European call and for a cash-or-nothing digital at the same
point, every Greek from every estimator that applies, with its own standard
error and its z-score against the closed form. Then it repeats the exercise
over independent seeds, so the column that actually decides which estimator to
use -- the spread of the estimate, not the estimate -- is in the same output.

What the tables show

- **On the call, all three estimators are unbiased and the bump is as good as
  the pathwise one at delta.** A common-random-numbers central difference of a
  Lipschitz payoff *is* the pathwise derivative plus `O(h**2)`. Where they part
  company is gamma: a second difference divides by `h**2`, and at the default
  bump that is a factor of 10 000 on the variance.
- **On the digital, the ordering reverses and the pathwise estimator is gone.**
  The almost-everywhere derivative of an indicator is identically zero, so the
  pathwise delta would return `0.0` -- not noisy, wrong -- and the engine
  refuses it. The likelihood ratio, which differentiates the density instead of
  the payoff, does not notice the jump at all.
- **The digital's bumped small-step Greeks are a lottery, and the spread table
  is the only place that shows it.** With `dt = 1e-04` the probability that a
  path crosses the strike when the maturity moves is about `2e-06`, so most
  runs contain no crossing at all: the sample is the discount factor's smooth
  part, the estimate is `r V`, and the reported standard error describes that
  part rather than the Greek. A run that *does* contain a crossing moves by a
  full `cash / (N dt)`. The estimator stays unbiased through all of this -- it
  is exactly unbiased for the finite difference it computes -- so its z-score
  at any one seed says nothing, and can be under 1 or over 500 depending on
  whether the lottery paid out (`examples/digital_option_cross_method.py`
  prints a seed where it reads +531). What does say something is the spread
  column: over 40 seeds the bumped theta is 2810 times the likelihood ratio's
  variance and the bumped gamma is 8.1e+09 times it.

`--case h` prints the other half of the story: the digital bump's bias and
standard deviation as the bump size sweeps three decimal orders, so the optimal
`h` is visible as a minimum in the RMSE column rather than asserted.

Deterministic: every run is seeded, and the seeds are listed in the output.

Run:  PYTHONPATH=src python examples/mc_greeks_estimators.py
      PYTHONPATH=src python examples/mc_greeks_estimators.py --case h

Derivations and the full measured tables:
`docs/notes/mc_greeks_pathwise_likelihood_ratio.md`.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from qpl.engines.mc.pricers import MCConfig
from qpl.exceptions import NotSupportedError
from qpl.instruments.options import DigitalOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import greeks
from qpl.validation import fit_convergence_order

SPOT = 100.0
STRIKE = 100.0
EXPIRY = 1.0
RATE = 0.05
DIVIDEND = 0.01
SIGMA = 0.20
CASH = 1.0

PATHS = 200_000
SEED = 123
SPREAD_PATHS = 20_000
SPREAD_SEEDS = tuple(range(2000, 2040))
BUMP_LEVELS = (10.0, 3.0, 1.0, 0.3, 0.1, 0.03)
BUMP_SEEDS = SPREAD_SEEDS[:24]

GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")
ESTIMATORS = ("bump", "pathwise", "likelihood_ratio")


def _market() -> Market:
    return Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )


def _mc(option, model, market, estimator: str, *, n_paths: int, seed: int, **kwargs):
    return greeks(
        option,
        model,
        market,
        method="mc",
        cfg=MCConfig(
            n_paths=n_paths, n_steps=1, seed=seed, greeks_estimator=estimator
        ),
        **kwargs,
    )


def _estimator_table(label: str, option, model, market) -> None:
    """One row per (estimator, Greek): estimate, stderr, closed form, z."""
    exact = greeks(option, model, market, method="analytic")
    print(f"{label}_table estimator greek estimate stderr analytic z")
    for estimator in ESTIMATORS:
        try:
            result = _mc(option, model, market, estimator, n_paths=PATHS, seed=SEED)
        except NotSupportedError:
            print(f"{label} {estimator} refused=payoff_derivative_is_zero_ae")
            continue
        assert result.meta is not None
        for greek in GREEK_NAMES:
            value = getattr(result, greek)
            stderr = result.meta["stderr"][greek]
            truth = getattr(exact, greek)
            z = (value - truth) / stderr if stderr > 0.0 else 0.0
            print(
                f"{label} {estimator} {greek} estimate={value:+.8f} "
                f"stderr={stderr:.3e} analytic={truth:+.8f} z={z:+.2f} "
                f"source={result.meta['estimator'][greek]}"
            )


def _spread_table(label: str, option, model, market) -> None:
    """The column that decides: the spread of the estimate over seeds."""
    print(f"{label}_spread greek estimator sd ratio_to_best")
    for greek in GREEK_NAMES:
        spreads: dict[str, float] = {}
        for estimator in ESTIMATORS:
            try:
                values = [
                    getattr(
                        _mc(
                            option,
                            model,
                            market,
                            estimator,
                            n_paths=SPREAD_PATHS,
                            seed=seed,
                        ),
                        greek,
                    )
                    for seed in SPREAD_SEEDS
                ]
            except NotSupportedError:
                continue
            spreads[estimator] = float(np.std(values, ddof=1))
        best = min(spreads.values())
        for estimator, sd in spreads.items():
            print(
                f"{label}_spread {greek} {estimator} sd={sd:.4e} "
                f"variance_ratio={(sd / best) ** 2:.2f}"
            )


def _bump_size_table(option, model, market) -> None:
    """The digital bump's bias/variance trade-off as `h` sweeps."""
    exact = greeks(option, model, market, method="analytic").delta
    print("bump_h_table h bias sd rmse")
    spreads, biases, rmse = [], [], []
    for h in BUMP_LEVELS:
        values = [
            _mc(
                option,
                model,
                market,
                "bump",
                n_paths=SPREAD_PATHS,
                seed=seed,
                bumps={"spot": h},
            ).delta
            for seed in BUMP_SEEDS
        ]
        sd = float(np.std(values, ddof=1))
        # The estimator is exactly unbiased for the finite difference it
        # computes, so its bias IS that difference's truncation error, which is
        # deterministic and worth computing exactly rather than through a noise
        # floor a hundred times larger.
        bias = _central_difference_delta(option, model, market, h) - exact
        spreads.append(sd)
        biases.append(abs(bias))
        rmse.append(math.hypot(bias, sd))
        print(f"bump_h h={h:g} bias={bias:+.4e} sd={sd:.4e} rmse={rmse[-1]:.4e}")
    sd_fit = fit_convergence_order(h=list(BUMP_LEVELS), err=spreads)
    bias_fit = fit_convergence_order(h=list(BUMP_LEVELS), err=biases)
    print(f"bump_h_sd_order={sd_fit.order:+.4f} residual={sd_fit.residual:.4f}")
    print(f"bump_h_bias_order={bias_fit.order:+.4f} residual={bias_fit.residual:.4f}")
    print(f"bump_h_optimal={BUMP_LEVELS[int(np.argmin(rmse))]:g}")


def _central_difference_delta(option, model, market, h: float) -> float:
    """`(V(S+h) - V(S-h)) / 2h` in closed form: the bump's exact target."""
    from qpl.engines.analytic.digital import digital_price

    kwargs = {
        "K": option.strike,
        "T": option.expiry,
        "r": market.rate(option.expiry),
        "sigma": model.sigma,
        "q": market.dividend_yield(option.expiry),
        "cash": option.cash,
        "kind": option.kind,
    }
    return (
        digital_price(S=market.spot + h, **kwargs)
        - digital_price(S=market.spot - h, **kwargs)
    ) / (2.0 * h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("all", "h"),
        default="all",
        help="'all' for the estimator tables, 'h' for the bump-size study",
    )
    args = parser.parse_args()

    market = _market()
    model = BlackScholesModel(sigma=SIGMA)
    call = EuropeanOption(kind="call", strike=STRIKE, expiry=EXPIRY)
    digital = DigitalOption(kind="call", strike=STRIKE, expiry=EXPIRY, cash=CASH)

    print("example=mc_greeks_estimators")
    print(f"case={args.case}")
    print(
        f"spot={SPOT:.2f} strike={STRIKE:.2f} expiry={EXPIRY:.2f} "
        f"rate={RATE:.4f} dividend={DIVIDEND:.4f} sigma={SIGMA:.4f}"
    )
    print(f"paths={PATHS} seed={SEED} spread_paths={SPREAD_PATHS} "
          f"spread_seeds={len(SPREAD_SEEDS)}")

    if args.case == "h":
        _bump_size_table(digital, model, market)
        return

    _estimator_table("call", call, model, market)
    _estimator_table("digital", digital, model, market)
    _spread_table("call", call, model, market)
    _spread_table("digital", digital, model, market)


if __name__ == "__main__":
    main()
