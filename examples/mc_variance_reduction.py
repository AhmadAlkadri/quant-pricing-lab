"""Variance reduction on one Monte Carlo price, with the cost held fixed.

Prints one row per estimator: the paths it ran, the **standard normals it
spent** (the axis the comparison is made on), the price, the reported standard
error, the z-score against the closed form, and two independent readings of the
variance factor -- one from the reported standard errors of a single run, one
from the spread of 20 independent seeds. Where theory gives a prediction
(`2/(1+rho_a)` for antithetic, `1/(1-rho^2)` for the control variate) the row
carries it too.

`--case digital` runs the same table on a cash-or-nothing digital and adds the
stratum scan, which is where the interesting failure lives: for a
discontinuous payoff the stratified gain is **not** monotone in the number of
strata, because all the residual variance sits in the single stratum that
contains the jump and the gain depends on where inside it the jump falls.

Deterministic: every seed is fixed, and the script prints the same bytes on
every run.

Reference for the estimators: Glasserman (2003), *Monte Carlo Methods in
Financial Engineering*, sections 4.1-4.3. Derivation and full tables:
`docs/notes/mc_variance_reduction.md`.
"""

from __future__ import annotations

import argparse
import math
from itertools import pairwise

import numpy as np

from qpl.engines.mc.pricers import MCConfig
from qpl.engines.mc.variance_reduction import terminal_spots_from_normals
from qpl.instruments.options import DigitalOption, EuropeanOption
from qpl.instruments.payoffs import call_payoff, digital_payoff
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price

N_DRAWS = 20_480
N_STRATA = 64
SEEDS = tuple(range(1, 21))
REPORT_SEED = 20250913
PILOT_DRAWS = 40_000
PILOT_SEED = 7

ESTIMATORS: tuple[tuple[str, str | tuple[str, ...]], ...] = (
    ("none", "none"),
    ("antithetic", "antithetic"),
    ("control_variate", "control_variate"),
    ("stratified", "stratified"),
    ("antithetic+control", ("antithetic", "control_variate")),
    ("stratified+control", ("stratified", "control_variate")),
)


def _market() -> Market:
    return Market(
        spot=100.0,
        rate_curve=FlatRateCurve(0.05),
        dividend_curve=FlatDividendCurve(0.0),
    )


def _paths_for(vr: str | tuple[str, ...], n_draws: int) -> int:
    """Antithetic uses each normal twice, so equal cost means double the paths."""
    return 2 * n_draws if "antithetic" in vr else n_draws


def _cfg(vr: str | tuple[str, ...], seed: int, n_strata: int = N_STRATA) -> MCConfig:
    return MCConfig(
        n_paths=_paths_for(vr, N_DRAWS),
        n_steps=1,
        seed=seed,
        variance_reduction=vr,
        n_strata=n_strata,
    )


def _pilot_correlations(option, model, market) -> tuple[float, float]:
    """`corr(Y(Z), Y(-Z))` and `corr(Y, e^{-rT} S_T)` on an independent pilot."""
    t = option.expiry
    mu = market.rate(t) - market.dividend_yield(t)
    df = market.df_r(t)
    if isinstance(option, DigitalOption):

        def payoff(spots: np.ndarray) -> np.ndarray:
            return np.asarray(
                digital_payoff(spots, option.strike, option.cash, option.kind),
                dtype=float,
            )
    else:

        def payoff(spots: np.ndarray) -> np.ndarray:
            return call_payoff(spots, option.strike)

    rng = np.random.default_rng(PILOT_SEED)
    z = rng.normal(size=(PILOT_DRAWS, 1))
    kwargs = {"s0": market.spot, "mu": mu, "sigma": model.sigma, "t": t}
    y = df * payoff(terminal_spots_from_normals(z, **kwargs))
    y_reflected = df * payoff(terminal_spots_from_normals(-z, **kwargs))
    x = df * terminal_spots_from_normals(z, **kwargs)
    return (
        float(np.corrcoef(y, y_reflected)[0, 1]),
        float(np.corrcoef(y, x)[0, 1]),
    )


def _seed_variance(option, model, market, vr, n_strata: int = N_STRATA) -> float:
    values = np.array(
        [
            price(option, model, market, method="mc", cfg=_cfg(vr, seed, n_strata)).value
            for seed in SEEDS
        ]
    )
    return float(values.var(ddof=1))


def _run_table(option, model, market, analytic: float) -> None:
    rho_anti, rho_ctrl = _pilot_correlations(option, model, market)
    print(f"pilot_rho_antithetic={rho_anti:+.4f}")
    print(f"pilot_rho_control={rho_ctrl:+.4f}")
    print(f"predicted_factor antithetic={2.0 / (1.0 + rho_anti):.2f}")
    print(f"predicted_factor control_variate={1.0 / (1.0 - rho_ctrl**2):.2f}")

    plain_stderr = None
    plain_variance = None
    for label, vr in ESTIMATORS:
        res = price(option, model, market, method="mc", cfg=_cfg(vr, REPORT_SEED))
        seed_variance = _seed_variance(option, model, market, vr)
        if plain_stderr is None:
            plain_stderr = res.stderr
            plain_variance = seed_variance
        stderr_factor = (plain_stderr / res.stderr) ** 2
        seed_factor = plain_variance / seed_variance
        z = (res.value - analytic) / res.stderr
        print(
            f"estimator={label:18s} "
            f"paths={res.meta['n_paths']:6d} "
            f"normals={res.meta['n_normal_draws']:6d} "
            f"price={res.value:.6f} "
            f"stderr={res.stderr:.6f} "
            f"z={z:+.2f} "
            f"stderr_factor={stderr_factor:6.2f} "
            f"seed_factor={seed_factor:7.2f}"
        )


STRATA_SCAN = (8, 16, 20, 32, 40, 64)
"""Stratum counts for the digital scan. Powers of two *and* their 1.25x
neighbours, because the point of the scan is that the gain depends on where the
jump falls inside its stratum and not on how many strata there are: 16 puts it
4.6% in and 20 puts it 80.8% in, so 20 strata are predicted to be worse than
16. Each divides the draw count, which proportional allocation requires."""


def _run_strata_scan(option, model, market) -> None:
    """Measured gain against `K p(1-p) / (f(1-f))`, the law it actually follows."""
    from scipy.stats import norm

    t = option.expiry
    mu = market.rate(t) - market.dividend_yield(t)
    z_star = (math.log(option.strike / market.spot) - (mu - 0.5 * model.sigma**2) * t) / (
        model.sigma * math.sqrt(t)
    )
    u_star = float(norm.cdf(z_star))
    p_itm = 1.0 - u_star
    print(f"jump_at_u={u_star:.6f} itm_probability={p_itm:.6f}")

    plain_variance = _seed_variance(option, model, market, "none")
    gains = {}
    for k in STRATA_SCAN:
        f = (k * u_star) % 1.0
        predicted = k * p_itm * (1.0 - p_itm) / (f * (1.0 - f))
        gains[k] = plain_variance / _seed_variance(option, model, market, "stratified", k)
        print(
            f"strata_scan K={k:3d} f={f:.4f} predicted={predicted:8.2f} "
            f"factor={gains[k]:8.2f}"
        )
    monotone = all(gains[a] <= gains[b] for a, b in pairwise(STRATA_SCAN))
    print(f"strata_gain_monotone_in_K={monotone}")
    print(f"strata_gain_K16_over_K20={gains[16] / gains[20]:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("vanilla", "digital"), default="vanilla")
    args = parser.parse_args()

    market = _market()
    model = BlackScholesModel(sigma=0.20)
    if args.case == "vanilla":
        option = EuropeanOption(kind="call", strike=100.0, expiry=1.0)
    else:
        option = DigitalOption(kind="call", strike=100.0, expiry=1.0, cash=1.0)

    analytic = price(option, model, market, method="analytic").value

    print("example=mc_variance_reduction")
    print(f"case={args.case}")
    print(f"instrument={type(option).__name__} kind={option.kind} strike={option.strike:g}")
    print(f"spot={market.spot:g} expiry={option.expiry:g} sigma={model.sigma:g}")
    print(f"draws={N_DRAWS} strata={N_STRATA} seeds={len(SEEDS)} report_seed={REPORT_SEED}")
    print(f"analytic={analytic:.10f}")
    _run_table(option, model, market, analytic)
    if args.case == "digital":
        _run_strata_scan(option, model, market)


if __name__ == "__main__":
    main()
