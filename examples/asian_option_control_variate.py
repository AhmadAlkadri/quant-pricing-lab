"""Pricing an arithmetic Asian with the geometric average as the control.

One table per fixing count. Each row is one estimator and carries the paths it
ran, the standard normals it spent (the cost axis: an Asian path costs one
normal per fixing), the price, the reported standard error, and the variance
factor read two ways -- from the standard errors of a single run and from the
spread of twenty independent seeds. The control-variate rows also print the
realised correlation with the control and the factor `1/(1-rho^2)` it predicts.

Around each table: the exact discrete geometric closed form (which is both a
price in its own right and the control's known mean), a 200 000-path
control-variate reference price with its standard error, the Turnbull-Wakeman
approximation, and the **signed gap** between that approximation and the
reference. The gap is the point of printing it: the approximation reproduces its
own published benchmark to 1e-05 and is 33 to 38 standard errors away from the
simulated price, always on the same side.

`--case fixings` runs the other direction instead: the geometric closed form
against its continuous-averaging limit as the fixing grid is refined, which is
order 1 in the number of fixings and not order 2.

Deterministic: every seed is fixed and the script prints the same bytes on every
run.

Reference: Kemna, A.G.Z. and Vorst, A.C.F. (1990), *Journal of Banking and
Finance* 14, 113-129, for the geometric closed form and the control variate;
Turnbull & Wakeman (1991), *JFQA* 26(3), for the moment-matching approximation;
Glasserman (2003) section 4.1 for the estimator. Derivation and full tables:
`docs/notes/asian_options_control_variate.md`.
"""

from __future__ import annotations

import argparse
import math
from itertools import pairwise

import numpy as np

from qpl.engines.analytic.asian import (
    expected_arithmetic_average,
    turnbull_wakeman_price,
)
from qpl.engines.mc.asian import asian_terminal_sample
from qpl.engines.mc.pricers import MCConfig
from qpl.instruments.options import AsianOption, uniform_fixing_times
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price

SPOT, STRIKE, EXPIRY, SIGMA, RATE, DIVIDEND = 100.0, 100.0, 1.0, 0.20, 0.05, 0.0
N_DRAWS = 41_600
"""`4160 * 10 == 800 * 52`: both fixing counts spend the same normals."""

FIXING_COUNTS = (10, 52)
SEEDS = tuple(range(1, 21))
REPORT_SEED = 20250913
PILOT_PATHS = 40_000
PILOT_SEED = 7
REFERENCE_PATHS = 200_000
REFERENCE_SEED = 7
"""The gap to the approximation is printed from a dedicated 200 000-path
control-variate run rather than from the table's 800-to-4160-path rows. At the
table's cost the standard error is 3.6e-03 to 6.5e-03 and the gap (about 0.018)
is only three to five standard errors -- visible, but not a number worth
printing to six decimals. The table exists to compare estimators at equal cost;
the gap line exists to state a bias."""

ESTIMATORS: tuple[tuple[str, str | tuple[str, ...]], ...] = (
    ("none", "none"),
    ("antithetic", "antithetic"),
    ("control_variate", "control_variate"),
    ("antithetic+control", ("antithetic", "control_variate")),
)

ORDER_LEVELS = (20, 40, 80, 160, 320, 640, 1280, 2560)


def _market() -> Market:
    return Market(
        spot=SPOT,
        rate_curve=FlatRateCurve(RATE),
        dividend_curve=FlatDividendCurve(DIVIDEND),
    )


def _option(n_fixings: int, averaging: str = "arithmetic") -> AsianOption:
    return AsianOption(
        kind="call",
        strike=STRIKE,
        expiry=EXPIRY,
        fixing_times=uniform_fixing_times(EXPIRY, n_fixings),
        averaging=averaging,
    )


def _paths_for(vr: str | tuple[str, ...], n_fixings: int) -> int:
    """An Asian path costs one normal per fixing; antithetic reuses each twice."""
    base = N_DRAWS // n_fixings
    return 2 * base if "antithetic" in vr else base


def _cfg(vr: str | tuple[str, ...], n_fixings: int, seed: int) -> MCConfig:
    return MCConfig(
        n_paths=_paths_for(vr, n_fixings),
        n_steps=1,
        seed=seed,
        variance_reduction=vr,
    )


def _seed_variance(option: AsianOption, model, market, vr) -> float:
    values = np.array(
        [
            price(
                option,
                model,
                market,
                method="mc",
                cfg=_cfg(vr, option.n_fixings, seed),
            ).value
            for seed in SEEDS
        ]
    )
    return float(values.var(ddof=1))


def _pilot_correlation(option: AsianOption, model, market) -> float:
    """`corr(arithmetic payoff, geometric payoff)` on an independent pilot."""
    sample = asian_terminal_sample(
        option,
        model,
        market,
        cfg=MCConfig(
            n_paths=PILOT_PATHS, seed=PILOT_SEED, variance_reduction="control_variate"
        ),
        methods=("control_variate",),
    )
    return float(np.corrcoef(sample.y, sample.x)[0, 1])


def _run_table(n_fixings: int, model, market) -> None:
    option = _option(n_fixings)
    geometric = price(_option(n_fixings, "geometric"), model, market).value
    approximation = turnbull_wakeman_price(
        S=SPOT,
        K=STRIKE,
        T=EXPIRY,
        r=market.rate(EXPIRY),
        sigma=SIGMA,
        fixing_times=option.fixing_times,
        q=market.dividend_yield(EXPIRY),
    )
    rho = _pilot_correlation(option, model, market)

    print(f"fixings={n_fixings:3d} draws={N_DRAWS} seeds={len(SEEDS)}")
    print(f"  geometric_closed_form={geometric:.10f}")
    print(f"  expected_arithmetic_average={expected_arithmetic_average(S=SPOT, mu=market.rate(EXPIRY) - market.dividend_yield(EXPIRY), fixing_times=option.fixing_times):.10f}")
    print(f"  pilot_rho={rho:.6f} predicted_factor={1.0 / (1.0 - rho * rho):9.1f}")

    plain_stderr = None
    plain_variance = None
    for label, vr in ESTIMATORS:
        res = price(
            option, model, market, method="mc", cfg=_cfg(vr, n_fixings, REPORT_SEED)
        )
        seed_variance = _seed_variance(option, model, market, vr)
        if plain_stderr is None:
            plain_stderr = res.stderr
            plain_variance = seed_variance
        meta = res.meta or {}
        realised = meta.get("control_correlation")
        rho_column = f"{realised:.6f}" if realised is not None else "        "
        print(
            f"  estimator={label:18s} "
            f"paths={meta['n_paths']:6d} "
            f"normals={meta['n_normal_draws']:6d} "
            f"price={res.value:.6f} "
            f"stderr={res.stderr:.6f} "
            f"stderr_factor={(plain_stderr / res.stderr) ** 2:9.1f} "
            f"seed_factor={plain_variance / seed_variance:9.1f} "
            f"rho={rho_column}"
        )

    reference = price(
        option,
        model,
        market,
        method="mc",
        cfg=MCConfig(
            n_paths=REFERENCE_PATHS,
            seed=REFERENCE_SEED,
            variance_reduction="control_variate",
        ),
    )
    gap = approximation - reference.value
    print(
        f"  reference_cv_mc={reference.value:.6f} stderr={reference.stderr:.6f} "
        f"paths={REFERENCE_PATHS}"
    )
    print(f"  turnbull_wakeman={approximation:.10f}")
    print(
        f"  tw_minus_reference={gap:+.6f} ({gap / reference.value * 100:+.3f}%) "
        f"= {gap / reference.stderr:+.0f} stderr"
    )
    print(f"  arithmetic_minus_geometric={reference.value - geometric:+.6f}")


def _continuous_geometric() -> float:
    """The Kemna-Vorst continuous-averaging limit, written out."""
    mu = RATE - DIVIDEND
    m = math.log(SPOT) + (mu - 0.5 * SIGMA * SIGMA) * EXPIRY / 2.0
    v = SIGMA * SIGMA * EXPIRY / 3.0
    forward = math.exp(m + 0.5 * v)
    sd = math.sqrt(v)
    d1 = (math.log(forward / STRIKE) + 0.5 * v) / sd
    d2 = d1 - sd
    ncdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))  # noqa: E731
    return math.exp(-RATE * EXPIRY) * (forward * ncdf(d1) - STRIKE * ncdf(d2))


def _run_fixing_scan(model, market) -> None:
    """The discrete geometric price against its continuous limit: order 1."""
    from qpl.validation import fit_convergence_order

    limit = _continuous_geometric()
    print(f"continuous_limit={limit:.10f}")
    errors = []
    for n in ORDER_LEVELS:
        value = price(_option(n, "geometric"), model, market).value
        errors.append(abs(value - limit))
        print(f"  fixings={n:5d} geometric={value:.10f} error={value - limit:+.6e}")
    fit = fit_convergence_order([1.0 / n for n in ORDER_LEVELS], errors)
    print(f"fixing_order={fit.order:+.4f} residual={fit.residual:.5f}")
    halving = all(
        0.45 < b / a < 0.55 for a, b in pairwise(errors)
    )
    print(f"errors_halve_at_each_level={halving}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("control_variate", "fixings"), default="control_variate")
    args = parser.parse_args()

    market = _market()
    model = BlackScholesModel(sigma=SIGMA)

    print("example=asian_option_control_variate")
    print(f"case={args.case}")
    print(
        f"spot={SPOT:g} strike={STRIKE:g} expiry={EXPIRY:g} "
        f"sigma={SIGMA:g} rate={RATE:g} dividend={DIVIDEND:g}"
    )
    if args.case == "control_variate":
        for n_fixings in FIXING_COUNTS:
            _run_table(n_fixings, model, market)
    else:
        _run_fixing_scan(model, market)


if __name__ == "__main__":
    main()
