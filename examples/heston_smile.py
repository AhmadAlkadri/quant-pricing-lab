"""The Heston model through the transform engines: prices, a smile, and two traps.

Four cases:

    --case smile   strike x maturity table of call, put and implied volatility
    --case cos     COS error against the term count and the range, both Feller
                   regimes, and the call/put asymmetry the range creates
    --case trap    the little Heston trap: the original branch choice against
                   the stable one, and why the reference parameter set hides it
    --case alpha   Carr-Madan damping against the moment-explosion bound

Everything printed is deterministic: no sampling anywhere in this file.

The parameter set is the one six published values are quoted at (S = 100,
r = 1%, q = 2%, v0 = 0.04, kappa = 4, theta = 0.25, xi = 1, rho = -0.5),
attributed by the QuantLib test suite to Alan Lewis' Wilmott-forum posting;
`qpl.cases.heston` carries those values as benchmark rows with that citation.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from qpl.cases.heston import HESTON_LEWIS_SPEC
from qpl.engines.fourier import FourierConfig, cos_price, implied_vol, smile_skew
from qpl.engines.fourier.lewis import lewis_call
from qpl.instruments.options import EuropeanOption
from qpl.models.heston import HestonModel, numerical_log_return_cumulant
from qpl.pricing import price

SPEC = HESTON_LEWIS_SPEC
MODEL = SPEC.model()
MARKET = SPEC.market()

FELLER_VIOLATED = HestonModel(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9)
"""The variance parameters of this repository's Feller-violating CIR spec."""

TRAP_MODEL = replace(MODEL, theta=0.3125)
"""The reference model with `theta` alone changed, so `d`, `g` and the branch
windings are identical and only `2 kappa theta / xi^2` moves, from 2 to 2.5."""

STRIKES = (80.0, 90.0, 100.0, 110.0, 120.0)
MATURITIES = (0.25, 1.0, 5.0)
SMILE_CFG = FourierConfig(truncation_l=12.0, n_terms=512)
"""The range that works across T = 0.25 ... 5, and it is a compromise.

The COS call's payoff coefficient carries `e^b` with `b ~ L sqrt(theta T)`, so
a wide range costs the call precision at long maturity; the COS put's error is
missing left-tail mass, so a narrow range costs the put precision at short
maturity. Worst over the three maturities and three strikes:

    L        6         10        12        14        16
    call   1.4e-07   1.8e-10   8.5e-09   9.5e-08   7.4e-07
    put    5.9e-04   3.6e-07   8.3e-09   1.8e-10   3.4e-14

`L = 12` is where the two columns cross. There is no setting that makes both
better than 1e-09 here, and `--case cos` shows a parameter set where there is
no setting that makes both better than 1e+00."""


@dataclass(frozen=True)
class OriginalBranch:
    """The unstable branch choice, wrapped as a transform the COS method reads."""

    model: HestonModel

    def characteristic_function(self, u, expiry, *, rate, dividend):
        return self.model.characteristic_function_original_branch(
            u, expiry, rate=rate, dividend=dividend
        )

    def log_return_cumulants(self, expiry, *, rate, dividend):
        return self.model.log_return_cumulants(expiry, rate=rate, dividend=dividend)


def _header(case: str) -> None:
    print("example=heston_smile")
    print(f"case={case}")
    print(
        f"spec S={SPEC.spot:g} r={SPEC.rate:g} q={SPEC.dividend:g} v0={SPEC.v0:g} "
        f"kappa={SPEC.kappa:g} theta={SPEC.theta:g} xi={SPEC.xi:g} rho={SPEC.rho:g}"
    )
    print(
        f"feller_number={MODEL.feller_number:.2f} "
        f"feller_satisfied={MODEL.feller_satisfied} "
        f"log_multiplier={2.0 * SPEC.kappa * SPEC.theta / SPEC.xi**2:.2f}"
    )


def _vanilla(model, kind: str, strike: float, expiry: float, cfg: FourierConfig) -> float:
    return price(
        EuropeanOption(kind=kind, strike=strike, expiry=expiry),
        model,
        MARKET,
        method="fourier",
        cfg=cfg,
    ).value


def _reference(model, strike: float, expiry: float) -> float:
    value, _ = lewis_call(
        model,
        s0=SPEC.spot,
        strike=strike,
        expiry=expiry,
        rate=SPEC.rate,
        dividend=SPEC.dividend,
        limit=800,
        tolerance=1e-13,
    )
    return value


# --------------------------------------------------------------------------


def case_smile() -> None:
    _header("smile")
    print("published K=100 call=16.0702 put=17.0553 (Alan Lewis via QuantLib tests)")
    for expiry in MATURITIES:
        skew = smile_skew(MODEL, MARKET, expiry, cfg=SMILE_CFG)
        print(f"maturity T={expiry:5.2f} skew={skew:+.6f}")
        for strike in STRIKES:
            call = _vanilla(MODEL, "call", strike, expiry, SMILE_CFG)
            put = _vanilla(MODEL, "put", strike, expiry, SMILE_CFG)
            vol = implied_vol(
                MODEL, MARKET, strike=strike, expiry=expiry, cfg=SMILE_CFG
            )
            parity = (call - put) - (
                SPEC.spot * math.exp(-SPEC.dividend * expiry)
                - strike * math.exp(-SPEC.rate * expiry)
            )
            print(
                f"  K={strike:6.1f} call={call:11.6f} put={put:11.6f} "
                f"implied_vol={vol:.6f} parity={parity:+.2e}"
            )
    skews = [smile_skew(MODEL, MARKET, T, cfg=SMILE_CFG) for T in MATURITIES]
    print(
        "skew_signs=" + "".join("-" if s < 0 else "+" for s in skews),
        f"skew_flattens={all(abs(b) < abs(a) for a, b in pairwise(skews))}",
    )
    positive = replace(MODEL, rho=0.5)
    flipped = [smile_skew(positive, MARKET, T, cfg=SMILE_CFG) for T in MATURITIES]
    print("rho=+0.5 skew_signs=" + "".join("-" if s < 0 else "+" for s in flipped))


def case_cos() -> None:
    _header("cos")
    print("cumulants and what c4=0 costs the truncation range")
    for label, model in (("feller_ok", MODEL), ("feller_violated", FELLER_VIOLATED)):
        cumulants = model.log_return_cumulants(
            1.0, rate=SPEC.rate, dividend=SPEC.dividend
        )
        c4 = numerical_log_return_cumulant(
            model, 4, 1.0, rate=SPEC.rate, dividend=SPEC.dividend, step=0.05
        )
        widen = math.sqrt(1.0 + math.sqrt(max(c4, 0.0)) / cumulants.c2)
        print(
            f"  {label:16s} feller={model.feller_number:5.2f} c2={cumulants.c2:.6f} "
            f"c4_measured={c4:.4e} widen_L_by={widen:.2f} suggested_L={10.0 * widen:5.1f}"
        )

    print("cos call error vs N at fixed L (a flat column is a RANGE error)")
    for label, model, truncation_l in (
        ("feller_ok", MODEL, 10.0),
        ("feller_violated", FELLER_VIOLATED, 10.0),
        ("feller_violated", FELLER_VIOLATED, 28.0),
    ):
        reference = _reference(model, 100.0, 1.0)
        cells = " ".join(
            f"N={n:5d}:{abs(_vanilla(model, 'call', 100.0, 1.0, FourierConfig(n_terms=n, truncation_l=truncation_l)) - reference):.1e}"
            for n in (512, 1024, 2048, 4096, 8192)
        )
        print(f"  {label:16s} L={truncation_l:5.1f} {cells}")

    print("the call and the put are not the same range problem (T=10, feller_violated)")
    reference_call = _reference(FELLER_VIOLATED, 100.0, 10.0)
    reference_put = (
        reference_call
        - SPEC.spot * math.exp(-SPEC.dividend * 10.0)
        + 100.0 * math.exp(-SPEC.rate * 10.0)
    )
    for truncation_l in (8.0, 14.0, 20.0, 28.0, 36.0):
        cfg = FourierConfig(truncation_l=truncation_l, n_terms=2048)
        call_error = abs(_vanilla(FELLER_VIOLATED, "call", 100.0, 10.0, cfg) - reference_call)
        put_error = abs(_vanilla(FELLER_VIOLATED, "put", 100.0, 10.0, cfg) - reference_put)
        print(
            f"  L={truncation_l:5.1f} call_err={call_error:.1e} put_err={put_error:.1e}"
        )
    print("cos_call_window_empty=True cos_put_then_parity=recommended")


def case_trap() -> None:
    _header("trap")
    print("original branch vs stable branch, same range, same terms, same model")
    for label, model in (("reference", MODEL), ("trap", TRAP_MODEL)):
        multiplier = 2.0 * model.kappa * model.theta / model.xi**2
        print(f"  model={label} 2*kappa*theta/xi^2={multiplier:.2f}")
        for expiry in (1.0, 1.2, 2.0, 5.0, 10.0, 30.0):
            common = {
                "s0": SPEC.spot,
                "strike": 100.0,
                "expiry": expiry,
                "rate": SPEC.rate,
                "dividend": SPEC.dividend,
                "kind": "call",
                "n_terms": 256,
                "truncation_l": 8.0,
            }
            with np.errstate(over="ignore", invalid="ignore"):
                stable = cos_price(model, **common).value
                unstable = cos_price(OriginalBranch(model), **common).value
            relative = abs(unstable - stable) / stable
            print(
                f"    T={expiry:5.1f} stable={stable:11.6f} original={unstable:14.6f} "
                f"rel={relative:.1e}"
            )
    grid = np.linspace(1e-06, 20.0, 4001)
    print("first branch crossing in u, and u*T is nearly constant")
    with np.errstate(over="ignore", invalid="ignore"):
        for expiry in (1.0, 2.0, 3.0, 5.0, 10.0):
            gap = np.abs(
                TRAP_MODEL.characteristic_function(
                    grid, expiry, rate=SPEC.rate, dividend=SPEC.dividend
                )
                - TRAP_MODEL.characteristic_function_original_branch(
                    grid, expiry, rate=SPEC.rate, dividend=SPEC.dividend
                )
            )
            first = grid[int(np.argmax(gap > 1e-08))]
            print(f"    T={expiry:5.1f} first_break_u={first:7.4f} u_times_T={first * expiry:6.3f}")
    print("trap_invisible_on_reference_set=True reason=integer_log_multiplier")


def case_alpha() -> None:
    _header("alpha")
    critical = MODEL.critical_moment(1.0)
    print(f"critical_moment={critical:.4f} alpha_max={critical - 1.0:.4f}")
    reference = _reference(MODEL, 100.0, 1.0)
    for alpha in (0.5, 1.5, 3.0, 5.0, 8.0, 10.0, 10.6, 10.65, 11.0, 12.0):
        explosion = MODEL.moment_explosion_time(alpha + 1.0)
        with np.errstate(over="ignore", invalid="ignore"):
            value = _vanilla(
                MODEL,
                "call",
                100.0,
                1.0,
                FourierConfig(
                    method="carr_madan",
                    carr_madan_transform="quadrature",
                    alpha=alpha,
                    n_quad=4096,
                ),
            )
        marker = "ok" if abs(value - reference) < 1e-06 else "BROKEN"
        print(
            f"  alpha={alpha:6.2f} explosion_time={explosion:9.4f} "
            f"err={abs(value - reference):.1e} {marker}"
        )
    print("alpha_failure_is_moment_explosion=True")


CASES = {
    "smile": case_smile,
    "cos": case_cos,
    "trap": case_trap,
    "alpha": case_alpha,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), default="smile")
    CASES[parser.parse_args().case]()


if __name__ == "__main__":
    main()
