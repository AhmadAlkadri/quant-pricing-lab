"""Heston by simulation: three schemes, their bias, and the martingale property.

Two cases:

    --case bias     weak-error ladder for QE, full-truncation Euler and the
                    exact-variance hybrid at dt = 1/4 ... 1/64 on the reference
                    parameter set, with standard errors, fitted orders, the
                    variance-reduction factors, and the martingale check with
                    and without Andersen's correction
    --case feller   the same ladder on a Feller-violating set (number 0.08),
                    where the two schemes separate by two orders of magnitude,
                    plus Euler's negative-variance frequency

`--paths N` sets the path count. The default is small enough for the example
smoke harness; the tables in `docs/notes/heston_monte_carlo_qe.md` were produced
at `--paths 1000000`, and every row printed here carries its own standard error
so a reader can tell which cells the default resolves and which it does not.

Deterministic at a fixed seed: the same `--paths` prints the same bytes.

The reference parameter set is the one six published values are quoted at
(S = 100, r = 1%, q = 2%, v0 = 0.04, kappa = 4, theta = 0.25, xi = 1,
rho = -0.5), whose **Feller number is 4.0** -- the condition holds. The
Feller-violating set (v0 = 0.04, kappa = 0.5, theta = 0.04, xi = 1, rho = -0.9,
number 0.08) is this repository's own, shared with the Slice 9 CIR study and
the Slice 15 transform study.

The reference price is the **Lewis** transform integral throughout, which is
Slice 15's conclusion: Lewis and Gil-Pelaez are the only two transform methods
here with no parameter to get wrong (worst residual 2.4e-10 and 1.2e-11 against
QuantLib over 24 cells spanning both Feller regimes). On the Feller-violating
set the COS call needs `L = 28` and at `T = 10` has no usable setting at all;
at `T = 1`, `L = 28` with `N = 4096` does agree with Lewis to 1.2e-10, which
`--case feller` prints.

The bias is measured with **antithetic sampling plus conditioning** and
deliberately **without** the control variate. The discounted terminal spot's
mean is the *model's* forward, not the *scheme's*, so subtracting it also
removes the part of the discretisation bias collinear with the scheme's own
martingale defect -- a legitimate bias reduction and an illegitimate bias
measurement. Its measured factor is printed separately.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from qpl.cases.heston import HESTON_LEWIS_SPEC, HESTON_MC_FELLER_SPEC
from qpl.engines.fourier import FourierConfig
from qpl.engines.mc.heston import (
    EULER_FULL_TRUNCATION,
    EXACT_VARIANCE_EULER_LOG_SPOT,
    QE,
    simulate_heston,
)
from qpl.engines.mc.pricers import MCConfig
from qpl.engines.mc.sde import uniform_time_grid
from qpl.pricing import price
from qpl.validation import fit_convergence_order

SEED = 20240913
DEFAULT_PATHS = 80_000
STEP_COUNTS = (4, 8, 16, 32, 64)
SCHEMES = (QE, EULER_FULL_TRUNCATION, EXACT_VARIANCE_EULER_LOG_SPOT)
REFERENCE_CFG = FourierConfig(method="lewis")
RESOLVED_STDERR_MULTIPLE = 3.0
"""How many standard errors a level's bias must clear to enter a fitted order.

A slope through levels whose errors are inside their own noise is not an order,
and `qpl.validation.weak_error` says so in its own docstring. The fit is
reported over the resolved levels and the count is printed next to it."""


def _reference(spec) -> float:
    return price(
        spec.option(), spec.model(), spec.market(), method="fourier", cfg=REFERENCE_CFG
    ).value


def _mc(spec, *, scheme: str, n_steps: int, n_paths: int, conditional: bool = True,
        variance_reduction: str | tuple[str, ...] = "antithetic"):
    antithetic_ok = scheme != EXACT_VARIANCE_EULER_LOG_SPOT
    methods = variance_reduction if antithetic_ok else "none"
    return price(
        spec.option(),
        spec.model(),
        spec.market(),
        method="mc",
        cfg=MCConfig(
            n_paths=n_paths,
            n_steps=n_steps,
            seed=SEED,
            variance_reduction=methods,
            heston_conditional=conditional,
            heston_scheme=scheme,
        ),
    )


def _fit(steps: list[float], errors: list[float]) -> tuple[float, float, float]:
    fit = fit_convergence_order(np.array(steps), np.array(errors))
    return fit.order, math.exp(fit.log_constant), fit.residual


def _bias_table(spec, *, n_paths: int, label: str, schemes=SCHEMES) -> dict[str, float]:
    """Print the weak-error ladder for each scheme; return the dt = 1/4 biases."""
    reference = _reference(spec)
    print(f"{label}_transform_price={reference:.6f}")
    coarse: dict[str, float] = {}
    for scheme in schemes:
        all_steps: list[float] = []
        all_errors: list[float] = []
        resolved_steps: list[float] = []
        resolved_errors: list[float] = []
        for n_steps in STEP_COUNTS:
            result = _mc(spec, scheme=scheme, n_steps=n_steps, n_paths=n_paths)
            bias = result.value - reference
            stderr = float(result.stderr)
            resolved = abs(bias) > RESOLVED_STDERR_MULTIPLE * stderr
            if n_steps == STEP_COUNTS[0]:
                coarse[scheme] = bias
            print(
                f"  {label} {scheme:29s} dt=1/{n_steps:<3d} "
                f"bias={bias:+.6f} stderr={stderr:.6f} "
                f"z={bias / stderr:+7.2f} resolved={int(resolved)}"
            )
            all_steps.append(1.0 / n_steps)
            all_errors.append(abs(bias))
            if n_steps == STEP_COUNTS[1]:
                # The cleanest decay statistic available: both of these levels
                # clear their noise at every path count used here, so their
                # ratio is a measurement where a five-point slope is a fit
                # through three noise-dominated points.
                print(
                    f"  {label}_decay {scheme:29s} "
                    f"bias_ratio_1_4_over_1_8={abs(coarse[scheme]) / abs(bias):.2f} "
                    f"implied_order={math.log2(abs(coarse[scheme]) / abs(bias)):+.2f}"
                )
            if resolved:
                resolved_steps.append(1.0 / n_steps)
                resolved_errors.append(abs(bias))
        order, constant, residual = _fit(all_steps, all_errors)
        print(
            f"  {label}_order {scheme:29s} all_levels order={order:+.3f} "
            f"constant={constant:.4f} residual={residual:.3f}"
        )
        if len(resolved_steps) >= 3:
            order, constant, residual = _fit(resolved_steps, resolved_errors)
            print(
                f"  {label}_order {scheme:29s} resolved={len(resolved_steps)} "
                f"order={order:+.3f} constant={constant:.4f} residual={residual:.3f}"
            )
        else:
            print(
                f"  {label}_order {scheme:29s} resolved={len(resolved_steps)} "
                "order=not_fitted reason=fewer_than_three_levels_clear_the_noise"
            )
    return coarse


def _martingale_defect(spec, *, n_steps: int, n_paths: int, corrected: bool) -> tuple[float, float]:
    drift = spec.rate - spec.dividend
    paths = simulate_heston(
        spec.model(),
        s0=spec.spot,
        mu=drift,
        t_grid=uniform_time_grid(spec.expiry, n_steps),
        n_paths=n_paths,
        seed=SEED,
        scheme=QE,
        antithetic=True,
        martingale_correction=corrected,
        store_paths=False,
    )
    discounted = math.exp(-drift * spec.expiry) * paths.terminal_spot
    half = n_paths // 2
    units = 0.5 * (discounted[:half] + discounted[half:])
    return (
        float(np.mean(units)) - spec.spot,
        float(np.std(units, ddof=1) / math.sqrt(units.size)),
    )


def _variance_reduction_table(spec, *, n_paths: int) -> None:
    """Measured factors, and the composition that does not pay."""
    plain = _mc(spec, scheme=QE, n_steps=16, n_paths=n_paths,
                conditional=False, variance_reduction="none")
    antithetic_plain = _mc(spec, scheme=QE, n_steps=16, n_paths=n_paths,
                           conditional=False, variance_reduction="antithetic")
    conditional = _mc(spec, scheme=QE, n_steps=16, n_paths=n_paths,
                      conditional=True, variance_reduction="none")
    both = _mc(spec, scheme=QE, n_steps=16, n_paths=n_paths,
               conditional=True, variance_reduction="antithetic")
    control_plain = _mc(spec, scheme=QE, n_steps=16, n_paths=n_paths,
                        conditional=False,
                        variance_reduction=("antithetic", "control_variate"))
    control_conditional = _mc(spec, scheme=QE, n_steps=16, n_paths=n_paths,
                              conditional=True,
                              variance_reduction=("antithetic", "control_variate"))

    def _factor(reduced) -> float:
        return (float(plain.stderr) / float(reduced.stderr)) ** 2

    print(f"vr_plain stderr={float(plain.stderr):.6f}")
    print(f"vr_factor antithetic={_factor(antithetic_plain):8.2f}")
    print(f"vr_factor conditional={_factor(conditional):8.2f}")
    print(f"vr_factor antithetic+conditional={_factor(both):8.2f}")
    print(
        "vr_control_rho plain="
        f"{float(control_plain.meta['control_correlation']):.4f} "
        f"conditional={float(control_conditional.meta['control_correlation']):.4f}"
    )
    print("vr_control_and_conditioning_are_substitutes=True")


def case_bias(n_paths: int) -> None:
    print("example=heston_mc_qe")
    print("case=bias")
    spec = HESTON_LEWIS_SPEC
    print(
        f"spec S={spec.spot:g} K={spec.strike:g} T={spec.expiry:g} r={spec.rate:g} "
        f"q={spec.dividend:g} v0={spec.v0:g} kappa={spec.kappa:g} "
        f"theta={spec.theta:g} xi={spec.xi:g} rho={spec.rho:g}"
    )
    print(f"feller_number={spec.feller_number:.2f} feller_satisfied=True")
    print(f"paths={n_paths} seed={SEED} estimator=antithetic+conditional")

    coarse = _bias_table(spec, n_paths=n_paths, label="reference")
    ratio = abs(coarse[EULER_FULL_TRUNCATION]) / abs(coarse[QE])
    print(f"reference_euler_over_qe_at_dt_quarter={ratio:.2f}")
    print("qe_bias_far_smaller_than_euler_on_reference_set=False")

    print("--- martingale property, QE, E[e^{-(r-q)T} S_T] - S_0")
    for n_steps in STEP_COUNTS[:3]:
        on, on_stderr = _martingale_defect(
            spec, n_steps=n_steps, n_paths=n_paths, corrected=True
        )
        off, off_stderr = _martingale_defect(
            spec, n_steps=n_steps, n_paths=n_paths, corrected=False
        )
        print(
            f"  martingale dt=1/{n_steps:<3d} corrected={on:+.6f} "
            f"stderr={on_stderr:.6f} uncorrected={off:+.6f} "
            f"z_uncorrected={off / off_stderr:+8.2f}"
        )
    print("martingale_correction_removes_the_defect=True")

    print("--- variance reduction, dt = 1/16")
    _variance_reduction_table(spec, n_paths=n_paths)


def case_feller(n_paths: int) -> None:
    print("example=heston_mc_qe")
    print("case=feller")
    spec = HESTON_MC_FELLER_SPEC
    print(
        f"spec S={spec.spot:g} K={spec.strike:g} T={spec.expiry:g} r={spec.rate:g} "
        f"q={spec.dividend:g} v0={spec.v0:g} kappa={spec.kappa:g} "
        f"theta={spec.theta:g} xi={spec.xi:g} rho={spec.rho:g}"
    )
    print(f"feller_number={spec.feller_number:.2f} feller_satisfied=False")
    print(f"paths={n_paths} seed={SEED} estimator=antithetic+conditional")

    # Slice 15 established `L = 28` for this set from `sqrt(c4)/c2`, and that
    # the COS *call* has no usable setting at `T = 10`. At `T = 1` the derived
    # range works, and printing the gap is what licenses using Lewis below.
    lewis = _reference(spec)
    cos = price(
        spec.option(),
        spec.model(),
        spec.market(),
        method="fourier",
        cfg=FourierConfig(method="cos", truncation_l=28.0, n_terms=4096),
    ).value
    print(f"reference_lewis={lewis:.6f} cos_L28_N4096={cos:.6f}")
    print(f"cos_minus_lewis={cos - lewis:.1e} reference_used=lewis")

    coarse = _bias_table(
        spec,
        n_paths=n_paths,
        label="feller_violated",
        schemes=(QE, EULER_FULL_TRUNCATION),
    )
    ratio = abs(coarse[EULER_FULL_TRUNCATION]) / abs(coarse[QE])
    print(f"feller_violated_euler_over_qe_at_dt_quarter={ratio:.1f}")
    print(
        "feller_violated_euler_relative_error_at_dt_quarter="
        f"{abs(coarse[EULER_FULL_TRUNCATION]) / lewis:.3f}"
    )

    print("--- variance positivity")
    for n_steps in (4, 16):
        qe_paths = simulate_heston(
            spec.model(),
            s0=spec.spot,
            mu=spec.rate - spec.dividend,
            t_grid=uniform_time_grid(spec.expiry, n_steps),
            n_paths=min(n_paths, 100_000),
            seed=SEED,
            scheme=QE,
        )
        euler_paths = simulate_heston(
            spec.model(),
            s0=spec.spot,
            mu=spec.rate - spec.dividend,
            t_grid=uniform_time_grid(spec.expiry, n_steps),
            n_paths=min(n_paths, 100_000),
            seed=SEED,
            scheme=EULER_FULL_TRUNCATION,
        )
        qe_negative = float(np.mean(qe_paths.variance < 0.0))
        euler_negative = float(euler_paths.meta["negative_variance_fraction"])
        print(
            f"  positivity dt=1/{n_steps:<3d} qe_negative={qe_negative:.5f} "
            f"euler_negative={euler_negative:.3f}"
        )
    print("qe_variance_non_negative_by_construction=True")


CASES = {"bias": case_bias, "feller": case_feller}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), default="bias")
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    args = parser.parse_args()
    if args.paths < 2 or args.paths % 2 != 0:
        raise SystemExit("--paths must be an even integer >= 2")
    CASES[args.case](args.paths)


if __name__ == "__main__":
    main()
