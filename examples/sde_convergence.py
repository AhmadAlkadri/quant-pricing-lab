"""Strong and weak convergence of the Euler and Milstein schemes, measured.

Prints, for GBM, one row per (scheme, step count): the strong error
``E|X^h_T - X_T|`` on a **shared** Brownian path, the weak error in ``f(x) = x``
and in the call payoff, and each one's own standard error and z-score. Then the
fitted orders and constants.

The z-score columns are the point of the table, not decoration. The weak errors
are measured by common random numbers against the exact sampler, so their noise
floor is the scheme's own *strong* error: Milstein's z-scores are flat down the
ladder and Euler's fall, because Euler's strong order is 1/2 while its weak
order is 1. Refining the grid makes the Euler weak measurement worse, and only
more paths help.

``--case cir`` runs the square-root diffusion instead: the frequency with which
plain Euler steps to a negative variance (and then to NaN), what Andersen's
full truncation does and does not repair, and the weak orders of full-truncation
Euler against the exact noncentral chi-square transition law, on both sides of
the Feller condition.

Deterministic: every seed is fixed and the script prints the same bytes on
every run.

References: Kloeden & Platen (1992) chapters 9-10 and Glasserman (2003)
sections 6.1-6.2 for the orders; Higham (2001), SIAM Review 43(3), section 5
for the coupled-increment design; Cox, Ingersoll & Ross (1985), Broadie & Kaya
(2006) and Andersen (2008) for the CIR parts. Derivation and full tables:
`docs/notes/sde_discretization.md`.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from qpl.cases import (
    CIR_LEVELS,
    CIR_NEGATIVITY_PATHS,
    CIR_NEGATIVITY_SEED,
    CIR_NEGATIVITY_STEPS,
    GBM_LEVELS,
    GBM_STUDY_SPEC,
)
from qpl.cases.sde_discretization import CIR_FELLER_OK, CIR_FELLER_VIOLATED, CIRSpec
from qpl.engines.mc.sde import (
    cir_expected_excess,
    cir_moments,
    cir_sde,
    coarsen_normals,
    gbm_sde,
    simulate,
    uniform_time_grid,
)
from qpl.models.black_scholes import bs_price
from qpl.validation import fit_convergence_order, strong_error, weak_error

GBM_PATHS = 50_000
GBM_SEED = 20250913
CIR_PATHS = 200_000
CIR_SEED = 4242
SCHEMES = ("euler", "milstein")


def _fit(levels, errors, expiry: float):
    return fit_convergence_order(
        [expiry / n for n in levels], [abs(e.value) for e in errors]
    )


def _run_gbm() -> None:
    spec = GBM_STUDY_SPEC
    model = gbm_sde(spec.mu, spec.sigma)
    fine = np.random.default_rng(GBM_SEED).normal(size=(GBM_PATHS, max(GBM_LEVELS)))

    print(f"paths={GBM_PATHS} seed={GBM_SEED} fine_steps={max(GBM_LEVELS)}")
    print(
        f"spec s0={spec.s0:g} mu={spec.mu:g} sigma={spec.sigma:g} "
        f"expiry={spec.expiry:g} strike={spec.strike:g}"
    )
    analytic = float(
        bs_price(
            S=spec.s0,
            K=spec.strike,
            T=spec.expiry,
            r=spec.mu,
            sigma=spec.sigma,
            q=0.0,
            kind="call",
        )
    )
    discount = math.exp(-spec.mu * spec.expiry)
    print(f"analytic_call={analytic:.10f}")

    fits: dict[str, float] = {}
    euler_bias: list[float] = []
    for scheme in SCHEMES:
        strong, weak_id, weak_call = [], [], []
        for n_steps in GBM_LEVELS:
            z = coarsen_normals(fine, n_steps)
            grid = uniform_time_grid(spec.expiry, n_steps)
            approx = simulate(
                model, spec.s0, grid, GBM_PATHS, scheme=scheme, normals=z
            ).terminal
            exact = simulate(
                model, spec.s0, grid, GBM_PATHS, scheme="exact", normals=z
            ).terminal
            strong.append(strong_error(approx, exact))
            weak_id.append(weak_error(approx, reference_values=exact))
            call = weak_error(
                np.maximum(approx - spec.strike, 0.0),
                reference_values=np.maximum(exact - spec.strike, 0.0),
            )
            weak_call.append(call)
            if scheme == "euler":
                euler_bias.append(discount * call.value)
            # E[X^h_T] = S0 (1 + mu h)^n exactly, for BOTH schemes: reported
            # as a z-score, since the sample mean carries the full O(1/sqrt N)
            # error of the payoff and not the tiny error of the coupled
            # difference.
            moment = spec.s0 * (1.0 + spec.mu * spec.expiry / n_steps) ** n_steps
            moment_z = (float(np.mean(approx)) - moment) / (
                float(np.std(approx, ddof=1)) / math.sqrt(approx.size)
            )
            print(
                f"{scheme:8s} n={n_steps:4d} "
                f"strong={strong[-1].value:.6e} strong_z={strong[-1].z:7.1f} "
                f"weak_id={weak_id[-1].value:+.6e} weak_id_z={weak_id[-1].z:+7.1f} "
                f"weak_call={call.value:+.6e} weak_call_z={call.z:+7.1f} "
                f"geometric_moment_z={moment_z:+.2f}"
            )
        for label, errors in (
            ("strong", strong),
            ("weak_identity", weak_id),
            ("weak_call", weak_call),
        ):
            fit = _fit(GBM_LEVELS, errors, spec.expiry)
            fits[f"{label}_{scheme}"] = fit.order
            print(
                f"{label}_order {scheme}={fit.order:+.4f} "
                f"residual={fit.residual:.4f} "
                f"constant={math.exp(fit.log_constant):.4f}"
            )
        z_first, z_last = abs(weak_call[0].z), abs(weak_call[-1].z)
        print(
            f"weak_call_z_decay {scheme}={z_last / z_first:.3f} "
            f"(first={z_first:.1f} last={z_last:.1f})"
        )

    closed_form = spec.s0 * math.exp(spec.mu * spec.expiry) * spec.mu**2 * spec.expiry / 2.0
    print(f"weak_identity_closed_form_constant={closed_form:.4f}")
    print(f"milstein_over_euler_strong_order={fits['strong_milstein'] / fits['strong_euler']:.4f}")

    bias_fit = fit_convergence_order(
        [spec.expiry / n for n in GBM_LEVELS], [abs(b) for b in euler_bias]
    )
    for n_steps, bias in zip(GBM_LEVELS, euler_bias, strict=True):
        print(
            f"price n={n_steps:4d} euler_price={analytic + bias:.6f} "
            f"bias={bias:+.6f} bias_pct={100.0 * bias / analytic:+.4f}"
        )
    print(
        f"price_bias_order={bias_fit.order:+.4f} residual={bias_fit.residual:.4f} "
        f"constant={math.exp(bias_fit.log_constant):.4f}"
    )


def _negativity(spec: CIRSpec, label: str) -> None:
    model = cir_sde(**spec.params)
    grid = uniform_time_grid(spec.expiry, CIR_NEGATIVITY_STEPS)
    plain = simulate(
        model,
        spec.v0,
        grid,
        CIR_NEGATIVITY_PATHS,
        scheme="euler",
        truncation="none",
        seed=CIR_NEGATIVITY_SEED,
    ).values
    full = simulate(
        model,
        spec.v0,
        grid,
        CIR_NEGATIVITY_PATHS,
        scheme="euler",
        truncation="full",
        seed=CIR_NEGATIVITY_SEED,
    ).values
    negative_plain = float(np.mean(np.nanmin(plain, axis=1) < 0.0))
    negative_full = float(np.mean(np.min(full, axis=1) < 0.0))
    print(
        f"negativity {label} feller_number={spec.feller_number:.2f} "
        f"plain_negative={negative_plain:.5f} full_negative={negative_full:.5f} "
        f"plain_nan={float(np.mean(np.isnan(plain).any(axis=1))):.5f} "
        f"full_nan={float(np.mean(np.isnan(full).any(axis=1))):.5f} "
        f"full_min={float(np.min(full)):+.4e}"
    )


def _run_cir() -> None:
    print(
        f"negativity_settings steps={CIR_NEGATIVITY_STEPS} "
        f"paths={CIR_NEGATIVITY_PATHS} seed={CIR_NEGATIVITY_SEED}"
    )
    print(f"weak_settings paths={CIR_PATHS} seed={CIR_SEED} levels={list(CIR_LEVELS)}")
    for label, spec in (
        ("feller_ok", CIR_FELLER_OK),
        ("feller_violated", CIR_FELLER_VIOLATED),
    ):
        print(
            f"spec {label} v0={spec.v0:g} kappa={spec.kappa:g} theta={spec.theta:g} "
            f"xi={spec.xi:g} strike={spec.strike:g} "
            f"feller_satisfied={spec.feller_satisfied}"
        )
        _negativity(spec, label)

        mean, variance = cir_moments(spec.v0, spec.expiry, **spec.params)
        reference = cir_expected_excess(
            spec.v0, spec.expiry, strike=spec.strike, **spec.params
        )
        print(
            f"exact {label} mean={mean:.8f} sd={math.sqrt(variance):.8f} "
            f"call={reference:.8e}"
        )
        mean_errors, call_errors = [], []
        for n_steps in CIR_LEVELS:
            terminal = simulate(
                cir_sde(**spec.params),
                spec.v0,
                uniform_time_grid(spec.expiry, n_steps),
                CIR_PATHS,
                scheme="euler",
                truncation="full",
                seed=CIR_SEED,
            ).terminal
            mean_errors.append(weak_error(terminal, reference_mean=mean))
            call_errors.append(
                weak_error(
                    np.maximum(terminal - spec.strike, 0.0), reference_mean=reference
                )
            )
            print(
                f"cir {label} n={n_steps:3d} "
                f"weak_mean={mean_errors[-1].value:+.6e} "
                f"z={mean_errors[-1].z:+6.1f} "
                f"weak_call={call_errors[-1].value:+.6e} "
                f"z={call_errors[-1].z:+6.1f} "
                f"negative_fraction={float(np.mean(terminal < 0.0)):.5f}"
            )
        for name, errors in (("mean", mean_errors), ("call", call_errors)):
            fit = _fit(CIR_LEVELS, errors, spec.expiry)
            max_z = max(abs(e.z) for e in errors)
            print(
                f"cir_order {label} {name}={fit.order:+.4f} "
                f"residual={fit.residual:.4f} "
                f"constant={math.exp(fit.log_constant):.4e} max_z={max_z:.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("gbm", "cir"), default="gbm")
    args = parser.parse_args()

    print("example=sde_convergence")
    print(f"case={args.case}")
    if args.case == "gbm":
        _run_gbm()
    else:
        _run_cir()


if __name__ == "__main__":
    main()
