import os
import re
import subprocess
import sys

import pytest

# (script name, command-line arguments, output keys that must appear).
#
# A script may appear more than once, once per curated invocation: the same
# example run two ways is two smoke cases, not one. The pytest id is built from
# the script name plus the arguments so the two are distinguishable in a
# failure report.
_SMOKE_CASES = [
    ("bs_analytic_greeks.py", [], ["example=bs_analytic_greeks", "price=", "delta=", "gamma="]),
    (
        "mc_pricing_and_stderr.py",
        [],
        ["example=mc_pricing_and_stderr", "analytic_price=", "mc_price=", "mc_stderr=", "z_score="],
    ),
    (
        "pde_theta_scheme.py",
        [],
        ["example=pde_theta_scheme", "analytic_price=", "theta=0.0", "theta=0.5", "theta=1.0"],
    ),
    (
        "tree_convergence.py",
        [],
        [
            "example=tree_convergence",
            "black_scholes=",
            "odd_order=",
            "even_order=",
            "odd_sign=above",
            "even_sign=below",
            "richardson_order=",
        ],
    ),
    (
        # The Leisen-Reimer leg of the same example. `sign=below` and the two
        # ratio keys are the curated part: a run that had lost the scheme would
        # still print an order, but not an order-2 sequence sitting three
        # decimal orders of magnitude inside CRR's.
        "tree_convergence.py",
        ["--scheme", "leisen-reimer"],
        [
            "example=tree_convergence",
            "scheme=leisen-reimer",
            "black_scholes=",
            "order=1.9",
            "sign=below",
            "error_ratio_vs_crr_at_n101=",
            "error_ratio_vs_crr_at_n801=",
            "richardson_order=",
        ],
    ),
    (
        # The smooth case: both time steppings land on order two in all three
        # grid Greeks, which is the baseline the start-up case departs from.
        "pde_greeks_demo.py",
        [],
        [
            "example=pde_greeks_demo",
            "case=smooth",
            "smooth_order theta_gamma=+2.",
            "smooth_order rannacher_gamma=+2.",
            "smooth_order rannacher_delta=+2.",
        ],
    ),
    (
        # The start-up case. `theta_gamma=-1.` is the curated part: a run that
        # had lost the pathology would still print an order, but not a NEGATIVE
        # one, and `rannacher_gamma=+1.8` pins the repair on the same grids.
        "pde_greeks_demo.py",
        ["--case", "startup"],
        [
            "example=pde_greeks_demo",
            "case=startup",
            "startup_order theta_gamma=-1.",
            "startup_order theta_theta=-1.",
            "startup_order rannacher_gamma=+1.8",
            "startup_order rannacher_delta=+2.",
        ],
    ),
    (
        # Two engines on the same American put. The curated keys are the ones a
        # run that had quietly lost the PSOR path would not print:
        # `psor_mean_sweeps` and `lcp_max_complementarity` exist only on the
        # finite-difference leg, `engine_gap=1.3` pins the cross-engine
        # agreement to one figure, and `pde_boundary_first_time=0.0000` next to
        # a non-zero `tree_boundary_first_time` is the grid-versus-lattice
        # difference the example exists to show.
        "american_put_cross_method.py",
        [],
        [
            "example=american_put_cross_method",
            "pde_american=6.0888",
            "tree_american=6.0902",
            "engine_gap=1.3",
            "early_exercise_premium=",
            "psor_mean_sweeps=",
            "psor_converged=True",
            "lcp_max_complementarity=",
            "lcp_min_constraint_slack=0.000e+00",
            "boundary t=",
            "pde_boundary_first_time=0.0000",
        ],
    ),
    (
        # One digital, four engines. The curated keys are the ones a run that
        # had lost the point of the example would not print: `tree_order
        # leisen-reimer=+1.99` next to `pde_order plain_cn_unaligned=+1.0` is
        # the whole slice in two lines -- the order-2 lattice on a jump and the
        # order-1 grid -- and `sawtooth_sign_changes=2` pins that CRR's error
        # off the money changes sign inside a single window of consecutive odd
        # n. The `mc_greeks` lines pin Slice 10's estimators: the pathwise one
        # is refused because its payoff derivative is identically zero, and the
        # bumped theta's z-score is three orders of magnitude larger than the
        # likelihood-ratio one on the same seed and the same paths.
        "digital_option_cross_method.py",
        [],
        [
            "example=digital_option_cross_method",
            "analytic=0.5323248155",
            "replication_residual=+0.000e+00",
            "tree_order crr=+1.0",
            "tree_order leisen-reimer=+1.99",
            "tree_error_ratio_crr_over_lr_at_n801=4546.4",
            "sawtooth_sign_changes=2",
            "pde_order plain_cn_unaligned=+1.0",
            "pde_order cn_unaligned_projected=+2.0",
            "pde_order rannacher_aligned=+1.9",
            "mc_stderr_order=+0.4999",
            "mc_greeks pathwise=refused_zero_derivative",
            "mc_greeks likelihood_ratio delta estimate=",
            "mc_greeks bump theta estimate=",
            "z=+531.",
        ],
    ),
    (
        # The finite-difference table on its own, which is where the
        # discontinuity pathology and its two repairs sit side by side.
        "digital_option_cross_method.py",
        ["--case", "pde"],
        [
            "case=pde",
            "pde plain_cn_unaligned n=800 price=0.5276356328",
            "pde rannacher_aligned n=800 price=0.5323250384",
            "pde_order plain_cn_unaligned=+1.0",
            "pde_order rannacher_unaligned_projected=+2.0",
        ],
    ),
    (
        # Six estimators on one price at a fixed cost. The curated keys are the
        # ones a run that had lost the point would not print: `paths= 40960
        # normals= 20480` is the equal-cost bookkeeping (antithetic buys two
        # paths per normal, so the comparison is at equal normals), and
        # `predicted_factor antithetic=4.0` next to the measured `seed_factor`
        # column is theory and measurement in the same table.
        "mc_variance_reduction.py",
        [],
        [
            "example=mc_variance_reduction",
            "case=vanilla",
            "analytic=10.4505835722",
            "pilot_rho_antithetic=-0.50",
            "predicted_factor antithetic=4.0",
            "predicted_factor control_variate=6.7",
            "estimator=none",
            "paths= 40960 normals= 20480",
            "estimator=stratified+control",
            "seed_factor=",
        ],
    ),
    (
        # The digital leg, which exists for the stratum scan:
        # `strata_gain_monotone_in_K=False` with `K= 16` beating `K= 20` by
        # 2.9x is the finding -- for a discontinuous payoff the gain is set by
        # where the jump falls inside its stratum, so more strata can mean less
        # gain, and the printed `predicted` column says so before the measured
        # one does.
        "mc_variance_reduction.py",
        ["--case", "digital"],
        [
            "case=digital",
            "analytic=0.5323248155",
            "pilot_rho_antithetic=-0.78",
            "jump_at_u=0.440382",
            "strata_scan K= 16 f=0.0461",
            "strata_scan K= 20 f=0.8076",
            "strata_gain_monotone_in_K=False",
            "strata_gain_K16_over_K20=2.9",
        ],
    ),
    (
        # The Kemna-Vorst control variate on an arithmetic Asian, at two fixing
        # counts and equal normal draws. The curated keys are the ones a run
        # that had lost the point would not print: `pilot_rho=0.999608` next to
        # `predicted_factor=   1276.7` is the whole slice in one line (the
        # control is worth three orders of magnitude because rho is 0.9996,
        # against the 7.6 the terminal spot buys on a vanilla in Slice 7), and
        # `tw_minus_reference=+0.017777` pins the approximation's bias with its
        # sign -- a number that must not quietly become zero.
        "asian_option_control_variate.py",
        [],
        [
            "example=asian_option_control_variate",
            "case=control_variate",
            "fixings= 10",
            "fixings= 52",
            "geometric_closed_form=6.0191160793",
            "geometric_closed_form=5.6374316204",
            "pilot_rho=0.999608",
            "predicted_factor=   1276.7",
            "estimator=none",
            "estimator=antithetic+control",
            "paths=  4160 normals= 41600",
            "paths=   800 normals= 41600",
            "turnbull_wakeman=6.2523156852",
            "tw_minus_reference=+0.017777",
            "arithmetic_minus_geometric=+0.215",
        ],
    ),
    (
        # The fixing-refinement leg, which exists for one number:
        # `fixing_order=+1.0002`. A right-endpoint fixing convention approaches
        # the continuous-averaging limit at order ONE, and
        # `errors_halve_at_each_level=True` is the assumption-free form of that.
        "asian_option_control_variate.py",
        ["--case", "fixings"],
        [
            "case=fixings",
            "continuous_limit=5.5468186338",
            "fixings=   20 geometric=5.7826160816",
            "fixings= 2560 geometric=5.5486582749",
            "fixing_order=+1.0002",
            "errors_halve_at_each_level=True",
        ],
    ),
    (
        # Euler against Milstein on one GBM, at five step counts, with the
        # exact sampler on the SAME Brownian path as both. The curated keys
        # are the ones a run that had lost the coupling or the point would not
        # print: `strong_order euler=+0.51` next to `milstein=+0.99` is the
        # slice in two lines, and `weak_call_z_decay euler=0.276` against
        # `milstein=0.981` is the finding -- the coupled weak estimator's
        # noise floor is the scheme's own STRONG error, so Euler's
        # signal-to-noise falls by 3.6x down the same ladder on which
        # Milstein's is flat.
        "sde_convergence.py",
        [],
        [
            "example=sde_convergence",
            "case=gbm",
            "analytic_call=19.6298871",
            "strong_order euler=+0.51",
            "strong_order milstein=+0.99",
            "weak_identity_order euler=+0.99",
            "weak_identity_order milstein=+0.99",
            "weak_call_order euler=+1.00",
            "weak_call_order milstein=+0.99",
            "weak_call_z_decay euler=0.27",
            "weak_call_z_decay milstein=0.98",
            "weak_identity_closed_form_constant=2.4428",
            "price_bias_order=+1.00",
            "bias_pct=-1.29",
        ],
    ),
    (
        # The square-root diffusion, where Euler meets a boundary it cannot
        # cross. `plain_negative=0.911` equal to `full_negative=0.911` with
        # `full_nan=0.00000` is the whole CIR finding: full truncation keeps
        # the recursion defined and changes the negative frequency not at all.
        "sde_convergence.py",
        ["--case", "cir"],
        [
            "case=cir",
            "feller_satisfied=True",
            "feller_satisfied=False",
            "negativity feller_ok feller_number=8.00 plain_negative=0.00026",
            "negativity feller_violated feller_number=0.08 plain_negative=0.911",
            "full_negative=0.911",
            "plain_nan=0.909",
            "full_nan=0.00000",
            "cir_order feller_ok call=+1.0",
            "cir_order feller_violated call=+0.96",
            "cir_order feller_violated mean=+0.64",
        ],
    ),
    (
        "american_put_binomial_dp.py",
        [],
        [
            "example=american_put_binomial_dp",
            "american_put=",
            "european_put=",
            "early_exercise_premium=",
            "early_exercise_nodes=",
            "boundary_first_time=",
            "boundary t=",
        ],
    ),
    ("quadrature_demo.py", [], ["example=quadrature_demo", "exact=", "trap=", "simpson=", "gauss="]),
    (
        "laplace_inversion_demo.py",
        [],
        ["example=laplace_inversion_demo", "max_abs_error=", "mean_abs_error="],
    ),
    (
        "copula_gaussian_demo.py",
        [],
        ["example=copula_gaussian_demo", "tau_emp=", "spearman_emp=", "coexceedance_corr=", "coexceedance_ind="],
    ),
]

_SMOKE_IDS = [
    name if not args else f"{name}{''.join(' ' + a for a in args)}"
    for name, args, _ in _SMOKE_CASES
]


@pytest.mark.parametrize(
    ("script_name", "script_args", "required_keys"), _SMOKE_CASES, ids=_SMOKE_IDS
)
def test_public_examples_smoke(
    script_name: str, script_args: list[str], required_keys: list[str]
) -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    script_path = os.path.join(repo_root, "examples", script_name)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(repo_root, "src")

    result = subprocess.run(
        [sys.executable, script_path, *script_args],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    for key in required_keys:
        assert key in stdout

    if "mc_stderr=" in stdout:
        match = re.search(r"mc_stderr=([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)", stdout)
        assert match is not None
        stderr_val = float(match.group(1))
        assert stderr_val >= 0.0


@pytest.mark.parametrize(
    ("script_name", "script_args"),
    [(name, args) for name, args, _ in _SMOKE_CASES],
    ids=_SMOKE_IDS,
)
def test_public_examples_are_deterministic(
    script_name: str, script_args: list[str]
) -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    script_path = os.path.join(repo_root, "examples", script_name)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(repo_root, "src")

    run_a = subprocess.run(
        [sys.executable, script_path, *script_args],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    run_b = subprocess.run(
        [sys.executable, script_path, *script_args],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert run_a.returncode == 0, run_a.stderr
    assert run_b.returncode == 0, run_b.stderr
    assert run_a.stdout == run_b.stdout
