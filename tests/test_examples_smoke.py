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
