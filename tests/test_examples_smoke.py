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
