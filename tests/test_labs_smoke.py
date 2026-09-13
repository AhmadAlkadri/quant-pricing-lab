import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "lab_filename",
    [
        "01_static_monte_carlo.ipynb",
        "02_dynamic_monte_carlo.ipynb",
        "03_dynamic_programming_stochastic_optimization.ipynb",
        "04_finite_difference_methods.ipynb",
        "05_numerical_linear_systems.ipynb",
        "06_quadrature_methods.ipynb",
        "07_laplace_transform_inversion.ipynb",
        "08_copula_functions.ipynb",
    ],
)
def test_lab_smoke(tmp_path: Path, lab_filename: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    lab_path = repo_root / "labs" / lab_filename
    output_name = f"{lab_path.stem}.executed.ipynb"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    env["MPLBACKEND"] = "Agg"
    env["QPL_LAB_SMOKE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(lab_path),
            "--output",
            output_name,
            "--output-dir",
            str(tmp_path),
            "--ExecutePreprocessor.timeout=180",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / output_name).exists()
    assert "Traceback (most recent call last)" not in result.stderr
