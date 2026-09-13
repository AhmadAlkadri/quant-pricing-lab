import os
import subprocess
import sys
from pathlib import Path

import pytest

LAB_FILENAMES = [
    "01_static_monte_carlo.ipynb",
    "02_dynamic_monte_carlo.ipynb",
    "03_dynamic_programming_stochastic_optimization.ipynb",
    "04_finite_difference_methods.ipynb",
    "05_numerical_linear_systems.ipynb",
    "06_quadrature_methods.ipynb",
    "07_laplace_transform_inversion.ipynb",
    "08_copula_functions.ipynb",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _labs_root(repo_root: Path | None = None) -> Path:
    """Resolve the private, gitignored labs/ directory.

    Factored out (rather than inlined) so a dedicated test can exercise the
    skip path by monkeypatching this function's return value, without ever
    moving or deleting the real labs/ directory.
    """
    return (repo_root or _repo_root()) / "labs"


def _resolve_lab_path(labs_root: Path, lab_filename: str) -> Path:
    """Return the notebook path, or skip cleanly if labs/ or the notebook is absent."""
    lab_path = labs_root / lab_filename
    if not labs_root.is_dir() or not lab_path.is_file():
        pytest.skip("private labs/ directory not present")
    return lab_path


@pytest.mark.parametrize("lab_filename", LAB_FILENAMES)
def test_lab_smoke(tmp_path: Path, lab_filename: str) -> None:
    repo_root = _repo_root()
    lab_path = _resolve_lab_path(_labs_root(repo_root), lab_filename)
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


def test_lab_smoke_skips_when_labs_dir_missing(tmp_path: Path) -> None:
    """Dedicated check for the skip path: a nonexistent labs/ root must skip
    cleanly with a clear reason, without touching the real (present) labs/ dir.
    """
    fake_labs_root = tmp_path / "no-labs-here"

    with pytest.raises(pytest.skip.Exception) as exc_info:
        _resolve_lab_path(fake_labs_root, LAB_FILENAMES[0])

    assert "private labs/ directory not present" in str(exc_info.value)


def test_lab_smoke_skips_when_notebook_missing(tmp_path: Path) -> None:
    """Same skip path, but for a labs/ dir that exists yet lacks the notebook."""
    labs_root = tmp_path / "labs"
    labs_root.mkdir()

    with pytest.raises(pytest.skip.Exception) as exc_info:
        _resolve_lab_path(labs_root, "99_does_not_exist.ipynb")

    assert "private labs/ directory not present" in str(exc_info.value)
