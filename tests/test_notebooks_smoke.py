from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.parametrize(
    "notebook_filename",
    [
        "01_pricing_overview.ipynb",
        "02_numerical_toolkit.ipynb",
        "03_dependence_and_copulas.ipynb",
    ],
)
def test_public_notebook_smoke(tmp_path: Path, notebook_filename: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / notebook_filename
    output_name = f"{notebook_path.stem}.executed.ipynb"

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
            str(notebook_path),
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
