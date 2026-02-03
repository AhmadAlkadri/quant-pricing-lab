import os
import re
import subprocess
import sys


def test_bs_mc_vs_analytic_example_smoke() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    script_path = os.path.join(repo_root, "examples", "bs_mc_vs_analytic.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(repo_root, "src")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "analytic=" in stdout
    assert "mc=" in stdout
    assert "stderr=" in stdout
    assert "z=" in stdout

    match = re.search(r"stderr=([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)", stdout)
    assert match is not None
    stderr_val = float(match.group(1))
    assert stderr_val >= 0.0
