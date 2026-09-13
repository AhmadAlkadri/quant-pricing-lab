"""
Plumbing check only: confirms QuantLib-Python is importable and reports a
version. Real oracle comparisons against qpl engines arrive in later slices.

The directory-level conftest.py already keeps this file out of collection
when QuantLib is absent; `pytest.importorskip` here is a second guard so
this file also skips cleanly if it is ever targeted directly by path
(e.g. `pytest tests/oracle/test_oracle_available.py`).
"""
import pytest

QuantLib = pytest.importorskip("QuantLib")


def test_quantlib_version_is_nonempty_string() -> None:
    assert isinstance(QuantLib.__version__, str)
    assert QuantLib.__version__ != ""
