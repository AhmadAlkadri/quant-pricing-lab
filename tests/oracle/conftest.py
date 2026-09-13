"""
Collection-time gate for tests/oracle/.

QuantLib-Python is an optional independent oracle (the `oracle` extra).
Everything under this directory is skipped, with a clear reason, when
QuantLib is not installed -- including when tests/oracle is passed directly
on the command line.

Note: a plain module-level `pytest.importorskip("QuantLib")` here would
raise during pytest's *initial* conftest loading (which happens for paths
given explicitly on the command line, e.g. `pytest tests/oracle`), and that
phase does not catch the resulting Skipped exception -- it turns into a hard
collection error instead of a clean skip. `pytest_ignore_collect` avoids
that failure mode while still skipping collection entirely when the
optional `oracle` extra is not installed.
"""
import importlib.util


def pytest_ignore_collect(collection_path, config):
    if importlib.util.find_spec("QuantLib") is None:
        return True
    return None
