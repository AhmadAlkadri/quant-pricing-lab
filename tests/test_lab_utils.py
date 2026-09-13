from __future__ import annotations

import numpy as np
import pytest

from qpl.exceptions import InvalidInputError
from qpl.utils.labs import choose_by_mode, is_smoke_mode, set_global_seed


def test_is_smoke_mode_from_mapping() -> None:
    assert not is_smoke_mode(env={})
    assert not is_smoke_mode(env={"QPL_LAB_SMOKE": "0"})
    assert is_smoke_mode(env={"QPL_LAB_SMOKE": "1"})


def test_choose_by_mode_returns_expected_value() -> None:
    assert choose_by_mode(True, smoke=10, full=100) == 10
    assert choose_by_mode(False, smoke=[1], full=[2, 3]) == [2, 3]


def test_set_global_seed_is_deterministic_for_global_and_generator() -> None:
    set_global_seed(7)
    global_a = np.random.normal(size=5)
    set_global_seed(7)
    global_b = np.random.normal(size=5)
    assert np.array_equal(global_a, global_b)

    rng_a = set_global_seed(13)
    vals_a = rng_a.uniform(size=6)
    rng_b = set_global_seed(13)
    vals_b = rng_b.uniform(size=6)
    assert np.array_equal(vals_a, vals_b)


def test_lab_utils_validation_errors() -> None:
    with pytest.raises(InvalidInputError):
        is_smoke_mode(env={}, env_var="")
    with pytest.raises(InvalidInputError):
        set_global_seed(-1)
    with pytest.raises(InvalidInputError):
        set_global_seed(3.14)  # type: ignore[arg-type]
