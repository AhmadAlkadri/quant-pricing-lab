"""The generic finite-horizon optimal-stopping solver and the spot lattice.

The American *option* engine that consumes them lives in
`qpl.engines.tree.american` and is tested in `tests/test_tree_american.py`;
this file is about the two mechanisms underneath it, exercised on inputs that
have nothing to do with options.
"""

import numpy as np
import pytest

from qpl.engines.dp import backward_induction_optimal_stopping
from qpl.engines.tree import build_recombining_spot_tree
from qpl.exceptions import InvalidInputError


def test_backward_induction_toy_recursion_closed_form() -> None:
    exercise_values = [
        np.array([0.0]),
        np.array([1.0, 2.0]),
        np.array([0.0, 4.0, 1.0]),
    ]

    def continuation(_level: int, next_values: np.ndarray) -> np.ndarray:
        return 0.5 * (next_values[:-1] + next_values[1:])

    value_tree, policy = backward_induction_optimal_stopping(
        exercise_values=exercise_values,
        continuation_operator=continuation,
    )

    assert np.allclose(value_tree[2], np.array([0.0, 4.0, 1.0]))
    assert np.allclose(value_tree[1], np.array([2.0, 2.5]))
    assert np.allclose(value_tree[0], np.array([2.25]))
    assert np.array_equal(policy[0], np.array([False]))
    assert np.array_equal(policy[1], np.array([False, False]))
    assert np.array_equal(policy[2], np.array([True, True, True]))


def test_backward_induction_validation_errors() -> None:
    with pytest.raises(InvalidInputError):
        backward_induction_optimal_stopping(exercise_values=[], continuation_operator=lambda _j, v: v)

    with pytest.raises(InvalidInputError):
        backward_induction_optimal_stopping(  # type: ignore[arg-type]
            exercise_values=[np.array([0.0])],
            continuation_operator=None,
        )

    with pytest.raises(InvalidInputError):
        backward_induction_optimal_stopping(
            exercise_values=[np.array([0.0]), np.array([1.0]), np.array([0.0, 1.0, 2.0])],
            continuation_operator=lambda _j, v: v[:-1],
        )

    with pytest.raises(InvalidInputError):
        backward_induction_optimal_stopping(
            exercise_values=[np.array([0.0]), np.array([1.0, 2.0])],
            continuation_operator=lambda _j, _v: np.array([1.0, np.nan]),
        )


def test_spot_tree_shape_and_values() -> None:
    tree = build_recombining_spot_tree(spot=100.0, up=1.1, down=0.9, n_steps=3)
    assert [len(level) for level in tree] == [1, 2, 3, 4]

    assert tree[0][0] == pytest.approx(100.0)
    assert tree[2][1] == pytest.approx(100.0 * 1.1 * 0.9)
    assert tree[3][0] == pytest.approx(100.0 * 0.9**3)
    assert tree[3][3] == pytest.approx(100.0 * 1.1**3)


def test_spot_tree_validation_errors() -> None:
    with pytest.raises(InvalidInputError):
        build_recombining_spot_tree(spot=0.0, up=1.1, down=0.9, n_steps=2)
    with pytest.raises(InvalidInputError):
        build_recombining_spot_tree(spot=100.0, up=0.0, down=0.9, n_steps=2)
    with pytest.raises(InvalidInputError):
        build_recombining_spot_tree(spot=100.0, up=1.1, down=-0.9, n_steps=2)
    with pytest.raises(InvalidInputError):
        build_recombining_spot_tree(spot=100.0, up=1.1, down=0.9, n_steps=0)
