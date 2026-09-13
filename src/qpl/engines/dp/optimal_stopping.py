from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ...exceptions import InvalidInputError


def _normalize_level(values: object, *, expected_size: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise InvalidInputError(f"{label} must be a 1D array")
    if arr.shape[0] != expected_size:
        raise InvalidInputError(f"{label} must have shape ({expected_size},)")
    if not np.all(np.isfinite(arr)):
        raise InvalidInputError(f"{label} must contain only finite values")
    return arr.copy()


def backward_induction_optimal_stopping(
    *,
    exercise_values: Sequence[Sequence[float] | np.ndarray],
    continuation_operator: Callable[[int, np.ndarray], Sequence[float] | np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Solve a finite-horizon optimal stopping problem via backward induction.

    Parameters
    ----------
    exercise_values
        Sequence of per-time intrinsic/exercise values. Level ``j`` must have shape ``(j + 1,)``.
    continuation_operator
        Function returning continuation values at time ``j`` from values at time ``j + 1``.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        Value tree and exercise-policy tree (boolean arrays), each indexed by time level.
    """
    if not exercise_values:
        raise InvalidInputError("exercise_values must contain at least one time level")
    if not callable(continuation_operator):
        raise InvalidInputError("continuation_operator must be callable")

    n_levels = len(exercise_values)
    normalized_exercise = [
        _normalize_level(level, expected_size=j + 1, label=f"exercise_values[{j}]")
        for j, level in enumerate(exercise_values)
    ]

    value_tree: list[np.ndarray] = [np.empty(j + 1, dtype=float) for j in range(n_levels)]
    exercise_policy: list[np.ndarray] = [np.empty(j + 1, dtype=bool) for j in range(n_levels)]

    last_idx = n_levels - 1
    value_tree[last_idx] = normalized_exercise[last_idx]
    exercise_policy[last_idx] = np.ones(last_idx + 1, dtype=bool)

    for j in range(last_idx - 1, -1, -1):
        continuation_raw = continuation_operator(j, value_tree[j + 1].copy())
        continuation = _normalize_level(
            continuation_raw,
            expected_size=j + 1,
            label=f"continuation_operator({j}, ...)",
        )
        exercise = normalized_exercise[j]
        value_tree[j] = np.maximum(exercise, continuation)
        exercise_policy[j] = exercise >= continuation

    return value_tree, exercise_policy
