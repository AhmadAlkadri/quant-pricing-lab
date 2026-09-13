from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TypeVar

import numpy as np

from ..exceptions import InvalidInputError

T = TypeVar("T")
SMOKE_ENV_VAR = "QPL_LAB_SMOKE"


def is_smoke_mode(*, env: Mapping[str, str] | None = None, env_var: str = SMOKE_ENV_VAR) -> bool:
    """Return whether smoke mode is enabled via an environment variable.

    Parameters
    ----------
    env
        Optional environment mapping. Defaults to `os.environ`.
    env_var
        Name of the environment variable used to toggle smoke mode.

    Returns
    -------
    bool
        `True` if `env_var` is set to `"1"`, else `False`.
    """
    if not isinstance(env_var, str) or not env_var:
        raise InvalidInputError("env_var must be a non-empty string")
    values = os.environ if env is None else env
    return str(values.get(env_var, "0")) == "1"


def choose_by_mode(smoke_mode: bool, *, smoke: T, full: T) -> T:
    """Return mode-specific value.

    Parameters
    ----------
    smoke_mode
        Smoke-mode flag.
    smoke
        Value returned when `smoke_mode` is `True`.
    full
        Value returned when `smoke_mode` is `False`.
    """
    return smoke if smoke_mode else full


def set_global_seed(seed: int = 123) -> np.random.Generator:
    """Set NumPy global seed and return a matching generator.

    Parameters
    ----------
    seed
        Non-negative integer seed.

    Returns
    -------
    np.random.Generator
        `default_rng` generator initialized with `seed`.
    """
    if not isinstance(seed, int):
        raise InvalidInputError("seed must be an integer")
    if seed < 0:
        raise InvalidInputError("seed must be >= 0")

    np.random.seed(seed)
    return np.random.default_rng(seed)
