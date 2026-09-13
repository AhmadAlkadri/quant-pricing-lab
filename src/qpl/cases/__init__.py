"""Benchmark case definitions: numerical claims as data, with their evidence."""

from .european_black_scholes import (
    ALL_CASES,
    KNOWN_VALUE_CASES,
    LIMIT_CASES,
    MONOTONICITY_CASES,
    PARITY_CASES,
    EuropeanBSCase,
    EuropeanBSSpec,
    parity_residual,
)

__all__ = [
    "EuropeanBSSpec",
    "EuropeanBSCase",
    "PARITY_CASES",
    "LIMIT_CASES",
    "KNOWN_VALUE_CASES",
    "MONOTONICITY_CASES",
    "ALL_CASES",
    "parity_residual",
]
