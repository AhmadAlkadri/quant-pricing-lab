"""Validation utilities: convergence-order measurement and evidence labelling."""

from .benchmark import BenchmarkRow, EvidenceClass
from .convergence import ConvergenceFit, fit_convergence_order, refinement_errors
from .stochastic import ErrorEstimate, strong_error, weak_error

__all__ = [
    "BenchmarkRow",
    "ConvergenceFit",
    "ErrorEstimate",
    "EvidenceClass",
    "fit_convergence_order",
    "refinement_errors",
    "strong_error",
    "weak_error",
]
