"""Evidence taxonomy for numerical benchmark cases.

Every numerical claim this repository makes is only as good as the thing it is
checked against.  A test that asserts ``abs(pde - analytic) < 1e-2`` is not
self-describing: the reader cannot tell whether the reference is exact, a
published table, an independent implementation, or a statistical estimate with
its own error bars.  :class:`EvidenceClass` forces that statement to be
explicit, and :class:`BenchmarkRow` carries it next to the number, the
tolerance, and the citation.

This is a labelling pattern, not a framework: rows are plain data that tests
parametrise over.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["BenchmarkRow", "EvidenceClass"]


class EvidenceClass(Enum):
    """How a benchmark expectation is justified."""

    EXACT_IDENTITY = "exact_identity"
    """A relation that holds exactly in the model (e.g. put-call parity).
    Tolerance is floating-point round-off, not model error."""

    CLOSED_FORM = "closed_form"
    """Compared against a closed-form formula evaluated in this repository."""

    PUBLISHED_BENCHMARK = "published_benchmark"
    """Compared against a number published in a cited source."""

    INDEPENDENT_ENGINE = "independent_engine"
    """Compared against a different engine or library computing the same
    quantity by a different route."""

    CONVERGENCE_ORDER = "convergence_order"
    """The claim is about the measured rate at which the error decays under
    refinement, not about a single price."""

    STATISTICAL = "statistical"
    """The estimate carries sampling error; the tolerance is stated as a
    multiple of the reported standard error."""

    NEGATIVE_FINDING = "negative_finding"
    """A documented failure mode, pinned by a test so that it cannot silently
    change or be mistaken for a bug elsewhere."""


@dataclass(frozen=True)
class BenchmarkRow:
    """One checkable numerical claim.

    Parameters
    ----------
    id
        Short stable identifier, used as the pytest parameter id.
    description
        One line saying what is being checked.
    expected
        Expected value.  For identity-style rows this is the value the residual
        should take (usually ``0.0``).
    tolerance
        Absolute tolerance for the comparison.
    evidence
        Which :class:`EvidenceClass` justifies ``expected`` and ``tolerance``.
    source
        Citation for the expectation.  Either a bibliographic reference, or
        ``"derived in-repo: <path>"`` when the number is produced by this
        repository rather than taken from a source.
    notes
        Optional free text: caveats, why the tolerance is what it is.
    """

    id: str
    description: str
    expected: float
    tolerance: float
    evidence: EvidenceClass
    source: str
    notes: str = ""
