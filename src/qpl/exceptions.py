from __future__ import annotations


class QPLError(Exception):
    """Base error for qpl."""



class InvalidInputError(QPLError):
    """Raised when inputs violate domain constraints."""



class ModelAssumptionError(QPLError):
    """Raised when model assumptions are violated."""



class NotSupportedError(QPLError):
    """Raised when an instrument/model/method combo is unsupported."""
