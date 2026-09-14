"""Inverse problems: fitting a model's parameters to a set of market quotes.

One entry point, `calibrate_heston`, plus the data types its answer is made of
and the diagnostics that say whether the answer means anything. See
`qpl.calibration.heston` for the derivation, the measured COS settings and the
identifiability caveat.
"""

from .heston import (
    CALIBRATION_METHODS,
    COS_PUT_LEG_THRESHOLD,
    DEFAULT_BOUNDS,
    DOMAIN_PENALTY,
    FELLER_SATISFIED_COS,
    FELLER_VIOLATED_COS,
    HESTON_PARAMETERS,
    IMPLIED_VOL_BRACKET,
    MULTISTART_SEED,
    OBJECTIVES,
    RANK_TOLERANCE,
    VEGA_FLOOR,
    CalibrationResult,
    CosSettings,
    OptionQuote,
    StartSummary,
    calibrate_heston,
    cos_call_prices,
    default_cos_settings,
    heston_charfn_gradient,
    heston_quote_values,
    parameter_covariance,
    residual_jacobian,
    vega_weights,
)

__all__ = [
    "CALIBRATION_METHODS",
    "COS_PUT_LEG_THRESHOLD",
    "DEFAULT_BOUNDS",
    "DOMAIN_PENALTY",
    "FELLER_SATISFIED_COS",
    "FELLER_VIOLATED_COS",
    "HESTON_PARAMETERS",
    "IMPLIED_VOL_BRACKET",
    "MULTISTART_SEED",
    "OBJECTIVES",
    "RANK_TOLERANCE",
    "VEGA_FLOOR",
    "CalibrationResult",
    "CosSettings",
    "OptionQuote",
    "StartSummary",
    "calibrate_heston",
    "cos_call_prices",
    "default_cos_settings",
    "heston_charfn_gradient",
    "heston_quote_values",
    "parameter_covariance",
    "residual_jacobian",
    "vega_weights",
]
