"""Dynamic-programming primitives, and one deprecated wrapper.

The CRR American-put pricer that used to live here is now
`qpl.engines.tree.price_american`, reached through the dispatcher as
``price(AmericanOption(...), model, market, method="tree", cfg=TreeConfig(...))``:
exercise style is a property of the instrument, not a separate engine entry
point. `price_american_put_binomial` survives only as a thin wrapper over that
engine, for notebooks written against the Slice 1 signature.

`backward_induction_optimal_stopping` is the part of this package that is
genuinely about dynamic programming rather than about options, and it stays.
"""

from .american_put_binomial import BinomialDPConfig, price_american_put_binomial
from .optimal_stopping import backward_induction_optimal_stopping

__all__ = [
    "BinomialDPConfig",
    "backward_induction_optimal_stopping",
    "price_american_put_binomial",
]
