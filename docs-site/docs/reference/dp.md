# Dynamic Programming

Dynamic-programming components are under `qpl.engines.dp`.

Current scope:

- generic finite-horizon optimal stopping backward induction
  (`backward_induction_optimal_stopping`),
- a deprecated keyword-based wrapper, `price_american_put_binomial`, kept for
  notebooks written against the older signature.

American options themselves are no longer priced from this package. Exercise
style is a property of the instrument, so an American put or call goes through
the ordinary dispatcher:

```python
from qpl.engines.tree import TreeConfig
from qpl.instruments import AmericanOption
from qpl.pricing import greeks, price

result = price(
    AmericanOption(kind="put", strike=100.0, expiry=1.0),
    model, market, method="tree", cfg=TreeConfig(n_steps=2000),
)
result.meta["exercise_boundary"]        # per time level, NaN where none
result.meta["early_exercise_node_count"]
```

`greeks(...)` works the same way. The analytic, Monte Carlo and PDE engines
are registered for European options only and raise `NotSupportedError` for an
`AmericanOption`.

See `examples/american_put_binomial_dp.py` and
`docs/notes/american_exercise_on_trees.md`.
