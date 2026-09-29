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

`greeks(...)` works the same way. `method="pde"` (projected SOR) and
`method="mc"` (least-squares Monte Carlo, which prices a Bermudan and has no
Greeks) also accept an `AmericanOption`; `method="analytic"` does not.

See `examples/american_put_binomial_dp.py`, `examples/american_put_cross_method.py`
and the note [American exercise on a binomial tree](/curriculum/notes/american_exercise_on_trees).
