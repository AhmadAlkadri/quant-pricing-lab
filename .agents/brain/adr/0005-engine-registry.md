# ADR-0005: Engine registry keyed by (instrument, model, method)

Status: accepted
Date: 2026-09-13

Context
- `qpl.pricing.price` and `qpl.pricing.greeks` were each an `isinstance`
  ladder with one branch per `method`, and every branch repeated the same
  four steps: check the kwargs, check the instrument type, check the model
  type, check the market type. Six copies of that shape for three methods.
- Adding the Phase 1 CRR tree makes it eight copies, and Phase 1/2/3 will add
  Leisen-Reimer, PSOR, Longstaff-Schwartz and Fourier engines on top of that.
  The duplication is already the largest source of accidental divergence in
  the dispatcher: nothing but discipline kept the three branches raising the
  same errors in the same order.
- The error *messages* and the `InvalidInputError` / `NotSupportedError` split
  are relied on by callers and by existing tests, so whatever replaces the
  ladder has to reproduce them exactly.

Decision
- Replace the ladders with two dictionaries in `src/qpl/engines/registry.py`,
  keyed by `(instrument type, model type, method)` -- one for prices, one for
  Greeks -- plus one `MethodSpec` per `method` string carrying that method's
  keyword contract (`cfg` type, and any extra kwargs such as MC's `bumps`,
  each with its validator). Lookup walks the MRO, preserving `isinstance`
  semantics.
- `register(...)` / `resolve_price(...)` / `resolve_greeks(...)` /
  `method_spec(...)` are the whole API. The `MethodSpec` for a method lives in
  the engine module that defines it (`MC_METHOD_SPEC` next to `MCConfig`, and
  so on); `qpl.pricing` imports the engines it ships and calls `register`
  explicitly at import time. No metaclass, no `supports()` predicate, no
  import-graph scan.
- `price(instrument, model, market, *, method=..., **kwargs)` and `greeks(...)`
  keep their signatures; each is now three lines: bind kwargs, resolve engine,
  call it.

Alternatives considered
- Keep the ladder: rejected. It works, but the copy count grows with engines
  times entry points, and a new engine must be hand-added to both ladders with
  the error ordering re-derived from memory each time.
- Engine classes with a `supports(instrument, model, market)` predicate:
  rejected. Resolution becomes a linear scan whose result depends on
  registration order, "unsupported" cannot be distinguished from "no engine
  matched yet", and every engine grows a predicate that is almost always a
  restatement of its type signature.
- `functools.singledispatch` on the instrument type: rejected. It dispatches
  on one argument, and this dispatch is genuinely three-way; encoding model
  and method inside the instrument-dispatched function reintroduces the ladder
  one level down.
- QuantLib-style engine objects attached to the instrument
  (`option.set_pricing_engine(...)`): rejected for this repository. It makes
  the instrument stateful and makes a pricing call's behaviour depend on
  mutation history, which is exactly the property that makes a numerical
  result hard to reproduce from a test.

Consequences
- To add an engine: define a `MethodSpec` next to the engine's config object,
  write `price_*`/`greeks_*` with the signature
  `fn(instrument, model, market, **bound_kwargs)`, and add one `register(...)`
  call in `qpl.pricing._register_builtin_engines`. Nothing else changes.
- Keyword validation happens before engine lookup, as in the ladder, so a bad
  `cfg` still reports `InvalidInputError` rather than `NotSupportedError`.
  `tests/test_engine_registry.py` pins the messages and that ordering.
- `Market` is not part of the key. It is a data container (spot plus curves)
  that every engine consumes through the same interface, not a dispatch axis:
  no engine here is selected *because of* the market type. It is still
  type-checked, once, inside `resolve_*`, which is what preserves the single
  "Unsupported instrument/model/market combination" message. If a second
  market type ever appears (a term-structure market, say), that is an
  interface change every engine must absorb, not a new branch to dispatch on;
  at that point this decision should be revisited rather than worked around.
- Cost: one more module and an indirection between `price()` and the engine.
  It buys back six duplicated validation blocks and makes the supported set
  enumerable (`known_methods()`), which the ladder never was.

Supersedes (optional)
- None.
