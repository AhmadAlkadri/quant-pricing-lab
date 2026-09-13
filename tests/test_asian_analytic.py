"""The discrete geometric Asian closed form, and what it is checked against.

Four kinds of claim live here and they are deliberately not mixed.

1. **EXACT_IDENTITY.** A one-fixing Asian *is* a European vanilla, whichever
   averaging is asked for, and the moment-matched approximation is exact there
   too (a lognormal fitted to the two moments of a lognormal is that
   lognormal). Asian put-call parity `C - P = e^{-rT}(E[A] - K)` holds for both
   averagings with the right `E[.]`. These are tested at round-off.
2. **CLOSED_FORM.** The collapsed variance formula
   `v = (sigma^2/n^2) sum_i (2(n-i)+1) t_i` against the `O(n^2)` double sum it
   was derived from, on irregular schedules as well as uniform ones.
3. **PUBLISHED_BENCHMARK.** Three cited values, all reproduced: the
   Clewlow-Strickland discrete geometric call (5.3425606635), the Haug
   continuous geometric put (4.6922) reached by making the fixing grid dense,
   and the Turnbull-Wakeman arithmetic approximation (19.5152) -- the last of
   which is a benchmark *for the approximation*, not for the true price.
4. **CONVERGENCE_ORDER.** The rate at which the discrete formula approaches the
   continuous one as the fixings are refined. Derived to be order 1 (both
   `tbar` and `v` carry an `O(1/n)` correction), measured at 1.0002 and 1.0096.

The fixing-time convention throughout: `t_i = i T / n`, `i = 1..n`, so the last
fixing coincides with expiry. It is built by
`qpl.instruments.uniform_fixing_times`, which uses `linspace` so that
`t_n == T` exactly.

Sources: Kemna & Vorst (1990), *Journal of Banking and Finance* 14, 113-129
(continuous geometric closed form, and the geometric control variate);
Turnbull & Wakeman (1991), *JFQA* 26(3), 377-389 and Levy (1992), *JIMF* 11,
474-491 (the two-moment lognormal fit); Clewlow & Strickland,
*Implementing Derivatives Models* (the discrete geometric reference value, as
carried in the QuantLib test suite); Haug, *The Complete Guide to Option
Pricing Formulas* (the continuous geometric reference value, likewise). Every
formula here was re-derived in `qpl.engines.analytic.asian`; the three
published numbers are used as fixtures with the citations above and nothing
else is taken from those sources.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from qpl.engines.analytic.asian import (
    arithmetic_average_moments,
    discrete_geometric_greeks,
    discrete_geometric_price,
    expected_arithmetic_average,
    expected_geometric_average,
    geometric_average_log_moments,
    turnbull_wakeman_price,
)
from qpl.exceptions import InvalidInputError, NotSupportedError
from qpl.instruments import (
    AsianOption,
    EuropeanOption,
    arithmetic_average,
    asian_payoff,
    geometric_average,
    uniform_fixing_times,
)
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel, bs_price
from qpl.pricing import greeks, price
from qpl.validation import EvidenceClass, fit_convergence_order

# --------------------------------------------------------------------------
# The three published points, one place.
# --------------------------------------------------------------------------

CLEWLOW_STRICKLAND = dict(S=100.0, K=100.0, T=1.0, r=0.06, q=0.03, sigma=0.20, n=10)
CLEWLOW_STRICKLAND_VALUE = 5.3425606635

HAUG_CONTINUOUS = dict(S=80.0, K=85.0, T=0.25, r=0.05, q=-0.03, sigma=0.20)
HAUG_CONTINUOUS_PUT = 4.6922

TURNBULL_WAKEMAN = dict(S=100.0, K=80.0, T=0.5, r=0.05, q=0.05, sigma=0.20, n=26)
TURNBULL_WAKEMAN_VALUE = 19.5152

_DENSE_FIXINGS = 20_000
"""Fixings used to approach the continuous limit. The discrete error is
`0.30 / n` at the Haug point, so 20 000 fixings leave 1.5e-05 of discretisation
against the 2.1e-05 that separates the exact continuous value from the
published four-decimal figure."""

_ORDER_LEVELS = (20, 40, 80, 160, 320, 640, 1280, 2560)


def _continuous_geometric(*, S, K, T, r, q, sigma, kind="call") -> float:
    """Kemna-Vorst continuous-averaging geometric price, written out here.

    Deliberately a second implementation rather than a call into the engine: it
    is the limit the discrete formula is measured *against*, so it must not
    share code with it. `log G_cont = log S + (mu - sigma^2/2) T/2 + ...` with
    variance `sigma^2 T / 3`, both being the `n -> inf` limits of the discrete
    moments (module docstring of `qpl.engines.analytic.asian`).
    """
    mu = r - q
    m = math.log(S) + (mu - 0.5 * sigma * sigma) * T / 2.0
    v = sigma * sigma * T / 3.0
    forward = math.exp(m + 0.5 * v)
    sd = math.sqrt(v)
    d1 = (math.log(forward / K) + 0.5 * v) / sd
    d2 = d1 - sd
    ncdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))  # noqa: E731
    disc = math.exp(-r * T)
    if kind == "call":
        return disc * (forward * ncdf(d1) - K * ncdf(d2))
    return disc * (K * ncdf(-d2) - forward * ncdf(-d1))


def _triple(*, S, K, T, r, q, sigma, times, averaging="geometric", kind="call"):
    return (
        AsianOption(
            kind=kind, strike=K, expiry=T, fixing_times=times, averaging=averaging
        ),
        BlackScholesModel(sigma=sigma),
        Market(
            spot=S,
            rate_curve=FlatRateCurve(r, allow_negative=True),
            dividend_curve=FlatDividendCurve(q, allow_negative=True),
        ),
    )


# --------------------------------------------------------------------------
# The instrument.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(kind="straddle"), "kind must be 'call' or 'put'"),
        (dict(strike=0.0), "strike must be > 0"),
        (dict(averaging="harmonic"), "averaging must be 'arithmetic' or 'geometric'"),
        (dict(fixing_times=()), "fixing_times must contain at least one time"),
        (dict(fixing_times=(0.0, 0.5)), "fixing_times must all be > 0"),
        (dict(fixing_times=(-0.5, 0.5)), "fixing_times must all be > 0"),
        (dict(fixing_times=(0.5, 0.5)), "fixing_times must be strictly increasing"),
        (dict(fixing_times=(0.5, 0.25)), "fixing_times must be strictly increasing"),
        (dict(fixing_times=(0.5, 1.5)), "fixing_times must all be <= expiry"),
        (dict(fixing_times=(0.5, float("nan"))), "fixing_times must be finite"),
    ],
)
def test_invalid_asian_options_are_refused_with_their_own_message(kwargs, message):
    """Validation at construction, one message per rule.

    A repeated fixing time is refused rather than handled: it is a legitimate
    contract (a date that counts twice) but it makes the fixing grid and the
    simulation grid two different objects, and the closed form would absorb the
    repeat silently. See the `AsianOption` docstring.
    """
    base = dict(
        kind="call",
        strike=100.0,
        expiry=1.0,
        fixing_times=(0.5, 1.0),
        averaging="arithmetic",
    )
    with pytest.raises(InvalidInputError, match=message):
        AsianOption(**{**base, **kwargs})


def test_fixing_times_are_normalised_to_a_tuple_and_the_option_stays_hashable():
    """Any iterable in, a tuple of floats out; the instrument is still a value."""
    option = AsianOption("call", 100.0, 1.0, [0.5, 1.0], "geometric")
    assert option.fixing_times == (0.5, 1.0)
    assert isinstance(option.fixing_times, tuple)
    assert option.n_fixings == 2
    assert hash(option) == hash(AsianOption("CALL", 100.0, 1.0, (0.5, 1.0), "GEOMETRIC"))


def test_uniform_fixing_times_lands_exactly_on_expiry():
    """The reason this helper exists rather than a comprehension.

    `(i + 1) * T / n` at `i = n - 1` is not `T` in general: at `T = 90/365` and
    `n = 2560` it is one ulp above, and `AsianOption` then correctly refuses a
    fixing after expiry. `linspace` pins the endpoint.
    """
    for expiry in (1.0, 0.5, 90.0 / 365.0, 0.25, 1.0 / 3.0):
        for n in (1, 10, 26, 52, 2560):
            times = uniform_fixing_times(expiry, n)
            assert len(times) == n
            assert times[-1] == expiry
            assert times[0] == pytest.approx(expiry / n, rel=1e-15)
            AsianOption("call", 100.0, expiry, times, "arithmetic")
    with pytest.raises(InvalidInputError, match="expiry must be finite and > 0"):
        uniform_fixing_times(0.0, 10)
    with pytest.raises(InvalidInputError, match="n_fixings must be >= 1"):
        uniform_fixing_times(1.0, 0)


# --------------------------------------------------------------------------
# The payoff reduction.
# --------------------------------------------------------------------------


def test_the_two_averages_and_the_am_gm_inequality_on_a_sample():
    """EXACT_IDENTITY: `A_arith >= A_geom` pathwise, with equality iff constant.

    This is the inequality that (d) in the slice statement lifts to prices. It
    is checked on the *payoff reduction* here, before any pricing, because that
    is where it is exact: AM-GM is a statement about a finite list of positive
    numbers and owes nothing to the model.
    """
    assert EvidenceClass.EXACT_IDENTITY
    rng = np.random.default_rng(11)
    fixings = np.exp(rng.normal(size=(500, 12)) * 0.3 + 4.6)
    arith = arithmetic_average(fixings)
    geom = geometric_average(fixings)
    assert np.all(arith >= geom - 1e-12)
    assert np.min(arith - geom) > 0.0  # no constant rows in this sample

    flat = np.full((3, 7), 100.0)
    assert np.allclose(arithmetic_average(flat), 100.0)
    assert np.allclose(geometric_average(flat), 100.0, atol=1e-12)

    # And the payoff inherits it, because max(. - K, 0) is non-decreasing.
    calls_a = asian_payoff(arith, 100.0, "call")
    calls_g = asian_payoff(geom, 100.0, "call")
    assert np.all(calls_a >= calls_g)


def test_geometric_average_is_computed_in_log_space_and_does_not_overflow():
    """A 52-fixing product of spots near 100 overflows; the mean of logs does not.

    `prod_i S_i` for 52 fixings at `S ~ 1e6` is `O(10**312)`, past the float64
    ceiling, and the naive `prod ** (1/n)` returns `inf`. This is why
    `geometric_average` is `exp(mean(log S))`.
    """
    fixings = np.full((2, 52), 1.0e6)
    with np.errstate(over="ignore"):
        assert np.isinf(np.prod(fixings[0]))
    assert np.allclose(geometric_average(fixings), 1.0e6, rtol=1e-12)


# --------------------------------------------------------------------------
# The closed form against its own derivation.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "times",
    [
        uniform_fixing_times(1.0, 1),
        uniform_fixing_times(1.0, 10),
        uniform_fixing_times(2.0, 37),
        (0.01, 0.4, 0.41, 1.7, 3.0),
        (0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
    ],
    ids=["n1", "n10", "n37", "irregular", "beyond_a_year"],
)
def test_collapsed_variance_equals_the_double_sum(times):
    """CLOSED_FORM: `sum_i (2(n-i)+1) t_i` is `sum_i sum_j min(t_i, t_j)`.

    The `O(n)` form in the engine is an algebraic collapse of the `O(n^2)`
    covariance sum that the derivation actually produces. This test evaluates
    the double sum directly, including on schedules that are not uniform, where
    an index slip would not cancel.
    """
    assert EvidenceClass.CLOSED_FORM
    sigma = 0.3
    _, v = geometric_average_log_moments(S=100.0, mu=0.02, sigma=sigma, fixing_times=times)
    t = np.asarray(times, dtype=float)
    n = t.size
    brute = sigma * sigma * float(np.sum(np.minimum(t[:, None], t[None, :]))) / (n * n)
    assert v == pytest.approx(brute, rel=1e-14, abs=1e-16)


def test_uniform_grid_moments_match_the_derived_closed_forms():
    """CLOSED_FORM: `tbar = (T/2)(1+1/n)` and `v = (sigma^2 T/3)(1+3/(2n)+1/(2n^2))`.

    These two expressions are what make the continuous limit order 1 rather
    than order 2, so they are checked directly rather than only through the
    fitted order below.
    """
    assert EvidenceClass.CLOSED_FORM
    S, sigma, mu, T = 100.0, 0.25, 0.04, 1.7
    for n in (1, 2, 10, 52, 501):
        times = uniform_fixing_times(T, n)
        m, v = geometric_average_log_moments(S=S, mu=mu, sigma=sigma, fixing_times=times)
        tbar = 0.5 * T * (1.0 + 1.0 / n)
        assert m == pytest.approx(
            math.log(S) + (mu - 0.5 * sigma**2) * tbar, rel=1e-14
        )
        expected_v = (sigma**2 * T / 3.0) * (1.0 + 1.5 / n + 0.5 / (n * n))
        assert v == pytest.approx(expected_v, rel=1e-13)


def test_expected_averages_are_exact_and_ordered():
    """EXACT_IDENTITY: `E[A] = (S/n) sum e^{mu t_i}` and `E[A] > E[G]` for sigma > 0.

    `E[A]` does not depend on `sigma` at all -- it is an average of forwards --
    which is the fact that makes Asian put-call parity volatility-independent.
    `E[G] = exp(m + v/2)` is below it by Jensen, strictly so whenever the
    variance is positive.
    """
    assert EvidenceClass.EXACT_IDENTITY
    times = uniform_fixing_times(1.5, 13)
    mu = 0.03
    ea = expected_arithmetic_average(S=100.0, mu=mu, fixing_times=times)
    assert ea == pytest.approx(
        100.0 * float(np.mean(np.exp(mu * np.asarray(times)))), rel=1e-15
    )
    for sigma in (0.05, 0.2, 0.6):
        eg = expected_geometric_average(S=100.0, mu=mu, sigma=sigma, fixing_times=times)
        assert eg < ea
    # sigma = 0 collapses the gap only for a single fixing; with several
    # fixings E[G] is still the geometric mean of distinct forwards.
    eg0 = expected_geometric_average(S=100.0, mu=mu, sigma=0.0, fixing_times=times)
    assert eg0 < ea
    single = uniform_fixing_times(1.5, 1)
    assert expected_geometric_average(
        S=100.0, mu=mu, sigma=0.0, fixing_times=single
    ) == pytest.approx(expected_arithmetic_average(S=100.0, mu=mu, fixing_times=single))


@pytest.mark.parametrize("averaging", ["arithmetic", "geometric"])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_a_single_fixing_at_expiry_is_a_european_vanilla(averaging, kind):
    """EXACT_IDENTITY: with `n = 1` and `t_1 = T`, `A = G = S_T`.

    The strongest available check on the geometric formula, because it ties it
    to an engine written years earlier and shared with nothing here: at one
    fixing the whole Asian apparatus has to collapse onto Black-Scholes. The
    moment-matched approximation is exact here too -- a lognormal fitted to the
    first two moments of a lognormal is that lognormal -- so
    `turnbull_wakeman_price` must also reproduce Black-Scholes to round-off,
    which is a check on the `M2` double sum that nothing else in this file
    provides.
    """
    assert EvidenceClass.EXACT_IDENTITY
    S, K, T, r, q, sigma = 103.0, 97.0, 1.3, 0.045, 0.017, 0.27
    times = uniform_fixing_times(T, 1)
    vanilla = float(bs_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind=kind))

    geo = discrete_geometric_price(
        S=S, K=K, T=T, r=r, sigma=sigma, fixing_times=times, q=q, kind=kind
    )
    assert geo == pytest.approx(vanilla, abs=1e-13)

    tw = turnbull_wakeman_price(
        S=S, K=K, T=T, r=r, sigma=sigma, fixing_times=times, q=q, kind=kind
    )
    assert tw == pytest.approx(vanilla, abs=1e-12)

    if averaging == "geometric":
        triple = _triple(S=S, K=K, T=T, r=r, q=q, sigma=sigma, times=times, kind=kind)
        assert price(*triple).value == pytest.approx(vanilla, abs=1e-12)
        european = (EuropeanOption(kind, K, T), triple[1], triple[2])
        assert price(*triple).value == pytest.approx(price(*european).value, abs=1e-12)


@pytest.mark.parametrize("n", [1, 5, 26, 52])
def test_geometric_put_call_parity_is_exact(n):
    """EXACT_IDENTITY: `C - P = e^{-rT}(E[G] - K)` for the geometric pair.

    Follows from `max(G - K, 0) - max(K - G, 0) = G - K` and linearity of the
    expectation; it is Black (1976) parity in the forward `E[G]`, and holds for
    any fixing schedule and any volatility.
    """
    assert EvidenceClass.EXACT_IDENTITY
    S, K, T, r, q, sigma = 100.0, 95.0, 1.0, 0.05, 0.02, 0.3
    times = uniform_fixing_times(T, n)
    args = dict(S=S, K=K, T=T, r=r, sigma=sigma, fixing_times=times, q=q)
    call = discrete_geometric_price(**args, kind="call")
    put = discrete_geometric_price(**args, kind="put")
    eg = expected_geometric_average(S=S, mu=r - q, sigma=sigma, fixing_times=times)
    assert call - put == pytest.approx(math.exp(-r * T) * (eg - K), abs=1e-12)


@pytest.mark.parametrize("n", [1, 5, 26, 52])
def test_turnbull_wakeman_put_call_parity_is_exact(n):
    """EXACT_IDENTITY: the approximation still satisfies parity exactly.

    Worth its own row because it separates two things that are easy to conflate.
    The moment-matched price is *wrong* (it prices the wrong distribution), but
    it is wrong in a way that preserves `C - P = e^{-rT}(E[A] - K)`, because the
    fitted lognormal has the exact first moment `M1 = E[A]` by construction. A
    parity check can therefore never detect the approximation error -- which is
    exactly why the gap is measured against Monte Carlo instead.
    """
    assert EvidenceClass.EXACT_IDENTITY
    S, K, T, r, q, sigma = 100.0, 95.0, 1.0, 0.05, 0.02, 0.3
    times = uniform_fixing_times(T, n)
    args = dict(S=S, K=K, T=T, r=r, sigma=sigma, fixing_times=times, q=q)
    call = turnbull_wakeman_price(**args, kind="call")
    put = turnbull_wakeman_price(**args, kind="put")
    ea = expected_arithmetic_average(S=S, mu=r - q, fixing_times=times)
    assert call - put == pytest.approx(math.exp(-r * T) * (ea - K), abs=1e-12)
    m1, _ = arithmetic_average_moments(S=S, mu=r - q, sigma=sigma, fixing_times=times)
    assert m1 == pytest.approx(ea, rel=1e-15)


def test_turnbull_wakeman_is_above_the_geometric_price():
    """The moment-matched arithmetic price exceeds the exact geometric one.

    Not an identity and not asserted as one: it follows from `E[A] > E[G]`
    (Jensen) dominating the variance difference at these points, and it is the
    ordering the control-variate Monte Carlo run has to reproduce. Measured gap
    at the Turnbull-Wakeman benchmark point: 19.5152 against 19.3535.
    """
    args = dict(
        S=TURNBULL_WAKEMAN["S"],
        K=TURNBULL_WAKEMAN["K"],
        T=TURNBULL_WAKEMAN["T"],
        r=TURNBULL_WAKEMAN["r"],
        sigma=TURNBULL_WAKEMAN["sigma"],
        q=TURNBULL_WAKEMAN["q"],
        fixing_times=uniform_fixing_times(
            TURNBULL_WAKEMAN["T"], TURNBULL_WAKEMAN["n"]
        ),
    )
    tw = turnbull_wakeman_price(**args)
    geo = discrete_geometric_price(**args)
    assert tw > geo
    assert tw - geo == pytest.approx(0.16166, abs=1e-4)


def test_zero_volatility_gives_the_discounted_intrinsic_of_the_average():
    """Degenerate limit: at `sigma = 0` the average is deterministic.

    `A = G = (1/n) sum_i S e^{mu t_i}` for the arithmetic case (and its
    geometric counterpart), so both formulas must return
    `e^{-rT} max(average forward - K, 0)`. `_black_76` reaches this through its
    `variance <= 0` branch rather than through a separate code path.
    """
    S, T, r, q = 100.0, 1.0, 0.05, 0.0
    times = uniform_fixing_times(T, 12)
    disc = math.exp(-r * T)
    eg = expected_geometric_average(S=S, mu=r - q, sigma=0.0, fixing_times=times)
    ea = expected_arithmetic_average(S=S, mu=r - q, fixing_times=times)
    for K in (80.0, 105.0):
        args = dict(S=S, K=K, T=T, r=r, sigma=0.0, fixing_times=times, q=q)
        assert discrete_geometric_price(**args, kind="call") == pytest.approx(
            disc * max(eg - K, 0.0), abs=1e-12
        )
        assert discrete_geometric_price(**args, kind="put") == pytest.approx(
            disc * max(K - eg, 0.0), abs=1e-12
        )
        assert turnbull_wakeman_price(**args, kind="call") == pytest.approx(
            disc * max(ea - K, 0.0), abs=1e-12
        )


def test_an_average_that_stops_before_settlement_only_changes_the_discount():
    """`T` enters the geometric formula only through `e^{-rT}`.

    Averaging that ends at `t_n < T` is a real convention, and the formula
    handles it by construction rather than by a special case: the moments depend
    on the fixings, the discount on the settlement date. Checked by pricing the
    same fixing schedule with two settlement dates and recovering the ratio of
    discount factors exactly.
    """
    S, K, r, q, sigma = 100.0, 100.0, 0.05, 0.01, 0.25
    times = uniform_fixing_times(1.0, 12)
    early = discrete_geometric_price(
        S=S, K=K, T=1.0, r=r, sigma=sigma, fixing_times=times, q=q
    )
    late = discrete_geometric_price(
        S=S, K=K, T=1.5, r=r, sigma=sigma, fixing_times=times, q=q
    )
    assert late / early == pytest.approx(math.exp(-r * 0.5), rel=1e-14)


# --------------------------------------------------------------------------
# (a) The published anchors.
# --------------------------------------------------------------------------


def test_discrete_geometric_matches_the_clewlow_strickland_reference():
    """PUBLISHED_BENCHMARK: 5.3425606635, matched to 5.8e-11.

    `S = K = 100`, `q = 3%`, `r = 6%`, `sigma = 20%`, ten fixings over a year
    on an Actual/360-style schedule, which under the convention used here is
    `T = 1` with `t_i = i/10`. The published figure is quoted to ten decimals,
    and the formula in `qpl.engines.analytic.asian` reproduces it to **5.8e-11**
    -- i.e. to the last published digit -- which is a statement about the
    derivation, since the two were written independently.

    Value: Clewlow & Strickland, *Implementing Derivatives Models*, as carried
    in the QuantLib test suite (`asianoptions.cpp`). The same value is checked
    against QuantLib's own engine in `tests/oracle/test_asian_vs_quantlib.py`,
    where the agreement is one ulp.
    """
    assert EvidenceClass.PUBLISHED_BENCHMARK
    p = CLEWLOW_STRICKLAND
    value = discrete_geometric_price(
        S=p["S"],
        K=p["K"],
        T=p["T"],
        r=p["r"],
        sigma=p["sigma"],
        fixing_times=uniform_fixing_times(p["T"], p["n"]),
        q=p["q"],
        kind="call",
    )
    assert abs(value - CLEWLOW_STRICKLAND_VALUE) < 1e-8
    assert abs(value - CLEWLOW_STRICKLAND_VALUE) < 1e-10  # measured 5.8e-11


def test_dense_fixings_reproduce_the_haug_continuous_geometric_value():
    """PUBLISHED_BENCHMARK: 4.6922, matched to 3.7e-05 at 20 000 fixings.

    `S = 80`, `K = 85`, `q = -3%`, `r = 5%`, `sigma = 20%`, 90 days, the
    **put**. Two things about this row are worth stating because neither was in
    the slice statement.

    - It is a *put*, not a call. The geometric-average call at this point is
      0.4714; the published 4.6922 is the put, and the put is what is checked.
    - The published figure sits on a **90/360 = 0.25** year fraction, not
      90/365. At `T = 0.25` the exact continuous value is 4.6922213 and rounds
      to the published four decimals; at `T = 90/365 = 0.246575` it is
      4.6924339, which misses the published figure by 2.34e-04 -- more than
      twice the 1e-04 tolerance a four-decimal quote deserves. The convention is
      therefore Actual/360 and is pinned separately below.

    The discrete formula reaches the continuous value at order 1, so this row
    also states the cost of that: 20 000 fixings leave 1.5e-05 of
    discretisation, and the total gap to the published figure is 3.67e-05.
    """
    assert EvidenceClass.PUBLISHED_BENCHMARK
    p = HAUG_CONTINUOUS
    value = discrete_geometric_price(
        S=p["S"],
        K=p["K"],
        T=p["T"],
        r=p["r"],
        sigma=p["sigma"],
        fixing_times=uniform_fixing_times(p["T"], _DENSE_FIXINGS),
        q=p["q"],
        kind="put",
    )
    assert abs(value - HAUG_CONTINUOUS_PUT) < 1e-4
    # The exact continuous limit, for reference: the residual above is almost
    # all rounding of the published quote, not error in this formula.
    exact = _continuous_geometric(**p, kind="put")
    assert abs(exact - HAUG_CONTINUOUS_PUT) == pytest.approx(2.131e-05, rel=0.05)
    assert abs(value - exact) == pytest.approx(1.536e-05, rel=0.05)


def test_the_haug_value_is_quoted_on_a_ninety_over_three_sixty_year_fraction():
    """NEGATIVE_FINDING: the day-count convention is load-bearing at 1e-04.

    The slice was written expecting `T = 90/365`. That is a different contract
    by 1.4% of a year, and at four published decimals the difference is visible:
    4.6924339 against 4.6922213, a gap of 2.13e-04. Pinned so that a future
    change to the convention fails here with the reason rather than failing the
    benchmark row with a bare number.
    """
    assert EvidenceClass.NEGATIVE_FINDING
    p = dict(HAUG_CONTINUOUS)
    on_360 = _continuous_geometric(**p, kind="put")
    p["T"] = 90.0 / 365.0
    on_365 = _continuous_geometric(**p, kind="put")
    assert on_360 == pytest.approx(4.692221312, abs=1e-8)
    assert on_365 == pytest.approx(4.692433890, abs=1e-8)
    assert abs(on_365 - HAUG_CONTINUOUS_PUT) > 2.0e-4
    assert abs(on_360 - HAUG_CONTINUOUS_PUT) < 1e-4


def test_turnbull_wakeman_matches_its_published_approximation_value():
    """PUBLISHED_BENCHMARK **for the approximation**: 19.5152, matched to 9.8e-06.

    `S = 100`, `K = 80`, `r = q = 5%`, `sigma = 20%`, 26 fixings over half a
    year at `t_i = i T / 26`. Note what this row does and does not claim. It
    says that this implementation of the Turnbull-Wakeman construction agrees
    with the published Turnbull-Wakeman number; it says **nothing** about
    whether either is the true arithmetic-Asian price. The distance from the
    truth is measured separately, against a control-variate Monte Carlo run, and
    is reported with its sign.

    The dividend yield is `q = 5%`, equal to `r`. That was not stated in the
    slice brief and was recovered by matching: at `q = 0` the same construction
    gives 20.7865, and the published figure is reproduced only at `q = r`, where
    `E[A] = S` exactly (every forward equals the spot).
    """
    assert EvidenceClass.PUBLISHED_BENCHMARK
    p = TURNBULL_WAKEMAN
    times = uniform_fixing_times(p["T"], p["n"])
    value = turnbull_wakeman_price(
        S=p["S"], K=p["K"], T=p["T"], r=p["r"], sigma=p["sigma"], fixing_times=times, q=p["q"]
    )
    assert abs(value - TURNBULL_WAKEMAN_VALUE) < 1e-4
    assert abs(value - TURNBULL_WAKEMAN_VALUE) < 2e-5  # measured 9.78e-06
    # `q = r` makes every forward the spot, hence `E[A] = S` exactly.
    assert expected_arithmetic_average(
        S=p["S"], mu=p["r"] - p["q"], fixing_times=times
    ) == pytest.approx(p["S"], rel=1e-15)


# --------------------------------------------------------------------------
# (a) The measured discrete -> continuous order.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "point", "kind", "expected_order"),
    [
        ("clewlow_call", CLEWLOW_STRICKLAND, "call", 1.0002),
        ("haug_put", HAUG_CONTINUOUS, "put", 1.0096),
    ],
)
def test_discrete_to_continuous_convergence_is_first_order(
    label, point, kind, expected_order
):
    """CONVERGENCE_ORDER: the fixing grid refines at order **1**, not 2.

    Derived before it was measured. With `t_i = iT/n` the two moments of
    `log G` are

        tbar = (T/2)(1 + 1/n),   v = (sigma^2 T/3)(1 + 3/(2n) + 1/(2n^2)),

    so each carries an `O(1/n)` correction and the price inherits it. There is
    no cancellation between them -- a left-endpoint rule would give
    `tbar = (T/2)(1 - 1/n)` and the trapezoidal placement would give order 2,
    but the convention that puts the last fixing at expiry is a right-endpoint
    rule and is order 1. This is a real modelling consequence and not a defect:
    a monthly-fixing Asian is genuinely not a continuously averaged one, and the
    difference is `O(1/12)` of the continuous correction.

    Measured over `n` in (20, 40, ..., 2560), against the closed-form continuous
    limit:

        point          fitted order   log-residual   err(n=20)   err(n=2560)
        Clewlow call      1.0002         0.00017      2.030e-01    1.584e-03
        Haug put          1.0096         0.00917      1.622e-02    1.200e-04

    Errors are one-signed and halve at every level. The Haug leg's slightly
    larger residual is the `1/n^2` term in `v` showing up against a leading
    constant an order of magnitude smaller.
    """
    assert EvidenceClass.CONVERGENCE_ORDER
    limit = _continuous_geometric(
        S=point["S"],
        K=point["K"],
        T=point["T"],
        r=point["r"],
        q=point["q"],
        sigma=point["sigma"],
        kind=kind,
    )
    signed = [
        discrete_geometric_price(
            S=point["S"],
            K=point["K"],
            T=point["T"],
            r=point["r"],
            sigma=point["sigma"],
            fixing_times=uniform_fixing_times(point["T"], n),
            q=point["q"],
            kind=kind,
        )
        - limit
        for n in _ORDER_LEVELS
    ]
    errs = [abs(e) for e in signed]
    fit = fit_convergence_order([1.0 / n for n in _ORDER_LEVELS], errs)
    assert fit.order == pytest.approx(expected_order, abs=0.05), fit.order
    assert fit.residual < 0.02, fit.residual
    assert all(e > 0 for e in signed) or all(e < 0 for e in signed), signed
    assert all(a > b for a, b in pairwise(errs)), errs
    # Order 1 means the error halves; order 2 would quarter it. Pinned as a
    # ratio so the claim is not only a fitted slope: over a 128-fold refinement
    # the measured ratios are 128.06 (Clewlow call) and 135.16 (Haug put),
    # against 16384 for order 2.
    ratio = errs[0] / errs[-1]
    assert ratio == pytest.approx(_ORDER_LEVELS[-1] / _ORDER_LEVELS[0], rel=0.12), ratio


# --------------------------------------------------------------------------
# The dispatcher.
# --------------------------------------------------------------------------


def test_analytic_prices_a_geometric_asian_through_the_dispatcher():
    """The registered route returns the module route's number, with metadata."""
    p = CLEWLOW_STRICKLAND
    triple = _triple(
        S=p["S"],
        K=p["K"],
        T=p["T"],
        r=p["r"],
        q=p["q"],
        sigma=p["sigma"],
        times=uniform_fixing_times(p["T"], p["n"]),
    )
    result = price(*triple)
    assert result.value == pytest.approx(CLEWLOW_STRICKLAND_VALUE, abs=1e-8)
    assert result.meta is not None
    assert result.meta["instrument"] == "asian"
    assert result.meta["averaging"] == "geometric"
    assert result.meta["n_fixings"] == 10
    assert result.meta["expected_average"] == pytest.approx(101.3287504596, abs=1e-8)


def test_analytic_refuses_an_arithmetic_asian_and_says_what_to_use():
    """NotSupportedError, naming the approximations and Monte Carlo.

    The registry *does* have an entry for `(AsianOption, BlackScholesModel,
    "analytic")`; the refusal is inside the engine and depends on a field of the
    instrument. Registering nothing would have produced the generic
    "Unsupported instrument/model/market combination", which is true and
    useless. The requirement this encodes is the one from the slice statement:
    the dispatcher must never return an approximation as if it were exact, so
    `turnbull_wakeman_price` is reachable only by name.
    """
    triple = _triple(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        times=uniform_fixing_times(1.0, 12),
        averaging="arithmetic",
    )
    with pytest.raises(NotSupportedError) as excinfo:
        price(*triple)
    message = str(excinfo.value)
    assert "turnbull_wakeman_price" in message
    assert "method='mc'" in message
    assert "control_variate" in message


GREEKS_SPEC = {
    "S": 100.0,
    "K": 100.0,
    "T": 1.0,
    "r": 0.05,
    "q": 0.03,
    "sigma": 0.25,
}
"""A dividend yield and a non-default volatility, so a Greek that confused `r`
with `mu = r - q` or dropped a `sigma` shows up."""

GREEKS_NAMES = ("delta", "gamma", "vega", "theta", "rho")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_a_one_fixing_geometric_asian_has_the_black_scholes_greeks(kind):
    """Evidence class: EXACT_IDENTITY, and it is what fixes the theta convention.

    With a single fixing at expiry the geometric average IS `S_T`, so the
    contract is a European vanilla and all five Greeks must agree with
    `qpl.engines.analytic.black_scholes` to round-off. Measured worst absolute
    gap over both kinds: 7.1e-15 (on vega, whose scale is 37.9), and delta,
    gamma and rho agree to 1.1e-16 or exactly.

    The load-bearing one is **theta**. The settlement date enters the geometric
    Asian price only through `e^{-rT}`, so `-dV/dT` would be `-r V` -- a true
    derivative, and not time decay. This engine instead reports the derivative
    along the *roll*, where settlement and every fixing shift back together,
    and that is the definition that reproduces the Black-Scholes theta here.
    Nothing else does.
    """
    times = (GREEKS_SPEC["T"],)
    asian = _triple(**GREEKS_SPEC, times=times, kind=kind)
    vanilla = (
        EuropeanOption(kind=kind, strike=GREEKS_SPEC["K"], expiry=GREEKS_SPEC["T"]),
        asian[1],
        asian[2],
    )
    asian_greeks = greeks(*asian)
    vanilla_greeks = greeks(*vanilla)
    for name in GREEKS_NAMES:
        assert getattr(asian_greeks, name) == pytest.approx(
            getattr(vanilla_greeks, name), rel=1e-12, abs=1e-13
        ), name


@pytest.mark.parametrize("kind", ["call", "put"])
def test_geometric_asian_greeks_match_central_differences_of_the_closed_form(kind):
    """Evidence class: CLOSED_FORM against finite differences of the same formula.

    Not a cross-engine check -- it differentiates the price this module already
    ships -- but it is what catches an algebra slip in the chain rule, which is
    the only way these expressions can be wrong. Twelve fixings; central
    differences with `h = 1e-4` in the spot, `1e-5` in volatility and rate, and
    `1e-5` along the roll (settlement and every fixing shifted together, which
    is the derivative theta claims to be).

    Measured worst gaps, call and put: delta 1.2e-10, vega 2.4e-09, theta
    2.4e-09, rho 3.4e-09, gamma 4.1e-06. Gamma is three decimal orders looser
    than the rest and that is the second difference's own round-off floor
    (`O(eps / h**2)` with `h = 1e-4` is `O(1e-8)` on a price of order 6, i.e.
    exactly this size), not a disagreement.
    """
    times = uniform_fixing_times(GREEKS_SPEC["T"], 12)
    exact = discrete_geometric_greeks(**GREEKS_SPEC, fixing_times=times, kind=kind)

    def _price(**overrides) -> float:
        spec = {**GREEKS_SPEC, "fixing_times": times, "kind": kind}
        spec.update(overrides)
        return discrete_geometric_price(**spec)

    assert exact["value"] == pytest.approx(_price(), rel=1e-14)

    h_s, h = 1e-4, 1e-5
    s0 = GREEKS_SPEC["S"]
    fd = {
        "delta": (_price(S=s0 + h_s) - _price(S=s0 - h_s)) / (2 * h_s),
        "gamma": (_price(S=s0 + h_s) - 2 * _price() + _price(S=s0 - h_s)) / h_s**2,
        "vega": (
            _price(sigma=GREEKS_SPEC["sigma"] + h)
            - _price(sigma=GREEKS_SPEC["sigma"] - h)
        )
        / (2 * h),
        "rho": (_price(r=GREEKS_SPEC["r"] + h) - _price(r=GREEKS_SPEC["r"] - h))
        / (2 * h),
        "theta": (
            _price(
                T=GREEKS_SPEC["T"] - h, fixing_times=tuple(t - h for t in times)
            )
            - _price(
                T=GREEKS_SPEC["T"] + h, fixing_times=tuple(t + h for t in times)
            )
        )
        / (2 * h),
    }
    tolerances = {
        "delta": 1e-8,
        "gamma": 1e-4,
        "vega": 1e-6,
        "theta": 1e-6,
        "rho": 1e-6,
    }
    for name, value in fd.items():
        assert exact[name] == pytest.approx(value, abs=tolerances[name]), name


def test_geometric_asian_greeks_reach_the_registry_and_the_arithmetic_one_refuses():
    """Evidence class: NEGATIVE_FINDING on the arithmetic half.

    Slice 8 refused *both* averagings so that `greeks(..., method="analytic")`
    would not succeed or fail depending on a field of the instrument. Slice 10
    reverses that: the geometric Greeks are exact and are shipped, and the
    arithmetic ones raise with a reason that is about the mathematics rather
    than about scope. The reason is not "not implemented" -- the moment-matched
    approximations *are* differentiable -- it is that a derivative of an
    approximation whose error has no rigorous control has no rigorous control
    either, and `method="analytic"` must not hand one back.

    Which is the right shape after all: the field decides whether a *closed
    form exists*, and that is a fact about the contract, not an accident of the
    dispatcher.
    """
    times = uniform_fixing_times(GREEKS_SPEC["T"], 12)
    geometric = greeks(*_triple(**GREEKS_SPEC, times=times, averaging="geometric"))
    assert geometric.meta is not None
    assert geometric.meta["averaging"] == "geometric"
    assert geometric.meta["theta_convention"].startswith("roll")
    assert geometric.delta > 0.0

    triple = _triple(**GREEKS_SPEC, times=times, averaging="arithmetic")
    with pytest.raises(NotSupportedError) as excinfo:
        greeks(*triple)
    message = str(excinfo.value)
    assert "no rigorous control" in message
    assert "greeks_estimator='pathwise'" in message


def test_geometric_asian_greeks_refuse_the_degenerate_limits():
    """Both limits make the average deterministic, so gamma is a point mass.

    `T = 0` cannot even be built as an `AsianOption` -- a fixing schedule inside
    `(0, expiry]` would be empty -- so it is checked at the function level,
    while `sigma = 0` is reachable through the dispatcher. The vanilla engine
    refuses the same two limits one derivative lower, and
    `discrete_geometric_price` still returns the correct price at both.
    """
    times = (0.5, 1.0)
    base = {
        "S": GREEKS_SPEC["S"],
        "K": GREEKS_SPEC["K"],
        "r": GREEKS_SPEC["r"],
        "q": GREEKS_SPEC["q"],
        "fixing_times": times,
    }
    with pytest.raises(InvalidInputError, match="T must be > 0 for Greeks"):
        discrete_geometric_greeks(**base, T=0.0, sigma=0.25)
    with pytest.raises(InvalidInputError, match="sigma must be > 0 for Greeks"):
        discrete_geometric_greeks(**base, T=1.0, sigma=0.0)

    triple = _triple(**{**GREEKS_SPEC, "sigma": 0.0}, times=times)
    with pytest.raises(InvalidInputError, match="sigma must be > 0 for Greeks"):
        greeks(*triple)
    # The price is still exact in that limit.
    assert price(*triple).value > 0.0


@pytest.mark.parametrize("method", ["tree", "pde"])
def test_the_lattice_and_the_grid_do_not_price_an_asian(method):
    """NotSupportedError by lookup, not by a guard inside those engines.

    Nothing registers `AsianOption` for `method="tree"` or `method="pde"`:
    pricing an average needs a second state variable, which is a different
    discretisation rather than a branch inside the existing one. The refusal
    therefore comes from the registry's generic message, which is the correct
    one here -- there is no alternative to name.
    """
    from qpl.engines.pde.pricers import PDEConfig
    from qpl.engines.tree.pricers import TreeConfig

    triple = _triple(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        times=uniform_fixing_times(1.0, 12),
    )
    cfg = TreeConfig(n_steps=50) if method == "tree" else PDEConfig(n_s=50, n_t=50)
    with pytest.raises(NotSupportedError, match="Unsupported instrument"):
        price(*triple, method=method, cfg=cfg)
