from __future__ import annotations

import math

from qpl.engines.pde.pricers import PDEConfig
from qpl.instruments.options import EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price
from qpl.validation import fit_convergence_order


def _market(spot: float, r: float, q: float) -> Market:
    return Market(
        spot=spot,
        rate_curve=FlatRateCurve(r),
        dividend_curve=FlatDividendCurve(q),
    )


def test_ch4_theta_prices_close_to_analytic_call() -> None:
    s = 100.0
    k = 100.0
    t = 1.0
    r = 0.05
    q = 0.01
    sigma = 0.2

    option = EuropeanOption(kind="call", strike=k, expiry=t)
    model = BlackScholesModel(sigma=sigma)
    market = _market(s, r, q)
    analytic = price(option, model, market, method="analytic").value

    for theta in (0.0, 0.5, 1.0):
        cfg = PDEConfig(n_s=50, n_t=50, theta=theta, s_max_multiplier=4.0)
        pde = price(option, model, market, method="pde", cfg=cfg).value
        assert math.isfinite(pde)
        assert abs(pde - analytic) <= 0.05


# Reference case used by the convergence tests below and by
# `docs/notes/pde_strike_alignment.md`: S = K = 100, r = 5%, q = 0,
# sigma = 20%, T = 1, European call. Held in one place so the measured error
# tables in the note and in the tests describe the same experiment.
_CONV_SPOT = 100.0
_CONV_STRIKE = 100.0
_CONV_EXPIRY = 1.0
_CONV_RATE = 0.05
_CONV_DIVIDEND = 0.0
_CONV_SIGMA = 0.2


def _conv_setup() -> tuple[EuropeanOption, BlackScholesModel, Market, float]:
    option = EuropeanOption(kind="call", strike=_CONV_STRIKE, expiry=_CONV_EXPIRY)
    model = BlackScholesModel(sigma=_CONV_SIGMA)
    market = _market(_CONV_SPOT, _CONV_RATE, _CONV_DIVIDEND)
    analytic = price(option, model, market, method="analytic").value
    return option, model, market, analytic


def _pde_error(
    option: EuropeanOption,
    model: BlackScholesModel,
    market: Market,
    analytic: float,
    *,
    n_s: int,
    n_t: int,
    theta: float,
    alignment: str,
    time_stepping: str = "theta",
) -> float:
    cfg = PDEConfig(
        n_s=n_s,
        n_t=n_t,
        theta=theta,
        s_max_multiplier=4.0,
        strike_alignment=alignment,  # type: ignore[arg-type]
        time_stepping=time_stepping,  # type: ignore[arg-type]
    )
    value = price(option, model, market, method="pde", cfg=cfg).value
    return abs(value - analytic)


def test_pde_cn_unaligned_strike_pathology_negative_finding() -> None:
    """Refining an unaligned grid can make the error *worse*, not better.

    Evidence class: NEGATIVE_FINDING. This test pins a known failure mode so
    that it cannot silently change and cannot be mistaken for a bug elsewhere.

    With `s_max_multiplier=4.0` and S=100 the grid spans [0, 400]. At n_s=50
    the spacing is 8 and the strike K=100 falls halfway between nodes 12 and
    13, which is the favourable case. At n_s=100 the spacing is 4 and K falls
    exactly on node 25. The payoff kink then sits on a grid point, the local
    truncation error of the stencil straddling it is no longer second order,
    and the error constant degrades enough to swamp the factor-of-four gain
    from halving the spacing. Measured absolute errors against the closed
    form are 7.6e-3 at n=50 and 4.0e-2 at n=100: refinement loses a factor
    of five.

    The remedy is `strike_alignment="midpoint"`, exercised by the next test.
    """
    option, model, market, analytic = _conv_setup()

    errs = {
        n: _pde_error(
            option, model, market, analytic, n_s=n, n_t=n, theta=0.5, alignment="none"
        )
        for n in (50, 100, 200)
    }

    # The pathology itself: refining 50 -> 100 increases the error.
    assert errs[100] > errs[50]
    # And it is large, not a marginal wobble.
    assert errs[100] > 4.0 * errs[50]
    # Refining further recovers, which is why a single monotonicity check over
    # a lucky grid triple hides this entirely.
    assert errs[200] < errs[100]


def test_pde_cn_aligned_strike_order_two() -> None:
    """Crank-Nicolson on a strike-aligned grid converges at order two.

    Evidence class: CONVERGENCE_ORDER. `n_s = n_t = n` is refined together, so
    ds and dt shrink at the same rate and the single fitted slope is the joint
    order of the scheme. Crank-Nicolson is O(dt**2) in time and the central
    spatial stencil is O(ds**2), so the expected slope is 2.

    The band [1.8, 2.2] is not a fudge factor: the fit is contaminated from
    below by the coarsest grids, which are not yet asymptotic, and from above
    by the truncation of the domain at s_max = 4S, whose error decays with
    s_max rather than with n and therefore does not participate in the rate.
    A band of +/- 0.2 admits those two effects and nothing else; a scheme that
    had genuinely dropped to first order (slope ~1) or that was not converging
    at all would fail. The measured slope on this case is 1.997 with a
    log-space residual of 0.022, comfortably inside the band.
    """
    option, model, market, analytic = _conv_setup()

    levels = (50, 100, 200, 400, 800)
    h = [1.0 / n for n in levels]
    errs = [
        _pde_error(
            option, model, market, analytic, n_s=n, n_t=n, theta=0.5, alignment="midpoint"
        )
        for n in levels
    ]

    fit = fit_convergence_order(h, errs)

    assert 1.8 <= fit.order <= 2.2
    # A clean power law: the residual is the RMS of the log-space deviations,
    # so 0.1 corresponds to roughly 10% scatter about the fitted line.
    assert fit.residual < 0.1
    assert fit.n_points == len(levels)


def test_pde_implicit_euler_first_order_in_time() -> None:
    """Fully implicit time stepping is first order in dt.

    Evidence class: CONVERGENCE_ORDER. Refining n_s and n_t together, as the
    Crank-Nicolson test does, would confound the O(dt) time error with the
    O(ds**2) space error. Here the spatial grid is held fixed and fine
    (n_s=800, strike aligned) while only n_t is refined, so the fitted slope
    is the temporal order alone.

    The spatial error at n_s=800 is a floor the refinement cannot cross. It
    was measured at 1.9e-4 by driving n_t to 6400 on the same grid. The
    temporal errors over n_t in (25, 50, 100, 200) run from 4.2e-2 down to
    5.3e-3, so the floor stays at least a factor of 25 below the smallest
    error in the fit and the temporal term dominates throughout. Measured
    slope 0.997 with log-space residual 3e-4.
    """
    option, model, market, analytic = _conv_setup()

    n_s = 800
    levels = (25, 50, 100, 200)
    h = [1.0 / n_t for n_t in levels]
    errs = [
        _pde_error(
            option,
            model,
            market,
            analytic,
            n_s=n_s,
            n_t=n_t,
            theta=1.0,
            alignment="midpoint",
        )
        for n_t in levels
    ]

    # The smallest error in the fit must stay well clear of the spatial floor,
    # otherwise the slope would be measuring the floor and not the scheme.
    assert min(errs) > 25.0 * 1.9e-4

    fit = fit_convergence_order(h, errs)

    assert 0.8 <= fit.order <= 1.2
    assert fit.residual < 0.05


def test_ch4_call_monotonicity_in_spot_for_each_theta() -> None:
    k = 100.0
    t = 1.0
    r = 0.05
    q = 0.01
    sigma = 0.2

    option = EuropeanOption(kind="call", strike=k, expiry=t)
    model = BlackScholesModel(sigma=sigma)
    spots = (90.0, 100.0, 110.0)

    for theta in (0.0, 0.5, 1.0):
        prices: list[float] = []
        for s in spots:
            cfg = PDEConfig(n_s=50, n_t=50, theta=theta, s_max_multiplier=4.0)
            pde = price(option, model, _market(s, r, q), method="pde", cfg=cfg).value
            prices.append(pde)

        assert prices[0] <= prices[1] + 1e-12
        assert prices[1] <= prices[2] + 1e-12


def test_pde_rannacher_preserves_second_order_price_convergence() -> None:
    """Rannacher start-up keeps the price at order two.

    Evidence class: CONVERGENCE_ORDER. Four fully implicit half steps replace
    the first two Crank-Nicolson steps, so the start-up is locally first order
    in dt over an interval of length 2*dt. That contributes O(dt**2) to the
    global error -- which is the whole point of the construction, and the
    reason it is four *half* steps rather than two *full* ones -- so the
    order-2 price convergence measured for plain Crank-Nicolson on an aligned
    grid must survive.

    Measured on the reference ATM call, n_s = n_t = n over (50, 100, 200, 400,
    800), strike-aligned: order 1.9973 with log-space residual 0.0213, against
    1.9972 / 0.0224 for plain Crank-Nicolson. The band is the same +/- 0.2 the
    plain-CN test uses and for the same two reasons.

    The constant is checked separately below; it is the only thing that moves.
    """
    option, model, market, analytic = _conv_setup()

    levels = (50, 100, 200, 400, 800)
    h = [1.0 / n for n in levels]
    errs = [
        _pde_error(
            option,
            model,
            market,
            analytic,
            n_s=n,
            n_t=n,
            theta=0.5,
            alignment="midpoint",
            time_stepping="rannacher",
        )
        for n in levels
    ]

    fit = fit_convergence_order(h, errs)

    assert 1.8 <= fit.order <= 2.2
    assert fit.residual < 0.1
    assert fit.n_points == len(levels)


def test_pde_rannacher_price_error_constant_is_slightly_worse_than_plain_cn() -> None:
    """The damping is not free: it costs about 5% on the price error constant.

    Evidence class: CONVERGENCE_ORDER (the claim is about the constant of a
    measured power law, not about a single price).

    Four implicit half steps are only first-order accurate over the 2*dt they
    cover, so they add an O(dt**2) term with the same sign as Crank-Nicolson's
    own. Measured scaled errors n**2 * |error| on the reference ATM call:

    | n   | theta   | rannacher |
    |-----|---------|-----------|
    | 50  | 19.0195 | 20.0666   |
    | 100 | 20.3437 | 21.3925   |
    | 200 | 19.2799 | 20.3290   |
    | 400 | 19.5074 | 20.5566   |
    | 800 | 19.6144 | 20.6636   |

    The ratio is 1.052-1.055 across the whole sequence: a constant factor, as a
    same-order perturbation must be, not a growing gap. Both engines undershoot
    the closed form at every n, so the comparison is between two same-signed
    errors. The bounds below are deliberately loose enough to survive a
    round-off-level change and tight enough that the ratio turning into a
    growth in n (i.e. an order loss) would fail.
    """
    option, model, market, analytic = _conv_setup()

    levels = (50, 100, 200, 400, 800)
    ratios = []
    for n in levels:
        base = _pde_error(
            option, model, market, analytic, n_s=n, n_t=n, theta=0.5, alignment="midpoint"
        )
        damped = _pde_error(
            option,
            model,
            market,
            analytic,
            n_s=n,
            n_t=n,
            theta=0.5,
            alignment="midpoint",
            time_stepping="rannacher",
        )
        ratios.append(damped / base)

    assert all(1.02 <= ratio <= 1.10 for ratio in ratios), ratios
    # A constant factor, not a drift: the spread across a 16x refinement is
    # measured at 0.003.
    assert max(ratios) - min(ratios) < 0.02, ratios
