from __future__ import annotations

from qpl.engines.tree import TreeConfig
from qpl.instruments.options import AmericanOption, EuropeanOption
from qpl.market.curves import FlatDividendCurve, FlatRateCurve
from qpl.market.market import Market
from qpl.models.black_scholes import BlackScholesModel
from qpl.pricing import price


def main() -> None:
    strike = 100.0
    expiry = 1.0

    market = Market(
        spot=100.0,
        rate_curve=FlatRateCurve(0.05),
        dividend_curve=FlatDividendCurve(0.01),
    )
    model = BlackScholesModel(sigma=0.20)
    cfg = TreeConfig(n_steps=200)

    american = price(
        AmericanOption(kind="put", strike=strike, expiry=expiry),
        model,
        market,
        method="tree",
        cfg=cfg,
    )
    european = price(
        EuropeanOption(kind="put", strike=strike, expiry=expiry),
        model,
        market,
        method="analytic",
    )

    premium = american.value - european.value
    meta = american.meta or {}

    print("example=american_put_binomial_dp")
    print(f"american_put={american.value:.6f}")
    print(f"european_put={european.value:.6f}")
    print(f"early_exercise_premium={premium:.6f}")
    print(f"early_exercise_nodes={int(meta.get('early_exercise_node_count', 0))}")


if __name__ == "__main__":
    main()
