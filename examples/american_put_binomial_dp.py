from __future__ import annotations

from qpl.engines.dp import BinomialDPConfig, price_american_put_binomial
from qpl.instruments.options import EuropeanOption
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

    dp_cfg = BinomialDPConfig(n_steps=200)
    american = price_american_put_binomial(
        strike=strike,
        expiry=expiry,
        market=market,
        model=model,
        cfg=dp_cfg,
        return_lattice=True,
    )

    european = price(
        EuropeanOption(kind="put", strike=strike, expiry=expiry),
        model,
        market,
        method="analytic",
    )

    premium = american.value - european.value
    early_ex_nodes = int((american.meta or {}).get("early_exercise_node_count", 0))

    print("example=american_put_binomial_dp")
    print(f"american_put={american.value:.6f}")
    print(f"european_put={european.value:.6f}")
    print(f"early_exercise_premium={premium:.6f}")
    print(f"early_exercise_nodes={early_ex_nodes}")


if __name__ == "__main__":
    main()
