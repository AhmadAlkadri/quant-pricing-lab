"""
Demo of Historical Volatility Estimation.

This script demonstrates:
1. Creating a synthetic price series.
2. Computing annualized historical volatility.
"""
import numpy as np
import matplotlib.pyplot as plt
from qpl.utils.statistics import historical_volatility

def main():
    print("--- Historical Volatility Demo ---")
    
    # 1. Generate synthetic prices (Geometric Brownian Motion)
    # S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    np.random.seed(42)  # For reproducibility
    
    sigma_true = 0.30   # 30% volatility
    mu = 0.05           # 5% drift
    dt = 1/252          # Daily steps
    n_days = 252 * 2    # 2 years of data
    S0 = 100.0
    
    prices = [S0]
    for _ in range(n_days):
        z = np.random.normal()
        ret = (mu - 0.5 * sigma_true**2) * dt + sigma_true * np.sqrt(dt) * z
        prices.append(prices[-1] * np.exp(ret))
    
    prices = np.array(prices)
    print(f"Generated {len(prices)} prices with true vol = {sigma_true:.2%}")
    print(f"Prices: {prices[:5]} ... {prices[-5:]}")
    
    # 2. Compute historical volatility
    vol_est = historical_volatility(prices, annualization_factor=252, demean=True)
    
    print(f"\nEstimated Volatility (demean=True):  {vol_est:.4f} ({vol_est:.2%})")
    print(f"Error compared to True Vol: {vol_est - sigma_true:.4f}")
    
    # Check without demeaning (assuming 0 mean return)
    vol_est_nodean = historical_volatility(prices, annualization_factor=252, demean=False)
    print(f"Estimated Volatility (demean=False): {vol_est_nodean:.4f} ({vol_est_nodean:.2%})")

if __name__ == "__main__":
    main()
