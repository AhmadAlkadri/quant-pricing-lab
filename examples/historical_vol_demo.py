"""
Demo: Historical Volatility Estimation from Market Data (Slice 3).
"""
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# --- MOCKING SETUP ---
# Since this demo might be run in an environment without internet or valid yfinance,
# we need to robustly handle the missing dependency or just use the system one if available.
# However, the user request asks for a "minimal demo that ties together: get_prices -> compute realized sigma".
# To ensure this runs reliably for the user right now (given previous issues), I will
# wrap the get_prices call.

try:
    from qpl.market.data import get_prices
except ImportError:
    print("Warning: qpl.market.data could not be imported. Check dependencies.")
    sys.exit(1)

from qpl.market.stats import log_returns, realized_volatility

def main():
    print("--- Historical Volatility from Time Series ---")
    
    ticker = "SPY"
    start = "2023-01-01"
    end = "2023-06-30"
    
    print(f"1. Fetching prices for {ticker} ({start} to {end})...")
    try:
        # Use cache_dir='.' or default to make it visible
        df = get_prices(ticker, start, end)
        print(f"   Retrieved {len(df)} records.")
        print(f"   Head:\n{df.head(3)}")
    except Exception as e:
        print(f"   Failed to fetch data: {e}")
        print("   -> Creating synthetic data for demonstration purposes.")
        dates = pd.date_range(start, end, freq="B") # Business days
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=len(dates))))
        df = pd.DataFrame({"Close": prices}, index=dates)
    
    # 2. Compute Log Returns
    # Access the numpy array from the DataFrame
    prices_arr = df["Close"].values
    
    try:
        rets = log_returns(prices_arr)
        print(f"\n2. Computed {len(rets)} log returns.")
        print(f"   First 5 returns: {rets[:5]}")
    except Exception as e:
        print(f"   Error computing returns: {e}")
        return

    # 3. Compute Realized Volatility
    # Default: annualization=252 (daily), demean=True
    vol_ann = realized_volatility(rets)
    
    print(f"\n3. Annualized Realized Volatility: {vol_ann:.4f} ({vol_ann:.2%})")
    print("   (Assumes 252 trading days per year)")

if __name__ == "__main__":
    main()
