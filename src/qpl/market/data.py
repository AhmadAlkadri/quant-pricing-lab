"""
Market data retrieval with local caching.
"""

import os
import hashlib
from typing import Optional
import pandas as pd
import yfinance as yf

# Common interval alias mapping to standardized filename part
INTERVAL_ALIASES = {
    "1d": "1d",
    "daily": "1d"
}

def get_prices(
    ticker: str,
    start: str,
    end: str,
    *,
    source: str = "yahoo",
    interval: str = "1d",
    cache_dir: str = ".market_cache"
) -> pd.DataFrame:
    """
    Fetch historical prices for a ticker, using local disk cache if available.

    Parameters
    ----------
    ticker : str
        The ticker symbol (e.g. "SPY", "^GSPC").
    start : str
        Start date string (YYYY-MM-DD).
    end : str
        End date string (YYYY-MM-DD).
    source : str, default "yahoo"
        The data source. Currently only "yahoo" is supported.
    interval : str, default "1d"
        Data interval (e.g. "1d").
    cache_dir : str, default ".market_cache"
        Directory to store cached data files.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and at least a 'Close' column.

    Raises
    ------
    ValueError
        If inputs are invalid or source is unsupported.
    IOError
        If data cannot be fetched and is not in cache.
    """
    if source != "yahoo":
        raise ValueError(f"Unsupported data source: {source}")

    interval = INTERVAL_ALIASES.get(interval, interval)
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create deterministic cache filename
    # We hash the inputs to handle special characters in tickers/dates safely
    key_str = f"{source}_{ticker}_{start}_{end}_{interval}"
    key_hash = hashlib.md5(key_str.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{ticker}_{key_hash}.parquet")
    
    # 1. Try to load from cache
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            print(f"[MarketData] Loaded {ticker} from cache: {cache_path}")
            return df
        except Exception as e:
            print(f"[MarketData] Cache load failed, refetching. Error: {e}")
            # If cache is corrupted, proceed to fetch
            pass

    # 2. Fetch from network
    print(f"[MarketData] Fetching {ticker} from {source}...")
    try:
        # yfinance download
        # auto_adjust=True gives adjusted close as 'Close'
        df = yf.download(
            ticker, 
            start=start, 
            end=end, 
            interval=interval, 
            auto_adjust=True, 
            progress=False
        )
        
        if df.empty:
            raise IOError(f"No data found for {ticker} from {source}")

        # Ensure we have a Close column
        if "Close" not in df.columns:
            # Fallback if auto_adjust failed to behave as expected
            if "Adj Close" in df.columns:
                df = df.rename(columns={"Adj Close": "Close"})
            elif "Close" not in df.columns:
                 raise IOError(f"Data for {ticker} missing 'Close' column")

        # 3. Save to cache
        # Use parquet for efficiency and type preservation
        df.to_parquet(cache_path)
        print(f"[MarketData] Saved {ticker} to cache: {cache_path}")
        
        return df

    except Exception as e:
        raise IOError(f"Failed to fetch data for {ticker}: {e}")
