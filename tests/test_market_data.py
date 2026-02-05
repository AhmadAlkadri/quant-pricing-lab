"""
Tests for market data retrieval.
Ensures caching works and network calls are mocked.
"""
import os
import shutil
import tempfile
import sys
from unittest.mock import patch, MagicMock

# Mock yfinance before importing qpl.market.data to avoid ModuleNotFoundError
sys.modules["yfinance"] = MagicMock()

import pandas as pd
import pytest
from qpl.market.data import get_prices


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for caching."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_yf_download():
    """Mock yfinance.download to avoid network access."""
    with patch("yfinance.download") as mock:
        yield mock

def test_fetch_and_cache_miss(temp_cache_dir, mock_yf_download):
    """Test that a cache miss calls download and saves file."""
    # Setup mock return
    dates = pd.date_range("2023-01-01", "2023-01-05")
    mock_df = pd.DataFrame({"Close": [100.0] * 5}, index=dates)
    mock_yf_download.return_value = mock_df

    # 1. First call: miss
    df1 = get_prices("FAKE", "2023-01-01", "2023-01-05", cache_dir=temp_cache_dir)
    
    assert mock_yf_download.called, "Should call yfinance on cache miss"
    assert len(df1) == 5
    assert not df1.empty
    
    # 2. Check file exists in cache
    files = os.listdir(temp_cache_dir)
    assert len(files) == 1
    assert files[0].startswith("FAKE_")
    assert files[0].endswith(".parquet")

def test_cache_hit(temp_cache_dir, mock_yf_download):
    """Test that a cache hit loads from file and does NOT call download."""
    # Setup: manually write a file to cache
    dates = pd.date_range("2023-01-01", "2023-01-05")
    mock_df = pd.DataFrame({"Close": [100.0] * 5}, index=dates)
    
    # Run once to populate cache (mocking the download)
    mock_yf_download.return_value = mock_df
    get_prices("FAKE", "2023-01-01", "2023-01-05", cache_dir=temp_cache_dir)
    
    # Reset mock
    mock_yf_download.reset_mock()
    
    # 2. Second call: hit
    df2 = get_prices("FAKE", "2023-01-01", "2023-01-05", cache_dir=temp_cache_dir)
    
    assert not mock_yf_download.called, "Should NOT call yfinance on cache hit"
    pd.testing.assert_frame_equal(df2, mock_df)

def test_bad_source_error(temp_cache_dir):
    with pytest.raises(ValueError, match="Unsupported data source"):
        get_prices("FAKE", "2023-01-01", "2023-01-05", source="invalid", cache_dir=temp_cache_dir)

def test_io_error_on_empty(temp_cache_dir, mock_yf_download):
    """Test that an IOError is raised if download returns empty."""
    mock_yf_download.return_value = pd.DataFrame() # Empty
    
    with pytest.raises(IOError, match="No data found"):
        get_prices("FAKE", "2023-01-01", "2023-01-05", cache_dir=temp_cache_dir)
