
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Mock yfinance and structlog before importing predictor_agent
sys.modules['yfinance'] = MagicMock()
sys.modules['structlog'] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath("Multi-Agent-Investment-System-main"))

from src.predictor.predictor_agent import TargetPredictorAgent, HistoricalAnalysis

def test_backtest_nan():
    agent = TargetPredictorAgent()
    
    # Mock plan
    plan = {
        'allocations': [
            {'symbol': 'TEST', 'shares': 10, 'weight': 1.0}
        ]
    }
    
    # Case 1: Zero prices (division by zero)
    dates = pd.date_range(start='2024-01-01', periods=100)
    prices = pd.Series(np.zeros(100), index=dates)
    historical_data = {'TEST': prices}
    
    print("Testing with zero prices...")
    try:
        result = agent.backtest_historical_performance(plan, historical_data)
        print(f"Result: {result}")
        if np.isnan(result.predicted_return):
            print("FAIL: predicted_return is NaN")
        else:
            print("PASS: predicted_return is not NaN")
    except Exception as e:
        print(f"Error: {e}")

    # Case 2: Constant prices (zero return, but valid)
    prices = pd.Series(np.ones(100) * 100, index=dates)
    historical_data = {'TEST': prices}
    
    print("\nTesting with constant prices...")
    try:
        result = agent.backtest_historical_performance(plan, historical_data)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Case 3: NaN in prices
    prices = pd.Series(np.random.randn(100) + 100, index=dates)
    prices.iloc[50] = np.nan
    historical_data = {'TEST': prices}
    
    print("\nTesting with NaN in prices...")
    try:
        result = agent.backtest_historical_performance(plan, historical_data)
        print(f"Result: {result}")
        if np.isnan(result.predicted_return):
            print("FAIL: predicted_return is NaN")
    except Exception as e:
        print(f"Error: {e}")
    
    # Case 4: Empty data (should be handled by length check, but let's see)
    dates_short = pd.date_range(start='2024-01-01', periods=10)
    prices_short = pd.Series(np.ones(10) * 100, index=dates_short)
    historical_data_short = {'TEST': prices_short}
    
    print("\nTesting with short data...")
    try:
        result = agent.backtest_historical_performance(plan, historical_data_short)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_backtest_nan()
