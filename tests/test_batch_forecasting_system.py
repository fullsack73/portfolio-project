
import pandas as pd
import numpy as np
import pytest

from src.batch_forecasting_system import BatchForecastingSystem
from src.batch_forecasting_config import get_default_batch_config

@pytest.fixture
def dummy_data():
    tickers = ['AAPL', 'GOOG', 'MSFT']
    dates = pd.to_datetime(pd.date_range('2023-01-01', periods=100))
    data = pd.DataFrame(np.random.rand(100, 3), index=dates, columns=tickers)
    return tickers, data

def test_batch_forecasting_system_init():
    """Test initialization of BatchForecastingSystem."""
    config = get_default_batch_config()
    system = BatchForecastingSystem(config)
    assert system is not None
    assert system.config == config

def test_batch_forecasting_system_e2e(dummy_data):
    """Test end-to-end batch forecasting."""
    tickers, data = dummy_data
    config = get_default_batch_config()
    system = BatchForecastingSystem(config)

    forecasts = system.forecast_batch_returns(tickers, data)

    assert isinstance(forecasts, dict)
    assert set(forecasts.keys()) == set(tickers)
    for ticker, value in forecasts.items():
        assert isinstance(value, float)
