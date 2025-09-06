from pmdarima import auto_arima
import numpy as np
import logging
from scipy.stats import linregress

logger = logging.getLogger(__name__)

class ARIMA():
    """ARIMA-based forecasting model to replace linear trend forecasting."""
    
    def __init__(self, seasonal=False, suppress_warnings=True):
        """
        Initialize ARIMA model.
        
        Args:
            seasonal: Whether to use seasonal ARIMA (SARIMA)
            suppress_warnings: Whether to suppress model fitting warnings
        """
        self.seasonal = seasonal
        self.suppress_warnings = suppress_warnings
    
    def forecast(self, prices):
        """
        Forecast annual return using ARIMA model.
        
        Args:
            prices: Array-like of historical prices
            
        Returns:
            float: Expected annual return percentage
        """
        if len(prices) < 10:
            logger.warning("Insufficient data points for ARIMA forecast")
            return 0.05
            
        try:
            # Convert prices to returns for better ARIMA performance
            returns = np.diff(prices) / prices[:-1]
            
            # Fit ARIMA model
            model = auto_arima(
                returns,
                seasonal=self.seasonal,
                suppress_warnings=self.suppress_warnings,
                error_action='ignore'
            )
            
            # Forecast next 252 days (1 year) of returns
            forecast_returns, conf_int = model.predict(
                n_periods=252,
                return_conf_int=True
            )
            
            # Calculate cumulative return
            cumulative_return = np.prod(1 + forecast_returns) - 1
            
            return cumulative_return
            
        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}")
            # Fallback to simple linear trend if ARIMA fails
            x = np.arange(len(prices)).reshape(-1, 1)
            slope, intercept, _, _, _ = linregress(x.flatten(), prices)
            future_price = slope * (len(prices) + 252) + intercept
            current_price = prices[-1]
            return (future_price / current_price) - 1 if current_price > 0 else 0.05
