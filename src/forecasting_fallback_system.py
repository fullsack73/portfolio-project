"""
Graceful Degradation and Fallback System for Forecasting

This module implements comprehensive fallback mechanisms when advanced models fail,
including cached prediction usage and historical mean fallbacks.
"""

import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from cache_manager import get_cache


class PredictionCache:
    """Manages cached predictions for fallback scenarios."""
    
    def __init__(self, cache_ttl_hours: int = 24):
        """
        Initialize prediction cache.
        
        Args:
            cache_ttl_hours: Time-to-live for cached predictions in hours
        """
        self.logger = logging.getLogger('forecasting.prediction_cache')
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.cache = get_cache()
    
    def _generate_cache_key(self, ticker: str, model_name: str, periods: int) -> str:
        """Generate cache key for prediction."""
        key_string = f"forecast_prediction_{ticker}_{model_name}_{periods}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def store_prediction(self, 
                        ticker: str, 
                        model_name: str, 
                        periods: int,
                        prediction: float,
                        confidence: Optional[float] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a prediction in cache.
        
        Args:
            ticker: Ticker symbol
            model_name: Name of the model that generated the prediction
            periods: Number of periods forecasted
            prediction: The prediction value
            confidence: Optional confidence score
            metadata: Optional additional metadata
        """
        try:
            cache_key = self._generate_cache_key(ticker, model_name, periods)
            
            cache_data = {
                'ticker': ticker,
                'model_name': model_name,
                'periods': periods,
                'prediction': float(prediction),
                'confidence': confidence,
                'timestamp': datetime.now(),
                'metadata': metadata or {}
            }
            
            # Store with TTL (use expire parameter for MultiLevelCache)
            try:
                self.cache.set(cache_key, cache_data, expire=int(self.cache_ttl.total_seconds()))
            except TypeError:
                # Fallback if expire parameter not supported
                self.cache.set(cache_key, cache_data)
            
            self.logger.debug(f"Cached prediction for {ticker}/{model_name}: {prediction}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache prediction for {ticker}: {e}")
    
    def get_cached_prediction(self, 
                            ticker: str, 
                            model_name: str, 
                            periods: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached prediction.
        
        Args:
            ticker: Ticker symbol
            model_name: Name of the model
            periods: Number of periods
            
        Returns:
            Cached prediction data or None if not found/expired
        """
        try:
            cache_key = self._generate_cache_key(ticker, model_name, periods)
            cached_data = self.cache.get(cache_key)
            
            if cached_data is None:
                return None
            
            # Check if cache is still valid
            cache_time = cached_data.get('timestamp')
            if cache_time and isinstance(cache_time, datetime):
                if datetime.now() - cache_time > self.cache_ttl:
                    self.logger.debug(f"Cached prediction expired for {ticker}/{model_name}")
                    return None
            
            self.logger.debug(f"Retrieved cached prediction for {ticker}/{model_name}: {cached_data['prediction']}")
            return cached_data
            
        except Exception as e:
            self.logger.warning(f"Failed to retrieve cached prediction for {ticker}: {e}")
            return None
    
    def get_best_cached_prediction(self, ticker: str, periods: int, exclude_models: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get the best available cached prediction for a ticker.
        
        Args:
            ticker: Ticker symbol
            periods: Number of periods
            exclude_models: List of model names to exclude from cache lookup
            
        Returns:
            Best cached prediction or None
        """
        exclude_models = exclude_models or []
        
        # List of model names to check in order of preference
        model_priority = [
            'ensemble', 'lstm', 'sarimax', 'xgboost', 'lightgbm', 
            'catboost', 'arima', 'exponential_smoothing', 'linear_trend',
            'volatility_adjusted', 'historical_mean'
        ]
        
        for model_name in model_priority:
            if model_name in exclude_models:
                self.logger.debug(f"Skipping cached prediction from excluded model {model_name} for {ticker}")
                continue
                
            cached_pred = self.get_cached_prediction(ticker, model_name, periods)
            if cached_pred is not None:
                self.logger.info(f"Using cached prediction from {model_name} for {ticker}")
                return cached_pred
        
        return None
    
    def clear_cache_for_ticker(self, ticker: str) -> None:
        """Clear all cached predictions for a ticker."""
        try:
            # This is a simplified implementation
            # In practice, you might need to iterate through possible cache keys
            self.logger.info(f"Cache clearing requested for {ticker}")
        except Exception as e:
            self.logger.warning(f"Failed to clear cache for {ticker}: {e}")


class HistoricalFallbackForecaster:
    """Provides historical-based fallback forecasting methods."""
    
    def __init__(self):
        self.logger = logging.getLogger('forecasting.historical_fallback')
    
    def historical_mean_forecast(self, 
                                data: pd.Series, 
                                periods: int = 1,
                                lookback_days: int = 252) -> float:
        """
        Generate forecast using historical mean.
        
        Args:
            data: Historical time series data
            periods: Number of periods to forecast (not used for mean)
            lookback_days: Number of days to look back for calculation
            
        Returns:
            Historical mean forecast
        """
        try:
            if data is None or data.empty:
                self.logger.warning("No data available for historical mean forecast")
                return 0.05  # Conservative default
            
            # Use recent data for better relevance
            recent_data = data.tail(min(lookback_days, len(data)))
            
            if recent_data.empty:
                return 0.05
            
            # Calculate returns if data appears to be prices
            if recent_data.min() > 0 and recent_data.mean() > 1:
                # Likely price data, calculate returns
                returns = recent_data.pct_change().dropna()
                if not returns.empty:
                    mean_return = float(returns.mean())
                    self.logger.debug(f"Historical mean return: {mean_return:.6f}")
                    return mean_return
            
            # Otherwise treat as returns data
            mean_value = float(recent_data.mean())
            self.logger.debug(f"Historical mean value: {mean_value:.6f}")
            return mean_value
            
        except Exception as e:
            self.logger.error(f"Historical mean forecast failed: {e}")
            return 0.05
    
    def exponential_smoothing_forecast(self, 
                                     data: pd.Series, 
                                     periods: int = 1,
                                     alpha: float = 0.3) -> float:
        """
        Generate forecast using exponential smoothing.
        
        Args:
            data: Historical time series data
            periods: Number of periods to forecast
            alpha: Smoothing parameter
            
        Returns:
            Exponentially smoothed forecast
        """
        try:
            if data is None or data.empty or len(data) < 2:
                return self.historical_mean_forecast(data, periods)
            
            # Convert to returns if necessary
            if data.min() > 0 and data.mean() > 1:
                returns = data.pct_change().dropna()
                if returns.empty:
                    return 0.05
                work_data = returns
            else:
                work_data = data
            
            # Apply exponential smoothing
            smoothed = float(work_data.iloc[0])
            for value in work_data.iloc[1:]:
                smoothed = alpha * float(value) + (1 - alpha) * smoothed
            
            self.logger.debug(f"Exponential smoothing forecast: {smoothed:.6f}")
            return smoothed
            
        except Exception as e:
            self.logger.error(f"Exponential smoothing forecast failed: {e}")
            return self.historical_mean_forecast(data, periods)
    
    def linear_trend_forecast(self, 
                            data: pd.Series, 
                            periods: int = 1,
                            lookback_days: int = 60) -> float:
        """
        Generate forecast using linear trend extrapolation.
        
        Args:
            data: Historical time series data
            periods: Number of periods to forecast
            lookback_days: Number of days to use for trend calculation
            
        Returns:
            Linear trend forecast
        """
        try:
            if data is None or data.empty or len(data) < 10:
                return self.historical_mean_forecast(data, periods)
            
            # Use recent data for trend calculation
            recent_data = data.tail(min(lookback_days, len(data)))
            
            # Convert to returns if necessary
            if recent_data.min() > 0 and recent_data.mean() > 1:
                returns = recent_data.pct_change().dropna()
                if returns.empty or len(returns) < 5:
                    return self.historical_mean_forecast(data, periods)
                work_data = returns
            else:
                work_data = recent_data
            
            # Calculate linear trend
            x = np.arange(len(work_data))
            y = work_data.values
            
            # Simple linear regression
            n = len(x)
            if n < 2:
                return self.historical_mean_forecast(data, periods)
            
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            # Calculate slope and intercept
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:  # Avoid division by zero
                return self.historical_mean_forecast(data, periods)
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
            
            # Forecast
            forecast_x = len(work_data) + periods - 1
            forecast = slope * forecast_x + intercept
            
            # Sanity check
            if abs(forecast) > 2.0:  # Cap at ±200%
                forecast = np.sign(forecast) * min(abs(forecast), 0.5)
                self.logger.warning(f"Capped extreme linear trend forecast to {forecast:.6f}")
            
            self.logger.debug(f"Linear trend forecast: {forecast:.6f}")
            return float(forecast)
            
        except Exception as e:
            self.logger.error(f"Linear trend forecast failed: {e}")
            return self.historical_mean_forecast(data, periods)
    
    def volatility_adjusted_forecast(self, 
                                   data: pd.Series, 
                                   periods: int = 1,
                                   lookback_days: int = 30) -> float:
        """
        Generate forecast adjusted for recent volatility.
        
        Args:
            data: Historical time series data
            periods: Number of periods to forecast
            lookback_days: Number of days for volatility calculation
            
        Returns:
            Volatility-adjusted forecast
        """
        try:
            if data is None or data.empty or len(data) < lookback_days:
                return self.historical_mean_forecast(data, periods)
            
            # Calculate base forecast using historical mean
            base_forecast = self.historical_mean_forecast(data, periods, lookback_days)
            
            # Calculate recent volatility
            recent_data = data.tail(lookback_days)
            
            if recent_data.min() > 0 and recent_data.mean() > 1:
                returns = recent_data.pct_change().dropna()
            else:
                returns = recent_data
            
            if returns.empty or len(returns) < 5:
                return base_forecast
            
            volatility = float(returns.std())
            
            # Adjust forecast based on volatility regime
            if volatility > returns.std() * 2:  # High volatility regime
                # Be more conservative
                adjustment_factor = 0.5
            elif volatility < returns.std() * 0.5:  # Low volatility regime
                # Can be slightly more aggressive
                adjustment_factor = 1.2
            else:
                adjustment_factor = 1.0
            
            adjusted_forecast = base_forecast * adjustment_factor
            
            self.logger.debug(f"Volatility-adjusted forecast: {adjusted_forecast:.6f} (volatility: {volatility:.6f})")
            return adjusted_forecast
            
        except Exception as e:
            self.logger.error(f"Volatility-adjusted forecast failed: {e}")
            return self.historical_mean_forecast(data, periods)


class GracefulDegradationSystem:
    """Main system for graceful degradation and fallback forecasting."""
    
    def __init__(self):
        self.logger = logging.getLogger('forecasting.graceful_degradation')
        self.prediction_cache = PredictionCache()
        self.historical_forecaster = HistoricalFallbackForecaster()
        
        # Fallback method priority order
        self.fallback_methods = [
            ('cached_prediction', self._get_cached_fallback),
            ('exponential_smoothing', self._get_exponential_smoothing_fallback),
            ('linear_trend', self._get_linear_trend_fallback),
            ('volatility_adjusted', self._get_volatility_adjusted_fallback),
            ('historical_mean', self._get_historical_mean_fallback),
            ('ultimate_fallback', self._get_ultimate_fallback)
        ]
    
    def get_fallback_prediction(self, 
                              ticker: str,
                              data: Optional[pd.Series] = None,
                              periods: int = 1,
                              failed_models: Optional[List[str]] = None) -> Tuple[float, str]:
        """
        Get fallback prediction using the best available method.
        
        Args:
            ticker: Ticker symbol
            data: Historical data (if available)
            periods: Number of periods to forecast
            failed_models: List of models that have already failed
            
        Returns:
            Tuple of (prediction_value, method_used)
        """
        failed_models = failed_models or []
        
        self.logger.info(f"Attempting fallback prediction for {ticker} (failed models: {failed_models})")
        
        for method_name, method_func in self.fallback_methods:
            # Skip methods that are in the failed models list
            if failed_models and method_name in failed_models:
                self.logger.debug(f"Skipping failed method {method_name} for {ticker}")
                continue
                
            try:
                # Handle cached prediction with failed model exclusion
                if method_name == 'cached_prediction':
                    prediction = self._get_cached_fallback(ticker, data, periods, failed_models)
                else:
                    prediction = method_func(ticker, data, periods)
                
                if prediction is not None and not np.isnan(prediction) and not np.isinf(prediction):
                    self.logger.info(f"Fallback prediction for {ticker} using {method_name}: {prediction:.6f}")
                    return float(prediction), method_name
                
            except Exception as e:
                self.logger.warning(f"Fallback method {method_name} failed for {ticker}: {e}")
                continue
        
        # If all methods fail, return ultimate fallback
        self.logger.error(f"All fallback methods failed for {ticker}, using ultimate fallback")
        return 0.05, 'ultimate_fallback'
    
    def _get_cached_fallback(self, ticker: str, data: Optional[pd.Series], periods: int, exclude_models: Optional[List[str]] = None) -> Optional[float]:
        """Get cached prediction fallback."""
        cached_data = self.prediction_cache.get_best_cached_prediction(ticker, periods, exclude_models)
        if cached_data:
            return cached_data['prediction']
        return None
    
    def _get_exponential_smoothing_fallback(self, ticker: str, data: Optional[pd.Series], periods: int) -> Optional[float]:
        """Get exponential smoothing fallback."""
        if data is not None and not data.empty:
            return self.historical_forecaster.exponential_smoothing_forecast(data, periods)
        return None
    
    def _get_linear_trend_fallback(self, ticker: str, data: Optional[pd.Series], periods: int) -> Optional[float]:
        """Get linear trend fallback."""
        if data is not None and not data.empty:
            return self.historical_forecaster.linear_trend_forecast(data, periods)
        return None
    
    def _get_volatility_adjusted_fallback(self, ticker: str, data: Optional[pd.Series], periods: int) -> Optional[float]:
        """Get volatility-adjusted fallback."""
        if data is not None and not data.empty:
            return self.historical_forecaster.volatility_adjusted_forecast(data, periods)
        return None
    
    def _get_historical_mean_fallback(self, ticker: str, data: Optional[pd.Series], periods: int) -> Optional[float]:
        """Get historical mean fallback."""
        if data is not None and not data.empty:
            return self.historical_forecaster.historical_mean_forecast(data, periods)
        return None
    
    def _get_ultimate_fallback(self, ticker: str, data: Optional[pd.Series], periods: int) -> float:
        """Ultimate fallback - always returns a value."""
        return 0.05  # Conservative 5% annual return assumption
    
    def cache_successful_prediction(self, 
                                  ticker: str,
                                  model_name: str,
                                  periods: int,
                                  prediction: float,
                                  confidence: Optional[float] = None) -> None:
        """
        Cache a successful prediction for future fallback use.
        
        Args:
            ticker: Ticker symbol
            model_name: Name of the model that generated the prediction
            periods: Number of periods forecasted
            prediction: The prediction value
            confidence: Optional confidence score
        """
        try:
            self.prediction_cache.store_prediction(
                ticker, model_name, periods, prediction, confidence
            )
        except Exception as e:
            self.logger.warning(f"Failed to cache prediction for {ticker}: {e}")
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get statistics about fallback usage."""
        # This would be implemented to track fallback method usage
        return {
            'fallback_methods_available': len(self.fallback_methods),
            'cache_enabled': True,
            'historical_methods_available': 4
        }


# Global instance for easy access
graceful_degradation_system = GracefulDegradationSystem()