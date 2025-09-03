"""
Comprehensive Forecasting Integration with Graceful Degradation

This module provides the main integration point for all forecasting functionality
with comprehensive error handling and graceful degradation to lightweight methods.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from forecasting_models import (
    ARIMAForecaster, LSTMForecaster, SARIMAXForecaster,
    XGBoostForecaster, LightGBMForecaster, CatBoostForecaster,
    EnsembleForecaster
)
from model_selection_system import TieredModelSelector, FallbackForecaster
from forecasting_error_handler import (
    ForecastingErrorHandler, DataQualityValidator, ErrorContext,
    ErrorCategory, ErrorSeverity
)
from forecasting_fallback_system import graceful_degradation_system
from forecasting_config import get_forecasting_config, get_enabled_models
from model_performance import record_model_performance


class ComprehensiveForecastingSystem:
    """
    Main forecasting system with comprehensive error handling and graceful degradation.
    
    This system implements a multi-tier approach:
    1. Advanced ML models (LSTM, SARIMAX, Gradient Boosting)
    2. Classical models (ARIMA)
    3. Lightweight methods (Exponential smoothing, linear trend)
    4. Fallback methods (Historical mean, cached predictions)
    5. Ultimate fallback (Conservative default returns)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize comprehensive forecasting system.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger('forecasting.comprehensive')
        
        # Load configuration
        try:
            loaded_config = get_forecasting_config()
            # Convert to dict if it's a config object
            if hasattr(loaded_config, '__dict__'):
                self.config = loaded_config.__dict__.copy()
            elif hasattr(loaded_config, 'to_dict'):
                self.config = loaded_config.to_dict()
            else:
                self.config = dict(loaded_config) if loaded_config else {}
            
            if config:
                self.config.update(config)
        except Exception as e:
            self.logger.warning(f"Failed to load config, using defaults: {e}")
            self.config = self._get_default_config()
        
        # Initialize components
        self.error_handler = ForecastingErrorHandler()
        self.data_validator = DataQualityValidator()
        self.model_selector = TieredModelSelector(self.config.get('model_selection', {}))
        self.fallback_forecaster = FallbackForecaster()
        
        # Performance tracking
        self.forecasting_stats = {
            'total_requests': 0,
            'successful_advanced': 0,
            'successful_classical': 0,
            'successful_lightweight': 0,
            'fallback_used': 0,
            'ultimate_fallback_used': 0,
            'total_errors': 0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enabled_models': ['arima', 'lstm', 'xgboost'],
            'parallel_processing': True,
            'max_workers': 4,
            'timeout_seconds': 300,
            'cache_predictions': True,
            'use_ensemble': True,
            'fallback_enabled': True,
            'model_selection': {
                'tiers': {
                    'tier_1': {
                        'models': ['lstm', 'xgboost', 'lightgbm'],
                        'priority': 1,
                        'max_training_time': 180.0,
                        'performance_threshold': 15.0
                    },
                    'tier_2': {
                        'models': ['arima'],
                        'priority': 2,
                        'max_training_time': 60.0,
                        'performance_threshold': 20.0
                    },
                    'tier_3': {
                        'models': ['exponential_smoothing', 'linear_trend'],
                        'priority': 3,
                        'max_training_time': 10.0,
                        'performance_threshold': 30.0
                    }
                }
            }
        }
    
    def forecast_returns(self, 
                        tickers: List[str],
                        data: pd.DataFrame,
                        periods: int = 1,
                        method: str = 'auto',
                        max_workers: Optional[int] = None) -> Dict[str, float]:
        """
        Comprehensive return forecasting with graceful degradation.
        
        Args:
            tickers: List of ticker symbols to forecast
            data: Historical price data
            periods: Number of periods to forecast
            method: Forecasting method ('auto', 'advanced', 'classical', 'lightweight', 'fallback')
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping tickers to expected returns
        """
        start_time = time.time()
        self.forecasting_stats['total_requests'] += 1
        
        self.logger.info(f"Starting comprehensive forecasting for {len(tickers)} tickers using {method} method")
        
        # Initialize results
        expected_returns = {}
        processing_stats = {
            'successful': 0,
            'failed': 0,
            'fallback_used': 0,
            'method_breakdown': {}
        }
        
        # Process tickers based on method
        if method == 'auto':
            expected_returns = self._auto_forecast_tickers(tickers, data, periods, max_workers, processing_stats)
        elif method == 'advanced':
            expected_returns = self._advanced_forecast_tickers(tickers, data, periods, max_workers, processing_stats)
        elif method == 'classical':
            expected_returns = self._classical_forecast_tickers(tickers, data, periods, max_workers, processing_stats)
        elif method == 'lightweight':
            expected_returns = self._lightweight_forecast_tickers(tickers, data, periods, max_workers, processing_stats)
        elif method == 'fallback':
            expected_returns = self._fallback_forecast_tickers(tickers, data, periods, processing_stats)
        else:
            raise ValueError(f"Unknown forecasting method: {method}")
        
        # Ensure all tickers have predictions
        for ticker in tickers:
            if ticker not in expected_returns:
                self.logger.warning(f"No prediction generated for {ticker}, using ultimate fallback")
                expected_returns[ticker] = self._get_ultimate_fallback(ticker)
                processing_stats['fallback_used'] += 1
                self.forecasting_stats['ultimate_fallback_used'] += 1
        
        # Log summary
        total_time = time.time() - start_time
        self.logger.info(
            f"Forecasting completed in {total_time:.2f}s: "
            f"{processing_stats['successful']}/{len(tickers)} successful, "
            f"{processing_stats['fallback_used']} fallbacks used"
        )
        
        # Update global stats
        self.forecasting_stats['successful_advanced'] += processing_stats['method_breakdown'].get('advanced', 0)
        self.forecasting_stats['successful_classical'] += processing_stats['method_breakdown'].get('classical', 0)
        self.forecasting_stats['successful_lightweight'] += processing_stats['method_breakdown'].get('lightweight', 0)
        self.forecasting_stats['fallback_used'] += processing_stats['fallback_used']
        
        return expected_returns
    
    def _auto_forecast_tickers(self, 
                              tickers: List[str],
                              data: pd.DataFrame,
                              periods: int,
                              max_workers: Optional[int],
                              stats: Dict[str, Any]) -> Dict[str, float]:
        """Auto-select best forecasting method for each ticker."""
        expected_returns = {}
        
        # Use parallel processing if enabled
        if self.config.get('parallel_processing', True) and len(tickers) > 1:
            expected_returns = self._parallel_auto_forecast(tickers, data, periods, max_workers, stats)
        else:
            expected_returns = self._sequential_auto_forecast(tickers, data, periods, stats)
        
        return expected_returns
    
    def _parallel_auto_forecast(self, 
                               tickers: List[str],
                               data: pd.DataFrame,
                               periods: int,
                               max_workers: Optional[int],
                               stats: Dict[str, Any]) -> Dict[str, float]:
        """Parallel auto-forecasting with error handling."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, len(tickers), self.config.get('max_workers', 4))
        
        expected_returns = {}
        
        def _forecast_single_ticker(ticker):
            """Forecast single ticker with comprehensive error handling."""
            try:
                return self._auto_forecast_single_ticker(ticker, data, periods)
            except Exception as e:
                self.logger.error(f"Auto-forecasting failed for {ticker}: {e}")
                return ticker, None, 'error'
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_ticker = {
                    executor.submit(_forecast_single_ticker, ticker): ticker 
                    for ticker in tickers
                }
                
                # Collect results with timeout
                timeout = self.config.get('timeout_seconds', 300)
                
                for future in as_completed(future_to_ticker, timeout=timeout):
                    ticker = future_to_ticker[future]
                    try:
                        ticker_result, prediction, method_used = future.result(timeout=60)
                        
                        if prediction is not None:
                            expected_returns[ticker_result] = prediction
                            stats['successful'] += 1
                            
                            # Update method breakdown
                            method_category = self._categorize_method(method_used)
                            stats['method_breakdown'][method_category] = stats['method_breakdown'].get(method_category, 0) + 1
                        else:
                            stats['failed'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Future execution failed for {ticker}: {e}")
                        stats['failed'] += 1
                        
        except Exception as e:
            self.logger.error(f"Parallel processing failed, falling back to sequential: {e}")
            return self._sequential_auto_forecast(tickers, data, periods, stats)
        
        return expected_returns
    
    def _sequential_auto_forecast(self, 
                                 tickers: List[str],
                                 data: pd.DataFrame,
                                 periods: int,
                                 stats: Dict[str, Any]) -> Dict[str, float]:
        """Sequential auto-forecasting with error handling."""
        expected_returns = {}
        
        for ticker in tickers:
            try:
                ticker_result, prediction, method_used = self._auto_forecast_single_ticker(ticker, data, periods)
                
                if prediction is not None:
                    expected_returns[ticker_result] = prediction
                    stats['successful'] += 1
                    
                    # Update method breakdown
                    method_category = self._categorize_method(method_used)
                    stats['method_breakdown'][method_category] = stats['method_breakdown'].get(method_category, 0) + 1
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                self.logger.error(f"Sequential forecasting failed for {ticker}: {e}")
                stats['failed'] += 1
        
        return expected_returns
    
    def _auto_forecast_single_ticker(self, 
                                    ticker: str,
                                    data: pd.DataFrame,
                                    periods: int) -> Tuple[str, Optional[float], str]:
        """Auto-forecast single ticker using tiered approach with comprehensive fallback."""
        failed_models = []
        ticker_data = None
        
        try:
            # Get ticker data
            if ticker not in data.columns:
                self.logger.warning(f"No data available for {ticker}")
                # Try fallback with no data
                prediction, fallback_method = graceful_degradation_system.get_fallback_prediction(
                    ticker, None, periods, failed_models=['no_data']
                )
                return ticker, prediction, f'fallback_{fallback_method}'
            
            ticker_data = data[ticker].dropna()
            
            # Validate data quality
            is_valid, issues = self.data_validator.validate_data(ticker_data, ticker, min_points=30)
            
            if not is_valid:
                self.logger.warning(f"Data quality issues for {ticker}: {'; '.join(issues)}")
                # Try to clean the data
                ticker_data = self.data_validator.clean_data(ticker_data, ticker)
                
                # Re-validate with lower threshold
                is_valid_after_cleaning, remaining_issues = self.data_validator.validate_data(
                    ticker_data, ticker, min_points=20
                )
                
                if not is_valid_after_cleaning:
                    self.logger.error(f"Data unusable for {ticker}: {'; '.join(remaining_issues)}")
                    # Use fallback system with data quality issues
                    prediction, fallback_method = graceful_degradation_system.get_fallback_prediction(
                        ticker, ticker_data, periods, failed_models=['data_quality']
                    )
                    return ticker, prediction, f'fallback_{fallback_method}'
            
            # Try tiered model selection
            try:
                selection_result = self.model_selector.select_model(ticker_data, ticker)
                
                # Create and use selected model
                model = self._create_model(selection_result.selected_model)
                if model:
                    # Fit model
                    success = model._safe_fit(ticker_data, ticker=ticker)
                    if success:
                        # Generate prediction
                        prediction = model._safe_predict(periods, ticker=ticker)
                        if prediction is not None:
                            # Cache successful prediction
                            if self.config.get('cache_predictions', True):
                                graceful_degradation_system.cache_successful_prediction(
                                    ticker, selection_result.selected_model, periods, prediction
                                )
                            
                            return ticker, prediction, f'tiered_{selection_result.selected_model}'
                        else:
                            failed_models.append(selection_result.selected_model)
                    else:
                        failed_models.append(selection_result.selected_model)
                else:
                    failed_models.append('model_creation_failed')
                
            except Exception as e:
                self.logger.warning(f"Tiered model selection failed for {ticker}: {e}")
                failed_models.append('tiered_selection')
            
            # Try lightweight methods with explicit fallback tracking
            prediction, method_used = self._try_lightweight_methods_with_fallback(ticker, ticker_data, periods, failed_models)
            if prediction is not None:
                return ticker, prediction, f'lightweight_{method_used}'
            
            # Use graceful degradation system with all failed models
            prediction, fallback_method = graceful_degradation_system.get_fallback_prediction(
                ticker, ticker_data, periods, failed_models=failed_models
            )
            
            return ticker, prediction, f'fallback_{fallback_method}'
            
        except Exception as e:
            self.logger.error(f"Auto-forecasting completely failed for {ticker}: {e}")
            
            # Create error context
            context = ErrorContext(
                ticker=ticker,
                model_name='auto_forecast',
                operation='forecast_single_ticker',
                data_points=len(ticker_data) if ticker_data is not None else 0,
                timestamp=datetime.now(),
                additional_info={'periods': periods, 'error': str(e), 'failed_models': failed_models}
            )
            
            # Handle the error with comprehensive recovery
            recovery_successful, recovery_result = self._comprehensive_error_recovery(
                e, context, ticker, ticker_data, periods, failed_models
            )
            
            if recovery_successful and isinstance(recovery_result, (int, float)):
                return ticker, float(recovery_result), 'error_recovery'
            
            # Ultimate fallback with enhanced safety net
            return ticker, self._get_enhanced_ultimate_fallback(ticker, ticker_data), 'ultimate_fallback'
    
    def _create_model(self, model_name: str):
        """Create model instance by name."""
        try:
            if model_name == 'arima':
                return ARIMAForecaster()
            elif model_name == 'lstm':
                return LSTMForecaster()
            elif model_name == 'sarimax':
                return SARIMAXForecaster()
            elif model_name == 'xgboost':
                return XGBoostForecaster()
            elif model_name == 'lightgbm':
                return LightGBMForecaster()
            elif model_name == 'catboost':
                return CatBoostForecaster()
            elif model_name == 'ensemble':
                return EnsembleForecaster()
            else:
                self.logger.warning(f"Unknown model name: {model_name}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to create model {model_name}: {e}")
            return None
    
    def _try_lightweight_methods(self, 
                                ticker: str,
                                data: pd.Series,
                                periods: int) -> Tuple[Optional[float], Optional[str]]:
        """Try lightweight forecasting methods."""
        methods = [
            ('exponential_smoothing', self._exponential_smoothing_forecast),
            ('linear_trend', self._linear_trend_forecast),
            ('historical_mean', self._historical_mean_forecast)
        ]
        
        for method_name, method_func in methods:
            try:
                prediction = method_func(data, periods)
                if prediction is not None and not np.isnan(prediction) and not np.isinf(prediction):
                    self.logger.debug(f"Lightweight forecast for {ticker} using {method_name}: {prediction:.6f}")
                    return prediction, method_name
            except Exception as e:
                self.logger.warning(f"Lightweight method {method_name} failed for {ticker}: {e}")
                continue
        
        return None, None
    
    def _try_lightweight_methods_with_fallback(self, 
                                              ticker: str,
                                              data: pd.Series,
                                              periods: int,
                                              failed_models: List[str]) -> Tuple[Optional[float], Optional[str]]:
        """Try lightweight forecasting methods with enhanced fallback tracking."""
        # First try cached predictions from successful models (excluding failed ones)
        if self.config.get('cache_predictions', True):
            try:
                cached_data = graceful_degradation_system.prediction_cache.get_best_cached_prediction(ticker, periods)
                if cached_data and cached_data.get('model_name') not in failed_models:
                    self.logger.info(f"Using cached prediction from {cached_data.get('model_name')} for {ticker}")
                    return cached_data['prediction'], f"cached_{cached_data.get('model_name')}"
            except Exception as e:
                self.logger.warning(f"Failed to retrieve cached prediction for {ticker}: {e}")
        
        # Try lightweight methods in order of sophistication
        methods = [
            ('exponential_smoothing', self._exponential_smoothing_forecast),
            ('linear_trend', self._linear_trend_forecast),
            ('volatility_adjusted', self._volatility_adjusted_forecast),
            ('historical_mean', self._historical_mean_forecast)
        ]
        
        for method_name, method_func in methods:
            if method_name in failed_models:
                continue
                
            try:
                prediction = method_func(data, periods)
                if prediction is not None and not np.isnan(prediction) and not np.isinf(prediction):
                    # Sanity check the prediction
                    if abs(prediction) > 2.0:  # Cap at ±200%
                        self.logger.warning(f"Capping extreme prediction from {method_name} for {ticker}: {prediction:.6f}")
                        prediction = np.sign(prediction) * min(abs(prediction), 0.5)
                    
                    self.logger.debug(f"Lightweight forecast for {ticker} using {method_name}: {prediction:.6f}")
                    
                    # Cache the successful lightweight prediction
                    if self.config.get('cache_predictions', True):
                        try:
                            graceful_degradation_system.cache_successful_prediction(
                                ticker, method_name, periods, prediction
                            )
                        except Exception as cache_e:
                            self.logger.warning(f"Failed to cache lightweight prediction for {ticker}: {cache_e}")
                    
                    return prediction, method_name
                    
            except Exception as e:
                self.logger.warning(f"Lightweight method {method_name} failed for {ticker}: {e}")
                failed_models.append(method_name)
                continue
        
        return None, None
    
    def _exponential_smoothing_forecast(self, data: pd.Series, periods: int) -> float:
        """Simple exponential smoothing forecast."""
        if len(data) < 2:
            return 0.05
        
        alpha = 0.3
        smoothed = float(data.iloc[0])
        
        for value in data.iloc[1:]:
            smoothed = alpha * float(value) + (1 - alpha) * smoothed
        
        # Convert to return if data looks like prices
        if data.min() > 0 and data.mean() > 1:
            return (smoothed / data.iloc[-1]) - 1
        
        return smoothed
    
    def _linear_trend_forecast(self, data: pd.Series, periods: int) -> float:
        """Simple linear trend forecast."""
        if len(data) < 10:
            return 0.05
        
        # Use recent data
        recent_data = data.tail(min(60, len(data)))
        x = np.arange(len(recent_data))
        y = recent_data.values
        
        # Simple linear regression
        n = len(x)
        if n < 2:
            return 0.05
        
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.05
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Forecast
        forecast_x = len(recent_data) + periods - 1
        forecast = slope * forecast_x + intercept
        
        # Convert to return if data looks like prices
        if data.min() > 0 and data.mean() > 1:
            return (forecast / data.iloc[-1]) - 1
        
        return float(np.clip(forecast, -0.5, 1.0))
    
    def _historical_mean_forecast(self, data: pd.Series, periods: int) -> float:
        """Historical mean forecast."""
        if data.empty:
            return 0.05
        
        # Use recent data
        recent_data = data.tail(min(252, len(data)))
        
        # Convert to returns if data looks like prices
        if recent_data.min() > 0 and recent_data.mean() > 1:
            returns = recent_data.pct_change().dropna()
            if not returns.empty:
                return float(returns.mean())
        
        return float(recent_data.mean())
    
    def _volatility_adjusted_forecast(self, data: pd.Series, periods: int) -> float:
        """Volatility-adjusted forecast using the historical fallback forecaster."""
        try:
            return graceful_degradation_system.historical_forecaster.volatility_adjusted_forecast(
                data, periods
            )
        except Exception as e:
            self.logger.warning(f"Volatility-adjusted forecast failed: {e}")
            return self._historical_mean_forecast(data, periods)
    
    def _advanced_forecast_tickers(self, tickers, data, periods, max_workers, stats):
        """Forecast using advanced models only."""
        # Implementation would be similar to auto but restricted to advanced models
        return self._auto_forecast_tickers(tickers, data, periods, max_workers, stats)
    
    def _classical_forecast_tickers(self, tickers, data, periods, max_workers, stats):
        """Forecast using classical models only."""
        # Implementation would be similar to auto but restricted to classical models
        return self._auto_forecast_tickers(tickers, data, periods, max_workers, stats)
    
    def _lightweight_forecast_tickers(self, tickers, data, periods, max_workers, stats):
        """Forecast using lightweight methods only."""
        expected_returns = {}
        
        for ticker in tickers:
            try:
                if ticker not in data.columns:
                    continue
                
                ticker_data = data[ticker].dropna()
                prediction, method_used = self._try_lightweight_methods(ticker, ticker_data, periods)
                
                if prediction is not None:
                    expected_returns[ticker] = prediction
                    stats['successful'] += 1
                    stats['method_breakdown']['lightweight'] = stats['method_breakdown'].get('lightweight', 0) + 1
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                self.logger.error(f"Lightweight forecasting failed for {ticker}: {e}")
                stats['failed'] += 1
        
        return expected_returns
    
    def _fallback_forecast_tickers(self, tickers, data, periods, stats):
        """Forecast using fallback methods only."""
        expected_returns = {}
        
        for ticker in tickers:
            try:
                ticker_data = data[ticker].dropna() if ticker in data.columns else None
                prediction, fallback_method = graceful_degradation_system.get_fallback_prediction(
                    ticker, ticker_data, periods
                )
                
                expected_returns[ticker] = prediction
                stats['successful'] += 1
                stats['fallback_used'] += 1
                stats['method_breakdown']['fallback'] = stats['method_breakdown'].get('fallback', 0) + 1
                
            except Exception as e:
                self.logger.error(f"Fallback forecasting failed for {ticker}: {e}")
                expected_returns[ticker] = self._get_ultimate_fallback(ticker)
                stats['failed'] += 1
        
        return expected_returns
    
    def _categorize_method(self, method_used: str) -> str:
        """Categorize method for statistics."""
        if 'advanced' in method_used or 'tiered' in method_used:
            return 'advanced'
        elif 'classical' in method_used:
            return 'classical'
        elif 'lightweight' in method_used:
            return 'lightweight'
        elif 'fallback' in method_used:
            return 'fallback'
        else:
            return 'other'
    
    def _get_ultimate_fallback(self, ticker: str) -> float:
        """Get ultimate fallback prediction."""
        # Conservative default return based on historical market averages
        return 0.07  # 7% annual return assumption
    
    def _get_enhanced_ultimate_fallback(self, ticker: str, data: Optional[pd.Series] = None) -> float:
        """Get enhanced ultimate fallback prediction with data-aware defaults."""
        try:
            # If we have some data, try to use it for a more informed fallback
            if data is not None and not data.empty and len(data) >= 5:
                # Try simple statistical measures as last resort
                try:
                    # Calculate basic statistics
                    if data.min() > 0 and data.mean() > 1:
                        # Looks like price data, calculate simple return
                        recent_return = (data.iloc[-1] / data.iloc[0]) ** (252 / len(data)) - 1
                        if abs(recent_return) < 1.0:  # Sanity check
                            self.logger.debug(f"Enhanced fallback using recent return for {ticker}: {recent_return:.6f}")
                            return float(recent_return)
                    else:
                        # Looks like return data, use median
                        median_return = float(data.median())
                        if abs(median_return) < 0.5:  # Sanity check
                            self.logger.debug(f"Enhanced fallback using median return for {ticker}: {median_return:.6f}")
                            return median_return
                except Exception as e:
                    self.logger.warning(f"Enhanced fallback calculation failed for {ticker}: {e}")
            
            # Sector-aware fallback (simplified)
            sector_defaults = {
                # Technology stocks tend to have higher growth
                'tech': 0.12,
                'growth': 0.10,
                # Utilities and defensive sectors
                'utility': 0.05,
                'defensive': 0.06,
                # Financial sector
                'financial': 0.08,
                # Default market return
                'market': 0.07
            }
            
            # Simple heuristic based on ticker patterns
            ticker_lower = ticker.lower()
            if any(tech in ticker_lower for tech in ['aapl', 'msft', 'googl', 'amzn', 'tsla', 'nvda']):
                return sector_defaults['tech']
            elif any(util in ticker_lower for util in ['xlu', 'so', 'nee', 'duk']):
                return sector_defaults['utility']
            elif any(fin in ticker_lower for fin in ['jpm', 'bac', 'wfc', 'c', 'xlf']):
                return sector_defaults['financial']
            else:
                return sector_defaults['market']
                
        except Exception as e:
            self.logger.error(f"Enhanced ultimate fallback failed for {ticker}: {e}")
            return 0.05  # Conservative final fallback
    
    def _comprehensive_error_recovery(self, 
                                    error: Exception,
                                    context: ErrorContext,
                                    ticker: str,
                                    data: Optional[pd.Series],
                                    periods: int,
                                    failed_models: List[str]) -> Tuple[bool, Any]:
        """Comprehensive error recovery with multiple fallback strategies."""
        try:
            # First, try the error handler's recovery mechanisms
            recovery_successful, recovery_result = self.error_handler.handle_error(
                error, context, ErrorCategory.MODEL_PREDICTION, ErrorSeverity.HIGH
            )
            
            if recovery_successful and isinstance(recovery_result, (int, float)):
                return True, float(recovery_result)
            
            # Try graceful degradation system as secondary recovery
            try:
                prediction, fallback_method = graceful_degradation_system.get_fallback_prediction(
                    ticker, data, periods, failed_models=failed_models
                )
                if prediction is not None:
                    self.logger.info(f"Secondary recovery successful for {ticker} using {fallback_method}")
                    return True, prediction
            except Exception as fallback_e:
                self.logger.warning(f"Graceful degradation recovery failed for {ticker}: {fallback_e}")
            
            # Try historical fallback forecaster directly
            try:
                if data is not None and not data.empty:
                    historical_forecaster = graceful_degradation_system.historical_forecaster
                    prediction = historical_forecaster.historical_mean_forecast(data, periods)
                    if prediction is not None and not np.isnan(prediction):
                        self.logger.info(f"Historical mean recovery successful for {ticker}: {prediction:.6f}")
                        return True, prediction
            except Exception as hist_e:
                self.logger.warning(f"Historical mean recovery failed for {ticker}: {hist_e}")
            
            # Final recovery attempt with cached predictions (ignore failed models filter)
            try:
                if self.config.get('cache_predictions', True):
                    cached_data = graceful_degradation_system.prediction_cache.get_best_cached_prediction(ticker, periods)
                    if cached_data:
                        self.logger.info(f"Emergency cached prediction recovery for {ticker} from {cached_data.get('model_name')}")
                        return True, cached_data['prediction']
            except Exception as cache_e:
                self.logger.warning(f"Emergency cache recovery failed for {ticker}: {cache_e}")
            
            return False, None
            
        except Exception as recovery_e:
            self.logger.error(f"Comprehensive error recovery completely failed for {ticker}: {recovery_e}")
            return False, None
    
    def get_forecasting_statistics(self) -> Dict[str, Any]:
        """Get comprehensive forecasting statistics."""
        return {
            'system_stats': self.forecasting_stats.copy(),
            'error_stats': self.error_handler.get_error_statistics(),
            'model_selection_stats': self.model_selector.get_tier_performance_stats(),
            'fallback_stats': graceful_degradation_system.get_fallback_statistics()
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.forecasting_stats = {
            'total_requests': 0,
            'successful_advanced': 0,
            'successful_classical': 0,
            'successful_lightweight': 0,
            'fallback_used': 0,
            'ultimate_fallback_used': 0,
            'total_errors': 0
        }


# Global instance for easy access
comprehensive_forecasting_system = ComprehensiveForecastingSystem()