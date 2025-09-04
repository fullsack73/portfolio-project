import logging
from logging import NullHandler
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.risk_models import CovarianceShrinkage
from prophet import Prophet
import warnings
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from scipy.stats import expon, linregress
from sklearn.linear_model import LinearRegression
from cache_manager import (
    get_cache, cached, cache_stock_data_key, 
    cache_forecast_key, cache_portfolio_key
)
from ticker_lists import get_ticker_group
from forecasting_models import (
    ARIMAForecaster, LSTMForecaster, SARIMAXForecaster,
    XGBoostForecaster, LightGBMForecaster, CatBoostForecaster,
    EnsembleForecaster, ForecastResult
)
from model_validator import ModelValidator
from forecasting_config import get_forecasting_config, get_enabled_models
from model_performance import record_model_performance, get_performance_report
from forecasting_error_handler import (
    ForecastingErrorHandler, DataQualityValidator, ErrorContext,
    ErrorCategory, ErrorSeverity, robust_forecasting_wrapper
)
from forecasting_fallback_system import graceful_degradation_system
from forecasting_integration import comprehensive_forecasting_system
import pickle
import hashlib

# Define a dummy function to suppress logging
# def silent_logger(*args, **kwargs):
#     pass

# Monkey-patch the logger to silence it
# cmdstanpy.utils.get_logger = silent_logger

# Configure logging for this module
logger = logging.getLogger(__name__)

# Re-enabled log suppression after debugging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
# Forcing prophet logs to be silent by adding a null handler
prophet_logger = logging.getLogger('prophet')
prophet_logger.addHandler(NullHandler())
prophet_logger.propagate = False

def sanitize_tickers(tickers):
    """Sanitize ticker symbols for yfinance compatibility with special case handling."""
    special_mappings = {
        'BRK.B': 'BRK-B',
        'BF.B': 'BF-B',
    }
    
    sanitized = []
    for ticker in tickers:
        if ticker in special_mappings:
            sanitized.append(special_mappings[ticker])
            logger.info(f"Applied special mapping: {ticker} -> {special_mappings[ticker]}")
        else:
            sanitized.append(ticker.replace('.', '-'))
    
    return sanitized

@cached(l1_ttl=900, l2_ttl=14400)  # 15 min L1, 4 hour L2 cache
def get_stock_data(tickers, start_date, end_date):
    """Fetch stock data for given tickers and date range using batch processing with aggressive caching."""
    logger.info(f"GET_STOCK_DATA: Starting BATCH fetch for {len(tickers)} tickers")
    
    # Sanitize tickers first
    original_tickers = tickers.copy()
    tickers = sanitize_tickers(tickers)
    sanitized_changes = set(original_tickers) - set(tickers)
    if sanitized_changes:
        logger.info(f"GET_STOCK_DATA: Sanitized {len(sanitized_changes)} ticker symbols")
    
    try:
        # Use batch download for better performance
        logger.info(f"GET_STOCK_DATA: Performing batch download for {len(tickers)} tickers")
        batch_data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        # Handle single ticker case (returns Series instead of DataFrame)
        if len(tickers) == 1:
            # For single ticker, yfinance returns a simple DataFrame
            if isinstance(batch_data.columns, pd.MultiIndex):
                # If somehow we get MultiIndex for single ticker
                close_data = batch_data['Close'] if 'Close' in batch_data.columns.get_level_values(0) else batch_data
            else:
                # Normal case for single ticker
                close_data = batch_data['Close'] if 'Close' in batch_data.columns else batch_data
            
            close_data.name = tickers[0]
            final_data = pd.DataFrame(close_data)
        else:
            # Extract Close prices for multiple tickers
            logger.info(f"GET_STOCK_DATA: Batch data columns structure: {type(batch_data.columns)}")
            logger.info(f"GET_STOCK_DATA: Batch data shape: {batch_data.shape}")
            
            if isinstance(batch_data.columns, pd.MultiIndex):
                # MultiIndex columns: ('Close', 'AAPL'), ('Close', 'MSFT'), etc.
                if 'Close' in batch_data.columns.get_level_values(0):
                    close_data = batch_data['Close']
                    logger.info(f"GET_STOCK_DATA: Extracted Close data shape: {close_data.shape}")
        
        # Clean the data
        close_data = close_data.ffill().dropna()
        logger.info(f"GET_STOCK_DATA: Data cleaned, final shape: {close_data.shape}")
        
        # Ensure we have data
        if close_data.empty:
            logger.error(f"GET_STOCK_DATA: No data returned after cleaning")
            return pd.DataFrame()
            
        final_data = close_data
        logger.info(f"GET_STOCK_DATA: BATCH fetch successful for {len(final_data.columns)} tickers")
        return final_data
        
    except Exception as e:
        logger.error(f"GET_STOCK_DATA: Batch fetch failed: {e}")
        logger.info(f"GET_STOCK_DATA: Falling back to individual ticker fetching")
        
        # Fallback to parallel individual ticker fetching
        individual_data = {}
        import os
        max_workers = min(os.cpu_count() or 4, len(tickers), 16) # Cap at 16 for I/O
        logger.info(f"GET_STOCK_DATA: Using {max_workers} parallel workers for individual fetch")

        def _fetch_single(ticker):
            try:
                ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if not ticker_data.empty and 'Close' in ticker_data.columns:
                    logger.info(f"GET_STOCK_DATA: Individual fetch successful for {ticker}")
                    return ticker, ticker_data['Close']
                else:
                    logger.warning(f"GET_STOCK_DATA: No data for {ticker}")
                    return ticker, None
            except Exception as ticker_error:
                logger.error(f"GET_STOCK_DATA: Failed to fetch {ticker}: {ticker_error}")
                return ticker, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(_fetch_single, ticker): ticker for ticker in tickers}
            for future in as_completed(future_to_ticker):
                ticker, data = future.result()
                if data is not None:
                    individual_data[ticker] = data
        
        if individual_data:
            final_data = pd.DataFrame(individual_data).ffill().dropna()
            logger.info(f"GET_STOCK_DATA: Individual fallback successful for {len(final_data.columns)} tickers")
            return final_data
        else:
            logger.error(f"GET_STOCK_DATA: All fetching methods failed")
            return pd.DataFrame()

def _exponential_smoothing_forecast(prices, alpha=0.3):
    """Fast exponential smoothing forecast."""
    if len(prices) < 2:
        return 0.05
    
    # Simple exponential smoothing
    smoothed = [prices[0]]
    for i in range(1, len(prices)):
        smoothed.append(alpha * prices[i] + (1 - alpha) * smoothed[i-1])
    
    # Calculate trend from last 30 days
    recent_data = smoothed[-30:] if len(smoothed) >= 30 else smoothed
    if len(recent_data) < 2:
        return 0.05
    
    # Linear trend extrapolation
    x = np.arange(len(recent_data))
    slope, intercept, _, _, _ = linregress(x, recent_data)
    
    # Project 252 days ahead (1 year)
    future_price = slope * (len(recent_data) + 252) + intercept
    current_price = recent_data[-1]
    
    if current_price <= 0:
        return 0.05
    
    return (future_price / current_price) - 1

def _linear_trend_forecast(prices):
    """Fast linear trend forecast."""
    if len(prices) < 10:
        return 0.05
    
    # Use last 90 days for trend analysis
    recent_prices = prices[-90:] if len(prices) >= 90 else prices
    x = np.arange(len(recent_prices)).reshape(-1, 1)
    y = recent_prices
    
    try:
        model = LinearRegression()
        model.fit(x, y)
        
        # Predict 252 days ahead
        future_x = np.array([[len(recent_prices) + 252]])
        future_price = model.predict(future_x)[0]
        current_price = recent_prices[-1]
        
        if current_price <= 0:
            return 0.05
        
        return (future_price / current_price) - 1
    except:
        return 0.05

def _historical_volatility_adjusted_forecast(prices):
    """Historical mean with volatility adjustment."""
    if len(prices) < 30:
        return 0.05
    
    returns = np.diff(prices) / prices[:-1]
    returns = returns[~np.isnan(returns)]  # Remove NaN values
    
    if len(returns) < 10:
        return 0.05
    
    mean_return = np.mean(returns)
    volatility = np.std(returns)
    
    # Annualized return with volatility adjustment
    annual_return = mean_return * 252
    
    # Apply volatility penalty for very volatile stocks
    if volatility > 0.05:  # 5% daily volatility threshold
        annual_return *= 0.8  # Reduce expected return for high volatility
    
    return annual_return

@cached(l1_ttl=1800, l2_ttl=7200)  # 30 min L1, 2 hour L2 cache for forecasts
def _forecast_single_ticker(ticker, ticker_data, use_lightweight=True):
    """Helper function to forecast returns for a single ticker with lightweight methods by default and caching."""
    try:
        prices = ticker_data.values
        
        # Skip if data is empty or insufficient
        if len(prices) < 10:
            logger.warning(f"Skipping forecast for {ticker}: insufficient data ({len(prices)} data points).")
            return ticker, 0.05
        
        # Check for too many missing values
        valid_prices = prices[~np.isnan(prices)]
        if len(valid_prices) < len(prices) * 0.5:
            logger.warning(f"Skipping forecast for {ticker}: too many missing values.")
            return ticker, 0.05
        
        if use_lightweight:
            # Use fast lightweight forecasting methods
            try:
                # Try multiple lightweight methods and use ensemble
                exp_forecast = _exponential_smoothing_forecast(valid_prices)
                trend_forecast = _linear_trend_forecast(valid_prices)
                vol_forecast = _historical_volatility_adjusted_forecast(valid_prices)
                
                # Ensemble: weighted average of the three methods
                forecast_value = (0.4 * exp_forecast + 0.3 * trend_forecast + 0.3 * vol_forecast)
                
                # Sanity check: cap extreme values
                forecast_value = np.clip(forecast_value, -0.5, 1.0)  # -50% to +100% annual return
                
                logger.info(f"LIGHTWEIGHT forecast for {ticker}: {forecast_value:.4f}")
                return ticker, forecast_value
                
            except Exception as lightweight_error:
                logger.warning(f"Lightweight forecasting failed for {ticker}: {lightweight_error}. Using Prophet fallback.")
                # Fall through to Prophet method
        
        # Prophet fallback (for high-priority tickers or when lightweight fails)
        logger.info(f"Using Prophet fallback for {ticker}")
        
        # Create DataFrame for Prophet
        df_prophet = pd.DataFrame({
            'ds': ticker_data.index,
            'y': valid_prices[:len(ticker_data.index)]  # Ensure same length
        })
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        
        try:
            model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
            model.fit(df_prophet)
            
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)
            
            if len(forecast) >= 365:
                expected_return = (forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[-365]) - 1
                logger.info(f"Prophet forecast for {ticker}: {expected_return:.4f}")
                return ticker, expected_return
            else:
                raise ValueError("Insufficient forecast data")
                
        except Exception as prophet_error:
            logger.warning(f"Prophet forecasting failed for {ticker}: {prophet_error}. Using simple mean.")
            # Final fallback to simple historical mean
            returns = np.diff(valid_prices) / valid_prices[:-1]
            returns = returns[~np.isnan(returns)]
            if len(returns) > 0:
                forecast_value = np.mean(returns) * 252  # Annualized
                return ticker, forecast_value
            else:
                return ticker, 0.05

    except Exception as e:
        logger.error(f"Critical error processing {ticker}: {e}")
        return ticker, 0.05


@robust_forecasting_wrapper
def robust_forecast_returns(tickers, data, method='advanced', max_workers=None, ticker_to_data=None):
    """
    Robust return forecasting with comprehensive error handling and fallback mechanisms.
    
    Args:
        tickers: List of ticker symbols
        data: Historical price data
        method: Forecasting method ('advanced', 'lightweight', 'fallback')
        max_workers: Maximum number of parallel workers
        ticker_to_data: Pre-computed ticker data mapping
        
    Returns:
        Dictionary of expected returns with error handling
    """
    logger.info(f"Starting robust forecast for {len(tickers)} tickers using {method} method")
    
    # Initialize error handler and fallback system
    error_handler = ForecastingErrorHandler()
    data_validator = DataQualityValidator()
    
    expected_returns = {}
    failed_tickers = []
    fallback_used = []
    
    # Get forecasting configuration
    try:
        config = get_forecasting_config()
        enabled_models = get_enabled_models()
    except Exception as e:
        logger.warning(f"Failed to load forecasting config, using defaults: {e}")
        config = {'default_method': 'arima', 'parallel_processing': True}
        enabled_models = ['arima']
    
    def _robust_single_ticker_forecast(ticker):
        """Forecast single ticker with comprehensive error handling."""
        try:
            # Get ticker data
            if ticker_to_data and ticker in ticker_to_data:
                ticker_data = ticker_to_data[ticker]
            elif ticker in data.columns:
                ticker_data = data[ticker].dropna()
            else:
                logger.warning(f"No data available for {ticker}")
                return ticker, None, 'no_data'
            
            # Validate data quality
            is_valid, issues = data_validator.validate_data(ticker_data, ticker, min_points=50)
            
            if not is_valid:
                logger.warning(f"Data quality issues for {ticker}: {'; '.join(issues)}")
                # Try to clean the data
                ticker_data = data_validator.clean_data(ticker_data, ticker)
                
                # Re-validate
                is_valid_after_cleaning, remaining_issues = data_validator.validate_data(
                    ticker_data, ticker, min_points=30  # Lower threshold after cleaning
                )
                
                if not is_valid_after_cleaning:
                    logger.error(f"Data unusable for {ticker} even after cleaning: {'; '.join(remaining_issues)}")
                    return ticker, None, 'data_quality_failed'
            
            # Try advanced forecasting if requested
            if method == 'advanced' and enabled_models:
                prediction, model_used = _try_advanced_forecasting(ticker, ticker_data, enabled_models)
                if prediction is not None:
                    # Cache successful prediction
                    graceful_degradation_system.cache_successful_prediction(
                        ticker, model_used, 1, prediction
                    )
                    return ticker, prediction, f'advanced_{model_used}'
            
            # Try lightweight methods
            if method in ['advanced', 'lightweight']:
                prediction, method_used = _try_lightweight_forecasting(ticker, ticker_data)
                if prediction is not None:
                    return ticker, prediction, f'lightweight_{method_used}'
            
            # Use fallback system
            prediction, fallback_method = graceful_degradation_system.get_fallback_prediction(
                ticker, ticker_data, periods=1
            )
            
            return ticker, prediction, f'fallback_{fallback_method}'
            
        except Exception as e:
            logger.error(f"Complete forecasting failure for {ticker}: {e}")
            
            # Create error context
            context = ErrorContext(
                ticker=ticker,
                model_name='robust_forecast',
                operation='forecast_returns',
                data_points=len(ticker_data) if 'ticker_data' in locals() else 0,
                timestamp=datetime.now(),
                additional_info={'method': method, 'error': str(e)}
            )
            
            # Handle the error
            recovery_successful, recovery_result = error_handler.handle_error(
                e, context, ErrorCategory.MODEL_PREDICTION, ErrorSeverity.HIGH
            )
            
            if recovery_successful and isinstance(recovery_result, (int, float)):
                return ticker, float(recovery_result), 'error_recovery'
            
            # Ultimate fallback
            return ticker, 0.05, 'ultimate_fallback'
    
    # Process tickers with parallel execution and error handling
    if max_workers is None:
        import os
        max_workers = min(os.cpu_count() or 4, len(tickers), 8)
    
    logger.info(f"Processing {len(tickers)} tickers with {max_workers} workers")
    
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(_robust_single_ticker_forecast, ticker): ticker 
                for ticker in tickers
            }
            
            # Collect results
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    ticker_result, prediction, method_used = future.result(timeout=60)  # 60 second timeout
                    
                    if prediction is not None:
                        expected_returns[ticker_result] = prediction
                        
                        if 'fallback' in method_used:
                            fallback_used.append(ticker_result)
                        
                        logger.debug(f"Forecast for {ticker_result}: {prediction:.6f} ({method_used})")
                    else:
                        failed_tickers.append(ticker_result)
                        expected_returns[ticker_result] = 0.05  # Conservative default
                        
                except Exception as e:
                    logger.error(f"Future execution failed for {ticker}: {e}")
                    failed_tickers.append(ticker)
                    expected_returns[ticker] = 0.05
                    
    except Exception as e:
        logger.error(f"Parallel processing failed, falling back to sequential: {e}")
        
        # Sequential fallback
        for ticker in tickers:
            try:
                ticker_result, prediction, method_used = _robust_single_ticker_forecast(ticker)
                expected_returns[ticker_result] = prediction if prediction is not None else 0.05
                
                if prediction is None:
                    failed_tickers.append(ticker_result)
                elif 'fallback' in method_used:
                    fallback_used.append(ticker_result)
                    
            except Exception as e:
                logger.error(f"Sequential processing failed for {ticker}: {e}")
                expected_returns[ticker] = 0.05
                failed_tickers.append(ticker)
    
    # Log summary statistics
    total_tickers = len(tickers)
    successful_forecasts = len([v for v in expected_returns.values() if v != 0.05])
    fallback_count = len(fallback_used)
    failed_count = len(failed_tickers)
    
    logger.info(f"Forecasting summary: {successful_forecasts}/{total_tickers} successful, "
               f"{fallback_count} used fallback, {failed_count} failed")
    
    if failed_tickers:
        logger.warning(f"Failed tickers: {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")
    
    return expected_returns


def _try_advanced_forecasting(ticker, data, enabled_models):
    """Try advanced forecasting models with error handling."""
    logger.debug(f"Trying advanced forecasting for {ticker}")
    
    # Model priority order
    model_priority = ['ensemble', 'lstm', 'sarimax', 'xgboost', 'lightgbm', 'catboost', 'arima']
    
    for model_name in model_priority:
        if model_name not in enabled_models:
            continue
            
        try:
            # Create and fit model
            if model_name == 'arima':
                model = ARIMAForecaster()
            elif model_name == 'lstm':
                model = LSTMForecaster()
            elif model_name == 'sarimax':
                model = SARIMAXForecaster()
            elif model_name == 'xgboost':
                model = XGBoostForecaster()
            elif model_name == 'lightgbm':
                model = LightGBMForecaster()
            elif model_name == 'catboost':
                model = CatBoostForecaster()
            elif model_name == 'ensemble':
                model = EnsembleForecaster()
            else:
                continue
            
            # Fit with timeout and error handling
            success = model._safe_fit(data, ticker=ticker)
            if not success:
                logger.warning(f"Model {model_name} fitting failed for {ticker}")
                continue
            
            # Generate prediction
            prediction = model._safe_predict(1, ticker=ticker)
            if prediction is not None:
                logger.debug(f"Advanced forecast for {ticker} using {model_name}: {prediction:.6f}")
                return prediction, model_name
                
        except Exception as e:
            logger.warning(f"Advanced model {model_name} failed for {ticker}: {e}")
            continue
    
    return None, None


def _try_lightweight_forecasting(ticker, data):
    """Try lightweight forecasting methods."""
    logger.debug(f"Trying lightweight forecasting for {ticker}")
    
    methods = [
        ('exponential_smoothing', _exponential_smoothing_forecast),
        ('linear_trend', _linear_trend_forecast),
    ]
    
    for method_name, method_func in methods:
        try:
            prediction = method_func(data.values)
            if prediction is not None and not np.isnan(prediction) and not np.isinf(prediction):
                logger.debug(f"Lightweight forecast for {ticker} using {method_name}: {prediction:.6f}")
                return prediction, method_name
        except Exception as e:
            logger.warning(f"Lightweight method {method_name} failed for {ticker}: {e}")
            continue
    
    return None, None


def _get_model_cache_key(ticker, model_name, data_hash):
    """Generate cache key for trained models."""
    return f"model_{model_name}_{ticker}_{data_hash}"


def _get_prediction_cache_key(ticker, model_name, data_hash, periods):
    """Generate cache key for predictions."""
    return f"prediction_{model_name}_{ticker}_{data_hash}_{periods}"


def _cache_trained_model(model, ticker, model_name, data_hash):
    """Cache a trained model."""
    try:
        cache = get_cache()
        cache_key = _get_model_cache_key(ticker, model_name, data_hash)
        
        # Serialize model
        model_data = pickle.dumps(model)
        cache.set(cache_key, model_data, ttl=3600)  # 1 hour TTL
        
        logger.debug(f"Cached trained model {model_name} for {ticker}")
    except Exception as e:
        logger.warning(f"Failed to cache model {model_name} for {ticker}: {e}")


def _load_cached_model(ticker, model_name, data_hash):
    """Load a cached trained model."""
    try:
        cache = get_cache()
        cache_key = _get_model_cache_key(ticker, model_name, data_hash)
        
        model_data = cache.get(cache_key)
        if model_data:
            model = pickle.loads(model_data)
            logger.debug(f"Loaded cached model {model_name} for {ticker}")
            return model
    except Exception as e:
        logger.debug(f"Failed to load cached model {model_name} for {ticker}: {e}")
    
    return None


def _cache_prediction(prediction, ticker, model_name, data_hash, periods):
    """Cache a model prediction."""
    try:
        cache = get_cache()
        cache_key = _get_prediction_cache_key(ticker, model_name, data_hash, periods)
        cache.set(cache_key, prediction, ttl=1800)  # 30 min TTL
        
        logger.debug(f"Cached prediction from {model_name} for {ticker}")
    except Exception as e:
        logger.warning(f"Failed to cache prediction from {model_name} for {ticker}: {e}")


def _load_cached_prediction(ticker, model_name, data_hash, periods):
    """Load a cached prediction."""
    try:
        cache = get_cache()
        cache_key = _get_prediction_cache_key(ticker, model_name, data_hash, periods)
        
        prediction = cache.get(cache_key)
        if prediction is not None:
            logger.debug(f"Loaded cached prediction from {model_name} for {ticker}")
            return prediction
    except Exception as e:
        logger.debug(f"Failed to load cached prediction from {model_name} for {ticker}: {e}")
    
    return None


@cached(l1_ttl=1800, l2_ttl=7200)  # 30 min L1, 2 hour L2 cache for advanced forecasts
def advanced_forecast_returns(data, use_ensemble=True, max_models_per_ticker=5, parallel_workers=None):
    """
    Advanced forecasting using ML models with automatic model selection and ensemble methods.
    
    Args:
        data: DataFrame with stock price data
        use_ensemble: Whether to use ensemble methods when multiple models succeed
        max_models_per_ticker: Maximum number of models to try per ticker
        parallel_workers: Number of parallel workers (None for auto-detection)
    
    Returns:
        pd.Series: Expected returns for each ticker
    """
    start_time = time.time()
    logger.info(f"Starting ADVANCED forecasting for {len(data.columns)} tickers")
    
    # Get configuration
    config = get_forecasting_config()
    enabled_models = get_enabled_models()[:max_models_per_ticker]
    
    # Determine optimal number of workers
    if parallel_workers is None:
        parallel_workers = min(config.performance.parallel_workers, len(data.columns), 8)
    
    logger.info(f"Using {parallel_workers} parallel workers with models: {enabled_models}")
    
    # Initialize model validator
    validator = ModelValidator()
    
    forecasts = {}
    model_usage_stats = {}
    
    # Use ProcessPoolExecutor for CPU-bound forecasting tasks
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        # Submit all forecasting tasks
        future_to_ticker = {}
        for ticker in data.columns:
            future_to_ticker[executor.submit(
                _advanced_forecast_single_ticker, 
                ticker, 
                data[ticker], 
                enabled_models,
                use_ensemble,
                config
            )] = ticker
        
        # Collect results as they complete
        completed_count = 0
        
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result_ticker, forecast_value, model_used, ensemble_info = future.result()
                forecasts[result_ticker] = forecast_value
                
                # Track model usage
                if model_used not in model_usage_stats:
                    model_usage_stats[model_used] = 0
                model_usage_stats[model_used] += 1
                
                completed_count += 1
                
                # Progress logging
                if completed_count % 25 == 0 or completed_count == len(data.columns):
                    logger.info(f"Advanced forecasting progress: {completed_count}/{len(data.columns)} completed")
                    
            except Exception as exc:
                logger.error(f"Advanced forecasting failed for {ticker}: {exc}")
                # Enhanced fallback using graceful degradation system
                try:
                    ticker_data = data[ticker].dropna()
                    fallback_forecast, fallback_method = graceful_degradation_system.get_fallback_prediction(
                        ticker, ticker_data, periods=1, failed_models=['advanced_models']
                    )
                    forecasts[ticker] = fallback_forecast
                    model_usage_stats[f'fallback_{fallback_method}'] = model_usage_stats.get(f'fallback_{fallback_method}', 0) + 1
                except Exception as fallback_exc:
                    logger.error(f"Graceful degradation also failed for {ticker}: {fallback_exc}")
                    # Try lightweight method as secondary fallback
                    try:
                        _, lightweight_forecast = _forecast_single_ticker(ticker, data[ticker], use_lightweight=True)
                        forecasts[ticker] = lightweight_forecast
                        model_usage_stats['lightweight_fallback'] = model_usage_stats.get('lightweight_fallback', 0) + 1
                    except Exception as lightweight_exc:
                        logger.error(f"All fallback methods failed for {ticker}: {lightweight_exc}")
                        # Enhanced ultimate fallback
                        try:
                            ticker_data = data[ticker].dropna() if ticker in data.columns else None
                            if ticker_data is not None and not ticker_data.empty and len(ticker_data) >= 5:
                                # Use recent performance as fallback
                                recent_return = (ticker_data.iloc[-1] / ticker_data.iloc[0]) ** (252 / len(ticker_data)) - 1
                                if abs(recent_return) < 1.0:  # Sanity check
                                    forecasts[ticker] = float(recent_return)
                                else:
                                    forecasts[ticker] = 0.05
                            else:
                                forecasts[ticker] = 0.05
                        except:
                            forecasts[ticker] = 0.05  # Final conservative fallback
                        model_usage_stats['ultimate_fallback'] = model_usage_stats.get('ultimate_fallback', 0) + 1
    
    elapsed_time = time.time() - start_time
    logger.info(f"ADVANCED forecasting completed in {elapsed_time:.2f} seconds for {len(forecasts)} tickers")
    logger.info(f"Model usage: {model_usage_stats}")
    
    return pd.Series(forecasts)


def _advanced_forecast_single_ticker(ticker, ticker_data, enabled_models, use_ensemble, config):
    """
    Helper function to forecast returns for a single ticker using advanced ML models.
    
    Args:
        ticker: Ticker symbol
        ticker_data: Price data for the ticker
        enabled_models: List of enabled model names
        use_ensemble: Whether to use ensemble methods
        config: Forecasting configuration
    
    Returns:
        Tuple of (ticker, forecast_value, model_used, ensemble_info)
    """
    try:
        # Validate data
        prices = ticker_data.dropna()
        if len(prices) < config.validation.min_data_points:
            logger.warning(f"Insufficient data for {ticker}: {len(prices)} points")
            raise ValueError(f"Insufficient data: {len(prices)} points")
        
        # Generate data hash for caching
        data_str = f"{len(prices)}_{prices.index[0]}_{prices.index[-1]}_{prices.sum():.6f}"
        data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
        
        # Check for cached ensemble prediction first
        if use_ensemble:
            cached_prediction = _load_cached_prediction(ticker, 'ensemble', data_hash, 252)
            if cached_prediction is not None:
                return ticker, cached_prediction, 'ensemble', {'cached': True}
        
        # Initialize models
        models = []
        model_instances = {}
        
        for model_name in enabled_models:
            try:
                model_config = config.models.get(model_name)
                if not model_config or not model_config.enabled:
                    continue
                
                # Check for cached prediction first
                cached_prediction = _load_cached_prediction(ticker, model_name, data_hash, 252)
                if cached_prediction is not None:
                    # Create a dummy model for ensemble use
                    if model_name == 'arima':
                        model = ARIMAForecaster(model_name=f"{model_name}_{ticker}")
                    elif model_name == 'lstm':
                        model = LSTMForecaster(model_name=f"{model_name}_{ticker}")
                    elif model_name == 'sarimax':
                        model = SARIMAXForecaster(model_name=f"{model_name}_{ticker}")
                    elif model_name == 'xgboost':
                        model = XGBoostForecaster(model_name=f"{model_name}_{ticker}")
                    elif model_name == 'lightgbm':
                        model = LightGBMForecaster(model_name=f"{model_name}_{ticker}")
                    elif model_name == 'catboost':
                        model = CatBoostForecaster(model_name=f"{model_name}_{ticker}")
                    else:
                        continue
                    
                    # Mark as fitted and store cached prediction
                    model.is_fitted = True
                    model._cached_prediction = cached_prediction
                    models.append(model)
                    model_instances[model_name] = model
                    continue
                
                # Try to load cached trained model
                cached_model = _load_cached_model(ticker, model_name, data_hash)
                if cached_model is not None:
                    models.append(cached_model)
                    model_instances[model_name] = cached_model
                    continue
                
                # Create and train new model
                if model_name == 'arima':
                    model = ARIMAForecaster(model_name=f"{model_name}_{ticker}")
                elif model_name == 'lstm':
                    model = LSTMForecaster(model_name=f"{model_name}_{ticker}")
                elif model_name == 'sarimax':
                    model = SARIMAXForecaster(model_name=f"{model_name}_{ticker}")
                elif model_name == 'xgboost':
                    model = XGBoostForecaster(model_name=f"{model_name}_{ticker}")
                elif model_name == 'lightgbm':
                    model = LightGBMForecaster(model_name=f"{model_name}_{ticker}")
                elif model_name == 'catboost':
                    model = CatBoostForecaster(model_name=f"{model_name}_{ticker}")
                else:
                    continue
                
                # Set model configuration
                if hasattr(model, 'config'):
                    model.config = model_config.params.copy()
                
                # Fit model with timeout
                fit_start = time.time()
                model.fit(prices)
                fit_time = time.time() - fit_start
                
                if fit_time > model_config.max_training_time:
                    logger.warning(f"Model {model_name} exceeded training time limit for {ticker}")
                    # Record failed training due to timeout
                    record_model_performance(
                        model_name=model_name,
                        ticker=ticker,
                        training_time=fit_time,
                        prediction_time=0.0,
                        success=False,
                        error_message="Training time exceeded limit"
                    )
                    continue
                
                # Cache the trained model
                _cache_trained_model(model, ticker, model_name, data_hash)
                
                models.append(model)
                model_instances[model_name] = model
                
            except Exception as e:
                logger.warning(f"Failed to initialize/fit {model_name} for {ticker}: {e}")
                # Record failed model initialization/training
                record_model_performance(
                    model_name=model_name,
                    ticker=ticker,
                    training_time=0.0,
                    prediction_time=0.0,
                    success=False,
                    error_message=str(e)
                )
                continue
        
        if not models:
            raise RuntimeError("No models successfully initialized")
        
        # Generate forecasts
        if len(models) == 1:
            # Single model
            model = models[0]
            model_name = model.model_name.split('_')[0]
            
            # Check if we have cached prediction
            if hasattr(model, '_cached_prediction'):
                forecast_value = model._cached_prediction
                prediction_time = 0.001  # Minimal time for cached prediction
            else:
                pred_start = time.time()
                forecast_value = model.predict(periods=252)  # 1 year forecast
                prediction_time = time.time() - pred_start
                _cache_prediction(forecast_value, ticker, model_name, data_hash, 252)
            
            # Record performance metrics
            record_model_performance(
                model_name=model_name,
                ticker=ticker,
                training_time=0.0,  # Training time already recorded during fit
                prediction_time=prediction_time,
                success=True
            )
            
            return ticker, forecast_value, model_name, {}
        
        elif use_ensemble and len(models) >= config.performance.ensemble_threshold:
            # Use ensemble
            try:
                ensemble = EnsembleForecaster(model_name=f"ensemble_{ticker}")
                
                # Add models to ensemble
                for model in models:
                    ensemble.add_model(model)
                
                # Fit ensemble
                ensemble.fit(prices)
                
                # Generate ensemble forecast
                pred_start = time.time()
                forecast_value = ensemble.predict(periods=252)
                prediction_time = time.time() - pred_start
                
                # Cache ensemble prediction
                _cache_prediction(forecast_value, ticker, 'ensemble', data_hash, 252)
                
                # Record ensemble performance
                record_model_performance(
                    model_name='ensemble',
                    ticker=ticker,
                    training_time=0.0,  # Ensemble fitting time is minimal
                    prediction_time=prediction_time,
                    success=True
                )
                
                ensemble_info = {
                    'models': [m.model_name.split('_')[0] for m in models],
                    'weights': ensemble._model_weights,
                    'method': ensemble.config.get('combination_method', 'performance_weighted')
                }
                
                return ticker, forecast_value, 'ensemble', ensemble_info
                
            except Exception as e:
                logger.warning(f"Ensemble failed for {ticker}: {e}")
                # Record ensemble failure
                record_model_performance(
                    model_name='ensemble',
                    ticker=ticker,
                    training_time=0.0,
                    prediction_time=0.0,
                    success=False,
                    error_message=str(e)
                )
                # Fall back to best single model
        
        # Select best model using validator
        try:
            validator = ModelValidator()
            ranking = validator.select_best_models(models, prices, ticker)
            
            best_model_name = ranking.best_model.split('_')[0]
            best_model = model_instances[best_model_name]
            
            # Check if we have cached prediction
            if hasattr(best_model, '_cached_prediction'):
                forecast_value = best_model._cached_prediction
                prediction_time = 0.001  # Minimal time for cached prediction
            else:
                pred_start = time.time()
                forecast_value = best_model.predict(periods=252)
                prediction_time = time.time() - pred_start
                _cache_prediction(forecast_value, ticker, best_model_name, data_hash, 252)
            
            # Record performance metrics
            record_model_performance(
                model_name=best_model_name,
                ticker=ticker,
                training_time=0.0,  # Training time already recorded
                prediction_time=prediction_time,
                success=True,
                accuracy_score=ranking.best_score if hasattr(ranking, 'best_score') else None
            )
            
            return ticker, forecast_value, best_model_name, {'ranking_score': ranking.best_score}
            
        except Exception as e:
            logger.warning(f"Model selection failed for {ticker}: {e}")
            # Use first available model
            model = models[0]
            model_name = model.model_name.split('_')[0]
            
            # Check if we have cached prediction
            if hasattr(model, '_cached_prediction'):
                forecast_value = model._cached_prediction
                prediction_time = 0.001  # Minimal time for cached prediction
            else:
                pred_start = time.time()
                forecast_value = model.predict(periods=252)
                prediction_time = time.time() - pred_start
                _cache_prediction(forecast_value, ticker, model_name, data_hash, 252)
            
            # Record performance metrics
            record_model_performance(
                model_name=model_name,
                ticker=ticker,
                training_time=0.0,  # Training time already recorded
                prediction_time=prediction_time,
                success=True
            )
            
            return ticker, forecast_value, model_name, {}
    
    except Exception as e:
        logger.error(f"Advanced forecasting failed for {ticker}: {e}")
        raise e


def forecast_returns(data, use_lightweight=True, prophet_ratio=0.1, use_robust_system=False):
    """
    Forecast expected returns using lightweight methods by default with optional robust system.
    
    Args:
        data: Historical price data
        use_lightweight: Whether to use lightweight forecasting methods
        prophet_ratio: Ratio of tickers to use Prophet for (when not using robust system)
        use_robust_system: Whether to use the robust forecasting system with error handling
    """
    start_time = time.time()
    
    # Use robust system if requested
    if use_robust_system:
        logger.info("Using comprehensive forecasting system with graceful degradation")
        tickers = list(data.columns)
        method = 'lightweight' if use_lightweight else 'auto'
        
        try:
            expected_returns = comprehensive_forecasting_system.forecast_returns(
                tickers=tickers,
                data=data,
                periods=1,
                method=method,
                max_workers=None
            )
            
            total_time = time.time() - start_time
            logger.info(f"Comprehensive forecasting completed in {total_time:.2f}s for {len(tickers)} tickers")
            
            return expected_returns
            
        except Exception as e:
            logger.error(f"Comprehensive forecasting system failed, falling back to lightweight: {e}")
            # Fall through to original lightweight system
    
    # Original lightweight forecasting system
    method = "LIGHTWEIGHT" if use_lightweight else "PROPHET"
    logger.info(f"Starting PARALLEL {method} forecasting for {len(data.columns)} tickers")
    
    # Determine optimal number of workers
    import os
    max_workers = min(os.cpu_count() or 4, len(data.columns), 8)
    logger.info(f"Using {max_workers} parallel workers for {method} forecasting")
    
    # For large portfolios, use Prophet only for a subset of tickers (highest volume/market cap)
    prophet_tickers = set()
    if use_lightweight and len(data.columns) > 50:
        # Use Prophet for a small percentage of tickers (e.g., 10%)
        num_prophet = max(1, int(len(data.columns) * prophet_ratio))
        # For now, randomly select tickers for Prophet (in production, would use market cap/volume)
        import random
        prophet_tickers = set(random.sample(list(data.columns), num_prophet))
        logger.info(f"Using Prophet for {len(prophet_tickers)} high-priority tickers, lightweight for the rest")
    
    forecasts = {}
    
    # Use ProcessPoolExecutor for CPU-bound forecasting tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all forecasting tasks
        future_to_ticker = {}
        for ticker in data.columns:
            # Use Prophet for selected high-priority tickers, lightweight for others
            use_lightweight_for_ticker = use_lightweight and (ticker not in prophet_tickers)
            future_to_ticker[executor.submit(_forecast_single_ticker, ticker, data[ticker], use_lightweight_for_ticker)] = ticker
        
        # Collect results as they complete
        completed_count = 0
        lightweight_count = 0
        prophet_count = 0
        
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result_ticker, forecast_value = future.result()
                forecasts[result_ticker] = forecast_value
                completed_count += 1
                
                # Track method usage
                if use_lightweight and ticker not in prophet_tickers:
                    lightweight_count += 1
                else:
                    prophet_count += 1
                
                # Progress logging
                if completed_count % 25 == 0 or completed_count == len(data.columns):
                    logger.info(f"Forecasting progress: {completed_count}/{len(data.columns)} completed (Lightweight: {lightweight_count}, Prophet: {prophet_count})")
                    
            except Exception as exc:
                logger.error(f"Forecasting generated an exception for {ticker}: {exc}")
                forecasts[ticker] = 0.05  # Default fallback
    
    elapsed_time = time.time() - start_time
    logger.info(f"PARALLEL {method} forecasting completed in {elapsed_time:.2f} seconds for {len(forecasts)} tickers")
    logger.info(f"Method breakdown: {lightweight_count} lightweight, {prophet_count} Prophet")
    
    return pd.Series(forecasts)

@cached(l1_ttl=600, l2_ttl=3600)  # 10 min L1, 1 hour L2 cache for portfolio optimization
def optimize_portfolio(start_date, end_date, risk_free_rate, ticker_group=None, tickers=None, target_return=None, risk_tolerance=None, 
                      use_advanced_forecasting=True, forecasting_method='auto', max_models_per_ticker=5):
    """
    Optimize portfolio based on user preferences using PyPortfolioOpt with advanced forecasting.
    
    Args:
        start_date: Start date for historical data
        end_date: End date for historical data
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        ticker_group: Predefined group of tickers to use
        tickers: Custom list of tickers (overrides ticker_group)
        target_return: Target return for optimization (optional)
        risk_tolerance: Risk tolerance level (optional)
        use_advanced_forecasting: Whether to use advanced ML models (default: True)
        forecasting_method: 'advanced', 'lightweight', or 'auto' (default: 'auto')
        max_models_per_ticker: Maximum number of models to try per ticker (default: 5)
    
    Returns:
        Dictionary containing optimization results
    """
    # Log cache performance at start of optimization
    cache = get_cache()
    cache_stats = cache.stats()
    logger.info(f"CACHE PERFORMANCE: L1 Hit: {cache_stats['hit_ratios']['l1']:.1%}, L2 Hit: {cache_stats['hit_ratios']['l2']:.1%}, Overall: {cache_stats['hit_ratios']['overall']:.1%}")
    logger.info(f"CACHE MEMORY: {cache_stats['l1_cache']['memory_usage_mb']:.1f}MB / {cache_stats['l1_cache']['memory_limit_mb']:.1f}MB ({cache_stats['l1_cache']['memory_utilization']:.1%})")
    
    # Get tickers from the selected group or use the provided list
    if tickers:
        pass
    elif ticker_group:
        tickers = get_ticker_group(ticker_group)
    else:
        raise ValueError("Either ticker_group or tickers must be provided.")

    # Fetch data with comprehensive logging
    logger.info(f"PIPELINE STAGE 1: Attempting to fetch data for {len(tickers)} tickers")
    logger.info(f"Initial ticker list: {tickers[:10]}{'...' if len(tickers) > 10 else ''}")
    
    data = get_stock_data(tickers, start_date, end_date)
    print("DEBUG: Calling get_stock_data")
    logger.info(f"PIPELINE STAGE 1 RESULT: Fetched data shape: {data.shape}")
    logger.info(f"Fetched data columns: {list(data.columns)[:10]}{'...' if len(data.columns) > 10 else ''}")

    # Check if data is empty after fetching
    if data.empty:
        logger.warning("Could not fetch any valid data for the given tickers and date range. Aborting optimization.")
        return {
            "error": "Could not fetch any valid data for the given tickers and date range."
        }
    
    # Data cleaning stage
    logger.info(f"PIPELINE STAGE 2: Data cleaning - before dropna: {len(data.columns)} columns")
    data = data.dropna(axis=1, how='all')
    logger.info(f"PIPELINE STAGE 2 RESULT: After dropna: {len(data.columns)} columns")
    
    dropped_tickers = set(tickers) - set(data.columns)
    if dropped_tickers:
        logger.warning(f"DROPPED DURING DATA CLEANING: {len(dropped_tickers)} tickers: {list(dropped_tickers)[:10]}{'...' if len(dropped_tickers) > 10 else ''}")
    
    tickers = data.columns.tolist()
    logger.info(f"PIPELINE STAGE 2 FINAL: Proceeding with {len(tickers)} tickers for forecasting")

    # Calculate expected returns using selected forecasting method
    if use_advanced_forecasting and forecasting_method in ['advanced', 'auto']:
        try:
            # Determine if we should use advanced forecasting
            use_advanced = True
            if forecasting_method == 'auto':
                # Use advanced forecasting for smaller portfolios or when explicitly requested
                if len(data.columns) > 100:
                    logger.info(f"Large portfolio ({len(data.columns)} tickers), using hybrid approach")
                    use_advanced = False
                else:
                    logger.info(f"Medium portfolio ({len(data.columns)} tickers), using advanced forecasting")
            
            if use_advanced:
                logger.info(f"Starting ADVANCED forecasting for {len(data.columns)} tickers")
                mu = advanced_forecast_returns(
                    data, 
                    use_ensemble=True, 
                    max_models_per_ticker=max_models_per_ticker,
                    parallel_workers=None
                )
                logger.info(f"Advanced forecasting completed. Got forecasts for {len(mu)} tickers")
            else:
                # Hybrid approach: advanced for top tickers, lightweight for others
                logger.info(f"Using HYBRID forecasting approach")
                # Sort tickers by volume/market cap proxy (using data variance as proxy)
                ticker_importance = data.var().sort_values(ascending=False)
                top_tickers = ticker_importance.head(min(50, len(data.columns) // 2)).index.tolist()
                
                # Advanced forecasting for important tickers
                if top_tickers:
                    logger.info(f"Advanced forecasting for {len(top_tickers)} top tickers")
                    mu_advanced = advanced_forecast_returns(
                        data[top_tickers], 
                        use_ensemble=True, 
                        max_models_per_ticker=max_models_per_ticker,
                        parallel_workers=None
                    )
                else:
                    mu_advanced = pd.Series(dtype=float)
                
                # Lightweight forecasting for remaining tickers
                remaining_tickers = [t for t in data.columns if t not in top_tickers]
                if remaining_tickers:
                    logger.info(f"Lightweight forecasting for {len(remaining_tickers)} remaining tickers")
                    mu_lightweight = forecast_returns(
                        data[remaining_tickers], 
                        use_lightweight=True, 
                        prophet_ratio=0.1
                    )
                else:
                    mu_lightweight = pd.Series(dtype=float)
                
                # Combine results
                mu = pd.concat([mu_advanced, mu_lightweight])
                logger.info(f"Hybrid forecasting completed. Advanced: {len(mu_advanced)}, Lightweight: {len(mu_lightweight)}")
                
        except Exception as e:
            logger.warning(f"Advanced forecasting failed: {e}. Falling back to lightweight forecasting.")
            mu = forecast_returns(data, use_lightweight=True, prophet_ratio=0.1)
            logger.info(f"Fallback lightweight forecasting completed for {len(mu)} tickers")
    else:
        # Use lightweight forecasting
        logger.info(f"Starting LIGHTWEIGHT forecasting for {len(data.columns)} tickers")
        mu = forecast_returns(data, use_lightweight=True, prophet_ratio=0.1)
        logger.info(f"Lightweight forecasting completed. Got forecasts for {len(mu)} tickers")
    
    # Check for any missing tickers
    missing_tickers = set(data.columns) - set(mu.index)
    if missing_tickers:
        logger.warning(f"Missing forecasts for {len(missing_tickers)} tickers: {list(missing_tickers)}")
    
    # Filter the historical data to align with the tickers that have a forecast
    aligned_data = data[mu.index]
    logger.info(f"Aligned data contains {len(aligned_data.columns)} tickers for optimization")

    # Calculate covariance matrix on the aligned data
    S = risk_models.CovarianceShrinkage(aligned_data).ledoit_wolf()

    # Initialize Efficient Frontier
    ef = EfficientFrontier(mu, S)

    # Set optimization objective
    if target_return:
        ef.efficient_return(target_return)
    elif risk_tolerance:
        ef.efficient_risk(risk_tolerance)
    else:
        ef.max_sharpe()

    # Get optimized weights
    weights = ef.clean_weights()
    
    # Filter out assets with near-zero weight
    final_weights = {ticker: weight for ticker, weight in weights.items() if weight > 1e-4}

    # Get performance metrics
    performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    optimized_return = performance[0]
    optimized_std_dev = performance[1]
    optimized_sharpe_ratio = performance[2]

    # Get the latest prices for the tickers in the final portfolio
    latest_prices = {}
    if final_weights:
        final_tickers = list(final_weights.keys())
        logger.info(f"Fetching latest prices for {len(final_tickers)} final tickers.")

        for ticker in final_tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                # Fetching history for the last 2 days to get the most recent closing price
                hist = ticker_obj.history(period="2d", auto_adjust=True)
                if not hist.empty and 'Close' in hist.columns:
                    latest_prices[ticker] = hist['Close'].iloc[-1]
                    logger.info(f"Successfully fetched latest price for {ticker}: {latest_prices[ticker]:.2f}")
                else:
                    logger.warning(f"Could not retrieve latest price for {ticker}. It might be delisted or data is unavailable.")
            except Exception as e:
                logger.error(f"An error occurred while fetching the latest price for {ticker}: {e}")


    # Get performance report if advanced forecasting was used
    performance_report = None
    if use_advanced_forecasting:
        try:
            performance_report = get_performance_report()
            logger.info(f"System health score: {performance_report.system_health_score:.1f}%")
            if performance_report.recommendations:
                logger.info(f"Performance recommendations: {performance_report.recommendations}")
        except Exception as e:
            logger.warning(f"Failed to generate performance report: {e}")

    result = {
        "weights": final_weights,
        "return": optimized_return,
        "risk": optimized_std_dev,
        "sharpe_ratio": optimized_sharpe_ratio,
        "prices": latest_prices
    }
    
    # Add performance metrics if available
    if performance_report:
        result["performance_metrics"] = {
            "system_health_score": performance_report.system_health_score,
            "total_forecasts": performance_report.total_forecasts,
            "successful_forecasts": performance_report.successful_forecasts,
            "model_usage_stats": {name: {
                "success_rate": stats.success_rate,
                "average_training_time": stats.average_training_time,
                "average_prediction_time": stats.average_prediction_time,
                "total_calls": stats.total_calls
            } for name, stats in performance_report.model_usage_stats.items()},
            "top_performing_models": performance_report.top_performing_models[:3],
            "recommendations": performance_report.recommendations
        }
    
    return result


def get_forecasting_performance_dashboard():
    """
    Get a comprehensive forecasting performance dashboard.
    
    Returns:
        Dictionary containing performance dashboard data
    """
    try:
        from model_performance import get_performance_monitor
        
        monitor = get_performance_monitor()
        report = monitor.generate_performance_report()
        
        # Get recent alerts
        recent_alerts = monitor.get_recent_alerts(hours=24)
        
        # Calculate additional metrics
        dashboard = {
            "system_overview": {
                "health_score": report.system_health_score,
                "total_forecasts": report.total_forecasts,
                "success_rate": (report.successful_forecasts / report.total_forecasts * 100) if report.total_forecasts > 0 else 0,
                "average_processing_time": report.average_processing_time,
                "uptime_hours": (datetime.now() - monitor.start_time).total_seconds() / 3600
            },
            "model_performance": {
                name: {
                    "success_rate": stats.success_rate,
                    "total_calls": stats.total_calls,
                    "avg_training_time": stats.average_training_time,
                    "avg_prediction_time": stats.average_prediction_time,
                    "avg_accuracy": stats.average_accuracy,
                    "last_used": stats.last_used.isoformat() if stats.last_used else None
                }
                for name, stats in report.model_usage_stats.items()
            },
            "top_models": report.top_performing_models,
            "recent_alerts": recent_alerts,
            "recommendations": report.recommendations,
            "cache_performance": {
                "enabled": True,  # Assuming caching is enabled
                "hit_rate": "N/A"  # Would need to integrate with cache manager
            }
        }
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Failed to generate performance dashboard: {e}")
        return {
            "error": f"Failed to generate dashboard: {str(e)}",
            "system_overview": {"health_score": 0, "status": "error"}
        }
