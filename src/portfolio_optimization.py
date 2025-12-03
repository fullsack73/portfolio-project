import json
import logging
from logging import NullHandler
from pathlib import Path
import time

import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.risk_models import CovarianceShrinkage
from prophet import Prophet
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from scipy.stats import expon, linregress
from sklearn.linear_model import LinearRegression
import gc
from cache_manager import (
    get_cache, cached, cache_stock_data_key, 
    cache_forecast_key, cache_portfolio_key
)
from ticker_lists import get_ticker_group
from forecast_models import ModelSelector, ARIMA, LSTMModel, XGBoostModel

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Re-enabled log suppression after debugging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
# Forcing prophet logs to be silent by adding a null handler
prophet_logger = logging.getLogger('prophet')
prophet_logger.addHandler(NullHandler())
prophet_logger.propagate = False

RESULTS_DIR = Path("logs/portfolio_results")


def _ensure_results_dir():
    """Create persistence directory if it does not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _to_serializable(value):
    """Convert numpy/pandas objects to JSON-serializable primitives."""
    if isinstance(value, (np.generic, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (pd.Series, pd.Index)):
        return [_to_serializable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return value


def save_portfolio_result(portfolio_id, result, metadata=None):
    """Persist portfolio optimization output and metadata to disk."""
    if not portfolio_id:
        raise ValueError("portfolio_id is required to save results")
    if not isinstance(result, dict):
        raise ValueError("result must be a dictionary")

    payload = {
        "portfolio_id": portfolio_id,
        "result": _to_serializable(result),
        "metadata": _to_serializable(metadata or {}),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    _ensure_results_dir()
    output_path = RESULTS_DIR / f"{portfolio_id}.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    logger.info(f"Saved portfolio result to {output_path}")


def load_portfolio_result(portfolio_id):
    """Load a previously saved portfolio optimization result."""
    if not portfolio_id:
        raise ValueError("portfolio_id is required to load results")

    output_path = RESULTS_DIR / f"{portfolio_id}.json"
    if not output_path.exists():
        logger.info(f"No saved portfolio result found for {portfolio_id}")
        return None

    with open(output_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    result = payload.get("result", {})
    result["metadata"] = payload.get("metadata", {})
    result["saved_at"] = payload.get("saved_at")
    result["portfolio_id"] = payload.get("portfolio_id", portfolio_id)
    logger.info(f"Loaded portfolio result from {output_path}")
    return result


def list_saved_portfolios():
    """Return available saved portfolio identifiers."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(p.stem for p in RESULTS_DIR.glob("*.json"))

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
        # Only drop rows that are entirely NaN across all tickers; keep partial data for per-ticker cleaning later
        close_data = close_data.ffill().dropna(how='all')
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

def forecast_returns(data, use_lightweight=True, prophet_ratio=0.1):
    """Forecast expected returns using lightweight methods by default with optional Prophet for high-priority tickers."""
    start_time = time.time()
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

# Rename existing forecast_returns to fallback_forecast_returns for fallback
fallback_forecast_returns = forecast_returns

# NOTE: 모델 객체를 캐시하지 않음 - 메모리 누수 방지를 위해 forecast 결과만 캐시
def _train_and_select_model(ticker, ticker_data):
    """Train ML models and select best performer for a single ticker.
    
    WARNING: 모델 객체는 캐시하지 않습니다. 메모리 누수를 방지하기 위해
    forecast 결과만 별도로 캐시됩니다.
    """
    try:
        prices = ticker_data.values
        
        # Validate data
        if len(prices) < 100:
            logger.warning(f"Insufficient data for ML training on {ticker}: {len(prices)} points")
            return None, None
        
        valid_prices = prices[~np.isnan(prices)]
        if len(valid_prices) < 100:
            logger.warning(f"Too many NaN values for {ticker}")
            return None, None
        
        # Split into train/validation
        train_size = int(len(valid_prices) * 0.8)
        train_data = valid_prices[:train_size]
        val_data = valid_prices[train_size:]
        
        # Use ModelSelector to find best model
        start_time = time.time()
        selector = ModelSelector()
        best_model, metrics = selector.select_best_model(train_data, val_data)
        elapsed = time.time() - start_time
        
        logger.info(f"ML Model Selection for {ticker}: {metrics['model_name']} "
                   f"(R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.6f}) "
                   f"in {elapsed:.2f}s")
        
        return best_model, metrics
        
    except Exception as e:
        logger.error(f"ML model training failed for {ticker}: {e}")
        return None, None
    finally:
        # 명시적 메모리 해제
        gc.collect()

@cached(l1_ttl=900, l2_ttl=14400)  # 15 min L1, 4 hour L2 cache for predictions
def _ml_forecast_single_ticker(ticker, ticker_data, use_lightweight=False):
    """Forecast returns for single ticker using ML models with caching.
    
    NOTE: forecast 결과값만 캐시됩니다. 모델 객체는 사용 후 즉시 해제됩니다.
    """
    try:
        prices = ticker_data.values
        
        # Validate data
        if len(prices) < 10:
            logger.warning(f"Insufficient data for {ticker}: {len(prices)} points")
            return ticker, 0.08
        
        # For lightweight mode, use fast forecasting
        if use_lightweight:
            valid_prices = prices[~np.isnan(prices)]
            
            # Use ensemble of lightweight methods
            exp_forecast = _exponential_smoothing_forecast(valid_prices)
            trend_forecast = _linear_trend_forecast(valid_prices)
            vol_forecast = _historical_volatility_adjusted_forecast(valid_prices)
            
            forecast_value = (0.4 * exp_forecast + 0.3 * trend_forecast + 0.3 * vol_forecast)
            forecast_value = np.clip(forecast_value, -0.5, 1.0)
            
            return ticker, forecast_value
        
        # ML mode: train and select best model
        best_model, metrics = _train_and_select_model(ticker, ticker_data)
        
        if best_model is None:
            # No fallback in ML-only mode; return default small prior
            logger.warning(f"ML training failed for {ticker}, no fallback (ML-only mode)")
            return ticker, 0.02
        
        # Get forecast from best model
        try:
            if isinstance(best_model, ARIMA):
                expected_return, volatility = best_model.forecast(prices[~np.isnan(prices)])
                logger.info(f"ARIMA forecast for {ticker}: return={expected_return:.4f}, vol={volatility:.4f}")
                return ticker, expected_return
            elif isinstance(best_model, (LSTMModel, XGBoostModel)):
                forecast = best_model.forecast()
                logger.info(f"{metrics['model_name']} forecast for {ticker}: {forecast:.4f}")
                return ticker, forecast
            else:
                # Fallback
                logger.warning(f"Unknown model type for {ticker}, using default")
                return ticker, 0.08
        finally:
            # 모델 객체 명시적 해제 - 메모리 누수 방지
            del best_model
            gc.collect()
            
    except Exception as e:
        logger.error(f"ML forecasting failed for {ticker}: {e}")
        return ticker, 0.02

def ml_forecast_returns(data, use_lightweight=False, batch_size=20):
    """
    Forecast expected returns using ML models with memory-efficient batch processing.
    
    Args:
        data: DataFrame with stock prices (dates as index, tickers as columns)
        use_lightweight: If True, use fast ensemble methods; if False, use full ML models
        batch_size: Number of tickers to process in each batch (메모리 관리용)
        
    Returns:
        pandas Series with expected annual returns for each ticker
    """
    start_time = time.time()
    method = "ML" if not use_lightweight else "LIGHTWEIGHT"
    logger.info(f"Starting BATCH {method} forecasting for {len(data.columns)} tickers")
    
    # 메모리 효율을 위해 worker 수 제한 (LSTM/TensorFlow가 프로세스당 많은 메모리 사용)
    # ThreadPoolExecutor 사용 - ProcessPoolExecutor는 TensorFlow와 함께 사용 시 
    # 각 프로세스마다 TF가 로드되어 메모리 폭발
    import os
    max_workers = min(os.cpu_count() or 4, len(data.columns), 4)  # 최대 4로 제한
    logger.info(f"Using {max_workers} parallel workers for {method} forecasting (memory-safe mode)")
    
    forecasts = {}
    tickers = list(data.columns)
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    try:
        # 배치 단위로 처리하여 메모리 관리
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(tickers))
            batch_tickers = tickers[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_tickers)} tickers)")
            
            # ThreadPoolExecutor 사용 - TensorFlow는 메인 프로세스에서만 로드됨
            # GIL 때문에 완전한 병렬화는 안되지만, I/O 대기 시간 활용 가능
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {}
                for ticker in batch_tickers:
                    future = executor.submit(_ml_forecast_single_ticker, ticker, data[ticker], use_lightweight)
                    future_to_ticker[future] = ticker
                
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result_ticker, forecast_value = future.result()
                        forecasts[result_ticker] = forecast_value
                    except Exception as exc:
                        logger.error(f"{method} forecasting exception for {ticker}: {exc}")
                        forecasts[ticker] = 0.08
            
            # 배치 완료 후 메모리 정리
            gc.collect()
            
            # 진행 상황 로깅
            completed = len(forecasts)
            logger.info(f"Batch {batch_idx + 1} complete. Total progress: {completed}/{len(tickers)} ({100*completed/len(tickers):.1f}%)")
            
            # 메모리 상태 체크
            import psutil
            mem = psutil.virtual_memory()
            logger.info(f"Memory usage: {mem.percent:.1f}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)")
            
            # 메모리 사용량이 85% 이상이면 경고 및 추가 정리
            if mem.percent > 85:
                logger.warning(f"High memory usage detected ({mem.percent:.1f}%). Forcing garbage collection.")
                gc.collect()
                # Keras/TensorFlow 세션 정리 시도
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except Exception:
                    pass
        
        elapsed_time = time.time() - start_time
        logger.info(f"BATCH {method} forecasting completed in {elapsed_time:.2f}s for {len(forecasts)} tickers")
        
        # Log cache performance
        cache = get_cache()
        cache_stats = cache.stats()
        logger.info(f"CACHE HIT RATES: L1={cache_stats['hit_ratios']['l1']:.1%}, "
                   f"L2={cache_stats['hit_ratios']['l2']:.1%}, "
                   f"Overall={cache_stats['hit_ratios']['overall']:.1%}")
        
        return pd.Series(forecasts)
        
    except Exception as e:
        logger.error(f"ML forecasting failed critically: {e}. Falling back to lightweight methods.")
        # Fallback to original forecast_returns
        return fallback_forecast_returns(data, use_lightweight=True, prophet_ratio=0.0)

@cached(l1_ttl=600, l2_ttl=3600)  # 10 min L1, 1 hour L2 cache for portfolio optimization
def optimize_portfolio(start_date, end_date, risk_free_rate, ticker_group=None, tickers=None,
                       target_return=None, risk_tolerance=None, portfolio_id=None,
                       persist_result=False, load_if_available=False):
    """Optimize portfolio and optionally persist or reuse saved results."""
    # Log cache performance at start of optimization
    cache = get_cache()
    cache_stats = cache.stats()
    logger.info(f"CACHE PERFORMANCE: L1 Hit: {cache_stats['hit_ratios']['l1']:.1%}, L2 Hit: {cache_stats['hit_ratios']['l2']:.1%}, Overall: {cache_stats['hit_ratios']['overall']:.1%}")
    logger.info(f"CACHE MEMORY: {cache_stats['l1_cache']['memory_usage_mb']:.1f}MB / {cache_stats['l1_cache']['memory_limit_mb']:.1f}MB ({cache_stats['l1_cache']['memory_utilization']:.1%})")
    
    # Short-circuit if saved result should be reused
    if portfolio_id and load_if_available:
        saved_result = load_portfolio_result(portfolio_id)
        if saved_result:
            logger.info(f"Returning previously saved result for {portfolio_id}")
            return saved_result

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

    # Calculate expected returns using ML-based forecasting with fallback to lightweight
    logger.info(f"Starting ML forecasting for {len(data.columns)} tickers: {list(data.columns)}")
    # Strict ML-only forecasting
    mu = ml_forecast_returns(data, use_lightweight=False)
    logger.info(f"ML forecasting completed. Got forecasts for {len(mu)} tickers: {list(mu.index)}")
    
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


    result_payload = {
        "weights": final_weights,
        "return": optimized_return,
        "risk": optimized_std_dev,
        "sharpe_ratio": optimized_sharpe_ratio,
        "prices": latest_prices
    }

    if portfolio_id and persist_result:
        metadata = {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "risk_free_rate": risk_free_rate,
            "ticker_group": ticker_group,
            "tickers": tickers,
            "target_return": target_return,
            "risk_tolerance": risk_tolerance
        }
        save_portfolio_result(portfolio_id, result_payload, metadata)
        result_payload["portfolio_id"] = portfolio_id
    elif persist_result and not portfolio_id:
        logger.warning("persist_result is True but portfolio_id is missing; skipping save")

    return result_payload
