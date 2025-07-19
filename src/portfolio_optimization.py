import logging
from logging import NullHandler
import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models
from prophet import Prophet
from ticker_lists import get_ticker_group
import cmdstanpy
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
    # Special mappings for problematic tickers
    special_mappings = {
        'BRK.B': 'BRK-B',
        'BF.B': 'BF-B',
        # Add more special cases as needed
    }
    
    sanitized = []
    for ticker in tickers:
        if ticker in special_mappings:
            sanitized.append(special_mappings[ticker])
            logger.info(f"Applied special mapping: {ticker} -> {special_mappings[ticker]}")
        else:
            sanitized.append(ticker.replace('.', '-'))
    
    return sanitized

def get_stock_data(tickers, start_date, end_date):
    """Fetch stock data for given tickers and date range using batch processing."""
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
                else:
                    # Fallback: assume the entire DataFrame is price data
                    close_data = batch_data
                    logger.warning("GET_STOCK_DATA: No Close column found in MultiIndex, using entire DataFrame")
            else:
                # Single level columns (shouldn't happen for multi-ticker, but handle it)
                close_data = batch_data
                logger.warning("GET_STOCK_DATA: Expected MultiIndex columns for multi-ticker download")
            
            # Fill missing values to create a clean, consistent dataset
            filled_data = close_data.ffill().bfill()
            final_data = filled_data.dropna(axis=1, how='all')  # Drop any columns that are still all NaN
            logger.info(f"GET_STOCK_DATA: Final data shape after cleaning: {final_data.shape}")
        
        successful_tickers = list(final_data.columns)
        failed_tickers = list(set(tickers) - set(successful_tickers))
        
        logger.info(f"GET_STOCK_DATA BATCH SUMMARY: {len(successful_tickers)} successful, {len(failed_tickers)} failed")
        logger.info(f"GET_STOCK_DATA SUCCESSFUL: {successful_tickers[:10]}{'...' if len(successful_tickers) > 10 else ''}")
        if failed_tickers:
            logger.warning(f"GET_STOCK_DATA FAILED: {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")
        
        logger.info(f"GET_STOCK_DATA FINAL: Returning data with shape {final_data.shape}")
        return final_data
        
    except Exception as e:
        logger.error(f"GET_STOCK_DATA: Batch download failed: {e}. Falling back to individual downloads.")
        
        # Fallback to individual downloads if batch fails
        all_data = []
        successful_tickers = []
        failed_tickers = []
        
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
                if data.empty:
                    failed_tickers.append(ticker)
                    continue
                data.name = ticker
                all_data.append(data)
                successful_tickers.append(ticker)
            except Exception as ticker_error:
                logger.error(f"GET_STOCK_DATA: Error fetching data for {ticker}: {ticker_error}")
                failed_tickers.append(ticker)
                continue
        
        if not all_data:
            logger.warning("GET_STOCK_DATA: No successful data fetches, returning empty DataFrame")
            return pd.DataFrame()
        
        combined_data = pd.concat(all_data, axis=1)
        filled_data = combined_data.ffill().bfill()
        final_data = filled_data.dropna(axis=1, how='all')
        
        logger.info(f"GET_STOCK_DATA FALLBACK FINAL: Returning data with shape {final_data.shape}")
        return final_data

def _forecast_single_ticker(ticker, ticker_data):
    """Helper function to forecast returns for a single ticker (for parallel processing)."""
    try:
        # Create a clean DataFrame for Prophet with explicit column structure
        df_prophet = pd.DataFrame({
            'ds': ticker_data.index,
            'y': ticker_data.values
        })
        
        # Ensure the date column is properly formatted
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

        # Skip if data is empty or insufficient
        if df_prophet.shape[0] < 2:
            logger.warning(f"Skipping forecast for {ticker}: insufficient data ({df_prophet.shape[0]} data points).")
            return ticker, 0
        
        # Additional data quality check
        if df_prophet['y'].isna().sum() > len(df_prophet) * 0.5:
            logger.warning(f"Skipping forecast for {ticker}: too many missing values ({df_prophet['y'].isna().sum()}/{len(df_prophet)}).")
            # Use simple historical mean as fallback
            historical_returns = df_prophet['y'].pct_change().dropna()
            if len(historical_returns) > 0:
                forecast_value = historical_returns.mean() * 252  # Annualized
                logger.info(f"Using historical mean fallback for {ticker}: {forecast_value:.4f}")
                return ticker, forecast_value
            else:
                return ticker, 0

        # Initialize and fit the model with error handling
        try:
            model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
            model.fit(df_prophet)
            logger.info(f"Prophet model fitted successfully for {ticker}")

            # Make future dataframe
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)

            # Calculate expected return (annualized)
            if len(forecast) >= 365:
                expected_return = (forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[-365]) - 1
                logger.info(f"Successfully forecasted returns for {ticker}: {expected_return:.4f}")
                return ticker, expected_return
            else:
                logger.warning(f"Insufficient forecast data for {ticker}, using historical mean fallback")
                raise ValueError("Insufficient forecast data")
                
        except Exception as prophet_error:
            logger.warning(f"Prophet forecasting failed for {ticker}: {prophet_error}. Using historical mean fallback.")
            # Fallback to historical mean calculation
            historical_returns = df_prophet['y'].pct_change().dropna()
            if len(historical_returns) > 0:
                forecast_value = historical_returns.mean() * 252
                logger.info(f"Using historical mean fallback for {ticker}: {forecast_value:.4f}")
                return ticker, forecast_value
            else:
                return ticker, 0.05

    except Exception as e:
        logger.error(f"Critical error processing {ticker}: {e}")
        return ticker, 0.05

def forecast_returns(data):
    """Forecast expected returns using Prophet with parallel processing."""
    start_time = time.time()
    logger.info(f"Starting PARALLEL return forecasting for {len(data.columns)} tickers")
    
    # Determine optimal number of workers (don't exceed CPU count or ticker count)
    import os
    max_workers = min(os.cpu_count() or 4, len(data.columns), 8)  # Cap at 8 to avoid overwhelming the system
    logger.info(f"Using {max_workers} parallel workers for forecasting")
    
    forecasts = {}
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all forecasting tasks
        future_to_ticker = {
            executor.submit(_forecast_single_ticker, ticker, data[ticker]): ticker 
            for ticker in data.columns
        }
        
        # Collect results as they complete
        completed_count = 0
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result_ticker, forecast_value = future.result()
                forecasts[result_ticker] = forecast_value
                completed_count += 1
                
                # Progress logging
                if completed_count % 10 == 0 or completed_count == len(data.columns):
                    logger.info(f"Forecasting progress: {completed_count}/{len(data.columns)} completed")
                    
            except Exception as exc:
                logger.error(f"Forecasting generated an exception for {ticker}: {exc}")
                forecasts[ticker] = 0.05  # Default fallback
    
    elapsed_time = time.time() - start_time
    logger.info(f"PARALLEL forecasting completed in {elapsed_time:.2f} seconds for {len(forecasts)} tickers")
    
    return pd.Series(forecasts)

def optimize_portfolio(start_date, end_date, risk_free_rate, ticker_group=None, tickers=None, target_return=None, risk_tolerance=None):
    """
    Optimize portfolio based on user preferences using PyPortfolioOpt.
    """
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

    # Calculate expected returns using ML forecast and sample covariance
    logger.info(f"Starting forecasting for {len(data.columns)} tickers: {list(data.columns)}")
    mu = forecast_returns(data)
    logger.info(f"Forecasting completed. Got forecasts for {len(mu)} tickers: {list(mu.index)}")
    
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
        price_data = yf.download(final_tickers, period='1d', auto_adjust=True)['Close']
        if not price_data.empty:
            for ticker in final_tickers:
                if ticker in price_data:
                    # Ensure we get a single price, even if only one ticker is requested
                    if len(final_tickers) == 1:
                        latest_prices[ticker] = price_data.iloc[-1]
                    else:
                        latest_prices[ticker] = price_data[ticker].iloc[-1]


    return {
        "weights": final_weights,
        "return": optimized_return,
        "risk": optimized_std_dev,
        "sharpe_ratio": optimized_sharpe_ratio,
        "prices": latest_prices
    }
