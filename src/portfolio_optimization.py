import logging
from logging import NullHandler
import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models
from prophet import Prophet
from ticker_lists import get_ticker_group
import cmdstanpy

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
    """Fetch stock data for given tickers and date range."""
    logger.info(f"GET_STOCK_DATA: Starting fetch for {len(tickers)} tickers")
    
    # Sanitize tickers first
    original_tickers = tickers.copy()
    tickers = sanitize_tickers(tickers)
    sanitized_changes = set(original_tickers) - set(tickers)
    if sanitized_changes:
        logger.info(f"GET_STOCK_DATA: Sanitized {len(sanitized_changes)} ticker symbols")
    
    all_data = []
    successful_tickers = []
    failed_tickers = []
    
    for ticker in tickers:
        try:
            logger.info(f"GET_STOCK_DATA: Fetching data for {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
            if data.empty:
                logger.warning(f"GET_STOCK_DATA: No data found for {ticker}, skipping.")
                failed_tickers.append(ticker)
                continue
            data.name = ticker
            all_data.append(data)
            successful_tickers.append(ticker)
            logger.info(f"GET_STOCK_DATA: Successfully fetched {len(data)} data points for {ticker}")
        except Exception as e:
            logger.error(f"GET_STOCK_DATA: Error fetching data for {ticker}: {e}")
            failed_tickers.append(ticker)
            continue

    logger.info(f"GET_STOCK_DATA SUMMARY: {len(successful_tickers)} successful, {len(failed_tickers)} failed")
    logger.info(f"GET_STOCK_DATA SUCCESSFUL: {successful_tickers[:10]}{'...' if len(successful_tickers) > 10 else ''}")
    if failed_tickers:
        logger.warning(f"GET_STOCK_DATA FAILED: {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")

    if not all_data:
        logger.warning("GET_STOCK_DATA: No successful data fetches, returning empty DataFrame")
        return pd.DataFrame()

    # Combine all successful downloads into a single DataFrame
    logger.info(f"GET_STOCK_DATA: Combining {len(all_data)} successful data series")
    combined_data = pd.concat(all_data, axis=1)
    logger.info(f"GET_STOCK_DATA: Combined data shape before filling: {combined_data.shape}")

    # Fill missing values to create a clean, consistent dataset
    # Forward-fill handles weekends/holidays, back-fill handles missing data at the start
    filled_data = combined_data.ffill().bfill()
    logger.info(f"GET_STOCK_DATA: Data shape after filling: {filled_data.shape}")
    
    final_data = filled_data.dropna(axis=1, how='all')  # Drop any columns that are still all NaN
    logger.info(f"GET_STOCK_DATA FINAL: Returning data with shape {final_data.shape} and columns {list(final_data.columns)[:10]}{'...' if len(final_data.columns) > 10 else ''}")
    
    return final_data

def forecast_returns(data):
    """Forecast expected returns using Prophet."""
    logger.info(f"Starting return forecasting for tickers: {', '.join(data.columns)}")
    forecasts = {}
    for ticker in data.columns:
        try:
            # Prepare data for Prophet with robust column handling
            df_single_ticker = data[[ticker]].copy()
            
            # Create a clean DataFrame for Prophet with explicit column structure
            df_prophet = pd.DataFrame({
                'ds': df_single_ticker.index,
                'y': df_single_ticker[ticker].values
            })
            
            # Ensure the date column is properly formatted
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

            # Skip if data is empty or insufficient
            if df_prophet.shape[0] < 2:
                logger.warning(f"Skipping forecast for {ticker}: insufficient data ({df_prophet.shape[0]} data points).")
                forecasts[ticker] = 0
                continue
            
            # Additional data quality check
            if df_prophet['y'].isna().sum() > len(df_prophet) * 0.5:
                logger.warning(f"Skipping forecast for {ticker}: too many missing values ({df_prophet['y'].isna().sum()}/{len(df_prophet)}).")
                # Use simple historical mean as fallback
                historical_returns = df_prophet['y'].pct_change().dropna()
                if len(historical_returns) > 0:
                    forecasts[ticker] = historical_returns.mean() * 252  # Annualized
                    logger.info(f"Using historical mean fallback for {ticker}: {forecasts[ticker]:.4f}")
                else:
                    forecasts[ticker] = 0
                continue

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
                    forecasts[ticker] = expected_return
                    logger.info(f"Successfully forecasted returns for {ticker}: {expected_return:.4f}")
                else:
                    logger.warning(f"Insufficient forecast data for {ticker}, using historical mean fallback")
                    raise ValueError("Insufficient forecast data")
                    
            except Exception as prophet_error:
                logger.warning(f"Prophet forecasting failed for {ticker}: {prophet_error}. Using historical mean fallback.")
                # Fallback to historical mean calculation
                historical_returns = df_prophet['y'].pct_change().dropna()
                if len(historical_returns) > 0:
                    forecasts[ticker] = historical_returns.mean() * 252
                    logger.info(f"Using historical mean fallback for {ticker}: {forecasts[ticker]:.4f}")
                else:
                    forecasts[ticker] = 0.05
                    logger.info(f"Using default return for {ticker}: 0.05")

        except Exception as e:
            logger.error(f"Critical error processing {ticker}: {e}")
            forecasts[ticker] = 0.05

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
