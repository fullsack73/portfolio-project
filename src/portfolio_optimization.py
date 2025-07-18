import numpy as np
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
from ticker_lists import get_ticker_group

def get_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

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

    # Fetch data
    data = get_stock_data(tickers, start_date, end_date)
    data = data.dropna(axis=1, how='all')
    tickers = data.columns.tolist()

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

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
        price_data = yf.download(final_tickers, period='1d')['Close']
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
