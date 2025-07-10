import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from ticker_lists import get_ticker_group

def get_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio performance metrics."""
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std_dev

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Calculate the negative Sharpe ratio for optimization."""
    p_returns, p_std_dev = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std_dev

def optimize_portfolio(ticker_group, start_date, end_date, risk_free_rate, target_return=None, risk_tolerance=None):
    """
    Optimize portfolio based on user preferences.
    """
    # Get tickers from the selected group
    tickers = get_ticker_group(ticker_group)

    # Fetch data
    data = get_stock_data(tickers, start_date, end_date)
    # Drop columns with all NaN values (e.g., stocks with no data in the given range)
    data = data.dropna(axis=1, how='all')
    tickers = data.columns.tolist()
    
    mean_returns = data.pct_change().mean()
    cov_matrix = data.pct_change().cov()
    num_assets = len(tickers)

    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    if target_return:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: calculate_portfolio_performance(x, mean_returns, cov_matrix)[0] - target_return}
        )
    
    if risk_tolerance:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: risk_tolerance - calculate_portfolio_performance(x, mean_returns, cov_matrix)[1]}
        )

    # Bounds
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Initial guess
    initial_weights = np.array(num_assets * [1. / num_assets])

    # Optimization
    result = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate),
                        method='SLSQP', bounds=bounds, constraints=constraints)

    # Optimized portfolio
    optimized_weights = result.x
    optimized_return, optimized_std_dev = calculate_portfolio_performance(optimized_weights, mean_returns, cov_matrix)
    optimized_sharpe_ratio = (optimized_return - risk_free_rate) / optimized_std_dev

    # Filter out assets with near-zero weight
    final_weights = {ticker: weight for ticker, weight in zip(tickers, optimized_weights) if weight > 1e-4}

    # Get the latest prices for the tickers in the final portfolio
    latest_prices = {}
    if final_weights:
        final_tickers = list(final_weights.keys())
        price_data = yf.download(final_tickers, period='1d')['Close']
        if not price_data.empty:
            for ticker in final_tickers:
                if ticker in price_data:
                    latest_prices[ticker] = price_data[ticker].iloc[-1]

    return {
        "weights": final_weights,
        "return": optimized_return,
        "risk": optimized_std_dev,
        "sharpe_ratio": optimized_sharpe_ratio,
        "prices": latest_prices
    }
