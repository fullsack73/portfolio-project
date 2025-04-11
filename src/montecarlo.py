import pandas as pd # not sure if i need this but fuck it, import it anyway.
import numpy as np
import scipy.optimize as sco
import yfinance as yf
from datetime import datetime, timedelta

def port_ret(weights, rets):
    return np.sum(rets.mean() * weights) * 252

# i won't even begin to claim that i know what the fuck is going on here.
def port_vol(weights, rets):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

def min_func_sharpe(weights, rets):
    return -port_ret(weights, rets) / port_vol(weights, rets)

def validate_date_range(start_date, end_date):
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Check if dates are valid
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
            
        # Check if dates are not too far in the past
        max_days_back = 365 * 10  # 10 years max
        if (datetime.now() - start_date).days > max_days_back:
            raise ValueError(f"Start date cannot be more than {max_days_back/365:.0f} years in the past")
            
        # Check if dates are not in the future
        if end_date > datetime.now():
            raise ValueError("End date cannot be in the future")
            
        return start_date, end_date
    except ValueError as e:
        raise ValueError(f"Invalid date range: {str(e)}")
    except Exception as e:
        raise ValueError(f"Invalid date format. Please use YYYY-MM-DD format")

def calculate_portfolio_metrics(tickers=None, start_date=None, end_date=None):
    # default tickers. to prevent the function from crashing. btw, why the fuck would anyone want to use this function without tickers??
    if tickers is None:
        tickers = ['SPY', 'GLD', 'AAPL', 'MSFT']
    
    noa = len(tickers)
    
    # use provided dates or default to 1 year from today
    if start_date is None or end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    else:
        start_date, end_date = validate_date_range(start_date, end_date)
    
    # fetch data using yfinance
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # use 'Adj Close' if available, otherwise fallback to 'Close'
    if 'Adj Close' in data.columns:
        data = data['Adj Close']
    else:
        data = data['Close']
    
    data = data.dropna()
    
    # calculate returns
    rets = np.log(data / data.shift(1))
    
    # initialize weights
    weights = np.random.random(noa)
    weights = weights / np.sum(weights)
    
    # set up constraints and bounds
    cons = ({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)})
    bounds = tuple((0, 1) for _ in range(noa))
    
    # optimization
    opts = sco.minimize(min_func_sharpe, weights, args=(rets,), method='SLSQP', bounds=bounds, constraints=cons)
    optv = sco.minimize(port_vol, weights, args=(rets,), method='SLSQP', bounds=bounds, constraints=cons)
    
    final_weights = opts['x'].round(3)
    final_ret = port_ret(final_weights, rets)
    final_vol = port_vol(final_weights, rets)
    final_sharpe = final_ret / final_vol
    
    return final_weights, final_ret, final_vol, final_sharpe, opts, optv, rets

def prepare_portfolio_data(opts, optv, rets):
    # generate random portfolios for visualization
    noa = len(rets.columns)
    prets = []
    pvols = []
    
    for p in range(2500):
        weights = np.random.random(noa)
        weights = weights / np.sum(weights)
        prets.append(port_ret(weights, rets))
        pvols.append(port_vol(weights, rets))
    
    prets = np.array(prets)
    pvols = np.array(pvols)
    
    # generate efficient frontier
    trets = np.linspace(0.05, 0.2, 50)
    tvols = []

    for tret in trets:
        cons = ({'type': 'eq', 'fun': lambda x: port_ret(x, rets) - tret}, 
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        res = sco.minimize(port_vol, np.random.random(noa), args=(rets,), 
                         method='SLSQP', 
                         bounds=tuple((0, 1) for _ in range(noa)), 
                         constraints=cons)
        tvols.append(res['fun'])

    tvols = np.array(tvols)

    # return the data instead of plotting
    return {
        'pvols': pvols.tolist(),
        'prets': prets.tolist(),
        'tvols': tvols.tolist(),
        'trets': trets.tolist(),
        'opt_vol': port_vol(opts['x'], rets),
        'opt_ret': port_ret(opts['x'], rets),
        'optv_vol': port_vol(optv['x'], rets),
        'optv_ret': port_ret(optv['x'], rets)
    }

# call the functions
final_weights, final_ret, final_vol, final_sharpe, opts, optv, rets = calculate_portfolio_metrics()
portfolio_data = prepare_portfolio_data(opts, optv, rets)
print("Final weights: ", final_weights)
print("Final return: ", final_ret)
print("Final volatility: ", final_vol)
print("Final Sharpe ratio: ", final_sharpe)