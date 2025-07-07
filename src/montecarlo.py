import pandas as pd # not sure if i need this but fuck it, import it anyway.
import numpy as np
import scipy.optimize as sco
import yfinance as yf
from datetime import datetime, timedelta

def port_ret(weights, rets):
    return np.sum(rets.mean() * weights) * 252

def port_vol(weights, rets):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

def min_func_sharpe(weights, rets, riskless_rate=0.0):
    # Negative Sharpe ratio (for minimization)
    port_return = port_ret(weights, rets)
    port_volatility = port_vol(weights, rets)
    return -((port_return - riskless_rate) / port_volatility)

def validate_date_range(start_date, end_date):
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Check if dates are valid
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
            
        # Check if dates are not in the future
        if end_date > datetime.now():
            raise ValueError("End date cannot be in the future")
            
        return start_date, end_date
    except ValueError as e:
        raise ValueError(f"Invalid date range: {str(e)}")
    except Exception as e:
        raise ValueError(f"Invalid date format. Please use YYYY-MM-DD format")

def calculate_portfolio_metrics(tickers=None, start_date=None, end_date=None, riskless_rate=0.0):
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL']
    
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
    
    # optimization for tangency portfolio (max Sharpe)
    opts = sco.minimize(min_func_sharpe, weights, args=(rets, riskless_rate), method='SLSQP', bounds=bounds, constraints=cons)
    optv = sco.minimize(port_vol, weights, args=(rets,), method='SLSQP', bounds=bounds, constraints=cons)
    
    final_weights = opts['x']
    final_ret = port_ret(final_weights, rets)
    final_vol = port_vol(final_weights, rets)
    final_sharpe = (final_ret - riskless_rate) / final_vol

    optimal_portfolio = {ticker: f'{weight:.2%}' for ticker, weight in zip(tickers, final_weights)}

    return optimal_portfolio, final_ret, final_vol, final_sharpe, opts, optv, rets

def prepare_portfolio_data(opts, optv, rets, riskless_rate=0.0):
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
    trets = np.linspace(min(prets.min(), 0), prets.max(), 50)
    tvols = []
    prev_weights = np.ones(noa) / noa  # start with equal weights

    for tret in trets:
        cons = ({'type': 'eq', 'fun': lambda x: port_ret(x, rets) - tret}, 
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        res = sco.minimize(port_vol, prev_weights, args=(rets,), 
                         method='SLSQP', 
                         bounds=tuple((0, 1) for _ in range(noa)), 
                         constraints=cons)
        if res.success:
            tvols.append(res.fun)
            prev_weights = res.x # warm start
        else:
            tvols.append(np.nan)

    tvols = np.array(tvols)
    # filter out nan values
    mask = ~np.isnan(tvols)
    tvols = tvols[mask]
    trets = trets[mask]

    # Tangency portfolio
    tangency_vol = port_vol(opts['x'], rets)
    tangency_ret = port_ret(opts['x'], rets)

    # Riskless asset point
    riskless_point = {'vol': 0.0, 'ret': riskless_rate}

    # Capital Market Line (CML): line from riskless asset through tangency portfolio
    # We'll plot from (0, riskless_rate) to a volatility a bit beyond the tangency portfolio
    cml_vols = np.linspace(0, max(tvols.max(), tangency_vol) * 1.2, 100)
    cml_slope = (tangency_ret - riskless_rate) / tangency_vol if tangency_vol > 0 else 0
    cml_rets = riskless_rate + cml_slope * cml_vols

    return {
        'pvols': pvols.tolist(),
        'prets': prets.tolist(),
        'tvols': tvols.tolist(),
        'trets': trets.tolist(),
        'opt_vol': tangency_vol,
        'opt_ret': tangency_ret,
        'optv_vol': port_vol(optv['x'], rets),
        'optv_ret': port_ret(optv['x'], rets),
        'riskless_point': riskless_point,
        'cml_vols': cml_vols.tolist(),
        'cml_rets': cml_rets.tolist()
    }

