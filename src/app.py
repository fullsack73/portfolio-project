from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import yfinance as yf # make sure to update the yf library to prevent interpreter from bitching
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from hedge_analysis import analyze_hedge_relationship
from montecarlo import calculate_portfolio_metrics, prepare_portfolio_data

app = Flask(__name__)

# don't touch the settings in CORS(). JUST DON'T. it took me fucking ages to get it working
CORS(app, 
     resources={
         r"/*": {
             "origins": ["http://localhost:5173", "http://127.0.0.1:5173", "https://gannet-included-jolly.ngrok-free.app"],
             "methods": ["GET", "POST", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization", "Accept"],
             "supports_credentials": True,
             "expose_headers": ["Content-Type", "Authorization"],
             "max_age": 3600
         }
     })

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

# maybe go with sliding window approach, but for now this is good
def generate_regression_data(ticker="", start_date=None, end_date=None):
    try:
        # default to 3 months to 'yesterday'. otherwise yfinance might start to fuck up
        if start_date and end_date:
            start_date, end_date = validate_date_range(start_date, end_date)
        else:
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=90)
        
        print(f"Fetching {ticker} data from {start_date} to {end_date}")
        
        # fetch stock data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                          end=end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            print(f"No data received from yfinance for {ticker}")
            return {}
            
        # prepare data for regression
        X = np.arange(len(df)).reshape(-1, 1)  # days as features
        y = df['Close'].values  # closing prices as target
        
        # scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # fit linear regression. built in linear regression is slow as fuck. but i've got no other options.
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # generate regression line
        y_pred = model.predict(X_scaled)
        
        # convert to date:value format for both original and regression data
        dates = df.index.strftime('%Y-%m-%d').tolist()
        original_data = {date: float(price) for date, price in zip(dates, y)}
        regression_data = {date: float(price) for date, price in zip(dates, y_pred)}
        
        # get stock info
        info = stock.info
        company_name = info.get('longName', ticker)
        
        # calculate the actual formula coefficients (unscaled)
        n_days = len(df)
        mean_days = (n_days - 1) / 2
        std_days = np.sqrt((n_days - 1) * (n_days + 1) / 12)
        
        # convert scaled coefficients to unscaled
        actual_slope = model.coef_[0] / std_days
        actual_intercept = model.intercept_ - (model.coef_[0] * mean_days / std_days)
        
        # format the formula
        formula = f"price = {actual_slope:.4f} * days + {actual_intercept:.2f}"
        
        return {
            'prices': original_data,
            'regression': regression_data,
            'companyName': company_name,
            'slope': float(actual_slope),
            'intercept': float(actual_intercept),
            'formula': formula
        }
        
    except Exception as e:
        print(f"Error generating regression data: {str(e)}")
        return {
            'prices': {}, 
            'regression': {}, 
            'companyName': ticker, 
            'slope': 0, 
            'intercept': 0,
            'formula': "price = 0 * days + 0"
        }

def generate_data(ticker="", start_date=None, end_date=None):
    try:
        # use provided dates or default to 3 months from 'yesterday'
        if start_date and end_date:
            start_date, end_date = validate_date_range(start_date, end_date)
        else:
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=90)
        
        print(f"Fetching {ticker} data from {start_date} to {end_date}")
        
        # fetch stock data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                          end=end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            print(f"No data received from yfinance for {ticker}")
            return {}
            
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns}")
        print(f"First few rows:\n{df.head()}")
        
        # get stock info
        info = stock.info
        company_name = info.get('longName', ticker)
        
        # convert to date:value format
        data = {date.strftime('%Y-%m-%d'): float(price) for date, price in zip(df.index, df['Close'])}
        print(f"Generated data dictionary with {len(data)} entries")
        
        return {
            'prices': data,
            'companyName': company_name
        }
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return {'prices': {}, 'companyName': ticker}
    

    
# API endpoint to send data to the frontend
@app.route('/get-data', methods=['GET', 'OPTIONS'])
def get_data():
    ticker = request.args.get('ticker', 'AAPL')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    include_regression = request.args.get('regression', 'false').lower() == 'true'

    if include_regression:
        data = generate_regression_data(ticker, start_date, end_date)
    else:
        data = generate_data(ticker, start_date, end_date)

    return jsonify(data)

# add new endpoint for hedge analysis
@app.route('/api/analyze-hedge', methods=['GET', 'OPTIONS'])
def analyze_hedge():
    ticker1 = request.args.get('ticker1')
    ticker2 = request.args.get('ticker2')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    if not ticker1 or not ticker2:
        return jsonify({'error': 'Both tickers are required'}), 400

    result = analyze_hedge_relationship(ticker1, ticker2, start_date, end_date)
    return jsonify(result)

# endpoint for portfolio metrics. keeping files seperated. it's fucking spagetti code otherwise.
@app.route('/portfolio-metrics', methods=['GET', 'OPTIONS'])
def get_portfolio_metrics():
    # get parameters from request
    tickers = request.args.get('tickers', '').split(',')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    riskless_rate = request.args.get('riskless_rate', default=0.0, type=float)

    if not tickers or tickers[0] == '':
        return jsonify({'error': 'At least one ticker is required'}), 400

    try:
        # calculate portfolio metrics.
        final_weights, final_ret, final_vol, final_sharpe, opts, optv, rets = calculate_portfolio_metrics(tickers, start_date, end_date, riskless_rate)
        portfolio_data = prepare_portfolio_data(opts, optv, rets, riskless_rate)
        
        data = {
            'final_weights': final_weights.tolist(),
            'final_return': final_ret,
            'final_volatility': final_vol,
            'final_sharpe_ratio': final_sharpe,
            'data': portfolio_data
        }

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)