from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf # make sure to update the yf library to prevent interpreter from bitching
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

app = Flask(__name__)
# Update CORS configuration to allow all necessary headers and methods
CORS(app, 
     resources={r"/*": {
         "origins": ["http://localhost:5173"],
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "Accept"],
         "supports_credentials": True
     }})


# maybe go with sliding window approach, but for now this is good
def generate_regression_data(ticker="", start_date=None, end_date=None):
    try:
        # Use provided dates or default to 3 months from yesterday
        if start_date and end_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=90)
        
        print(f"Fetching {ticker} data from {start_date} to {end_date}")
        
        # Fetch stock data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                          end=end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            print(f"No data received from yfinance for {ticker}")
            return {}
            
        # Prepare data for regression
        X = np.arange(len(df)).reshape(-1, 1)  # Days as features
        y = df['Close'].values  # Closing prices as target
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Generate regression line
        y_pred = model.predict(X_scaled)
        
        # Convert to date:value format for both original and regression data
        dates = df.index.strftime('%Y-%m-%d').tolist()
        original_data = {date: float(price) for date, price in zip(dates, y)}
        regression_data = {date: float(price) for date, price in zip(dates, y_pred)}
        
        # Get stock info
        info = stock.info
        company_name = info.get('longName', ticker)
        
        # Calculate the actual formula coefficients (unscaled)
        n_days = len(df)
        mean_days = (n_days - 1) / 2
        std_days = np.sqrt((n_days - 1) * (n_days + 1) / 12)
        
        # Convert scaled coefficients to unscaled
        actual_slope = model.coef_[0] / std_days
        actual_intercept = model.intercept_ - (model.coef_[0] * mean_days / std_days)
        
        # Format the formula
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

def analyze_hedge_relationship(ticker1, ticker2, start_date=None, end_date=None):
    try:
        # Use provided dates or default to 3 months from yesterday
        if start_date and end_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=90)
        
        print(f"Analyzing hedge relationship between {ticker1} and {ticker2}")
        
        # Fetch data for both tickers
        stock1 = yf.Ticker(ticker1)
        stock2 = yf.Ticker(ticker2)
        
        df1 = stock1.history(start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'))
        df2 = stock2.history(start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'))
        
        if df1.empty or df2.empty:
            return {
                'error': 'No data available for one or both tickers',
                'is_hedge': 0,  # Changed from False to 0
                'correlation': 0,
                'p_value': 0,
                'strength': 'None'
            }
        
        # Calculate daily returns
        returns1 = df1['Close'].pct_change().dropna()
        returns2 = df2['Close'].pct_change().dropna()
        
        # Ensure both series have the same dates
        common_dates = returns1.index.intersection(returns2.index)
        returns1 = returns1[common_dates]
        returns2 = returns2[common_dates]
        
        # Calculate correlation and p-value
        correlation, p_value = stats.pearsonr(returns1, returns2)
        
        # Determine hedge relationship
        is_hedge = correlation < -0.5  # Strong negative correlation
        strength = 'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.5 else 'Weak'
        
        # Get company names
        info1 = stock1.info
        info2 = stock2.info
        company1 = info1.get('longName', ticker1)
        company2 = info2.get('longName', ticker2)
        
        return {
            'is_hedge': 1 if is_hedge else 0,  # Convert boolean to integer
            'correlation': float(correlation),
            'p_value': float(p_value),
            'strength': strength,
            'company1': company1,
            'company2': company2,
            'ticker1': ticker1,
            'ticker2': ticker2,
            'period': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
        }
        
    except Exception as e:
        print(f"Error analyzing hedge relationship: {str(e)}")
        return {
            'error': str(e),
            'is_hedge': 0,  # Changed from False to 0
            'correlation': 0,
            'p_value': 0,
            'strength': 'None'
        }

def generate_data(ticker="", start_date=None, end_date=None):
    try:
        # Use provided dates or default to 3 months from yesterday
        if start_date and end_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=90)
        
        print(f"Fetching {ticker} data from {start_date} to {end_date}")
        
        # Fetch stock data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                          end=end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            print(f"No data received from yfinance for {ticker}")
            return {}
            
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns}")
        print(f"First few rows:\n{df.head()}")
        
        # Get stock info
        info = stock.info
        company_name = info.get('longName', ticker)
        
        # Convert to date:value format
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
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response

    ticker = request.args.get('ticker', 'AAPL')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    include_regression = request.args.get('regression', 'false').lower() == 'true'

    if include_regression:
        data = generate_regression_data(ticker, start_date, end_date)
    else:
        data = generate_data(ticker, start_date, end_date)

    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Add new endpoint for hedge analysis
@app.route('/analyze-hedge', methods=['GET', 'OPTIONS'])
def analyze_hedge():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response

    ticker1 = request.args.get('ticker1')
    ticker2 = request.args.get('ticker2')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    if not ticker1 or not ticker2:
        return jsonify({'error': 'Both tickers are required'}), 400

    result = analyze_hedge_relationship(ticker1, ticker2, start_date, end_date)
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == '__main__':
    app.run(debug=True)
    