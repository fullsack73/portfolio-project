from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import yfinance as yf # make sure to update the yf library to prevent interpreter from bitching
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from hedge_analysis import analyze_hedge_relationship
from montecarlo import calculate_portfolio_metrics, prepare_portfolio_data
import lightgbm as lgb
import pandas as pd
from financial_statement import get_financial_ratios

app = Flask(__name__)

# don't touch the settings in CORS(). JUST DON'T. it took me fucking ages to get it working
CORS(app, 
     resources={
         "/*": {
             "origins": ["http://localhost:5173", "http://127.0.0.1:*", "https://gannet-included-jolly.ngrok-free.app","http://127.0.0.1:63200"],
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
def generate_regression_data(ticker="", start_date=None, end_date=None, future_days=0):
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
            
        # Feature Engineering: Adding moving averages, lagged price, and volume
        df['Time'] = np.arange(len(df)) # Original time index
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean() # A common short-medium term moving average
        df['Lag1'] = df['Close'].shift(1) # Previous day's closing price
        # 'Volume' is typically available in df from yf.Ticker().history()

        # Drop rows with NaN values created by rolling window and shift operations
        # This ensures the model trains on complete data points
        df.dropna(inplace=True)
        
        if df.empty:
            print(f"Not enough data for {ticker} after feature engineering (NaN drop). Consider a longer date range or simpler features.")
            return {}
            
        # Prepare data for regression using the new features
        feature_columns = ['Time', 'MA7', 'MA21', 'Lag1', 'Volume']
        X = df[feature_columns].values
        y = df['Close'].values  # Target variable remains the closing price
        
        # fit LightGBM model
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31)
        model.fit(X, y)
        
        # generate regression line
        y_pred = model.predict(X)
        
        # convert to date:value format for both original and regression data
        dates = df.index.strftime('%Y-%m-%d').tolist()
        original_data = {date: float(price) for date, price in zip(dates, y)}
        regression_data = {date: float(price) for date, price in zip(dates, y_pred)}
        
        # get stock info
        info = stock.info
        company_name = info.get('longName', ticker)

        future_predictions = {}
        if future_days > 0 and X.shape[0] > 0: # Ensure there's data to predict from
            last_date = df.index[-1]
            # Keep a rolling window of prices for MA calculation, including historical and new predictions
            all_prices = list(y)
            # Last known volume (simplification)
            last_volume = df['Volume'].iloc[-1] if 'Volume' in df.columns and not df['Volume'].empty else 0 
            # Last time index
            last_time_index = df['Time'].iloc[-1]

            current_features_df = df[feature_columns].copy()

            for i in range(future_days):
                next_date = last_date + timedelta(days=i + 1)
                
                # Prepare features for the next day
                lag1_future = all_prices[-1]
                
                temp_prices_for_ma = pd.Series(all_prices)
                ma7_future = temp_prices_for_ma.rolling(window=7).mean().iloc[-1] if len(temp_prices_for_ma) >= 7 else np.nan
                ma21_future = temp_prices_for_ma.rolling(window=21).mean().iloc[-1] if len(temp_prices_for_ma) >= 21 else np.nan

                time_future = last_time_index + i + 1

                future_feature_vector = np.array([[time_future, ma7_future, ma21_future, lag1_future, last_volume]])
                
                if np.isnan(future_feature_vector[0, 1]): # MA7
                    future_feature_vector[0, 1] = current_features_df['MA7'].iloc[-1] if not current_features_df['MA7'].empty else 0
                if np.isnan(future_feature_vector[0, 2]): # MA21
                    future_feature_vector[0, 2] = current_features_df['MA21'].iloc[-1] if not current_features_df['MA21'].empty else 0

                # Predict
                predicted_price = model.predict(future_feature_vector)[0]
                
                # Store prediction and update for next iteration
                future_predictions[next_date.strftime('%Y-%m-%d')] = float(predicted_price)
                all_prices.append(predicted_price)
        
        return {
            'prices': original_data,
            'regression': regression_data,
            'future_predictions': future_predictions,
            'companyName': company_name,
            'slope': 'N/A',
            'intercept': 'N/A'
        }
        
    except Exception as e:
        print(f"Error generating regression data: {str(e)}")
        return {
            'prices': {},
            'regression': {},
            'companyName': ticker, 
            'slope': 'N/A', 
            'intercept': 'N/A',
            'future_predictions': {}
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
            'companyName': company_name,
            'regression': {},
            'future_predictions': {}
        }
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return {'prices': {}, 'companyName': ticker, 'regression': {}, 'future_predictions': {}}
    

    
# API endpoint to send data to the frontend
@app.route('/get-data', methods=['GET', 'OPTIONS'])
def get_data():
    ticker = request.args.get('ticker', 'AAPL')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    include_regression = request.args.get('regression', 'false').lower() == 'true'
    future_days_str = request.args.get('future_days', '0')
    try:
        future_days = int(future_days_str)
        if future_days < 0:
            future_days = 0 # Prevent negative days
    except ValueError:
        future_days = 0 # Default to 0 if conversion fails

    if include_regression:
        data = generate_regression_data(ticker, start_date, end_date, future_days=future_days)
    else:
        data = generate_data(ticker, start_date, end_date)

    return jsonify(data)

# add new endpoint for hedge analysis
@app.route('/analyze-hedge', methods=['GET', 'OPTIONS'])
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
        optimal_portfolio, final_ret, final_vol, final_sharpe, opts, optv, rets = calculate_portfolio_metrics(tickers, start_date, end_date, riskless_rate)
        portfolio_data = prepare_portfolio_data(opts, optv, rets, riskless_rate)

        print("Optimal portfolio weights:", optimal_portfolio) # This will now print the ticker and weight
        
        data = {
            'final_weights': optimal_portfolio,
            'final_return': final_ret,
            'final_volatility': final_vol,
            'final_sharpe_ratio': final_sharpe,
            'data': portfolio_data
        }

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/financial-statement', methods=['GET'])
def financial_statement():
    print("--- Received request for financial statement ---")
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400

    data = get_financial_ratios(ticker)
    if 'error' in data:
        return jsonify(data), 404

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)