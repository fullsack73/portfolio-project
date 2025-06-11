from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import yfinance as yf # make sure to update the yf library to prevent interpreter from bitching
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from hedge_analysis import analyze_hedge_relationship
from montecarlo import calculate_portfolio_metrics, prepare_portfolio_data
from sklearn.svm import SVR
import pandas as pd

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
        
        # Scale the features
        # Scaling is important for SVR and many other algorithms
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # fit SVR model
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
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
        
        # SVR does not provide a simple formula
        formula = "N/A (SVR model)"

        future_predictions = {}
        if future_days > 0 and X_scaled.shape[0] > 0: # Ensure there's data to predict from
            last_date = df.index[-1]
            last_price = y[-1]
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
                # Lag1 is the last predicted price (or last actual price for the first future step)
                lag1_future = all_prices[-1]
                
                # MA7 and MA21: need at least 7 and 21 points respectively
                # Append the latest price to all_prices to calculate rolling MAs
                # For the first few predictions, MAs might rely more on historical data
                temp_prices_for_ma = pd.Series(all_prices)
                ma7_future = temp_prices_for_ma.rolling(window=7).mean().iloc[-1] if len(temp_prices_for_ma) >= 7 else np.nan
                ma21_future = temp_prices_for_ma.rolling(window=21).mean().iloc[-1] if len(temp_prices_for_ma) >= 21 else np.nan

                # Time index continues to increment
                time_future = last_time_index + i + 1

                # Construct feature vector for the future day
                # Order must match 'Time', 'MA7', 'MA21', 'Lag1', 'Volume'
                future_feature_vector = np.array([[time_future, ma7_future, ma21_future, lag1_future, last_volume]])
                
                # Handle potential NaNs from MA calculation if not enough data points
                # A simple strategy: use the last known MA if NaN
                if np.isnan(future_feature_vector[0, 1]): # MA7
                    future_feature_vector[0, 1] = current_features_df['MA7'].iloc[-1] if not current_features_df['MA7'].empty else 0
                if np.isnan(future_feature_vector[0, 2]): # MA21
                    future_feature_vector[0, 2] = current_features_df['MA21'].iloc[-1] if not current_features_df['MA21'].empty else 0

                # Scale features
                future_feature_vector_scaled = scaler.transform(future_feature_vector)
                
                # Predict
                predicted_price = model.predict(future_feature_vector_scaled)[0]
                
                # Store prediction and update for next iteration
                future_predictions[next_date.strftime('%Y-%m-%d')] = float(predicted_price)
                all_prices.append(predicted_price)
        
        return {
            'prices': original_data,
            'regression': regression_data,
            'future_predictions': future_predictions,
            'companyName': company_name,
            'slope': 'N/A',
            'intercept': 'N/A',
            'formula': formula
        }
        
    except Exception as e:
        print(f"Error generating regression data: {str(e)}")
        return {
            'prices': {}, 
            'regression': {}, 
            'companyName': ticker, 
            'slope': 'N/A', 
            'intercept': 'N/A',
            'formula': "N/A (SVR model)",
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
            'formula': "",
            'future_predictions': {}
        }
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return {'prices': {}, 'companyName': ticker, 'regression': {}, 'formula': "", 'future_predictions': {}}
    

    
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