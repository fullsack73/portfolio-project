from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"], supports_credentials=True)

def generate_data():
    try:
        # Calculate date range (3 months from yesterday)
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=90)     # 3 months before yesterday
        
        print(f"Fetching data from {start_date} to {end_date}")
        
        # Fetch Apple stock data
        apple = yf.Ticker("AAPL")
        df = apple.history(start=start_date.strftime('%Y-%m-%d'), 
                          end=end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            print("No data received from yfinance")
            return {}
            
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns}")
        print(f"First few rows:\n{df.head()}")
        
        # Convert to date:value format
        data = {date.strftime('%Y-%m-%d'): float(price) for date, price in zip(df.index, df['Close'])}
        print(f"Generated data dictionary with {len(data)} entries")
        return data
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return {}

# API endpoint to send data to the frontend
@app.route('/get-data', methods=['GET'])
def get_data():
    data = generate_data()  # Call the Python function
    return jsonify(data)    # Send data as JSON

if __name__ == '__main__':
    app.run(debug=True)
    