from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats

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
