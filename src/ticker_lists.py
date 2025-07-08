import pandas as pd

def get_sp500_tickers():
    """Gets the list of S&P 500 tickers from Wikipedia."""
    try:
        # Wikipedia page with the list of S&P 500 companies
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # Read the HTML table into a pandas DataFrame
        tables = pd.read_html(url)
        sp500_table = tables[0]
        # The tickers are in the 'Symbol' column
        tickers = sp500_table['Symbol'].tolist()
        # Some tickers on Wikipedia might be in the format 'BF.B', but yfinance expects 'BF-B'
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        # Return a smaller, hardcoded list as a fallback
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'V', 'JNJ', 'WMT', 'PG']

TICKER_GROUPS = {
    "SP500": get_sp500_tickers
}

def get_ticker_group(group_name):
    """Returns a list of tickers for a given group name."""
    if group_name in TICKER_GROUPS:
        return TICKER_GROUPS[group_name]()
    else:
        raise ValueError(f"Unknown ticker group: {group_name}")
