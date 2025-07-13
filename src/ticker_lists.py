import pandas as pd

def get_sp500_tickers():
    """Gets the list of S&P 500 tickers from snp.csv."""
    try:
        df = pd.read_csv('snp.csv')
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []

def get_dow_tickers():
    """Gets the list of Dow Jones tickers from dow.csv."""
    try:
        df = pd.read_csv('dow.csv')
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching Dow Jones tickers: {e}")
        return []

TICKER_GROUPS = {
    "SP500": get_sp500_tickers,
    "DOW": get_dow_tickers
}

def get_ticker_group(group_name):
    """Returns a list of tickers for a given group name."""
    if group_name in TICKER_GROUPS:
        return TICKER_GROUPS[group_name]()
    else:
        raise ValueError(f"Unknown ticker group: {group_name}")
