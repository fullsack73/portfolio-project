import logging
from finvizfinance.screener.overview import Overview
from cache_manager import cached
import pandas as pd

# Configure logging for this module
logger = logging.getLogger(__name__)

@cached(l1_ttl=3600, l2_ttl=86400)  # 1 hour L1, 24 hour L2 cache
def search_stocks(filters, v=2):
    """
    Searches for stocks using finvizfinance and returns a filtered set of columns.
    The 'v' parameter is for cache-busting.

    Args:
        filters (dict): A dictionary of filter criteria.
        v (int): Cache version number.

    Returns:
        list: A list of dictionaries, each representing a stock with filtered data.
    """
    logger.info(f"Starting stock screen with filters: {filters} (v={v})")
    try:
        foverview = Overview()
        foverview.set_filter(filters_dict=filters)
        
        df = foverview.screener_view()
        
        if df.empty:
            logger.warning(f"No stocks found for the given filters: {filters}")
            return []

        # Define columns to keep, based on frontend METRICS + identifiers
        # Frontend METRICS in StockScreener.jsx: 'P/E', 'P/B', 'Debt/Equity', 'ROE', 'Price/Cash'
        # Mapped to corresponding finvizfinance columns:
        columns_to_keep = [
            'Ticker', 
            'P/E',
            'P/B',
            'Debt/Eq',  # Corresponds to 'Debt/Equity'
            'ROE',
            'P/C'       # Corresponds to 'Price/Cash'
        ]

        # Filter the DataFrame to only include columns that actually exist in the result
        existing_columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        filtered_df = df[existing_columns_to_keep]
            
        # Convert DataFrame to a list of dictionaries
        results = filtered_df.to_dict('records')
        logger.info(f"Found {len(results)} stocks, returning {len(existing_columns_to_keep)} columns.")
        return results

    except Exception as e:
        logger.error(f"An error occurred during the finvizfinance screen: {e}")
        # In case of an error (e.g., website down, parsing issue), return an empty list
        return []

