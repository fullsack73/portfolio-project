import logging
from finvizfinance.screener.overview import Overview
from cache_manager import cached
import pandas as pd

# Configure logging for this module
logger = logging.getLogger(__name__)

@cached(l1_ttl=3600, l2_ttl=86400)  # 1 hour L1, 24 hour L2 cache
def search_stocks(filters):
    """
    Searches for stocks using finvizfinance based on a dictionary of filters.

    Args:
        filters (dict): A dictionary where keys are filter names (e.g., 'Index', 'P/E')
                        and values are the desired filter values (e.g., 'S&P 500', 'Under 15').

    Returns:
        list: A list of dictionaries, where each dictionary represents a stock
              and its data. Returns an empty list if an error occurs or no stocks are found.
    """
    logger.info(f"Starting stock screen with filters: {filters}")
    try:
        foverview = Overview()
        foverview.set_filter(filters_dict=filters)
        
        df = foverview.screener_view()
        
        if df.empty:
            logger.warning(f"No stocks found for the given filters: {filters}")
            return []
            
        # Convert DataFrame to a list of dictionaries
        results = df.to_dict('records')
        logger.info(f"Found {len(results)} stocks matching the criteria.")
        return results

    except Exception as e:
        logger.error(f"An error occurred during the finvizfinance screen: {e}")
        # In case of an error (e.g., website down, parsing issue), return an empty list
        return []
