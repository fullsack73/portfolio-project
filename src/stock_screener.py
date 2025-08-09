import logging
from finvizfinance.screener.valuation import Valuation
from finvizfinance.screener.financial import Financial
from cache_manager import cached
import pandas as pd
import numpy as np

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
        # Use Valuation and Financial screeners to get all necessary columns
        valuation = Valuation()
        valuation.set_filter(filters_dict=filters)
        df_valuation = valuation.screener_view()

        if df_valuation.empty:
            logger.warning(f"No stocks found for the given filters: {filters}")
            return []

        financial = Financial()
        financial.set_filter(filters_dict=filters)
        df_financial = financial.screener_view()
        
        # If financial screener fails, we can still proceed with valuation data
        if df_financial.empty:
            df = df_valuation
        else:
            # Merge valuation and financial dataframes
            df = pd.merge(df_valuation, df_financial.drop(columns=[c for c in df_financial.columns if c in df_valuation.columns and c != 'Ticker']), on='Ticker', how='left')

        if df.empty:
            logger.warning(f"No stocks found after merging for the given filters: {filters}")
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
            'P/S'       
        ]

        # Filter the DataFrame to only include columns that actually exist in the result
        existing_columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        filtered_df = df[existing_columns_to_keep]
            
        # Rename columns for frontend consistency
        rename_map = {
            'Debt/Eq': 'Debt/Equity',
            'P/S': 'Price/Sales'
        }
        filtered_df = filtered_df.rename(columns=rename_map)

        # Replace NaN with None for JSON compatibility
        filtered_df = filtered_df.replace(np.nan, None)

        # Convert DataFrame to a list of dictionaries
        results = filtered_df.to_dict('records')
        logger.info(f"Found {len(results)} stocks, returning data with columns: {list(filtered_df.columns)}")
        return results

    except Exception as e:
        logger.error(f"An error occurred during the finvizfinance screen: {e}")
        # In case of an error (e.g., website down, parsing issue), return an empty list
        return []

