import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

class SharedFeatureExtractor:
    """
    Extracts market-wide and individual features for a batch of tickers.
    """

    def __init__(self, cache_size: int = 100):
        self.feature_cache: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        self.cache_size = cache_size

    def extract_features(self, data: pd.DataFrame, tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts individual and shared features for a batch of tickers.

        Args:
            data (pd.DataFrame): DataFrame with historical price data, with tickers as columns.
            tickers (List[str]): The list of tickers to process.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                - individual_features: Features specific to each ticker.
                - shared_features: Features shared across all tickers in the batch.
        """
        data_hash = pd.util.hash_pandas_object(data).sum()
        if data_hash in self.feature_cache:
            return self.feature_cache[data_hash]

        individual_features = self._extract_individual_features(data, tickers)
        shared_features = self._extract_market_features(data)

        if len(self.feature_cache) >= self.cache_size:
            self.feature_cache.pop(next(iter(self.feature_cache)))  # Remove oldest item

        self.feature_cache[data_hash] = (individual_features, shared_features)

        return individual_features, shared_features

    def _extract_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts market-wide features using vectorized operations.

        Args:
            data (pd.DataFrame): DataFrame with historical price data.

        Returns:
            pd.DataFrame: DataFrame containing market-wide features.
        """
        market_features = pd.DataFrame(index=data.index)
        
        # Market return (equal-weighted)
        market_return = data.pct_change().mean(axis=1)
        market_features['market_return'] = market_return
        
        # Market volatility (30-day rolling std of market return)
        market_features['market_volatility_30d'] = market_return.rolling(window=30).std()
        
        return market_features.fillna(0)

    def _extract_individual_features(self, data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """
        Extracts individual features for each ticker using vectorized operations.

        Args:
            data (pd.DataFrame): DataFrame with historical price data for the tickers.
            tickers (List[str]): The list of tickers to process.

        Returns:
            pd.DataFrame: A multi-index DataFrame with features for each ticker.
        """
        all_features = []
        
        returns = data.pct_change()
        
        for ticker in tickers:
            ticker_features = pd.DataFrame(index=data.index)
            ticker_features['ticker'] = ticker
            
            # Momentum
            ticker_features['momentum_1m'] = returns[ticker].rolling(window=21).sum()
            ticker_features['momentum_3m'] = returns[ticker].rolling(window=63).sum()
            
            # Volatility
            ticker_features['volatility_21d'] = returns[ticker].rolling(window=21).std()
            ticker_features['volatility_63d'] = returns[ticker].rolling(window=63).std()
            
            # Moving Averages
            ticker_features['ma_21d'] = data[ticker].rolling(window=21).mean()
            ticker_features['ma_63d'] = data[ticker].rolling(window=63).mean()
            
            # RSI (14-day)
            delta = data[ticker].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            ticker_features['rsi_14d'] = 100 - (100 / (1 + rs))
            
            all_features.append(ticker_features)
            
        if not all_features:
            return pd.DataFrame()

        features_df = pd.concat(all_features)
        features_df = features_df.set_index([features_df.index, 'ticker'])
        
        return features_df.fillna(0)
