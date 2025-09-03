"""
Feature Engineering Utilities for Gradient Boosting Models

This module provides technical indicator calculations, lagged features,
and market regime indicators for use with gradient boosting forecasting models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger('forecasting.feature_engineering')


class TechnicalIndicators:
    """
    Technical indicator calculations for financial time series.
    
    This class provides methods to calculate various technical indicators
    commonly used in financial analysis and machine learning models.
    """
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Price series (typically closing prices)
            window: Period for RSI calculation
            
        Returns:
            RSI values as pandas Series
        """
        if len(data) < window + 1:
            logger.warning(f"Insufficient data for RSI calculation. Need {window + 1}, got {len(data)}")
            return pd.Series(index=data.index, dtype=float)
        
        # Calculate price changes
        delta = data.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=window, min_periods=window).mean()
        avg_losses = losses.rolling(window=window, min_periods=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price series (typically closing prices)
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary containing MACD line, signal line, and histogram
        """
        if len(data) < slow + signal:
            logger.warning(f"Insufficient data for MACD calculation. Need {slow + signal}, got {len(data)}")
            empty_series = pd.Series(index=data.index, dtype=float)
            return {'macd': empty_series, 'signal': empty_series, 'histogram': empty_series}
        
        # Calculate EMAs
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price series (typically closing prices)
            window: Period for moving average and standard deviation
            num_std: Number of standard deviations for bands
            
        Returns:
            Dictionary containing upper band, lower band, and middle band (SMA)
        """
        if len(data) < window:
            logger.warning(f"Insufficient data for Bollinger Bands calculation. Need {window}, got {len(data)}")
            empty_series = pd.Series(index=data.index, dtype=float)
            return {'upper': empty_series, 'middle': empty_series, 'lower': empty_series}
        
        # Calculate middle band (SMA)
        middle_band = data.rolling(window=window).mean()
        
        # Calculate standard deviation
        std_dev = data.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_window: Period for %K calculation
            d_window: Period for %D smoothing
            
        Returns:
            Dictionary containing %K and %D values
        """
        if len(close) < k_window + d_window:
            logger.warning(f"Insufficient data for Stochastic calculation. Need {k_window + d_window}, got {len(close)}")
            empty_series = pd.Series(index=close.index, dtype=float)
            return {'%K': empty_series, '%D': empty_series}
        
        # Calculate %K
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (smoothed %K)
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            '%K': k_percent,
            '%D': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Period for calculation
            
        Returns:
            Williams %R values as pandas Series
        """
        if len(close) < window:
            logger.warning(f"Insufficient data for Williams %R calculation. Need {window}, got {len(close)}")
            return pd.Series(index=close.index, dtype=float)
        
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r


class FeatureEngineer:
    """
    Main feature engineering class for creating ML-ready features from financial data.
    
    This class combines technical indicators, lagged features, and market regime
    indicators to create a comprehensive feature set for gradient boosting models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Configuration dictionary for feature parameters
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger('forecasting.feature_engineering')
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for feature engineering."""
        return {
            'lagged_returns': {
                'lags': [1, 2, 3, 5, 10, 20],
                'include_log_returns': True
            },
            'moving_averages': {
                'windows': [5, 10, 20, 50],
                'include_ratios': True
            },
            'volatility': {
                'windows': [5, 10, 20],
                'include_garch': False  # Set to True if GARCH is available
            },
            'technical_indicators': {
                'rsi': {'window': 14},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger': {'window': 20, 'num_std': 2.0},
                'stochastic': {'k_window': 14, 'd_window': 3},
                'williams_r': {'window': 14}
            },
            'market_regime': {
                'volatility_threshold': 0.02,
                'trend_window': 20,
                'regime_window': 60
            }
        }
    
    def create_features(self, data: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """
        Create comprehensive feature set from price data.
        
        Args:
            data: DataFrame with OHLCV data
            target_col: Column name for target variable (typically 'close')
            
        Returns:
            DataFrame with engineered features
        """
        if data.empty or target_col not in data.columns:
            raise ValueError(f"Data is empty or missing target column '{target_col}'")
        
        self.logger.info(f"Creating features for {len(data)} data points")
        
        # Initialize feature DataFrame
        features = pd.DataFrame(index=data.index)
        
        # Add original price data
        features['price'] = data[target_col]
        
        # Calculate returns
        features['returns'] = data[target_col].pct_change(fill_method=None)
        if self.config['lagged_returns']['include_log_returns']:
            features['log_returns'] = np.log(data[target_col] / data[target_col].shift(1))
        
        # Add lagged return features
        features = self._add_lagged_features(features, data[target_col])
        
        # Add moving average features
        features = self._add_moving_average_features(features, data[target_col])
        
        # Add volatility features
        features = self._add_volatility_features(features, data[target_col])
        
        # Add technical indicators
        features = self._add_technical_indicators(features, data)
        
        # Add market regime indicators
        features = self._add_market_regime_features(features, data[target_col])
        
        # Add volume features if available
        if 'volume' in data.columns:
            features = self._add_volume_features(features, data)
        
        # Remove rows with excessive NaN values (keep some data for short series)
        initial_length = len(features)
        
        # For short series, be more lenient with NaN values
        if len(features) < 100:
            # Keep rows that have at least 50% non-NaN values
            min_valid_features = max(1, len(features.columns) // 2)
            features = features.dropna(thresh=min_valid_features)
        else:
            # For longer series, drop rows with any NaN in key features
            key_features = ['price', 'returns']
            features = features.dropna(subset=[col for col in key_features if col in features.columns])
        
        dropped_rows = initial_length - len(features)
        
        if dropped_rows > 0:
            self.logger.info(f"Dropped {dropped_rows} rows with NaN values")
        
        self.logger.info(f"Created {len(features.columns)} features for {len(features)} data points")
        
        return features
    
    def _add_lagged_features(self, features: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
        """Add lagged return and price features."""
        lags = self.config['lagged_returns']['lags']
        
        for lag in lags:
            # Lagged returns
            features[f'return_lag_{lag}'] = features['returns'].shift(lag)
            
            # Lagged log returns if enabled
            if self.config['lagged_returns']['include_log_returns'] and 'log_returns' in features.columns:
                features[f'log_return_lag_{lag}'] = features['log_returns'].shift(lag)
            
            # Lagged price ratios
            features[f'price_ratio_lag_{lag}'] = price_series / price_series.shift(lag) - 1
        
        return features
    
    def _add_moving_average_features(self, features: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
        """Add moving average features."""
        windows = self.config['moving_averages']['windows']
        
        for window in windows:
            # Simple moving average
            ma = price_series.rolling(window=window).mean()
            features[f'sma_{window}'] = ma
            
            # Price to MA ratio
            if self.config['moving_averages']['include_ratios']:
                features[f'price_to_sma_{window}'] = price_series / ma - 1
            
            # Exponential moving average
            ema = price_series.ewm(span=window).mean()
            features[f'ema_{window}'] = ema
            
            # Price to EMA ratio
            if self.config['moving_averages']['include_ratios']:
                features[f'price_to_ema_{window}'] = price_series / ema - 1
        
        return features
    
    def _add_volatility_features(self, features: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
        """Add volatility-based features."""
        windows = self.config['volatility']['windows']
        
        for window in windows:
            # Rolling standard deviation of returns
            if 'returns' in features.columns:
                features[f'volatility_{window}'] = features['returns'].rolling(window=window).std()
            
            # Rolling standard deviation of log returns
            if 'log_returns' in features.columns:
                features[f'log_volatility_{window}'] = features['log_returns'].rolling(window=window).std()
            
            # Price range volatility (high-low based if available)
            features[f'price_volatility_{window}'] = price_series.rolling(window=window).std()
        
        # Add realized volatility (if we have intraday-like data)
        if 'returns' in features.columns:
            features['realized_volatility'] = features['returns'].rolling(window=20).apply(
                lambda x: np.sqrt(np.sum(x**2))
            )
        
        return features
    
    def _add_technical_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        price_col = 'close' if 'close' in data.columns else data.columns[0]
        price_series = data[price_col]
        
        # RSI
        if 'rsi' in self.config['technical_indicators']:
            rsi_config = self.config['technical_indicators']['rsi']
            features['rsi'] = TechnicalIndicators.rsi(price_series, **rsi_config)
        
        # MACD
        if 'macd' in self.config['technical_indicators']:
            macd_config = self.config['technical_indicators']['macd']
            macd_data = TechnicalIndicators.macd(price_series, **macd_config)
            features['macd'] = macd_data['macd']
            features['macd_signal'] = macd_data['signal']
            features['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        if 'bollinger' in self.config['technical_indicators']:
            bb_config = self.config['technical_indicators']['bollinger']
            bb_data = TechnicalIndicators.bollinger_bands(price_series, **bb_config)
            features['bb_upper'] = bb_data['upper']
            features['bb_middle'] = bb_data['middle']
            features['bb_lower'] = bb_data['lower']
            
            # Bollinger Band position
            features['bb_position'] = (price_series - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
            features['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        
        # Add other indicators if OHLC data is available
        if all(col in data.columns for col in ['high', 'low', 'close']):
            # Stochastic Oscillator
            if 'stochastic' in self.config['technical_indicators']:
                stoch_config = self.config['technical_indicators']['stochastic']
                stoch_data = TechnicalIndicators.stochastic_oscillator(
                    data['high'], data['low'], data['close'], **stoch_config
                )
                features['stoch_k'] = stoch_data['%K']
                features['stoch_d'] = stoch_data['%D']
            
            # Williams %R
            if 'williams_r' in self.config['technical_indicators']:
                wr_config = self.config['technical_indicators']['williams_r']
                features['williams_r'] = TechnicalIndicators.williams_r(
                    data['high'], data['low'], data['close'], **wr_config
                )
        
        return features
    
    def _add_market_regime_features(self, features: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
        """Add market regime indicators."""
        regime_config = self.config['market_regime']
        
        # Volatility regime
        if 'returns' in features.columns:
            vol_threshold = regime_config['volatility_threshold']
            rolling_vol = features['returns'].rolling(window=20).std()
            features['high_volatility_regime'] = (rolling_vol > vol_threshold).astype(int)
        
        # Trend regime
        trend_window = regime_config['trend_window']
        sma_trend = price_series.rolling(window=trend_window).mean()
        features['uptrend'] = (price_series > sma_trend).astype(int)
        features['trend_strength'] = (price_series / sma_trend - 1)
        
        # Market regime classification
        regime_window = regime_config['regime_window']
        if 'returns' in features.columns:
            # Calculate rolling statistics for regime classification
            rolling_mean = features['returns'].rolling(window=regime_window).mean()
            rolling_std = features['returns'].rolling(window=regime_window).std()
            
            # Define regimes based on return patterns
            features['bull_market'] = ((rolling_mean > 0) & (rolling_std < rolling_std.median())).astype(int)
            features['bear_market'] = ((rolling_mean < 0) & (rolling_std < rolling_std.median())).astype(int)
            features['volatile_market'] = (rolling_std > rolling_std.quantile(0.75)).astype(int)
        
        return features
    
    def _add_volume_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features if volume data is available."""
        if 'volume' not in data.columns:
            return features
        
        volume = data['volume']
        price = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        
        # Volume moving averages
        for window in [5, 10, 20]:
            features[f'volume_sma_{window}'] = volume.rolling(window=window).mean()
            features[f'volume_ratio_{window}'] = volume / features[f'volume_sma_{window}']
        
        # Price-volume features
        features['price_volume'] = price * volume
        features['vwap_5'] = (features['price_volume'].rolling(window=5).sum() / 
                             volume.rolling(window=5).sum())
        
        # Volume rate of change
        features['volume_roc'] = volume.pct_change()
        
        # On-Balance Volume (OBV)
        if 'returns' in features.columns:
            obv_direction = np.where(features['returns'] > 0, volume, 
                                   np.where(features['returns'] < 0, -volume, 0))
            features['obv'] = obv_direction.cumsum()
        
        return features
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups for importance analysis.
        
        Returns:
            Dictionary mapping feature group names to lists of feature patterns
        """
        return {
            'lagged_returns': ['return_lag_', 'log_return_lag_', 'price_ratio_lag_'],
            'moving_averages': ['sma_', 'ema_', 'price_to_sma_', 'price_to_ema_'],
            'volatility': ['volatility_', 'log_volatility_', 'price_volatility_', 'realized_volatility'],
            'technical_indicators': ['rsi', 'macd', 'bb_', 'stoch_', 'williams_r'],
            'market_regime': ['high_volatility_regime', 'uptrend', 'trend_strength', 
                            'bull_market', 'bear_market', 'volatile_market'],
            'volume': ['volume_', 'vwap_', 'obv', 'price_volume']
        }


def create_features_for_ticker(data: pd.DataFrame, ticker: str, 
                              config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function to create features for a single ticker.
    
    Args:
        data: OHLCV DataFrame for the ticker
        ticker: Ticker symbol (for logging)
        config: Optional configuration for feature engineering
        
    Returns:
        DataFrame with engineered features
    """
    logger.info(f"Creating features for ticker: {ticker}")
    
    try:
        engineer = FeatureEngineer(config)
        features = engineer.create_features(data)
        
        # Add ticker column for identification
        features['ticker'] = ticker
        
        logger.info(f"Successfully created {len(features.columns)} features for {ticker}")
        return features
        
    except Exception as e:
        logger.error(f"Failed to create features for {ticker}: {str(e)}")
        raise


def validate_feature_data(features: pd.DataFrame, min_samples: int = 100) -> bool:
    """
    Validate feature data for ML model training.
    
    Args:
        features: DataFrame with engineered features
        min_samples: Minimum number of samples required
        
    Returns:
        True if data is valid for training
        
    Raises:
        ValueError: If data validation fails
    """
    if features.empty:
        raise ValueError("Feature data is empty")
    
    if len(features) < min_samples:
        raise ValueError(f"Insufficient samples. Required: {min_samples}, Got: {len(features)}")
    
    # Check for excessive NaN values
    nan_ratio = features.isnull().sum().sum() / (len(features) * len(features.columns))
    if nan_ratio > 0.1:  # More than 10% NaN values
        logger.warning(f"High NaN ratio in features: {nan_ratio:.2%}")
    
    # Check for infinite values
    inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        raise ValueError(f"Found {inf_count} infinite values in features")
    
    # Check for constant features
    constant_features = []
    for col in features.select_dtypes(include=[np.number]).columns:
        if features[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        logger.warning(f"Found constant features: {constant_features}")
    
    logger.info(f"Feature validation passed. Shape: {features.shape}, NaN ratio: {nan_ratio:.2%}")
    return True