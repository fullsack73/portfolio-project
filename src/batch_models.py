import pandas as pd
import xgboost as xgb
from typing import List, Dict, Optional

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from .forecasting_models import BaseForecaster

class BatchXGBoostForecaster(BaseForecaster):
    """
    XGBoost model with multi-output regression for batch processing.
    Primary model for speed and accuracy balance.
    """

    def __init__(self, model_name: str = "batch_xgboost", **kwargs):
        super().__init__(model_name)
        self.config = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'n_jobs': -1,
        }
        self.config.update(kwargs)
        self._model = xgb.XGBRegressor(**self.config)
        self.tickers = []

    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        raise NotImplementedError("This is a batch forecaster. Use fit_batch instead.")

    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        raise NotImplementedError("This is a batch forecaster. Use predict_batch instead.")

    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        raise NotImplementedError("This is a batch forecaster and does not support single series validation.")

    def fit_batch(self, individual_features: pd.DataFrame, shared_features: pd.DataFrame, tickers: List[str], returns: pd.DataFrame) -> None:
        """
        Fit multi-output XGBoost model with hardcoded optimal parameters.

        Args:
            individual_features (pd.DataFrame): Multi-index DataFrame of individual ticker features.
            shared_features (pd.DataFrame): DataFrame of shared market features.
            tickers (List[str]): List of tickers in the batch.
            returns (pd.DataFrame): DataFrame of returns for each ticker, used as the target.
        """
        self.tickers = tickers
        
        # Combine features
        X = individual_features.unstack(level='ticker')
        
        # Align shared features
        shared_features_aligned = shared_features.reindex(X.index).ffill().bfill()
        
        # Merge shared features for each ticker
        for ticker in tickers:
            for col in shared_features_aligned.columns:
                X[(ticker, col)] = shared_features_aligned[col]
        
        X.columns = ['_'.join(map(str, col)) for col in X.columns]
        
        # Align target variable y
        y = returns[tickers].reindex(X.index).fillna(0)
        
        self._model.fit(X, y)
        self.is_fitted = True

    def predict_batch(self, periods: int, individual_features: pd.DataFrame, shared_features: pd.DataFrame) -> Dict[str, float]:
        """
        Generate batch predictions using multi-output regression.

        Args:
            periods (int): Number of periods to forecast (used to select the latest features).
            individual_features (pd.DataFrame): Features for the prediction period.
            shared_features (pd.DataFrame): Shared features for the prediction period.

        Returns:
            Dict[str, float]: Dictionary of ticker to predicted return.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit_batch first.")

        # Prepare the latest features for prediction
        latest_features = individual_features.unstack(level='ticker').tail(periods)
        
        # Align and merge shared features
        shared_features_aligned = shared_features.reindex(latest_features.index).ffill().bfill()
        for ticker in self.tickers:
            for col in shared_features_aligned.columns:
                latest_features[(ticker, col)] = shared_features_aligned[col]

        latest_features.columns = ['_'.join(map(str,col)) for col in latest_features.columns]

        predictions = self._model.predict(latest_features)
        
        # Extract the last prediction for each ticker
        last_predictions = predictions[-1]
        
        return {ticker: float(pred) for ticker, pred in zip(self.tickers, last_predictions)}


class BatchLinearForecaster(BaseForecaster):
    """
    Simple linear regression model for batch processing.
    Fast fallback model when XGBoost fails.
    """

    def __init__(self, model_name: str = "batch_linear", **kwargs):
        super().__init__(model_name)
        self.config = kwargs
        self._model = MultiOutputRegressor(LinearRegression(n_jobs=-1))
        self.tickers = []

    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        raise NotImplementedError("This is a batch forecaster. Use fit_batch instead.")

    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        raise NotImplementedError("This is a batch forecaster. Use predict_batch instead.")

    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        raise NotImplementedError("This is a batch forecaster and does not support single series validation.")

    def fit_batch(self, individual_features: pd.DataFrame, shared_features: pd.DataFrame, tickers: List[str], returns: pd.DataFrame) -> None:
        """
        Fit multi-output linear regression model.

        Args:
            individual_features (pd.DataFrame): Multi-index DataFrame of individual ticker features.
            shared_features (pd.DataFrame): DataFrame of shared market features.
            tickers (List[str]): List of tickers in the batch.
            returns (pd.DataFrame): DataFrame of returns for each ticker, used as the target.
        """
        self.tickers = tickers
        
        # Combine features
        X = individual_features.unstack(level='ticker')
        
        # Align shared features
        shared_features_aligned = shared_features.reindex(X.index).ffill().bfill()
        
        # Merge shared features for each ticker
        for ticker in tickers:
            for col in shared_features_aligned.columns:
                X[(ticker, col)] = shared_features_aligned[col]
        
        X.columns = ['_'.join(map(str, col)) for col in X.columns]
        
        # Align target variable y
        y = returns[tickers].reindex(X.index).fillna(0)
        
        self._model.fit(X, y)
        self.is_fitted = True

    def predict_batch(self, periods: int, individual_features: pd.DataFrame, shared_features: pd.DataFrame) -> Dict[str, float]:
        """
        Generate batch predictions using linear regression.

        Args:
            periods (int): Number of periods to forecast (used to select the latest features).
            individual_features (pd.DataFrame): Features for the prediction period.
            shared_features (pd.DataFrame): Shared features for the prediction period.

        Returns:
            Dict[str, float]: Dictionary of ticker to predicted return.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit_batch first.")

        # Prepare the latest features for prediction
        latest_features = individual_features.unstack(level='ticker').tail(periods)
        
        # Align and merge shared features
        shared_features_aligned = shared_features.reindex(latest_features.index).ffill().bfill()
        for ticker in self.tickers:
            for col in shared_features_aligned.columns:
                latest_features[(ticker, col)] = shared_features_aligned[col]

        latest_features.columns = ['_'.join(map(str, col)) for col in latest_features.columns]

        predictions = self._model.predict(latest_features)
        
        # Extract the last prediction for each ticker
        last_predictions = predictions[-1]
        
        return {ticker: float(pred) for ticker, pred in zip(self.tickers, last_predictions)}
