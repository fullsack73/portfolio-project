"""
Automatic Model Selection and Fallback System

This module implements the tiered model selection strategy with graceful
fallback mechanisms for robust forecasting.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from src.forecasting_models import BaseForecaster, ModelFactory, ForecastResult
from src.model_validator import ModelValidator, ValidationResult, ModelRanking


# Configure logging
logger = logging.getLogger('model_selection')


@dataclass
class ModelTier:
    """Configuration for a model tier."""
    tier_name: str
    models: List[str]
    priority: int
    max_training_time: float
    performance_threshold: float
    fallback_on_failure: bool = True


@dataclass
class SelectionResult:
    """Result of model selection process."""
    ticker: str
    selected_model: str
    tier_used: str
    selection_score: float
    fallback_used: bool
    selection_time: float
    validation_results: List[ValidationResult]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ticker': self.ticker,
            'selected_model': self.selected_model,
            'tier_used': self.tier_used,
            'selection_score': self.selection_score,
            'fallback_used': self.fallback_used,
            'selection_time': self.selection_time,
            'timestamp': self.timestamp.isoformat()
        }


class TieredModelSelector:
    """
    Tiered model selection system with automatic fallback.
    
    Implements a three-tier strategy:
    - Tier 1: Advanced models (LSTM, SARIMAX, Gradient Boosting)
    - Tier 2: Classical models (ARIMA, Enhanced methods)
    - Tier 3: Fallback models (Simple methods, historical averages)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize tiered model selector.
        
        Args:
            config: Configuration dictionary for model tiers and thresholds
        """
        self.logger = logging.getLogger('model_selection.tiered')
        
        # Default configuration
        default_config = {
            'tiers': {
                'tier_1': {
                    'models': ['lstm', 'sarimax', 'xgboost', 'lightgbm', 'catboost'],
                    'priority': 1,
                    'max_training_time': 300.0,  # 5 minutes
                    'performance_threshold': 15.0,  # MAPE threshold
                    'fallback_on_failure': True
                },
                'tier_2': {
                    'models': ['arima'],
                    'priority': 2,
                    'max_training_time': 120.0,  # 2 minutes
                    'performance_threshold': 20.0,
                    'fallback_on_failure': True
                },
                'tier_3': {
                    'models': ['exponential_smoothing', 'linear_trend', 'historical_mean'],
                    'priority': 3,
                    'max_training_time': 30.0,  # 30 seconds
                    'performance_threshold': 30.0,
                    'fallback_on_failure': False
                }
            },
            'selection': {
                'min_data_points': 100,
                'validation_test_size': 0.2,
                'ensemble_threshold': 3,
                'parallel_validation': True
            }
        }
        
        # Merge with provided config
        self.config = default_config
        if config:
            self._merge_config(self.config, config)
        
        # Create model tiers
        self.tiers = self._create_model_tiers()
        
        # Initialize model validator
        self.validator = ModelValidator(
            cv_config={
                'test_size': self.config['selection']['validation_test_size'],
                'n_splits': 3  # Reduced for faster selection
            }
        )
        
        # Performance tracking
        self.selection_history: Dict[str, List[SelectionResult]] = {}
        
    def _merge_config(self, base: Dict, override: Dict) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _create_model_tiers(self) -> List[ModelTier]:
        """Create model tier objects from configuration."""
        tiers = []
        
        for tier_name, tier_config in self.config['tiers'].items():
            tier = ModelTier(
                tier_name=tier_name,
                models=tier_config['models'],
                priority=tier_config['priority'],
                max_training_time=tier_config['max_training_time'],
                performance_threshold=tier_config['performance_threshold'],
                fallback_on_failure=tier_config.get('fallback_on_failure', True)
            )
            tiers.append(tier)
        
        # Sort by priority
        tiers.sort(key=lambda x: x.priority)
        return tiers
    
    def select_model(self, 
                    data: pd.Series,
                    ticker: str,
                    exog: Optional[pd.DataFrame] = None,
                    force_tier: Optional[str] = None) -> SelectionResult:
        """
        Select the best model using tiered approach with fallback.
        
        Args:
            data: Time series data for model selection
            ticker: Ticker symbol
            exog: Optional exogenous variables
            force_tier: Force selection from specific tier (for testing)
            
        Returns:
            SelectionResult with selected model and metadata
            
        Raises:
            RuntimeError: If all tiers fail to produce a valid model
        """
        start_time = time.time()
        
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Invalid data provided for model selection")
        
        if len(data) < self.config['selection']['min_data_points']:
            raise ValueError(f"Insufficient data points: {len(data)} < "
                           f"{self.config['selection']['min_data_points']}")
        
        self.logger.info(f"Starting tiered model selection for {ticker}")
        
        # Determine which tiers to try
        tiers_to_try = self.tiers
        if force_tier:
            tiers_to_try = [tier for tier in self.tiers if tier.tier_name == force_tier]
            if not tiers_to_try:
                raise ValueError(f"Unknown tier: {force_tier}")
        
        last_error = None
        
        for tier in tiers_to_try:
            try:
                self.logger.info(f"Trying {tier.tier_name} models for {ticker}")
                
                result = self._try_tier(tier, data, ticker, exog)
                if result:
                    selection_time = time.time() - start_time
                    result.selection_time = selection_time
                    
                    # Store in history
                    if ticker not in self.selection_history:
                        self.selection_history[ticker] = []
                    self.selection_history[ticker].append(result)
                    
                    self.logger.info(f"Model selection completed for {ticker}: "
                                   f"{result.selected_model} from {result.tier_used} "
                                   f"(score: {result.selection_score:.4f})")
                    
                    return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"{tier.tier_name} failed for {ticker}: {e}")
                
                if not tier.fallback_on_failure:
                    # If fallback is disabled for this tier, re-raise the error
                    raise e
                
                continue
        
        # If we get here, all tiers failed
        error_msg = f"All model tiers failed for {ticker}"
        if last_error:
            error_msg += f". Last error: {last_error}"
        
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def _try_tier(self, 
                 tier: ModelTier,
                 data: pd.Series,
                 ticker: str,
                 exog: Optional[pd.DataFrame] = None) -> Optional[SelectionResult]:
        """
        Try to select a model from a specific tier.
        
        Args:
            tier: Model tier to try
            data: Time series data
            ticker: Ticker symbol
            exog: Optional exogenous variables
            
        Returns:
            SelectionResult if successful, None if tier fails
        """
        try:
            # Create models for this tier
            models = self._create_tier_models(tier)
            
            if not models:
                self.logger.warning(f"No valid models created for {tier.tier_name}")
                return None
            
            # Validate models with time limit
            validation_results = self._validate_tier_models(
                models, data, ticker, exog, tier.max_training_time
            )
            
            if not validation_results:
                self.logger.warning(f"No models passed validation in {tier.tier_name}")
                return None
            
            # Filter by performance threshold
            valid_results = [
                result for result in validation_results
                if result.mape <= tier.performance_threshold
            ]
            
            if not valid_results:
                self.logger.warning(f"No models met performance threshold "
                                  f"({tier.performance_threshold}%) in {tier.tier_name}")
                # Use best available model if no models meet threshold
                valid_results = [min(validation_results, key=lambda x: x.mape)]
            
            # Select best model
            best_result = min(valid_results, key=lambda x: x.mape)
            
            return SelectionResult(
                ticker=ticker,
                selected_model=best_result.model_name,
                tier_used=tier.tier_name,
                selection_score=best_result.mape,
                fallback_used=len(valid_results) == 1 and valid_results[0].mape > tier.performance_threshold,
                selection_time=0.0,  # Will be set by caller
                validation_results=validation_results,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Tier {tier.tier_name} failed: {e}")
            return None
    
    def _create_tier_models(self, tier: ModelTier) -> List[BaseForecaster]:
        """Create model instances for a tier."""
        models = []
        
        for model_name in tier.models:
            try:
                # Try to create model using factory
                if model_name in ModelFactory.get_available_models():
                    model = ModelFactory.create_model(model_name)
                    models.append(model)
                else:
                    # Handle fallback models that might not be in factory
                    model = self._create_fallback_model(model_name)
                    if model:
                        models.append(model)
                    
            except Exception as e:
                self.logger.warning(f"Failed to create model {model_name}: {e}")
                continue
        
        return models
    
    def _create_fallback_model(self, model_name: str) -> Optional[BaseForecaster]:
        """Create fallback models that might not be in the main factory."""
        try:
            if model_name == 'exponential_smoothing':
                return ExponentialSmoothingForecaster()
            elif model_name == 'linear_trend':
                return LinearTrendForecaster()
            elif model_name == 'historical_mean':
                return HistoricalMeanForecaster()
            else:
                self.logger.warning(f"Unknown fallback model: {model_name}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to create fallback model {model_name}: {e}")
            return None
    
    def _validate_tier_models(self, 
                            models: List[BaseForecaster],
                            data: pd.Series,
                            ticker: str,
                            exog: Optional[pd.DataFrame],
                            max_time: float) -> List[ValidationResult]:
        """Validate models with time constraints."""
        validation_results = []
        start_time = time.time()
        
        for model in models:
            # Check if we've exceeded time limit
            if time.time() - start_time > max_time:
                self.logger.warning(f"Time limit exceeded for tier validation ({max_time}s)")
                break
            
            try:
                # Set a timeout for individual model validation
                model_start = time.time()
                result = self.validator.cross_validator.validate_model(model, data, exog)
                result.ticker = ticker
                
                model_time = time.time() - model_start
                if model_time > max_time * 0.8:  # If single model takes >80% of tier time
                    self.logger.warning(f"Model {model.model_name} took {model_time:.2f}s "
                                      f"(>{max_time * 0.8:.2f}s threshold)")
                
                validation_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Validation failed for {model.model_name}: {e}")
                continue
        
        return validation_results
    
    def get_selection_history(self, ticker: str) -> List[SelectionResult]:
        """Get selection history for a ticker."""
        return self.selection_history.get(ticker, [])
    
    def get_tier_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for each tier."""
        stats = {}
        
        for tier in self.tiers:
            tier_selections = []
            for ticker_history in self.selection_history.values():
                tier_selections.extend([
                    result for result in ticker_history
                    if result.tier_used == tier.tier_name
                ])
            
            if tier_selections:
                stats[tier.tier_name] = {
                    'total_selections': len(tier_selections),
                    'avg_score': np.mean([r.selection_score for r in tier_selections]),
                    'avg_selection_time': np.mean([r.selection_time for r in tier_selections]),
                    'fallback_rate': np.mean([r.fallback_used for r in tier_selections]),
                    'success_rate': len(tier_selections) / len(self.selection_history) if self.selection_history else 0
                }
            else:
                stats[tier.tier_name] = {
                    'total_selections': 0,
                    'avg_score': 0,
                    'avg_selection_time': 0,
                    'fallback_rate': 0,
                    'success_rate': 0
                }
        
        return stats


class FallbackForecaster:
    """
    Fallback forecasting system for when all advanced models fail.
    
    Provides simple, reliable forecasting methods as a last resort.
    """
    
    def __init__(self):
        """Initialize fallback forecaster."""
        self.logger = logging.getLogger('model_selection.fallback')
        
    def forecast_with_fallback(self, 
                             data: pd.Series,
                             periods: int = 1,
                             method: str = 'historical_mean') -> float:
        """
        Generate forecast using fallback methods.
        
        Args:
            data: Time series data
            periods: Number of periods to forecast
            method: Fallback method to use
            
        Returns:
            Forecasted value
        """
        try:
            if method == 'historical_mean':
                return self._historical_mean_forecast(data, periods)
            elif method == 'exponential_smoothing':
                return self._exponential_smoothing_forecast(data, periods)
            elif method == 'linear_trend':
                return self._linear_trend_forecast(data, periods)
            else:
                self.logger.warning(f"Unknown fallback method: {method}, using historical mean")
                return self._historical_mean_forecast(data, periods)
                
        except Exception as e:
            self.logger.error(f"Fallback forecasting failed: {e}")
            # Ultimate fallback - return last known value
            return float(data.iloc[-1]) if not data.empty else 0.0
    
    def _historical_mean_forecast(self, data: pd.Series, periods: int) -> float:
        """Simple historical mean forecast."""
        # Use recent data for better relevance
        recent_data = data.tail(min(252, len(data)))  # Last year or available data
        return float(recent_data.mean())
    
    def _exponential_smoothing_forecast(self, data: pd.Series, periods: int) -> float:
        """Simple exponential smoothing forecast."""
        alpha = 0.3  # Smoothing parameter
        
        if len(data) < 2:
            return float(data.iloc[-1])
        
        # Initialize with first value
        smoothed = float(data.iloc[0])
        
        # Apply exponential smoothing
        for value in data.iloc[1:]:
            smoothed = alpha * float(value) + (1 - alpha) * smoothed
        
        return smoothed
    
    def _linear_trend_forecast(self, data: pd.Series, periods: int) -> float:
        """Simple linear trend forecast."""
        if len(data) < 10:
            return float(data.mean())
        
        # Use recent data for trend calculation
        recent_data = data.tail(min(60, len(data)))
        
        # Calculate simple linear trend
        x = np.arange(len(recent_data))
        y = recent_data.values
        
        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Forecast
        forecast_x = len(recent_data) + periods - 1
        forecast = slope * forecast_x + intercept
        
        return float(forecast)


# Simple fallback model implementations
class ExponentialSmoothingForecaster(BaseForecaster):
    """Simple exponential smoothing forecaster."""
    
    def __init__(self, model_name: str = "exponential_smoothing"):
        super().__init__(model_name)
        self.alpha = 0.3
        self.smoothed_value = None
        
    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        self._validate_data(data, min_points=10)
        self._training_data = data.copy()
        
        # Apply exponential smoothing
        self.smoothed_value = float(data.iloc[0])
        for value in data.iloc[1:]:
            self.smoothed_value = self.alpha * float(value) + (1 - self.alpha) * self.smoothed_value
        
        self.is_fitted = True
        
    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.smoothed_value
        
    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        # Simple validation implementation
        split_point = int(len(data) * (1 - test_size))
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        self.fit(train_data)
        predictions = [self.predict(1) for _ in range(len(test_data))]
        actuals = test_data.values
        
        mse = np.mean((actuals - predictions) ** 2)
        mae = np.mean(np.abs(actuals - predictions))
        mape = np.mean(np.abs((actuals - predictions) / np.where(actuals != 0, actuals, 1))) * 100
        
        return {'mse': float(mse), 'mae': float(mae), 'mape': float(mape), 'rmse': float(np.sqrt(mse))}


class LinearTrendForecaster(BaseForecaster):
    """Simple linear trend forecaster."""
    
    def __init__(self, model_name: str = "linear_trend"):
        super().__init__(model_name)
        self.slope = 0
        self.intercept = 0
        
    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        self._validate_data(data, min_points=10)
        self._training_data = data.copy()
        
        # Calculate linear trend
        x = np.arange(len(data))
        y = data.values
        
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        self.intercept = (sum_y - self.slope * sum_x) / n
        
        self.is_fitted = True
        
    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        forecast_x = len(self._training_data) + periods - 1
        return float(self.slope * forecast_x + self.intercept)
        
    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        # Simple validation implementation
        split_point = int(len(data) * (1 - test_size))
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        self.fit(train_data)
        predictions = [self.predict(i+1) for i in range(len(test_data))]
        actuals = test_data.values
        
        mse = np.mean((actuals - predictions) ** 2)
        mae = np.mean(np.abs(actuals - predictions))
        mape = np.mean(np.abs((actuals - predictions) / np.where(actuals != 0, actuals, 1))) * 100
        
        return {'mse': float(mse), 'mae': float(mae), 'mape': float(mape), 'rmse': float(np.sqrt(mse))}


class HistoricalMeanForecaster(BaseForecaster):
    """Historical mean forecaster."""
    
    def __init__(self, model_name: str = "historical_mean"):
        super().__init__(model_name)
        self.mean_value = 0
        
    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        self._validate_data(data, min_points=5)
        self._training_data = data.copy()
        
        # Use recent data for better relevance
        recent_data = data.tail(min(252, len(data)))
        self.mean_value = float(recent_data.mean())
        
        self.is_fitted = True
        
    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.mean_value
        
    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        # Simple validation implementation
        split_point = int(len(data) * (1 - test_size))
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        self.fit(train_data)
        predictions = [self.predict(1) for _ in range(len(test_data))]
        actuals = test_data.values
        
        mse = np.mean((actuals - predictions) ** 2)
        mae = np.mean(np.abs(actuals - predictions))
        mape = np.mean(np.abs((actuals - predictions) / np.where(actuals != 0, actuals, 1))) * 100
        
        return {'mse': float(mse), 'mae': float(mae), 'mape': float(mape), 'rmse': float(np.sqrt(mse))}