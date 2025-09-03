"""
Advanced Forecasting Models Module

This module provides the base interfaces and abstract classes for implementing
various forecasting models including ARIMA, LSTM, SARIMAX, and gradient boosting models.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List
import pandas as pd
import numpy as np


# Configure logging for forecasting module
def setup_forecasting_logger():
    """Set up logging configuration for the forecasting module."""
    logger = logging.getLogger('forecasting')
    
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    
    return logger


# Initialize logger
forecasting_logger = setup_forecasting_logger()


@dataclass
class ModelPerformance:
    """Data class for tracking model performance metrics."""
    ticker: str
    model_name: str
    mse: float
    mae: float
    mape: float
    training_time: float
    prediction_time: float
    timestamp: datetime
    data_points: int


@dataclass
class ForecastResult:
    """Data class for forecast results."""
    ticker: str
    expected_return: float
    confidence_interval: Tuple[float, float]
    model_used: str
    ensemble_weights: Dict[str, float]
    validation_score: float
    timestamp: datetime


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    
    This class defines the common interface that all forecasting models must implement,
    ensuring consistency across different model types (ARIMA, LSTM, SARIMAX, etc.).
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the base forecaster.
        
        Args:
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.is_fitted = False
        self.logger = logging.getLogger(f'forecasting.{model_name}')
        self._model = None
        self._training_data = None
        
    @abstractmethod
    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        Fit the forecasting model to the provided data.
        
        Args:
            data: Time series data for training
            exog: Optional exogenous variables for models that support them
            
        Raises:
            ValueError: If data is insufficient or invalid
            RuntimeError: If model training fails
        """
        pass
    
    @abstractmethod
    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        """
        Generate forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            exog: Optional exogenous variables for prediction
            
        Returns:
            Expected return for the forecast period
            
        Raises:
            RuntimeError: If model is not fitted or prediction fails
        """
        pass
    
    @abstractmethod
    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        """
        Perform model validation using time series cross-validation.
        
        Args:
            data: Time series data for validation
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing validation metrics (MSE, MAE, MAPE)
            
        Raises:
            ValueError: If validation parameters are invalid
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model configuration and status.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'training_data_points': len(self._training_data) if self._training_data is not None else 0,
            'model_type': self.__class__.__name__
        }
    
    def _validate_data(self, data: pd.Series, min_points: int = 100) -> None:
        """
        Validate input data for training or prediction.
        
        Args:
            data: Time series data to validate
            min_points: Minimum number of data points required
            
        Raises:
            ValueError: If data is invalid or insufficient
        """
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty")
        
        if len(data) < min_points:
            raise ValueError(f"Insufficient data points. Required: {min_points}, Got: {len(data)}")
        
        if data.isnull().any():
            self.logger.warning("Data contains null values, consider preprocessing")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Data index is not DatetimeIndex, time series operations may be affected")
    
    def _log_performance(self, performance: ModelPerformance) -> None:
        """
        Log model performance metrics.
        
        Args:
            performance: ModelPerformance object containing metrics
        """
        self.logger.info(
            f"Model: {performance.model_name}, Ticker: {performance.ticker}, "
            f"MAPE: {performance.mape:.4f}, MAE: {performance.mae:.4f}, "
            f"Training Time: {performance.training_time:.2f}s"
        )


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble forecasting model that combines predictions from multiple models.
    
    This implementation supports multiple combination methods including simple averaging,
    performance-weighted averaging, and stacking approaches for improved forecast accuracy.
    """
    
    def __init__(self, model_name: str = "ensemble", **kwargs):
        """
        Initialize ensemble forecaster.
        
        Args:
            model_name: Name identifier for the ensemble
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name)
        
        # Get configuration
        try:
            from forecasting_config import get_model_config
            config = get_model_config('ensemble')
        except ImportError:
            # Fallback if config module not available
            config = None
        
        if config:
            self.config = config.params
        else:
            # Default configuration
            self.config = {
                'combination_method': 'performance_weighted',  # 'simple', 'performance_weighted', 'stacking'
                'min_models': 3,
                'max_models': 5,
                'diversity_threshold': 0.7,  # Correlation threshold for diversity
                'performance_window': 30,  # Days to consider for recent performance
                'stacking_model': 'linear',  # 'linear', 'ridge', 'lasso'
                'weight_decay': 0.95,  # Decay factor for historical performance
                'min_weight': 0.05,  # Minimum weight for any model
                'validation_method': 'walk_forward'
            }
        
        # Override with any provided kwargs
        self.config.update(kwargs)
        
        # Ensemble state
        self._base_models = {}  # Dict[str, BaseForecaster]
        self._model_weights = {}  # Dict[str, float]
        self._model_performances = {}  # Dict[str, List[float]]
        self._stacking_model = None
        self._ensemble_validation_scores = []
        self._last_predictions = {}  # Store last predictions for dynamic weighting
        
    def add_model(self, model: BaseForecaster, weight: Optional[float] = None) -> None:
        """
        Add a base model to the ensemble.
        
        Args:
            model: Fitted forecasting model to add
            weight: Optional initial weight (will be calculated if not provided)
            
        Raises:
            ValueError: If model is not fitted or invalid
        """
        if not isinstance(model, BaseForecaster):
            raise ValueError("Model must inherit from BaseForecaster")
        
        if not model.is_fitted:
            raise ValueError(f"Model {model.model_name} must be fitted before adding to ensemble")
        
        # Check for diversity if we have other models
        if len(self._base_models) > 0 and self.config.get('diversity_threshold'):
            if not self._check_model_diversity(model):
                self.logger.warning(f"Model {model.model_name} may be too similar to existing models")
        
        self._base_models[model.model_name] = model
        
        # Set initial weight
        if weight is not None:
            self._model_weights[model.model_name] = max(weight, self.config.get('min_weight', 0.05))
        else:
            # Equal weight initially, will be updated during fitting
            self._model_weights[model.model_name] = 1.0 / len(self._base_models)
        
        # Normalize all weights to sum to 1
        total_weight = sum(self._model_weights.values())
        if total_weight > 0:
            for model_name in self._model_weights:
                self._model_weights[model_name] /= total_weight
        
        # Initialize performance tracking
        self._model_performances[model.model_name] = []
        
        self.logger.info(f"Added model {model.model_name} to ensemble (weight: {self._model_weights[model.model_name]:.3f})")
    
    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        Fit the ensemble by calculating optimal weights and training stacking model if needed.
        
        Args:
            data: Time series data for training
            exog: Optional exogenous variables
            
        Raises:
            ValueError: If insufficient models or data
            RuntimeError: If ensemble fitting fails
        """
        import time
        start_time = time.time()
        
        try:
            # Validate we have enough models
            if len(self._base_models) < self.config.get('min_models', 3):
                raise ValueError(f"Ensemble requires at least {self.config.get('min_models', 3)} models, got {len(self._base_models)}")
            
            # Validate input data
            self._validate_data(data, min_points=self.config.get('min_data_points', 200))
            
            # Store training data
            self._training_data = data.copy()
            
            # Calculate model weights based on validation performance
            self._calculate_model_weights(data, exog)
            
            # Train stacking model if using stacking method
            if self.config.get('combination_method') == 'stacking':
                self._train_stacking_model(data, exog)
            
            # Mark as fitted
            self.is_fitted = True
            
            training_time = time.time() - start_time
            self.logger.info(
                f"Ensemble fitted successfully in {training_time:.2f}s. "
                f"Models: {list(self._base_models.keys())}, "
                f"Method: {self.config.get('combination_method')}"
            )
            
        except ValueError as e:
            self.logger.error(f"Ensemble validation failed: {str(e)}")
            raise e
        except Exception as e:
            self.logger.error(f"Failed to fit ensemble: {str(e)}")
            raise RuntimeError(f"Ensemble fitting failed: {str(e)}")
    
    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        """
        Generate ensemble forecast by combining predictions from base models.
        
        Args:
            periods: Number of periods to forecast
            exog: Optional exogenous variables
            
        Returns:
            Combined expected return from ensemble
            
        Raises:
            RuntimeError: If ensemble is not fitted or prediction fails
        """
        import time
        start_time = time.time()
        
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before making predictions")
        
        if not self._base_models:
            raise RuntimeError("No base models available for prediction")
        
        try:
            # Get predictions from all base models
            model_predictions = {}
            successful_models = []
            
            for model_name, model in self._base_models.items():
                try:
                    prediction = model.predict(periods, exog)
                    model_predictions[model_name] = prediction
                    successful_models.append(model_name)
                    
                except Exception as e:
                    self.logger.warning(f"Model {model_name} prediction failed: {e}")
                    continue
            
            if not model_predictions:
                raise RuntimeError("All base models failed to generate predictions")
            
            # Store predictions for dynamic weighting
            self._last_predictions.update(model_predictions)
            
            # Combine predictions based on method
            if self.config.get('combination_method') == 'simple':
                ensemble_prediction = self._simple_average(model_predictions)
            elif self.config.get('combination_method') == 'performance_weighted':
                ensemble_prediction = self._weighted_average(model_predictions, successful_models)
            elif self.config.get('combination_method') == 'stacking':
                ensemble_prediction = self._stacking_prediction(model_predictions, exog)
            else:
                # Default to performance weighted
                ensemble_prediction = self._weighted_average(model_predictions, successful_models)
            
            prediction_time = time.time() - start_time
            self.logger.debug(f"Ensemble prediction completed in {prediction_time:.4f}s")
            
            return float(ensemble_prediction)
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {str(e)}")
            raise RuntimeError(f"Ensemble prediction failed: {str(e)}")
    
    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        """
        Perform ensemble validation using time series cross-validation.
        
        Args:
            data: Time series data for validation
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing validation metrics
        """
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        
        try:
            # Split data for validation
            split_point = int(len(data) * (1 - test_size))
            train_data = data[:split_point]
            test_data = data[split_point:]
            
            if len(train_data) < 100 or len(test_data) < 20:
                raise ValueError("Insufficient data for validation split")
            
            # Create temporary ensemble for validation
            temp_ensemble = EnsembleForecaster(model_name=f"{self.model_name}_validation")
            temp_ensemble.config = self.config.copy()
            
            # Add fitted base models to temp ensemble
            for model_name, model in self._base_models.items():
                if model.is_fitted:
                    temp_ensemble.add_model(model)
            
            # Fit ensemble on training data
            temp_ensemble.fit(train_data)
            
            # Generate predictions for test period
            predictions = []
            actuals = list(test_data.values)
            
            # Walk-forward validation
            current_train = train_data.copy()
            
            for i in range(len(test_data)):
                try:
                    # Predict next value
                    pred = temp_ensemble.predict(periods=1)
                    predictions.append(pred)
                    
                    # Update training data for next iteration (if not last)
                    if i < len(test_data) - 1:
                        new_point = pd.Series([actuals[i]], index=[test_data.index[i]])
                        current_train = pd.concat([current_train, new_point])
                        # Re-fit ensemble with updated data
                        temp_ensemble.fit(current_train)
                        
                except Exception as e:
                    self.logger.warning(f"Ensemble prediction failed at step {i}: {e}")
                    # Use last prediction or mean
                    if predictions:
                        predictions.append(predictions[-1])
                    else:
                        predictions.append(current_train.mean())
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            mape = np.mean(np.abs((actuals - predictions) / np.where(actuals != 0, actuals, 1))) * 100
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'mape': float(mape),
                'rmse': float(np.sqrt(mse)),
                'validation_points': len(test_data),
                'successful_models': len(self._base_models)
            }
            
            self.logger.info(f"Ensemble validation completed. MAPE: {mape:.4f}%, Models: {len(self._base_models)}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Ensemble validation failed: {str(e)}")
            raise RuntimeError(f"Ensemble validation failed: {str(e)}")
    
    def _calculate_model_weights(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        Calculate optimal weights for base models based on validation performance.
        
        Args:
            data: Training data for validation
            exog: Optional exogenous variables
        """
        try:
            # Validate each model and collect performance scores
            model_scores = {}
            
            for model_name, model in self._base_models.items():
                try:
                    # Perform validation on the model
                    validation_result = model.validate(data, test_size=0.2)
                    
                    # Use inverse MAPE as score (lower MAPE = higher score)
                    mape = validation_result.get('mape', 100.0)
                    score = 1.0 / (1.0 + mape / 100.0)  # Normalize to 0-1 range
                    
                    model_scores[model_name] = score
                    self._model_performances[model_name].append(score)
                    
                except Exception as e:
                    self.logger.warning(f"Validation failed for model {model_name}: {e}")
                    model_scores[model_name] = 0.1  # Low score for failed models
            
            # Calculate weights based on combination method
            if self.config.get('combination_method') == 'simple':
                # Equal weights
                weight = 1.0 / len(model_scores)
                for model_name in model_scores:
                    self._model_weights[model_name] = weight
                    
            elif self.config.get('combination_method') in ['performance_weighted', 'stacking']:
                # Performance-based weights
                total_score = sum(model_scores.values())
                
                if total_score > 0:
                    for model_name, score in model_scores.items():
                        weight = score / total_score
                        # Apply minimum weight constraint
                        weight = max(weight, self.config.get('min_weight', 0.05))
                        self._model_weights[model_name] = weight
                else:
                    # Fallback to equal weights if all scores are zero
                    weight = 1.0 / len(model_scores)
                    for model_name in model_scores:
                        self._model_weights[model_name] = weight
            
            # Normalize weights to sum to 1
            total_weight = sum(self._model_weights.values())
            if total_weight > 0:
                for model_name in self._model_weights:
                    self._model_weights[model_name] /= total_weight
            
            self.logger.info(f"Model weights calculated: {self._model_weights}")
            
        except Exception as e:
            self.logger.error(f"Weight calculation failed: {e}")
            # Fallback to equal weights
            weight = 1.0 / len(self._base_models)
            for model_name in self._base_models:
                self._model_weights[model_name] = weight
    
    def _simple_average(self, predictions: Dict[str, float]) -> float:
        """
        Calculate simple average of model predictions.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Simple average prediction
        """
        if not predictions:
            raise ValueError("No predictions available for averaging")
        
        return sum(predictions.values()) / len(predictions)
    
    def _weighted_average(self, predictions: Dict[str, float], successful_models: List[str]) -> float:
        """
        Calculate weighted average of model predictions.
        
        Args:
            predictions: Dictionary of model predictions
            successful_models: List of models that successfully generated predictions
            
        Returns:
            Weighted average prediction
        """
        if not predictions:
            raise ValueError("No predictions available for weighted averaging")
        
        # Get weights for successful models only
        active_weights = {}
        total_weight = 0.0
        
        for model_name in successful_models:
            if model_name in self._model_weights:
                active_weights[model_name] = self._model_weights[model_name]
                total_weight += self._model_weights[model_name]
        
        # Normalize weights for active models
        if total_weight > 0:
            for model_name in active_weights:
                active_weights[model_name] /= total_weight
        else:
            # Fallback to equal weights
            weight = 1.0 / len(successful_models)
            for model_name in successful_models:
                active_weights[model_name] = weight
        
        # Calculate weighted average
        weighted_sum = 0.0
        for model_name, prediction in predictions.items():
            if model_name in active_weights:
                weighted_sum += prediction * active_weights[model_name]
        
        return weighted_sum
    
    def _train_stacking_model(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        Train a meta-learner for stacking ensemble method.
        
        Args:
            data: Training data
            exog: Optional exogenous variables
        """
        try:
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.model_selection import TimeSeriesSplit
            import numpy as np
            
            # Generate out-of-fold predictions for training the meta-learner
            tscv = TimeSeriesSplit(n_splits=5)
            meta_features = []
            meta_targets = []
            
            for train_idx, val_idx in tscv.split(data):
                train_fold = data.iloc[train_idx]
                val_fold = data.iloc[val_idx]
                
                # Get predictions from each base model on validation fold
                fold_predictions = []
                
                for model_name, model in self._base_models.items():
                    try:
                        # Create temporary model for this fold
                        temp_model = type(model)(model_name=f"{model_name}_fold")
                        temp_model.config = getattr(model, 'config', {})
                        temp_model.fit(train_fold)
                        
                        # Predict on validation fold
                        predictions = []
                        for i in range(len(val_fold)):
                            pred = temp_model.predict(periods=1)
                            predictions.append(pred)
                        
                        fold_predictions.append(predictions)
                        
                    except Exception as e:
                        self.logger.warning(f"Stacking fold prediction failed for {model_name}: {e}")
                        # Use mean prediction as fallback
                        fold_predictions.append([train_fold.mean()] * len(val_fold))
                
                # Add fold predictions to meta-features
                if fold_predictions:
                    fold_features = np.array(fold_predictions).T  # Shape: (n_samples, n_models)
                    meta_features.extend(fold_features.tolist())
                    meta_targets.extend(val_fold.values.tolist())
            
            # Train meta-learner
            if meta_features and meta_targets:
                X_meta = np.array(meta_features)
                y_meta = np.array(meta_targets)
                
                # Select stacking model type
                stacking_type = self.config.get('stacking_model', 'linear')
                if stacking_type == 'ridge':
                    self._stacking_model = Ridge(alpha=1.0)
                elif stacking_type == 'lasso':
                    self._stacking_model = Lasso(alpha=0.1)
                else:
                    self._stacking_model = LinearRegression()
                
                self._stacking_model.fit(X_meta, y_meta)
                self.logger.info(f"Stacking model ({stacking_type}) trained successfully")
            else:
                self.logger.warning("No meta-features available for stacking, falling back to weighted average")
                self._stacking_model = None
                
        except ImportError:
            self.logger.warning("Scikit-learn not available for stacking, falling back to weighted average")
            self._stacking_model = None
        except Exception as e:
            self.logger.error(f"Stacking model training failed: {e}")
            self._stacking_model = None
    
    def _stacking_prediction(self, predictions: Dict[str, float], exog: Optional[pd.DataFrame] = None) -> float:
        """
        Generate prediction using stacking meta-learner.
        
        Args:
            predictions: Dictionary of base model predictions
            exog: Optional exogenous variables
            
        Returns:
            Stacked prediction
        """
        if self._stacking_model is None:
            self.logger.warning("Stacking model not available, falling back to weighted average")
            return self._weighted_average(predictions, list(predictions.keys()))
        
        try:
            import numpy as np
            
            # Prepare features for meta-learner
            model_names = sorted(predictions.keys())  # Ensure consistent ordering
            features = np.array([predictions[name] for name in model_names]).reshape(1, -1)
            
            # Generate stacked prediction
            stacked_pred = self._stacking_model.predict(features)[0]
            
            return float(stacked_pred)
            
        except Exception as e:
            self.logger.error(f"Stacking prediction failed: {e}")
            # Fallback to weighted average
            return self._weighted_average(predictions, list(predictions.keys()))
    
    def _check_model_diversity(self, new_model: BaseForecaster) -> bool:
        """
        Check if new model provides sufficient diversity to the ensemble.
        
        Args:
            new_model: Model to check for diversity
            
        Returns:
            True if model is sufficiently diverse
        """
        # This is a simplified diversity check
        # In practice, you might want to compare predictions on a validation set
        
        diversity_threshold = self.config.get('diversity_threshold', 0.7)
        
        # Check model type diversity
        new_model_type = type(new_model).__name__
        existing_types = [type(model).__name__ for model in self._base_models.values()]
        
        # If it's a different model type, consider it diverse
        if new_model_type not in existing_types:
            return True
        
        # If same type, check if we have too many of the same type
        same_type_count = existing_types.count(new_model_type)
        max_same_type = max(1, len(self._base_models) // 2)  # Allow at most half to be same type
        
        return same_type_count < max_same_type
    
    def update_weights(self, recent_performances: Dict[str, List[float]]) -> None:
        """
        Update model weights based on recent performance.
        
        Args:
            recent_performances: Dictionary of recent performance scores for each model
        """
        try:
            decay_factor = self.config.get('weight_decay', 0.95)
            
            # Update performance history with decay
            for model_name, recent_scores in recent_performances.items():
                if model_name in self._model_performances:
                    # Apply decay to historical scores
                    historical_scores = self._model_performances[model_name]
                    decayed_scores = [score * decay_factor for score in historical_scores]
                    
                    # Add recent scores
                    self._model_performances[model_name] = decayed_scores + recent_scores
                    
                    # Keep only recent history
                    max_history = self.config.get('performance_window', 30)
                    if len(self._model_performances[model_name]) > max_history:
                        self._model_performances[model_name] = self._model_performances[model_name][-max_history:]
            
            # Recalculate weights based on updated performance
            self._recalculate_weights()
            
            self.logger.info(f"Updated ensemble weights: {self._model_weights}")
            
        except Exception as e:
            self.logger.error(f"Weight update failed: {e}")
    
    def _recalculate_weights(self) -> None:
        """Recalculate model weights based on current performance history."""
        try:
            model_scores = {}
            
            for model_name, performance_history in self._model_performances.items():
                if performance_history:
                    # Use recent average performance as score
                    recent_window = min(10, len(performance_history))
                    recent_scores = performance_history[-recent_window:]
                    avg_score = sum(recent_scores) / len(recent_scores)
                    model_scores[model_name] = avg_score
                else:
                    model_scores[model_name] = 0.5  # Neutral score for models without history
            
            # Calculate new weights
            total_score = sum(model_scores.values())
            
            if total_score > 0:
                for model_name, score in model_scores.items():
                    weight = score / total_score
                    weight = max(weight, self.config.get('min_weight', 0.05))
                    self._model_weights[model_name] = weight
            else:
                # Equal weights fallback
                weight = 1.0 / len(model_scores)
                for model_name in model_scores:
                    self._model_weights[model_name] = weight
            
            # Normalize weights
            total_weight = sum(self._model_weights.values())
            if total_weight > 0:
                for model_name in self._model_weights:
                    self._model_weights[model_name] /= total_weight
                    
        except Exception as e:
            self.logger.error(f"Weight recalculation failed: {e}")
    
    def optimize_ensemble(self, recent_data: pd.Series, optimizer=None) -> Dict[str, Any]:
        """
        Optimize ensemble using advanced optimization techniques.
        
        Args:
            recent_data: Recent time series data for optimization
            optimizer: Optional EnsembleOptimizer instance
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            if optimizer is None:
                from ensemble_optimizer import EnsembleOptimizer
                optimizer = EnsembleOptimizer()
            
            # Collect recent performance metrics
            recent_performances = {}
            for model_name in self._base_models.keys():
                if model_name in self._model_performances:
                    recent_performances[model_name] = self._model_performances[model_name][-5:]  # Last 5 scores
            
            # Run optimization
            optimization_result = optimizer.optimize_ensemble(self, recent_data, recent_performances)
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Ensemble optimization failed: {e}")
            return {'error': str(e)}
    
    def monitor_performance(self, 
                          actual_values: List[float],
                          predicted_values: List[float],
                          individual_predictions: Dict[str, List[float]],
                          optimizer=None) -> Any:
        """
        Monitor ensemble performance and update tracking.
        
        Args:
            actual_values: Actual observed values
            predicted_values: Ensemble predictions
            individual_predictions: Individual model predictions
            optimizer: Optional EnsembleOptimizer instance
            
        Returns:
            EnsemblePerformance object
        """
        try:
            if optimizer is None:
                from ensemble_optimizer import EnsembleOptimizer
                optimizer = EnsembleOptimizer()
            
            performance = optimizer.monitor_ensemble_performance(
                self, actual_values, predicted_values, individual_predictions
            )
            
            # Update internal performance tracking
            for model_name, predictions in individual_predictions.items():
                if model_name in self._model_performances:
                    # Calculate individual model accuracy
                    if len(predictions) == len(actual_values):
                        accuracy = optimizer._calculate_prediction_accuracy(actual_values, predictions)
                        self._model_performances[model_name].append(accuracy)
                        
                        # Keep only recent history
                        max_history = self.config.get('performance_window', 30)
                        if len(self._model_performances[model_name]) > max_history:
                            self._model_performances[model_name] = self._model_performances[model_name][-max_history:]
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {e}")
            return None
    
    def get_diversity_metrics(self, data: pd.Series, optimizer=None) -> Any:
        """
        Calculate current ensemble diversity metrics.
        
        Args:
            data: Data for diversity calculation
            optimizer: Optional EnsembleOptimizer instance
            
        Returns:
            DiversityMetrics object
        """
        try:
            if optimizer is None:
                from ensemble_optimizer import EnsembleOptimizer
                optimizer = EnsembleOptimizer()
            
            return optimizer._calculate_diversity_metrics(self, data)
            
        except Exception as e:
            self.logger.error(f"Diversity calculation failed: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the ensemble configuration.
        
        Returns:
            Dictionary containing ensemble information
        """
        base_info = super().get_model_info()
        
        ensemble_info = {
            'base_models': list(self._base_models.keys()),
            'model_weights': self._model_weights.copy(),
            'combination_method': self.config.get('combination_method'),
            'model_performances': {k: v[-5:] for k, v in self._model_performances.items()},  # Last 5 scores
            'config': self.config,
            'stacking_model_available': self._stacking_model is not None
        }
        
        base_info.update(ensemble_info)
        return base_info


class ModelFactory:
    """
    Factory class for creating forecasting model instances.
    
    This factory provides a centralized way to create different types of
    forecasting models and manage their instantiation.
    """
    
    _model_registry = {}
    
    @classmethod
    def register_model(cls, model_name: str, model_class: type) -> None:
        """
        Register a new model class with the factory.
        
        Args:
            model_name: Name identifier for the model
            model_class: Class that implements BaseForecaster
        """
        if not issubclass(model_class, BaseForecaster):
            raise ValueError(f"Model class must inherit from BaseForecaster")
        
        cls._model_registry[model_name] = model_class
        forecasting_logger.info(f"Registered model: {model_name}")
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> BaseForecaster:
        """
        Create an instance of the specified model.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional arguments to pass to model constructor
            
        Returns:
            Instance of the requested model
            
        Raises:
            ValueError: If model name is not registered
        """
        if model_name not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
        
        model_class = cls._model_registry[model_name]
        return model_class(model_name=model_name, **kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """
        Get list of available model names.
        
        Returns:
            List of registered model names
        """
        return list(cls._model_registry.keys())


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecasting model.
    
    This implementation includes automatic parameter selection using AIC/BIC criteria
    and time series cross-validation for model validation.
    """
    
    def __init__(self, model_name: str = "arima", **kwargs):
        """
        Initialize ARIMA forecaster.
        
        Args:
            model_name: Name identifier for the model
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name)
        
        # Import required libraries
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
            from statsmodels.stats.diagnostic import acorr_ljungbox
            import warnings
            warnings.filterwarnings('ignore')
            
            self.ARIMA = ARIMA
            self.adfuller = adfuller
            self.acorr_ljungbox = acorr_ljungbox
            
        except ImportError as e:
            raise ImportError(f"Required dependencies not available: {e}")
        
        # Get configuration
        try:
            from forecasting_config import get_model_config
            config = get_model_config('arima')
        except ImportError:
            # Fallback if config module not available
            config = None
        
        if config:
            self.config = config.params
        else:
            # Default configuration
            self.config = {
                'order': (1, 1, 1),
                'trend': 'c',
                'enforce_stationarity': False,
                'enforce_invertibility': False,
                'auto_arima': True,
                'max_p': 5,
                'max_q': 5,
                'max_d': 2
            }
        
        # Override with any provided kwargs
        self.config.update(kwargs)
        
        # Model state
        self._fitted_model = None
        self._best_order = None
        self._aic_score = None
        self._bic_score = None
        
    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        Fit ARIMA model with automatic parameter selection.
        
        Args:
            data: Time series data for training
            exog: Optional exogenous variables (not used in basic ARIMA)
            
        Raises:
            ValueError: If data is insufficient or invalid
            RuntimeError: If model training fails
        """
        import time
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_data(data, min_points=self.config.get('min_data_points', 100))
            
            # Store training data
            self._training_data = data.copy()
            
            # Prepare data (handle missing values)
            clean_data = self._prepare_data(data)
            
            # Check stationarity and suggest differencing
            if self.config.get('auto_arima', True):
                self._best_order = self._auto_arima_selection(clean_data)
            else:
                self._best_order = tuple(self.config['order'])
            
            # Fit the model with best parameters
            self.logger.info(f"Fitting ARIMA{self._best_order} model")
            
            # Determine appropriate trend based on differencing order
            p, d, q = self._best_order
            if d > 0:
                # For integrated models, use no trend or linear trend
                trend = 'n'  # No trend for differenced data
            else:
                trend = self.config.get('trend', 'c')
            
            model = self.ARIMA(
                clean_data,
                order=self._best_order,
                trend=trend,
                enforce_stationarity=self.config.get('enforce_stationarity', False),
                enforce_invertibility=self.config.get('enforce_invertibility', False)
            )
            
            self._fitted_model = model.fit()
            
            # Store model performance metrics
            self._aic_score = self._fitted_model.aic
            self._bic_score = self._fitted_model.bic
            
            # Mark as fitted
            self.is_fitted = True
            
            training_time = time.time() - start_time
            self.logger.info(
                f"ARIMA model fitted successfully in {training_time:.2f}s. "
                f"Order: {self._best_order}, AIC: {self._aic_score:.4f}, BIC: {self._bic_score:.4f}"
            )
            
        except ValueError as e:
            # Re-raise ValueError as-is (for data validation errors)
            self.logger.error(f"ARIMA model validation failed: {str(e)}")
            raise e
        except Exception as e:
            self.logger.error(f"Failed to fit ARIMA model: {str(e)}")
            raise RuntimeError(f"ARIMA model fitting failed: {str(e)}")
    
    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        """
        Generate forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            exog: Optional exogenous variables (not used in basic ARIMA)
            
        Returns:
            Expected return for the forecast period
            
        Raises:
            RuntimeError: If model is not fitted or prediction fails
        """
        import time
        start_time = time.time()
        
        if not self.is_fitted or self._fitted_model is None:
            raise RuntimeError("Model must be fitted before making predictions")
        
        try:
            # Generate forecast
            forecast_result = self._fitted_model.forecast(steps=periods)
            
            # Return the final period forecast (expected return)
            if isinstance(forecast_result, pd.Series):
                expected_return = forecast_result.iloc[-1]
            else:
                expected_return = forecast_result[-1] if hasattr(forecast_result, '__getitem__') else forecast_result
            
            prediction_time = time.time() - start_time
            self.logger.debug(f"ARIMA prediction completed in {prediction_time:.4f}s")
            
            return float(expected_return)
            
        except Exception as e:
            self.logger.error(f"ARIMA prediction failed: {str(e)}")
            raise RuntimeError(f"ARIMA prediction failed: {str(e)}")
    
    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        """
        Perform model validation using time series cross-validation.
        
        Args:
            data: Time series data for validation
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing validation metrics (MSE, MAE, MAPE)
            
        Raises:
            ValueError: If validation parameters are invalid
        """
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        
        try:
            # Prepare data
            clean_data = self._prepare_data(data)
            
            # Split data for time series validation
            split_point = int(len(clean_data) * (1 - test_size))
            train_data = clean_data[:split_point]
            test_data = clean_data[split_point:]
            
            if len(train_data) < 50 or len(test_data) < 10:
                raise ValueError("Insufficient data for validation split")
            
            # Fit model on training data
            temp_forecaster = ARIMAForecaster(model_name=f"{self.model_name}_validation")
            temp_forecaster.config = self.config.copy()
            temp_forecaster.fit(train_data)
            
            # Generate predictions for test period
            predictions = []
            actuals = list(test_data.values)
            
            # Walk-forward validation
            current_train = train_data.copy()
            
            for i in range(len(test_data)):
                try:
                    # Predict next value
                    pred = temp_forecaster.predict(periods=1)
                    predictions.append(pred)
                    
                    # Update training data with actual value for next iteration
                    if i < len(test_data) - 1:
                        new_point = pd.Series([actuals[i]], index=[test_data.index[i]])
                        current_train = pd.concat([current_train, new_point])
                        temp_forecaster.fit(current_train)
                        
                except Exception as e:
                    self.logger.warning(f"Prediction failed at step {i}: {e}")
                    # Use last known prediction or mean
                    if predictions:
                        predictions.append(predictions[-1])
                    else:
                        predictions.append(current_train.mean())
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((actuals - predictions) / np.where(actuals != 0, actuals, 1))) * 100
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'mape': float(mape),
                'rmse': float(np.sqrt(mse)),
                'validation_points': len(test_data)
            }
            
            self.logger.info(f"ARIMA validation completed. MAPE: {mape:.4f}%, MAE: {mae:.6f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"ARIMA validation failed: {str(e)}")
            raise RuntimeError(f"ARIMA validation failed: {str(e)}")
    
    def _prepare_data(self, data: pd.Series) -> pd.Series:
        """
        Prepare data for ARIMA modeling.
        
        Args:
            data: Raw time series data
            
        Returns:
            Cleaned and prepared data
        """
        # Handle missing values
        if data.isnull().any():
            self.logger.warning("Filling missing values with forward fill")
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        # Ensure we have enough data
        if len(data) < 50:
            raise ValueError(f"Insufficient data after cleaning: {len(data)} points")
        
        return data
    
    def _auto_arima_selection(self, data: pd.Series) -> Tuple[int, int, int]:
        """
        Automatically select ARIMA parameters using AIC/BIC criteria.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (p, d, q) parameters
        """
        import itertools
        import numpy as np
        
        max_p = self.config.get('max_p', 5)
        max_q = self.config.get('max_q', 5)
        max_d = self.config.get('max_d', 2)
        
        # Determine differencing order
        d = self._determine_differencing_order(data, max_d)
        
        # Grid search for p and q
        best_aic = np.inf
        best_order = (1, d, 1)
        
        self.logger.info(f"Starting auto-ARIMA selection with d={d}")
        
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                try:
                    order = (p, d, q)
                    
                    # Skip if all parameters are zero
                    if p == 0 and q == 0:
                        continue
                    
                    # Fit temporary model
                    temp_model = self.ARIMA(
                        data,
                        order=order,
                        trend=self.config.get('trend', 'c'),
                        enforce_stationarity=self.config.get('enforce_stationarity', False),
                        enforce_invertibility=self.config.get('enforce_invertibility', False)
                    )
                    
                    fitted_temp = temp_model.fit()
                    
                    # Check AIC
                    if fitted_temp.aic < best_aic:
                        best_aic = fitted_temp.aic
                        best_order = order
                        
                except Exception as e:
                    # Skip problematic parameter combinations
                    self.logger.debug(f"Skipping ARIMA{order}: {str(e)}")
                    continue
        
        self.logger.info(f"Auto-ARIMA selected order: {best_order} with AIC: {best_aic:.4f}")
        return best_order
    
    def _determine_differencing_order(self, data: pd.Series, max_d: int = 2) -> int:
        """
        Determine the appropriate differencing order using ADF test.
        
        Args:
            data: Time series data
            max_d: Maximum differencing order to test
            
        Returns:
            Optimal differencing order
        """
        import numpy as np
        
        # Test original series
        current_data = data.copy()
        
        for d in range(max_d + 1):
            try:
                # Perform Augmented Dickey-Fuller test
                adf_result = self.adfuller(current_data.dropna())
                p_value = adf_result[1]
                
                # If p-value < 0.05, series is stationary
                if p_value < 0.05:
                    self.logger.debug(f"Series is stationary with d={d} (p-value: {p_value:.4f})")
                    return d
                
                # If not stationary and we haven't reached max_d, difference the series
                if d < max_d:
                    current_data = current_data.diff().dropna()
                    
            except Exception as e:
                self.logger.warning(f"ADF test failed for d={d}: {e}")
                # If ADF test fails, return conservative differencing
                return min(1, max_d)
        
        # If no stationarity achieved, return max_d
        self.logger.warning(f"Series not stationary after {max_d} differences, using d={max_d}")
        return max_d
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the fitted ARIMA model.
        
        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()
        
        arima_info = {
            'best_order': self._best_order,
            'aic_score': self._aic_score,
            'bic_score': self._bic_score,
            'config': self.config
        }
        
        if self._fitted_model is not None:
            try:
                arima_info.update({
                    'log_likelihood': self._fitted_model.llf,
                    'params': self._fitted_model.params.to_dict() if hasattr(self._fitted_model.params, 'to_dict') else None,
                    'residuals_ljung_box_p': self._get_ljung_box_p_value()
                })
            except Exception as e:
                self.logger.debug(f"Could not extract additional model info: {e}")
        
        base_info.update(arima_info)
        return base_info
    
    def _get_ljung_box_p_value(self) -> Optional[float]:
        """Get Ljung-Box test p-value for residual autocorrelation."""
        try:
            if self._fitted_model is not None:
                residuals = self._fitted_model.resid
                ljung_box_result = self.acorr_ljungbox(residuals, lags=10, return_df=True)
                return float(ljung_box_result['lb_pvalue'].iloc[-1])
        except Exception as e:
            self.logger.debug(f"Ljung-Box test failed: {e}")
        return None


class SARIMAXForecaster(BaseForecaster):
    """
    SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) forecasting model.
    
    This implementation includes automatic seasonal parameter detection and support for
    market indicators as exogenous variables to improve forecast accuracy.
    """
    
    def __init__(self, model_name: str = "sarimax", **kwargs):
        """
        Initialize SARIMAX forecaster.
        
        Args:
            model_name: Name identifier for the model
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name)
        
        # Import required libraries
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.tsa.stattools import adfuller
            from statsmodels.tsa.seasonal import seasonal_decompose
            from statsmodels.stats.diagnostic import acorr_ljungbox
            import warnings
            warnings.filterwarnings('ignore')
            
            self.SARIMAX = SARIMAX
            self.adfuller = adfuller
            self.seasonal_decompose = seasonal_decompose
            self.acorr_ljungbox = acorr_ljungbox
            
        except ImportError as e:
            raise ImportError(f"Required dependencies not available: {e}")
        
        # Get configuration
        try:
            from forecasting_config import get_model_config
            config = get_model_config('sarimax')
        except ImportError:
            # Fallback if config module not available
            config = None
        
        if config:
            self.config = config.params
        else:
            # Default configuration
            self.config = {
                'order': (1, 1, 1),
                'seasonal_order': (1, 1, 1, 12),
                'trend': 'c',
                'enforce_stationarity': False,
                'enforce_invertibility': False,
                'auto_seasonal': True,
                'seasonal_periods': [12, 4, 52],  # Monthly, quarterly, weekly patterns
                'max_p': 3,
                'max_q': 3,
                'max_P': 2,
                'max_Q': 2,
                'max_d': 2,
                'max_D': 1
            }
        
        # Override with any provided kwargs
        self.config.update(kwargs)
        
        # Model state
        self._fitted_model = None
        self._best_order = None
        self._best_seasonal_order = None
        self._aic_score = None
        self._bic_score = None
        self._exog_columns = None
        self._seasonal_period = None
        
    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        Fit SARIMAX model with seasonal parameter detection and exogenous variables.
        
        Args:
            data: Time series data for training
            exog: Optional exogenous variables (market indicators)
            
        Raises:
            ValueError: If data is insufficient or invalid
            RuntimeError: If model training fails
        """
        import time
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_data(data, min_points=self.config.get('min_data_points', 150))
            
            # Store training data
            self._training_data = data.copy()
            
            # Prepare data (handle missing values)
            clean_data = self._prepare_data(data)
            
            # Prepare exogenous variables if provided
            clean_exog = None
            if exog is not None:
                clean_exog = self._prepare_exog_data(exog, clean_data.index)
                self._exog_columns = list(clean_exog.columns)
                self.logger.info(f"Using {len(self._exog_columns)} exogenous variables: {self._exog_columns}")
            
            # Detect seasonal patterns and parameters
            if self.config.get('auto_seasonal', True):
                self._seasonal_period = self._detect_seasonality(clean_data)
                self._best_order, self._best_seasonal_order = self._auto_sarimax_selection(clean_data, clean_exog)
            else:
                self._best_order = tuple(self.config['order'])
                self._best_seasonal_order = tuple(self.config['seasonal_order'])
                self._seasonal_period = self._best_seasonal_order[3] if len(self._best_seasonal_order) > 3 else 12
            
            # Fit the model with best parameters
            self.logger.info(f"Fitting SARIMAX{self._best_order}x{self._best_seasonal_order} model")
            
            model = self.SARIMAX(
                clean_data,
                exog=clean_exog,
                order=self._best_order,
                seasonal_order=self._best_seasonal_order,
                trend=self.config.get('trend', 'c'),
                enforce_stationarity=self.config.get('enforce_stationarity', False),
                enforce_invertibility=self.config.get('enforce_invertibility', False)
            )
            
            self._fitted_model = model.fit(disp=False)
            
            # Store model performance metrics
            self._aic_score = self._fitted_model.aic
            self._bic_score = self._fitted_model.bic
            
            # Mark as fitted
            self.is_fitted = True
            
            training_time = time.time() - start_time
            self.logger.info(
                f"SARIMAX model fitted successfully in {training_time:.2f}s. "
                f"Order: {self._best_order}x{self._best_seasonal_order}, "
                f"AIC: {self._aic_score:.4f}, BIC: {self._bic_score:.4f}"
            )
            
        except ValueError as e:
            # Re-raise ValueError as-is (for data validation errors)
            self.logger.error(f"SARIMAX model validation failed: {str(e)}")
            raise e
        except Exception as e:
            self.logger.error(f"Failed to fit SARIMAX model: {str(e)}")
            raise RuntimeError(f"SARIMAX model fitting failed: {str(e)}")
    
    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        """
        Generate forecast for the specified number of periods with exogenous variable forecasting.
        
        Args:
            periods: Number of periods to forecast
            exog: Optional exogenous variables for prediction period
            
        Returns:
            Expected return for the forecast period
            
        Raises:
            RuntimeError: If model is not fitted or prediction fails
        """
        import time
        start_time = time.time()
        
        if not self.is_fitted or self._fitted_model is None:
            raise RuntimeError("Model must be fitted before making predictions")
        
        try:
            # Prepare exogenous variables for prediction if needed
            forecast_exog = None
            if self._exog_columns is not None:
                if exog is not None:
                    # Use provided exogenous variables
                    forecast_exog = self._prepare_forecast_exog(exog, periods)
                else:
                    # Generate forecasted exogenous variables
                    forecast_exog = self._forecast_exog_variables(periods)
            
            # Generate forecast
            forecast_result = self._fitted_model.forecast(steps=periods, exog=forecast_exog)
            
            # Return the final period forecast (expected return)
            if isinstance(forecast_result, pd.Series):
                expected_return = forecast_result.iloc[-1]
            else:
                expected_return = forecast_result[-1] if hasattr(forecast_result, '__getitem__') else forecast_result
            
            prediction_time = time.time() - start_time
            self.logger.debug(f"SARIMAX prediction completed in {prediction_time:.4f}s")
            
            return float(expected_return)
            
        except Exception as e:
            self.logger.error(f"SARIMAX prediction failed: {str(e)}")
            raise RuntimeError(f"SARIMAX prediction failed: {str(e)}")
    
    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        """
        Perform model validation using time series cross-validation.
        
        Args:
            data: Time series data for validation
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing validation metrics (MSE, MAE, MAPE)
            
        Raises:
            ValueError: If validation parameters are invalid
        """
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        
        try:
            # Prepare data
            clean_data = self._prepare_data(data)
            
            # Split data for time series validation
            split_point = int(len(clean_data) * (1 - test_size))
            train_data = clean_data[:split_point]
            test_data = clean_data[split_point:]
            
            if len(train_data) < 100 or len(test_data) < 10:
                raise ValueError("Insufficient data for validation split")
            
            # Fit model on training data
            temp_forecaster = SARIMAXForecaster(model_name=f"{self.model_name}_validation")
            temp_forecaster.config = self.config.copy()
            temp_forecaster.fit(train_data)
            
            # Generate predictions for test period
            predictions = []
            actuals = list(test_data.values)
            
            # Walk-forward validation
            current_train = train_data.copy()
            
            for i in range(len(test_data)):
                try:
                    # Predict next value
                    pred = temp_forecaster.predict(periods=1)
                    predictions.append(pred)
                    
                    # Update training data with actual value for next iteration
                    if i < len(test_data) - 1:
                        new_point = pd.Series([actuals[i]], index=[test_data.index[i]])
                        current_train = pd.concat([current_train, new_point])
                        temp_forecaster.fit(current_train)
                        
                except Exception as e:
                    self.logger.warning(f"Prediction failed at step {i}: {e}")
                    # Use last known prediction or mean
                    if predictions:
                        predictions.append(predictions[-1])
                    else:
                        predictions.append(current_train.mean())
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((actuals - predictions) / np.where(actuals != 0, actuals, 1))) * 100
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'mape': float(mape),
                'rmse': float(np.sqrt(mse)),
                'validation_points': len(test_data),
                'seasonal_period': self._seasonal_period
            }
            
            self.logger.info(f"SARIMAX validation completed. MAPE: {mape:.4f}%, MAE: {mae:.6f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"SARIMAX validation failed: {str(e)}")
            raise RuntimeError(f"SARIMAX validation failed: {str(e)}")
    
    def _prepare_data(self, data: pd.Series) -> pd.Series:
        """
        Prepare data for SARIMAX modeling.
        
        Args:
            data: Raw time series data
            
        Returns:
            Cleaned and prepared data
        """
        # Handle missing values
        if data.isnull().any():
            self.logger.warning("Filling missing values with forward fill")
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        # Ensure we have enough data
        min_points = self.config.get('min_data_points', 100)
        if len(data) < min_points:
            raise ValueError(f"Insufficient data after cleaning: {len(data)} points")
        
        return data
    
    def _prepare_exog_data(self, exog: pd.DataFrame, data_index: pd.Index) -> pd.DataFrame:
        """
        Prepare exogenous variables for model fitting.
        
        Args:
            exog: Raw exogenous variables
            data_index: Index from the main time series data
            
        Returns:
            Cleaned and aligned exogenous variables
        """
        # Align exogenous data with main data index
        aligned_exog = exog.reindex(data_index)
        
        # Handle missing values
        if aligned_exog.isnull().any().any():
            self.logger.warning("Filling missing values in exogenous variables")
            aligned_exog = aligned_exog.fillna(method='ffill').fillna(method='bfill')
        
        # Remove columns with all NaN values
        aligned_exog = aligned_exog.dropna(axis=1, how='all')
        
        if aligned_exog.empty:
            raise ValueError("No valid exogenous variables after cleaning")
        
        return aligned_exog
    
    def _detect_seasonality(self, data: pd.Series) -> int:
        """
        Detect seasonal patterns in the time series data.
        
        Args:
            data: Time series data
            
        Returns:
            Detected seasonal period
        """
        import numpy as np
        from scipy import signal
        
        try:
            # Test different seasonal periods
            seasonal_periods = self.config.get('seasonal_periods', [12, 4, 52])
            best_period = 12  # Default
            best_strength = 0
            
            for period in seasonal_periods:
                if len(data) < 2 * period:
                    continue
                
                try:
                    # Perform seasonal decomposition
                    decomposition = self.seasonal_decompose(
                        data, 
                        model='additive', 
                        period=period,
                        extrapolate_trend='freq'
                    )
                    
                    # Calculate seasonal strength
                    seasonal_var = np.var(decomposition.seasonal.dropna())
                    residual_var = np.var(decomposition.resid.dropna())
                    
                    if residual_var > 0:
                        seasonal_strength = seasonal_var / (seasonal_var + residual_var)
                        
                        if seasonal_strength > best_strength:
                            best_strength = seasonal_strength
                            best_period = period
                            
                except Exception as e:
                    self.logger.debug(f"Seasonal decomposition failed for period {period}: {e}")
                    continue
            
            self.logger.info(f"Detected seasonal period: {best_period} (strength: {best_strength:.4f})")
            return best_period
            
        except Exception as e:
            self.logger.warning(f"Seasonality detection failed: {e}, using default period 12")
            return 12
    
    def _auto_sarimax_selection(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Automatically select SARIMAX parameters using AIC/BIC criteria.
        
        Args:
            data: Time series data
            exog: Optional exogenous variables
            
        Returns:
            Tuple of (order, seasonal_order) parameters
        """
        import itertools
        import numpy as np
        
        max_p = self.config.get('max_p', 3)
        max_q = self.config.get('max_q', 3)
        max_P = self.config.get('max_P', 2)
        max_Q = self.config.get('max_Q', 2)
        max_d = self.config.get('max_d', 2)
        max_D = self.config.get('max_D', 1)
        
        # Determine differencing orders
        d = self._determine_differencing_order(data, max_d)
        D = self._determine_seasonal_differencing_order(data, self._seasonal_period, max_D)
        
        # Grid search for p, q, P, Q
        best_aic = np.inf
        best_order = (1, d, 1)
        best_seasonal_order = (1, D, 1, self._seasonal_period)
        
        self.logger.info(f"Starting auto-SARIMAX selection with d={d}, D={D}, s={self._seasonal_period}")
        
        # Limit search space for efficiency
        p_range = range(0, min(max_p + 1, 4))
        q_range = range(0, min(max_q + 1, 4))
        P_range = range(0, min(max_P + 1, 3))
        Q_range = range(0, min(max_Q + 1, 3))
        
        for p, q, P, Q in itertools.product(p_range, q_range, P_range, Q_range):
            try:
                order = (p, d, q)
                seasonal_order = (P, D, Q, self._seasonal_period)
                
                # Skip if all parameters are zero
                if p == 0 and q == 0 and P == 0 and Q == 0:
                    continue
                
                # Fit temporary model
                temp_model = self.SARIMAX(
                    data,
                    exog=exog,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=self.config.get('trend', 'c'),
                    enforce_stationarity=self.config.get('enforce_stationarity', False),
                    enforce_invertibility=self.config.get('enforce_invertibility', False)
                )
                
                fitted_temp = temp_model.fit(disp=False)
                
                # Check AIC
                if fitted_temp.aic < best_aic:
                    best_aic = fitted_temp.aic
                    best_order = order
                    best_seasonal_order = seasonal_order
                    
            except Exception as e:
                # Skip problematic parameter combinations
                self.logger.debug(f"Skipping SARIMAX{order}x{seasonal_order}: {str(e)}")
                continue
        
        self.logger.info(f"Auto-SARIMAX selected order: {best_order}x{best_seasonal_order} with AIC: {best_aic:.4f}")
        return best_order, best_seasonal_order
    
    def _determine_differencing_order(self, data: pd.Series, max_d: int = 2) -> int:
        """
        Determine the appropriate differencing order using ADF test.
        
        Args:
            data: Time series data
            max_d: Maximum differencing order to test
            
        Returns:
            Optimal differencing order
        """
        import numpy as np
        
        # Test original series
        current_data = data.copy()
        
        for d in range(max_d + 1):
            try:
                # Perform Augmented Dickey-Fuller test
                adf_result = self.adfuller(current_data.dropna())
                p_value = adf_result[1]
                
                # If p-value < 0.05, series is stationary
                if p_value < 0.05:
                    self.logger.debug(f"Series is stationary with d={d} (p-value: {p_value:.4f})")
                    return d
                
                # If not stationary and we haven't reached max_d, difference the series
                if d < max_d:
                    current_data = current_data.diff().dropna()
                    
            except Exception as e:
                self.logger.warning(f"ADF test failed for d={d}: {e}")
                # If ADF test fails, return conservative differencing
                return min(1, max_d)
        
        # If no stationarity achieved, return max_d
        self.logger.warning(f"Series not stationary after {max_d} differences, using d={max_d}")
        return max_d
    
    def _determine_seasonal_differencing_order(self, data: pd.Series, seasonal_period: int, max_D: int = 1) -> int:
        """
        Determine the appropriate seasonal differencing order.
        
        Args:
            data: Time series data
            seasonal_period: Seasonal period
            max_D: Maximum seasonal differencing order to test
            
        Returns:
            Optimal seasonal differencing order
        """
        import numpy as np
        
        if len(data) < 2 * seasonal_period:
            return 0
        
        # Test seasonal differencing
        current_data = data.copy()
        
        for D in range(max_D + 1):
            try:
                # Test for seasonal unit root
                seasonal_data = current_data[seasonal_period:] - current_data[:-seasonal_period].values
                
                if len(seasonal_data) < 50:
                    break
                
                # Perform ADF test on seasonally differenced data
                adf_result = self.adfuller(seasonal_data.dropna())
                p_value = adf_result[1]
                
                # If p-value < 0.05, seasonally stationary
                if p_value < 0.05:
                    self.logger.debug(f"Series is seasonally stationary with D={D} (p-value: {p_value:.4f})")
                    return D
                
                # If not stationary and we haven't reached max_D, continue
                if D < max_D:
                    current_data = seasonal_data
                    
            except Exception as e:
                self.logger.warning(f"Seasonal ADF test failed for D={D}: {e}")
                return 0
        
        # Default to no seasonal differencing if tests fail
        return 0
    
    def _prepare_forecast_exog(self, exog: pd.DataFrame, periods: int) -> pd.DataFrame:
        """
        Prepare exogenous variables for forecasting period.
        
        Args:
            exog: Exogenous variables for forecast period
            periods: Number of forecast periods
            
        Returns:
            Prepared exogenous variables
        """
        if len(exog) != periods:
            raise ValueError(f"Exogenous data length ({len(exog)}) must match forecast periods ({periods})")
        
        # Ensure columns match training data
        if self._exog_columns:
            missing_cols = set(self._exog_columns) - set(exog.columns)
            if missing_cols:
                raise ValueError(f"Missing exogenous columns: {missing_cols}")
            
            # Reorder columns to match training data
            exog = exog[self._exog_columns]
        
        return exog
    
    def _forecast_exog_variables(self, periods: int) -> pd.DataFrame:
        """
        Generate forecasted exogenous variables when not provided.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Forecasted exogenous variables
        """
        import numpy as np
        
        if not self._exog_columns:
            return None
        
        # Simple forecasting: use last known values or trends
        # In a real implementation, you might use separate models for each exogenous variable
        forecast_exog = pd.DataFrame(index=range(periods), columns=self._exog_columns)
        
        # For now, use the mean of the last few observations
        # This is a simplified approach - in practice, you'd want more sophisticated forecasting
        for col in self._exog_columns:
            # Use mean of last 10 observations or available data
            last_values = self._training_data.tail(10) if hasattr(self, '_training_data') else None
            if last_values is not None and len(last_values) > 0:
                forecast_value = last_values.mean()
            else:
                forecast_value = 0.0
            
            forecast_exog[col] = forecast_value
        
        self.logger.warning("Using simplified exogenous variable forecasting. Consider providing forecast values.")
        return forecast_exog
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the fitted SARIMAX model.
        
        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()
        
        sarimax_info = {
            'best_order': self._best_order,
            'best_seasonal_order': self._best_seasonal_order,
            'seasonal_period': self._seasonal_period,
            'exog_columns': self._exog_columns,
            'aic_score': self._aic_score,
            'bic_score': self._bic_score,
            'config': self.config
        }
        
        if self._fitted_model is not None:
            try:
                sarimax_info.update({
                    'log_likelihood': self._fitted_model.llf,
                    'params': self._fitted_model.params.to_dict() if hasattr(self._fitted_model.params, 'to_dict') else None,
                    'residuals_ljung_box_p': self._get_ljung_box_p_value()
                })
            except Exception as e:
                self.logger.debug(f"Could not extract additional model info: {e}")
        
        base_info.update(sarimax_info)
        return base_info
    
    def _get_ljung_box_p_value(self) -> Optional[float]:
        """Get Ljung-Box test p-value for residual autocorrelation."""
        try:
            if self._fitted_model is not None:
                residuals = self._fitted_model.resid
                ljung_box_result = self.acorr_ljungbox(residuals, lags=10, return_df=True)
                return float(ljung_box_result['lb_pvalue'].iloc[-1])
        except Exception as e:
            self.logger.debug(f"Ljung-Box test failed: {e}")
        return None


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost forecasting model with feature engineering integration.
    
    This implementation uses gradient boosting with engineered features including
    technical indicators, lagged returns, and market regime indicators.
    """
    
    def __init__(self, model_name: str = "xgboost", **kwargs):
        """
        Initialize XGBoost forecaster.
        
        Args:
            model_name: Name identifier for the model
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name)
        
        # Import required libraries
        try:
            import xgboost as xgb
            from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from feature_engineering import FeatureEngineer, validate_feature_data
            
            self.xgb = xgb
            self.TimeSeriesSplit = TimeSeriesSplit
            self.GridSearchCV = GridSearchCV
            self.mean_squared_error = mean_squared_error
            self.mean_absolute_error = mean_absolute_error
            self.FeatureEngineer = FeatureEngineer
            self.validate_feature_data = validate_feature_data
            
        except ImportError as e:
            raise ImportError(f"Required dependencies not available: {e}")
        
        # Get configuration
        try:
            from forecasting_config import get_model_config
            config = get_model_config('xgboost')
        except ImportError:
            # Fallback if config module not available
            config = None
        
        if config:
            self.config = config.params
        else:
            # Default configuration
            self.config = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'early_stopping_rounds': 10,
                'hyperparameter_tuning': True,
                'cv_folds': 3,
                'feature_selection': True,
                'target_periods': 1
            }
        
        # Override with any provided kwargs
        self.config.update(kwargs)
        
        # Model state
        self._fitted_model = None
        self._feature_engineer = None
        self._feature_columns = None
        self._best_params = None
        self._feature_importance = None
        self._cv_scores = None
        
    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        Fit XGBoost model with feature engineering and hyperparameter tuning.
        
        Args:
            data: Time series data for training
            exog: Optional exogenous variables (will be integrated with engineered features)
            
        Raises:
            ValueError: If data is insufficient or invalid
            RuntimeError: If model training fails
        """
        import time
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_data(data, min_points=self.config.get('min_data_points', 200))
            
            # Store training data
            self._training_data = data.copy()
            
            # Prepare OHLCV-like data for feature engineering
            ohlcv_data = self._prepare_ohlcv_data(data, exog)
            
            # Initialize feature engineer
            self._feature_engineer = self.FeatureEngineer()
            
            # Create features
            self.logger.info("Creating features for XGBoost model")
            features = self._feature_engineer.create_features(ohlcv_data, target_col='close')
            
            # Validate feature data
            self.validate_feature_data(features, min_samples=100)
            
            # Prepare training data
            X, y = self._prepare_training_data(features)
            
            if len(X) < 50:
                raise ValueError(f"Insufficient training samples after feature engineering: {len(X)}")
            
            # Store feature columns
            self._feature_columns = list(X.columns)
            
            # Hyperparameter tuning if enabled
            if self.config.get('hyperparameter_tuning', True):
                self.logger.info("Performing hyperparameter tuning")
                self._best_params = self._tune_hyperparameters(X, y)
            else:
                self._best_params = {
                    'n_estimators': self.config['n_estimators'],
                    'max_depth': self.config['max_depth'],
                    'learning_rate': self.config['learning_rate'],
                    'subsample': self.config['subsample'],
                    'colsample_bytree': self.config['colsample_bytree'],
                    'random_state': self.config['random_state'],
                    'n_jobs': self.config['n_jobs']
                }
            
            # Train final model
            self.logger.info(f"Training XGBoost model with {len(X)} samples and {len(X.columns)} features")
            
            self._fitted_model = self.xgb.XGBRegressor(**self._best_params)
            
            # Use early stopping if validation data is available
            if len(X) > 100:
                # Split for early stopping
                split_point = int(len(X) * 0.8)
                X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
                y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
                
                # Try new XGBoost API first, fallback to old API
                try:
                    self._fitted_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                except Exception:
                    # Fallback to basic fitting without early stopping
                    self._fitted_model.fit(X_train, y_train)
            else:
                self._fitted_model.fit(X, y)
            
            # Store feature importance
            self._feature_importance = dict(zip(
                self._feature_columns,
                self._fitted_model.feature_importances_
            ))
            
            # Mark as fitted
            self.is_fitted = True
            
            training_time = time.time() - start_time
            self.logger.info(
                f"XGBoost model fitted successfully in {training_time:.2f}s. "
                f"Features: {len(self._feature_columns)}, Best params: {self._best_params}"
            )
            
        except ValueError as e:
            # Re-raise ValueError as-is (for data validation errors)
            self.logger.error(f"XGBoost model validation failed: {str(e)}")
            raise e
        except Exception as e:
            self.logger.error(f"Failed to fit XGBoost model: {str(e)}")
            raise RuntimeError(f"XGBoost model fitting failed: {str(e)}")
    
    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        """
        Generate forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            exog: Optional exogenous variables for prediction
            
        Returns:
            Expected return for the forecast period
            
        Raises:
            RuntimeError: If model is not fitted or prediction fails
        """
        import time
        start_time = time.time()
        
        if not self.is_fitted or self._fitted_model is None:
            raise RuntimeError("Model must be fitted before making predictions")
        
        try:
            # For XGBoost, we need to create features for the prediction period
            # We'll use the last available data and extend it
            
            # Get recent data for feature creation
            recent_data = self._training_data.tail(max(100, periods * 2))
            
            # Prepare OHLCV-like data
            ohlcv_data = self._prepare_ohlcv_data(recent_data, exog)
            
            # Create features
            features = self._feature_engineer.create_features(ohlcv_data, target_col='close')
            
            # Get the most recent feature vector
            if len(features) == 0:
                raise RuntimeError("No valid features could be created for prediction")
            
            # Use the last available feature vector
            last_features = features.iloc[-1:][self._feature_columns]
            
            # Handle any missing features
            for col in self._feature_columns:
                if col not in last_features.columns:
                    last_features[col] = 0.0  # Default value for missing features
            
            # Ensure correct column order
            last_features = last_features[self._feature_columns]
            
            # Make prediction
            prediction = self._fitted_model.predict(last_features)[0]
            
            prediction_time = time.time() - start_time
            self.logger.debug(f"XGBoost prediction completed in {prediction_time:.4f}s")
            
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"XGBoost prediction failed: {str(e)}")
            raise RuntimeError(f"XGBoost prediction failed: {str(e)}")
    
    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        """
        Perform model validation using time series cross-validation.
        
        Args:
            data: Time series data for validation
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing validation metrics (MSE, MAE, MAPE)
            
        Raises:
            ValueError: If validation parameters are invalid
        """
        import numpy as np
        
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        
        try:
            # Prepare OHLCV-like data
            ohlcv_data = self._prepare_ohlcv_data(data)
            
            # Create features
            temp_engineer = self.FeatureEngineer()
            features = temp_engineer.create_features(ohlcv_data, target_col='close')
            
            if len(features) < 100:
                raise ValueError("Insufficient data for validation after feature engineering")
            
            # Prepare training data
            X, y = self._prepare_training_data(features)
            
            # Time series cross-validation
            tscv = self.TimeSeriesSplit(n_splits=self.config.get('cv_folds', 3))
            
            mse_scores = []
            mae_scores = []
            mape_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train temporary model
                temp_model = self.xgb.XGBRegressor(
                    n_estimators=50,  # Reduced for faster validation
                    max_depth=self.config['max_depth'],
                    learning_rate=self.config['learning_rate'],
                    random_state=self.config['random_state'],
                    n_jobs=self.config['n_jobs']
                )
                
                temp_model.fit(X_train, y_train)
                y_pred = temp_model.predict(X_val)
                
                # Calculate metrics
                mse = self.mean_squared_error(y_val, y_pred)
                mae = self.mean_absolute_error(y_val, y_pred)
                
                # Calculate MAPE
                mape = np.mean(np.abs((y_val - y_pred) / np.where(y_val != 0, y_val, 1))) * 100
                
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)
            
            # Store CV scores
            self._cv_scores = {
                'mse_scores': mse_scores,
                'mae_scores': mae_scores,
                'mape_scores': mape_scores
            }
            
            metrics = {
                'mse': float(np.mean(mse_scores)),
                'mae': float(np.mean(mae_scores)),
                'mape': float(np.mean(mape_scores)),
                'rmse': float(np.sqrt(np.mean(mse_scores))),
                'mse_std': float(np.std(mse_scores)),
                'mae_std': float(np.std(mae_scores)),
                'mape_std': float(np.std(mape_scores)),
                'validation_folds': len(mse_scores)
            }
            
            self.logger.info(f"XGBoost validation completed. MAPE: {metrics['mape']:.4f}% ± {metrics['mape_std']:.4f}%")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"XGBoost validation failed: {str(e)}")
            raise RuntimeError(f"XGBoost validation failed: {str(e)}")
    
    def _prepare_ohlcv_data(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare OHLCV-like data from price series for feature engineering.
        
        Args:
            data: Price series
            exog: Optional exogenous variables
            
        Returns:
            DataFrame with OHLCV-like structure
        """
        ohlcv = pd.DataFrame(index=data.index)
        ohlcv['close'] = data
        
        # Create synthetic OHLC data if not available
        ohlcv['open'] = data.shift(1).fillna(data.iloc[0])
        ohlcv['high'] = data * (1 + np.abs(np.random.normal(0, 0.005, len(data))))
        ohlcv['low'] = data * (1 - np.abs(np.random.normal(0, 0.005, len(data))))
        
        # Add synthetic volume if not available
        ohlcv['volume'] = np.random.randint(1000000, 10000000, len(data))
        
        # Add exogenous variables if provided
        if exog is not None:
            for col in exog.columns:
                if col not in ohlcv.columns:
                    # Align exog data with price data index
                    aligned_exog = exog[col].reindex(data.index, method='ffill')
                    ohlcv[col] = aligned_exog
        
        return ohlcv
    
    def _prepare_training_data(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from features.
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Tuple of (X, y) for training
        """
        # Target variable: next period return
        target_periods = self.config.get('target_periods', 1)
        
        if 'returns' not in features.columns:
            raise ValueError("Returns column not found in features")
        
        # Create target variable (future returns)
        y = features['returns'].shift(-target_periods)
        
        # Remove target-related columns from features
        feature_cols = [col for col in features.columns 
                       if not col.startswith('returns') and col != 'price' and col != 'ticker']
        
        X = features[feature_cols].copy()
        
        # Remove rows with NaN target values
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Feature selection if enabled
        if self.config.get('feature_selection', True) and len(X.columns) > 50:
            X = self._select_features(X, y)
        
        return X, y
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 50) -> pd.DataFrame:
        """
        Select most important features using correlation and variance filtering.
        
        Args:
            X: Feature matrix
            y: Target variable
            max_features: Maximum number of features to select
            
        Returns:
            DataFrame with selected features
        """
        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1]
        X = X.drop(columns=constant_features)
        
        if len(constant_features) > 0:
            self.logger.info(f"Removed {len(constant_features)} constant features")
        
        # Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns 
                             if any(upper_triangle[column] > 0.95)]
        X = X.drop(columns=high_corr_features)
        
        if len(high_corr_features) > 0:
            self.logger.info(f"Removed {len(high_corr_features)} highly correlated features")
        
        # Select top features by correlation with target
        if len(X.columns) > max_features:
            target_corr = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = target_corr.head(max_features).index
            X = X[selected_features]
            
            self.logger.info(f"Selected top {len(selected_features)} features by target correlation")
        
        return X
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Tune hyperparameters using time series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of best parameters
        """
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Use time series cross-validation
        tscv = self.TimeSeriesSplit(n_splits=3)
        
        # Create base model
        base_model = self.xgb.XGBRegressor(
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs']
        )
        
        # Grid search
        grid_search = self.GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=1,  # XGBoost already uses multiple cores
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        best_params = grid_search.best_params_
        best_params.update({
            'random_state': self.config['random_state'],
            'n_jobs': self.config['n_jobs']
        })
        
        self.logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the fitted XGBoost model.
        
        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()
        
        xgboost_info = {
            'best_params': self._best_params,
            'feature_columns': self._feature_columns,
            'num_features': len(self._feature_columns) if self._feature_columns else 0,
            'feature_importance': self._feature_importance,
            'cv_scores': self._cv_scores,
            'config': self.config
        }
        
        if self._fitted_model is not None:
            try:
                xgboost_info.update({
                    'n_estimators': self._fitted_model.n_estimators,
                    'max_depth': self._fitted_model.max_depth,
                    'learning_rate': self._fitted_model.learning_rate,
                    'best_iteration': getattr(self._fitted_model, 'best_iteration', None)
                })
            except Exception as e:
                self.logger.debug(f"Could not extract additional model info: {e}")
        
        base_info.update(xgboost_info)
        return base_info
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self._feature_importance:
            return {}
        
        # Sort by importance and return top N
        sorted_features = sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features[:top_n])


class LightGBMForecaster(BaseForecaster):
    """
    LightGBM forecasting model optimized for speed and large datasets.
    
    This implementation uses LightGBM's gradient boosting with efficient training,
    early stopping, and overfitting prevention for fast and accurate forecasting.
    """
    
    def __init__(self, model_name: str = "lightgbm", **kwargs):
        """
        Initialize LightGBM forecaster.
        
        Args:
            model_name: Name identifier for the model
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name)
        
        # Import required libraries
        try:
            import lightgbm as lgb
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from feature_engineering import FeatureEngineer, validate_feature_data
            
            self.lgb = lgb
            self.TimeSeriesSplit = TimeSeriesSplit
            self.mean_squared_error = mean_squared_error
            self.mean_absolute_error = mean_absolute_error
            self.FeatureEngineer = FeatureEngineer
            self.validate_feature_data = validate_feature_data
            
        except ImportError as e:
            raise ImportError(f"Required dependencies not available: {e}")
        
        # Get configuration
        try:
            from forecasting_config import get_model_config
            config = get_model_config('lightgbm')
        except ImportError:
            # Fallback if config module not available
            config = None
        
        if config:
            self.config = config.params
        else:
            # Default configuration optimized for speed
            self.config = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'num_iterations': 100,
                'early_stopping_rounds': 10,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1,
                'hyperparameter_tuning': True,
                'cv_folds': 3,
                'feature_selection': True,
                'target_periods': 1
            }
        
        # Override with any provided kwargs, but preserve essential config
        if kwargs:
            for key, value in kwargs.items():
                self.config[key] = value
        
        # Model state
        self._fitted_model = None
        self._feature_engineer = None
        self._feature_columns = None
        self._best_params = None
        self._feature_importance = None
        self._cv_scores = None
        self._training_history = None
        
    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        Fit LightGBM model with efficient training and early stopping.
        
        Args:
            data: Time series data for training
            exog: Optional exogenous variables (will be integrated with engineered features)
            
        Raises:
            ValueError: If data is insufficient or invalid
            RuntimeError: If model training fails
        """
        import time
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_data(data, min_points=self.config.get('min_data_points', 150))
            
            # Store training data
            self._training_data = data.copy()
            
            # Prepare OHLCV-like data for feature engineering
            ohlcv_data = self._prepare_ohlcv_data(data, exog)
            
            # Initialize feature engineer
            self._feature_engineer = self.FeatureEngineer()
            
            # Create features
            self.logger.info("Creating features for LightGBM model")
            features = self._feature_engineer.create_features(ohlcv_data, target_col='close')
            
            # Validate feature data
            self.validate_feature_data(features, min_samples=80)
            
            # Prepare training data
            X, y = self._prepare_training_data(features)
            
            if len(X) < 30:
                raise ValueError(f"Insufficient training samples after feature engineering: {len(X)}")
            
            # Store feature columns
            self._feature_columns = list(X.columns)
            
            # Hyperparameter tuning if enabled
            if self.config.get('hyperparameter_tuning', True) and len(X) > 100:
                self.logger.info("Performing hyperparameter tuning")
                self._best_params = self._tune_hyperparameters(X, y)
            else:
                self._best_params = {
                    'objective': self.config.get('objective', 'regression'),
                    'metric': self.config.get('metric', 'rmse'),
                    'boosting_type': self.config.get('boosting_type', 'gbdt'),
                    'num_leaves': self.config.get('num_leaves', 31),
                    'learning_rate': self.config.get('learning_rate', 0.1),
                    'feature_fraction': self.config.get('feature_fraction', 0.8),
                    'bagging_fraction': self.config.get('bagging_fraction', 0.8),
                    'bagging_freq': self.config.get('bagging_freq', 5),
                    'min_child_samples': self.config.get('min_child_samples', 20),
                    'verbose': self.config.get('verbose', -1),
                    'random_state': self.config.get('random_state', 42),
                    'n_jobs': self.config.get('n_jobs', -1)
                }
            
            # Train final model
            self.logger.info(f"Training LightGBM model with {len(X)} samples and {len(X.columns)} features")
            
            # Create LightGBM datasets
            if len(X) > 50:
                # Split for early stopping
                split_point = int(len(X) * 0.8)
                X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
                y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
                
                train_data = self.lgb.Dataset(X_train, label=y_train)
                valid_data = self.lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train with early stopping
                self._fitted_model = self.lgb.train(
                    self._best_params,
                    train_data,
                    num_boost_round=self.config.get('num_iterations', 100),
                    valid_sets=[valid_data],
                    callbacks=[
                        self.lgb.early_stopping(self.config.get('early_stopping_rounds', 10)),
                        self.lgb.log_evaluation(0)  # Suppress output
                    ]
                )
            else:
                # Train without validation for small datasets
                train_data = self.lgb.Dataset(X, label=y)
                
                self._fitted_model = self.lgb.train(
                    self._best_params,
                    train_data,
                    num_boost_round=min(50, self.config.get('num_iterations', 100)),
                    callbacks=[self.lgb.log_evaluation(0)]
                )
            
            # Store feature importance
            self._feature_importance = dict(zip(
                self._feature_columns,
                self._fitted_model.feature_importance(importance_type='gain')
            ))
            
            # Mark as fitted
            self.is_fitted = True
            
            training_time = time.time() - start_time
            self.logger.info(
                f"LightGBM model fitted successfully in {training_time:.2f}s. "
                f"Features: {len(self._feature_columns)}, Best iteration: {self._fitted_model.best_iteration}"
            )
            
        except ValueError as e:
            # Re-raise ValueError as-is (for data validation errors)
            self.logger.error(f"LightGBM model validation failed: {str(e)}")
            raise e
        except Exception as e:
            self.logger.error(f"Failed to fit LightGBM model: {str(e)}")
            raise RuntimeError(f"LightGBM model fitting failed: {str(e)}")
    
    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        """
        Generate forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            exog: Optional exogenous variables for prediction
            
        Returns:
            Expected return for the forecast period
            
        Raises:
            RuntimeError: If model is not fitted or prediction fails
        """
        import time
        start_time = time.time()
        
        if not self.is_fitted or self._fitted_model is None:
            raise RuntimeError("Model must be fitted before making predictions")
        
        try:
            # Get recent data for feature creation
            recent_data = self._training_data.tail(max(100, periods * 2))
            
            # Prepare OHLCV-like data
            ohlcv_data = self._prepare_ohlcv_data(recent_data, exog)
            
            # Create features
            features = self._feature_engineer.create_features(ohlcv_data, target_col='close')
            
            # Get the most recent feature vector
            if len(features) == 0:
                raise RuntimeError("No valid features could be created for prediction")
            
            # Use the last available feature vector
            last_features = features.iloc[-1:][self._feature_columns]
            
            # Handle any missing features
            for col in self._feature_columns:
                if col not in last_features.columns:
                    last_features[col] = 0.0  # Default value for missing features
            
            # Ensure correct column order
            last_features = last_features[self._feature_columns]
            
            # Make prediction
            prediction = self._fitted_model.predict(last_features, num_iteration=self._fitted_model.best_iteration)[0]
            
            prediction_time = time.time() - start_time
            self.logger.debug(f"LightGBM prediction completed in {prediction_time:.4f}s")
            
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"LightGBM prediction failed: {str(e)}")
            raise RuntimeError(f"LightGBM prediction failed: {str(e)}")
    
    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        """
        Perform model validation using time series cross-validation.
        
        Args:
            data: Time series data for validation
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing validation metrics (MSE, MAE, MAPE)
            
        Raises:
            ValueError: If validation parameters are invalid
        """
        import numpy as np
        
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        
        try:
            # Prepare OHLCV-like data
            ohlcv_data = self._prepare_ohlcv_data(data)
            
            # Create features
            temp_engineer = self.FeatureEngineer()
            features = temp_engineer.create_features(ohlcv_data, target_col='close')
            
            if len(features) < 50:
                raise ValueError("Insufficient data for validation after feature engineering")
            
            # Prepare training data
            X, y = self._prepare_training_data(features)
            
            # Time series cross-validation
            tscv = self.TimeSeriesSplit(n_splits=min(3, len(X) // 20))
            
            mse_scores = []
            mae_scores = []
            mape_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create LightGBM datasets
                train_data = self.lgb.Dataset(X_train, label=y_train)
                
                # Train temporary model with reduced iterations for speed
                temp_params = self._best_params.copy() if self._best_params else self.config.copy()
                temp_params['verbose'] = -1
                
                temp_model = self.lgb.train(
                    temp_params,
                    train_data,
                    num_boost_round=30,  # Reduced for faster validation
                    callbacks=[self.lgb.log_evaluation(0)]
                )
                
                y_pred = temp_model.predict(X_val)
                
                # Calculate metrics
                mse = self.mean_squared_error(y_val, y_pred)
                mae = self.mean_absolute_error(y_val, y_pred)
                
                # Calculate MAPE
                mape = np.mean(np.abs((y_val - y_pred) / np.where(y_val != 0, y_val, 1))) * 100
                
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)
            
            # Store CV scores
            self._cv_scores = {
                'mse_scores': mse_scores,
                'mae_scores': mae_scores,
                'mape_scores': mape_scores
            }
            
            metrics = {
                'mse': float(np.mean(mse_scores)),
                'mae': float(np.mean(mae_scores)),
                'mape': float(np.mean(mape_scores)),
                'rmse': float(np.sqrt(np.mean(mse_scores))),
                'mse_std': float(np.std(mse_scores)),
                'mae_std': float(np.std(mae_scores)),
                'mape_std': float(np.std(mape_scores)),
                'validation_folds': len(mse_scores)
            }
            
            self.logger.info(f"LightGBM validation completed. MAPE: {metrics['mape']:.4f}% ± {metrics['mape_std']:.4f}%")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"LightGBM validation failed: {str(e)}")
            raise RuntimeError(f"LightGBM validation failed: {str(e)}")
    
    def _prepare_ohlcv_data(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare OHLCV-like data from price series for feature engineering.
        
        Args:
            data: Price series
            exog: Optional exogenous variables
            
        Returns:
            DataFrame with OHLCV-like structure
        """
        ohlcv = pd.DataFrame(index=data.index)
        ohlcv['close'] = data
        
        # Create synthetic OHLC data if not available
        ohlcv['open'] = data.shift(1).fillna(data.iloc[0])
        ohlcv['high'] = data * (1 + np.abs(np.random.normal(0, 0.005, len(data))))
        ohlcv['low'] = data * (1 - np.abs(np.random.normal(0, 0.005, len(data))))
        
        # Add synthetic volume if not available
        ohlcv['volume'] = np.random.randint(1000000, 10000000, len(data))
        
        # Add exogenous variables if provided
        if exog is not None:
            for col in exog.columns:
                if col not in ohlcv.columns:
                    # Align exog data with price data index
                    aligned_exog = exog[col].reindex(data.index, method='ffill')
                    ohlcv[col] = aligned_exog
        
        return ohlcv
    
    def _prepare_training_data(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from features.
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Tuple of (X, y) for training
        """
        # Target variable: next period return
        target_periods = self.config.get('target_periods', 1)
        
        if 'returns' not in features.columns:
            raise ValueError("Returns column not found in features")
        
        # Create target variable (future returns)
        y = features['returns'].shift(-target_periods)
        
        # Remove target-related columns from features
        feature_cols = [col for col in features.columns 
                       if not col.startswith('returns') and col != 'price' and col != 'ticker']
        
        X = features[feature_cols].copy()
        
        # Remove rows with NaN target values
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Feature selection if enabled
        if self.config.get('feature_selection', True) and len(X.columns) > 40:
            X = self._select_features(X, y)
        
        return X, y
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 40) -> pd.DataFrame:
        """
        Select most important features using correlation and variance filtering.
        
        Args:
            X: Feature matrix
            y: Target variable
            max_features: Maximum number of features to select
            
        Returns:
            DataFrame with selected features
        """
        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1]
        X = X.drop(columns=constant_features)
        
        if len(constant_features) > 0:
            self.logger.info(f"Removed {len(constant_features)} constant features")
        
        # Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns 
                             if any(upper_triangle[column] > 0.95)]
        X = X.drop(columns=high_corr_features)
        
        if len(high_corr_features) > 0:
            self.logger.info(f"Removed {len(high_corr_features)} highly correlated features")
        
        # Select top features by correlation with target
        if len(X.columns) > max_features:
            target_corr = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = target_corr.head(max_features).index
            X = X[selected_features]
            
            self.logger.info(f"Selected top {len(selected_features)} features by target correlation")
        
        return X
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Tune hyperparameters using time series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of best parameters
        """
        # Define parameter grid (smaller for speed)
        param_combinations = [
            {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'verbose': -1,
                'random_state': self.config['random_state'],
                'n_jobs': self.config['n_jobs']
            },
            {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 15,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9,
                'bagging_freq': 3,
                'min_child_samples': 10,
                'verbose': -1,
                'random_state': self.config['random_state'],
                'n_jobs': self.config['n_jobs']
            },
            {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 63,
                'learning_rate': 0.2,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7,
                'bagging_freq': 7,
                'min_child_samples': 30,
                'verbose': -1,
                'random_state': self.config['random_state'],
                'n_jobs': self.config['n_jobs']
            }
        ]
        
        # Use time series cross-validation
        tscv = self.TimeSeriesSplit(n_splits=3)
        
        best_score = float('inf')
        best_params = param_combinations[0]
        
        for params in param_combinations:
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create datasets
                train_data = self.lgb.Dataset(X_train, label=y_train)
                
                # Train model
                model = self.lgb.train(
                    params,
                    train_data,
                    num_boost_round=50,  # Reduced for speed
                    callbacks=[self.lgb.log_evaluation(0)]
                )
                
                # Predict and score
                y_pred = model.predict(X_val)
                score = self.mean_squared_error(y_val, y_pred)
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
        
        self.logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the fitted LightGBM model.
        
        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()
        
        lightgbm_info = {
            'best_params': self._best_params,
            'feature_columns': self._feature_columns,
            'num_features': len(self._feature_columns) if self._feature_columns else 0,
            'feature_importance': self._feature_importance,
            'cv_scores': self._cv_scores,
            'config': self.config
        }
        
        if self._fitted_model is not None:
            try:
                lightgbm_info.update({
                    'best_iteration': self._fitted_model.best_iteration,
                    'num_trees': self._fitted_model.num_trees(),
                    'objective': self._fitted_model.params.get('objective', 'unknown'),
                    'boosting_type': self._fitted_model.params.get('boosting_type', 'unknown')
                })
            except Exception as e:
                self.logger.debug(f"Could not extract additional model info: {e}")
        
        base_info.update(lightgbm_info)
        return base_info
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self._feature_importance:
            return {}
        
        # Sort by importance and return top N
        sorted_features = sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features[:top_n])


class CatBoostForecaster(BaseForecaster):
    """
    CatBoost forecasting model with categorical feature handling and built-in overfitting protection.
    
    This implementation uses CatBoost's gradient boosting with automatic categorical feature
    handling, robust training, and built-in overfitting protection.
    """
    
    def __init__(self, model_name: str = "catboost", **kwargs):
        """
        Initialize CatBoost forecaster.
        
        Args:
            model_name: Name identifier for the model
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name)
        
        # Import required libraries
        try:
            import catboost as cb
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from feature_engineering import FeatureEngineer, validate_feature_data
            
            self.cb = cb
            self.TimeSeriesSplit = TimeSeriesSplit
            self.mean_squared_error = mean_squared_error
            self.mean_absolute_error = mean_absolute_error
            self.FeatureEngineer = FeatureEngineer
            self.validate_feature_data = validate_feature_data
            
        except ImportError as e:
            raise ImportError(f"Required dependencies not available: {e}")
        
        # Get configuration
        try:
            from forecasting_config import get_model_config
            config = get_model_config('catboost')
        except ImportError:
            # Fallback if config module not available
            config = None
        
        if config:
            self.config = config.params
        else:
            # Default configuration with built-in overfitting protection
            self.config = {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1,
                'od_type': 'Iter',
                'od_wait': 10,
                'random_seed': 42,
                'logging_level': 'Silent',
                'thread_count': -1,
                'hyperparameter_tuning': True,
                'cv_folds': 3,
                'feature_selection': True,
                'target_periods': 1,
                'categorical_features': []
            }
        
        # Override with any provided kwargs
        self.config.update(kwargs)
        
        # Model state
        self._fitted_model = None
        self._feature_engineer = None
        self._feature_columns = None
        self._categorical_features = None
        self._best_params = None
        self._feature_importance = None
        self._cv_scores = None
        
    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        Fit CatBoost model with categorical feature handling and overfitting protection.
        
        Args:
            data: Time series data for training
            exog: Optional exogenous variables (will be integrated with engineered features)
            
        Raises:
            ValueError: If data is insufficient or invalid
            RuntimeError: If model training fails
        """
        import time
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_data(data, min_points=self.config.get('min_data_points', 150))
            
            # Store training data
            self._training_data = data.copy()
            
            # Prepare OHLCV-like data for feature engineering
            ohlcv_data = self._prepare_ohlcv_data(data, exog)
            
            # Initialize feature engineer
            self._feature_engineer = self.FeatureEngineer()
            
            # Create features
            self.logger.info("Creating features for CatBoost model")
            features = self._feature_engineer.create_features(ohlcv_data, target_col='close')
            
            # Validate feature data
            self.validate_feature_data(features, min_samples=80)
            
            # Prepare training data
            X, y = self._prepare_training_data(features)
            
            if len(X) < 30:
                raise ValueError(f"Insufficient training samples after feature engineering: {len(X)}")
            
            # Store feature columns and identify categorical features
            self._feature_columns = list(X.columns)
            self._categorical_features = self._identify_categorical_features(X)
            
            # Hyperparameter tuning if enabled
            if self.config.get('hyperparameter_tuning', True) and len(X) > 100:
                self.logger.info("Performing hyperparameter tuning")
                self._best_params = self._tune_hyperparameters(X, y)
            else:
                self._best_params = {
                    'iterations': self.config['iterations'],
                    'learning_rate': self.config['learning_rate'],
                    'depth': self.config['depth'],
                    'l2_leaf_reg': self.config['l2_leaf_reg'],
                    'bootstrap_type': self.config['bootstrap_type'],
                    'bagging_temperature': self.config['bagging_temperature'],
                    'od_type': self.config['od_type'],
                    'od_wait': self.config['od_wait'],
                    'random_seed': self.config['random_seed'],
                    'logging_level': self.config['logging_level'],
                    'thread_count': self.config['thread_count']
                }
            
            # Train final model
            self.logger.info(f"Training CatBoost model with {len(X)} samples and {len(X.columns)} features")
            
            # Create CatBoost model
            self._fitted_model = self.cb.CatBoostRegressor(**self._best_params)
            
            # Prepare training data with validation split if enough data
            if len(X) > 50:
                # Split for early stopping
                split_point = int(len(X) * 0.8)
                X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
                y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
                
                # Train with validation
                self._fitted_model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    cat_features=self._categorical_features,
                    verbose=False,
                    plot=False
                )
            else:
                # Train without validation for small datasets
                self._fitted_model.fit(
                    X, y,
                    cat_features=self._categorical_features,
                    verbose=False,
                    plot=False
                )
            
            # Store feature importance
            self._feature_importance = dict(zip(
                self._feature_columns,
                self._fitted_model.feature_importances_
            ))
            
            # Mark as fitted
            self.is_fitted = True
            
            training_time = time.time() - start_time
            self.logger.info(
                f"CatBoost model fitted successfully in {training_time:.2f}s. "
                f"Features: {len(self._feature_columns)}, Categorical: {len(self._categorical_features)}"
            )
            
        except ValueError as e:
            # Re-raise ValueError as-is (for data validation errors)
            self.logger.error(f"CatBoost model validation failed: {str(e)}")
            raise e
        except Exception as e:
            self.logger.error(f"Failed to fit CatBoost model: {str(e)}")
            raise RuntimeError(f"CatBoost model fitting failed: {str(e)}")
    
    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        """
        Generate forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            exog: Optional exogenous variables for prediction
            
        Returns:
            Expected return for the forecast period
            
        Raises:
            RuntimeError: If model is not fitted or prediction fails
        """
        import time
        start_time = time.time()
        
        if not self.is_fitted or self._fitted_model is None:
            raise RuntimeError("Model must be fitted before making predictions")
        
        try:
            # Get recent data for feature creation
            recent_data = self._training_data.tail(max(100, periods * 2))
            
            # Prepare OHLCV-like data
            ohlcv_data = self._prepare_ohlcv_data(recent_data, exog)
            
            # Create features
            features = self._feature_engineer.create_features(ohlcv_data, target_col='close')
            
            # Get the most recent feature vector
            if len(features) == 0:
                raise RuntimeError("No valid features could be created for prediction")
            
            # Use the last available feature vector
            last_features = features.iloc[-1:][self._feature_columns]
            
            # Handle any missing features
            for col in self._feature_columns:
                if col not in last_features.columns:
                    last_features[col] = 0.0  # Default value for missing features
            
            # Ensure correct column order
            last_features = last_features[self._feature_columns]
            
            # Make prediction
            prediction = self._fitted_model.predict(last_features)[0]
            
            prediction_time = time.time() - start_time
            self.logger.debug(f"CatBoost prediction completed in {prediction_time:.4f}s")
            
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"CatBoost prediction failed: {str(e)}")
            raise RuntimeError(f"CatBoost prediction failed: {str(e)}")
    
    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        """
        Perform model validation using time series cross-validation.
        
        Args:
            data: Time series data for validation
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing validation metrics (MSE, MAE, MAPE)
            
        Raises:
            ValueError: If validation parameters are invalid
        """
        import numpy as np
        
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        
        try:
            # Prepare OHLCV-like data
            ohlcv_data = self._prepare_ohlcv_data(data)
            
            # Create features
            temp_engineer = self.FeatureEngineer()
            features = temp_engineer.create_features(ohlcv_data, target_col='close')
            
            if len(features) < 50:
                raise ValueError("Insufficient data for validation after feature engineering")
            
            # Prepare training data
            X, y = self._prepare_training_data(features)
            
            # Identify categorical features
            categorical_features = self._identify_categorical_features(X)
            
            # Time series cross-validation
            tscv = self.TimeSeriesSplit(n_splits=min(3, len(X) // 20))
            
            mse_scores = []
            mae_scores = []
            mape_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train temporary model with reduced iterations for speed
                temp_params = self._best_params.copy() if self._best_params else self.config.copy()
                temp_params['iterations'] = 30  # Reduced for faster validation
                temp_params['logging_level'] = 'Silent'
                
                temp_model = self.cb.CatBoostRegressor(**temp_params)
                temp_model.fit(
                    X_train, y_train,
                    cat_features=categorical_features,
                    verbose=False,
                    plot=False
                )
                
                y_pred = temp_model.predict(X_val)
                
                # Calculate metrics
                mse = self.mean_squared_error(y_val, y_pred)
                mae = self.mean_absolute_error(y_val, y_pred)
                
                # Calculate MAPE
                mape = np.mean(np.abs((y_val - y_pred) / np.where(y_val != 0, y_val, 1))) * 100
                
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)
            
            # Store CV scores
            self._cv_scores = {
                'mse_scores': mse_scores,
                'mae_scores': mae_scores,
                'mape_scores': mape_scores
            }
            
            metrics = {
                'mse': float(np.mean(mse_scores)),
                'mae': float(np.mean(mae_scores)),
                'mape': float(np.mean(mape_scores)),
                'rmse': float(np.sqrt(np.mean(mse_scores))),
                'mse_std': float(np.std(mse_scores)),
                'mae_std': float(np.std(mae_scores)),
                'mape_std': float(np.std(mape_scores)),
                'validation_folds': len(mse_scores)
            }
            
            self.logger.info(f"CatBoost validation completed. MAPE: {metrics['mape']:.4f}% ± {metrics['mape_std']:.4f}%")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"CatBoost validation failed: {str(e)}")
            raise RuntimeError(f"CatBoost validation failed: {str(e)}")
    
    def _prepare_ohlcv_data(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare OHLCV-like data from price series for feature engineering.
        
        Args:
            data: Price series
            exog: Optional exogenous variables
            
        Returns:
            DataFrame with OHLCV-like structure
        """
        ohlcv = pd.DataFrame(index=data.index)
        ohlcv['close'] = data
        
        # Create synthetic OHLC data if not available
        ohlcv['open'] = data.shift(1).fillna(data.iloc[0])
        ohlcv['high'] = data * (1 + np.abs(np.random.normal(0, 0.005, len(data))))
        ohlcv['low'] = data * (1 - np.abs(np.random.normal(0, 0.005, len(data))))
        
        # Add synthetic volume if not available
        ohlcv['volume'] = np.random.randint(1000000, 10000000, len(data))
        
        # Add exogenous variables if provided
        if exog is not None:
            for col in exog.columns:
                if col not in ohlcv.columns:
                    # Align exog data with price data index
                    aligned_exog = exog[col].reindex(data.index, method='ffill')
                    ohlcv[col] = aligned_exog
        
        return ohlcv
    
    def _prepare_training_data(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from features.
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Tuple of (X, y) for training
        """
        # Target variable: next period return
        target_periods = self.config.get('target_periods', 1)
        
        if 'returns' not in features.columns:
            raise ValueError("Returns column not found in features")
        
        # Create target variable (future returns)
        y = features['returns'].shift(-target_periods)
        
        # Remove target-related columns from features
        feature_cols = [col for col in features.columns 
                       if not col.startswith('returns') and col != 'price' and col != 'ticker']
        
        X = features[feature_cols].copy()
        
        # Remove rows with NaN target values
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Feature selection if enabled
        if self.config.get('feature_selection', True) and len(X.columns) > 40:
            X = self._select_features(X, y)
        
        return X, y
    
    def _identify_categorical_features(self, X: pd.DataFrame) -> List[int]:
        """
        Identify categorical features in the dataset.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of column indices for categorical features
        """
        categorical_indices = []
        
        for i, col in enumerate(X.columns):
            # Check if column contains categorical-like data
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_indices.append(i)
            elif X[col].nunique() <= 10 and X[col].dtype in ['int64', 'int32']:
                # Integer columns with few unique values might be categorical
                categorical_indices.append(i)
            elif col in ['uptrend', 'high_volatility_regime', 'bull_market', 'bear_market', 'volatile_market']:
                # Known binary/categorical features from feature engineering
                categorical_indices.append(i)
        
        return categorical_indices
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 40) -> pd.DataFrame:
        """
        Select most important features using correlation and variance filtering.
        
        Args:
            X: Feature matrix
            y: Target variable
            max_features: Maximum number of features to select
            
        Returns:
            DataFrame with selected features
        """
        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1]
        X = X.drop(columns=constant_features)
        
        if len(constant_features) > 0:
            self.logger.info(f"Removed {len(constant_features)} constant features")
        
        # Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns 
                             if any(upper_triangle[column] > 0.95)]
        X = X.drop(columns=high_corr_features)
        
        if len(high_corr_features) > 0:
            self.logger.info(f"Removed {len(high_corr_features)} highly correlated features")
        
        # Select top features by correlation with target
        if len(X.columns) > max_features:
            target_corr = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = target_corr.head(max_features).index
            X = X[selected_features]
            
            self.logger.info(f"Selected top {len(selected_features)} features by target correlation")
        
        return X
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Tune hyperparameters using time series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of best parameters
        """
        # Define parameter combinations (smaller for speed)
        param_combinations = [
            {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1,
                'od_type': 'Iter',
                'od_wait': 10,
                'random_seed': self.config['random_seed'],
                'logging_level': 'Silent',
                'thread_count': self.config['thread_count']
            },
            {
                'iterations': 50,
                'learning_rate': 0.05,
                'depth': 4,
                'l2_leaf_reg': 1,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 0.5,
                'od_type': 'Iter',
                'od_wait': 5,
                'random_seed': self.config['random_seed'],
                'logging_level': 'Silent',
                'thread_count': self.config['thread_count']
            },
            {
                'iterations': 150,
                'learning_rate': 0.2,
                'depth': 8,
                'l2_leaf_reg': 5,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1.5,
                'od_type': 'Iter',
                'od_wait': 15,
                'random_seed': self.config['random_seed'],
                'logging_level': 'Silent',
                'thread_count': self.config['thread_count']
            }
        ]
        
        # Use time series cross-validation
        tscv = self.TimeSeriesSplit(n_splits=3)
        
        # Identify categorical features
        categorical_features = self._identify_categorical_features(X)
        
        best_score = float('inf')
        best_params = param_combinations[0]
        
        for params in param_combinations:
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model = self.cb.CatBoostRegressor(**params)
                model.fit(
                    X_train, y_train,
                    cat_features=categorical_features,
                    verbose=False,
                    plot=False
                )
                
                # Predict and score
                y_pred = model.predict(X_val)
                score = self.mean_squared_error(y_val, y_pred)
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
        
        self.logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the fitted CatBoost model.
        
        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()
        
        catboost_info = {
            'best_params': self._best_params,
            'feature_columns': self._feature_columns,
            'categorical_features': self._categorical_features,
            'num_features': len(self._feature_columns) if self._feature_columns else 0,
            'num_categorical': len(self._categorical_features) if self._categorical_features else 0,
            'feature_importance': self._feature_importance,
            'cv_scores': self._cv_scores,
            'config': self.config
        }
        
        if self._fitted_model is not None:
            try:
                catboost_info.update({
                    'tree_count': self._fitted_model.tree_count_,
                    'learning_rate': self._fitted_model.learning_rate_,
                    'depth': self._fitted_model.get_param('depth'),
                    'l2_leaf_reg': self._fitted_model.get_param('l2_leaf_reg')
                })
            except Exception as e:
                self.logger.debug(f"Could not extract additional model info: {e}")
        
        base_info.update(catboost_info)
        return base_info
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self._feature_importance:
            return {}
        
        # Sort by importance and return top N
        sorted_features = sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features[:top_n])


# Register models with the factory
ModelFactory.register_model('arima', ARIMAForecaster)
class LSTMForecaster(BaseForecaster):
    """
    LSTM (Long Short-Term Memory) neural network forecasting model.
    
    This implementation uses TensorFlow/Keras to create a multi-layer LSTM architecture
    with dropout for time series forecasting. It includes sequence data preparation
    and proper handling of time series input.
    """
    
    def __init__(self, model_name: str = "lstm", **kwargs):
        """
        Initialize LSTM forecaster.
        
        Args:
            model_name: Name identifier for the model
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name)
        
        # Import required libraries
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            from sklearn.preprocessing import MinMaxScaler
            import warnings
            
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            warnings.filterwarnings('ignore')
            
            self.tf = tf
            self.keras = keras
            self.layers = layers
            self.MinMaxScaler = MinMaxScaler
            
        except ImportError as e:
            raise ImportError(f"TensorFlow/Keras not available: {e}")
        
        # Get configuration
        try:
            from forecasting_config import get_model_config
            config = get_model_config('lstm')
        except ImportError:
            # Fallback if config module not available
            config = None
        
        if config:
            self.config = config.params
        else:
            # Default configuration
            self.config = {
                'sequence_length': 60,  # Number of time steps to look back
                'lstm_units': [64, 32],  # Units in each LSTM layer
                'dropout_rate': 0.2,
                'dense_units': 16,
                'batch_size': 32,
                'epochs': 100,
                'validation_split': 0.2,
                'early_stopping_patience': 10,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'loss': 'mse',
                'metrics': ['mae'],
                'use_technical_indicators': True,
                'feature_columns': ['returns', 'volume', 'volatility']
            }
        
        # Override with any provided kwargs
        self.config.update(kwargs)
        
        # Model state
        self._model = None
        self._scaler = None
        self._feature_scaler = None
        self._sequence_length = self.config['sequence_length']
        self._feature_columns = self.config.get('feature_columns', ['returns'])
        self._model_checkpoint_path = None
        
    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        Fit LSTM model with sequence data preparation and early stopping.
        
        Args:
            data: Time series data for training
            exog: Optional exogenous variables (technical indicators)
            
        Raises:
            ValueError: If data is insufficient or invalid
            RuntimeError: If model training fails
        """
        import time
        import tempfile
        import os
        start_time = time.time()
        
        try:
            # Validate input data
            min_points = self._sequence_length + 50  # Need extra points for sequences
            self._validate_data(data, min_points=min_points)
            
            # Store training data
            self._training_data = data.copy()
            
            # Prepare features and sequences
            features_df = self._prepare_features(data, exog)
            X, y = self._create_sequences(features_df)
            
            if len(X) < 50:
                raise ValueError(f"Insufficient sequences after preparation: {len(X)}")
            
            self.logger.info(f"Created {len(X)} sequences with {X.shape[2]} features")
            
            # Build model architecture
            self._model = self._build_model(input_shape=(X.shape[1], X.shape[2]))
            
            # Set up callbacks
            callbacks = self._setup_callbacks()
            
            # Train the model
            self.logger.info("Starting LSTM model training...")
            
            history = self._model.fit(
                X, y,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_split=self.config['validation_split'],
                callbacks=callbacks,
                verbose=0  # Suppress training output
            )
            
            # Mark as fitted
            self.is_fitted = True
            
            training_time = time.time() - start_time
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
            
            self.logger.info(
                f"LSTM model trained successfully in {training_time:.2f}s. "
                f"Final loss: {final_loss:.6f}"
                + (f", Val loss: {final_val_loss:.6f}" if final_val_loss else "")
            )
            
        except ValueError as e:
            # Re-raise ValueError as-is (for data validation errors)
            self.logger.error(f"LSTM model validation failed: {str(e)}")
            raise e
        except Exception as e:
            self.logger.error(f"LSTM model training failed: {str(e)}")
            raise RuntimeError(f"LSTM model training failed: {str(e)}")
    
    def predict(self, periods: int, exog: Optional[pd.DataFrame] = None) -> float:
        """
        Generate forecast for the specified number of periods with proper sequence handling.
        
        Args:
            periods: Number of periods to forecast
            exog: Optional exogenous variables for prediction periods
            
        Returns:
            Expected return for the forecast period
            
        Raises:
            RuntimeError: If model is not fitted or prediction fails
        """
        import time
        import numpy as np
        start_time = time.time()
        
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before making predictions")
        
        try:
            # Get the last sequence from training data for prediction
            last_sequence = self._get_last_sequence_for_prediction(exog)
            
            # Generate multi-step forecast
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(periods):
                # Predict next value
                pred = self._model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
                pred_value = pred[0, 0]
                predictions.append(pred_value)
                
                # Update sequence for next prediction
                # Remove first element and add prediction
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1, 0] = pred_value  # Update the target feature
                
                # If we have exogenous features, we'd need to update them too
                # For now, we'll keep the last known values
            
            # Inverse transform the final prediction
            final_prediction = predictions[-1]
            if self._scaler is not None:
                # Create dummy array for inverse transform
                dummy = np.zeros((1, len(self._feature_columns)))
                dummy[0, 0] = final_prediction
                final_prediction = self._scaler.inverse_transform(dummy)[0, 0]
            
            prediction_time = time.time() - start_time
            self.logger.debug(f"LSTM prediction completed in {prediction_time:.4f}s")
            
            return float(final_prediction)
            
        except Exception as e:
            self.logger.error(f"LSTM prediction failed: {str(e)}")
            raise RuntimeError(f"LSTM prediction failed: {str(e)}")
    
    def validate(self, data: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        """
        Perform model validation using time series cross-validation.
        
        Args:
            data: Time series data for validation
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing validation metrics (MSE, MAE, MAPE)
            
        Raises:
            ValueError: If validation parameters are invalid
        """
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        
        try:
            # Prepare features
            features_df = self._prepare_features(data)
            
            # Create sequences
            X, y = self._create_sequences(features_df)
            
            if len(X) < 20:
                raise ValueError("Insufficient sequences for validation")
            
            # Split data for time series validation
            split_point = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            if len(X_train) < 10 or len(X_test) < 5:
                raise ValueError("Insufficient data for validation split")
            
            # Create and train temporary model
            temp_model = self._build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # Train with reduced epochs for validation
            temp_model.fit(
                X_train, y_train,
                batch_size=min(self.config['batch_size'], len(X_train)),
                epochs=min(50, self.config['epochs']),
                validation_split=0.1,
                verbose=0
            )
            
            # Generate predictions
            predictions = temp_model.predict(X_test, verbose=0).flatten()
            actuals = y_test.flatten()
            
            # Inverse transform if scaler was used
            if self._scaler is not None:
                # Create dummy arrays for inverse transform
                pred_dummy = np.zeros((len(predictions), len(self._feature_columns)))
                actual_dummy = np.zeros((len(actuals), len(self._feature_columns)))
                pred_dummy[:, 0] = predictions
                actual_dummy[:, 0] = actuals
                
                predictions = self._scaler.inverse_transform(pred_dummy)[:, 0]
                actuals = self._scaler.inverse_transform(actual_dummy)[:, 0]
            
            # Calculate metrics
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((actuals - predictions) / np.where(actuals != 0, actuals, 1))) * 100
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'mape': float(mape),
                'rmse': float(np.sqrt(mse)),
                'validation_points': len(y_test)
            }
            
            self.logger.info(f"LSTM validation completed. MAPE: {mape:.4f}%, MAE: {mae:.6f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"LSTM validation failed: {str(e)}")
            raise RuntimeError(f"LSTM validation failed: {str(e)}")
    
    def _prepare_features(self, data: pd.Series, exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare features for LSTM training including technical indicators.
        
        Args:
            data: Time series data
            exog: Optional exogenous variables
            
        Returns:
            DataFrame with prepared features
        """
        import numpy as np
        
        # Start with returns as the primary feature
        features_df = pd.DataFrame(index=data.index)
        features_df['returns'] = data.values
        
        # Add technical indicators if enabled
        if self.config.get('use_technical_indicators', True):
            features_df = self._add_technical_indicators(features_df, data)
        
        # Add exogenous variables if provided
        if exog is not None:
            # Align exog data with features_df
            aligned_exog = exog.reindex(features_df.index, method='ffill')
            for col in aligned_exog.columns:
                if not aligned_exog[col].isnull().all():
                    features_df[f'exog_{col}'] = aligned_exog[col]
        
        # Handle missing values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        features_df = features_df.dropna()
        
        # Update feature columns list
        self._feature_columns = list(features_df.columns)
        
        # Scale features
        self._scaler = self.MinMaxScaler()
        scaled_features = self._scaler.fit_transform(features_df.values)
        
        # Create scaled DataFrame
        scaled_df = pd.DataFrame(
            scaled_features,
            index=features_df.index,
            columns=features_df.columns
        )
        
        return scaled_df
    
    def _add_technical_indicators(self, features_df: pd.DataFrame, data: pd.Series) -> pd.DataFrame:
        """
        Add technical indicators as features.
        
        Args:
            features_df: Existing features DataFrame
            data: Original time series data
            
        Returns:
            DataFrame with additional technical indicators
        """
        import numpy as np
        
        try:
            # Moving averages
            for window in [5, 10, 20]:
                features_df[f'ma_{window}'] = data.rolling(window=window).mean()
                features_df[f'ma_ratio_{window}'] = data / features_df[f'ma_{window}']
            
            # Volatility (rolling standard deviation)
            for window in [5, 10, 20]:
                features_df[f'volatility_{window}'] = data.rolling(window=window).std()
            
            # RSI-like indicator (simplified)
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # Momentum indicators
            for period in [5, 10]:
                features_df[f'momentum_{period}'] = data.diff(period)
                features_df[f'roc_{period}'] = data.pct_change(period)
            
            # Bollinger Bands
            bb_window = 20
            bb_ma = data.rolling(window=bb_window).mean()
            bb_std = data.rolling(window=bb_window).std()
            bb_upper = bb_ma + (bb_std * 2)
            bb_lower = bb_ma - (bb_std * 2)
            features_df['bb_upper'] = bb_upper
            features_df['bb_lower'] = bb_lower
            features_df['bb_position'] = (data - bb_lower) / (bb_upper - bb_lower)
            
        except Exception as e:
            self.logger.warning(f"Error adding technical indicators: {e}")
        
        return features_df
    
    def _create_sequences(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            features_df: DataFrame with prepared features
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        import numpy as np
        
        data = features_df.values
        X, y = [], []
        
        for i in range(self._sequence_length, len(data)):
            # Input sequence (all features)
            X.append(data[i-self._sequence_length:i])
            # Target (next value of the first feature - returns)
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> 'keras.Model':
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input sequences (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        model = self.keras.Sequential()
        
        # First LSTM layer
        lstm_units = self.config['lstm_units']
        if isinstance(lstm_units, list) and len(lstm_units) > 0:
            first_units = lstm_units[0]
            return_sequences = len(lstm_units) > 1
        else:
            first_units = 64
            return_sequences = False
        
        model.add(self.layers.LSTM(
            first_units,
            return_sequences=return_sequences,
            input_shape=input_shape,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['dropout_rate']
        ))
        
        # Additional LSTM layers
        if isinstance(lstm_units, list) and len(lstm_units) > 1:
            for i, units in enumerate(lstm_units[1:], 1):
                return_sequences = i < len(lstm_units) - 1
                model.add(self.layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config['dropout_rate'],
                    recurrent_dropout=self.config['dropout_rate']
                ))
        
        # Dense layers
        model.add(self.layers.Dense(
            self.config['dense_units'],
            activation='relu'
        ))
        model.add(self.layers.Dropout(self.config['dropout_rate']))
        
        # Output layer
        model.add(self.layers.Dense(1, activation='linear'))
        
        # Compile model
        optimizer = self.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss=self.config['loss'],
            metrics=self.config['metrics']
        )
        
        return model
    
    def _setup_callbacks(self) -> List:
        """
        Set up training callbacks including early stopping and model checkpointing.
        
        Returns:
            List of Keras callbacks
        """
        import tempfile
        import os
        
        callbacks = []
        
        # Early stopping
        early_stopping = self.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=0
        )
        callbacks.append(early_stopping)
        
        # Model checkpointing
        checkpoint_dir = tempfile.mkdtemp()
        self._model_checkpoint_path = os.path.join(checkpoint_dir, 'lstm_model.h5')
        
        checkpoint = self.keras.callbacks.ModelCheckpoint(
            self._model_checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        callbacks.append(checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = self.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def _get_last_sequence_for_prediction(self, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Get the last sequence from training data for making predictions.
        
        Args:
            exog: Optional exogenous variables for prediction
            
        Returns:
            Last sequence array for prediction
        """
        import numpy as np
        
        if self._training_data is None:
            raise RuntimeError("No training data available for prediction")
        
        # Prepare features for the last sequence
        features_df = self._prepare_features(self._training_data, exog)
        
        # Get the last sequence
        if len(features_df) < self._sequence_length:
            raise RuntimeError(f"Insufficient data for sequence. Need {self._sequence_length}, got {len(features_df)}")
        
        last_sequence = features_df.values[-self._sequence_length:]
        
        return last_sequence
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the fitted LSTM model.
        
        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()
        
        lstm_info = {
            'sequence_length': self._sequence_length,
            'feature_columns': self._feature_columns,
            'config': self.config
        }
        
        if self._model is not None:
            try:
                lstm_info.update({
                    'total_params': self._model.count_params(),
                    'trainable_params': sum([self.tf.keras.backend.count_params(w) for w in self._model.trainable_weights]),
                    'model_layers': len(self._model.layers),
                    'input_shape': self._model.input_shape,
                    'output_shape': self._model.output_shape
                })
            except Exception as e:
                self.logger.debug(f"Could not extract additional model info: {e}")
        
        base_info.update(lstm_info)
        return base_info


ModelFactory.register_model('arima', ARIMAForecaster)
ModelFactory.register_model('sarimax', SARIMAXForecaster)
ModelFactory.register_model('xgboost', XGBoostForecaster)
ModelFactory.register_model('lightgbm', LightGBMForecaster)
ModelFactory.register_model('catboost', CatBoostForecaster)
ModelFactory.register_model('lstm', LSTMForecaster)
ModelFactory.register_model('ensemble', EnsembleForecaster)

# Initialize the module logger
forecasting_logger.info("Forecasting models module initialized")