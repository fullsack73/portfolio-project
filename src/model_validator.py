"""
Model Validation and Selection System

This module provides comprehensive model validation, selection, and performance
tracking capabilities for the advanced forecasting system.
"""

import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os

from forecasting_models import BaseForecaster, ModelPerformance, ForecastResult


# Configure logging
logger = logging.getLogger('model_validator')


@dataclass
class ValidationResult:
    """Data class for storing validation results."""
    model_name: str
    ticker: str
    mse: float
    mae: float
    mape: float
    rmse: float
    validation_points: int
    training_time: float
    prediction_time: float
    cross_validation_scores: List[float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class ModelRanking:
    """Data class for model ranking results."""
    ticker: str
    ranked_models: List[Tuple[str, float]]  # (model_name, score)
    best_model: str
    best_score: float
    ensemble_candidates: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class TimeSeriesCrossValidator:
    """
    Time series cross-validation implementation.
    
    Provides walk-forward validation and expanding window validation
    specifically designed for time series forecasting models.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: float = 0.2,
                 gap: int = 0,
                 expanding_window: bool = True):
        """
        Initialize time series cross-validator.
        
        Args:
            n_splits: Number of cross-validation splits
            test_size: Proportion of data for testing in each split
            gap: Number of periods to skip between train and test
            expanding_window: If True, use expanding window; if False, use sliding window
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding_window = expanding_window
        self.logger = logging.getLogger('model_validator.cv')
        
    def split(self, data: pd.Series) -> List[Tuple[pd.Series, pd.Series]]:
        """
        Generate train/test splits for time series cross-validation.
        
        Args:
            data: Time series data to split
            
        Returns:
            List of (train, test) tuples
            
        Raises:
            ValueError: If data is insufficient for cross-validation
        """
        if len(data) < 100:
            raise ValueError(f"Insufficient data for cross-validation: {len(data)} points")
        
        splits = []
        n_test = max(int(len(data) * self.test_size / self.n_splits), 10)
        
        # Calculate split points
        total_test_size = n_test * self.n_splits
        if total_test_size >= len(data) * 0.8:
            # Adjust if test size is too large
            n_test = max(int(len(data) * 0.6 / self.n_splits), 5)
            total_test_size = n_test * self.n_splits
        
        min_train_size = max(50, int(len(data) * 0.3))
        
        for i in range(self.n_splits):
            # Calculate test period
            test_end = len(data) - (self.n_splits - i - 1) * n_test
            test_start = test_end - n_test
            
            # Calculate train period
            if self.expanding_window:
                train_start = 0
            else:
                # Sliding window
                train_size = max(min_train_size, test_start - self.gap)
                train_start = max(0, test_start - self.gap - train_size)
            
            train_end = max(train_start + min_train_size, test_start - self.gap)
            
            # Ensure we have enough training data
            if train_end - train_start < min_train_size:
                continue
                
            # Create splits
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            if len(train_data) >= min_train_size and len(test_data) >= 5:
                splits.append((train_data, test_data))
        
        if not splits:
            raise ValueError("Could not create valid cross-validation splits")
        
        self.logger.debug(f"Created {len(splits)} CV splits with avg train size: "
                         f"{np.mean([len(s[0]) for s in splits]):.0f}, "
                         f"avg test size: {np.mean([len(s[1]) for s in splits]):.0f}")
        
        return splits
    
    def validate_model(self, 
                      model: BaseForecaster, 
                      data: pd.Series,
                      exog: Optional[pd.DataFrame] = None) -> ValidationResult:
        """
        Perform cross-validation on a single model.
        
        Args:
            model: Forecasting model to validate
            data: Time series data for validation
            exog: Optional exogenous variables
            
        Returns:
            ValidationResult containing validation metrics
            
        Raises:
            RuntimeError: If validation fails
        """
        start_time = time.time()
        
        try:
            # Generate cross-validation splits
            splits = self.split(data)
            
            all_predictions = []
            all_actuals = []
            cv_scores = []
            training_times = []
            prediction_times = []
            
            for i, (train_data, test_data) in enumerate(splits):
                try:
                    # Prepare exogenous data for this split if provided
                    train_exog = None
                    test_exog = None
                    if exog is not None:
                        train_exog = exog.loc[train_data.index] if not exog.empty else None
                        test_exog = exog.loc[test_data.index] if not exog.empty else None
                    
                    # Create a fresh model instance for this fold
                    fold_model = model.__class__(model_name=f"{model.model_name}_fold_{i}")
                    if hasattr(model, 'config'):
                        fold_model.config = model.config.copy()
                    
                    # Train model
                    train_start = time.time()
                    fold_model.fit(train_data, exog=train_exog)
                    train_time = time.time() - train_start
                    training_times.append(train_time)
                    
                    # Generate predictions
                    pred_start = time.time()
                    fold_predictions = []
                    fold_actuals = list(test_data.values)
                    
                    # Walk-forward prediction within the test period
                    current_train = train_data.copy()
                    current_exog = train_exog.copy() if train_exog is not None else None
                    
                    for j in range(len(test_data)):
                        try:
                            # Predict next value
                            pred_exog = None
                            if test_exog is not None and j < len(test_exog):
                                pred_exog = test_exog.iloc[[j]]
                            
                            pred = fold_model.predict(periods=1, exog=pred_exog)
                            fold_predictions.append(pred)
                            
                            # Update training data for next prediction (if not last)
                            if j < len(test_data) - 1:
                                new_point = pd.Series([fold_actuals[j]], 
                                                    index=[test_data.index[j]])
                                current_train = pd.concat([current_train, new_point])
                                
                                if current_exog is not None and test_exog is not None:
                                    new_exog = test_exog.iloc[[j]]
                                    current_exog = pd.concat([current_exog, new_exog])
                                
                                # Refit model with updated data
                                fold_model.fit(current_train, exog=current_exog)
                                
                        except Exception as e:
                            self.logger.warning(f"Prediction failed at step {j} in fold {i}: {e}")
                            # Use last prediction or mean as fallback
                            if fold_predictions:
                                fold_predictions.append(fold_predictions[-1])
                            else:
                                fold_predictions.append(current_train.mean())
                    
                    pred_time = time.time() - pred_start
                    prediction_times.append(pred_time)
                    
                    # Calculate fold metrics
                    fold_predictions = np.array(fold_predictions)
                    fold_actuals = np.array(fold_actuals)
                    
                    fold_mse = mean_squared_error(fold_actuals, fold_predictions)
                    cv_scores.append(fold_mse)
                    
                    all_predictions.extend(fold_predictions)
                    all_actuals.extend(fold_actuals)
                    
                    self.logger.debug(f"Fold {i+1}/{len(splits)} completed. MSE: {fold_mse:.6f}")
                    
                except Exception as e:
                    self.logger.warning(f"Fold {i} failed: {e}")
                    continue
            
            if not all_predictions:
                raise RuntimeError("All cross-validation folds failed")
            
            # Calculate overall metrics
            all_predictions = np.array(all_predictions)
            all_actuals = np.array(all_actuals)
            
            mse = mean_squared_error(all_actuals, all_predictions)
            mae = mean_absolute_error(all_actuals, all_predictions)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE with protection against division by zero
            mape = np.mean(np.abs((all_actuals - all_predictions) / 
                                np.where(np.abs(all_actuals) > 1e-8, all_actuals, 1))) * 100
            
            total_time = time.time() - start_time
            avg_training_time = np.mean(training_times) if training_times else 0
            avg_prediction_time = np.mean(prediction_times) if prediction_times else 0
            
            result = ValidationResult(
                model_name=model.model_name,
                ticker="validation",  # Will be set by caller
                mse=float(mse),
                mae=float(mae),
                mape=float(mape),
                rmse=float(rmse),
                validation_points=len(all_predictions),
                training_time=float(avg_training_time),
                prediction_time=float(avg_prediction_time),
                cross_validation_scores=cv_scores,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"Cross-validation completed for {model.model_name}. "
                           f"MAPE: {mape:.4f}%, MAE: {mae:.6f}, "
                           f"Total time: {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed for {model.model_name}: {e}")
            raise RuntimeError(f"Cross-validation failed: {e}")


class ModelSelector:
    """
    Model selection and ranking system.
    
    Provides functionality to compare models, rank them based on performance,
    and select the best model for each ticker.
    """
    
    def __init__(self, 
                 primary_metric: str = 'mape',
                 secondary_metric: str = 'mae',
                 performance_threshold: float = 20.0,
                 ensemble_threshold: int = 3):
        """
        Initialize model selector.
        
        Args:
            primary_metric: Primary metric for model ranking ('mape', 'mae', 'mse')
            secondary_metric: Secondary metric for tie-breaking
            performance_threshold: Maximum acceptable MAPE for model selection
            ensemble_threshold: Minimum number of models required for ensemble
        """
        self.primary_metric = primary_metric
        self.secondary_metric = secondary_metric
        self.performance_threshold = performance_threshold
        self.ensemble_threshold = ensemble_threshold
        self.logger = logging.getLogger('model_validator.selector')
        
    def rank_models(self, 
                   validation_results: List[ValidationResult],
                   ticker: str) -> ModelRanking:
        """
        Rank models based on validation performance.
        
        Args:
            validation_results: List of validation results for different models
            ticker: Ticker symbol for this ranking
            
        Returns:
            ModelRanking object with ranked models and selection
            
        Raises:
            ValueError: If no valid models are provided
        """
        if not validation_results:
            raise ValueError("No validation results provided for ranking")
        
        # Filter models that meet performance threshold
        valid_models = [
            result for result in validation_results
            if result.mape <= self.performance_threshold
        ]
        
        if not valid_models:
            self.logger.warning(f"No models meet performance threshold ({self.performance_threshold}%) "
                              f"for {ticker}. Using best available model.")
            valid_models = validation_results
        
        # Sort by primary metric (lower is better for all our metrics)
        ranked_models = sorted(valid_models, 
                             key=lambda x: (getattr(x, self.primary_metric),
                                          getattr(x, self.secondary_metric)))
        
        # Create ranking list with scores
        ranking_list = [(model.model_name, getattr(model, self.primary_metric)) 
                       for model in ranked_models]
        
        # Select best model
        best_model = ranked_models[0]
        
        # Select ensemble candidates (top models that meet threshold)
        ensemble_candidates = [
            model.model_name for model in ranked_models[:self.ensemble_threshold]
            if model.mape <= self.performance_threshold
        ]
        
        # Ensure we have at least 2 models for ensemble if possible
        if len(ensemble_candidates) < 2 and len(ranked_models) >= 2:
            ensemble_candidates = [model.model_name for model in ranked_models[:2]]
        
        ranking = ModelRanking(
            ticker=ticker,
            ranked_models=ranking_list,
            best_model=best_model.model_name,
            best_score=getattr(best_model, self.primary_metric),
            ensemble_candidates=ensemble_candidates,
            timestamp=datetime.now()
        )
        
        self.logger.info(f"Model ranking for {ticker}: Best={best_model.model_name} "
                        f"({self.primary_metric}={getattr(best_model, self.primary_metric):.4f}), "
                        f"Ensemble candidates: {len(ensemble_candidates)}")
        
        return ranking
    
    def select_best_model(self, 
                         validation_results: List[ValidationResult],
                         ticker: str) -> Tuple[str, float]:
        """
        Select the best performing model for a ticker.
        
        Args:
            validation_results: List of validation results
            ticker: Ticker symbol
            
        Returns:
            Tuple of (best_model_name, best_score)
        """
        ranking = self.rank_models(validation_results, ticker)
        return ranking.best_model, ranking.best_score
    
    def get_ensemble_candidates(self, 
                              validation_results: List[ValidationResult],
                              ticker: str) -> List[str]:
        """
        Get list of models suitable for ensemble.
        
        Args:
            validation_results: List of validation results
            ticker: Ticker symbol
            
        Returns:
            List of model names suitable for ensemble
        """
        ranking = self.rank_models(validation_results, ticker)
        return ranking.ensemble_candidates


class ModelValidator:
    """
    Main model validation and selection system.
    
    Coordinates cross-validation, model comparison, and selection processes.
    """
    
    def __init__(self, 
                 cv_config: Optional[Dict[str, Any]] = None,
                 selector_config: Optional[Dict[str, Any]] = None,
                 cache_results: bool = True,
                 results_dir: str = "logs/model_validation"):
        """
        Initialize model validator.
        
        Args:
            cv_config: Configuration for cross-validation
            selector_config: Configuration for model selection
            cache_results: Whether to cache validation results
            results_dir: Directory to store validation results
        """
        # Cross-validation configuration
        cv_defaults = {
            'n_splits': 5,
            'test_size': 0.2,
            'gap': 0,
            'expanding_window': True
        }
        cv_config = cv_config or {}
        cv_config = {**cv_defaults, **cv_config}
        
        # Model selection configuration
        selector_defaults = {
            'primary_metric': 'mape',
            'secondary_metric': 'mae',
            'performance_threshold': 20.0,
            'ensemble_threshold': 3
        }
        selector_config = selector_config or {}
        selector_config = {**selector_defaults, **selector_config}
        
        # Initialize components
        self.cross_validator = TimeSeriesCrossValidator(**cv_config)
        self.model_selector = ModelSelector(**selector_config)
        
        # Configuration
        self.cache_results = cache_results
        self.results_dir = results_dir
        
        # Create results directory
        if self.cache_results:
            os.makedirs(self.results_dir, exist_ok=True)
        
        # Logging
        self.logger = logging.getLogger('model_validator')
        
        # Results storage
        self.validation_cache: Dict[str, List[ValidationResult]] = {}
        self.ranking_cache: Dict[str, ModelRanking] = {}
    
    def validate_models(self, 
                       models: List[BaseForecaster],
                       data: pd.Series,
                       ticker: str,
                       exog: Optional[pd.DataFrame] = None,
                       force_revalidation: bool = False) -> List[ValidationResult]:
        """
        Validate multiple models on the same dataset.
        
        Args:
            models: List of forecasting models to validate
            data: Time series data for validation
            ticker: Ticker symbol
            exog: Optional exogenous variables
            force_revalidation: Force revalidation even if cached results exist
            
        Returns:
            List of validation results for all models
            
        Raises:
            ValueError: If no models provided or data is invalid
        """
        if not models:
            raise ValueError("No models provided for validation")
        
        if data is None or data.empty:
            raise ValueError("Invalid data provided for validation")
        
        # Check cache
        cache_key = f"{ticker}_{len(data)}_{hash(str(data.index))}"
        if not force_revalidation and cache_key in self.validation_cache:
            self.logger.info(f"Using cached validation results for {ticker}")
            return self.validation_cache[cache_key]
        
        self.logger.info(f"Starting validation for {len(models)} models on {ticker}")
        
        validation_results = []
        
        for model in models:
            try:
                self.logger.info(f"Validating {model.model_name} for {ticker}")
                
                result = self.cross_validator.validate_model(model, data, exog)
                result.ticker = ticker  # Set the ticker
                
                validation_results.append(result)
                
                self.logger.info(f"Validation completed for {model.model_name}: "
                               f"MAPE={result.mape:.4f}%, MAE={result.mae:.6f}")
                
            except Exception as e:
                self.logger.error(f"Validation failed for {model.model_name} on {ticker}: {e}")
                continue
        
        if not validation_results:
            raise RuntimeError(f"All model validations failed for {ticker}")
        
        # Cache results
        if self.cache_results:
            self.validation_cache[cache_key] = validation_results
            self._save_validation_results(validation_results, ticker)
        
        self.logger.info(f"Validation completed for {ticker}. "
                        f"Successfully validated {len(validation_results)}/{len(models)} models")
        
        return validation_results
    
    def select_best_models(self, 
                          models: List[BaseForecaster],
                          data: pd.Series,
                          ticker: str,
                          exog: Optional[pd.DataFrame] = None) -> ModelRanking:
        """
        Validate models and select the best performing ones.
        
        Args:
            models: List of forecasting models
            data: Time series data
            ticker: Ticker symbol
            exog: Optional exogenous variables
            
        Returns:
            ModelRanking with best model selection and ensemble candidates
        """
        # Validate all models
        validation_results = self.validate_models(models, data, ticker, exog)
        
        # Rank and select models
        ranking = self.model_selector.rank_models(validation_results, ticker)
        
        # Cache ranking
        if self.cache_results:
            self.ranking_cache[ticker] = ranking
            self._save_ranking_results(ranking)
        
        return ranking
    
    def get_performance_comparison(self, 
                                 validation_results: List[ValidationResult]) -> pd.DataFrame:
        """
        Create a performance comparison DataFrame.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            DataFrame with model performance comparison
        """
        if not validation_results:
            return pd.DataFrame()
        
        comparison_data = []
        for result in validation_results:
            comparison_data.append({
                'Model': result.model_name,
                'MAPE (%)': result.mape,
                'MAE': result.mae,
                'MSE': result.mse,
                'RMSE': result.rmse,
                'Training Time (s)': result.training_time,
                'Prediction Time (s)': result.prediction_time,
                'Validation Points': result.validation_points,
                'CV Std': np.std(result.cross_validation_scores) if result.cross_validation_scores else 0
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('MAPE (%)')
        return df
    
    def _save_validation_results(self, 
                               validation_results: List[ValidationResult],
                               ticker: str) -> None:
        """Save validation results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_{ticker}_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            results_data = [result.to_dict() for result in validation_results]
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
                
            self.logger.debug(f"Validation results saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save validation results: {e}")
    
    def _save_ranking_results(self, ranking: ModelRanking) -> None:
        """Save ranking results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ranking_{ranking.ticker}_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(ranking.to_dict(), f, indent=2)
                
            self.logger.debug(f"Ranking results saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save ranking results: {e}")