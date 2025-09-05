"""
Main batch forecasting system for processing multiple tickers simultaneously.

This module provides the core BatchForecastingSystem class that orchestrates
batch processing, feature extraction, model training, and prediction.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

from batch_forecasting_config import BatchPerformanceMetrics
from src.batch_performance_monitor import BatchPerformanceMonitor
from simple_batch_manager import SimpleBatchManager
from src.fallback_coordinator import FallbackCoordinator
from src.batch_error_handler import BatchContext
from cache_manager import cached

logger = logging.getLogger(__name__)


class BatchForecastingSystem:
    """
    Main system for batch-based forecasting with intelligent grouping and fallback.
    
    This class orchestrates the entire batch forecasting pipeline:
    1. Batch creation and management
    2. Feature extraction (shared and individual)
    3. Model training and prediction
    4. Error handling and fallback
    5. Performance monitoring
    """
    
    def __init__(self, config: Optional[BatchForecastingConfig] = None):
        """
        Initialize batch forecasting system.
        
        Args:
            config: Batch forecasting configuration
        """
        self.config = config or BatchForecastingConfig()
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid batch forecasting configuration")
        
        # Initialize components
        self.batch_manager = SimpleBatchManager(self.config.batch)
        self.feature_extractor = None  # Will be initialized when needed
        self.batch_models = {}  # Will be initialized when needed
        self.fallback_coordinator = FallbackCoordinator(self.config.fallback)
        self.performance_monitor = BatchPerformanceMonitor()
        
        self.logger = logging.getLogger('batch_forecasting.system')
        self.logger.info("BatchForecastingSystem initialized")
    
    @cached(l1_ttl=1800, l2_ttl=7200)
    def forecast_batch_returns(self, 
                              tickers: List[str], 
                              data: pd.DataFrame, 
                              periods: int = 1) -> Dict[str, float]:
        """
        Main entry point for batch forecasting.
        
        Args:
            tickers: List of ticker symbols
            data: Historical price data with tickers as columns
            periods: Forecast horizon (currently only supports 1)
            
        Returns:
            Dictionary mapping tickers to expected returns
        """
        start_time = time.time()
        self.logger.info(f"Starting batch forecasting for {len(tickers)} tickers")
        
        # Validate inputs
        if not tickers:
            self.logger.warning("Empty ticker list provided")
            return {}
        
        if data.empty:
            self.logger.warning("Empty data provided")
            return {ticker: 0.05 for ticker in tickers}  # Conservative default
        
        # Check if batch processing is enabled
        if not self.config.batch.enable_batch_processing:
            self.logger.info("Batch processing disabled, falling back to individual processing")
            return self._fallback_to_individual_processing(tickers, data, periods)
        
        try:
            # Create batches
            batches = self.batch_manager.create_batches(
                tickers, 
                memory_limit_mb=self.config.fallback.memory_threshold_mb
            )
            
            # Validate batches
            if not self.batch_manager.validate_batches(batches, tickers):
                raise ValueError("Batch validation failed")
            
            # Process each batch
            all_forecasts = {}
            batch_metrics = []
            
            for i, ticker_batch in enumerate(batches):
                self.logger.info(f"Processing batch {i+1}/{len(batches)} with {len(ticker_batch)} tickers")
                
                try:
                    batch_forecasts, metrics = self._process_single_batch(
                        ticker_batch, data, periods, batch_id=i
                    )
                    all_forecasts.update(batch_forecasts)
                    batch_metrics.append(metrics)
                    
                except Exception as e:
                    self.logger.error(f"Batch {i+1} processing failed: {e}")
                    
                    # Try fallback for this batch
                    fallback_forecasts = self._handle_batch_failure(ticker_batch, data, periods, e)
                    all_forecasts.update(fallback_forecasts)
            
            # Ensure all tickers have forecasts
            for ticker in tickers:
                if ticker not in all_forecasts:
                    self.logger.warning(f"No forecast generated for {ticker}, using default")
                    all_forecasts[ticker] = 0.05
            
            
            
            self.logger.info(f"Batch forecasting completed in {total_time:.2f}s for {len(tickers)} tickers")
            return all_forecasts
            
        except Exception as e:
            self.logger.error(f"Batch forecasting system failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Complete fallback to individual processing
            return self._fallback_to_individual_processing(tickers, data, periods)
    
    def _process_single_batch(self, 
                             ticker_group: List[str], 
                             data: pd.DataFrame,
                             periods: int,
                             batch_id: int = 0) -> Tuple[Dict[str, float], BatchPerformanceMetrics]:
        """
        Process a single batch of tickers.
        
        Args:
            ticker_group: List of tickers in this batch
            data: Historical price data
            periods: Forecast horizon
            batch_id: Batch identifier for logging
            
        Returns:
            Tuple of (forecasts, performance_metrics)
        """
        self.performance_monitor.start_batch(len(ticker_group))
        feature_extraction_time = 0
        model_training_time = 0
        prediction_time = 0
        model_used = ""

        try:
            # Extract batch data
            batch_data = self._extract_batch_data(ticker_group, data)
            if batch_data.empty:
                raise ValueError("No valid data for batch")
            
            # Feature extraction
            feature_start = time.time()
            individual_features, shared_features = self._extract_batch_features(ticker_group, batch_data)
            feature_extraction_time = time.time() - feature_start
            
            # Model training and prediction
            model_start = time.time()
            forecasts, model_used = self._train_and_predict_batch(
                ticker_group, individual_features, shared_features, periods
            )
            model_time = time.time() - model_start
            model_training_time = model_time * 0.7  # Estimate training portion
            prediction_time = model_time * 0.3  # Estimate prediction portion
            
            self.performance_monitor.end_batch(model_used, feature_extraction_time, model_training_time, prediction_time)
            
            self.logger.info(f"Batch {batch_id} completed: {len(forecasts)} forecasts in {model_time + feature_extraction_time:.2f}s")
            
            return forecasts, self.performance_monitor.all_metrics[-1]
            
        except Exception as e:
            self.logger.error(f"Batch {batch_id} processing failed: {e}")
            self.performance_monitor.end_batch("failure", feature_extraction_time, model_training_time, prediction_time, fallback_rate=1.0)
            raise
    
    def _extract_batch_data(self, tickers: List[str], data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and validate data for a batch of tickers.
        
        Args:
            tickers: List of ticker symbols
            data: Full dataset
            
        Returns:
            DataFrame with data for the specified tickers
        """
        # Get available tickers
        available_tickers = [ticker for ticker in tickers if ticker in data.columns]
        
        if not available_tickers:
            self.logger.error(f"No data available for any ticker in batch: {tickers}")
            return pd.DataFrame()
        
        if len(available_tickers) < len(tickers):
            missing = set(tickers) - set(available_tickers)
            self.logger.warning(f"Missing data for tickers: {missing}")
        
        # Extract batch data
        batch_data = data[available_tickers].copy()
        
        # Basic data cleaning
        batch_data = batch_data.ffill().dropna()
        
        if batch_data.empty:
            self.logger.error("No valid data remaining after cleaning")
            return pd.DataFrame()
        
        self.logger.debug(f"Extracted batch data: {batch_data.shape} for {len(available_tickers)} tickers")
        return batch_data
    
    def _extract_batch_features(self, 
                               ticker_group: List[str], 
                               batch_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract individual and shared features for a batch.
        
        Args:
            ticker_group: List of tickers in the batch
            batch_data: Price data for the batch
            
        Returns:
            Tuple of (individual_features, shared_features)
        """
        # Lazy initialization of feature extractor
        if self.feature_extractor is None:
            from .shared_feature_extractor import SharedFeatureExtractor
            self.feature_extractor = SharedFeatureExtractor(self.config.features)
        
        try:
            individual_features, shared_features = self.feature_extractor.extract_features(
                batch_data, ticker_group
            )
            
            self.logger.debug(f"Extracted features: individual={individual_features.shape}, "
                            f"shared={shared_features.shape}")
            
            return individual_features, shared_features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            # Return minimal features as fallback
            individual_features = pd.DataFrame(index=batch_data.index)
            shared_features = pd.DataFrame(index=batch_data.index)
            return individual_features, shared_features
    
    def _train_and_predict_batch(self,
                                ticker_group: List[str],
                                individual_features: pd.DataFrame,
                                shared_features: pd.DataFrame,
                                periods: int) -> Tuple[Dict[str, float], str]:
        """
        Train model and generate predictions for a batch.
        
        Args:
            ticker_group: List of tickers in the batch
            individual_features: Individual ticker features
            shared_features: Shared market features
            periods: Forecast horizon
            
        Returns:
            Tuple of (forecasts, model_used)
        """
        # Lazy initialization of batch models
        if not self.batch_models:
            self._initialize_batch_models()
        
        # Try primary model first
        primary_model_name = self.config.models.primary_model
        
        try:
            if primary_model_name in self.batch_models:
                model = self.batch_models[primary_model_name]
                forecasts = self._try_model_prediction(
                    model, ticker_group, individual_features, shared_features, periods
                )
                
                if forecasts:
                    self.logger.debug(f"Primary model {primary_model_name} succeeded")
                    return forecasts, primary_model_name
            
        except Exception as e:
            self.logger.warning(f"Primary model {primary_model_name} failed: {e}")
        
        # Try fallback model
        fallback_model_name = self.config.models.fallback_model
        
        try:
            if fallback_model_name in self.batch_models and fallback_model_name != primary_model_name:
                model = self.batch_models[fallback_model_name]
                forecasts = self._try_model_prediction(
                    model, ticker_group, individual_features, shared_features, periods
                )
                
                if forecasts:
                    self.logger.info(f"Fallback model {fallback_model_name} succeeded")
                    return forecasts, fallback_model_name
            
        except Exception as e:
            self.logger.warning(f"Fallback model {fallback_model_name} failed: {e}")
        
        # Ultimate fallback: simple historical mean
        self.logger.warning("All batch models failed, using simple historical mean")
        return self._simple_historical_forecast(ticker_group, individual_features), "historical_mean"
    
    def _try_model_prediction(self,
                             model: Any,
                             ticker_group: List[str],
                             individual_features: pd.DataFrame,
                             shared_features: pd.DataFrame,
                             periods: int) -> Optional[Dict[str, float]]:
        """
        Try to get predictions from a specific model.
        
        Args:
            model: Batch forecasting model
            ticker_group: List of tickers
            individual_features: Individual ticker features
            shared_features: Shared market features
            periods: Forecast horizon
            
        Returns:
            Dictionary of forecasts or None if failed
        """
        try:
            # Fit the model
            model.fit_batch(individual_features, shared_features, ticker_group)
            
            # Generate predictions
            forecasts = model.predict_batch(periods, shared_features)
            
            # Validate predictions
            if self._validate_predictions(forecasts, ticker_group):
                return forecasts
            else:
                self.logger.warning("Model predictions failed validation")
                return None
                
        except Exception as e:
            self.logger.warning(f"Model prediction attempt failed: {e}")
            return None
    
    def _validate_predictions(self, forecasts: Dict[str, float], expected_tickers: List[str]) -> bool:
        """
        Validate model predictions.
        
        Args:
            forecasts: Dictionary of predictions
            expected_tickers: List of expected tickers
            
        Returns:
            True if predictions are valid
        """
        try:
            # Check if all tickers have predictions
            missing_tickers = set(expected_tickers) - set(forecasts.keys())
            if missing_tickers:
                self.logger.warning(f"Missing predictions for tickers: {missing_tickers}")
                return False
            
            # Check for valid numeric values
            for ticker, forecast in forecasts.items():
                if not isinstance(forecast, (int, float)) or np.isnan(forecast) or np.isinf(forecast):
                    self.logger.warning(f"Invalid prediction for {ticker}: {forecast}")
                    return False
                
                # Sanity check: reasonable return range
                if forecast < -1.0 or forecast > 5.0:  # -100% to +500% annual return
                    self.logger.warning(f"Extreme prediction for {ticker}: {forecast}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Prediction validation failed: {e}")
            return False
    
    def _simple_historical_forecast(self, ticker_group: List[str], features: pd.DataFrame) -> Dict[str, float]:
        """
        Simple historical mean forecast as ultimate fallback.
        
        Args:
            ticker_group: List of tickers
            features: Feature data (not used in this simple method)
            
        Returns:
            Dictionary of simple forecasts
        """
        forecasts = {}
        
        for ticker in ticker_group:
            # Use conservative default
            forecasts[ticker] = 0.05  # 5% annual return
        
        self.logger.info(f"Generated simple historical forecasts for {len(ticker_group)} tickers")
        return forecasts
    
    def _initialize_batch_models(self):
        """Initialize batch forecasting models."""
        try:
            # Import models dynamically to avoid circular imports
            from batch_models import BatchXGBoostForecaster, BatchLinearForecaster
            
            # Initialize primary model
            if self.config.models.primary_model == "batch_xgboost":
                self.batch_models["batch_xgboost"] = BatchXGBoostForecaster(self.config.models)
            elif self.config.models.primary_model == "batch_linear":
                self.batch_models["batch_linear"] = BatchLinearForecaster(self.config.models)
            
            # Initialize fallback model if different
            if (self.config.models.fallback_model != self.config.models.primary_model and
                self.config.models.fallback_model not in self.batch_models):
                
                if self.config.models.fallback_model == "batch_linear":
                    self.batch_models["batch_linear"] = BatchLinearForecaster(self.config.models)
                elif self.config.models.fallback_model == "batch_xgboost":
                    self.batch_models["batch_xgboost"] = BatchXGBoostForecaster(self.config.models)
            
            self.logger.info(f"Initialized batch models: {list(self.batch_models.keys())}")
            
        except ImportError as e:
            self.logger.error(f"Failed to import batch models: {e}")
            self.batch_models = {}
        except Exception as e:
            self.logger.error(f"Failed to initialize batch models: {e}")
            self.batch_models = {}
    
    def _handle_batch_failure(self, 
                             ticker_batch: List[str], 
                             data: pd.DataFrame, 
                             periods: int,
                             error: Exception) -> Dict[str, float]:
        """
        Handle batch processing failure with fallback strategies.
        
        Args:
            ticker_batch: Failed batch of tickers
            data: Historical data
            periods: Forecast horizon
            error: The error that caused the failure
            
        Returns:
            Dictionary of fallback forecasts
        """
        self.logger.warning(f"Handling batch failure for {len(ticker_batch)} tickers: {error}")

        context = BatchContext(
            tickers=ticker_batch,
            model_name="", # model is not known here
            operation="process_batch",
            batch_size=len(ticker_batch)
        )

        recovery_result = self.fallback_coordinator.handle_error(error, context)

        if recovery_result.recovery_successful:
            self.logger.info("Batch failure recovery successful.")
            new_batches = recovery_result.result
            
            all_forecasts = {}
            if isinstance(new_batches, list) and all(isinstance(b, list) for b in new_batches):
                 for i, new_ticker_batch in enumerate(new_batches):
                    self.logger.info(f"Processing recovered batch {i+1}/{len(new_batches)} with {len(new_ticker_batch)} tickers")
                    try:
                        batch_forecasts, _ = self._process_single_batch(
                            new_ticker_batch, data, periods, batch_id=f"recovered-{i}"
                        )
                        all_forecasts.update(batch_forecasts)
                    except Exception as e:
                        self.logger.error(f"Recovered batch processing failed: {e}")
                        # If even the smaller batch fails, fallback to individual processing for these tickers.
                        all_forecasts.update(self._fallback_to_individual_processing(new_ticker_batch, data, periods))

            return all_forecasts

        else:
            self.logger.error("Batch failure recovery failed. Falling back to individual processing for this batch.")
            return self._fallback_to_individual_processing(ticker_batch, data, periods)
    
    def _fallback_to_individual_processing(self, 
                                          tickers: List[str], 
                                          data: pd.DataFrame, 
                                          periods: int) -> Dict[str, float]:
        """
        Fallback to individual ticker processing.
        
        Args:
            tickers: List of tickers to process
            data: Historical data
            periods: Forecast horizon
            
        Returns:
            Dictionary of individual forecasts
        """
        self.logger.info(f"Falling back to individual processing for {len(tickers)} tickers")
        
        # Import the existing individual forecasting function
        try:
            from portfolio_optimization import _forecast_single_ticker
            
            forecasts = {}
            for ticker in tickers:
                if ticker in data.columns:
                    _, forecast = _forecast_single_ticker(ticker, data[ticker], use_lightweight=True)
                    forecasts[ticker] = forecast
                else:
                    forecasts[ticker] = 0.05
            
            return forecasts
            
        except ImportError as e:
            self.logger.error(f"Failed to import individual forecasting: {e}")
            return {ticker: 0.05 for ticker in tickers}
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _record_overall_performance(self, 
                                   num_tickers: int, 
                                   total_time: float, 
                                   batch_metrics: List[BatchPerformanceMetrics]):
        """Record overall performance metrics."""
        self.total_processing_time += total_time
        self.total_tickers_processed += num_tickers
        
        if batch_metrics:
            avg_batch_time = sum(m.processing_time for m in batch_metrics) / len(batch_metrics)
            avg_memory = sum(m.memory_usage for m in batch_metrics) / len(batch_metrics)
            
            self.logger.info(f"Performance summary: {num_tickers} tickers in {total_time:.2f}s "
                           f"(avg batch: {avg_batch_time:.2f}s, memory: {avg_memory:.1f}MB)")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        return self.performance_monitor.get_summary()