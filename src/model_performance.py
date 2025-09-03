"""
Model Performance Tracking Infrastructure

This module provides dataclasses and utilities for tracking model performance,
logging metrics, and storing forecast results for the advanced forecasting system.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class ModelPerformance:
    """Tracks performance metrics for a forecasting model on a specific ticker."""
    ticker: str
    model_name: str
    mse: float
    mae: float
    mape: float
    training_time: float
    prediction_time: float
    timestamp: datetime
    data_points: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelPerformance':
        """Create instance from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ForecastResult:
    """Stores the result of a forecasting operation."""
    ticker: str
    expected_return: float
    confidence_interval: Tuple[float, float]
    model_used: str
    ensemble_weights: Dict[str, float]
    validation_score: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ForecastResult':
        """Create instance from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['confidence_interval'] = tuple(data['confidence_interval'])
        return cls(**data)


class PerformanceMetrics:
    """Utility class for calculating model validation metrics."""
    
    @staticmethod
    def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return float(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100))
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(PerformanceMetrics.calculate_mse(y_true, y_pred)))
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all standard metrics."""
        return {
            'mse': PerformanceMetrics.calculate_mse(y_true, y_pred),
            'mae': PerformanceMetrics.calculate_mae(y_true, y_pred),
            'mape': PerformanceMetrics.calculate_mape(y_true, y_pred),
            'rmse': PerformanceMetrics.calculate_rmse(y_true, y_pred)
        }


class PerformanceLogger:
    """Handles logging and storage of model performance data."""
    
    def __init__(self, log_dir: str = "logs/performance"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger('model_performance')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler if not already exists
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_dir / 'performance.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_performance(self, performance: ModelPerformance) -> None:
        """Log model performance metrics."""
        self.logger.info(
            f"Model Performance - Ticker: {performance.ticker}, "
            f"Model: {performance.model_name}, "
            f"MSE: {performance.mse:.6f}, "
            f"MAE: {performance.mae:.6f}, "
            f"MAPE: {performance.mape:.2f}%, "
            f"Training Time: {performance.training_time:.2f}s, "
            f"Prediction Time: {performance.prediction_time:.4f}s, "
            f"Data Points: {performance.data_points}"
        )
    
    def log_forecast_result(self, result: ForecastResult) -> None:
        """Log forecast result."""
        self.logger.info(
            f"Forecast Result - Ticker: {result.ticker}, "
            f"Expected Return: {result.expected_return:.4f}, "
            f"Model: {result.model_used}, "
            f"Validation Score: {result.validation_score:.4f}, "
            f"Ensemble Weights: {result.ensemble_weights}"
        )
    
    def save_performance_data(self, performances: List[ModelPerformance], 
                            filename: Optional[str] = None) -> None:
        """Save performance data to JSON file."""
        if filename is None:
            filename = f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        data = [perf.to_dict() for perf in performances]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved {len(performances)} performance records to {filepath}")
    
    def load_performance_data(self, filename: str) -> List[ModelPerformance]:
        """Load performance data from JSON file."""
        filepath = self.log_dir / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return [ModelPerformance.from_dict(item) for item in data]
    
    def save_forecast_results(self, results: List[ForecastResult], 
                            filename: Optional[str] = None) -> None:
        """Save forecast results to JSON file."""
        if filename is None:
            filename = f"forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        data = [result.to_dict() for result in results]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved {len(results)} forecast results to {filepath}")


class PerformanceTracker:
    """Main class for tracking and managing model performance."""
    
    def __init__(self, log_dir: str = "logs/performance"):
        self.logger = PerformanceLogger(log_dir)
        self.performance_history: List[ModelPerformance] = []
        self.forecast_history: List[ForecastResult] = []
    
    def record_performance(self, ticker: str, model_name: str, 
                         y_true: np.ndarray, y_pred: np.ndarray,
                         training_time: float, prediction_time: float,
                         data_points: int) -> ModelPerformance:
        """Record model performance metrics."""
        metrics = PerformanceMetrics.calculate_all_metrics(y_true, y_pred)
        
        performance = ModelPerformance(
            ticker=ticker,
            model_name=model_name,
            mse=metrics['mse'],
            mae=metrics['mae'],
            mape=metrics['mape'],
            training_time=training_time,
            prediction_time=prediction_time,
            timestamp=datetime.now(),
            data_points=data_points
        )
        
        self.performance_history.append(performance)
        self.logger.log_performance(performance)
        
        return performance
    
    def record_forecast(self, ticker: str, expected_return: float,
                       confidence_interval: Tuple[float, float],
                       model_used: str, ensemble_weights: Dict[str, float],
                       validation_score: float) -> ForecastResult:
        """Record forecast result."""
        result = ForecastResult(
            ticker=ticker,
            expected_return=expected_return,
            confidence_interval=confidence_interval,
            model_used=model_used,
            ensemble_weights=ensemble_weights,
            validation_score=validation_score,
            timestamp=datetime.now()
        )
        
        self.forecast_history.append(result)
        self.logger.log_forecast_result(result)
        
        return result
    
    def get_model_performance_summary(self, model_name: Optional[str] = None,
                                   ticker: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for models."""
        filtered_data = self.performance_history
        
        if model_name:
            filtered_data = [p for p in filtered_data if p.model_name == model_name]
        
        if ticker:
            filtered_data = [p for p in filtered_data if p.ticker == ticker]
        
        if not filtered_data:
            return {}
        
        mse_values = [p.mse for p in filtered_data]
        mae_values = [p.mae for p in filtered_data]
        mape_values = [p.mape for p in filtered_data]
        training_times = [p.training_time for p in filtered_data]
        
        return {
            'count': len(filtered_data),
            'avg_mse': np.mean(mse_values),
            'avg_mae': np.mean(mae_values),
            'avg_mape': np.mean(mape_values),
            'avg_training_time': np.mean(training_times),
            'best_mape': min(mape_values),
            'worst_mape': max(mape_values)
        }
    
    def get_best_models_by_ticker(self, top_n: int = 3) -> Dict[str, List[str]]:
        """Get best performing models for each ticker."""
        ticker_models = {}
        
        for perf in self.performance_history:
            if perf.ticker not in ticker_models:
                ticker_models[perf.ticker] = []
            ticker_models[perf.ticker].append(perf)
        
        best_models = {}
        for ticker, performances in ticker_models.items():
            # Sort by MAPE (lower is better)
            sorted_perfs = sorted(performances, key=lambda x: x.mape)
            best_models[ticker] = [p.model_name for p in sorted_perfs[:top_n]]
        
        return best_models
    
    def save_all_data(self) -> None:
        """Save all performance and forecast data."""
        if self.performance_history:
            self.logger.save_performance_data(self.performance_history)
        
        if self.forecast_history:
            self.logger.save_forecast_results(self.forecast_history)