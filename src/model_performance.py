"""
Model Performance Monitoring and Reporting System

This module provides real-time performance metrics collection, model usage statistics,
success rate tracking, and performance reporting for the advanced forecasting system.
"""

import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelUsageStats:
    """Statistics for model usage and performance."""
    model_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_training_time: float = 0.0
    total_prediction_time: float = 0.0
    average_accuracy: float = 0.0
    last_used: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100
    
    @property
    def average_training_time(self) -> float:
        """Calculate average training time."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_training_time / self.successful_calls
    
    @property
    def average_prediction_time(self) -> float:
        """Calculate average prediction time."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_prediction_time / self.successful_calls


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    timestamp: datetime
    model_name: str
    ticker: str
    training_time: float
    prediction_time: float
    accuracy_score: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class SystemPerformanceReport:
    """System-wide performance report."""
    report_timestamp: datetime
    total_forecasts: int
    successful_forecasts: int
    failed_forecasts: int
    average_processing_time: float
    model_usage_stats: Dict[str, ModelUsageStats]
    top_performing_models: List[Tuple[str, float]]
    system_health_score: float
    recommendations: List[str]


class PerformanceMonitor:
    """
    Real-time performance monitoring system for forecasting models.
    
    Tracks model usage, success rates, timing metrics, and provides
    alerting and reporting capabilities.
    """
    
    def __init__(self, 
                 metrics_retention_days: int = 30,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 log_directory: str = "logs/model_performance"):
        """
        Initialize performance monitor.
        
        Args:
            metrics_retention_days: Number of days to retain detailed metrics
            alert_thresholds: Thresholds for performance alerts
            log_directory: Directory to store performance logs
        """
        self.metrics_retention_days = metrics_retention_days
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'min_success_rate': 80.0,  # Minimum success rate percentage
            'max_training_time': 300.0,  # Maximum training time in seconds
            'max_prediction_time': 30.0,  # Maximum prediction time in seconds
            'min_accuracy': 0.15,  # Minimum MAPE threshold
            'max_memory_usage': 1024.0  # Maximum memory usage in MB
        }
        
        # In-memory storage for real-time metrics
        self.model_stats: Dict[str, ModelUsageStats] = {}
        self.recent_metrics: deque = deque(maxlen=10000)  # Last 10k metrics
        self.alerts: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_forecasts = 0
        self.successful_forecasts = 0
        
        logger.info(f"Performance monitor initialized. Logging to: {self.log_directory}")
    
    def record_model_usage(self, 
                          model_name: str,
                          ticker: str,
                          training_time: float,
                          prediction_time: float,
                          success: bool = True,
                          accuracy_score: Optional[float] = None,
                          memory_usage_mb: Optional[float] = None,
                          error_message: Optional[str] = None) -> None:
        """
        Record model usage metrics.
        
        Args:
            model_name: Name of the model used
            ticker: Ticker symbol
            training_time: Time taken for training (seconds)
            prediction_time: Time taken for prediction (seconds)
            success: Whether the operation was successful
            accuracy_score: Model accuracy score (optional)
            memory_usage_mb: Memory usage in MB (optional)
            error_message: Error message if failed (optional)
        """
        timestamp = datetime.now()
        
        # Update model statistics
        if model_name not in self.model_stats:
            self.model_stats[model_name] = ModelUsageStats(model_name=model_name)
        
        stats = self.model_stats[model_name]
        stats.total_calls += 1
        stats.last_used = timestamp
        
        if success:
            stats.successful_calls += 1
            stats.total_training_time += training_time
            stats.total_prediction_time += prediction_time
            
            if accuracy_score is not None:
                # Update running average of accuracy
                current_avg = stats.average_accuracy
                n = stats.successful_calls
                stats.average_accuracy = ((current_avg * (n - 1)) + accuracy_score) / n
        else:
            stats.failed_calls += 1
        
        # Create performance metric record
        metric = PerformanceMetrics(
            timestamp=timestamp,
            model_name=model_name,
            ticker=ticker,
            training_time=training_time,
            prediction_time=prediction_time,
            accuracy_score=accuracy_score,
            memory_usage_mb=memory_usage_mb,
            success=success,
            error_message=error_message
        )
        
        # Store in recent metrics
        self.recent_metrics.append(metric)
        
        # Update global counters
        self.total_forecasts += 1
        if success:
            self.successful_forecasts += 1
        
        # Check for alerts
        self._check_alerts(metric, stats)
        
        # Log to file
        self._log_metric(metric)
        
        logger.debug(f"Recorded usage for {model_name} on {ticker}: "
                    f"Success={success}, Training={training_time:.2f}s, "
                    f"Prediction={prediction_time:.4f}s")
    
    def get_model_stats(self, model_name: Optional[str] = None) -> Dict[str, ModelUsageStats]:
        """
        Get model usage statistics.
        
        Args:
            model_name: Specific model name (optional, returns all if None)
            
        Returns:
            Dictionary of model statistics
        """
        if model_name:
            return {model_name: self.model_stats.get(model_name, ModelUsageStats(model_name))}
        return self.model_stats.copy()
    
    def get_system_health_score(self) -> float:
        """
        Calculate overall system health score (0-100).
        
        Returns:
            Health score percentage
        """
        if not self.model_stats:
            return 100.0
        
        # Calculate weighted health score based on multiple factors
        factors = []
        
        # Success rate factor
        if self.total_forecasts > 0:
            success_rate = (self.successful_forecasts / self.total_forecasts) * 100
            factors.append(min(success_rate / self.alert_thresholds['min_success_rate'], 1.0))
        
        # Model performance factor
        model_scores = []
        for stats in self.model_stats.values():
            if stats.total_calls > 0:
                # Success rate component
                success_component = stats.success_rate / 100.0
                
                # Timing component (inverse of average times)
                timing_component = 1.0
                if stats.average_training_time > 0:
                    timing_component *= min(self.alert_thresholds['max_training_time'] / stats.average_training_time, 1.0)
                if stats.average_prediction_time > 0:
                    timing_component *= min(self.alert_thresholds['max_prediction_time'] / stats.average_prediction_time, 1.0)
                
                # Accuracy component
                accuracy_component = 1.0
                if stats.average_accuracy > 0:
                    # Convert MAPE to score (lower MAPE = higher score)
                    accuracy_component = max(0, 1.0 - (stats.average_accuracy / 100.0))
                
                model_score = (success_component * 0.5 + timing_component * 0.3 + accuracy_component * 0.2)
                model_scores.append(model_score)
        
        if model_scores:
            factors.append(np.mean(model_scores))
        
        # Alert factor (penalize for recent alerts)
        recent_alerts = [a for a in self.alerts if 
                        (datetime.now() - datetime.fromisoformat(a['timestamp'])).total_seconds() < 3600]
        alert_penalty = min(len(recent_alerts) * 0.1, 0.5)  # Max 50% penalty
        factors.append(1.0 - alert_penalty)
        
        # Calculate final score
        health_score = np.mean(factors) * 100 if factors else 100.0
        return max(0.0, min(100.0, health_score))
    
    def generate_performance_report(self, 
                                  include_recommendations: bool = True) -> SystemPerformanceReport:
        """
        Generate comprehensive performance report.
        
        Args:
            include_recommendations: Whether to include performance recommendations
            
        Returns:
            SystemPerformanceReport object
        """
        # Calculate top performing models
        top_models = []
        for model_name, stats in self.model_stats.items():
            if stats.successful_calls > 0:
                # Score based on success rate and accuracy
                score = (stats.success_rate / 100.0) * 0.6
                if stats.average_accuracy > 0:
                    accuracy_score = max(0, 1.0 - (stats.average_accuracy / 100.0))
                    score += accuracy_score * 0.4
                top_models.append((model_name, score))
        
        top_models.sort(key=lambda x: x[1], reverse=True)
        
        # Generate recommendations
        recommendations = []
        if include_recommendations:
            recommendations = self._generate_recommendations()
        
        # Calculate average processing time
        avg_processing_time = 0.0
        if self.recent_metrics:
            total_time = sum(m.training_time + m.prediction_time for m in self.recent_metrics)
            avg_processing_time = total_time / len(self.recent_metrics)
        
        report = SystemPerformanceReport(
            report_timestamp=datetime.now(),
            total_forecasts=self.total_forecasts,
            successful_forecasts=self.successful_forecasts,
            failed_forecasts=self.total_forecasts - self.successful_forecasts,
            average_processing_time=avg_processing_time,
            model_usage_stats=self.model_stats.copy(),
            top_performing_models=top_models[:5],  # Top 5 models
            system_health_score=self.get_system_health_score(),
            recommendations=recommendations
        )
        
        return report
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent alerts within specified time window.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts 
                if datetime.fromisoformat(alert['timestamp']) > cutoff_time]
    
    def export_metrics(self, 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      format: str = 'json') -> str:
        """
        Export metrics to file.
        
        Args:
            start_date: Start date for export (optional)
            end_date: End date for export (optional)
            format: Export format ('json' or 'csv')
            
        Returns:
            Path to exported file
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)  # Default to last 7 days
        
        # Filter metrics by date range
        filtered_metrics = [
            m for m in self.recent_metrics
            if start_date <= m.timestamp <= end_date
        ]
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'json':
            filename = f"performance_metrics_{timestamp_str}.json"
            filepath = self.log_directory / filename
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'metrics': [asdict(m) for m in filtered_metrics],
                'model_stats': {name: asdict(stats) for name, stats in self.model_stats.items()},
                'system_health_score': self.get_system_health_score()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        elif format.lower() == 'csv':
            filename = f"performance_metrics_{timestamp_str}.csv"
            filepath = self.log_directory / filename
            
            # Convert metrics to DataFrame
            metrics_data = []
            for m in filtered_metrics:
                metrics_data.append({
                    'timestamp': m.timestamp.isoformat(),
                    'model_name': m.model_name,
                    'ticker': m.ticker,
                    'training_time': m.training_time,
                    'prediction_time': m.prediction_time,
                    'accuracy_score': m.accuracy_score,
                    'memory_usage_mb': m.memory_usage_mb,
                    'success': m.success,
                    'error_message': m.error_message
                })
            
            df = pd.DataFrame(metrics_data)
            df.to_csv(filepath, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(filtered_metrics)} metrics to {filepath}")
        return str(filepath)
    
    def _check_alerts(self, metric: PerformanceMetrics, stats: ModelUsageStats) -> None:
        """Check for performance alerts based on thresholds."""
        alerts_triggered = []
        
        # Success rate alert
        if stats.total_calls >= 10 and stats.success_rate < self.alert_thresholds['min_success_rate']:
            alerts_triggered.append({
                'type': 'low_success_rate',
                'message': f"Model {metric.model_name} success rate ({stats.success_rate:.1f}%) below threshold",
                'severity': 'warning'
            })
        
        # Training time alert
        if metric.success and metric.training_time > self.alert_thresholds['max_training_time']:
            alerts_triggered.append({
                'type': 'high_training_time',
                'message': f"Model {metric.model_name} training time ({metric.training_time:.1f}s) exceeded threshold",
                'severity': 'warning'
            })
        
        # Prediction time alert
        if metric.success and metric.prediction_time > self.alert_thresholds['max_prediction_time']:
            alerts_triggered.append({
                'type': 'high_prediction_time',
                'message': f"Model {metric.model_name} prediction time ({metric.prediction_time:.1f}s) exceeded threshold",
                'severity': 'warning'
            })
        
        # Accuracy alert
        if metric.accuracy_score and metric.accuracy_score > self.alert_thresholds['min_accuracy']:
            alerts_triggered.append({
                'type': 'low_accuracy',
                'message': f"Model {metric.model_name} accuracy ({metric.accuracy_score:.4f}) below threshold",
                'severity': 'warning'
            })
        
        # Memory usage alert
        if metric.memory_usage_mb and metric.memory_usage_mb > self.alert_thresholds['max_memory_usage']:
            alerts_triggered.append({
                'type': 'high_memory_usage',
                'message': f"Model {metric.model_name} memory usage ({metric.memory_usage_mb:.1f}MB) exceeded threshold",
                'severity': 'warning'
            })
        
        # Add alerts to list
        for alert in alerts_triggered:
            alert.update({
                'timestamp': metric.timestamp.isoformat(),
                'model_name': metric.model_name,
                'ticker': metric.ticker
            })
            self.alerts.append(alert)
            
            # Log alert
            logger.warning(f"ALERT: {alert['message']}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on current metrics."""
        recommendations = []
        
        if not self.model_stats:
            return recommendations
        
        # Analyze model performance
        low_performing_models = []
        high_memory_models = []
        slow_models = []
        
        for model_name, stats in self.model_stats.items():
            if stats.total_calls >= 5:  # Only consider models with sufficient data
                
                # Check success rate
                if stats.success_rate < self.alert_thresholds['min_success_rate']:
                    low_performing_models.append((model_name, stats.success_rate))
                
                # Check training time
                if stats.average_training_time > self.alert_thresholds['max_training_time']:
                    slow_models.append((model_name, stats.average_training_time))
        
        # Generate recommendations
        if low_performing_models:
            models_str = ', '.join([f"{name} ({rate:.1f}%)" for name, rate in low_performing_models])
            recommendations.append(f"Consider disabling or reconfiguring low-performing models: {models_str}")
        
        if slow_models:
            models_str = ', '.join([f"{name} ({time:.1f}s)" for name, time in slow_models])
            recommendations.append(f"Optimize training parameters for slow models: {models_str}")
        
        # System-level recommendations
        if self.total_forecasts > 0:
            overall_success_rate = (self.successful_forecasts / self.total_forecasts) * 100
            if overall_success_rate < 90:
                recommendations.append("Overall system success rate is low. Consider reviewing data quality and model configurations.")
        
        # Performance recommendations
        if len(self.recent_metrics) > 100:
            avg_total_time = np.mean([m.training_time + m.prediction_time for m in self.recent_metrics])
            if avg_total_time > 60:  # More than 1 minute average
                recommendations.append("Consider enabling more aggressive caching or reducing the number of models per ticker.")
        
        return recommendations
    
    def _log_metric(self, metric: PerformanceMetrics) -> None:
        """Log metric to file."""
        try:
            log_date = metric.timestamp.strftime("%Y%m%d")
            log_file = self.log_directory / f"performance_{log_date}.log"
            
            log_entry = {
                'timestamp': metric.timestamp.isoformat(),
                'model_name': metric.model_name,
                'ticker': metric.ticker,
                'training_time': metric.training_time,
                'prediction_time': metric.prediction_time,
                'success': metric.success,
                'accuracy_score': metric.accuracy_score,
                'memory_usage_mb': metric.memory_usage_mb,
                'error_message': metric.error_message
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.warning(f"Failed to log metric to file: {e}")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def record_model_performance(model_name: str,
                           ticker: str,
                           training_time: float,
                           prediction_time: float,
                           success: bool = True,
                           accuracy_score: Optional[float] = None,
                           memory_usage_mb: Optional[float] = None,
                           error_message: Optional[str] = None) -> None:
    """
    Convenience function to record model performance metrics.
    
    Args:
        model_name: Name of the model used
        ticker: Ticker symbol
        training_time: Time taken for training (seconds)
        prediction_time: Time taken for prediction (seconds)
        success: Whether the operation was successful
        accuracy_score: Model accuracy score (optional)
        memory_usage_mb: Memory usage in MB (optional)
        error_message: Error message if failed (optional)
    """
    monitor = get_performance_monitor()
    monitor.record_model_usage(
        model_name=model_name,
        ticker=ticker,
        training_time=training_time,
        prediction_time=prediction_time,
        success=success,
        accuracy_score=accuracy_score,
        memory_usage_mb=memory_usage_mb,
        error_message=error_message
    )


def get_performance_report() -> SystemPerformanceReport:
    """Get current system performance report."""
    monitor = get_performance_monitor()
    return monitor.generate_performance_report()


def export_performance_metrics(days: int = 7, format: str = 'json') -> str:
    """
    Export performance metrics for the specified number of days.
    
    Args:
        days: Number of days to export
        format: Export format ('json' or 'csv')
        
    Returns:
        Path to exported file
    """
    monitor = get_performance_monitor()
    start_date = datetime.now() - timedelta(days=days)
    return monitor.export_metrics(start_date=start_date, format=format)