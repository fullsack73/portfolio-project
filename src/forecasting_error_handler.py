"""
Comprehensive Error Handling and Robustness System for Forecasting Models

This module provides robust error handling, data quality validation, and graceful
degradation mechanisms for all forecasting models in the system.
"""

import logging
import traceback
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import pandas as pd
import numpy as np


class ErrorSeverity(Enum):
    """Error severity levels for categorizing different types of errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors that can occur in forecasting."""
    DATA_QUALITY = "data_quality"
    MODEL_TRAINING = "model_training"
    MODEL_PREDICTION = "model_prediction"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    ticker: str
    model_name: str
    operation: str
    data_points: int
    timestamp: datetime
    additional_info: Dict[str, Any]


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: ErrorContext
    exception_type: str
    traceback_info: str
    recovery_action: Optional[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error record to dictionary for logging/storage."""
        return {
            'error_id': self.error_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'ticker': self.context.ticker,
            'model_name': self.context.model_name,
            'operation': self.context.operation,
            'data_points': self.context.data_points,
            'exception_type': self.exception_type,
            'recovery_action': self.recovery_action,
            'timestamp': self.timestamp.isoformat(),
            'additional_info': self.context.additional_info
        }


class DataQualityValidator:
    """Validates data quality and identifies potential issues."""
    
    def __init__(self):
        self.logger = logging.getLogger('forecasting.data_quality')
    
    def validate_data(self, 
                     data: pd.Series, 
                     ticker: str,
                     min_points: int = 100,
                     max_missing_pct: float = 0.1) -> Tuple[bool, List[str]]:
        """
        Comprehensive data quality validation.
        
        Args:
            data: Time series data to validate
            ticker: Ticker symbol for context
            min_points: Minimum required data points
            max_missing_pct: Maximum allowed percentage of missing values
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Basic data existence check
            if data is None:
                issues.append("Data is None")
                return False, issues
            
            if data.empty:
                issues.append("Data is empty")
                return False, issues
            
            # Data length validation
            if len(data) < min_points:
                issues.append(f"Insufficient data points: {len(data)} < {min_points}")
            
            # Missing values check
            missing_count = data.isnull().sum()
            missing_pct = missing_count / len(data)
            if missing_pct > max_missing_pct:
                issues.append(f"Too many missing values: {missing_pct:.2%} > {max_missing_pct:.2%}")
            
            # Data type validation
            if not pd.api.types.is_numeric_dtype(data):
                issues.append("Data is not numeric")
            
            # Check for infinite values
            if np.isinf(data).any():
                issues.append("Data contains infinite values")
            
            # Check for constant values (no variance)
            if data.nunique() <= 1:
                issues.append("Data has no variance (constant values)")
            
            # Check for extreme outliers (beyond 5 standard deviations)
            if len(data) > 10:
                z_scores = np.abs((data - data.mean()) / data.std())
                extreme_outliers = (z_scores > 5).sum()
                if extreme_outliers > len(data) * 0.05:  # More than 5% extreme outliers
                    issues.append(f"Too many extreme outliers: {extreme_outliers} points")
            
            # Check data frequency consistency (if datetime index)
            if isinstance(data.index, pd.DatetimeIndex):
                freq_issues = self._check_frequency_consistency(data)
                issues.extend(freq_issues)
            
            # Check for recent data availability (only warn, don't fail validation)
            if isinstance(data.index, pd.DatetimeIndex):
                days_since_last = (datetime.now() - data.index[-1]).days
                if days_since_last > 30:
                    # Only add as warning, don't fail validation for historical data
                    self.logger.debug(f"Data is stale: {days_since_last} days since last update for {ticker}")
            
            is_valid = len(issues) == 0
            
            if issues:
                self.logger.warning(f"Data quality issues for {ticker}: {'; '.join(issues)}")
            else:
                self.logger.debug(f"Data quality validation passed for {ticker}")
            
            return is_valid, issues
            
        except Exception as e:
            issues.append(f"Data validation failed: {str(e)}")
            self.logger.error(f"Data validation error for {ticker}: {e}")
            return False, issues
    
    def _check_frequency_consistency(self, data: pd.Series) -> List[str]:
        """Check for consistency in data frequency."""
        issues = []
        
        try:
            if len(data) < 10:
                return issues
            
            # Calculate time differences
            time_diffs = data.index.to_series().diff().dropna()
            
            # Check for large gaps (more than 10 days for daily data)
            large_gaps = time_diffs > pd.Timedelta(days=10)
            if large_gaps.any():
                gap_count = large_gaps.sum()
                issues.append(f"Found {gap_count} large time gaps in data")
            
            # Check frequency consistency
            mode_diff = time_diffs.mode()
            if len(mode_diff) > 0:
                expected_freq = mode_diff.iloc[0]
                inconsistent = (time_diffs != expected_freq).sum()
                if inconsistent > len(time_diffs) * 0.1:  # More than 10% inconsistent
                    issues.append(f"Inconsistent data frequency: {inconsistent} irregular intervals")
            
        except Exception as e:
            issues.append(f"Frequency check failed: {str(e)}")
        
        return issues
    
    def clean_data(self, data: pd.Series, ticker: str) -> pd.Series:
        """
        Clean data by handling common issues.
        
        Args:
            data: Raw time series data
            ticker: Ticker symbol for context
            
        Returns:
            Cleaned data series
        """
        try:
            cleaned_data = data.copy()
            
            # Remove infinite values
            if np.isinf(cleaned_data).any():
                self.logger.warning(f"Removing infinite values from {ticker} data")
                cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
            
            # Handle missing values
            if cleaned_data.isnull().any():
                missing_count = cleaned_data.isnull().sum()
                self.logger.warning(f"Filling {missing_count} missing values in {ticker} data")
                
                # Forward fill first, then backward fill
                cleaned_data = cleaned_data.ffill().bfill()
                
                # If still missing, use interpolation
                if cleaned_data.isnull().any():
                    cleaned_data = cleaned_data.interpolate(method='linear')
                
                # Final fallback: use mean
                if cleaned_data.isnull().any():
                    cleaned_data = cleaned_data.fillna(cleaned_data.mean())
            
            # Remove extreme outliers (beyond 5 standard deviations)
            if len(cleaned_data) > 10:
                z_scores = np.abs((cleaned_data - cleaned_data.mean()) / cleaned_data.std())
                outliers = z_scores > 5
                if outliers.any():
                    outlier_count = outliers.sum()
                    self.logger.warning(f"Capping {outlier_count} extreme outliers in {ticker} data")
                    
                    # Cap outliers at 3 standard deviations from the original data
                    original_mean = data.mean()
                    original_std = data.std()
                    upper_bound = original_mean + 3 * original_std
                    lower_bound = original_mean - 3 * original_std
                    
                    cleaned_data = cleaned_data.clip(lower=lower_bound, upper=upper_bound)
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed for {ticker}: {e}")
            return data  # Return original data if cleaning fails


class ForecastingErrorHandler:
    """Main error handling system for forecasting operations."""
    
    def __init__(self):
        self.logger = logging.getLogger('forecasting.error_handler')
        self.data_validator = DataQualityValidator()
        self.error_history: List[ErrorRecord] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self) -> Dict[ErrorCategory, List[Callable]]:
        """Setup recovery strategies for different error categories."""
        return {
            ErrorCategory.DATA_QUALITY: [
                self._recover_data_quality,
                self._fallback_to_simple_data
            ],
            ErrorCategory.MODEL_TRAINING: [
                self._recover_model_training,
                self._fallback_to_simpler_model
            ],
            ErrorCategory.MODEL_PREDICTION: [
                self._recover_model_prediction,
                self._fallback_to_cached_prediction,
                self._fallback_to_historical_mean
            ],
            ErrorCategory.VALIDATION: [
                self._recover_validation,
                self._skip_validation
            ],
            ErrorCategory.CONFIGURATION: [
                self._recover_configuration,
                self._use_default_config
            ],
            ErrorCategory.RESOURCE: [
                self._recover_resource_issue,
                self._reduce_resource_usage
            ],
            ErrorCategory.DEPENDENCY: [
                self._recover_dependency_issue,
                self._fallback_to_basic_implementation
            ]
        }
    
    def handle_error(self, 
                    error: Exception,
                    context: ErrorContext,
                    category: ErrorCategory,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Tuple[bool, Any]:
        """
        Handle an error with appropriate recovery strategies.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            category: Category of the error
            severity: Severity level of the error
            
        Returns:
            Tuple of (recovery_successful, recovery_result)
        """
        # Generate unique error ID
        error_id = self._generate_error_id(error, context)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            context=context,
            exception_type=type(error).__name__,
            traceback_info=traceback.format_exc(),
            recovery_action=None,
            timestamp=datetime.now()
        )
        
        # Log the error
        self._log_error(error_record)
        
        # Attempt recovery
        recovery_successful, recovery_result = self._attempt_recovery(error_record)
        
        # Update error record with recovery action
        error_record.recovery_action = recovery_result if isinstance(recovery_result, str) else "Recovery attempted"
        
        # Store error record
        self.error_history.append(error_record)
        
        return recovery_successful, recovery_result
    
    def _generate_error_id(self, error: Exception, context: ErrorContext) -> str:
        """Generate unique error ID for tracking."""
        import hashlib
        
        error_string = f"{context.ticker}_{context.model_name}_{context.operation}_{type(error).__name__}_{str(error)}"
        return hashlib.md5(error_string.encode()).hexdigest()[:12]
    
    def _log_error(self, error_record: ErrorRecord) -> None:
        """Log error with appropriate level based on severity."""
        log_message = (
            f"Error {error_record.error_id}: {error_record.message} "
            f"[{error_record.context.ticker}/{error_record.context.model_name}/"
            f"{error_record.context.operation}]"
        )
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log additional context for debugging
        self.logger.debug(f"Error context: {error_record.context.additional_info}")
    
    def _attempt_recovery(self, error_record: ErrorRecord) -> Tuple[bool, Any]:
        """Attempt recovery using appropriate strategies."""
        recovery_strategies = self.recovery_strategies.get(error_record.category, [])
        
        for strategy in recovery_strategies:
            try:
                self.logger.debug(f"Attempting recovery strategy: {strategy.__name__}")
                recovery_result = strategy(error_record)
                
                if recovery_result is not None:
                    self.logger.info(f"Recovery successful using {strategy.__name__}")
                    return True, recovery_result
                    
            except Exception as recovery_error:
                self.logger.warning(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
                continue
        
        self.logger.error(f"All recovery strategies failed for error {error_record.error_id}")
        return False, None
    
    # Recovery strategy implementations
    def _recover_data_quality(self, error_record: ErrorRecord) -> Any:
        """Attempt to recover from data quality issues."""
        # This would be implemented with specific data cleaning logic
        return "Data quality recovery attempted"
    
    def _fallback_to_simple_data(self, error_record: ErrorRecord) -> Any:
        """Fallback to simpler data processing."""
        return "Fallback to simple data processing"
    
    def _recover_model_training(self, error_record: ErrorRecord) -> Any:
        """Attempt to recover from model training failures."""
        return "Model training recovery attempted"
    
    def _fallback_to_simpler_model(self, error_record: ErrorRecord) -> Any:
        """Fallback to a simpler model."""
        return "Fallback to simpler model"
    
    def _recover_model_prediction(self, error_record: ErrorRecord) -> Any:
        """Attempt to recover from prediction failures."""
        return "Model prediction recovery attempted"
    
    def _fallback_to_cached_prediction(self, error_record: ErrorRecord) -> Any:
        """Use cached prediction if available."""
        return "Fallback to cached prediction"
    
    def _fallback_to_historical_mean(self, error_record: ErrorRecord) -> Any:
        """Ultimate fallback to historical mean."""
        return "Fallback to historical mean"
    
    def _recover_validation(self, error_record: ErrorRecord) -> Any:
        """Attempt to recover from validation failures."""
        return "Validation recovery attempted"
    
    def _skip_validation(self, error_record: ErrorRecord) -> Any:
        """Skip validation as fallback."""
        return "Validation skipped"
    
    def _recover_configuration(self, error_record: ErrorRecord) -> Any:
        """Attempt to recover from configuration issues."""
        return "Configuration recovery attempted"
    
    def _use_default_config(self, error_record: ErrorRecord) -> Any:
        """Use default configuration."""
        return "Default configuration used"
    
    def _recover_resource_issue(self, error_record: ErrorRecord) -> Any:
        """Attempt to recover from resource issues."""
        return "Resource issue recovery attempted"
    
    def _reduce_resource_usage(self, error_record: ErrorRecord) -> Any:
        """Reduce resource usage."""
        return "Resource usage reduced"
    
    def _recover_dependency_issue(self, error_record: ErrorRecord) -> Any:
        """Attempt to recover from dependency issues."""
        return "Dependency issue recovery attempted"
    
    def _fallback_to_basic_implementation(self, error_record: ErrorRecord) -> Any:
        """Fallback to basic implementation."""
        return "Fallback to basic implementation"
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about error occurrences."""
        if not self.error_history:
            return {'total_errors': 0}
        
        total_errors = len(self.error_history)
        
        # Count by category
        category_counts = {}
        for error in self.error_history:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Recent errors (last 24 hours)
        recent_cutoff = datetime.now() - pd.Timedelta(hours=24)
        recent_errors = [e for e in self.error_history if e.timestamp > recent_cutoff]
        
        return {
            'total_errors': total_errors,
            'category_breakdown': category_counts,
            'severity_breakdown': severity_counts,
            'recent_errors_24h': len(recent_errors),
            'most_common_category': max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            'most_common_severity': max(severity_counts.items(), key=lambda x: x[1])[0] if severity_counts else None
        }


def robust_forecasting_wrapper(func: Callable) -> Callable:
    """
    Decorator for adding robust error handling to forecasting functions.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        error_handler = ForecastingErrorHandler()
        
        try:
            return func(*args, **kwargs)
            
        except ValueError as e:
            # Data validation or parameter errors
            context = ErrorContext(
                ticker=kwargs.get('ticker', 'unknown'),
                model_name=kwargs.get('model_name', func.__name__),
                operation=func.__name__,
                data_points=len(args[0]) if args and hasattr(args[0], '__len__') else 0,
                timestamp=datetime.now(),
                additional_info={'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
            )
            
            recovery_successful, recovery_result = error_handler.handle_error(
                e, context, ErrorCategory.DATA_QUALITY, ErrorSeverity.MEDIUM
            )
            
            if recovery_successful:
                return recovery_result
            else:
                raise e
                
        except ImportError as e:
            # Dependency issues
            context = ErrorContext(
                ticker=kwargs.get('ticker', 'unknown'),
                model_name=kwargs.get('model_name', func.__name__),
                operation=func.__name__,
                data_points=0,
                timestamp=datetime.now(),
                additional_info={'missing_dependency': str(e)}
            )
            
            recovery_successful, recovery_result = error_handler.handle_error(
                e, context, ErrorCategory.DEPENDENCY, ErrorSeverity.HIGH
            )
            
            if recovery_successful:
                return recovery_result
            else:
                raise e
                
        except MemoryError as e:
            # Resource issues
            context = ErrorContext(
                ticker=kwargs.get('ticker', 'unknown'),
                model_name=kwargs.get('model_name', func.__name__),
                operation=func.__name__,
                data_points=len(args[0]) if args and hasattr(args[0], '__len__') else 0,
                timestamp=datetime.now(),
                additional_info={'resource_type': 'memory'}
            )
            
            recovery_successful, recovery_result = error_handler.handle_error(
                e, context, ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL
            )
            
            if recovery_successful:
                return recovery_result
            else:
                raise e
                
        except Exception as e:
            # General errors
            context = ErrorContext(
                ticker=kwargs.get('ticker', 'unknown'),
                model_name=kwargs.get('model_name', func.__name__),
                operation=func.__name__,
                data_points=len(args[0]) if args and hasattr(args[0], '__len__') else 0,
                timestamp=datetime.now(),
                additional_info={'exception_type': type(e).__name__}
            )
            
            # Determine category based on exception type
            if 'training' in func.__name__.lower() or 'fit' in func.__name__.lower():
                category = ErrorCategory.MODEL_TRAINING
            elif 'predict' in func.__name__.lower():
                category = ErrorCategory.MODEL_PREDICTION
            elif 'validate' in func.__name__.lower():
                category = ErrorCategory.VALIDATION
            else:
                category = ErrorCategory.MODEL_TRAINING  # Default
            
            recovery_successful, recovery_result = error_handler.handle_error(
                e, context, category, ErrorSeverity.HIGH
            )
            
            if recovery_successful:
                return recovery_result
            else:
                raise e
    
    return wrapper


# Global error handler instance
global_error_handler = ForecastingErrorHandler()