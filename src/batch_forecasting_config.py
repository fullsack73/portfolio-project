"""
Configuration classes for batch forecasting system.

This module defines the configuration data structures used throughout
the batch forecasting optimization system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    
    strategy: str = "single_batch"  # "single_batch", "chunked"
    max_batch_size: Optional[int] = None  # None means no limit, process all together
    chunk_size: int = 50  # Size of chunks if using chunked strategy
    enable_batch_processing: bool = True  # Master switch for batch processing


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    
    include_market_features: bool = True
    include_sector_features: bool = True
    include_technical_indicators: bool = True
    lookback_window: int = 252  # Days of historical data to use
    cache_features: bool = True  # Enable feature caching
    
    # Technical indicator settings
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    
    # Moving average windows
    ma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class ModelConfig:
    """Configuration for batch models."""
    
    primary_model: str = "batch_xgboost"  # Single hardcoded model for speed
    fallback_model: str = "batch_linear"  # Simple fallback model
    use_ensemble: bool = False  # Disabled for speed
    parallel_training: bool = True
    max_training_time: int = 60  # Reduced timeout for speed
    
    # XGBoost specific parameters (hardcoded for speed)
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    })


@dataclass
class FallbackConfig:
    """Configuration for fallback system."""
    
    enable_fallback: bool = True
    max_retries: int = 3
    fallback_to_individual: bool = True  # Fall back to individual processing if batch fails
    memory_threshold_mb: int = 4096  # Memory threshold for automatic batch size reduction
    timeout_seconds: int = 300  # Timeout for batch processing


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    
    enable_monitoring: bool = True
    log_batch_metrics: bool = True
    track_memory_usage: bool = True
    benchmark_against_individual: bool = True
    performance_report_interval: int = 100  # Log performance every N tickers


@dataclass
class BatchPerformanceMetrics:
    """Performance metrics for batch processing."""
    
    batch_size: int
    processing_time: float
    memory_usage: float
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    speedup_factor: float = 0.0  # Compared to individual processing
    fallback_rate: float = 0.0  # Percentage of tickers that fell back to individual processing
    model_used: str = ""
    feature_extraction_time: float = 0.0
    model_training_time: float = 0.0
    prediction_time: float = 0.0


@dataclass
class BatchForecastingConfig:
    """Main configuration for batch forecasting system."""
    
    # Component configurations
    batch: BatchConfig = field(default_factory=BatchConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Global settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 1800  # 30 minutes
    log_level: str = "INFO"
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate batch configuration
            if self.batch.strategy not in ["single_batch", "chunked"]:
                raise ValueError(f"Invalid batch strategy: {self.batch.strategy}")
            
            if self.batch.chunk_size <= 0:
                raise ValueError(f"Chunk size must be positive: {self.batch.chunk_size}")
            
            # Validate feature configuration
            if self.features.lookback_window <= 0:
                raise ValueError(f"Lookback window must be positive: {self.features.lookback_window}")
            
            if not self.features.ma_windows:
                raise ValueError("Moving average windows cannot be empty")
            
            # Validate model configuration
            if self.models.primary_model not in ["batch_xgboost", "batch_linear"]:
                raise ValueError(f"Invalid primary model: {self.models.primary_model}")
            
            if self.models.fallback_model not in ["batch_linear", "batch_xgboost"]:
                raise ValueError(f"Invalid fallback model: {self.models.fallback_model}")
            
            if self.models.max_training_time <= 0:
                raise ValueError(f"Training timeout must be positive: {self.models.max_training_time}")
            
            # Validate fallback configuration
            if self.fallback.max_retries < 0:
                raise ValueError(f"Max retries cannot be negative: {self.fallback.max_retries}")
            
            if self.fallback.memory_threshold_mb <= 0:
                raise ValueError(f"Memory threshold must be positive: {self.fallback.memory_threshold_mb}")
            
            logger.info("Batch forecasting configuration validation passed")
            return True
            
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


def get_default_batch_config() -> BatchForecastingConfig:
    """Get default batch forecasting configuration."""
    return BatchForecastingConfig()


def get_speed_optimized_config() -> BatchForecastingConfig:
    """Get speed-optimized configuration for maximum performance."""
    config = BatchForecastingConfig()
    
    # Optimize for speed
    config.batch.strategy = "single_batch"  # Process all tickers together
    config.batch.max_batch_size = None  # No size limit
    
    # Minimal feature set for speed
    config.features.include_sector_features = False  # Disable complex sector analysis
    config.features.ma_windows = [10, 20]  # Fewer moving averages
    config.features.volatility_windows = [10]  # Single volatility window
    config.features.lookback_window = 126  # Shorter lookback (6 months)
    
    # Fast model settings
    config.models.primary_model = "batch_linear"  # Use linear model for speed
    config.models.fallback_model = "batch_linear"
    config.models.max_training_time = 30  # Shorter timeout
    config.models.xgboost_params['n_estimators'] = 50  # Fewer trees
    
    # Reduced monitoring overhead
    config.performance.track_memory_usage = False
    config.performance.benchmark_against_individual = False
    
    logger.info("Created speed-optimized batch forecasting configuration")
    return config


def get_accuracy_optimized_config() -> BatchForecastingConfig:
    """Get accuracy-optimized configuration for best forecast quality."""
    config = BatchForecastingConfig()
    
    # Optimize for accuracy
    config.batch.strategy = "single_batch"  # Still use single batch for cross-ticker learning
    
    # Full feature set
    config.features.include_market_features = True
    config.features.include_sector_features = True
    config.features.include_technical_indicators = True
    config.features.ma_windows = [5, 10, 20, 50, 100]  # More moving averages
    config.features.volatility_windows = [5, 10, 20, 50]  # More volatility windows
    config.features.lookback_window = 504  # Longer lookback (2 years)
    
    # Advanced model settings
    config.models.primary_model = "batch_xgboost"  # Use XGBoost for accuracy
    config.models.max_training_time = 120  # Longer timeout
    config.models.xgboost_params['n_estimators'] = 200  # More trees
    config.models.xgboost_params['max_depth'] = 8  # Deeper trees
    
    # Full monitoring
    config.performance.track_memory_usage = True
    config.performance.benchmark_against_individual = True
    
    logger.info("Created accuracy-optimized batch forecasting configuration")
    return config