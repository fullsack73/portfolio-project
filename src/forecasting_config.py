"""
Configuration management system for advanced forecasting models.

This module provides centralized configuration management for all forecasting models,
including parameter settings, validation, and environment-specific configurations.
"""

import os
import json
import logging
import copy
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for individual forecasting models."""
    enabled: bool
    priority: int
    params: Dict[str, Any]
    max_training_time: Optional[int] = None
    accuracy_threshold: Optional[float] = None


@dataclass
class ValidationConfig:
    """Configuration for model validation and cross-validation."""
    test_size: float = 0.2
    cv_folds: int = 3
    metrics: List[str] = None
    min_data_points: int = 100
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['mse', 'mae', 'mape']


@dataclass
class PerformanceConfig:
    """Configuration for performance thresholds and limits."""
    max_training_time: int = 300  # seconds
    max_prediction_time: int = 30  # seconds
    accuracy_threshold: float = 0.15  # MAPE
    ensemble_threshold: int = 3  # minimum models for ensemble
    memory_limit_mb: int = 1024
    parallel_workers: Optional[int] = None
    
    def __post_init__(self):
        if self.parallel_workers is None:
            self.parallel_workers = min(4, os.cpu_count() or 1)


@dataclass
class CachingConfig:
    """Configuration for model and prediction caching."""
    enabled: bool = True
    model_cache_ttl: int = 3600  # seconds
    prediction_cache_ttl: int = 1800  # seconds
    max_cache_size_mb: int = 512
    cache_directory: str = ".cache/forecasting"


@dataclass
class ForecastingConfig:
    """Main configuration class containing all forecasting settings."""
    models: Dict[str, ModelConfig]
    validation: ValidationConfig
    performance: PerformanceConfig
    caching: CachingConfig
    environment: str = "development"
    logging_level: str = "INFO"


# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    'lstm': ModelConfig(
        enabled=True,
        priority=1,
        params={
            'sequence_length': 60,
            'lstm_units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'early_stopping_patience': 10,
            'validation_split': 0.2
        },
        max_training_time=600,
        accuracy_threshold=0.12
    ),
    'xgboost': ModelConfig(
        enabled=True,
        priority=2,
        params={
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        },
        max_training_time=300,
        accuracy_threshold=0.15
    ),
    'lightgbm': ModelConfig(
        enabled=True,
        priority=3,
        params={
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        },
        max_training_time=200,
        accuracy_threshold=0.15
    ),
    'catboost': ModelConfig(
        enabled=True,
        priority=4,
        params={
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        },
        max_training_time=400,
        accuracy_threshold=0.15
    ),
    'sarimax': ModelConfig(
        enabled=True,
        priority=5,
        params={
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 12),
            'trend': 'c',
            'enforce_stationarity': False,
            'enforce_invertibility': False
        },
        max_training_time=180,
        accuracy_threshold=0.18
    ),
    'arima': ModelConfig(
        enabled=True,
        priority=6,
        params={
            'order': (1, 1, 1),
            'trend': 'c',
            'enforce_stationarity': False,
            'enforce_invertibility': False,
            'auto_arima': True,
            'max_p': 5,
            'max_q': 5,
            'max_d': 2
        },
        max_training_time=120,
        accuracy_threshold=0.20
    )
}

# Default configuration
DEFAULT_CONFIG = ForecastingConfig(
    models=DEFAULT_MODEL_CONFIGS,
    validation=ValidationConfig(),
    performance=PerformanceConfig(),
    caching=CachingConfig()
)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigurationManager:
    """Manages loading, validation, and access to forecasting configurations."""
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (development, production, testing)
        """
        self.environment = environment or os.getenv('FORECASTING_ENV', 'development')
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[ForecastingConfig] = None
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path based on environment."""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        if self.environment == 'testing':
            return str(config_dir / "forecasting_test.json")
        elif self.environment == 'production':
            return str(config_dir / "forecasting_prod.json")
        else:
            return str(config_dir / "forecasting_dev.json")
    
    def load_config(self, force_reload: bool = False) -> ForecastingConfig:
        """
        Load configuration from file or return cached configuration.
        
        Args:
            force_reload: Force reload from file even if cached
            
        Returns:
            ForecastingConfig: Loaded configuration
            
        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        if self._config is not None and not force_reload:
            return self._config
            
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                self._config = self._parse_config_data(config_data)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                self._config = self._create_default_config()
                self.save_config(self._config)
                logger.info(f"Created default configuration at {self.config_path}")
                
            self._validate_config(self._config)
            return self._config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def _parse_config_data(self, config_data: Dict[str, Any]) -> ForecastingConfig:
        """Parse configuration data from dictionary."""
        # Parse models
        models = {}
        for model_name, model_data in config_data.get('models', {}).items():
            models[model_name] = ModelConfig(**model_data)
        
        # Parse other sections
        validation = ValidationConfig(**config_data.get('validation', {}))
        performance = PerformanceConfig(**config_data.get('performance', {}))
        caching = CachingConfig(**config_data.get('caching', {}))
        
        return ForecastingConfig(
            models=models,
            validation=validation,
            performance=performance,
            caching=caching,
            environment=config_data.get('environment', self.environment),
            logging_level=config_data.get('logging_level', 'INFO')
        )
    
    def _create_default_config(self) -> ForecastingConfig:
        """Create default configuration for the current environment."""
        # Create deep copies to avoid modifying global defaults
        models = {}
        for name, model in DEFAULT_MODEL_CONFIGS.items():
            model_dict = asdict(model)
            models[name] = ModelConfig(**model_dict)
        
        validation = ValidationConfig(**asdict(DEFAULT_CONFIG.validation))
        performance = PerformanceConfig(**asdict(DEFAULT_CONFIG.performance))
        caching = CachingConfig(**asdict(DEFAULT_CONFIG.caching))
        
        config = ForecastingConfig(
            models=models,
            validation=validation,
            performance=performance,
            caching=caching,
            environment=self.environment,
            logging_level=DEFAULT_CONFIG.logging_level
        )
        
        # Adjust settings based on environment
        if self.environment == 'production':
            config.performance.max_training_time = 600
            config.performance.parallel_workers = min(8, os.cpu_count() or 1)
            config.caching.model_cache_ttl = 7200
            config.logging_level = 'WARNING'
        elif self.environment == 'testing':
            config.performance.max_training_time = 60
            config.performance.parallel_workers = 1
            config.caching.enabled = False
            config.logging_level = 'DEBUG'
            # Disable resource-intensive models for testing
            config.models['lstm'].enabled = False
            
        return config
    
    def save_config(self, config: ForecastingConfig) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            
        Raises:
            ConfigurationError: If saving fails
        """
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Convert to dictionary
            config_dict = self._config_to_dict(config)
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            logger.info(f"Saved configuration to {self.config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    def _config_to_dict(self, config: ForecastingConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        return {
            'models': {name: asdict(model_config) for name, model_config in config.models.items()},
            'validation': asdict(config.validation),
            'performance': asdict(config.performance),
            'caching': asdict(config.caching),
            'environment': config.environment,
            'logging_level': config.logging_level
        }
    
    def _validate_config(self, config: ForecastingConfig) -> None:
        """
        Validate configuration settings.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Validate environment
        valid_environments = ['development', 'production', 'testing']
        if config.environment not in valid_environments:
            raise ConfigurationError(f"Invalid environment: {config.environment}")
        
        # Validate logging level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.logging_level not in valid_log_levels:
            raise ConfigurationError(f"Invalid logging level: {config.logging_level}")
        
        # Validate models
        if not config.models:
            raise ConfigurationError("No models configured")
        
        enabled_models = [name for name, model in config.models.items() if model.enabled]
        if not enabled_models:
            raise ConfigurationError("No models are enabled")
        
        # Validate model priorities
        priorities = [model.priority for model in config.models.values() if model.enabled]
        if len(set(priorities)) != len(priorities):
            raise ConfigurationError("Model priorities must be unique")
        
        # Validate validation config
        if not (0 < config.validation.test_size < 1):
            raise ConfigurationError("test_size must be between 0 and 1")
        
        if config.validation.cv_folds < 2:
            raise ConfigurationError("cv_folds must be at least 2")
        
        if config.validation.min_data_points < 50:
            raise ConfigurationError("min_data_points must be at least 50")
        
        # Validate performance config
        if config.performance.max_training_time <= 0:
            raise ConfigurationError("max_training_time must be positive")
        
        if config.performance.accuracy_threshold <= 0:
            raise ConfigurationError("accuracy_threshold must be positive")
        
        if config.performance.ensemble_threshold < 2:
            raise ConfigurationError("ensemble_threshold must be at least 2")
        
        # Validate caching config
        if config.caching.enabled:
            if config.caching.model_cache_ttl <= 0:
                raise ConfigurationError("model_cache_ttl must be positive")
            
            if config.caching.prediction_cache_ttl <= 0:
                raise ConfigurationError("prediction_cache_ttl must be positive")
    
    def get_config(self) -> ForecastingConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model."""
        config = self.get_config()
        return config.models.get(model_name)
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled model names sorted by priority."""
        config = self.get_config()
        enabled_models = [(name, model.priority) for name, model in config.models.items() if model.enabled]
        return [name for name, _ in sorted(enabled_models, key=lambda x: x[1])]
    
    def update_model_config(self, model_name: str, **kwargs) -> None:
        """Update configuration for specific model."""
        config = self.get_config()
        if model_name not in config.models:
            raise ConfigurationError(f"Model {model_name} not found in configuration")
        
        model_config = config.models[model_name]
        for key, value in kwargs.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
            elif key in model_config.params:
                model_config.params[key] = value
            else:
                raise ConfigurationError(f"Invalid configuration key: {key}")
        
        self._validate_config(config)


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(config_path: Optional[str] = None, environment: Optional[str] = None) -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_path, environment)
    return _config_manager


def get_forecasting_config() -> ForecastingConfig:
    """Get current forecasting configuration."""
    return get_config_manager().get_config()


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for specific model."""
    return get_config_manager().get_model_config(model_name)


def get_enabled_models() -> List[str]:
    """Get list of enabled model names sorted by priority."""
    return get_config_manager().get_enabled_models()