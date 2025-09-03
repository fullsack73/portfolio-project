"""
Financial Analysis and Portfolio Optimization Package

This package provides advanced forecasting models, portfolio optimization,
and financial analysis tools.
"""

__version__ = "1.0.0"
__author__ = "Financial Analysis Team"

# Import main modules for easier access
try:
    from .portfolio_optimization import optimize_portfolio, get_forecasting_performance_dashboard, advanced_forecast_returns
    from .forecasting_models import ForecastingModelManager
    from .model_performance import ModelPerformanceTracker
    from .cache_manager import CacheManager
    
    __all__ = [
        'optimize_portfolio',
        'get_forecasting_performance_dashboard', 
        'advanced_forecast_returns',
        'ForecastingModelManager',
        'ModelPerformanceTracker',
        'CacheManager'
    ]
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = []