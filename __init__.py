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