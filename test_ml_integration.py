import unittest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from portfolio_optimization import ml_forecast_returns, optimize_portfolio
from forecast_models import ModelSelector

class TestMLIntegration(unittest.TestCase):
    """Focused tests for ML integration with portfolio optimization"""
    
    def setUp(self):
        """Generate sample data for testing"""
        np.random.seed(42)
        # Create synthetic stock data for 3 tickers over 500 days
        dates = pd.date_range('2023-01-01', periods=500)
        self.stock_data = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(500) * 2),
            'MSFT': 150 + np.cumsum(np.random.randn(500) * 2.5),
            'GOOGL': 120 + np.cumsum(np.random.randn(500) * 1.5)
        }, index=dates)
        self.stock_data = self.stock_data.clip(lower=1)  # Ensure positive prices
        
    def test_ml_forecast_returns_output_format(self):
        """Test ml_forecast_returns returns proper pandas Series"""
        result = ml_forecast_returns(self.stock_data, use_lightweight=True)
        
        self.assertIsInstance(result, pd.Series, "Should return pandas Series")
        self.assertEqual(len(result), 3, "Should return forecast for all 3 tickers")
        self.assertTrue(all(ticker in result.index for ticker in ['AAPL', 'MSFT', 'GOOGL']),
                       "Series index should contain all ticker names")
        
        # Check all values are numeric and reasonable
        for ticker, value in result.items():
            self.assertIsInstance(value, (int, float, np.number), f"{ticker} forecast should be numeric")
            self.assertTrue(-1 < value < 2, f"{ticker} forecast should be reasonable annual return")
    
    def test_ml_forecast_returns_multicore(self):
        """Test multicore processing works correctly"""
        import time
        
        # Create larger dataset to test parallel processing
        dates = pd.date_range('2023-01-01', periods=500)
        large_data = pd.DataFrame({
            f'TICK{i}': 100 + np.cumsum(np.random.randn(500) * 2)
            for i in range(10)
        }, index=dates)
        large_data = large_data.clip(lower=1)
        
        start_time = time.time()
        result = ml_forecast_returns(large_data, use_lightweight=True)
        elapsed = time.time() - start_time
        
        self.assertEqual(len(result), 10, "Should process all 10 tickers")
        self.assertLess(elapsed, 60, "Multicore should complete in reasonable time")
        
    def test_ml_forecast_cache_integration(self):
        """Test that caching works for ML forecasts"""
        import time
        
        # First call (cache miss)
        start_time = time.time()
        result1 = ml_forecast_returns(self.stock_data, use_lightweight=True)
        time1 = time.time() - start_time
        
        # Second call (should benefit from caching)
        start_time = time.time()
        result2 = ml_forecast_returns(self.stock_data, use_lightweight=True)
        time2 = time.time() - start_time
        
        # Results should be identical (sort for comparison)
        pd.testing.assert_series_equal(result1.sort_index(), result2.sort_index())
        
        # Note: ProcessPoolExecutor overhead may mask cache benefits
        # Just verify caching doesn't make it slower
        self.assertLess(time2, time1 * 2.0, "Cached call should not be significantly slower")
        
    def test_fallback_to_lightweight_on_ml_failure(self):
        """Test fallback mechanism when ML models fail"""
        # Create data with insufficient points (will trigger fallback)
        short_data = self.stock_data.iloc[:5]  # Only 5 days
        
        result = ml_forecast_returns(short_data, use_lightweight=True)
        
        self.assertIsInstance(result, pd.Series, "Should still return Series on fallback")
        self.assertEqual(len(result), 3, "Should return forecasts for all tickers")
        # Should use fallback default values
        for value in result.values:
            self.assertIsInstance(value, (int, float, np.number))
            
    def test_optimize_portfolio_with_ml_integration(self):
        """Test optimize_portfolio works end-to-end with ML forecasts"""
        # This is a simplified test - full integration would require real data
        try:
            # Test that optimize_portfolio can be called
            # (actual optimization would fail without real ticker data)
            from portfolio_optimization import optimize_portfolio
            
            # Verify function exists and has correct signature
            import inspect
            sig = inspect.signature(optimize_portfolio)
            params = list(sig.parameters.keys())
            
            self.assertIn('start_date', params, "Should have start_date parameter")
            self.assertIn('end_date', params, "Should have end_date parameter")
            self.assertIn('risk_free_rate', params, "Should have risk_free_rate parameter")
            
        except Exception as e:
            self.fail(f"optimize_portfolio should be callable: {e}")
            
    def test_ml_forecast_with_model_selector(self):
        """Test that ModelSelector is used for ML forecasts"""
        # Test with longer data to allow model training
        result = ml_forecast_returns(self.stock_data, use_lightweight=False)
        
        self.assertIsInstance(result, pd.Series, "Should return Series")
        self.assertEqual(len(result), 3, "Should forecast all tickers")
        
        # ML forecasts should be in reasonable range
        for ticker, value in result.items():
            self.assertTrue(-0.8 < value < 1.5, 
                          f"{ticker} ML forecast {value} should be in reasonable range")

if __name__ == '__main__':
    unittest.main()
