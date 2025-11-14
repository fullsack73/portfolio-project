import unittest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from forecast_models import ARIMA, LSTMModel, XGBoostModel, ModelSelector

class TestMLModels(unittest.TestCase):
    """Focused tests for ML forecast models"""
    
    def setUp(self):
        """Generate sample data for testing"""
        np.random.seed(42)
        # Create 500 days of synthetic stock price data
        self.prices = 100 + np.cumsum(np.random.randn(500) * 2)
        self.prices = np.maximum(self.prices, 1)  # Ensure positive prices
        
    def test_arima_returns_volatility_tuple(self):
        """Test ARIMA returns both expected return and volatility"""
        model = ARIMA(seasonal=False, suppress_warnings=True)
        result = model.forecast(self.prices)
        
        self.assertIsInstance(result, tuple, "ARIMA should return tuple")
        self.assertEqual(len(result), 2, "ARIMA should return (return, volatility)")
        
        expected_return, volatility = result
        self.assertIsInstance(expected_return, (int, float), "Expected return should be numeric")
        self.assertIsInstance(volatility, (int, float), "Volatility should be numeric")
        self.assertGreater(volatility, 0, "Volatility should be positive")
        
    def test_lstm_forecast_output(self):
        """Test LSTM produces valid forecast output"""
        model = LSTMModel(layers=2, units=32, dropout=0.2)
        model.train(self.prices)
        forecast = model.forecast()
        
        self.assertIsInstance(forecast, (int, float), "LSTM forecast should be numeric")
        self.assertTrue(-1 < forecast < 2, "LSTM forecast should be reasonable annual return")
        
    def test_xgboost_feature_engineering(self):
        """Test XGBoost creates proper features and forecasts"""
        model = XGBoostModel()
        model.train(self.prices)
        forecast = model.forecast()
        
        self.assertIsInstance(forecast, (int, float), "XGBoost forecast should be numeric")
        self.assertTrue(-1 < forecast < 2, "XGBoost forecast should be reasonable annual return")
        
    def test_arima_fallback_on_insufficient_data(self):
        """Test ARIMA fallback mechanism with insufficient data"""
        model = ARIMA(seasonal=False, suppress_warnings=True)
        short_prices = self.prices[:5]  # Only 5 data points
        result = model.forecast(short_prices)
        
        self.assertIsInstance(result, tuple, "Should still return tuple on fallback")
        expected_return, volatility = result
        self.assertIsNotNone(expected_return, "Should return fallback value")
        
    def test_model_selector_selects_best(self):
        """Test ModelSelector identifies best-performing model"""
        selector = ModelSelector()
        
        # Split data: 80% train, 20% validation
        train_size = int(len(self.prices) * 0.8)
        train_data = self.prices[:train_size]
        val_data = self.prices[train_size:]
        
        best_model, metrics = selector.select_best_model(train_data, val_data)
        
        self.assertIsNotNone(best_model, "Should select a best model")
        self.assertIn('model_name', metrics, "Metrics should include model name")
        self.assertIn('r2', metrics, "Metrics should include R²")
        self.assertIn('rmse', metrics, "Metrics should include RMSE")
        
    def test_lstm_error_handling(self):
        """Test LSTM error handling with bad data"""
        model = LSTMModel(layers=2, units=32, dropout=0.2)
        
        # Test with very short data
        short_data = np.array([100, 101, 102])
        try:
            model.train(short_data)
            forecast = model.forecast()
            # Should either work with fallback or return reasonable value
            self.assertIsInstance(forecast, (int, float))
        except Exception as e:
            self.fail(f"LSTM should handle short data gracefully: {e}")
            
    def test_xgboost_error_handling(self):
        """Test XGBoost error handling with edge cases"""
        model = XGBoostModel()
        
        # Test with constant prices (no variation)
        constant_prices = np.ones(100) * 100
        try:
            model.train(constant_prices)
            forecast = model.forecast()
            # Should handle gracefully
            self.assertIsInstance(forecast, (int, float))
        except Exception as e:
            self.fail(f"XGBoost should handle constant data gracefully: {e}")

if __name__ == '__main__':
    unittest.main()
