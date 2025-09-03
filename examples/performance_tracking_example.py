"""
Example usage of the model performance tracking infrastructure.

This script demonstrates how to use the performance tracking system
to monitor and compare different forecasting models.
"""

import numpy as np
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_performance import PerformanceTracker, PerformanceMetrics


def simulate_model_training_and_evaluation():
    """Simulate training and evaluating different models."""
    
    print("🚀 Starting Model Performance Tracking Example")
    print("=" * 50)
    
    # Initialize performance tracker
    tracker = PerformanceTracker(log_dir="logs/example_performance")
    
    # Simulate some realistic stock return data
    np.random.seed(42)
    tickers = ["AAPL", "GOOGL", "MSFT"]
    models = ["ARIMA", "LSTM", "XGBoost", "SARIMAX"]
    
    print(f"📊 Simulating performance for {len(tickers)} tickers and {len(models)} models")
    print()
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        
        # Generate realistic daily returns (mean ~0.1%, std ~2%)
        actual_returns = np.random.normal(0.001, 0.02, 100)
        
        for model in models:
            # Simulate different model performance characteristics
            model_configs = {
                "ARIMA": {"noise_factor": 0.6, "training_time": 8.0, "pred_time": 0.15},
                "LSTM": {"noise_factor": 0.4, "training_time": 45.0, "pred_time": 0.8},
                "XGBoost": {"noise_factor": 0.5, "training_time": 20.0, "pred_time": 0.3},
                "SARIMAX": {"noise_factor": 0.45, "training_time": 15.0, "pred_time": 0.2}
            }
            
            config = model_configs[model]
            
            # Generate predictions with model-specific noise
            noise = np.random.normal(0, config["noise_factor"] * np.std(actual_returns), 100)
            predicted_returns = actual_returns + noise
            
            # Record model performance
            performance = tracker.record_performance(
                ticker=ticker,
                model_name=model,
                y_true=actual_returns,
                y_pred=predicted_returns,
                training_time=config["training_time"],
                prediction_time=config["pred_time"],
                data_points=len(actual_returns)
            )
            
            print(f"  ✅ {model}: MAPE = {performance.mape:.2f}%, "
                  f"Training = {performance.training_time:.1f}s")
            
            # Record a forecast result
            recent_pred = np.mean(predicted_returns[-5:])
            pred_std = np.std(predicted_returns[-20:])
            
            forecast = tracker.record_forecast(
                ticker=ticker,
                expected_return=recent_pred,
                confidence_interval=(recent_pred - 1.96*pred_std, recent_pred + 1.96*pred_std),
                model_used=model,
                ensemble_weights={model: 1.0},
                validation_score=max(0.5, 1.0 - performance.mape/100)
            )
        
        print()
    
    # Analyze results
    print("📈 Performance Analysis")
    print("=" * 30)
    
    # Overall performance summary
    overall_summary = tracker.get_model_performance_summary()
    print(f"Total models evaluated: {overall_summary['count']}")
    print(f"Average MAPE: {overall_summary['avg_mape']:.2f}%")
    print(f"Average training time: {overall_summary['avg_training_time']:.1f}s")
    print()
    
    # Model-specific performance
    print("Model Performance Comparison:")
    for model in models:
        model_summary = tracker.get_model_performance_summary(model_name=model)
        print(f"  {model:8s}: MAPE = {model_summary['avg_mape']:.2f}%, "
              f"Time = {model_summary['avg_training_time']:.1f}s")
    
    print()
    
    # Best models by ticker
    print("🏆 Best Models by Ticker:")
    best_models = tracker.get_best_models_by_ticker(top_n=2)
    for ticker, models_list in best_models.items():
        print(f"  {ticker}: {', '.join(models_list[:2])}")
    
    print()
    
    # Demonstrate metrics calculation
    print("🔢 Performance Metrics Example:")
    y_true_example = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
    y_pred_example = np.array([0.012, 0.018, -0.008, 0.016, 0.007])
    
    metrics = PerformanceMetrics.calculate_all_metrics(y_true_example, y_pred_example)
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    
    print()
    
    # Save all data
    print("💾 Saving performance data...")
    tracker.save_all_data()
    print("✅ Data saved successfully!")
    
    print()
    print("🎉 Example completed successfully!")
    print("Check the 'logs/example_performance' directory for saved data.")


if __name__ == "__main__":
    simulate_model_training_and_evaluation()