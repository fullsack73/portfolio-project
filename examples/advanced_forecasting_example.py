#!/usr/bin/env python3
"""
Advanced Forecasting Integration Example

This example demonstrates how to use the new advanced forecasting capabilities
integrated into the portfolio optimization system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolio_optimization import (
    optimize_portfolio, 
    get_forecasting_performance_dashboard,
    advanced_forecast_returns
)

def main():
    """Demonstrate advanced forecasting integration."""
    
    print("=== Advanced Forecasting Integration Example ===\n")
    
    # Example parameters
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    risk_free_rate = 0.02
    
    # Example tickers (small set for demonstration)
    example_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print("1. Portfolio Optimization with Advanced Forecasting")
    print(f"   Tickers: {example_tickers}")
    print(f"   Date Range: {start_date} to {end_date}")
    print(f"   Risk-free Rate: {risk_free_rate}")
    print()
    
    try:
        # Run portfolio optimization with advanced forecasting
        print("Running portfolio optimization with advanced forecasting...")
        result = optimize_portfolio(
            start_date=start_date,
            end_date=end_date,
            risk_free_rate=risk_free_rate,
            tickers=example_tickers,
            use_advanced_forecasting=True,
            forecasting_method='auto',  # Auto-select based on portfolio size
            max_models_per_ticker=3  # Limit models for faster execution
        )
        
        print("✓ Portfolio optimization completed successfully!")
        print()
        
        # Display results
        print("2. Optimization Results")
        print(f"   Expected Return: {result['return']:.4f}")
        print(f"   Risk (Std Dev): {result['risk']:.4f}")
        print(f"   Sharpe Ratio: {result['sharpe_ratio']:.4f}")
        print()
        
        print("   Portfolio Weights:")
        for ticker, weight in result['weights'].items():
            print(f"     {ticker}: {weight:.4f} ({weight*100:.2f}%)")
        print()
        
        # Display performance metrics if available
        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            print("3. Forecasting Performance Metrics")
            print(f"   System Health Score: {metrics['system_health_score']:.1f}%")
            print(f"   Total Forecasts: {metrics['total_forecasts']}")
            print(f"   Successful Forecasts: {metrics['successful_forecasts']}")
            print()
            
            if metrics['model_usage_stats']:
                print("   Model Usage Statistics:")
                for model_name, stats in metrics['model_usage_stats'].items():
                    print(f"     {model_name}:")
                    print(f"       Success Rate: {stats['success_rate']:.1f}%")
                    print(f"       Avg Training Time: {stats['average_training_time']:.2f}s")
                    print(f"       Avg Prediction Time: {stats['average_prediction_time']:.4f}s")
                    print(f"       Total Calls: {stats['total_calls']}")
                print()
            
            if metrics['top_performing_models']:
                print("   Top Performing Models:")
                for i, (model_name, score) in enumerate(metrics['top_performing_models'], 1):
                    print(f"     {i}. {model_name} (Score: {score:.4f})")
                print()
            
            if metrics['recommendations']:
                print("   Performance Recommendations:")
                for rec in metrics['recommendations']:
                    print(f"     • {rec}")
                print()
        
        # Demonstrate performance dashboard
        print("4. Performance Dashboard")
        dashboard = get_forecasting_performance_dashboard()
        
        if 'error' not in dashboard:
            overview = dashboard['system_overview']
            print(f"   System Health: {overview['health_score']:.1f}%")
            print(f"   Success Rate: {overview['success_rate']:.1f}%")
            print(f"   Avg Processing Time: {overview['average_processing_time']:.4f}s")
            print(f"   System Uptime: {overview['uptime_hours']:.2f} hours")
            print()
            
            if dashboard['recent_alerts']:
                print("   Recent Alerts:")
                for alert in dashboard['recent_alerts'][-3:]:  # Show last 3 alerts
                    print(f"     • {alert['message']} ({alert['severity']})")
                print()
        else:
            print(f"   Dashboard Error: {dashboard['error']}")
            print()
        
        print("5. Comparison: Advanced vs Lightweight Forecasting")
        print("   Running lightweight forecasting for comparison...")
        
        # Run with lightweight forecasting for comparison
        result_lightweight = optimize_portfolio(
            start_date=start_date,
            end_date=end_date,
            risk_free_rate=risk_free_rate,
            tickers=example_tickers,
            use_advanced_forecasting=False  # Use lightweight forecasting
        )
        
        print("   Comparison Results:")
        print(f"     Advanced   - Return: {result['return']:.4f}, Risk: {result['risk']:.4f}, Sharpe: {result['sharpe_ratio']:.4f}")
        print(f"     Lightweight - Return: {result_lightweight['return']:.4f}, Risk: {result_lightweight['risk']:.4f}, Sharpe: {result_lightweight['sharpe_ratio']:.4f}")
        
        # Calculate improvement
        sharpe_improvement = ((result['sharpe_ratio'] - result_lightweight['sharpe_ratio']) / result_lightweight['sharpe_ratio']) * 100
        print(f"     Sharpe Ratio Improvement: {sharpe_improvement:+.2f}%")
        print()
        
    except Exception as e:
        print(f"❌ Error during portfolio optimization: {e}")
        print("This might be due to missing dependencies or network issues.")
        print("The advanced forecasting system requires additional ML libraries.")
        print()
        
        # Show what would be available
        print("Available forecasting methods:")
        print("  • Advanced ML models (ARIMA, LSTM, XGBoost, LightGBM, CatBoost, SARIMAX)")
        print("  • Ensemble forecasting with automatic model selection")
        print("  • Performance monitoring and caching")
        print("  • Automatic fallback to lightweight methods")
        print()
    
    print("=== Example Complete ===")


if __name__ == "__main__":
    main()