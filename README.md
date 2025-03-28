# Stock Data Visualization and Regression Analysis Program

## Overview
This program fetches historical stock market data, visualizes it, and performs regression analysis to identify trends and make predictions. It's designed for investors, analysts, and anyone interested in quantitative financial analysis.

## Features

### 1. Stock Data Retrieval
- Fetches historical stock prices from financial APIs (Yahoo Finance, Alpha Vantage, etc.)
- Supports multiple stocks and custom date ranges
- Handles dividend adjustments and stock splits

### 2. Data Visualization
- Interactive candlestick charts showing Open-High-Low-Close (OHLC) data
- Volume indicators
- Moving averages (SMA, EMA) with customizable periods
- Bollinger Bands and other technical indicators

### 3. Regression Analysis
- Linear regression models to identify trends
- Polynomial regression for non-linear relationships
- Time series forecasting with ARIMA models
- R-squared values and other statistical metrics to evaluate model fit

## How It Works

### Data Flow
1. **Input**: User selects stock symbol(s) and date range
2. **Data Fetching**: Program retrieves historical data from API
3. **Preprocessing**: Cleans and formats data (handling missing values, adjusting for splits)
4. **Analysis**: Performs selected regression models
5. **Visualization**: Generates interactive charts
6. **Output**: Displays results with statistical insights