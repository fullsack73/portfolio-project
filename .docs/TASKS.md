# Project Tasks

This file outlines the tasks completed during the development of the portfolio analysis and optimization tool.

## ✅ Completed Tasks

### Project Setup & Foundation
- [x] Initialize React frontend and Python backend.
- [x] Establish communication between frontend and backend.
- [x] Set up Git, ESLint, and project structure.

### Backend Development
- [x] Implement APIs for historical stock data, financial statements, and hedging analysis.
- [x] Develop Monte Carlo simulation for portfolio projection.
- [x] Implement portfolio optimization using the Markowitz model.
- [x] Add utility to read stock tickers from CSV files.

### Frontend Development
- [x] Design main application layout and UI components.
- [x] Create components for ticker input, portfolio allocation, and date selection.
- [x] Implement charts for historical data, portfolio performance, and Monte Carlo simulations.
- [x] Build components for financial statements, optimization, and hedging analysis.

### Internationalization (i18n)
- [x] Integrate `i18next` for English and Korean language support.
- [x] Implement a language selector and wrap UI text with translation functions.

### ML-Based Enhancements
- [x] Integrate `Prophet` for time-series forecasting of asset returns.
- [x] Update backend and frontend to support ML-based forecasting as an option.
- [x] Set ML forecasting as the default and implement a resilient data pipeline.
- [x] Refactor regression model to predict price changes for better trend extrapolation.

### Performance Optimization
- [x] Implement concurrent processing for model training and data fetching.
- [x] Optimize data fetching with batch downloads.
- [x] Introduce lightweight forecasting models (ARIMA, Exponential Smoothing).
- [x] Implement multi-level caching (In-Memory & Redis/Disk) with cache warming.

### Refactoring & Bug Fixes
- [x] Refactor frontend state management for centralized data updates and auto-updates on input change.
- [x] Fix request spamming by implementing debouncing and request cancellation.
- [x] Resolve various bugs, including data alignment errors, Prophet model issues, and `NameError` exceptions.
- [x] Corrected data handling for `yfinance` MultiIndex DataFrames.
- [x] Fixed flat line issue in the regression forecast graph.

### Stock Screener Feature
- [x] **Backend:** Implement `finvizfinance`-based stock screener with a dedicated API endpoint.
- [x] **Screener Module:** Create `stock_screener.py` for screening logic and filter translation.
- [x] **Frontend:** Develop `StockScreener.jsx` component with dynamic filter controls.
- [x] **State & API:** Manage state with hooks and integrate with the backend screener API.
- [x] **Results Display:** Show screener results in a table and include metric values.
- [x] **Save Results:** Add a button to save screener results to a CSV file.
- [x] **i18n & UI:** Add translations and fix UI styling issues.

### Removing Montecarlo Based Portfolio Analysis
- [x] **Frontend:** Delete components and instances that are related to "Portfolio Analysis" feature
- [x] **Backend:** Delete files and instances that are related to "Portfolio Analysis" feature

### Documentation
- [x] Create `README.md`, `REQUIREMENTS.md`, `DESIGN.md`, and `TASKS.md`.
- [x] Keep documentation updated with new features and architectural changes.

## TODO

### Replacing Prophet

Replace current procedure of Prophet + ensemble in port opt for enhanced accuracy.
Candidates are: XGBoost, ARIMA, LSTM and possibly Fedot

- [ ] 1. Create Module for ML models
    - [ ] ARIMA
    - [ ] XGBoost
    - [ ] LSTM
- [ ] 2. Implement to current pipeline
    - Make Module that handles multiple tickers at once as a batch