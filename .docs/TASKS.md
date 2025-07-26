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

### Documentation
- [x] Create `README.md`, `REQUIREMENTS.md`, `DESIGN.md`, and `TASKS.md`.
- [x] Keep documentation updated with new features and architectural changes.

## 🚧 Planned Future Optimizations(for portfolio optimizer)
- [ ] Memory-intensive pre-processing and model pre-training.
- [ ] **Ultra-Parallel Processing**
    - [x] **Hybrid Processing Model:**
        - [x] Use `ProcessPoolExecutor` for CPU-intensive model training (`Prophet`, `ARIMA`).
        - [x] Use `ThreadPoolExecutor` for I/O-bound tasks (API calls, cache operations).
        - [x] Integrate `asyncio` for non-blocking I/O.
    - [ ] **GPU Acceleration (Optional):**
        - [ ] Research `CuPy`/`Numba` for GPU-accelerated matrix operations.
        - [ ] Implement GPU-based covariance matrix calculations.
- [ ] Optimized data structures and predictive prefetching.
- [ ] Load test caching system and analyze hit ratios.