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

## 🆕 Stock Screener Feature (Requirement 4.3)

Implement a feature to search/screen for stocks based on financial ratio conditions.

### Backend (Implementation via `finvizfinance`)

- [ ] **Dependency:**
    - [ ] Add `finvizfinance` to `requirements.txt`.
- [ ] **Screener Module:**
    - [ ] Create a new file `src/stock_screener.py` to house the screening logic.
    - [ ] In this file, create a function `search_stocks(filters)` that accepts filter criteria from the API.
- [ ] **API Endpoint (`/api/stock-screener`):**
    - [ ] In `app.py`, create the endpoint that receives search criteria from the frontend.
    - [ ] The endpoint will call `stock_screener.search_stocks(filters)`.
- [ ] **Filtering Logic:**
    - [ ] The `search_stocks` function will translate the frontend criteria into the format required by `finvizfinance`.
    - [ ] It will use `finvizfinance.screener.overview.Overview` to set the filters and execute the search.
    - [ ] The function will handle potential errors if the external Finviz service is unavailable.
    - [ ] The results (a pandas DataFrame) will be converted to JSON to be sent to the frontend.

### Frontend

- [ ] **UI Component (`StockScreener.jsx`):**
    - [ ] Create a new parent component to house the feature.
    - [ ] Integrate this component into the `FinancialStatement.jsx` view.
- [ ] **Input Controls:**
    - [ ] Add a selector for the Ticker Group (S&P 500, Dow, Custom List). Can reuse "optimizer-select" at 'Optimizer.jsx'.
    - [ ] Design and build a dynamic form for adding/removing filter conditions. Each condition row should have:
        - A dropdown for the financial metric (P/E, P/B, etc.).
        - A dropdown for the operator (>, <, =).
        - A number input for the value.
- [ ] **State Management:**
    - [ ] Use React hooks (`useState`, `useReducer`) to manage the state of the ticker group, custom tickers, filter conditions, and search results.
- [ ] **API Integration:**
    - [ ] Implement the client-side logic to call the new `/api/stock-screener` endpoint when the user submits the form.
    - [ ] Handle loading and error states during the API call.
- [ ] **Results Display:**
    - [ ] Create a component to display the list of matching stock tickers in a clear, readable table or list format.
        - [ ] Reuse "allocation-table" at 'Optimizer.jsx'.
    - [ ] Include the values of the metrics that were used in the filter conditions in the results display.
        - [ ] Reuse "optimizer-results-grid" and "optimizer-result-card" at 'Optimizer.jsx'.
    - [ ] Add a popup to show the metrics of the selected stock if mouse hovered.
        - [ ] a simple table class to display metrics will have to do.
