
# Project Tasks

This file outlines the tasks completed during the development of the portfolio analysis and optimization tool.

## Project Setup & Foundation
- [x] Initialize React frontend application.
- [x] Set up Python backend server.
- [x] Establish communication between frontend and backend.
- [x] Initialize Git for version control.
- [x] Define project structure and file organization.
- [x] Configure ESLint for code quality.

## Backend Development
- [x] Create API endpoint to fetch historical stock data.
- [x] Implement Monte Carlo simulation for portfolio projection.
- [x] Develop portfolio optimization logic using the Markowitz model.
- [x] Implement hedging analysis to find correlated assets.
- [x] Create endpoint to serve financial statement data.
- [x] Add utility to read stock tickers from CSV files.

## Frontend Development
- [x] Design the main application layout and UI components.
- [x] Create `TickerInput` for stock selection.
- [x] Develop `PortfolioInput` to define asset allocations.
- [x] Implement `StockChart` to visualize historical price data.
- [x] Create `PortfolioGraph` to display portfolio performance.
- [x] Build `FinancialStatement` component to show company financials.
- [x] Develop `Optimizer` component for portfolio optimization.
- [x] Create `Hedge` component for hedging analysis.
- [x] Implement `FutureChart` to display Monte Carlo simulation results.
- [x] Add date input components for selecting time ranges.

## Internationalization (i18n)
- [x] Integrate `i18next` for internationalization.
- [x] Create translation files for English (`en`) and Korean (`ko`).
- [x] Implement `LanguageSelector` component.
- [x] Wrap UI text with translation functions.

## Documentation
- [x] Write `README.md` with a project overview and setup instructions.
- [x] Create `REQUIREMENTS.md` to detail project requirements.
- [x] Write `DESIGN.md` to outline the technical design and architecture.
- [x] Create this `TASKS.md` file to track project development progress.

## ML-Based Portfolio Optimization
- [x] **Backend:** Integrate `Prophet` for time-series forecasting of asset returns.
- [x] **Backend:** Update `portfolio_optimization.py` to include a function for ML-based return forecasting.
- [x] **Backend:** Modify the `optimize_portfolio` function and the API endpoint in `app.py` to handle a `use_ml_forecast` flag.
- [x] **Frontend:** Add a checkbox to the `Optimizer.jsx` component to enable ML forecasting.
- [x] **Frontend:** Update `handleSubmit` in `Optimizer.jsx` to send the `use_ml_forecast` flag to the backend.
- [x] **i18n:** Add translations for the new UI element in both English and Korean.
- [x] **Documentation:** Update `REQUIREMENTS.md` and `DESIGN.md` to reflect the new feature.

## Data Pipeline & Forecasting Engine
- [x] **ML Forecasting as Default:** Enabled Prophet-based ML forecasting as the default.
- [x] **Robust Data Pipeline:** Implemented a resilient data fetching and processing pipeline with ticker sanitization, intelligent data filling, and graceful failure handling.
- [x] **Critical Bug Fixes:**
    - [x] Resolved data alignment errors between returns and covariance matrix.
    - [x] Fixed Prophet "Length mismatch" errors with standardized DataFrame structure.
    - [x] Addressed various `NameError` issues.
- [x] **Log Suppression:** Silenced verbose logs from third-party libraries.

## Performance Optimization
- [x] **Parallel Processing:** Implemented concurrent model training and data fetching.
- [x] **Batch Data Fetching:** Switched to batch downloads for multiple tickers.
- [x] **Lightweight Forecasting:** Introduced faster alternatives to Prophet (ARIMA, exponential smoothing).
- [x] **Aggressive Caching:**
    - [x] Implemented multi-level caching (In-Memory & Redis/Disk).
    - [x] Added cache warming and smart invalidation.
- [ ] **Future Optimizations (Planned):**
    - [ ] Memory-intensive pre-processing and model pre-training.
    - [ ] Advanced parallel processing (Multi-processing + Multi-threading).
    - [ ] GPU acceleration for matrix operations.
    - [ ] Optimized data structures and predictive prefetching.

## Frontend Refactoring & UX Improvements
- [x] **Unified Update Logic:** Refactored `App.jsx` to use a single, centralized data update function.
- [x] **Auto-Updates:** Removed manual "Update" buttons from input components (`DateInput`, `FutureDateInput`).
- [x] **Debouncing:** Added debouncing to inputs to prevent excessive API calls.
- [x] **State Synchronization:** Fixed race conditions and state lag issues.
- [x] **API Call Optimization:** Implemented request cancellation (`AbortController`) to prevent redundant fetches.

## Ongoing Debugging & Issues
- [x] **`get_ticker_group` NameError:** Resolved missing import in `portfolio_optimization.py`.
- [x] **`stats` NameError:** Fixed missing import for lightweight forecasting.
- [x] **Batch Fetching Regression:** Corrected handling of yfinance MultiIndex DataFrames.
- [ ] **Caching Performance:** Test caching system performance and hit ratios under load.
- [ ] **Input Flow:** Final verification of user input flow after refactoring.